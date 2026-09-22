# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Tests against the new identified MD01 models: current-controlled ``md01i``.

The params files are BAM's campaign-2 fits (``bam/params/md01-3/``, servo 6,
``data_md01-6v2``), copied into ``bam_actuators/params/md01i/`` so these tests do
not depend on the BAM checkout being present.

Three independent checks are made:

1. :func:`test_matches_the_documented_m3_equation` re-implements the ``M3``
   equation *from the documentation* (``docs/theory/models.rst``) with the doc's own
   symbols, so a transcription error in either direction shows up.
2. :func:`test_parity_with_bams_reference_implementation` compares every bundled
   variant - friction budget *and* control law - against ``bam.model.load_model``
   itself, which is what the parameters were fitted against.
3. :func:`test_the_params_file_alone_can_run_the_servo` pins the difference from the
   STS3215 fits: these files carry the motor parameters too, so ``md01i/m3`` is a
   complete actuator with no ``motor_params`` and no recording.

Runs with ``torch`` only; ``bam`` is imported from source if the checkout exists.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bam_actuators.motors import get_motor
from bam_actuators.params import available_bundled, resolve_params_file

from bam_paths import bam_root

#: The working model of the campaign (BAM's README also calls out m6).
FIXTURE = Path(resolve_params_file("md01i/m3"))
BAM_ROOT = bam_root()

#: Firmware settings that live on BAM's actuator *class*, not in the params file.
BAM_MD01I_FIRMWARE = {"vin": 12.0, "kp": 80.0, "error_gain": 0.030, "max_pwm": 1.0}

DT = 0.005


@pytest.fixture
def params() -> dict:
    return json.loads(FIXTURE.read_text())


def _bam_load(path):
    """``bam.model.load_model`` for a params file, or skip the test."""
    if not BAM_ROOT.is_dir():
        pytest.skip(f"BAM checkout not found at {BAM_ROOT}")
    if str(BAM_ROOT) not in sys.path:
        sys.path.insert(0, str(BAM_ROOT))
    try:
        from bam.model import load_model
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not import bam from {BAM_ROOT}: {exc}")
    return load_model(str(path))


# ----------------------------------------------------------------------
# The bundled files are genuine md01i models
# ----------------------------------------------------------------------


def test_all_six_variants_are_bundled():
    """The campaign covers m1..m6; a partial copy would silently shrink the suite."""
    assert available_bundled()["md01i"] == ["m1", "m2", "m3", "m4", "m5", "m6"]


@pytest.mark.parametrize("model", ["m1", "m3", "m5", "m6"])
def test_fixture_is_a_real_md01i_model(model):
    """The file names the actuator and the friction maths, and we honour both."""
    params = json.loads(Path(resolve_params_file(f"md01i/{model}")).read_text())

    assert params["actuator"] == "md01i"
    assert params["model"] == model

    from bam_actuators.friction import BamFrictionModel

    friction = BamFrictionModel.from_json(resolve_params_file(f"md01i/{model}"))
    assert friction.model_name == model

    # The motor the file selects really is the current-controlled one.
    motor = get_motor(params["actuator"])()
    assert motor.control_unit == "amps"


def test_every_friction_parameter_in_the_file_is_loaded(params):
    from bam_actuators.friction import PARAMETER_NAMES, BamFrictionModel

    model = BamFrictionModel.from_json(str(FIXTURE))

    for name in PARAMETER_NAMES:
        if name in params:
            assert getattr(model, name) == pytest.approx(params[name]), name


def test_every_motor_parameter_in_the_file_is_loaded(params):
    """Unlike the STS3215 fits, these files carry the identified motor too."""
    motor = get_motor("md01i")()

    applied = motor.set_params(params)

    assert set(applied) >= {"kt", "R", "armature", "current_limit", "error_gain_ratio", "q_offset"}
    for name in applied:
        assert getattr(motor, name) == pytest.approx(params[name]), name


def test_the_params_file_alone_can_run_the_servo(params):
    """``params_file="md01i/m3"`` has to be enough - no ``motor_params`` escape hatch.

    The firmware constants (``vin``, ``kp``, ``error_gain``, ``max_pwm``) come from
    the module, the identified quantities from the file, and the combination has to
    produce a torque ceiling in the band BAM's README reports (0.43-0.46 Nm for
    m1/m3/m5/m6).
    """
    motor = get_motor("md01i")()
    motor.set_params(params)

    for name, value in BAM_MD01I_FIRMWARE.items():
        assert getattr(motor, name) == pytest.approx(value), name

    assert 0.40 <= motor.torque_limit <= 0.50
    assert abs(float(motor.control(0.2, 0.0, 0.0, DT))) > 0.0


# ----------------------------------------------------------------------
# 1. Against the documented equation
# ----------------------------------------------------------------------


def _documented_m3(params: dict, motor, external, dq):
    r"""``M3`` exactly as written in ``docs/theory/models.rst``.

    .. math::

        \tau_{fm} = K_v|\dot\theta| + K_c + K_l|\tau_m - \tau_e|

    The budget returned is ``(frictionloss, damping)`` with
    ``|tau| <= frictionloss + damping * |dq|``, i.e. ``K_l`` is the load-dependent
    constant, and ``damping`` carries the viscous term.
    """
    K_v = params["friction_viscous"]
    K_c = params["friction_base"]
    K_l = params["load_friction_base"]

    frictionloss = K_c + K_l * (external - motor).abs()
    damping = torch.full_like(frictionloss, K_v)
    return frictionloss, damping


@pytest.fixture
def states() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Motor torque, external torque and velocity, covering the sign quadrants."""
    motor = torch.tensor([[-0.4], [-0.05], [0.0], [0.3], [0.2], [-0.2]], dtype=torch.float64)
    external = torch.tensor([[0.4], [0.05], [0.0], [-0.3], [-0.2], [0.2]], dtype=torch.float64)
    dq = torch.tensor([[0.0], [0.1], [-0.1], [1.0], [-1.0], [2.0]], dtype=torch.float64)
    return motor, external, dq


def test_matches_the_documented_m3_equation(params, states):
    from bam_actuators.friction import BamFrictionModel

    model = BamFrictionModel.from_json(str(FIXTURE))
    motor, external, dq = states

    expected, expected_damping = _documented_m3(params, motor, external, dq)
    frictionloss, damping = model.compute(motor, external, dq)

    torch.testing.assert_close(frictionloss, expected, rtol=0.0, atol=1e-15)
    torch.testing.assert_close(damping, expected_damping, rtol=0.0, atol=1e-15)


# ----------------------------------------------------------------------
# 2. Against BAM
# ----------------------------------------------------------------------


@pytest.mark.parametrize("model", ["m1", "m2", "m3", "m4", "m5", "m6"])
def test_parity_with_bams_reference_implementation(model):
    """Friction budget and control law must match BAM for every bundled variant.

    Both sides are float64, and the laws are stateless, so this is an exact
    comparison rather than a tolerance: any difference is a porting error.
    """
    from bam_actuators.friction import BamFrictionModel

    path = resolve_params_file(f"md01i/{model}")
    reference = _bam_load(path)

    friction = BamFrictionModel.from_json(path)
    motor = get_motor("md01i")()
    motor.set_params(json.loads(Path(path).read_text()))

    motor_torques = np.array([-0.4, -0.05, 0.0, 0.3], dtype=np.float64)
    external_torques = np.array([0.4, 0.05, 0.0, -0.3], dtype=np.float64)
    velocities = np.array([0.0, 0.1, -1.0, 2.0], dtype=np.float64)

    # Friction budget -- BAM and us, on the same (tau_m, tau_e, dq) grid.
    ours_loss, ours_damping = friction.compute(
        torch.tensor(motor_torques), torch.tensor(external_torques), torch.tensor(velocities)
    )
    ref_loss, ref_damping = reference.compute_frictions(motor_torques, external_torques, velocities)
    np.testing.assert_allclose(ours_loss.numpy(), ref_loss, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(ours_damping.numpy(), ref_damping, rtol=0.0, atol=1e-15)

    # Control law and torque, including the fitted errors and current cap.
    errors = np.array([0.0, 0.05, 0.3, -0.5, 2.0], dtype=np.float64)
    for error, dq in zip(errors, velocities):
        ours_current = float(motor.control(error, 0.0, dq, DT))
        ref_current = float(np.asarray(reference.actuator.compute_control(error, 0.0, dq, DT)))
        assert ours_current == pytest.approx(ref_current, rel=0, abs=1e-15)

        ours_torque = float(motor.torque(ours_current, True, 0.0, dq))
        ref_torque = float(np.asarray(reference.actuator.compute_torque(ref_current, True, 0.0, dq)))
        assert ours_torque == pytest.approx(ref_torque, rel=0, abs=1e-15)


# ----------------------------------------------------------------------
# Behaviour the fit implies
# ----------------------------------------------------------------------


@pytest.mark.parametrize("model", ["m1", "m3", "m5", "m6"])
def test_the_current_saturates_at_the_fitted_limit(model):
    """The current loop saturates, and the torque ceiling is that saturation."""
    path = resolve_params_file(f"md01i/{model}")
    motor = get_motor("md01i")()
    motor.set_params(json.loads(Path(path).read_text()))

    currents = [float(motor.control(error, 0.0, 0.0, DT)) for error in (0.0, 0.05, 0.2, 1.0, 10.0)]

    assert currents[0] == pytest.approx(0.0, abs=1e-15)
    assert currents == sorted(currents), "the P law must be monotone in the error"
    assert currents[-1] == pytest.approx(motor.current_limit, rel=1e-12)
    assert all(abs(current) <= motor.current_limit + 1e-12 for current in currents)

    # ...and the torque that saturation buys is the ceiling the README quotes.
    assert float(motor.torque(currents[-1], True, 0.0, 0.0)) == pytest.approx(motor.torque_limit)
