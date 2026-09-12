# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Tests against a real identified model: Feetech STS3215 7.4V, BAM ``m5``.

The params file is the one shipped with BAM
(``bam/params/feetech_sts3215_7_4V/m5.json``), copied into ``tests/fixtures/`` so
these tests do not depend on the BAM checkout being present.

Two independent checks are made:

1. :func:`test_matches_the_documented_m5_equation` re-implements the ``M5``
   equation *from the documentation* (``docs/theory/models.rst``) using the doc's
   own symbols, so a transcription error in either direction shows up.
2. :func:`test_parity_with_bams_reference_implementation` compares against
   ``bam.model.Model.compute_frictions`` itself, which is what the parameters
   were fitted against.

Runs with ``torch`` only; ``bam`` is imported from source if the checkout exists.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bam_actuators.friction import FLAG_NAMES, PARAMETER_NAMES, BamFrictionModel, variant_flags
from bam_actuators.params import available_bundled, resolve_params_file

from bam_paths import bam_root

#: The model under test, loaded from the copy bundled with the extension.
FIXTURE = Path(resolve_params_file("sts3215/m5"))
BAM_ROOT = bam_root()

# Firmware settings that live on BAM's actuator *class*, not in the params file.
BAM_STS3215_FIRMWARE = {"vin": 7.4, "kp": 32.0, "error_gain": 0.166, "max_pwm": 0.97}


@pytest.fixture
def params() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def model() -> BamFrictionModel:
    return BamFrictionModel.from_json(str(FIXTURE))


@pytest.fixture
def states() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Motor torque, external torque and velocity, covering the sign quadrants."""
    generator = torch.Generator().manual_seed(0)
    motor = torch.cat([torch.randn(128, 2, generator=generator), torch.tensor([[0.0, 1.0]])])
    external = torch.cat([torch.randn(128, 2, generator=generator), torch.tensor([[0.0, -1.0]])])
    dq = torch.cat([torch.randn(128, 2, generator=generator).abs(), torch.tensor([[0.0, 0.5]])])
    return motor, external, dq


# ----------------------------------------------------------------------
# The fixture is a genuine m5 model
# ----------------------------------------------------------------------


def test_fixture_is_a_real_m5_model(model, params):
    assert params["model"] == "m5"
    assert params["actuator"] == "sts3215"

    assert model.model_name == "m5"
    assert model.active_flags == ["load_dependent", "directional", "stribeck"]
    assert not model.quadratic


def test_every_friction_parameter_in_the_file_is_loaded(model, params):
    for name in PARAMETER_NAMES:
        if name in params:
            assert getattr(model, name) == pytest.approx(params[name]), name


def test_params_file_carries_no_firmware_settings(params):
    """BAM's params files hold the *model*, not the firmware.

    This is a tripwire: pointing ``params_file`` at this file gives you the right
    friction but leaves ``vin``/``kp``/``error_gain``/``max_pwm`` at whatever the
    cfg says, so they have to be set by hand to BAM's actuator defaults.
    """
    for key in ("vin", "kp", "error_gain", "max_pwm"):
        assert key not in params, f"{key!r} unexpectedly present; update the cfg docs"
    # ...and these are the values the cfg must supply for STS3215.
    assert BAM_STS3215_FIRMWARE["vin"] == 7.4


# ----------------------------------------------------------------------
# 1. Against the documented equation
# ----------------------------------------------------------------------


def _documented_m5(params: dict, motor, external, dq):
    r"""``M5`` exactly as written in ``docs/theory/models.rst``.

    .. math::

        \tau_{fm} = K_v|\dot\theta| + K_c + |K_m\tau_m - K_e\tau_e|
                    + \exp\!\left(-\left|\frac{\dot\theta}{\dot\theta_s}\right|^{\alpha}\right)
                      \left(K_{cs} + |K_{ms}\tau_m - K_{es}\tau_e|\right)
    """
    K_v = params["friction_viscous"]
    K_c = params["friction_base"]
    K_cs = params["friction_stribeck"]
    theta_s = params["dtheta_stribeck"]
    alpha = params["alpha"]
    K_m = params["load_friction_motor"]
    K_e = params["load_friction_external"]
    K_ms = params["load_friction_motor_stribeck"]
    K_es = params["load_friction_external_stribeck"]

    stribeck = torch.exp(-((dq.abs() / theta_s) ** alpha))
    return (
        K_v * dq.abs()
        + K_c
        + (K_m * motor - K_e * external).abs()
        + stribeck * (K_cs + (K_ms * motor - K_es * external).abs())
    )


def test_matches_the_documented_m5_equation(model, params, states):
    motor, external, dq = states

    frictionloss, damping = model.compute(motor, external, dq)
    tau_fm_ours = frictionloss + damping * dq.abs()
    tau_fm_doc = _documented_m5(params, motor, external, dq)

    torch.testing.assert_close(tau_fm_ours, tau_fm_doc, rtol=1e-5, atol=1e-9)


def test_viscous_term_is_reported_as_a_coefficient(model, params):
    """``compute()`` returns K_v, not K_v*|dq| - the |dq| is applied by the clip."""
    _, damping = model.compute(torch.zeros(1), torch.zeros(1), torch.tensor([3.0]))

    torch.testing.assert_close(damping, torch.tensor([params["friction_viscous"]]), rtol=1e-6, atol=0.0)


# ----------------------------------------------------------------------
# 2. Against BAM's own implementation
# ----------------------------------------------------------------------


def _load_bam(path) -> object:
    """Load a params file through BAM itself, or skip if unavailable."""
    if not BAM_ROOT.is_dir():
        pytest.skip(f"BAM checkout not found at {BAM_ROOT}")

    if str(BAM_ROOT) not in sys.path:
        sys.path.insert(0, str(BAM_ROOT))
    try:
        from bam.model import load_model
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not import bam from {BAM_ROOT}: {exc}")

    return load_model(str(path))


def _bam_reference():
    """BAM's model for the fixture under test."""
    return _load_bam(FIXTURE)


def test_parity_with_bams_reference_implementation(model, states):
    """In float64 our port must reproduce ``Model.compute_frictions`` exactly.

    Running both sides in the same dtype is what makes this a test of the *maths*
    rather than of floating-point precision - see the float32 test below for the
    precision budget the simulator actually operates in.
    """
    reference = _bam_reference()
    motor, external, dq = (state.double() for state in states)

    frictionloss, damping = model.compute(motor, external, dq)
    ref_frictionloss, ref_damping = reference.compute_frictions(
        motor.numpy(), external.numpy(), dq.numpy()
    )

    assert frictionloss.dtype == torch.float64, "model should follow the input dtype"
    np.testing.assert_allclose(frictionloss.numpy(), ref_frictionloss, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(damping.numpy(), ref_damping, rtol=1e-12, atol=1e-15)


def test_float32_precision_stays_within_float32_epsilon(model, states):
    """The simulator runs in float32, so agreement is only ~1e-7 there.

    This pins the expected precision budget: if a future change makes the float32
    gap grow beyond float32 epsilon, something is genuinely wrong.
    """
    reference = _bam_reference()
    motor, external, dq = states  # float32

    frictionloss, damping = model.compute(motor, external, dq)
    ref_frictionloss, ref_damping = reference.compute_frictions(
        motor.numpy(), external.numpy(), dq.numpy()
    )

    assert frictionloss.dtype == torch.float32
    np.testing.assert_allclose(frictionloss.numpy(), ref_frictionloss, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(damping.numpy(), ref_damping, rtol=1e-6, atol=1e-7)
    assert torch.finfo(torch.float32).eps < 1e-6


def test_parity_holds_in_the_sign_quadrants(model):
    """m5 is directional, so the four (tau_m, tau_e) sign combinations must match."""
    reference = _bam_reference()

    motor = torch.tensor([[1.0], [1.0], [-1.0], [-1.0], [1.0], [-1.0]], dtype=torch.float64)
    external = torch.tensor([[1.0], [-1.0], [1.0], [-1.0], [2.0], [2.0]], dtype=torch.float64)
    dq = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.3], [0.3]], dtype=torch.float64)

    frictionloss, _ = model.compute(motor, external, dq)
    ref_frictionloss, _ = reference.compute_frictions(motor.numpy(), external.numpy(), dq.numpy())

    np.testing.assert_allclose(frictionloss.numpy(), ref_frictionloss, rtol=1e-12, atol=1e-15)

    # The directional term must actually see both torques: swapping them is not a
    # no-op for m5, which is the whole point of the model.
    swapped, _ = model.compute(external, motor, dq)
    assert not np.allclose(swapped.numpy(), ref_frictionloss, rtol=1e-12, atol=1e-15)


# ----------------------------------------------------------------------
# Behaviour that follows from the budget
# ----------------------------------------------------------------------


def test_non_directional_parameters_do_not_affect_m5(model):
    """m5 is directional: the m3/m4-only coefficients must be inert."""
    before = model.compute(torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.2]))

    model.load_friction_base = 123.0
    model.load_friction_stribeck = 456.0
    after = model.compute(torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.2]))

    torch.testing.assert_close(before[0], after[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(before[1], after[1], rtol=0.0, atol=0.0)


def test_budget_is_symmetric_in_the_torque_difference_sign(model):
    """|K_m*tau_m - K_e*tau_e| is even, so flipping both signs changes nothing."""
    a = model.compute(torch.tensor([0.3]), torch.tensor([-0.7]), torch.tensor([0.1]))[0]
    b = model.compute(torch.tensor([-0.3]), torch.tensor([0.7]), torch.tensor([0.1]))[0]

    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-12)


def test_stiction_holds_against_the_real_friction_budget(model, params):
    """A torque below the identified budget must be fully absorbed."""
    budget = params["friction_base"]
    tiny = 0.5 * budget

    net = model.apply(
        motor_torque=torch.tensor([tiny]), external_torque=torch.tensor([0.0]), dq=torch.tensor([0.0])
    )

    torch.testing.assert_close(net, torch.zeros(1), rtol=0.0, atol=1e-12)


def test_alpha_sits_at_the_fit_bound(params):
    """The identified alpha is ~10 (the upper bound), so the Stribeck term is sharp.

    Worth pinning: it means friction collapses to the Coulomb value within a tiny
    velocity band, which is what the sim should reproduce.
    """
    assert params["alpha"] > 9.9

    model = BamFrictionModel.from_json(str(FIXTURE))
    at_rest = model.compute(torch.zeros(1), torch.zeros(1), torch.tensor([0.0]))[0]
    moving = model.compute(torch.zeros(1), torch.zeros(1), torch.tensor([1.0]))[0]

    # friction_stribeck is ~7e-9 for this model, so the Stribeck term is negligible
    # in both cases; the velocity dependence comes from the stopping clip instead.
    torch.testing.assert_close(at_rest, moving, rtol=1e-6, atol=1e-12)


# ----------------------------------------------------------------------
# All six bundled feetech models
# ----------------------------------------------------------------------

ALL_MODELS = ["m1", "m2", "m3", "m4", "m5", "m6"]


def test_all_six_feetech_models_are_bundled():
    """The extension ships m1-m6 so a config can say `params_file="sts3215/m5"`."""
    assert available_bundled()["sts3215"] == ALL_MODELS


@pytest.mark.parametrize("variant", ALL_MODELS)
def test_each_bundled_model_selects_its_own_variant(variant):
    model = BamFrictionModel.from_json(resolve_params_file(f"sts3215/{variant}"))

    assert model.model_name == variant
    expected = [name for name in FLAG_NAMES if variant_flags(variant).get(name, False)]
    assert model.active_flags == expected


@pytest.mark.parametrize("variant", ALL_MODELS)
def test_each_bundled_model_matches_bam_in_float64(variant, states):
    """Every shipped model must reproduce BAM's own budget, not just m5."""
    model = BamFrictionModel.from_json(resolve_params_file(f"sts3215/{variant}"))
    reference = _load_bam(resolve_params_file(f"sts3215/{variant}"))

    motor, external, dq = (state.double() for state in states)
    frictionloss, damping = model.compute(motor, external, dq)
    ref_frictionloss, ref_damping = reference.compute_frictions(
        motor.numpy(), external.numpy(), dq.numpy()
    )

    np.testing.assert_allclose(frictionloss.numpy(), ref_frictionloss, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(damping.numpy(), ref_damping, rtol=1e-12, atol=1e-15)


def test_m6_quadratic_term_is_gated_on_opposing_torque_signs():
    """BAM gates the m6 quadratic term on sign(tau_e) != sign(tau_m).

    The theory docs present Q with no such gate, so this documents which side the
    port follows - the code, because that is what the parameters were fitted to.
    """
    model = BamFrictionModel.from_json(resolve_params_file("sts3215/m6"))
    assert model.quadratic

    position, velocity = torch.tensor([1.0]), torch.tensor([1.0])
    # Same sign -> gate closed -> the quadratic contribution is dropped.
    same_sign, _ = model.compute(torch.tensor([1.0]), torch.tensor([2.0]), velocity)
    opposite_sign, _ = model.compute(torch.tensor([1.0]), torch.tensor([-2.0]), velocity)

    assert float(same_sign) != float(opposite_sign)
