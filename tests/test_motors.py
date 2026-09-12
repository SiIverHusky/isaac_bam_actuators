# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for the per-motor control laws against BAM's actuator classes.

Each motor module is checked against the corresponding class in
``bam/<vendor>/actuator.py`` - both the firmware constants and the control law
itself, driven over a trajectory that exercises the rate limiter and the current
limiter. All of this is torch/numpy only, so it runs without Isaac Lab.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bam_actuators.motors import available_motors, get_motor
from bam_actuators.params import resolve_params_file

BAM_ROOT = Path("/home/hharis/Mangdang/BAM")

#: Motors we have a BAM counterpart for, and a bundled params file to drive them.
PARITY_CASES = [
    ("sts3215", "sts3215/m5"),
    ("md01", None),
]

DT = 0.02


def _bam_imports():
    """Import BAM from source, or skip the test."""
    if not BAM_ROOT.is_dir():
        pytest.skip(f"BAM checkout not found at {BAM_ROOT}")
    if str(BAM_ROOT) not in sys.path:
        sys.path.insert(0, str(BAM_ROOT))
    try:
        from bam.actuators import actuators
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not import bam from {BAM_ROOT}: {exc}")
    return actuators


def _bam_model(motor: str, params: str | None):
    """A BAM ``Model`` (with its actuator) for a motor, optionally with params."""
    _bam_imports()
    if params is None:
        from bam.actuators import actuators

        from bam.model import models

        model = models["m5"]()
        model.set_actuator(actuators[motor]())
        return model

    from bam.model import load_model

    return load_model(resolve_params_file(params))


def _our_motor(motor: str, params: str | None):
    """Our motor module, fed the same identified parameters."""
    instance = get_motor(motor)()
    if params is not None:
        instance.set_params(json.loads(Path(resolve_params_file(params)).read_text()))
    return instance


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


def test_registry_contains_the_ported_motors():
    assert set(available_motors()) >= {"generic", "sts3215", "md01"}


def test_unknown_motor_raises_with_guidance():
    with pytest.raises(KeyError, match="Available motors"):
        get_motor("nope")


# ----------------------------------------------------------------------
# Firmware constants
# ----------------------------------------------------------------------


@pytest.mark.parametrize("motor, params", PARITY_CASES)
def test_firmware_constants_match_bam(motor, params):
    """vin / kp / error_gain / max_pwm / max_current / stateful must be identical."""
    _bam_imports()
    reference = _bam_model(motor, params).actuator
    ours = _our_motor(motor, params)

    assert ours.vin == pytest.approx(reference.vin)
    assert ours.kp == pytest.approx(reference.kp)
    assert ours.error_gain == pytest.approx(reference.error_gain)
    assert ours.max_pwm == pytest.approx(reference.max_pwm)
    assert ours.stateful == type(reference).stateful

    if reference.max_current is None:
        assert ours.max_current is None
    else:
        assert ours.max_current == pytest.approx(reference.max_current)


# ----------------------------------------------------------------------
# Control law
# ----------------------------------------------------------------------


def _drive_control(step, q_targets, positions, velocities, dt):
    """Apply ``step`` over a trajectory, returning one control value per step."""
    return np.array(
        [np.asarray(step(qt, q, dq, dt), dtype=np.float64) for qt, q, dq in zip(q_targets, positions, velocities)]
    )


def _trajectory(n_steps: int = 25):
    """A step command that exercises the slew limiter, with the joint creeping."""
    q_targets = [0.0] * 5 + [1.0] * (n_steps - 5)
    positions = [min(0.3, 0.02 * i) for i in range(n_steps)]
    velocities = [0.5] * n_steps
    return q_targets, positions, velocities


@pytest.mark.parametrize("motor, params", PARITY_CASES)
def test_control_law_matches_bam_over_a_trajectory(motor, params):
    """The control signal must match BAM step for step, including state.

    Both implementations are stateful for the STS3215, so they are driven over the
    same sequence from the same initial condition - which is what makes this a
    test of the control law rather than of a single call.
    """
    reference = _bam_model(motor, params).actuator
    ours = _our_motor(motor, params)

    q_targets, positions, velocities = _trajectory()

    ref = _drive_control(reference.compute_control, q_targets, positions, velocities, DT)
    got = _drive_control(ours.control, q_targets, positions, velocities, DT)

    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)


def test_sts3215_rate_limiter_actually_slews():
    """A step command must not be followed instantly - that is the whole point.

    Note this is asserted on the *internal target*, not on the control signal:
    the firmware gain is large enough that the duty cycle saturates within a few
    steps of the step command, so the voltage stops being informative long before
    the target has arrived.
    """
    motor = get_motor("sts3215")()
    q_targets, positions, velocities = _trajectory()

    targets, volts = [], []
    for q_target, q, dq in zip(q_targets, positions, velocities):
        volts.append(float(motor.control(q_target, q, dq, DT)))
        targets.append(float(motor.q_target_smooth))

    max_step = motor.max_velocity * DT

    # Never moves further than the slew rate allows...
    assert np.all(np.abs(np.diff([0.0] + targets)) <= max_step + 1e-12)

    # ...so 1.0 rad at ~0.1043 rad/step needs 10 calls: still short at index 8,
    # arrived by index 20.
    assert targets[8] < 1.0
    assert targets[20] == pytest.approx(1.0)

    # And the step command does not slam the duty cycle on its first call, which
    # an unlimited P law would (error 0.9 rad -> duty >> 1).
    assert volts[5] < 0.2 * motor.vin * motor.max_pwm


def test_rate_limiter_step_scales_with_dt():
    """Halving dt must halve how far the internal target can travel per call."""
    motor = get_motor("sts3215")()
    motor.set_params({"kt": 0.78, "R": 2.0, "armature": 1e-4, "error_gain_ratio": 1.0})

    at_20ms = float(motor.control(1.0, 0.0, 0.0, 0.02))
    at_10ms = float(get_motor("sts3215")().control(1.0, 0.0, 0.0, 0.01))

    assert at_10ms < at_20ms


def test_stateful_motor_requires_dt():
    """Without dt the slew rate is undefined, so it must fail loudly."""
    with pytest.raises(ValueError, match="needs dt|dt.*required"):
        get_motor("sts3215")().control(1.0, 0.0, 0.0, None)


def test_md01_current_limiter_bounds_the_duty_cycle():
    """The MD01 firmware caps the duty cycle to hold the current within max_current."""
    motor = get_motor("md01")()
    assert motor.max_current is not None

    # A huge position error would demand a duty cycle far beyond the current cap.
    volts = float(motor.control(100.0, 0.0, 0.0, DT))
    span = motor.R * motor.max_current / motor.vin

    assert volts <= motor.vin * min(span, motor.max_pwm) + 1e-12


# ----------------------------------------------------------------------
# Torque
# ----------------------------------------------------------------------


@pytest.mark.parametrize("motor, params", PARITY_CASES)
def test_torque_matches_bam(motor, params):
    reference = _bam_model(motor, params).actuator
    ours = _our_motor(motor, params)

    controls = np.array([0.0, 1.0, -1.0, 3.7, -3.7])
    velocities = np.array([0.0, 0.5, -0.5, 2.0, -2.0])

    ref = np.array(
        [np.asarray(reference.compute_torque(c, True, 0.0, dq), dtype=np.float64) for c, dq in zip(controls, velocities)]
    )
    got = np.array([float(ours.torque(torch.tensor(c), True, 0.0, torch.tensor(dq))) for c, dq in zip(controls, velocities)])

    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)


def test_torque_is_zero_when_disabled():
    motor = get_motor("sts3215")()
    motor.set_params({"kt": 0.78, "R": 2.0})

    assert float(motor.torque(torch.tensor(12.0), False, 0.0, torch.tensor(0.0))) == 0.0


# ----------------------------------------------------------------------
# State handling
# ----------------------------------------------------------------------


def test_reset_reseeds_the_internal_target_from_the_current_position():
    """After a teleport the slew limiter must start from where the joint now is."""
    _bam_imports()
    reference = _bam_model("sts3215", "sts3215/m5").actuator
    ours = _our_motor("sts3215", "sts3215/m5")

    # Drive both away from the origin first, so the internal target is non-zero.
    for _ in range(5):
        reference.compute_control(1.0, 0.0, 0.0, DT)
        ours.control(1.0, 0.0, 0.0, DT)

    # Teleport to 0.5 and reset: a zero command error must give zero volts,
    # which only happens if the internal target was re-seeded to the new position.
    reference.reset()
    ours.reset()
    ref = float(np.asarray(reference.compute_control(0.5, 0.5, 0.0, DT)))
    got = float(ours.control(0.5, 0.5, 0.0, DT))

    assert got == pytest.approx(ref, abs=1e-15)
    assert got == pytest.approx(0.0, abs=1e-15)


def test_get_and_set_state_round_trip():
    motor = _our_motor("sts3215", "sts3215/m5")
    motor.control(1.0, 0.0, 0.0, DT)
    snapshot = motor.get_state()

    for _ in range(5):
        motor.control(1.0, 0.0, 0.0, DT)
    advanced = motor.get_state()

    motor.set_state(snapshot)
    assert not torch.equal(torch.as_tensor(advanced[0]), torch.as_tensor(motor.get_state()[0]))

    motor.set_state(advanced)
    assert torch.equal(torch.as_tensor(advanced[0]), torch.as_tensor(motor.get_state()[0]))
