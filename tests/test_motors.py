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

from bam_paths import bam_root

BAM_ROOT = bam_root()

#: Motors we have a BAM counterpart for, and a bundled params file to drive them.
#: ``md01i`` appears twice because the params file carries the fitted current limit
#: and gain ratio, so two variants exercise two very different control ranges.
PARITY_CASES = [
    ("sts3215", "sts3215/m5"),
    ("md01", None),
    ("md01i", "md01i/m3"),
    ("md01i", "md01i/m6"),
    ("md01c", None),
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
    assert set(available_motors()) >= {"generic", "sts3215", "md01", "md01i", "md01c"}


def test_unknown_motor_raises_with_guidance():
    with pytest.raises(KeyError, match="Available motors"):
        get_motor("nope")


# ----------------------------------------------------------------------
# Firmware constants
# ----------------------------------------------------------------------


@pytest.mark.parametrize("motor, params", PARITY_CASES)
def test_firmware_constants_match_bam(motor, params):
    """vin / kp / error_gain / max_pwm / current cap / stateful must be identical."""
    _bam_imports()
    reference = _bam_model(motor, params).actuator
    ours = _our_motor(motor, params)

    assert ours.vin == pytest.approx(reference.vin)
    assert ours.kp == pytest.approx(reference.kp)
    assert ours.error_gain == pytest.approx(reference.error_gain)
    assert ours.max_pwm == pytest.approx(reference.max_pwm)
    assert ours.stateful == type(reference).stateful
    # The unit is what BAM uses to pick the torque equation, so it has to agree too.
    assert ours.control_unit == reference.control_unit()

    # The current cap lives on the actuator for the voltage law and on the model
    # (named current_limit) for the current laws, so BAM's classes do not all have
    # a max_current attribute.
    if getattr(reference, "max_current", None) is None:
        assert getattr(ours, "max_current", None) is None
    else:
        assert ours.max_current == pytest.approx(reference.max_current)


def test_md01_family_seeds_match_bams_initialize():
    """An unfitted MD01 must start from the same numbers as BAM's ``initialize()``.

    Without a params file these seeds *are* the model, so a drift here would show
    up as a plausible-looking but wrong rollout rather than as an error.
    """
    _bam_imports()
    from bam.actuators import actuators
    from bam.model import models

    for motor in ("md01", "md01i", "md01c"):
        bam_model = models["m5"]()
        bam_model.set_actuator(actuators[motor]())
        ours = get_motor(motor)()

        for name in ours.parameters:
            parameter = getattr(bam_model, name, None)
            if parameter is None:
                continue
            assert getattr(ours, name) == pytest.approx(parameter.value), f"{motor}.{name}"

        # ...and the firmware the params files never carry.
        for name in ("kp_current", "kff_current", "cap_ma"):
            if hasattr(ours, name):
                assert getattr(ours, name) == pytest.approx(getattr(bam_model.actuator, name)), f"{motor}.{name}"


def test_md01c_loop_firmware_matches_the_measured_values():
    """The md01c loop gains are measured flash values, so pin them literally."""
    motor = get_motor("md01c")()

    assert motor.vin == 12.0
    assert motor.cap_ma == 1500.0
    assert motor.max_pwm == pytest.approx(0.99)
    assert motor.kp_current == pytest.approx(6e-4)
    assert motor.kff_current == pytest.approx(3e-4)

    _bam_imports()
    from bam.actuators import actuators
    from bam.model import models

    bam_model = models["m5"]()
    bam_model.set_actuator(actuators["md01c"]())
    for name in ("kp_current", "kff_current", "max_pwm", "cap_ma"):
        assert getattr(motor, name) == pytest.approx(getattr(bam_model.actuator, name))


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


def test_md01i_bounds_the_current_command():
    """md01i is the inverse: the *current setpoint* is what gets capped.

    The cap is the fitted ``current_limit``, and it has to bind on the command
    itself - not merely somewhere downstream of the duty cycle - because that is
    the saturation the fit was identified with.
    """
    motor = _our_motor("md01i", "md01i/m3")

    current = float(motor.control(100.0, 0.0, 0.0, DT))
    assert abs(current) <= motor.current_limit + 1e-12
    # The fitted limit is the binding one: the H-bridge is not the constraint at rest.
    assert abs(current) == pytest.approx(motor.current_limit, rel=1e-12)

    # A demanding command with the ratio folded in must not escape either.
    assert abs(float(motor.control(100.0, 0.0, 5.0, DT))) <= motor.current_limit + 1e-12


def test_md01i_torque_ceiling_is_the_physical_quantity():
    """``kt`` is a gauge in the md01i law; ``kt * current_limit`` is not.

    BAM's README reports 0.43-0.46 Nm for m1/m3/m5/m6, so a fit whose ceiling
    drifts out of that band means the params file was not identified on this bench.
    """
    for model in ("m1", "m3", "m5", "m6"):
        motor = _our_motor("md01i", f"md01i/{model}")
        assert 0.40 <= motor.torque_limit <= 0.50, f"md01i/{model}: {motor.torque_limit}"


def test_md01i_scales_the_error_into_amps():
    """The law is a P law in amps: the setpoint is linear in the error."""
    motor = get_motor("md01i")()
    motor.set_params({"error_gain_ratio": 1.0, "current_limit": 100.0, "R": 1.0, "kt": 0.5})

    at_rest = float(motor.control(0.1, 0.0, 0.0, DT))
    assert at_rest == pytest.approx(0.1 * motor.kp * motor.error_gain)
    assert float(motor.control(-0.1, 0.0, 0.0, DT)) == pytest.approx(-at_rest)

    # Doubling the ratio doubles the current - that is what makes it a gauge.
    motor.set_params({"error_gain_ratio": 2.0})
    assert float(motor.control(0.1, 0.0, 0.0, DT)) == pytest.approx(2.0 * at_rest)


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
