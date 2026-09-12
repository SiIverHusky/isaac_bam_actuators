# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity: our motor+friction composition vs BAM on a real recorded log.

This is the closest thing to the Isaac Lab test that can run without a GPU. It
drives BAM's pendulum testbench from a recorded trajectory two ways:

* **reference** - ``bam.model.Model`` + ``bam.simulate.Simulator``
* **ours** - ``bam_actuators.motors`` + ``bam_actuators.friction`` +
  :class:`tests.offline_rollout.OfflineSimulator`

Both sides get the same goal sequence, the same initial state and the same
``dt``, so any divergence comes from our components. Two motors are covered:

* ``md01`` - stateless control law with the firmware current limiter,
* ``sts3215`` with the bundled ``m5`` - stateful slew-limited target plus
  load-dependent, directional friction (which needs the external torque, so this
  also exercises the ``external_torque`` path that Isaac Lab cannot test yet).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bam_actuators.friction import BamFrictionModel
from bam_actuators.motors import get_motor
from bam_actuators.params import resolve_params_file

from offline_rollout import OfflineSimulator, drive, load_raw_log

BAM_ROOT = Path("/home/hharis/Mangdang/BAM")
LOG_DIR = BAM_ROOT / "data_raw"
#: A "steps" trajectory at kp=8: step commands exercise a stateful slew limiter.
LOG_NAME = "2026-09-10_15h22m23.json"

N_STEPS = 400


def _bam():
    """Import BAM from source, or skip."""
    if not LOG_DIR.is_dir():
        pytest.skip(f"recorded logs not found at {LOG_DIR}")
    if str(BAM_ROOT) not in sys.path:
        sys.path.insert(0, str(BAM_ROOT))
    try:
        import bam.simulate  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"could not import bam from {BAM_ROOT}: {exc}")


class BamSide:
    """Reference side: BAM's own components, driven through BAM's own ``Simulator``.

    Only the loop is ours; every computation is BAM's.
    """

    diagnostics = None  # no per-step recording on this side

    def __init__(self, model, dt: float):
        from bam.simulate import Simulator

        self.model = model
        self.dt = dt
        self.sim = Simulator(model)

    def control(self, goal):
        # Mirrors Simulator.rollout_log: the control law sees the RAW joint angle.
        return self.model.actuator.compute_control(goal, self.sim.q, self.sim.dq, self.dt)

    def step(self, control, torque_enable):
        self.sim.step(control, torque_enable, self.dt)

    @property
    def state(self):
        return float(np.asarray(self.sim.q).reshape(-1)[0]), float(np.asarray(self.sim.dq).reshape(-1)[0])

    def reset(self, q=0.0, dq=0.0):
        self.sim.reset(q, dq)


def _load(log_path: Path):
    """The recorded log plus the per-step goal sequence both sides will follow."""
    _bam()
    log = load_raw_log(str(log_path))
    entries = log["entries"][:N_STEPS]
    goals = [entry["goal_position"] for entry in entries]
    return log, entries, goals


def _run_pair(log, entries, goals, bam_model, our_motor, our_friction):
    """Drive both sides and return their recorded trajectories."""
    bam_actuator = bam_model.actuator

    # BAM's rollout calls load_log: it attaches the testbench, and lets each motor's
    # *own* override decide which firmware the recording supplies. That differs per
    # motor - DCMotorActuator.load_log sets only kp (and vin); MD01Actuator also takes
    # error_gain and max_pwm; STS3215Actuator overrides nothing, so its error_gain
    # stays at the class default 0.166 no matter what the log says. Reading the
    # values back instead of assuming keeps our side exactly in step.
    bam_actuator.load_log(log)
    our_motor.set_params(
        {
            "kp": bam_actuator.kp,
            "vin": bam_actuator.vin,
            "error_gain": bam_actuator.error_gain,
            "max_pwm": bam_actuator.max_pwm,
        }
    )

    testbench = bam_actuator.testbench
    ours = OfflineSimulator(
        testbench=testbench,
        motor=our_motor,
        friction=our_friction,
        dt=log["dt"],
        q_offset=our_motor.q_offset,
    )
    reference = BamSide(bam_model, log["dt"])

    start = (entries[0]["position"], entries[0].get("speed", 0.0))
    ours.reset(*start)
    reference.reset(*start)

    return drive(ours, entries, goals), drive(reference, entries, goals)


# ----------------------------------------------------------------------
# Case A: MD01 - stateless P law + firmware current limiter
# ----------------------------------------------------------------------


@pytest.fixture
def md01_pair():
    _bam()
    from bam.actuators import actuators
    from bam.model import models

    log, entries, goals = _load(LOG_DIR / LOG_NAME)

    # No identified params file exists for MD01 in BAM, so seed the motor values
    # identically on both sides - this is a test of our composition, not of physics.
    motor_params = {"kt": 0.5, "R": 1.0, "armature": 1e-4}

    bam_model = models["m5"]()
    bam_model.set_actuator(actuators["md01"]())
    for key, value in motor_params.items():
        getattr(bam_model, key).value = value

    our_motor = get_motor("md01")()
    our_motor.set_params(motor_params)
    our_friction = BamFrictionModel.from_bam_model(bam_model)

    ours, reference = _run_pair(log, entries, goals, bam_model, our_motor, our_friction)
    return ours, reference


def test_md01_trajectory_matches_bam_on_a_real_log(md01_pair):
    ours, reference = md01_pair

    np.testing.assert_allclose(ours["positions"], reference["positions"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(ours["velocities"], reference["velocities"], rtol=0, atol=1e-9)


def test_md01_control_signal_matches_bam_step_by_step(md01_pair):
    ours, reference = md01_pair

    np.testing.assert_allclose(ours["controls"], reference["controls"], rtol=1e-12, atol=1e-15)


def test_md01_run_is_not_degenerate(md01_pair):
    """Guard against a vacuous comparison: the pendulum must actually move.

    Note the PWM clamp is never reached here - the firmware current limiter bites
    first (``max_current=1.4`` bounds the duty cycle to roughly +/-0.12 duty at
    rest), which is exactly the behaviour being ported.
    """
    ours, _ = md01_pair
    diagnostics = ours["diagnostics"]

    assert np.ptp(ours["positions"]) > 1e-3, "the pendulum should move"
    assert np.ptp(ours["controls"]) > 1e-3, "the control should vary"
    assert np.ptp(diagnostics["frictionloss"]) > 1e-6, "friction budget should vary"

    duty = ours["controls"] / 12.0  # log vin
    window = 1.0 * 1.4 / 12.0  # R * max_current / vin at rest
    assert np.all(np.abs(duty) <= window + 1e-6), "duty must stay inside the current window"


# ----------------------------------------------------------------------
# Case B: STS3215 with the bundled m5 - stateful slew limiter, load-dependent
# directional friction (exercises external_torque)
# ----------------------------------------------------------------------


@pytest.fixture
def sts3215_pair():
    _bam()
    from bam.model import load_model

    log, entries, goals = _load(LOG_DIR / LOG_NAME)
    params_path = resolve_params_file("sts3215/m5")

    bam_model = load_model(params_path)

    our_motor = get_motor("sts3215")()
    our_motor.set_params(json.loads(Path(params_path).read_text()))
    our_friction = BamFrictionModel.from_json(params_path)

    ours, reference = _run_pair(log, entries, goals, bam_model, our_motor, our_friction)
    return ours, reference


def test_sts3215_m5_trajectory_matches_bam_on_a_real_log(sts3215_pair):
    ours, reference = sts3215_pair

    np.testing.assert_allclose(ours["positions"], reference["positions"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(ours["velocities"], reference["velocities"], rtol=0, atol=1e-9)


def test_sts3215_m5_control_signal_matches_bam_step_by_step(sts3215_pair):
    ours, reference = sts3215_pair

    np.testing.assert_allclose(ours["controls"], reference["controls"], rtol=1e-12, atol=1e-15)


def test_external_torque_reaches_the_load_dependent_friction(sts3215_pair):
    """The m5 budget must actually depend on the external (gravity) torque.

    Without an external torque the load-dependent terms vanish, which is exactly
    the failure mode Isaac Lab will hit until an env calls ``set_external_torque``.
    """
    ours, _ = sts3215_pair

    frictionloss = ours["diagnostics"]["frictionloss"]
    bias = ours["diagnostics"]["bias_torque"]

    assert np.ptp(bias) > 1e-3, "the pendulum should swing (non-constant gravity torque)"
    # friction_base alone is ~0.05; the load-dependent terms make the budget vary.
    correlated = np.corrcoef(np.abs(bias), frictionloss - frictionloss.min())[0, 1]
    assert correlated > 0.5, f"friction budget does not track the load (r={correlated:.3f})"


# ----------------------------------------------------------------------
# The q_offset convention
# ----------------------------------------------------------------------


def test_q_offset_convention_is_mirrored():
    """BAM applies q_offset to the physics only, not to the control law.

    The identified q_offset is only meaningful if our side resolves it the same
    way, so this pins the convention with a deliberately large offset.
    """
    _bam()
    from bam.actuators import actuators
    from bam.model import models
    from bam.simulate import Simulator

    log, entries, goals = _load(LOG_DIR / LOG_NAME)

    offset = 0.05
    bam_model = models["m5"]()
    bam_model.set_actuator(actuators["md01"]())
    bam_model.kt.value, bam_model.R.value, bam_model.armature.value = 0.5, 1.0, 1e-4
    bam_model.q_offset.value = offset

    our_motor = get_motor("md01")()
    our_motor.set_params({"kt": 0.5, "R": 1.0, "armature": 1e-4, "q_offset": offset})
    our_friction = BamFrictionModel.from_bam_model(bam_model)

    ours, reference = _run_pair(log, entries, goals, bam_model, our_motor, our_friction)

    np.testing.assert_allclose(ours["positions"], reference["positions"], rtol=0, atol=1e-9)

    # And the offset must matter: with it, the control laws now see a joint angle
    # that differs from the physical one, so the trajectory must differ from the
    # offset-free run.
    bam_model.q_offset.value = 0.0
    our_motor.set_params({"q_offset": 0.0})
    ours_zero, _ = _run_pair(log, entries, goals, bam_model, our_motor, our_friction)

    assert not np.allclose(ours["positions"], ours_zero["positions"], rtol=0, atol=1e-9)
