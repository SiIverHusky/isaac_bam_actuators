# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Offline rollout harness: BAM's pendulum loop, driven by *our* components.

``bam.simulate.Simulator`` integrates a single-axis pendulum testbench::

    bias         = testbench.bias(q + q_offset)
    motor_torque = actuator.torque(control, enable, q + q_offset, dq)
    frictionloss, damping = model.frictions(motor_torque, bias, dq)
    inertia      = testbench.mass(q + q_offset) + actuator.extra_inertia
    tau_stop     = (inertia / dt) * dq + motor_torque + bias
    net          = motor_torque + bias
                 - sign(tau_stop) * min(|tau_stop|, frictionloss + damping * |dq|)
    dq += (net / inertia) * dt ;  dq = clip(dq, -100, 100) ;  q += dq * dt

:class:`OfflineSimulator` is the same loop with the control law and the friction
budget supplied by this extension instead of by BAM. Driving both through
:func:`drive` - same goal sequence, same initial state, same ordering - makes any
difference attributable to our components rather than to the integration scheme.

.. important::
    BAM applies ``q_offset`` to the **physics only** (bias torque and motor torque)
    and passes the *raw* joint angle to the control law - see
    ``Simulator.rollout_log``, which calls ``compute_control(goal, self.q, ...)``
    while ``step`` uses ``self.q + q_offset``. This harness reproduces that
    exactly, because it is the convention the identified parameters were fitted
    with.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

#: Velocity clamp applied by BAM after each integration step.
MAX_VELOCITY = 100.0


def raw_dt(entries: list[dict]) -> float:
    """Timestep of a raw (unprocessed) log, as the median of its timestamp gaps.

    ``bam.process`` resamples raw logs onto a uniform grid; our recorded logs in
    ``data_raw/`` are pre-processing, so we take the median gap. Both sides of a
    comparison get the same value, so the idealisation cancels out.
    """
    timestamps = np.array([entry["timestamp"] for entry in entries], dtype=np.float64)
    return float(np.median(np.diff(timestamps)))


def load_raw_log(path: str) -> dict:
    """Load a raw recorded log and annotate it with its timestep."""
    import json

    with open(path) as f:
        log = json.load(f)
    log["dt"] = raw_dt(log["entries"])
    log["filename"] = path
    return log


def _as_float(value: Any) -> float:
    """Scalar float from a torch tensor, numpy scalar or Python number."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0])
    return float(np.asarray(value).reshape(-1)[0])


class OfflineSimulator:
    """BAM's pendulum loop with our motor control law and friction budget.

    :param testbench: Object exposing ``compute_bias(q, dq)`` and
        ``compute_mass(q, dq)`` - BAM's ``testbench.Pendulum`` is the reference.
    :param motor: A :class:`bam_actuators.motors.base.MotorBase` instance.
    :param friction: A :class:`bam_actuators.friction.BamFrictionModel`.
    :param dt: Timestep [s].
    :param q_offset: Rig offset, applied to the physics only (see module docstring).
    :param dtype: Torch dtype. Use float64 to compare against BAM's numpy float64.
    """

    def __init__(
        self,
        testbench: Any,
        motor: Any,
        friction: Any,
        dt: float,
        q_offset: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.testbench = testbench
        self.motor = motor
        self.friction = friction
        self.dt = dt
        self.q_offset = q_offset
        self.dtype = dtype

        self.q = torch.zeros((), dtype=dtype)
        self.dq = torch.zeros((), dtype=dtype)
        #: Per-step intermediate values, for exact component-level comparison.
        self.diagnostics: dict[str, float] | None = None

    # -- protocol shared with the BAM-side wrapper ----------------------

    def control(self, goal: Any):
        """Control signal for a goal position.

        The control law sees the *raw* joint angle, matching BAM's rollout.
        """
        return self.motor.control(_tensor(goal, self.dtype), self.q, self.dq, self.dt)

    def step(self, control: Any, torque_enable: bool) -> None:
        """Advance one timestep (identical arithmetic and ordering to BAM)."""
        dt = self.dt
        q = self.q + self.q_offset

        bias = self.testbench.compute_bias(_as_float(q), _as_float(self.dq))
        motor_torque = self.motor.torque(control, torque_enable, q, self.dq)
        frictionloss, damping = self.friction.compute(motor_torque, bias, self.dq)
        inertia = self.testbench.compute_mass(_as_float(q), _as_float(self.dq)) + self.motor.armature

        net_torque = motor_torque + bias
        tau_stop = (inertia / dt) * self.dq + net_torque
        budget = frictionloss + damping * self.dq.abs()
        net_torque = net_torque - torch.sign(tau_stop) * torch.minimum(tau_stop.abs(), budget)

        self.dq = torch.clamp(self.dq + (net_torque / inertia) * dt, -MAX_VELOCITY, MAX_VELOCITY)
        self.q = self.q + self.dq * dt

        self.diagnostics = {
            "control": _as_float(control),
            "motor_torque": _as_float(motor_torque),
            "bias_torque": float(bias),
            "frictionloss": _as_float(frictionloss),
            "damping": _as_float(damping),
            "inertia": _as_float(inertia),
        }

    @property
    def state(self) -> tuple[float, float]:
        return _as_float(self.q), _as_float(self.dq)

    def reset(self, q: float = 0.0, dq: float = 0.0) -> None:
        self.q = _tensor(q, self.dtype)
        self.dq = _tensor(dq, self.dtype)
        self.motor.reset()
        self.diagnostics = None


def _tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype)
    return torch.tensor(float(value), dtype=dtype)


def drive(side: Any, entries: list[dict], goals: list[float]) -> dict[str, np.ndarray]:
    """Step ``side`` over a log, recording the trajectory and diagnostics.

    Mirrors ``bam.simulate.Simulator.rollout_log``'s ordering: the state is
    recorded *before* the step, and the control law is evaluated at the state the
    controller would have seen.

    :param side: :class:`OfflineSimulator`, or any object with the same
        ``control`` / ``step`` / ``state`` / ``diagnostics`` protocol.
    :param entries: Log entries (``goal_position`` is overridden by ``goals``).
    :param goals: Goal position per step - pass the same list to both sides.
    :returns: Arrays ``positions``, ``velocities``, ``controls`` and a dict of
        per-step diagnostic arrays.
    """
    positions, velocities, controls = [], [], []
    diagnostics: dict[str, list[float]] = {}

    for entry, goal in zip(entries, goals):
        control = side.control(goal)

        positions.append(side.state[0])
        velocities.append(side.state[1])
        controls.append(_as_float(control))

        side.step(control, entry["torque_enable"])

        for key, value in (side.diagnostics or {}).items():
            diagnostics.setdefault(key, []).append(value)

    return {
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "controls": np.array(controls),
        "diagnostics": {key: np.array(values) for key, values in diagnostics.items()},
    }
