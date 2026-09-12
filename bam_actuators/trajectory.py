# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""BAM's identification trajectories.

These are the motions ``bam.trajectory`` drives a testbench with while recording
the data the friction models are fitted to. This is a faithful port - same names,
same durations, same arithmetic - so a params file can be exercised against
exactly the motions it was identified from. ``tests/test_trajectory.py`` pins the
port against BAM's own implementation sample-by-sample.

A trajectory is a callable ``t -> (angle [rad], torque_enable)``. The second
element mirrors BAM's per-step ``torque_enable`` flag: ``False`` means the servo
is unpowered, so it produces no torque while gravity and friction still act. Only
:class:`LiftAndDrop` and :class:`Nothing` use it.

Deliberately torch-free and Isaac-free, so it can drive the offline harness and
the Isaac scenes alike.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np


def cubic_interpolate(keyframes: list[list[float]], t: float) -> float:
    """Interpolate a scalar signal through keyframes with cubic splines.

    Each keyframe is a triplet ``[t, x, dx/dt]``. Outside the keyframe range the
    signal is held at the nearest end value.

    The 4x4 system is solved from scratch on every call, exactly as BAM does. The
    trajectories are sampled a few thousand times per run, so caching is not worth
    the state it would add.

    :param keyframes: ``[t, x, dx/dt]`` triplets, sorted by time.
    :param t: Query time.
    :returns: Interpolated value at time ``t``.
    """
    if t < keyframes[0][0]:
        return float(keyframes[0][1])
    if t > keyframes[-1][0]:
        return float(keyframes[-1][1])

    for i in range(len(keyframes) - 1):
        if keyframes[i][0] <= t <= keyframes[i + 1][0]:
            t0, x0, x0p = keyframes[i]
            t1, x1, x1p = keyframes[i + 1]

            a = [
                [1, t0, t0**2, t0**3],
                [0, 1, 2 * t0, 3 * t0**2],
                [1, t1, t1**2, t1**3],
                [0, 1, 2 * t1, 3 * t1**2],
            ]
            b = [x0, x0p, x1, x1p]
            w = np.linalg.solve(a, b)

            return float(w[0] + w[1] * t + w[2] * t**2 + w[3] * t**3)

    # Unreachable - the two clamps above cover everything outside the range.
    return float(keyframes[-1][1])


class Trajectory:
    """A recorded identification motion.

    All of BAM's built-in trajectories run for 6 seconds.
    """

    #: Nominal length of the motion [s].
    duration: ClassVar[float | None] = None

    def __call__(self, t: float) -> tuple[float, bool]:
        """Return ``(angle [rad], torque_enable)`` at time ``t`` [s]."""
        raise NotImplementedError


class LiftAndDrop(Trajectory):
    """Cubic move to -π/2 over 2 s, then torque disabled (gravity drop).

    Identifies backdrivability and Stribeck effects at very low speed, since the
    arm falls freely under gravity once the motor is released.
    """

    duration = 6.0

    def __call__(self, t: float) -> tuple[float, bool]:
        keyframes = [[0.0, 0.0, 0.0], [2.0, -np.pi / 2, 0.0]]
        return cubic_interpolate(keyframes, t), t < 2.0


class SinusTimeSquare(Trajectory):
    """Progressively faster sinusoid, :math:`\\sin(t^2)`.

    Sweeps a wide velocity range in a single run. BAM recommends it as the primary
    identification trajectory.
    """

    duration = 6.0

    def __call__(self, t: float) -> tuple[float, bool]:
        return float(np.sin(t**2)), True


class UpAndDown(Trajectory):
    """Slow cubic path 0 → π/2 → 0.8·π/2.

    Emphasises static friction and load-dependent effects at low to medium speed.
    """

    duration = 6.0

    def __call__(self, t: float) -> tuple[float, bool]:
        keyframes = [
            [0.0, 0.0, 0.0],
            [3.0, np.pi / 2, 0.0],
            [6.0, 0.8 * np.pi / 2, 0.0],
        ]
        return cubic_interpolate(keyframes, t), True


class HalfSine(Trajectory):
    """Slow half-sine path 0 → π/2."""

    duration = 6.0

    def __call__(self, t: float) -> tuple[float, bool]:
        return float(np.sin(t / 2) * np.pi / 2), True


class Steps(Trajectory):
    """A small staircase: 0 → 0.75 → 1.55 → 0.75 → 0 rad.

    Steps are what exercise a stateful control law (the STS3215's slew-limited
    internal target) hardest, because the command jumps faster than the firmware
    can follow.
    """

    duration = 6.0

    #: ``(time [s], angle [rad])`` breakpoints. The angle holds until the next one.
    stages = ((0.0, 0.0), (1.0, 0.75), (2.5, 1.55), (4.0, 0.75), (5.0, 0.0), (6.0, 0.0))

    def __call__(self, t: float) -> tuple[float, bool]:
        for (_, angle), (next_time, _) in zip(self.stages, self.stages[1:]):
            if t <= next_time:
                return angle, True
        # Past the end. BAM's version falls out of its loop and returns None here;
        # holding the final angle is the only sane reading of the same motion.
        return self.stages[-1][1], True


class SinSin(Trajectory):
    """Multi-frequency path, :math:`\\sin(t)\\pi/2 + \\sin(5t)\\cdot 0.5\\sin(2t)`.

    Rich spectral content covers a broad range of velocities and accelerations.
    """

    duration = 6.0

    def __call__(self, t: float) -> tuple[float, bool]:
        angle = np.sin(t) * np.pi / 2 + np.sin(5.0 * t) * 0.5 * np.sin(t * 2.0)
        return float(angle), True


class Nothing(Trajectory):
    """Zero torque for the full duration - the pure gravity response.

    Isolates backdrivability and the passive dynamics.
    """

    duration = 6.0

    def __call__(self, t: float) -> tuple[float, bool]:
        return 0.0, False


#: Every built-in trajectory, keyed by the name used on the command line. Mirrors
#: BAM's ``bam.trajectory.trajectories``, minus its unused ``--trajectory`` help.
TRAJECTORIES: dict[str, Trajectory] = {
    "lift_and_drop": LiftAndDrop(),
    "sin_time_square": SinusTimeSquare(),
    "up_and_down": UpAndDown(),
    "sin_sin": SinSin(),
    "half_sine": HalfSine(),
    "steps": Steps(),
    "nothing": Nothing(),
}


def get_trajectory(name: str) -> Trajectory:
    """Look up a trajectory by name.

    :param name: One of the keys of :data:`TRAJECTORIES`.
    """
    try:
        return TRAJECTORIES[name]
    except KeyError:
        known = ", ".join(sorted(TRAJECTORIES))
        raise KeyError(f"unknown trajectory {name!r}; known trajectories: {known}") from None


def steps_for(trajectory: Trajectory, dt: float) -> int:
    """Number of samples covering the trajectory's whole duration at ``dt``."""
    if trajectory.duration is None:
        raise ValueError("trajectory has no duration")
    return int(round(trajectory.duration / dt))


def sample(
    trajectory: Trajectory, dt: float, steps: int | None = None, start: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a trajectory into arrays, for driving a simulator.

    :param trajectory: The trajectory to sample.
    :param dt: Timestep [s].
    :param steps: Number of samples. Defaults to the whole trajectory duration.
    :param start: Time of the first sample [s].
    :returns: ``(times, angles, enables)``, each of length ``steps``.
    """
    if steps is None:
        steps = steps_for(trajectory, dt)

    times = start + np.arange(steps, dtype=float) * dt
    angles = np.empty(steps, dtype=float)
    enables = np.empty(steps, dtype=bool)
    for i, t in enumerate(times):
        angles[i], enables[i] = trajectory(float(t))
    return times, angles, enables
