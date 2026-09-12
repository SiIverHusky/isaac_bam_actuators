# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the port of BAM's identification trajectories.

The port itself needs only numpy, so most of this runs anywhere. The
sample-by-sample comparison against ``bam.trajectory`` needs a BAM checkout (see
``bam_paths``) and skips without one.
"""

import math

import numpy as np
import pytest

from bam_actuators.trajectory import (
    TRAJECTORIES,
    HalfSine,
    LiftAndDrop,
    Nothing,
    SinSin,
    SinusTimeSquare,
    Steps,
    UpAndDown,
    cubic_interpolate,
    get_trajectory,
    sample,
    steps_for,
)

from bam_paths import bam_root

#: Dense enough to catch a wrong branch: a 6 s motion at 5 kHz.
DENSE_DT = 2e-4


def bam_trajectories():
    """BAM's own trajectory registry, or ``None`` if there is no checkout."""
    root = bam_root()
    if not (root / "bam" / "trajectory.py").is_file():
        return None
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from bam.trajectory import trajectories
    except ImportError:
        return None
    return trajectories


# ----------------------------------------------------------------------
# Fidelity against BAM
# ----------------------------------------------------------------------


def test_registry_matches_bam():
    """Same names, same durations - a missing entry would silently skip the parity test."""
    reference = bam_trajectories()
    if reference is None:
        pytest.skip("no BAM checkout; set BAM_ROOT")

    assert sorted(TRAJECTORIES) == sorted(reference), "trajectory registry differs from BAM"
    for name, ours in TRAJECTORIES.items():
        assert ours.duration == reference[name].duration, f"duration differs for {name}"


def test_matches_bam_sample_by_sample():
    """The port must reproduce BAM's angle and torque-enable exactly.

    Exact equality, not a tolerance: this is the same arithmetic on the same
    float64 inputs, so any difference at all is a porting mistake rather than
    rounding. (BAM's ``Steps`` returns ``None`` past its 6 s duration, where the
    port holds the final angle instead, so those samples are skipped.)
    """
    reference = bam_trajectories()
    if reference is None:
        pytest.skip("no BAM checkout; set BAM_ROOT")

    for name, ours in TRAJECTORIES.items():
        theirs = reference[name]
        for t in np.arange(0.0, ours.duration + DENSE_DT, DENSE_DT):
            expected = theirs(float(t))
            if expected is None:
                continue
            angle, enabled = ours(float(t))
            assert angle == float(expected[0]), f"{name}: angle differs at t={t}"
            assert enabled == bool(expected[1]), f"{name}: torque_enable differs at t={t}"


# ----------------------------------------------------------------------
# The trajectories themselves
# ----------------------------------------------------------------------


def test_cubic_interpolate_clamps_outside_the_keyframe_range():
    """Both ends hold, rather than extrapolating a cubic off into space."""
    keyframes = [[0.0, 0.0, 0.0], [2.0, 1.5, 0.0]]

    assert cubic_interpolate(keyframes, -1.0) == 0.0
    assert cubic_interpolate(keyframes, 99.0) == 1.5


def test_cubic_interpolate_hits_its_keyframes_and_slopes():
    """A cubic spline through ``(t, x, x')`` must reproduce the keyframe values."""
    keyframes = [[0.0, 0.0, 0.0], [2.0, -math.pi / 2, 0.0]]

    assert cubic_interpolate(keyframes, 0.0) == pytest.approx(0.0)
    assert cubic_interpolate(keyframes, 2.0) == pytest.approx(-math.pi / 2)


def test_cubic_interpolate_reproduces_the_closed_form_smoothstep():
    """With zero slope at both ends the cubic is Hermite, whose closed form is
    ``X * (3u^2 - 2u^3)`` for ``u = t / T`` - an independent check of the 4x4 solve."""
    end, total = -math.pi / 2, 2.0
    keyframes = [[0.0, 0.0, 0.0], [total, end, 0.0]]

    for t in np.linspace(0.0, total, 50):
        u = t / total
        assert cubic_interpolate(keyframes, t) == pytest.approx(end * (3 * u**2 - 2 * u**3))


def test_steps_reproduces_its_staircase():
    """The staircase, including which stage owns each breakpoint.

    BAM scans with ``t0 <= t <= t1`` and returns the *earlier* stage, so a
    breakpoint resolves to the value the arm already had and the new target only
    takes effect just after it. The port keeps that - it is what the identifier
    saw - so ``t = 1.0`` is still 0.0 rad, not 0.75.
    """
    trajectory = Steps()

    assert trajectory(0.0)[0] == pytest.approx(0.0)
    assert trajectory(0.999)[0] == pytest.approx(0.0)
    assert trajectory(1.0)[0] == pytest.approx(0.0)
    assert trajectory(1.001)[0] == pytest.approx(0.75)
    assert trajectory(2.5)[0] == pytest.approx(0.75)
    assert trajectory(2.501)[0] == pytest.approx(1.55)
    assert trajectory(4.0)[0] == pytest.approx(1.55)
    assert trajectory(4.001)[0] == pytest.approx(0.75)
    assert trajectory(5.0)[0] == pytest.approx(0.75)
    assert trajectory(5.001)[0] == pytest.approx(0.0)
    assert trajectory(6.0)[0] == pytest.approx(0.0)


def test_steps_holds_the_last_angle_past_the_end():
    """BAM falls out of its loop and returns None here; holding is the sane reading."""
    angle, enabled = Steps()(7.5)

    assert angle == pytest.approx(0.0)
    assert enabled is True


def test_only_lift_and_drop_and_nothing_cut_the_power():
    """Which trajectories are unpowered, and when - these back-drive under gravity."""
    assert LiftAndDrop()(0.0)[1] is True
    assert LiftAndDrop()(1.999)[1] is True
    assert LiftAndDrop()(2.0)[1] is False
    assert LiftAndDrop()(5.0)[1] is False

    assert Nothing()(0.0)[1] is False
    assert Nothing()(5.0)[1] is False

    for trajectory in (SinusTimeSquare(), UpAndDown(), HalfSine(), SinSin(), Steps()):
        times = np.arange(0.0, trajectory.duration, 0.1)
        assert all(trajectory(float(t))[1] is True for t in times), type(trajectory).__name__


def test_lift_and_drop_is_a_negative_quarter_turn():
    """BAM lifts to -pi/2 before releasing, not to +pi/2."""
    assert LiftAndDrop()(0.5)[0] < 0.0
    assert LiftAndDrop()(2.0)[0] == pytest.approx(-math.pi / 2)

    # Monotonic on the way up, then held while the arm falls.
    samples = [LiftAndDrop()(t)[0] for t in np.linspace(0.0, 6.0, 200)]
    assert samples[:67] == sorted(samples[:67], reverse=True)


def test_sin_time_square_sweeps_a_growing_frequency():
    """The point of this one is the velocity range, so it must actually accelerate."""
    trajectory = SinusTimeSquare()

    times = np.arange(0.0, 6.0, 1e-3)
    values = np.array([trajectory(t)[0] for t in times])
    is_peak = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    peak_times = times[1:-1][is_peak]

    # sin(t^2) completes more of its oscillation in the second half than the first.
    assert 0 < (peak_times < 3.0).sum() < (peak_times >= 3.0).sum()


def test_every_trajectory_starts_at_zero():
    """No trajectory commands a step at t=0, so the pendulum starts from rest."""
    for name, trajectory in TRAJECTORIES.items():
        assert trajectory(0.0)[0] == pytest.approx(0.0), name


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def test_get_trajectory_rejects_an_unknown_name():
    assert get_trajectory("steps") is TRAJECTORIES["steps"]

    with pytest.raises(KeyError, match="steps"):
        get_trajectory("staircase")


def test_steps_for_covers_the_whole_duration():
    assert steps_for(SinusTimeSquare(), 0.005) == 1200  # 6 s at 200 Hz
    assert steps_for(SinusTimeSquare(), 0.01) == 600


def test_sample_returns_matching_arrays():
    times, angles, enables = sample(SinusTimeSquare(), 0.005, steps=100)

    assert times.shape == angles.shape == enables.shape == (100,)
    assert times[0] == 0.0
    assert times[1] == pytest.approx(0.005)
    assert enables.dtype == np.bool_
    assert enables.all()

    # Defaulting to the whole trajectory is what the scene relies on.
    assert len(sample(SinusTimeSquare(), 0.005)[0]) == 1200


def test_sample_carries_the_torque_enable_boundary():
    """The release instant must land on the right sample, or the arm is dropped early."""
    times, _, enables = sample(LiftAndDrop(), 0.005, steps=1200)

    released = np.flatnonzero(~enables)
    assert released[0] == 400, "torque should be cut at t=2.0 s"
    assert times[released[0]] == pytest.approx(2.0)
    assert enables[:400].all() and not enables[400:].any()
