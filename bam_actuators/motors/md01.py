# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Mangdang MD01 control laws.

Ported from ``bam/mangdang/actuator.py``, which models the same servo three ways.
They differ in *how much of the AT32 driver board* is modelled, and each has a name
of its own, so a params file's ``"actuator"`` key selects the right one:

* ``md01`` - voltage P law (``duty = clip(kp * error_gain * dq, +/-max_pwm)``) with
  the firmware current limit modelled as a duty-cycle window. Unit: volts.
* ``md01i`` - current setpoint ``I = clip(kp * error_gain * ratio * dq,
  +/-current_limit)``, then the H-bridge duty clamp. Unit: amps.
* ``md01c`` - the *measured* AT32 position and current loops, with the stall plant
  ``I = (duty * vin - V0 * sign) / R``. Unit: amps.

``md01i`` is the law the bundled fits use (``params_file="md01i/m3"``): on the
servo-6 bench recordings the measured current turned out to be a function of
``kp * error`` alone, so the current law reproduces the recorded current where the
voltage law needs 3-10x the real current to match the same angles. ``md01c`` adds
the two nested loops measured at a blocked output (2026-09-21), which pins its
plant (``R``, ``V0``) rather than fitting it.

.. note::

    ``md01`` follows BAM's ``MD01Actuator`` exactly, and BAM's class declares no
    ``error_gain_ratio``, so a fitted ratio in an ``md01`` params file is ignored
    on both sides. The ratio *does* apply under ``md01i``/``md01c``.

.. note::

    All three are stateless: the AT32's position loop settles within a control
    period, which BAM absorbs with the ``command_delay`` rig parameter rather than
    with a stateful target (compare the STS3215's slew-limited one).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from .base import ArrayLike, MotorBase, _clamp, _sign

#: Degrees per radian, for the ``md01c`` loop, whose gains are per degree.
_DEG_PER_RAD = 180.0 / math.pi


class MD01Motor(MotorBase):
    """Mangdang MD01 as a plain voltage-controlled servo (name ``md01``)."""

    name: ClassVar[str] = "md01"
    stateful: ClassVar[bool] = False
    control_unit: ClassVar[str] = "volts"

    firmware: ClassVar[dict[str, Any]] = {
        # Nominal MD01 bus voltage; the driver board has no rail ADC.
        "vin": 12.0,
        "kp": 80.0,
        # Measured with an oscilloscope (see BAM's ADDING_A_MOTOR.md).
        "error_gain": 0.0104,
        # Still inherited from the measurements on the other BAM voltage-controlled
        # servos - BAM carries a TODO to measure it on the MD01.
        "max_pwm": 1.0,
        # Effective current limit [A]: the 900 mA cap the recorder sends plateaus at
        # ~440 mA on the bench, which is the limit the pendulum actually sees.
        "max_current": 0.45,
    }

    parameters: ClassVar[dict[str, Any]] = {
        # Bench stall gives kt ~ 0.5 Nm/A at the output; the fit trades it against R.
        "kt": 0.5,
        "R": 10.0,
        "armature": 3e-4,
        "q_offset": 0.0,
        "command_delay": 0.0,
    }

    def control(self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float | None) -> ArrayLike:
        """P law -> duty cycle, then the firmware current limit, then the PWM limit."""
        duty_cycle = self._duty_from_error(q_target - q)
        duty_cycle = self._apply_current_limit(duty_cycle, dq)
        return self._volts(duty_cycle)


class MD01CurrentMotor(MotorBase):
    """Mangdang MD01 as a current-controlled servo (name ``md01i``).

    The AT32 runs a position loop whose output is a *current* setpoint for an inner
    current loop, so the control signal is a current and the torque is ``kt * I``.
    ``error_gain`` converts ``kp * error`` into amps; on servo 3 it was measured at
    ~30 mA per rad of error per unit of ``kp``, linear up to a saturation that the
    fit captures as ``current_limit``.

    This is the law the bundled ``md01i`` fits were identified with, and the one
    whose ``kt`` is only a gauge: rescaling it together with ``error_gain_ratio``
    and ``current_limit`` leaves the position error unchanged. The physical
    quantity is the torque ceiling ``kt * current_limit``.
    """

    name: ClassVar[str] = "md01i"
    stateful: ClassVar[bool] = False
    control_unit: ClassVar[str] = "amps"

    firmware: ClassVar[dict[str, Any]] = {
        "vin": 12.0,
        "kp": 80.0,
        # A per rad of error per unit of kp; measured on servo 3 in the linear
        # region of the I(kp * error) curve at |dq| < 0.3 rad/s.
        "error_gain": 0.030,
        "max_pwm": 1.0,
    }

    parameters: ClassVar[dict[str, Any]] = {
        "kt": 0.5,
        "R": 10.0,
        "armature": 3e-4,
        # Sustained current limit of the current loop as configured [A]. A bench
        # property, not a rating: the robot's presets reach ~0.9 A.
        "current_limit": 0.45,
        # Scale on error_gain, so the measured 0.030 is only a starting point.
        "error_gain_ratio": 1.0,
        "q_offset": 0.0,
        "command_delay": 0.0,
    }

    @property
    def torque_limit(self) -> float:
        """Torque ceiling at the current cap [Nm] - the physically meaningful value.

        ``kt`` and ``error_gain_ratio`` are only a gauge in this law, so this product
        is what the fit actually pins (~0.43-0.46 Nm for the bundled fits).
        """
        return self.kt * self.current_limit

    def control(self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float | None) -> ArrayLike:
        """P law -> current setpoint, cap it, then let the H-bridge deliver it."""
        current = (q_target - q) * self.kp * self.error_gain * getattr(self, "error_gain_ratio", 1.0)
        current = _clamp(current, -self.current_limit, self.current_limit)
        return self._current_from_duty(current, dq)


class MD01LoopMotor(MotorBase):
    """MD01 with the AT32's measured position and current loops (name ``md01c``).

    Both loops were measured at a blocked output on 2026-09-21, exact to three
    digits over three gain sets::

        i_set = clip(kp_position [mA/deg] * error_deg, +/-cap)
        duty  = clip(kp_current * (i_set - i) + kff_current * i_set, +/-max_pwm)

    against the stall plant ``i = (duty * vin - V0 * sign) / R``. Rather than
    integrating the loops, the steady state is solved per step:

    .. math::

        i = \\text{sign}(x)\\,\\frac{\\max(|x| - V_0, 0)}{R + v_{in} k_{p,c}},
        \\qquad x = v_{in} (k_{p,c} + k_{ff}) i_{set} - k_t \\dot{q}

    The firmware constants come from the log (``kp``, ``kp_current``,
    ``kff_current``, ``max_pwm``, ``cap_ma``), so a change of preset needs no
    refit. The plant is measured rather than fitted (see BAM's
    ``MD01LoopActuator``), and ``kp_ratio`` is a check parameter expected to fit
    near 1.

    With the default seeds this runs with no params file at all; point it at a
    fitted ``md01c`` params file to adapt it to a particular unit.
    """

    name: ClassVar[str] = "md01c"
    stateful: ClassVar[bool] = False
    control_unit: ClassVar[str] = "amps"

    firmware: ClassVar[dict[str, Any]] = {
        "vin": 12.0,
        "kp": 80.0,
        # The law works in degrees, so error_gain is a no-op here; BAM sets it to 1.
        "error_gain": 1.0,
        # Flash values read on 2026-09-21; a log overrides them when it carries them.
        "kp_current": 6e-4,
        "kff_current": 3e-4,
        "max_pwm": 0.99,
        # Frame current cap [mA]: the robot's preset, not the bench's 900 mA.
        "cap_ma": 1500.0,
    }

    parameters: ClassVar[dict[str, Any]] = {
        "kt": 0.6,
        # The stall-test spread: R_eff 11-13.4 ohm cold to warm, V0 ~ 1.2 V.
        "R": 11.5,
        "V0": 1.2,
        "armature": 3e-4,
        "kp_ratio": 1.0,
        "q_offset": 0.0,
        "command_delay": 0.0,
    }

    def control(self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float | None) -> ArrayLike:
        """Solve the two loops at steady state and return the current reached [A]."""
        kt = self.kt
        R = self.R
        V0 = self.V0
        vin = self.vin

        # Position loop: current setpoint [A], capped by the frame's cap.
        cap = self.cap_ma / 1000.0
        i_set = (q_target - q) * _DEG_PER_RAD * self.kp * self.kp_ratio / 1000.0
        i_set = _clamp(i_set, -cap, cap)

        # Current loop (P + feed-forward, gains per mA -> per A) in steady state
        # with the plant i = (duty * vin - V0 * sign - kt * dq) / R:
        #   i (R + vin kp_c) = vin (kp_c + kff) i_set - V0 sign(x) - kt dq
        kp_c = self.kp_current * 1000.0
        kff = self.kff_current * 1000.0
        x = vin * (kp_c + kff) * i_set - kt * dq
        sign = _sign(x)
        magnitude = _clamp(sign * x - V0, 0.0, math.inf)
        current = sign * magnitude / (R + vin * kp_c)

        # What the bridge can actually apply: clip the duty and recompute.
        duty = (current * R + V0 * sign + kt * dq) / vin
        duty = _clamp(duty, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty
        return (duty * vin - V0 * sign - kt * dq) / R

