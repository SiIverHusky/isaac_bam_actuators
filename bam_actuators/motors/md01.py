# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Mangdang MD01 control law.

Ported from ``bam/mangdang/actuator.py``, which subclasses BAM's
``VoltageControlledActuator``: a plain firmware P law on the position error, with
the firmware current limiter modelled as a bound on the duty cycle.

.. note::

    BAM's MD01 declares an ``error_gain_ratio`` parameter in ``initialize()`` but
    never applies it - only :class:`bam.feetech.actuator.STS3215Actuator` and
    ``ST3025Actuator`` multiply it into the duty cycle. This port therefore does
    **not** declare it, which matches what BAM's simulator actually computes: a
    fitted ``error_gain_ratio`` in an MD01 params file is ignored on both sides.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import ArrayLike, MotorBase


class MD01Motor(MotorBase):
    """Mangdang MD01 voltage-controlled servo."""

    name: ClassVar[str] = "md01"
    stateful: ClassVar[bool] = False
    control_unit: ClassVar[str] = "volts"

    firmware: ClassVar[dict[str, Any]] = {
        # Nominal MD01 bus voltage; the driver board has no rail ADC.
        "vin": 12.0,
        "kp": 32.0,
        # Measured with an oscilloscope (see BAM's ADDING_A_MOTOR.md).
        "error_gain": 0.0104,
        "max_pwm": 0.97,
        # The AT32 reads the torque field of a position command as a max current cap.
        "max_current": 1.4,
    }

    parameters: ClassVar[dict[str, Any]] = {
        # BAM seeds kt at 0.0 with a TODO for the datasheet value, so an unfitted
        # MD01 produces no torque. Supply a fitted params file to get motion.
        "kt": 0.0,
        "R": 1.0,
        "armature": 0.0001,
        "q_offset": 0.0,
    }

    def control(self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float | None) -> ArrayLike:
        """P law -> duty cycle, then the firmware current limit, then the PWM limit."""
        duty_cycle = self._duty_from_error(q_target - q)
        duty_cycle = self._apply_current_limit(duty_cycle, dq)
        return self._volts(duty_cycle)
