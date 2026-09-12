# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""A generic voltage-controlled servo: plain P law plus an optional current cap.

This mirrors BAM's ``VoltageControlledActuator`` and is the fallback for servos
that have no dedicated module. It is stateless: the position error is turned into
a duty cycle every step with no internal filtering.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import ArrayLike, MotorBase


class GenericServoMotor(MotorBase):
    """Voltage-controlled servo with a firmware P position controller."""

    name: ClassVar[str] = "generic"
    stateful: ClassVar[bool] = False
    control_unit: ClassVar[str] = "volts"

    firmware: ClassVar[dict[str, Any]] = {
        "vin": 12.0,
        "kp": 32.0,
        "error_gain": 1.0,
        "max_pwm": 1.0,
        "max_current": None,
    }

    parameters: ClassVar[dict[str, Any]] = {
        "kt": 0.1,
        "R": 1.0,
        "armature": 0.0,
        "q_offset": 0.0,
    }

    def control(self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float | None) -> ArrayLike:
        """P law -> duty cycle, optionally current-limited, clipped to the PWM range."""
        duty_cycle = self._duty_from_error(q_target - q)
        duty_cycle = self._apply_current_limit(duty_cycle, dq)
        return self._volts(duty_cycle)
