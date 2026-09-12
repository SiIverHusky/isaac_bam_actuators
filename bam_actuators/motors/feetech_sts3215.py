# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Feetech STS3215 (7.4 V) control law.

Ported from ``bam/feetech/actuator.py``. Two things make this servo different from
a plain P controller:

1. the firmware **rate-limits its internal target position** at ``max_velocity``,
   so a step command is slewed rather than followed instantly. That internal
   target is state, which makes this motor :attr:`stateful <MotorBase.stateful>`.
2. the identified gain is scaled by ``error_gain_ratio`` on top of ``error_gain``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

from .base import ArrayLike, MotorBase, _clamp

#: Firmware step rate: 3400 steps/s over 4096 steps/rev.
DEFAULT_MAX_VELOCITY = (3400 * 2 * math.pi) / 4096


class STS3215Motor(MotorBase):
    """Feetech STS3215 with the firmware's slew-limited target position."""

    name: ClassVar[str] = "sts3215"
    stateful: ClassVar[bool] = True
    control_unit: ClassVar[str] = "volts"

    firmware: ClassVar[dict[str, Any]] = {
        "vin": 7.4,
        "kp": 32.0,
        # Determined with an oscilloscope; converts kp * dq into a duty cycle.
        "error_gain": 0.166,
        "max_pwm": 0.97,
        "max_current": None,
    }

    parameters: ClassVar[dict[str, Any]] = {
        "kt": 0.784532,
        "R": 2.0,
        "armature": 0.0001,
        "error_gain_ratio": 1.0,
        "q_offset": 0.0,
        "max_velocity": DEFAULT_MAX_VELOCITY,
        "command_delay": 0.0,
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        #: Firmware's internal target position. Lazily seeded to the current joint
        #: position on the first call (the way the servo behaves when it powers
        #: on), so it inherits the dtype and shape of the state it is fed.
        self.q_target_smooth: ArrayLike | None = None

        #: Environments whose internal target still has to be re-seeded.
        self._to_reset_env_ids: list[Any] = []

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | slice | None = ...) -> None:
        """Defer a re-seed of the internal target to the next :meth:`control`.

        The internal target is a joint position, which is only known when the
        controller runs, so the reset cannot be applied here.
        """
        if env_ids is ...:
            self.q_target_smooth = None
            self._to_reset_env_ids.clear()
        else:
            self._to_reset_env_ids.append(env_ids)

    def get_state(self):
        """The internal target plus any pending re-seeds."""
        return self.q_target_smooth, list(self._to_reset_env_ids)

    def set_state(self, state) -> None:
        """Restore a snapshot from :meth:`get_state`."""
        q_target_smooth, pending = (None, []) if state is None else state
        self.q_target_smooth = q_target_smooth
        self._to_reset_env_ids = list(pending)

    # ------------------------------------------------------------------
    # Control law
    # ------------------------------------------------------------------

    def control(self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float | None) -> ArrayLike:
        """Slew the internal target towards ``q_target``, then run the P law on it.

        :param dt: Timestep [s]. **Required** - the slew rate is ``max_velocity * dt``.
        :raises ValueError: If ``dt`` is ``None``.
        """
        if dt is None:
            raise ValueError(
                "The STS3215 firmware rate-limits its internal target, so `dt` is required. "
                "Set `physics_dt` on the actuator config."
            )

        q_target_smooth = self.q_target_smooth
        if q_target_smooth is None:
            # First call, or first call after a full reset.
            q_target_smooth = q
            self._to_reset_env_ids.clear()
        elif self._to_reset_env_ids:
            # Environments teleported since the last call: their internal target
            # has to follow their new position.
            for env_ids in self._to_reset_env_ids:
                q_target_smooth[env_ids] = q[env_ids]
            self._to_reset_env_ids.clear()

        max_step = self.max_velocity * dt
        self.q_target_smooth = _clamp(q_target, q_target_smooth - max_step, q_target_smooth + max_step)

        duty_cycle = self._duty_from_error(self.q_target_smooth - q)
        return self._volts(duty_cycle)
