# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Base class for the per-motor control laws.

A motor owns two things:

* the **firmware**: the control law from joint state to a control signal
  (voltage), plus its constant settings (supply voltage, P gain, PWM limit, ...),
* the **motor parameters**: the identified quantities that vary per unit
  (``kt``, ``R``, ``armature``, ``q_offset``, ...), which BAM stores alongside the
  friction parameters in the same params JSON.

The friction budget is deliberately *not* here: it belongs to
:mod:`bam_actuators.friction` and is model-dependent rather than motor-dependent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import torch

#: Type of the state arguments of the control law: elementwise over a batch.
ArrayLike = Any


def _clamp(value: ArrayLike, low: ArrayLike, high: ArrayLike) -> ArrayLike:
    """``clamp`` that works for both torch tensors and Python floats."""
    if isinstance(value, torch.Tensor):
        return torch.clamp(value, low, high)
    return min(max(value, low), high)


class MotorBase:
    """Control law of one servo family.

    Subclasses declare their constants in :attr:`firmware` and the identified
    quantities they use in :attr:`parameters`, then implement :meth:`control`.
    Values loaded from a params file or set on the config land on the instance as
    plain attributes.

    :param params: Overrides for any name in :attr:`firmware` / :attr:`parameters`
        (typically the contents of a BAM params file, or non-``None`` config
        fields). ``None`` values are ignored so callers can pass a cfg verbatim.
    """

    #: Name used in BAM params files (the ``"actuator"`` key) and in configs.
    name: ClassVar[str] = ""

    #: Whether :meth:`control` carries state between calls. Such a motor must be
    #: stepped exactly once per timestep, in chronological order, and honours
    #: :meth:`reset` / :meth:`get_state` / :meth:`set_state`.
    stateful: ClassVar[bool] = False

    #: Physical unit of the control signal returned by :meth:`control`.
    control_unit: ClassVar[str] = "volts"

    #: Firmware constants, overridable per instance.
    firmware: ClassVar[dict[str, Any]] = {}

    #: Identified motor parameters with their seed values.
    parameters: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        for key, value in self.firmware.items():
            setattr(self, key, value)
        for key, value in self.parameters.items():
            setattr(self, key, value)
        self.set_params(params)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @classmethod
    def known_parameters(cls) -> dict[str, Any]:
        """Every settable name: the firmware constants plus the motor parameters."""
        return {**cls.firmware, **cls.parameters}

    def set_params(self, params: Mapping[str, Any]) -> list[str]:
        """Apply values for any known parameter present in ``params``.

        Unknown keys (friction parameters, ``"model"``, ...) are ignored, so a raw
        BAM params file can be handed over directly.

        :param params: Mapping of parameter name to value.
        :returns: The names that were actually applied.
        """
        applied = []
        for key in self.known_parameters():
            value = params.get(key)
            if value is not None:
                setattr(self, key, float(value))
                applied.append(key)
        return applied

    def get_params(self) -> dict[str, float]:
        """Current values of every known parameter."""
        return {key: float(getattr(self, key)) for key in self.known_parameters()}

    # ------------------------------------------------------------------
    # Motor torque (shared DC motor equation)
    # ------------------------------------------------------------------

    def torque(self, control: ArrayLike | None, torque_enable: bool, q: ArrayLike, dq: ArrayLike) -> ArrayLike:
        r"""Motor torque from the control signal, including back-EMF.

        .. math:: \tau = k_t V / R - k_t^2 \dot{q} / R

        Shared by every voltage-controlled servo, mirroring BAM's
        ``DCMotorActuator``. Current-controlled motors would override this.

        :param control: Control signal from :meth:`control` (volts).
        :param torque_enable: Whether the servo is powered; ``False`` gives zero torque.
        :param q: Joint position [rad] (unused here).
        :param dq: Joint velocity [rad/s].
        :returns: Motor torque [Nm].
        """
        volts = control
        tau = self.kt * volts / self.R - (self.kt**2) * dq / self.R
        return tau * torque_enable

    # ------------------------------------------------------------------
    # Control law
    # ------------------------------------------------------------------

    def control(self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float | None) -> ArrayLike:
        """Control signal (volts) for the current state.

        :param q_target: Target joint position [rad].
        :param q: Current joint position [rad].
        :param dq: Current joint velocity [rad/s].
        :param dt: Timestep [s]. Stateful motors that rate-limit their internal
            target need it; others may ignore it.
        :returns: Control signal, in :attr:`control_unit`.
        """
        raise NotImplementedError

    # -- helpers for building a control law --------------------------------

    def _duty_from_error(self, error: ArrayLike) -> ArrayLike:
        """Firmware P law: position error -> duty cycle.

        Includes ``error_gain_ratio`` when the motor declares it (it defaults to
        1.0, so motors without it are unaffected).
        """
        return error * self.kp * self.error_gain * getattr(self, "error_gain_ratio", 1.0)

    def _apply_current_limit(self, duty: ArrayLike, dq: ArrayLike) -> ArrayLike:
        """Bound the duty cycle so the motor current stays within ``max_current``.

        Mirrors BAM's firmware limiter: solving
        :math:`|(\\text{duty} \\cdot v_{in} - k_t \\dot{q}) / R| \\le I_{max}` gives
        the admissible duty window. This is only an *attempt* - the physical PWM
        clamp is applied afterwards, so at high back-EMF the limit is unreachable.
        """
        if self.max_current is None:
            return duty
        back_emf = self.kt * dq
        span = self.R * self.max_current / self.vin
        center = back_emf / self.vin
        return _clamp(duty, center - span, center + span)

    def _volts(self, duty: ArrayLike) -> ArrayLike:
        """Apply the physical PWM limit (battery voltage) and scale to volts."""
        return self.vin * _clamp(duty, -self.max_pwm, self.max_pwm)

    # ------------------------------------------------------------------
    # State (for stateful control laws)
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Reset the control law's internal state.

        When evaluated over a batch, ``env_ids`` selects which environments to
        reset. Because the internal state is expressed in terms of the joint
        state, a stateful motor may *defer* the reset to the next :meth:`control`
        call, where the joint position is available.
        """
        pass

    def get_state(self) -> Any:
        """Opaque snapshot of the internal state, for :meth:`set_state`."""
        return None

    def set_state(self, state: Any) -> None:
        """Restore a snapshot previously returned by :meth:`get_state`."""
        pass

    def __str__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} stateful={self.stateful}>"
