# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Vectorized (torch) implementation of the BAM extended friction budget.

This mirrors :meth:`bam.model.Model.compute_frictions` from the BAM project
(https://github.com/Rhoban/bam) but is written against ``torch`` so the whole
friction budget can be evaluated over ``(num_envs, num_joints)`` tensors on the
GPU, which is the shape Isaac Lab actuators operate on.

The friction is expressed as a *budget*: a maximum resistive torque split into

* ``frictionloss`` -- the velocity-independent (Coulomb / Stribeck / load
  dependent) part [Nm],
* ``damping`` -- the viscous coefficient times velocity [Nm/(rad/s)].

The torque actually applied is obtained by clipping the *stopping torque* (the
torque needed to bring the joint to rest within one timestep) into
``±(frictionloss + damping * |dq|)``. See :meth:`BamFrictionModel.apply`.

Supported model variants, matching BAM's ``m1``-``m6``:

===========  ==========================================================
``m1``       Coulomb (constant ``friction_base`` + viscous term)
``m2``       + Stribeck (extra friction near zero velocity)
``m3``       + load-dependent friction
``m4``       + Stribeck x load-dependent
``m5``       + directional (motor-side vs external-side) load friction
``m6``       + quadratic load-friction coupling
===========  ==========================================================
"""

from __future__ import annotations

import json
from typing import Any

import torch

#: Names of every parameter the model can expose / read back from a BAM JSON.
PARAMETER_NAMES = (
    "friction_base",
    "friction_stribeck",
    "friction_viscous",
    "dtheta_stribeck",
    "alpha",
    "load_friction_base",
    "load_friction_motor",
    "load_friction_external",
    "load_friction_stribeck",
    "load_friction_motor_stribeck",
    "load_friction_external_stribeck",
    "load_friction_motor_quad",
    "load_friction_external_quad",
)

#: The four boolean terms that select which model maths is evaluated.
FLAG_NAMES = ("load_dependent", "directional", "stribeck", "quadratic")

#: Which flags each BAM model variant turns on. Mirrors ``bam.model.models``.
MODEL_VARIANTS: dict[str, dict[str, bool]] = {
    "m1": {},
    "m2": {"stribeck": True},
    "m3": {"load_dependent": True},
    "m4": {"load_dependent": True, "stribeck": True},
    "m5": {"load_dependent": True, "stribeck": True, "directional": True},
    "m6": {"load_dependent": True, "stribeck": True, "directional": True, "quadratic": True},
}


def variant_flags(model: str) -> dict[str, bool]:
    """Return the flags enabled by a BAM model variant (``"m1"``-``"m6"``).

    :param model: Variant name, e.g. ``"m5"``.
    :returns: Mapping of flag name to ``True`` for the terms that variant uses.
    :raises ValueError: If the variant name is not known.
    """
    try:
        return dict(MODEL_VARIANTS[model.lower()])
    except (AttributeError, KeyError):
        raise ValueError(f"Unknown BAM model variant {model!r}. Expected one of {sorted(MODEL_VARIANTS)}.") from None


def _zeros_like(ref: Any) -> Any:
    """Return a zero scalar/tensor broadcastable with ``ref``.

    Keeps :meth:`BamFrictionModel.compute` shape-agnostic: the same code path
    works for Python floats (scalar identification runs) and for
    ``(num_envs, num_joints)`` tensors (RL rollouts).
    """
    if isinstance(ref, torch.Tensor):
        return torch.zeros_like(ref)
    return 0.0


def _abs(value: Any) -> Any:
    """Absolute value that works for both torch tensors and Python floats."""
    if isinstance(value, torch.Tensor):
        return value.abs()
    return abs(value)


class BamFrictionModel:
    r"""Torch implementation of the BAM friction budget.

    All parameters are plain Python floats by default so a single
    :class:`BamFrictionModel` can be evaluated over any batch shape. Per-joint or
    per-environment parameters can be injected afterwards by assigning tensors
    to the corresponding attributes (broadcasting follows standard torch rules).

    :param load_dependent: Enable load-dependent friction (BAM ``m3``+).
    :param directional: Distinguish motor-side from external-side load friction
        (BAM ``m5``+). Requires ``load_dependent=True``.
    :param stribeck: Enable the Stribeck effect (BAM ``m2``+).
    :param quadratic: Enable the quadratic load-friction coupling term
        (BAM ``m6``). Requires ``directional=True`` and ``stribeck=True``.
    """

    def __init__(
        self,
        *,
        load_dependent: bool = False,
        directional: bool = False,
        stribeck: bool = False,
        quadratic: bool = False,
        friction_base: float = 0.05,
        friction_stribeck: float = 0.05,
        friction_viscous: float = 0.1,
        dtheta_stribeck: float = 0.2,
        alpha: float = 1.35,
        load_friction_base: float = 0.05,
        load_friction_motor: float = 0.05,
        load_friction_external: float = 0.05,
        load_friction_stribeck: float = 0.05,
        load_friction_motor_stribeck: float = 0.05,
        load_friction_external_stribeck: float = 0.05,
        load_friction_motor_quad: float = 0.0,
        load_friction_external_quad: float = 0.0,
    ) -> None:
        if quadratic and not (directional and stribeck):
            raise ValueError("`quadratic=True` requires `directional=True` and `stribeck=True` (BAM m6).")
        if directional and not load_dependent:
            raise ValueError("`directional=True` requires `load_dependent=True` (BAM m5/m6).")

        self.load_dependent = bool(load_dependent)
        self.directional = bool(directional)
        self.stribeck = bool(stribeck)
        self.quadratic = bool(quadratic)

        self.friction_base = float(friction_base)
        self.friction_stribeck = float(friction_stribeck)
        self.friction_viscous = float(friction_viscous)
        self.dtheta_stribeck = float(dtheta_stribeck)
        self.alpha = float(alpha)
        self.load_friction_base = float(load_friction_base)
        self.load_friction_motor = float(load_friction_motor)
        self.load_friction_external = float(load_friction_external)
        self.load_friction_stribeck = float(load_friction_stribeck)
        self.load_friction_motor_stribeck = float(load_friction_motor_stribeck)
        self.load_friction_external_stribeck = float(load_friction_external_stribeck)
        self.load_friction_motor_quad = float(load_friction_motor_quad)
        self.load_friction_external_quad = float(load_friction_external_quad)

        #: Name of the BAM variant this model was built from, if any (``"m5"``, ...).
        self.model_name: str | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_params(
        cls, data: dict[str, Any], *, model: str | None = None, **flags: bool
    ) -> "BamFrictionModel":
        """Build a model from a BAM parameter dict, honouring its ``"model"`` key.

        This is what makes a params file self-describing: pointing at ``m5.json``
        selects the ``m5`` maths without having to set any flag by hand. An
        explicit ``model`` argument is ignored when the data carries its own
        ``"model"`` key, so the file always describes itself.

        Precedence, lowest to highest:

        1. the defaults (all flags off, i.e. ``m1``),
        2. the variant named by ``data["model"]`` (or the ``model`` argument),
        3. any flag passed explicitly as a keyword argument.

        :param data: Parsed BAM params file, e.g. ``json.load(open("m5.json"))``.
            Unknown keys (``kt``, ``R``, ``actuator``, ...) are ignored.
        :param model: Variant name to use when ``data`` has no ``"model"`` key.
        :param flags: Individual flags that override the selected variant.
        :returns: A configured :class:`BamFrictionModel`.
        """
        variant = data.get("model", model)
        resolved: dict[str, bool] = dict.fromkeys(FLAG_NAMES, False)
        if variant is not None:
            resolved.update(variant_flags(variant))
        resolved.update(flags)

        params = {name: float(data[name]) for name in PARAMETER_NAMES if name in data}
        instance = cls(**resolved, **params)
        instance.model_name = None if variant is None else str(variant).lower()
        return instance

    @classmethod
    def from_json(cls, path: str, *, model: str | None = None, **flags: bool) -> "BamFrictionModel":
        """Load a BAM params JSON file and build a model from it.

        :param path: Path to a BAM params file (``bam/params/<motor>/<model>.json``).
        :param model: Variant name to use when the file has no ``"model"`` key.
        :param flags: Individual flags that override the selected variant.
        :returns: A configured :class:`BamFrictionModel`.
        """
        with open(path) as f:
            return cls.from_params(json.load(f), model=model, **flags)

    @property
    def active_flags(self) -> list[str]:
        """The names of the friction terms currently enabled, in a stable order."""
        return [name for name in FLAG_NAMES if getattr(self, name)]

    @classmethod
    def from_bam_model(cls, bam_model: Any) -> "BamFrictionModel":
        """Build from a ``bam.model.Model`` instance (duck-typed).

        Reading the flags and ``Parameter.value`` off the BAM object keeps this
        extension free of a hard dependency on the ``bam`` package while still
        letting identified models be reused verbatim.

        :param bam_model: An object exposing the BAM model attributes
            (``load_dependent``, ``stribeck``, ... and ``Parameter`` objects with
            a ``.value`` attribute).
        :returns: A new :class:`BamFrictionModel`.
        """
        kwargs: dict[str, Any] = {
            flag: bool(getattr(bam_model, flag, False))
            for flag in ("load_dependent", "directional", "stribeck", "quadratic")
        }
        for name in PARAMETER_NAMES:
            param = getattr(bam_model, name, None)
            if param is not None:
                kwargs[name] = float(getattr(param, "value", param))
        instance = cls(**kwargs)
        name = getattr(bam_model, "name", None)
        instance.model_name = None if name is None else str(name).lower()
        return instance

    def load_params(self, data: dict[str, Any]) -> list[str]:
        """Apply identified parameter values from a flat ``{name: value}`` dict.

        Unknown keys are ignored, which makes it safe to feed the JSON files
        produced by BAM (they also contain actuator-level values such as ``kt``,
        ``R`` and ``kp``).

        :param data: Mapping of parameter name to value.
        :returns: The list of parameter names that were actually applied.
        """
        applied = []
        for name in PARAMETER_NAMES:
            if name in data:
                setattr(self, name, float(data[name]))
                applied.append(name)
        return applied

    def get_parameter_values(self) -> dict[str, float]:
        """Return the current scalar parameter values as a ``{name: value}`` dict."""
        return {name: getattr(self, name) for name in PARAMETER_NAMES}

    # ------------------------------------------------------------------
    # Friction budget
    # ------------------------------------------------------------------

    def compute(
        self, motor_torque: Any, external_torque: Any, dq: Any
    ) -> tuple[Any, Any]:
        """Compute the friction budget for the current state.

        :param motor_torque: Torque produced by the motor [Nm].
        :param external_torque: External (gravity / load) torque seen at the
            joint [Nm]. This is what makes load-dependent friction possible; see
            :meth:`BamActuator.set_external_torque`.
        :param dq: Joint velocity [rad/s].
        :returns: Tuple ``(frictionloss, damping)`` -- the constant part [Nm] and
            the viscous coefficient [Nm/(rad/s)].
        """
        # Torque applied to the gearbox.
        if self.directional:
            gearbox_torque = (
                external_torque * self.load_friction_external - motor_torque * self.load_friction_motor
            ).abs()
        else:
            gearbox_torque = (external_torque - motor_torque).abs()

        if self.stribeck:
            # Stribeck coefficient: 1 when stopped, 0 when moving fast.
            stribeck_coeff = torch.exp(-((_abs(dq) / self.dtheta_stribeck) ** self.alpha))
            if self.directional:
                gearbox_torque_stribeck = (
                    external_torque * self.load_friction_external_stribeck
                    - motor_torque * self.load_friction_motor_stribeck
                ).abs()

        # Static friction.
        frictionloss = self.friction_base + _zeros_like(motor_torque)
        if self.load_dependent:
            if self.directional:
                frictionloss = frictionloss + gearbox_torque
            else:
                frictionloss = frictionloss + self.load_friction_base * gearbox_torque

        if self.stribeck:
            frictionloss = frictionloss + stribeck_coeff * self.friction_stribeck

            if self.load_dependent:
                if self.directional:
                    frictionloss = frictionloss + gearbox_torque_stribeck * stribeck_coeff
                else:
                    frictionloss = frictionloss + (
                        self.load_friction_stribeck * gearbox_torque * stribeck_coeff
                    )

                if self.quadratic:
                    enable_quadratic = torch.sign(external_torque) != torch.sign(motor_torque)
                    direction_motor = external_torque.abs() < motor_torque.abs()
                    direction_external = external_torque.abs() > motor_torque.abs()

                    gearbox_torque2_motor = self.load_friction_external_quad * external_torque.abs() ** 2
                    gearbox_torque2_external = self.load_friction_motor_quad * motor_torque.abs() ** 2

                    frictionloss = frictionloss + (
                        stribeck_coeff
                        * (direction_motor * gearbox_torque2_motor + direction_external * gearbox_torque2_external)
                        * enable_quadratic
                    )

        # Viscous friction.
        damping = self.friction_viscous + _zeros_like(motor_torque)

        return frictionloss, damping

    def apply(
        self,
        motor_torque: Any,
        external_torque: Any,
        dq: Any,
        inertia: Any | None = None,
        dt: float | None = None,
    ) -> Any:
        """Return the net joint torque after applying the friction budget.

        This is BAM's Algorithm 1: the *stopping torque* is the torque needed to
        reach zero velocity within ``dt``; friction can cancel at most
        ``frictionloss + damping * |dq|`` of it, which is what produces proper
        stiction (a joint at rest does not drift) instead of a plain
        ``-sign(dq)`` Coulomb term.

        :param motor_torque: Torque produced by the motor [Nm].
        :param external_torque: External (gravity / load) torque [Nm].
        :param dq: Joint velocity [rad/s].
        :param inertia: Reflected inertia [kg·m²] used for the stopping-torque
            term. When ``None``, the term is dropped and the friction degenerates
            to a static/Coulomb clip on the net torque (still stable, but without
            the inertial "hold" behaviour).
        :param dt: Physics timestep [s], required together with ``inertia``.
        :returns: Net joint torque [Nm] including friction.
        """
        frictionloss, damping = self.compute(motor_torque, external_torque, dq)

        net_torque = motor_torque + external_torque
        if inertia is not None and dt is not None:
            tau_stop = net_torque + (inertia / dt) * dq
        else:
            tau_stop = net_torque

        budget = frictionloss + damping * _abs(dq)
        friction = -torch.sign(tau_stop) * torch.minimum(torch.abs(tau_stop), budget)
        return net_torque + friction
