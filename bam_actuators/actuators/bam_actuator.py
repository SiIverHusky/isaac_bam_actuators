# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""An explicit Isaac Lab actuator driven by a per-motor control law.

Isaac Lab distinguishes *implicit* actuators (the PD law and the joint friction
are handed to the physics solver) from *explicit* actuators (the actuator model
computes the joint effort itself and writes it into the articulation action).
Because the friction model here depends on the instantaneous motor torque, it
cannot be expressed through the solver's constant joint-friction parameters, so
it has to be an **explicit** actuator.

This class is deliberately **agnostic to the motor**: it resolves a control law
from :mod:`bam_actuators.motors` (one module per servo family), feeds it the joint
state, then applies the friction budget. Per timestep and per
``(num_envs, num_joints)`` element::

    q             = joint_pos + q_offset                                         # rig offset
    control       = motor.control(q_target, q, dq, dt)                           # per-motor firmware
    motor_torque  = motor.torque(control, True, q, dq)                           # DC motor + back-EMF
    net_torque    = motor_torque + external_torque
    tau_stop      = net_torque + (inertia / dt) * dq                              # torque to stop in dt
    tau_friction  = -sign(tau_stop) * min(|tau_stop|, frictionloss + damping*|dq|)
    applied       = clip(net_torque + tau_friction, -effort_limit, +effort_limit)

``frictionloss`` and ``damping`` come from
:class:`bam_actuators.friction.BamFrictionModel` (BAM ``m1``-``m6``), and the
control law from a :class:`bam_actuators.motors.base.MotorBase` subclass. Both are
selected by the params file that is loaded.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import field

import torch

from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from ..friction import FLAG_NAMES, BamFrictionModel
from ..motors import get_motor
from ..params import resolve_params_file


@configclass
class BamFrictionCfg:
    """Configuration of the BAM friction budget (see ``bam_actuators.friction``).

    There are two ways to pick which maths is evaluated:

    * set :attr:`model` to ``"m1"``-``"m6"`` to select a variant wholesale, or
    * set the individual flags (left at ``None`` to mean "not specified").

    When a :attr:`BamActuatorCfg.params_file` is loaded, the ``"model"`` key
    inside the JSON wins, so pointing at ``m5.json`` gives you the ``m5`` maths
    with no further configuration. Flags set explicitly here are applied *on
    top* of the selected variant, which is useful for ablating a single term.
    """

    model: str | None = None
    """BAM model variant to evaluate: ``"m1"`` (Coulomb), ``"m2"`` (Stribeck),
    ``"m3"`` (load-dependent), ``"m4"`` (Stribeck + load-dependent), ``"m5"``
    (directional) or ``"m6"`` (quadratic). Overridden by the params file's
    ``"model"`` key. ``None`` leaves the flags below in charge."""

    load_dependent: bool | None = None
    """Enable load-dependent friction (BAM ``m3``+)."""

    directional: bool | None = None
    """Distinguish motor-side from external-side load friction (BAM ``m5``+)."""

    stribeck: bool | None = None
    """Enable the Stribeck effect: extra friction near zero velocity (BAM ``m2``+)."""

    quadratic: bool | None = None
    """Enable the quadratic load-friction coupling term (BAM ``m6``)."""

    friction_base: float = 0.05
    """Velocity-independent friction term [Nm]."""

    friction_stribeck: float = 0.05
    """Additional Stribeck friction at zero velocity [Nm]."""

    friction_viscous: float = 0.1
    """Viscous friction coefficient [Nm/(rad/s)]."""

    dtheta_stribeck: float = 0.2
    """Stribeck velocity scale [rad/s]."""

    alpha: float = 1.35
    """Stribeck curvature exponent."""

    load_friction_base: float = 0.05
    """Load-friction coupling coefficient (non-directional, BAM ``m3``/``m4``)."""

    load_friction_motor: float = 0.05
    """Motor-side load-friction coefficient (directional, BAM ``m5``/``m6``)."""

    load_friction_external: float = 0.05
    """External-side load-friction coefficient (directional, BAM ``m5``/``m6``)."""

    load_friction_stribeck: float = 0.05
    """Stribeck x load-friction coupling (non-directional)."""

    load_friction_motor_stribeck: float = 0.05
    """Stribeck x motor-side load-friction coupling (directional)."""

    load_friction_external_stribeck: float = 0.05
    """Stribeck x external-side load-friction coupling (directional)."""

    load_friction_motor_quad: float = 0.0
    """Quadratic motor-side coupling (BAM ``m6``)."""

    load_friction_external_quad: float = 0.0
    """Quadratic external-side coupling (BAM ``m6``)."""

    def explicit_flags(self) -> dict[str, bool]:
        """The flags the user set here explicitly, ignoring the unset (``None``) ones.

        :returns: Mapping of flag name to value for flags that are not ``None``.
        """
        return {name: getattr(self, name) for name in FLAG_NAMES if getattr(self, name) is not None}

    def build(self, data: dict | None = None) -> BamFrictionModel:
        """Build the friction model this config describes.

        :param data: Parsed params file, if one was configured. Its ``"model"``
            key takes precedence over :attr:`model`.
        :returns: A configured :class:`~bam_actuators.friction.BamFrictionModel`.
        """
        return BamFrictionModel.from_params(data or {}, model=self.model, **self.explicit_flags())


class BamActuator(ActuatorBase):
    r"""Explicit actuator that folds a per-motor control law and a friction budget into one effort.

    The actuator itself is motor-agnostic - it resolves a control law from
    :mod:`bam_actuators.motors` (selected by the params file's ``"actuator"`` key,
    or :attr:`BamActuatorCfg.motor`) and folds three things into a single joint
    effort:

    1. the per-motor **firmware control law** (:meth:`MotorBase.control`), which
       turns joint state into a control signal,
    2. the **motor torque**, including back-EMF (:meth:`MotorBase.torque`), and
    3. the **friction model**, applied as a stopping-torque clip.

    .. note::
        The physics solver's own joint friction (``friction``, ``dynamic_friction``
        and ``viscous_friction`` of :class:`~isaaclab.actuators.ActuatorBaseCfg`)
        should be left at ``0`` for these joints, otherwise friction is applied
        twice.

    .. note::
        Because load-dependent friction needs the external (gravity / contact)
        torque seen at the joint, and Isaac Lab does not pass it to the actuator,
        it defaults to zero. Call :meth:`set_external_torque` from your env (e.g.
        from the torque the articulation reports, or from a gravity model) to
        enable BAM ``m3``-``m6``.
    """

    cfg: BamActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: BamActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # A BAM params file is self-describing: it names the motor
        # (``"actuator": "sts3215"``), the friction model (``"model": "m5"``) and
        # the identified parameters of both, so read it once and let it drive.
        data = self._read_params_file(cfg.params_file) if cfg.params_file is not None else None

        # --- 1. pick the control law -----------------------------------
        self.motor_name = self._resolve_motor_name(data)
        self.motor = get_motor(self.motor_name)()

        # --- 2. hand it the identified + configured values -------------
        applied = self.motor.set_params(self._motor_overrides(data))

        # --- 3. friction model: ``"model"`` in the file selects m1..m6 --
        # NOTE: this must NOT be called ``self.friction``. ActuatorBase already
        # defines that as the solver's joint friction coefficient, and shadowing it
        # makes IsaacLab's ``write_joint_friction_coefficient_to_sim(actuator.friction,
        # ...)`` fail during Articulation init with
        # "can't assign a BamFrictionModel to a torch.cuda.FloatTensor".
        self.friction_model = cfg.friction_model.build(data)

        # --- 4. armature is a motor parameter --------------------------
        # It becomes the joint's armature unless the cfg pinned one explicitly.
        if cfg.armature is None and getattr(self.motor, "armature", 0.0):
            self.armature[:] = float(self.motor.armature)

        if data is not None:
            self._report_params_file(cfg.params_file, data, applied)

        # --- 5. guardrails ---------------------------------------------
        if self.motor.stateful and cfg.physics_dt is None:
            raise ValueError(
                f"The {self.motor_name!r} control law is stateful (it rate-limits its internal "
                f"target), so `physics_dt` must be set on BamActuatorCfg - it never receives "
                f"the timestep from Isaac Lab."
            )

        # --- per-(env, joint) state ------------------------------------
        # External torque used by the load-dependent friction terms. Isaac Lab
        # never hands it to us, so it stays zero unless the env sets it.
        self._external_torque = torch.zeros(self._num_envs, self.num_joints, device=self._device)

        # Whether the servo is powered, mirroring BAM's per-step ``torque_enable``.
        # An unpowered servo produces no motor torque but still back-drives, so
        # gravity and friction act unopposed (BAM's ``LiftAndDrop`` / ``Nothing``).
        # Note this must not be called ``torque_enable``: keep it private so it can
        # never collide with an :class:`ActuatorBase` attribute.
        self._torque_enable = torch.ones(
            self._num_envs, self.num_joints, dtype=torch.bool, device=self._device
        )

        # Reflected inertia used by the stopping-torque term.
        self._inertia: torch.Tensor | None = self.armature if cfg.use_joint_armature_as_inertia else None

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_params_file(params_file: str) -> dict:
        """Read a BAM params file (or a bundled ``<motor>/<model>`` reference)."""
        with open(resolve_params_file(params_file)) as f:
            return json.load(f)

    def _resolve_motor_name(self, data: dict | None) -> str:
        """Decide which control law to use.

        The params file wins (its ``"actuator"`` key names the servo the model was
        identified on), then the cfg, and ``"generic"`` if neither says.
        """
        if data and data.get("actuator"):
            return str(data["actuator"])
        if self.cfg.motor:
            return str(self.cfg.motor)
        return "generic"

    def _motor_overrides(self, data: dict | None) -> dict:
        """Merge the params file with the cfg into one set of motor values.

        The file carries the identified quantities (``kt``, ``R``, ``armature``,
        ``q_offset``, ``max_velocity``, ...). Firmware constants are properties of
        the motor module, so a non-``None`` cfg field overrides the file and the
        module default.
        """
        overrides: dict = dict(data or {})

        # Free-form overrides for identified values (kt, R, armature, q_offset,
        # max_velocity, error_gain_ratio, ...). Needed in particular for a motor
        # whose values are not identified yet - e.g. MD01 seeds kt at 0.0, so
        # without this there is no way to give it a torque constant.
        overrides.update({key: value for key, value in self.cfg.motor_params.items() if value is not None})

        for cfg_field, motor_name in (
            ("vin", "vin"),
            ("error_gain", "error_gain"),
            ("max_pwm", "max_pwm"),
            ("max_current", "max_current"),
            ("firmware_kp", "kp"),
        ):
            value = getattr(self.cfg, cfg_field)
            if value is not None:
                overrides[motor_name] = value

        # Legacy alias: `stiffness` used to *be* the firmware P gain.
        if self.cfg.stiffness is not None:
            overrides.setdefault("kp", self.cfg.stiffness)

        return overrides

    def _report_params_file(self, params_file: str, data: dict, applied: Sequence[str]) -> None:
        """Print what the params file was used for."""
        variant = self.friction_model.model_name or "custom"
        terms = ", ".join(self.friction_model.active_flags) or "none"
        print(
            f"[BamActuator] '{params_file}': motor={self.motor_name}, "
            f"model={data.get('model', 'n/a')} ({variant}), friction terms=[{terms}], "
            f"motor params={sorted(applied)}"
        )
        if getattr(self.motor, "command_delay", 0.0):
            print(
                f"[BamActuator]   note: command_delay={self.motor.command_delay:.4f}s is identified "
                "but not applied (no transport-delay buffer yet)."
            )

    def set_external_torque(self, external_torque: torch.Tensor, env_ids: Sequence[int] | slice | None = None) -> None:
        """Provide the external (gravity / load) torque seen at the joint [Nm].

        Only needed for load-dependent friction models (BAM ``m3``-``m6``). The
        tensor is expected to have shape ``(num_envs, num_joints)`` or to be
        broadcastable to it; ``env_ids`` selects which environments to update.

        :param external_torque: External torque seen at the joint [Nm].
        :param env_ids: Environments to update. Defaults to all of them.
        """
        if env_ids is None or env_ids == slice(None):
            self._external_torque[:] = external_torque
        else:
            self._external_torque[env_ids] = external_torque

    def set_torque_enable(
        self, enabled: bool | torch.Tensor, env_ids: Sequence[int] | slice | None = None
    ) -> None:
        """Power the servo on or off, per environment.

        Mirrors the ``torque_enable`` argument of BAM's ``Simulator.step``. While
        disabled the motor contributes no torque, but the friction budget and the
        external torque still apply - the joint back-drives and settles under
        gravity alone. That is what BAM's ``lift_and_drop`` and ``nothing``
        trajectories measure, so it is not the same as commanding zero torque.

        :param enabled: Whether the servo is powered. Scalars broadcast; a tensor
            must be broadcastable to ``(num_envs, num_joints)``.
        :param env_ids: Environments to update. Defaults to all of them.
        """
        if env_ids is None or env_ids == slice(None):
            self._torque_enable[:] = enabled
        else:
            self._torque_enable[env_ids] = enabled

    # ------------------------------------------------------------------
    # ActuatorBase interface
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        """Reset the actuator internals for the given environments."""
        # An unpowered state is a per-run experiment, never a resting state, so
        # every env comes back powered.
        if env_ids is None or env_ids == slice(None):
            self._external_torque.zero_()
            self._torque_enable.fill_(True)
        else:
            self._external_torque[env_ids] = 0.0
            self._torque_enable[env_ids] = True

        # The motor uses BAM's `...` sentinel for "all environments".
        self.motor.reset(... if env_ids is None else env_ids)

    def get_state(self):
        """Snapshot of the control law's state, for :meth:`set_state`."""
        return self.motor.get_state()

    def set_state(self, state) -> None:
        """Restore a snapshot from :meth:`get_state`."""
        self.motor.set_state(state)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Compute the joint effort from the position command.

        :param control_action: Desired joint positions (and optionally velocities
            and feed-forward efforts).
        :param joint_pos: Current joint positions. Shape is ``(num_envs, num_joints)``.
        :param joint_vel: Current joint velocities. Shape is ``(num_envs, num_joints)``.
        :returns: The control action with ``joint_efforts`` set to the applied torque.
        """
        cfg = self.cfg

        # q_offset is a rig-level identified parameter, and BAM resolves it
        # asymmetrically: the firmware acts on the *raw* joint angle while the
        # physics sees the offset one (``Simulator.rollout_log`` passes ``self.q``
        # to ``compute_control``, ``Simulator.step`` uses ``self.q + q_offset`` for
        # the bias and motor torque). The identified value is only meaningful under
        # that convention, so replicate it exactly - see
        # tests/test_offline_rollout.py::test_q_offset_convention_is_mirrored.
        q_physics = joint_pos + getattr(self.motor, "q_offset", 0.0)

        q_target = control_action.joint_positions
        if q_target is None:
            # Effort/velocity command: no position error to drive the P law.
            q_target = joint_pos

        # --- 1. firmware control law (per-motor) ------------------------
        control = self.motor.control(q_target, joint_pos, joint_vel, cfg.physics_dt)

        # --- 2. motor torque (shared DC motor equation) -----------------
        # An unpowered servo produces no torque but still back-drives.
        motor_torque = self.motor.torque(control, self._torque_enable, q_physics, joint_vel)

        # Stored for logging / reward shaping (torque the motor would produce).
        self.computed_effort = motor_torque + self._external_torque

        # --- 3. friction budget + stopping-torque clip ------------------
        net_torque = self.friction_model.apply(
            motor_torque,
            self._external_torque,
            joint_vel,
            inertia=self._inertia,
            dt=cfg.physics_dt,
        )
        self.applied_effort = self._clip_effort(net_torque)

        # --- 4. hand the effort to the articulation ---------------------
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class BamActuatorCfg(ActuatorBaseCfg):
    """Configuration for :class:`BamActuator`.

    The control law is chosen by :attr:`motor`, or automatically from the params
    file's ``"actuator"`` key. Firmware constants (``vin``, ``firmware_kp``,
    ``error_gain``, ``max_pwm``, ``max_current``) live on the motor module and only
    need to be set here to *override* them - leave them ``None`` to take the
    motor's own values.

    The inherited ``stiffness`` / ``damping`` PD gains are unused: the firmware
    control law has its own gains, so they are relaxed to ``None`` to avoid
    having to pass meaningless values. ``stiffness`` is still accepted as a
    legacy alias for :attr:`firmware_kp`.
    """

    class_type: type = BamActuator

    # --- which control law ---
    motor: str | None = None
    """Name of the motor module to use, e.g. ``"sts3215"`` or ``"md01"``. Taken
    from the params file's ``"actuator"`` key when the file provides one, then
    falling back to ``"generic"``. See
    :func:`bam_actuators.motors.available_motors`."""

    # --- identified model ---
    params_file: str | None = None
    """BAM params file to load, used as-is.

    Either a path, or a ``"<motor>/<model>"`` shorthand for a bundled model (e.g.
    ``"sts3215/m5"``). The file decides the motor (``"actuator"``), the friction
    maths (``"model"``) and the identified parameters of both."""

    friction_model: BamFrictionCfg = field(default_factory=BamFrictionCfg)
    """Friction budget configuration. The model variant is taken from
    :attr:`params_file` when the file carries a ``"model"`` key."""

    motor_params: dict = field(default_factory=dict)
    """Overrides for identified motor values, applied on top of :attr:`params_file`.

    Anything the motor module declares: ``kt``, ``R``, ``armature``, ``q_offset``,
    ``max_velocity``, ``error_gain_ratio``, ... Useful to seed a value that is not
    identified yet - MD01's ``kt`` defaults to ``0.0``, so it needs one to produce
    any torque. The named firmware fields below take precedence over this.
    """

    # --- firmware overrides (None -> the motor module's default) ---
    vin: float | None = None
    """Supply (battery) voltage [V]."""

    firmware_kp: float | None = None
    """Firmware P gain (duty cycle per radian of position error)."""

    error_gain: float | None = None
    """Converts the firmware's ``kp * error`` into a duty cycle."""

    max_pwm: float | None = None
    """Maximum duty-cycle magnitude."""

    max_current: float | None = None
    """Firmware current limit [A]. ``None`` means "use the motor's default";
    pass ``float("inf")`` to disable the limiter."""

    # --- PD gains, relaxed: the firmware law owns its own gains ---
    stiffness: float | None = None
    """Unused. Accepted as a legacy alias for :attr:`firmware_kp`."""

    damping: float | None = None
    """Unused."""

    # --- stopping-torque term ---
    use_joint_armature_as_inertia: bool = False
    """Use the joint armature as the reflected inertia in the stopping-torque term.

    Note this is only the *rotor* inertia, whereas the stopping torque is defined
    with the total joint inertia, so the term is a lower bound while this is on."""

    physics_dt: float | None = None
    """Physics timestep [s]. Required by stateful control laws (e.g. ``sts3215``,
    which rate-limits its internal target) and for the stopping-torque term."""
