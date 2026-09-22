# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Single-joint pendulum scene for exercising BamActuator against a real recording.

BAM's ``testbench.Pendulum`` is a point mass at the tip of a uniform rod::

    I_pivot = mass * L^2 + (arm_mass / 3) * L^2
    tau_gravity(q) = (mass + arm_mass / 2) * g * L * sin(q)      [g = -9.80665]

Rather than tuning link geometry until Isaac happens to agree, this script writes
the URDF's ``<inertial>`` block so the articulation is *analytically* the same
body: one link of mass ``mass + arm_mass`` with its COM at the distance that
reproduces ``tau_gravity`` exactly, and a diagonal inertia chosen so the swing
inertia about the pivot equals ``I_pivot`` exactly. The visual geometry is
cosmetic and does not affect the dynamics.

It then drives the actuator with a command sequence and reports two comparisons:

* **vs the offline harness** - our model simulated twice, in Isaac (PhysX) and in
  BAM's own loop. Differences here are integrator/solver, not model.
* **vs the recording** - the model against the physical servo. This is the one that
  answers "does it react like it's supposed to".

With a recording (``--log``) the command sequence is the log's own
``goal_position``. Without one the pendulum is driven by one of **BAM's own
identification trajectories** (``--command``), the same motions the params file was
fitted from - including their ``torque_enable`` flag, so ``nothing`` and
``lift_and_drop`` let the arm fall unpowered under gravity alone.

**Comparing models.** ``--model1``/``--model2`` (or ``--models`` for any number) put
several models in the scene at once, each as its own pendulum driven by the *same*
command, so the only difference between them is the friction model. The arms are
colour-coded by variant (m1..m6), one joint each, and each gets its own actuator group
and params file - see :mod:`bam_actuators.testbench`.

**Watching it.** A 6 s trajectory simulates in a fraction of a second, so without
pacing the whole run is over before the viewport draws a first frame. ``--speed``
plays it back in real time (or slower), and the camera is framed on the row of arms,
which are only 15 cm long.

Usage::

    python scripts/pendulum_scene.py                              # headless, real-time paced
    python scripts/pendulum_scene.py --visual --loop              # watch it, over and over
    python scripts/pendulum_scene.py --visual --model1 m1 --model2 m6
    python scripts/pendulum_scene.py --visual --models m1 m3 m6   # any number
    python scripts/pendulum_scene.py --visual --command lift_and_drop --speed 0.25
    python scripts/pendulum_scene.py --command nothing --params none
    python scripts/pendulum_scene.py --command nothing --initial-angle 1.2 --visual
    python scripts/pendulum_scene.py --csv /tmp/traj.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

# BAM's identification trajectories drive the synthetic runs and provide the
# --command choices. Pure numpy, so this is safe before Isaac Sim is up - which
# matters because argparse has to see the choices first.
from bam_actuators.trajectory import TRAJECTORIES, get_trajectory, sample

# Answer --list-motors before anything Isaac-specific is imported: it is most
# useful exactly when the environment is misconfigured, so it must not need a
# working isaaclab to tell you the valid motor names.
if "--list-motors" in sys.argv:
    from bam_actuators.motors import available_motors

    print("available motors:", ", ".join(available_motors()))
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

# The offline harness lives with the tests; import it rather than duplicate it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from bam_paths import bam_root  # noqa: E402

#: BAM's source checkout. Only the *recordings* and the firmware-mirroring need it;
#: the scene, the bundled models and the comparison all work without it.
BAM_ROOT = bam_root()

parser = argparse.ArgumentParser(description="Pendulum test scene for BamActuator.")
parser.add_argument(
    "--log",
    type=str,
    default=None,
    help="Recorded log to replay. Default: $BAM_ROOT/data_raw/<default>.json. "
    "Pass 'none' to drive a synthetic command sequence instead.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Number of steps to simulate. Default: the whole trajectory, or the whole recording.",
)
parser.add_argument("--motor", type=str, default="sts3215", help="Motor name as BAM's params files write it, e.g. 'sts3215', 'md01' (voltage law), "
    "'md01i' (current law), 'md01c' (measured AT32 loops) - not the module filename. "
    "The params file's 'actuator' key overrides this. See --list-motors.")
parser.add_argument(
    "--params",
    type=str,
    default=None,
    help="Params file for the single-pendulum case, or 'none'. Default: sts3215/m5; try "
    "md01i/m3 for the new MD01 campaign. Ignored when --model1/--model2/--models are given.",
)
parser.add_argument("--list-motors", action="store_true", help="List valid --motor values and exit.")
# Which models to show. One pendulum each, so they can be compared side by side.
parser.add_argument(
    "--model1",
    type=str,
    default=None,
    help="Show this model as the first pendulum, e.g. 'm1'. A bare variant name is "
    "resolved against --motor, so 'm1' means '<motor>/m1.json'.",
)
parser.add_argument(
    "--model2",
    type=str,
    default=None,
    help="Show this model as the second pendulum, e.g. 'm5'.",
)
parser.add_argument(
    "--models",
    type=str,
    nargs="+",
    default=None,
    help="General form of --model1/--model2: any number of models to show side by "
    "side, e.g. --models m1 m3 m5.",
)
# Rig and drive, used when there is no recording.
parser.add_argument("--mass", type=float, default=0.5, help="Tip mass [kg].")
parser.add_argument("--arm-mass", type=float, default=0.02, help="Arm mass [kg].")
parser.add_argument("--length", type=float, default=0.15, help="Arm length [m].")
parser.add_argument("--dt", type=float, default=0.005, help="Timestep [s] for synthetic runs.")
parser.add_argument(
    "--initial-angle",
    type=float,
    default=0.0,
    help="Starting angle [rad] for synthetic runs; 0 is the arm hanging down, which is "
    "also the gravity equilibrium. Only recordings override it (they carry their own).",
)
parser.add_argument(
    "--command",
    type=str,
    default="steps",
    choices=sorted(TRAJECTORIES),
    help="BAM identification trajectory to drive the pendulum with when there is no "
    "recording. All of them run for 6 s; see bam_actuators.trajectory.",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Playback speed: 1.0 is real time, 0.25 is quarter speed, 0 runs as fast as "
    "possible. Only affects --visual.",
)
parser.add_argument(
    "--loop",
    action="store_true",
    help="After reporting, keep replaying the sequence so the motion stays on screen.",
)
parser.add_argument(
    "--overlay",
    action="store_true",
    help="Put every pendulum on the same pivot instead of spreading them out, so their "
    "arcs coincide and only the model can pull them apart. The arms are separated "
    "along their own rotation axis, which does not change the dynamics, and the rig "
    "carries no collision geometry, so overlapping the arcs is safe.",
)
parser.add_argument("--csv", type=str, default=None, help="Write the trajectories to this CSV.")
parser.add_argument("--visual", action="store_true", help="Show the viewport (default: headless).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.visual

# (--list-motors was already handled above, before the isaaclab import.)

# Let "import bam" work from the source checkout, the way the tests do.
if BAM_ROOT.is_dir():
    sys.path.insert(0, str(BAM_ROOT))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Kit must be up before these are imported -------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from bam_actuators.actuators import BamActuatorCfg  # noqa: E402
from bam_actuators.params import resolve_params_file  # noqa: E402
from bam_actuators.testbench import (  # noqa: E402
    GRAVITY,
    Arm,
    arm_for,
    build_urdf,
    pendulum_spacing,
)

from offline_rollout import OfflineSimulator, drive, load_raw_log  # noqa: E402

# The rig geometry, the model colour palette and BAM's gravity live in the package so
# they can be checked without Isaac Sim - see bam_actuators/testbench.py.


def resolve_model_arg(value: str, motor: str) -> str:
    """Resolve a ``--model``-style reference to a params file.

    A bundled variant may be given as a bare variant name (``m5``), which is
    expanded against ``--motor``, or in any form :func:`resolve_params_file`
    already accepts.

    :param value: e.g. ``"m5"``, ``"sts3215/m5"``, or a path to a ``.json``.
    :param motor: Motor name used to expand a bare variant name.
    """
    if "/" in value or value.endswith(".json"):
        return resolve_params_file(value)
    return resolve_params_file(f"{motor}/{value}")


def default_log() -> Path:
    """The recording the parity tests use, if BAM's checkout is where we expect."""
    return BAM_ROOT / "data_raw" / "2026-09-10_15h22m23.json"


#: Firmware constants a motor module declares that ``BamActuatorCfg`` has no field
#: for, so they have to travel through ``motor_params``. ``md01c`` is the only one
#: today: its loop gains come from the AT32 flash, and an MD01 recording writes them
#: into the log, so they have to reach the actuator somehow.
MOTOR_PARAM_FIRMWARE = ("kp_current", "kff_current", "cap_ma")


def firmware_overrides(log: dict | None, motor: str) -> tuple[dict, dict]:
    """Firmware constants to configure the actuator with.

    Firmware is **not** in params files, so it comes from the recording. Which
    fields a recording supplies is motor-specific - ``DCMotorActuator.load_log``
    sets ``kp`` and ``vin``, ``MD01Actuator`` also takes ``error_gain`` and
    ``max_pwm``, ``MD01LoopActuator`` also takes the AT32 loop gains,
    ``STS3215Actuator`` overrides nothing - so ask BAM rather than hard-coding the
    rule.

    With no recording there is nothing to mirror, so return nothing and let the
    motor module's own defaults stand. Both sides of the comparison then use those
    same defaults, so the comparison is still fair.

    :param log: The recorded log, or ``None`` for a synthetic run.
    :param motor: Motor name, as BAM's ``actuators`` registry writes it.
    :returns: ``(firmware_fields, motor_params)`` - the first for the named cfg
        fields, the second for the constants that have no field of their own (and
        so are empty for every motor but ``md01c``).
    """
    if log is None:
        return {}, {}

    try:
        from bam.actuators import actuators

        reference = actuators[motor]()
        reference.load_log(log)
        firmware = {
            "vin": reference.vin,
            "firmware_kp": reference.kp,
            "error_gain": reference.error_gain,
            "max_pwm": reference.max_pwm,
        }
        motor_params = {
            name: getattr(reference, name) for name in MOTOR_PARAM_FIRMWARE if hasattr(reference, name)
        }
        return firmware, motor_params
    except Exception as exc:  # noqa: BLE001 - fall back, but say so
        print(f"[scene] BAM not usable for firmware ({exc}); using kp/vin from the log.")
        print("[scene] WARNING: a motor that also takes error_gain/max_pwm from the log will differ.")
        return {"vin": log["vin"], "firmware_kp": log["kp"]}, {}


def make_scene_cfg(urdf_path: str, actuators: dict[str, BamActuatorCfg], num_envs: int = 1):
    """Scene with the articulated pendulum rig and a light.

    :param actuators: One actuator group per pendulum, keyed by the name `main` will
        look the actuator up by. Each group has its own ``joint_names_expr``, so each
        one reads its own params file.
    """

    @configclass
    class _PendulumSceneCfg(InteractiveSceneCfg):
        """Every pendulum, as the joints of one articulation anchored to the world.

        Deliberately **no ground plane**: the pivots sit at the env origin and the arms
        hang to ``z = -length``, so a floor at ``z = 0`` would intersect them and
        introduce contacts that BAM's testbench does not have.

        One articulation rather than one asset per model, because that is what makes
        per-model parameters possible: an articulation may carry several actuator
        groups, each covering a different subset of joints, and each group is its own
        :class:`BamActuator` reading its own params file. The arms are also then
        guaranteed to share a physics step, which is what makes the comparison fair.
        """

        light = AssetBaseCfg(
            prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
        )
        pendulum = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/pendulum",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=urdf_path,
                fix_base=True,
                make_instanceable=False,
                # The rig emits no collision geometry at all (see
                # bam_actuators.testbench). Both of these are already the defaults, but
                # state them: the arms must never collide with each other, or an
                # overlaid run would drive the solver into a deep interpenetration.
                self_collision=False,
                collision_from_visuals=False,
                # The importer keeps the URDF's `<inertial>` values - there is no
                # recompute-from-geometry option - which is essential here, since they
                # were derived to match BAM's testbench analytically. `link_density`
                # only applies to links with no inertial block, and ours all have one.
                #
                # The importer requires explicit drive gains (`PDGainsCfg.stiffness`
                # has no default), so zero them: BamActuator is the only source of
                # joint torque, and `target_type="none"` zeroes them regardless.
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    target_type="none",
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0.0, damping=0.0
                    ),
                ),
            ),
            # Every joint starts hanging straight down. The arms are placed by the
            # URDF's own `<origin>` offsets, so no joint names are needed here.
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
            ),
            actuators=actuators,
        )

    return _PendulumSceneCfg(num_envs=num_envs, env_spacing=2.0)


def colour_links(root_prim: str, pendulums: list[Arm]) -> None:
    """Reinforce the per-arm colours with an explicit material binding.

    The URDF already carries a ``<material>`` per arm, so the colour travels with the
    asset. Not every importer is guaranteed to honour that, and ``UrdfConverterCfg``
    has no material option of its own, so bind a spawned material on top as well.

    Cosmetic and best-effort: if this fails the arms keep whatever the asset gave them,
    which is worth a warning but never worth failing the run.

    :param root_prim: The articulation's prim path, e.g. ``/World/envs/env_0/pendulum``.
    :param pendulums: The arms whose link names and colours to use.
    """
    for pendulum in pendulums:
        material_path = f"/World/Looks/{pendulum.slug}"
        try:
            material = sim_utils.PreviewSurfaceCfg(diffuse_color=pendulum.colour)
            material.func(material_path, material)
            sim_utils.bind_visual_material(
                f"{root_prim}/{pendulum.link_name}", material_path
            )
        except Exception as exc:  # noqa: BLE001 - cosmetic, never fatal
            print(f"[scene] could not bind a colour to {pendulum.link_name}: {exc}")
            print("[scene]   the arms keep the <material> from the URDF instead.")
            return


def main() -> int:
    # --- decide what to replay -------------------------------------
    log_path: Path | None = None
    if args.log and args.log.lower() != "none":
        log_path = Path(args.log)
        if not log_path.is_file():
            print(f"[scene] log not found: {log_path}", file=sys.stderr)
            return 1
    elif args.log is None:
        candidate = default_log()
        if candidate.is_file():
            log_path = candidate
        else:
            print(f"[scene] no recording at {candidate}")
            print("[scene]   set BAM_ROOT to your BAM checkout, pass --log <file>, or --log none")
            print("[scene]   falling back to a synthetic run.\n")

    if log_path is not None:
        log = load_raw_log(str(log_path))
        entries = log["entries"]
        if args.steps is not None:
            entries = entries[: args.steps]
        goals = [entry["goal_position"] for entry in entries]
        enabled = np.array([bool(entry.get("torque_enable", True)) for entry in entries])
        recorded = np.array([entry["position"] for entry in entries])
        dt = log["dt"]
        mass, arm_mass, length = log["mass"], log["arm_mass"], log["length"]
        q0, dq0 = float(entries[0]["position"]), float(entries[0].get("speed", 0.0))
        provenance = log_path.name
    else:
        # Nothing recorded: build the rig from the CLI and drive it with one of BAM's
        # own identification trajectories, so the arm performs the very motion the
        # params file was fitted from - torque-enable flag and all.
        dt, mass, arm_mass, length = args.dt, args.mass, args.arm_mass, args.length
        trajectory = get_trajectory(args.command)
        _, angles, enabled = sample(trajectory, dt, args.steps)
        goals = [float(angle) for angle in angles]
        entries = [{"torque_enable": bool(flag)} for flag in enabled]
        log, recorded, q0, dq0 = None, None, args.initial_angle, 0.0
        provenance = f"BAM trajectory {args.command!r} ({trajectory.duration:g}s)"

    # Which pendulums to show, and which model drives each. With none of the
    # --model1/--model2/--models options there is exactly one, chosen by --params.
    requested = [value for value in (args.model1, args.model2) if value is not None]
    requested += [value for group in (args.models or []) for value in group.split(",")]

    if requested:
        if args.params is not None:
            print(
                f"[scene] note: --params {args.params!r} ignored; the --model options "
                "pick the models."
            )
        pendulums: list[Arm] = []
        for value in requested:
            path = resolve_model_arg(value, args.motor)
            label = Path(path).stem
            if any(existing.label == label for existing in pendulums):
                print(
                    f"[scene] {label!r} requested twice; each model needs its own pendulum.",
                    file=sys.stderr,
                )
                return 1
            pendulums.append(arm_for(label, path))
    else:
        params_arg = args.params if args.params is not None else "sts3215/m5"
        if params_arg.lower() == "none":
            pendulums = [arm_for("model")]
        else:
            resolved = resolve_params_file(params_arg)
            pendulums = [arm_for(Path(resolved).stem, resolved)]

    firmware, loop_firmware = firmware_overrides(log, args.motor)

    # Rendering is decimated to roughly 60 frames per second of wall-clock time: at
    # dt = 5 ms that is one frame every few physics steps. Scaling it with --speed
    # keeps slow motion smooth instead of dropping to a stutter.
    render_interval = max(1, round(args.speed / (60.0 * dt))) if args.speed > 0.0 else 1

    print(f"[scene] source    : {provenance}  ({len(goals)} steps, dt={dt:.6f}s)")
    print(f"[scene] testbench : mass={mass} arm_mass={arm_mass} length={length}  "
          f"start={q0:+.3f} rad")
    print(f"[scene] motor     : {args.motor}")
    print("[scene] firmware  : " + (" ".join(f"{k}={v:g}" for k, v in firmware.items())
                                     or "(motor module defaults)"))
    if loop_firmware:
        print("[scene] loop gains: " + " ".join(f"{k}={v:g}" for k, v in loop_firmware.items()))
    print(f"[scene] pendulums : {len(pendulums)}")
    for pendulum in pendulums:
        swatch = "".join(f"{int(round(255 * channel)):02x}" for channel in pendulum.colour)
        print(f"    {pendulum.label:<10} #{swatch}  params={pendulum.params_file}")
    if len(pendulums) > 1:
        print(
            "[scene] layout    : "
            + ("overlaid on one pivot (no collisions)" if args.overlay else "spread along X")
        )
    if args.visual:
        print(f"[scene] playback  : speed={args.speed:g}x, rendering every "
              f"{render_interval} physics step(s)")
    if not enabled.all():
        print(f"[scene] note      : torque is disabled for {int((~enabled).sum())} of "
              f"{len(enabled)} steps; the arm back-drives under gravity there.")

    actuator_cfgs = {
        pendulum.slug: BamActuatorCfg(
            # Anchored: an unanchored "joint_m1" would also match "joint_m10".
            joint_names_expr=[f"^{re.escape(pendulum.joint_name)}$"],
            effort_limit=100.0,
            velocity_limit=100.0,
            physics_dt=dt,  # must equal the sim dt or the slew limiter is wrong
            motor=args.motor,
            params_file=pendulum.params_file,
            # All friction lives in the actuator model, so the solver must add none
            # of its own or it would be double-counted.
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
            motor_params=loop_firmware,
            **firmware,
        )
        for pendulum in pendulums
    }

    # --- build the scene -------------------------------------------
    urdf = Path(tempfile.mkdtemp()) / "bam_pendulum.urdf"
    urdf.write_text(build_urdf(mass, arm_mass, length, pendulums, overlay=args.overlay))

    sim = SimulationContext(
        SimulationCfg(
            dt=dt,
            render_interval=render_interval,
            device=args.device,
            gravity=(0.0, 0.0, GRAVITY),
        )
    )
    if args.visual:
        # Frame the rig. The arms all rotate about +Y, so a camera looking along -Y shows
        # every swing face-on. Without this, arms a few tens of centimetres long are
        # specks near the world origin.
        if args.overlay:
            # The pivots coincide, so the extent is one arm's swing circle rather than a
            # row. Yaw off the rotation axis, though: the arms are separated *along* it,
            # and looking straight down it would hide all but the nearest one.
            span = length
            eye = [1.15 * span, -2.45 * span, 0.9 * span]
        else:
            span = max(
                length, (len(pendulums) - 1) * pendulum_spacing(length) / 2.0 + length
            )
            eye = [0.25 * span, -2.6 * span, 0.9 * span]
        sim.set_camera_view(eye=eye, target=[0.0, 0.0, -0.5 * length])
    scene = InteractiveScene(make_scene_cfg(str(urdf), actuator_cfgs))
    sim.reset()

    robot = scene["pendulum"]
    device = robot.device
    num_joints = robot.num_joints

    # One env, so the regex namespace resolves to env_0. If the scene already
    # substituted it, the replace is a no-op either way.
    colour_links(
        robot.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0"), pendulums
    )

    # Look the joints up by name rather than trusting the articulation's joint order.
    joint_ids = {p.slug: robot.find_joints(p.joint_name)[0][0] for p in pendulums}
    actuators = {p.slug: robot.actuators[p.slug] for p in pendulums}
    offsets = {
        p.slug: float(getattr(actuators[p.slug].motor, "q_offset", 0.0)) for p in pendulums
    }
    gravity_gain = (mass + arm_mass / 2.0) * GRAVITY * length

    def rollout() -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """One pass over the whole command sequence, paced for viewing.

        Same ordering as ``offline_rollout.drive``: the state is recorded *before*
        the step, and the controllers see the state they would have seen.
        """
        # Isaac Lab's documented way to restore an articulation: write the joint
        # state, then call ``reset`` to drop the internal buffers *and* the
        # actuators' own state - which is where the slew-limited internal target
        # lives. Skipping it would leak one pass's target into the next.
        robot.write_joint_state_to_sim(
            torch.full((1, num_joints), q0, device=device),
            torch.full((1, num_joints), dq0, device=device),
        )
        robot.reset()

        positions = {p.slug: [] for p in pendulums}
        velocities = {p.slug: [] for p in pendulums}
        wall_start = time.perf_counter()
        for i, (goal, enable) in enumerate(zip(goals, enabled)):
            # Record the state the controllers are about to act on (harness ordering).
            for pendulum in pendulums:
                index = joint_ids[pendulum.slug]
                positions[pendulum.slug].append(float(robot.data.joint_pos[0, index]))
                velocities[pendulum.slug].append(float(robot.data.joint_vel[0, index]))

            # Every pendulum gets the same command, so the only difference between
            # them is the model under test.
            robot.set_joint_position_target(
                torch.full((1, num_joints), float(goal), device=device)
            )

            for pendulum in pendulums:
                key = pendulum.slug
                # Load-dependent friction needs the external torque. BAM passes the
                # testbench bias here; the env has to supply it because PhysX will
                # not, and each arm's bias depends on its own angle.
                q_physics = robot.data.joint_pos[0, joint_ids[key]] + offsets[key]
                actuators[key].set_external_torque(gravity_gain * torch.sin(q_physics))
                actuators[key].set_torque_enable(bool(enable))

            robot.write_data_to_sim()  # runs every actuator group's compute()
            sim.step(render=False)
            robot.update(dt)

            # Wait out the wall-clock time this step represents. Without this the
            # whole run finishes in a fraction of a second, long before the viewport
            # has drawn anything, so the window shows a pendulum that never moves.
            if args.speed > 0.0:
                remaining = wall_start + (i + 1) * dt / args.speed - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)

            if args.visual and i % render_interval == 0:
                sim.render()

        return positions, velocities

    positions, velocities = rollout()

    # --- comparison: against the offline harness, once per model -----
    import json

    from bam_actuators.friction import BamFrictionModel
    from bam_actuators.motors import get_motor

    class _Testbench:
        """BAM's ``testbench.Pendulum``, inlined.

        Three lines of arithmetic, so there is no reason to require the BAM
        checkout just to run this comparison.
        """

        def compute_bias(self, q, dq):
            return gravity_gain * np.sin(q)

        def compute_mass(self, q, dq):
            return mass * length**2 + (arm_mass / 3.0) * length**2

    offline = {}
    for pendulum in pendulums:
        live = actuators[pendulum.slug]
        # Use the motor the actuator actually resolved - a params file can override
        # --motor - so both sides are guaranteed to run the same control law, and
        # copy *every* parameter rather than a hand-picked list: the current laws
        # have values (current_limit, error_gain_ratio, V0, kp_ratio, the md01c loop
        # gains) that a kp/vin/error_gain/max_pwm list would silently drop.
        motor = get_motor(live.motor_name)()
        motor.set_params(live.motor.get_params())
        friction = BamFrictionModel.from_params({}, model=None)
        if pendulum.params_file:
            motor.set_params(json.loads(Path(pendulum.params_file).read_text()))
            friction = BamFrictionModel.from_json(pendulum.params_file)

        harness = OfflineSimulator(
            _Testbench(), motor, friction, dt, q_offset=offsets[pendulum.slug]
        )
        harness.reset(q0, dq0)
        offline[pendulum.slug] = drive(harness, entries, goals)

    # --- report -----------------------------------------------------
    def gap(measured: np.ndarray, reference: np.ndarray) -> str:
        difference = measured - reference
        return (
            f"max|dq|={np.abs(difference).max():.6f}  "
            f"rms={np.sqrt((difference**2).mean()):.6f}"
        )

    print(f"\n[scene] {len(goals)} steps simulated, {len(pendulums)} pendulum(s)")
    for pendulum in pendulums:
        key = pendulum.slug
        # A params file can override --motor, so label each row with what it loaded.
        row = (
            f"  {pendulum.label:<8} motor={actuators[key].motor_name:<8}"
            f" final isaac={positions[key][-1]:+.4f}"
            f"  offline={offline[key]['positions'][-1]:+.4f}"
        )
        if recorded is not None:
            row += f"  recorded={recorded[-1]:+.4f}"
        print(row)
        measured = np.array(positions[key])
        print(f"  {'':<8} Isaac vs offline harness: {gap(measured, offline[key]['positions'])} rad")
        if recorded is not None:
            print(f"  {'':<8} Isaac vs recording:       {gap(measured, recorded)} rad")

    if args.csv:
        columns = [np.array(goals), enabled.astype(float)]
        header = ["goal", "torque_enable"]
        if recorded is not None:
            columns.append(recorded)
            header.append("recorded_pos")
        for pendulum in pendulums:
            key = pendulum.slug
            columns += [
                np.array(positions[key]),
                np.array(velocities[key]),
                offline[key]["positions"],
            ]
            header += [f"{key}_isaac_pos", f"{key}_isaac_vel", f"{key}_offline_pos"]
        np.savetxt(
            args.csv, np.column_stack(columns), delimiter=",", header=",".join(header), comments=""
        )
        print(f"\n[scene] wrote {args.csv}")

    print("\n[scene] done. A large 'vs offline' gap means the scene differs (timestep, "
          "\ninertia or armature). If you replayed a recording, a large 'vs recording' gap "
          "\nmeans the model disagrees with the servo - which is the one worth tuning.")

    # --- keep the window useful after the run ----------------------
    if args.visual and args.loop:
        print("\n[scene] looping. Close the window (or Ctrl+C) to stop.")
        try:
            while simulation_app.is_running():
                rollout()
        except KeyboardInterrupt:
            pass
    elif args.visual:
        print("\n[scene] window left open so you can look at the final pose. Close it (or "
              "Ctrl+C)"
              "\n[scene] to exit, or pass --loop to keep it moving.")
        try:
            while simulation_app.is_running():
                sim.render()
        except KeyboardInterrupt:
            pass

    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        simulation_app.close()
    sys.exit(code)
