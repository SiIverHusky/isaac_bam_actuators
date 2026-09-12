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

It then replays a recorded log's ``goal_position`` sequence through the actuator
and reports two comparisons:

* **vs the offline harness** - our model simulated twice, in Isaac (PhysX) and in
  BAM's own loop. Differences here are integrator/solver, not model.
* **vs the recording** - the model against the physical servo. This is the one that
  answers "does it react like it's supposed to".

Usage::

    python scripts/pendulum_scene.py                          # headless, sensible defaults
    python scripts/pendulum_scene.py --visual                 # with a viewport
    python scripts/pendulum_scene.py --motor md01 --params none --steps 300
    python scripts/pendulum_scene.py --csv /tmp/traj.csv
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

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
parser.add_argument("--steps", type=int, default=400, help="Number of steps to simulate.")
parser.add_argument(
    "--motor",
    type=str,
    default="sts3215",
    help="Motor name as BAM's params files write it, e.g. 'sts3215' or 'md01' - not the "
    "module filename. The params file's 'actuator' key overrides this. See --list-motors.",
)
parser.add_argument("--params", type=str, default="sts3215/m5", help="Params file, or 'none'.")
parser.add_argument("--list-motors", action="store_true", help="List valid --motor values and exit.")
# Rig and drive, used when there is no recording.
parser.add_argument("--mass", type=float, default=0.5, help="Tip mass [kg].")
parser.add_argument("--arm-mass", type=float, default=0.02, help="Arm mass [kg].")
parser.add_argument("--length", type=float, default=0.15, help="Arm length [m].")
parser.add_argument("--dt", type=float, default=0.005, help="Timestep [s] for synthetic runs.")
parser.add_argument(
    "--command",
    type=str,
    default="steps",
    choices=["steps", "square", "sine", "hold"],
    help="Command sequence to drive the pendulum with when there is no recording.",
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

from offline_rollout import OfflineSimulator, drive, load_raw_log  # noqa: E402

#: Must match BAM's ``bam.testbench.Pendulum`` exactly.
GRAVITY = -9.80665
#: Small, so the tip mass behaves like the point mass BAM models it as.
TIP_RADIUS = 0.01
ROD_RADIUS = 0.005


def default_log() -> Path:
    """The recording the parity tests use, if BAM's checkout is where we expect."""
    return BAM_ROOT / "data_raw" / "2026-09-10_15h22m23.json"


def build_urdf(mass: float, arm_mass: float, length: float) -> str:
    """A pendulum whose rigid-body dynamics equal BAM's testbench.

    Derived quantities, with ``m = mass + arm_mass``:

    * ``d = (mass + arm_mass / 2) * L / m`` -- COM distance that reproduces the
      gravity torque ``(mass + arm_mass/2) * g * L * sin(q)``;
    * ``I_com = I_pivot - m * d^2`` -- parallel-axis shift so the swing inertia
      about the pivot is ``I_pivot = mass*L^2 + (arm_mass/3)*L^2``.

    The joint is continuous about +Y, so ``q = 0`` is the arm hanging down and
    positive ``q`` is counter-clockwise - BAM's convention, which makes
    ``tau_y = -(mass + arm_mass/2) * 9.80665 * L * sin(q)`` on both sides.

    Everything lives on one link on purpose: a separate tip link joined by a fixed
    joint would be merged by the importer, perturbing the inertia derived above.
    There is no ``<collision>`` geometry either - nothing in this scene should
    contact anything, and a shape here would be a chance for the scene to stop
    matching BAM's contact-free single-axis model.
    """
    total_mass = mass + arm_mass
    com_distance = (mass + arm_mass / 2.0) * length / total_mass
    inertia_pivot = mass * length**2 + (arm_mass / 3.0) * length**2
    inertia_com = inertia_pivot - total_mass * com_distance**2
    # Isotropic: only Iyy matters for this 1-DOF swing, and it is exact.
    i = inertia_com

    return f"""<?xml version="1.0"?>
<robot name="bam_pendulum">
  <link name="base"/>
  <joint name="pivot" type="continuous">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.0" friction="0.0"/>
  </joint>
  <link name="arm">
    <inertial>
      <origin xyz="0 0 {-com_distance:.12g}" rpy="0 0 0"/>
      <mass value="{total_mass:.12g}"/>
      <inertia ixx="{i:.12g}" ixy="0" ixz="0" iyy="{i:.12g}" iyz="0" izz="{i:.12g}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {-length / 2.0:.12g}" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROD_RADIUS}" length="{length:.12g}"/></geometry>
    </visual>
    <visual>
      <origin xyz="0 0 {-length:.12g}" rpy="0 0 0"/>
      <geometry><sphere radius="{TIP_RADIUS}"/></geometry>
    </visual>
  </link>
</robot>
"""


def firmware_overrides(log: dict | None, motor: str) -> dict:
    """Firmware constants to configure the actuator with.

    Firmware is **not** in params files, so it comes from the recording. Which
    fields a recording supplies is motor-specific - ``DCMotorActuator.load_log``
    sets ``kp`` and ``vin``, ``MD01Actuator`` also takes ``error_gain`` and
    ``max_pwm``, ``STS3215Actuator`` overrides nothing - so ask BAM rather than
    hard-coding the rule.

    With no recording there is nothing to mirror, so return nothing and let the
    motor module's own defaults stand. Both sides of the comparison then use those
    same defaults, so the comparison is still fair.

    :param log: The recorded log, or ``None`` for a synthetic run.
    :param motor: Motor name, as BAM's ``actuators`` registry writes it.
    """
    if log is None:
        return {}

    try:
        from bam.actuators import actuators

        reference = actuators[motor]()
        reference.load_log(log)
        return {
            "vin": reference.vin,
            "firmware_kp": reference.kp,
            "error_gain": reference.error_gain,
            "max_pwm": reference.max_pwm,
        }
    except Exception as exc:  # noqa: BLE001 - fall back, but say so
        print(f"[scene] BAM not usable for firmware ({exc}); using kp/vin from the log.")
        print("[scene] WARNING: a motor that also takes error_gain/max_pwm from the log will differ.")
        return {"vin": log["vin"], "firmware_kp": log["kp"]}


def synthetic_commands(name: str, steps: int, dt: float) -> list[float]:
    """A goal-position sequence for runs that have no recording to replay.

    :param name: One of ``"steps"``, ``"square"``, ``"sine"``, ``"hold"``.
    :param steps: Number of samples.
    :param dt: Timestep [s].
    :returns: Target joint positions [rad].
    """
    t = np.arange(steps) * dt
    if name == "steps":
        # A step command is what exercises a stateful slew limiter hardest.
        return [0.0] * (steps // 10) + [1.0] * (steps - steps // 10)
    if name == "square":
        return list(0.6 * np.sign(np.sin(2.0 * np.pi * 0.5 * t)))
    if name == "sine":
        return list(0.8 * np.sin(2.0 * np.pi * 0.5 * t))
    return [0.0] * steps  # "hold"


def make_scene_cfg(urdf_path: str, dt: float, actuator_cfg: BamActuatorCfg, num_envs: int = 1):
    """Scene with the pendulum, a ground plane and a light."""

    @configclass
    class _PendulumSceneCfg(InteractiveSceneCfg):
        """A single pendulum anchored to the world.

        Deliberately **no ground plane**: the joint sits at the env origin and the arm
        hangs to ``z = -length``, so a floor at ``z = 0`` would intersect it and
        introduce contacts that BAM's testbench does not have.
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
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={"pivot": 0.0}, joint_vel={"pivot": 0.0}
            ),
            actuators={"joint": actuator_cfg},
        )

    return _PendulumSceneCfg(num_envs=num_envs, env_spacing=2.0)


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
        entries = log["entries"][: args.steps]
        goals = [entry["goal_position"] for entry in entries]
        recorded = np.array([entry["position"] for entry in entries])
        dt = log["dt"]
        mass, arm_mass, length = log["mass"], log["arm_mass"], log["length"]
        q0, dq0 = float(entries[0]["position"]), float(entries[0].get("speed", 0.0))
        provenance = log_path.name
    else:
        # Nothing recorded: build the rig from the CLI and drive it ourselves.
        dt, mass, arm_mass, length = args.dt, args.mass, args.arm_mass, args.length
        goals = synthetic_commands(args.command, args.steps, dt)
        entries = [{"torque_enable": True} for _ in goals]
        log, recorded, q0, dq0 = None, None, 0.0, 0.0
        provenance = f"synthetic ({args.command})"

    params_file = None if args.params.lower() == "none" else resolve_params_file(args.params)
    firmware = firmware_overrides(log, args.motor)

    print(f"[scene] source    : {provenance}  ({len(goals)} steps, dt={dt:.6f}s)")
    print(f"[scene] testbench : mass={mass} arm_mass={arm_mass} length={length}")
    print(f"[scene] motor     : {args.motor}  params={params_file}")
    print("[scene] firmware  : " + (" ".join(f"{k}={v:g}" for k, v in firmware.items())
                                     or "(motor module defaults)"))

    actuator_cfg = BamActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit=100.0,
        velocity_limit=100.0,
        physics_dt=dt,  # must equal the sim dt or the slew limiter is wrong
        motor=args.motor,
        params_file=params_file,
        **firmware,
    )

    # --- build the scene -------------------------------------------
    urdf = Path(tempfile.mkdtemp()) / "bam_pendulum.urdf"
    urdf.write_text(build_urdf(mass, arm_mass, length))

    sim = SimulationContext(
        SimulationCfg(dt=dt, device=args.device, gravity=(0.0, 0.0, GRAVITY))
    )
    scene = InteractiveScene(make_scene_cfg(str(urdf), dt, actuator_cfg))
    sim.reset()

    robot = scene["pendulum"]
    actuator = robot.actuators["joint"]
    device = robot.device

    robot.write_joint_state_to_sim(
        torch.full((1, 1), q0, device=device), torch.full((1, 1), dq0, device=device)
    )
    robot.reset()

    q_offset = float(getattr(actuator.motor, "q_offset", 0.0))
    gravity_gain = (mass + arm_mass / 2.0) * GRAVITY * length

    positions, velocities = [], []
    for goal in goals:
        # Record the state the controller is about to act on (harness ordering).
        positions.append(float(robot.data.joint_pos[0, 0]))
        velocities.append(float(robot.data.joint_vel[0, 0]))

        robot.set_joint_position_target(torch.full((1, 1), float(goal), device=device))

        # Load-dependent friction needs the external torque. BAM passes the
        # testbench bias here; the env has to supply it because PhysX will not.
        q_physics = robot.data.joint_pos + q_offset
        actuator.set_external_torque(gravity_gain * torch.sin(q_physics))

        robot.write_data_to_sim()  # runs actuator.compute()
        sim.step()
        robot.update(dt)

    isaac = {"positions": np.array(positions), "velocities": np.array(velocities)}

    # --- comparison 1: against the offline harness -------------------
    import json

    from bam_actuators.friction import BamFrictionModel
    from bam_actuators.motors import get_motor

    # Use the motor the actuator actually resolved - the params file can override
    # --motor - so both sides are guaranteed to run the same control law.
    our_motor = get_motor(actuator.motor_name)()
    our_motor.set_params(
        {
            "kp": actuator.motor.kp,
            "vin": actuator.motor.vin,
            "error_gain": actuator.motor.error_gain,
            "max_pwm": actuator.motor.max_pwm,
        }
    )
    our_friction = BamFrictionModel.from_params({}, model=None)
    if params_file:
        our_motor.set_params(json.loads(Path(params_file).read_text()))
        our_friction = BamFrictionModel.from_json(params_file)

    class _Testbench:
        """BAM's ``testbench.Pendulum``, inlined.

        Three lines of arithmetic, so there is no reason to require the BAM
        checkout just to run this comparison.
        """

        def compute_bias(self, q, dq):
            return gravity_gain * np.sin(q)

        def compute_mass(self, q, dq):
            return mass * length**2 + (arm_mass / 3.0) * length**2

    harness = OfflineSimulator(_Testbench(), our_motor, our_friction, dt, q_offset=q_offset)
    harness.reset(q0, dq0)
    offline = drive(harness, entries, goals)

    # --- report -----------------------------------------------------
    def summarise(label: str, reference: np.ndarray) -> None:
        delta = isaac["positions"] - reference
        print(f"  {label:<28} max|dq|={np.abs(delta).max():.6f} rad   "
              f"rms={np.sqrt((delta**2).mean()):.6f} rad")

    print(f"\n[scene] {len(positions)} steps simulated")
    finals = f"isaac={positions[-1]:+.4f}  offline={offline['positions'][-1]:+.4f}"
    if recorded is not None:
        finals += f"  recorded={recorded[-1]:+.4f}"
    print(f"  final position : {finals} rad")

    print("  Isaac vs offline harness (same model, different integrator):")
    summarise("position", offline["positions"])
    if recorded is not None:
        print("  Isaac vs the recording (model vs physical servo):")
        summarise("position", recorded)

    if args.csv:
        columns = [np.array(goals), isaac["positions"], isaac["velocities"], offline["positions"]]
        header = "goal,isaac_pos,isaac_vel,offline_pos"
        if recorded is not None:
            columns.insert(3, recorded)
            header = "goal,isaac_pos,isaac_vel,recorded_pos,offline_pos"
        np.savetxt(args.csv, np.column_stack(columns), delimiter=",", header=header, comments="")
        print(f"\n[scene] wrote {args.csv}")

    print("\n[scene] done. A large 'vs offline' gap means the scene differs (timestep, "
          "\ninertia or armature). If you replayed a recording, a large 'vs recording' gap "
          "\nmeans the model disagrees with the servo - which is the one worth tuning.")
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
