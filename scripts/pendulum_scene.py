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

from isaaclab.app import AppLauncher

# The offline harness lives with the tests; import it rather than duplicate it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

parser = argparse.ArgumentParser(description="Pendulum test scene for BamActuator.")
parser.add_argument("--log", type=str, default=None, help="Recorded log to replay (default: BAM data_raw).")
parser.add_argument("--steps", type=int, default=400, help="Number of log steps to replay.")
parser.add_argument("--motor", type=str, default="sts3215", help="Motor module name.")
parser.add_argument("--params", type=str, default="sts3215/m5", help="Params file, or 'none'.")
parser.add_argument("--csv", type=str, default=None, help="Write the trajectories to this CSV.")
parser.add_argument("--visual", action="store_true", help="Show the viewport (default: headless).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.visual

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
    """The log the parity tests use, if BAM's checkout is where we expect."""
    from bam_paths import bam_root

    return bam_root() / "data_raw" / "2026-09-10_15h22m23.json"


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
    <collision>
      <origin xyz="0 0 {-length / 2.0:.12g}" rpy="0 0 0"/>
      <geometry><cylinder radius="{ROD_RADIUS}" length="{length:.12g}"/></geometry>
    </collision>
  </link>
  <link name="tip">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.000001"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual><geometry><sphere radius="{TIP_RADIUS}"/></geometry></visual>
  </link>
  <joint name="tip_fixed" type="fixed">
    <parent link="arm"/>
    <child link="tip"/>
    <origin xyz="0 0 {-length:.12g}" rpy="0 0 0"/>
  </joint>
</robot>
"""


def firmware_overrides(log: dict, motor: str) -> dict:
    """Mirror BAM's ``Actuator.load_log`` so the Isaac run matches the harness.

    Which fields a recording supplies is motor-specific, so ask BAM when it is
    importable rather than hard-coding the rule.
    """
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
        print(f"[scene] BAM not available ({exc}); falling back to kp/vin from the log.")
        print("[scene] WARNING: motors that also take error_gain/max_pwm from the log will differ.")
        return {"vin": log["vin"], "firmware_kp": log["kp"]}


def make_scene_cfg(urdf_path: str, dt: float, actuator_cfg: BamActuatorCfg, num_envs: int = 1):
    """Scene with the pendulum, a ground plane and a light."""

    @configclass
    class _PendulumSceneCfg(InteractiveSceneCfg):
        """A single pendulum on a ground plane."""

        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(
            prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
        )
        pendulum = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/pendulum",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=urdf_path,
                fix_base=True,
                make_instanceable=False,
                # No drive: BamActuator is the only source of joint torque.
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(target_type="none"),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={"pivot": 0.0}, joint_vel={"pivot": 0.0}
            ),
            actuators={"joint": actuator_cfg},
        )

    return _PendulumSceneCfg(num_envs=num_envs, env_spacing=2.0)


def main() -> int:
    log_path = Path(args.log) if args.log else default_log()
    if not log_path.is_file():
        print(f"[scene] log not found: {log_path}\n"
              f"        pass --log /path/to/data_raw/<file>.json", file=sys.stderr)
        return 1

    log = load_raw_log(str(log_path))
    entries = log["entries"][: args.steps]
    goals = [entry["goal_position"] for entry in entries]
    dt = log["dt"]
    mass, arm_mass, length = log["mass"], log["arm_mass"], log["length"]

    params_file = None if args.params.lower() == "none" else resolve_params_file(args.params)
    firmware = firmware_overrides(log, args.motor)

    print(f"[scene] log       : {log_path.name}  ({len(entries)} steps, dt={dt:.6f}s)")
    print(f"[scene] testbench : mass={mass} arm_mass={arm_mass} length={length}")
    print(f"[scene] motor     : {args.motor}  params={params_file}")
    print(f"[scene] firmware  : " + " ".join(f"{k}={v:g}" for k, v in firmware.items()))

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

    # Start where the recording starts, exactly like the harness does.
    q0, dq0 = float(entries[0]["position"]), float(entries[0].get("speed", 0.0))
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
    recorded = np.array([entry["position"] for entry in entries])

    # --- comparison 1: against the offline harness -------------------
    from bam_actuators.friction import BamFrictionModel
    from bam_actuators.motors import get_motor

    our_motor = get_motor(args.motor)()
    our_motor.set_params(
        {
            "kp": firmware["firmware_kp"],
            "vin": firmware["vin"],
            "error_gain": firmware["error_gain"],
            "max_pwm": firmware["max_pwm"],
        }
    )
    our_friction = BamFrictionModel.from_params({}, model=None)
    if params_file:
        import json

        data = json.loads(Path(params_file).read_text())
        our_motor.set_params(data)
        our_friction = BamFrictionModel.from_json(params_file)

    class _Bias:
        """Minimal testbench: BAM's bias torque, with BAM's inertia."""

        def __init__(self):
            from bam.testbench import Pendulum

            self._pendulum = Pendulum(log)

        def compute_bias(self, q, dq):
            return gravity_gain * np.sin(q)

        def compute_mass(self, q, dq):
            return self._pendulum.compute_mass(q, dq)

    harness = OfflineSimulator(_Bias(), our_motor, our_friction, dt, q_offset=q_offset)
    harness.reset(q0, dq0)
    offline = drive(harness, entries, goals)

    # --- report -----------------------------------------------------
    def summarise(label: str, reference: np.ndarray) -> None:
        delta = isaac["positions"] - reference
        print(f"  {label:<28} max|dq|={np.abs(delta).max():.6f} rad   "
              f"rms={np.sqrt((delta**2).mean()):.6f} rad")

    print(f"\n[scene] {len(positions)} steps simulated")
    print(f"  final position : isaac={positions[-1]:+.4f}  "
          f"offline={offline['positions'][-1]:+.4f}  recorded={recorded[-1]:+.4f} rad")
    print("  Isaac vs offline harness (same model, different integrator):")
    summarise("position", offline["positions"])
    print("  Isaac vs the recording (model vs physical servo):")
    summarise("position", recorded)

    if args.csv:
        np.savetxt(
            args.csv,
            np.column_stack([np.array(goals), isaac["positions"], isaac["velocities"],
                             recorded, offline["positions"]]),
            delimiter=",",
            header="goal,isaac_pos,isaac_vel,recorded_pos,offline_pos",
            comments="",
        )
        print(f"\n[scene] wrote {args.csv}")

    print("\n[scene] done. A large 'vs offline' gap means the scene differs (timestep, "
          "\ninertia or armature); a large 'vs recording' gap means the model disagrees "
          "\nwith the servo - and that is the one worth tuning.")
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
