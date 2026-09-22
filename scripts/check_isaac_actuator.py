# Copyright (c) 2026, Haris Husnain
# SPDX-License-Identifier: Apache-2.0

"""Framework-level check of BamActuator inside Isaac Lab.

Instantiates the actuator *directly* - no scene, no articulation, no asset. That is
enough to exercise the entire Isaac-facing surface, which is the part the offline
tests cannot reach:

* the ``@configclass`` definitions (including relaxing ``stiffness``/``damping``),
* ``ActuatorBase`` construction, ``class_type`` resolution and ``_clip_effort``,
* the ``ArticulationActions`` contract, and
* ``compute()`` returning finite efforts of the right shape.

One case per control-law family: the voltage law (``md01``), the current law with a
bundled identified model (``md01i/m3``), the measured AT32 loops (``md01c``) and the
stateful STS3215 (``sts3215/m5``).

Run it before wiring the actuator into a task: if this passes, anything that then
goes wrong is about the scene, not about the extension.

    python scripts/check_isaac_actuator.py            # headless
    python scripts/check_isaac_actuator.py --visual   # with a viewport

Exit code is 0 on success, 1 on the first failure.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-check BamActuator without a scene.")
parser.add_argument("--visual", action="store_true", help="Show the viewport (default: headless).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.visual

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Kit must be up before these are imported -------------------------
import torch  # noqa: E402

from isaaclab.utils.types import ArticulationActions  # noqa: E402

from bam_actuators.actuators import BamActuator, BamActuatorCfg  # noqa: E402

DT = 0.005
NUM_ENVS = 2


def base_cfg(**overrides) -> BamActuatorCfg:
    """A minimal actuator config: one joint group, deliberately tight limits.

    ``effort_limit`` is small on purpose. Every case has to exceed it under the huge
    command below, or the clipping check is vacuous - and the lowest of them (the
    voltage-law ``md01``, whose duty cycle is capped by its current limit) lands at
    about 0.175 Nm.
    """
    defaults = dict(
        joint_names_expr=[".*"],
        effort_limit=0.15,
        velocity_limit=30.0,
        physics_dt=DT,
        # All friction lives in the actuator model, so the solver must add none of
        # its own or it would be double-counted.
        friction=0.0,
        dynamic_friction=0.0,
        viscous_friction=0.0,
    )
    defaults.update(overrides)
    return BamActuatorCfg(**defaults)


def build(cfg: BamActuatorCfg) -> BamActuator:
    """Construct the actuator the way Isaac Lab does - through ``cfg.class_type``."""
    return cfg.class_type(
        cfg=cfg,
        joint_names=["j0"],
        joint_ids=torch.tensor([0], device=args.device),
        num_envs=NUM_ENVS,
        device=args.device,
        # USB defaults for the PD gains; our firmware law does not use them.
        stiffness=0.0,
        damping=0.0,
    )


def check(label: str, cfg: BamActuatorCfg) -> None:
    print(f"\n--- {label} ---")
    actuator = build(cfg)
    print(actuator)  # ActuatorBase.__str__ - proves Isaac Lab's machinery engaged
    print(f"  motor={actuator.motor_name}  friction model={actuator.friction_model.model_name} "
          f"terms={actuator.friction_model.active_flags}")

    assert actuator.is_implicit_model is False, "BamActuator must be explicit"

    q = torch.full((NUM_ENVS, 1), 0.3, device=args.device)
    dq = torch.zeros_like(q)
    q_target = torch.full((NUM_ENVS, 1), 0.5, device=args.device)

    # A couple of steps, so a stateful control law exercises its slew limiter too.
    efforts = []
    for _ in range(3):
        action = actuator.compute(
            ArticulationActions(joint_positions=q_target, joint_velocities=None, joint_efforts=None),
            q,
            dq,
        )
        assert action.joint_positions is None, "positions must be handed over as efforts"
        assert action.joint_velocities is None, "velocities must be handed over as efforts"
        assert action.joint_efforts is not None
        assert action.joint_efforts.shape == (NUM_ENVS, 1), action.joint_efforts.shape
        assert torch.isfinite(action.joint_efforts).all(), "non-finite effort"

        efforts.append(float(action.joint_efforts[0, 0]))
        print(f"    step: applied_effort={efforts[-1]:+.6f} Nm "
              f"(computed={float(actuator.computed_effort[0, 0]):+.6f})")

    assert any(abs(value) > 1e-9 for value in efforts), "a 0.2 rad error produced no torque at all"

    # Clipping: a huge command must still respect effort_limit.
    action = actuator.compute(
        ArticulationActions(joint_positions=q_target + 10.0, joint_velocities=None, joint_efforts=None),
        q,
        dq,
    )
    peak = float(action.joint_efforts.abs().max())
    assert peak <= cfg.effort_limit + 1e-6, f"effort {peak} exceeded effort_limit {cfg.effort_limit}"
    print(f"  effort_limit respected (peak {peak:.6f} <= {cfg.effort_limit})")


def main() -> int:
    # 1. The voltage law with no params file at all: the module's own seeds are the
    #    model, exactly as BAM's MD01Actuator.initialize() seeds it.
    check("md01 (voltage law, module defaults)", base_cfg(motor="md01"))

    # 2. The current law with a bundled identified model. The file decides the motor
    #    ("actuator": "md01i"), the friction maths ("model": "m3") and every
    #    identified value - including the current limit, which no cfg field carries.
    check("md01i/m3 (bundled params file)", base_cfg(params_file="md01i/m3"))

    # 3. The measured AT32 loops, whose firmware constants (kp_current, kff_current,
    #    cap_ma) are module defaults rather than params-file values.
    check("md01c (loop law, module defaults)", base_cfg(motor="md01c"))

    # 4. A stateful control law: the STS3215's slew limiter is only exercised if the
    #    same actuator instance is stepped more than once.
    check("sts3215/m5 (bundled params file)", base_cfg(params_file="sts3215/m5"))

    print("\nOK: BamActuator works inside Isaac Lab.")
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
