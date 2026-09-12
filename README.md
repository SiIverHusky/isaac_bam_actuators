# isaac_bam_actuators

[BAM](https://github.com/Rhoban/bam) extended friction models (`m1`–`m6`) as an
**Isaac Lab extension**, exposed as an *explicit* actuator.

The repository is laid out the way Isaac Lab expects an external extension to be:

```
isaac_bam_actuators/
├── config/
│   └── extension.toml          # Isaac Sim extension manifest (required)
├── setup.py                    # reads extension.toml, pip-installable
├── pyproject.toml
├── bam_actuators/              # the Python module named in extension.toml
│   ├── __init__.py             # lazy public API
│   ├── friction.py             # torch port of BAM's friction budget (m1–m6)
│   ├── params.py               # locates the bundled identified models
│   ├── params/                 # identified models, shipped with the extension
│   │   └── sts3215/m1..m6.json
│   ├── motors/                 # ONE MODULE PER SERVO: the control laws
│   │   ├── __init__.py         # auto-discovers the modules below
│   │   ├── base.py             # MotorBase: DC torque + helpers + state hooks
│   │   ├── generic.py          # plain voltage-P servo (fallback)
│   │   ├── feetech_sts3215.py  # STS3215: slew-limited target + gain ratio
│   │   └── md01.py             # MD01: P law + firmware current limit
│   └── actuators/
│       ├── __init__.py
│       └── bam_actuator.py     # BamActuator: motor + friction -> joint effort
└── tests/
    ├── test_friction.py        # the m1–m6 maths
    ├── test_motors.py          # each control law vs BAM's actuator class
    └── test_sts3215_m5.py      # real identified models, end to end
```

Everything under `bam_actuators/` except `actuators/` is **pure torch** — no Isaac
Lab, no hardware — so the friction maths and the control laws can be tested
against BAM's reference implementations without launching a simulator.

## Adding a motor

Drop a `<name>.py` into `bam_actuators/motors/` with a `MotorBase` subclass; it is
picked up automatically (the `name` is what BAM params files call the servo):

```python
from .base import MotorBase

class MyServoMotor(MotorBase):
    name = "my_servo"                 # matches "actuator" in the params JSON
    firmware = {"vin": 12.0, "kp": 32.0, "error_gain": 0.166,
                "max_pwm": 0.97, "max_current": None}
    parameters = {"kt": 0.78, "R": 2.0, "armature": 1e-4, "q_offset": 0.0}

    def control(self, q_target, q, dq, dt):
        return self._volts(self._duty_from_error(q_target - q))
```

`self._duty_from_error`, `self._apply_current_limit` and `self._volts` are the
shared firmware building blocks; override `control` to do something else (as the
STS3215 does with its slew limiter), and set `stateful = True` if you keep state.


## How an Isaac Lab extension works

A few facts that drive the whole layout:

| Piece | Why it exists |
| --- | --- |
| `config/extension.toml` | The **extension manifest**. Isaac Sim's extension manager reads it to discover the extension, and `[[python.module]]` tells it which Python module to import. Without this file the directory is just a folder. |
| `setup.py` | Makes the extension `pip install -e`-able. It reads the version/repository/keywords out of `extension.toml` so the two never drift. Isaac Lab discovers tasks and assets through the installed package, not through the source tree. |
| `[dependencies]` in the manifest | Extensions, not pip packages. `"isaaclab" = {}` means "Isaac Sim must have the `isaaclab` extension enabled before mine loads". |
| Module `__init__.py` | Whatever is imported here is what the extension "is". Registering a `gym` task happens here; for a pure actuator library there is nothing to register — you opt in from your own `ArticulationCfg`. |

To see a freshly generated reference layout for your exact Isaac Lab version:

```bash
/path/to/IsaacLab/isaaclab.sh --new      # generates an external project
```

## Install

```bash
# from the Isaac Lab python environment
python -m pip install -e /home/hharis/Mangdang/isaac_bam_actuators
```

## Use

Explicit actuators are selected by putting their config into the `actuators`
dictionary of an `ArticulationCfg`:

```python
from isaaclab.assets import ArticulationCfg
from bam_actuators.actuators import BamActuatorCfg

MY_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=MY_USD_CFG,
    actuators={
        "legs": BamActuatorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit=20.0,
            velocity_limit=30.0,
            # Stateful control laws (sts3215) need the timestep, which Isaac Lab
            # never passes to an actuator.
            physics_dt=1.0 / 200.0,
            # Point at a bundled model and use it as-is: the file names the motor
            # ("actuator"), the friction maths ("model"), and the identified
            # parameters of both.
            params_file="sts3215/m5",
        ),
    },
)
```

Firmware constants (`vin`, `firmware_kp`, `error_gain`, `max_pwm`, `max_current`)
live on the motor module and only need setting to *override* them:

```python
BamActuatorCfg(
    joint_names_expr=[".*_joint"],
    effort_limit=20.0, velocity_limit=30.0,
    motor="md01",                  # or omit and let params_file's "actuator" decide
    vin=11.1,                      # override the motor's nominal 12.0 V
    physics_dt=1.0 / 200.0,
)
```

### Loading BAM params

A BAM params file is self-describing, so loading one is all you need:

```python
BamActuatorCfg(
    joint_names_expr=[".*_joint"],
    effort_limit=20.0, velocity_limit=30.0, physics_dt=1.0 / 200.0,
    params_file="sts3215/m5",      # bundled, or any path to a .json
)
```

| Key in the JSON | Applied to |
| --- | --- |
| `actuator` (`"sts3215"`) | **which control law runs** (from `bam_actuators/motors/`) |
| `model` (`"m5"`) | which friction terms are evaluated |
| `kt`, `R`, `armature`, `q_offset` | the motor model |
| `error_gain_ratio`, `max_velocity`, `command_delay` | the motor model, if that motor uses them |
| `friction_*`, `load_friction_*`, `dtheta_stribeck`, `alpha` | the friction model |

`kt`, `R`, `armature` and `q_offset` are the only parameters a *friction* model needs
but that belong to the motor, so the actuator routes them to the motor module.

Firmware constants are **not** in params files at all — they live on the motor module.
BAM's ``load_log`` decides which of them a *recording* supplies, and that differs per
motor: ``DCMotorActuator.load_log`` sets ``kp`` and ``vin``; ``MD01Actuator`` also takes
``error_gain`` and ``max_pwm``; ``STS3215Actuator`` overrides nothing, so its
``error_gain`` stays at the class default 0.166 whatever the log says. Read them off
BAM's actuator rather than assuming.

If you have no file, pick the friction variant directly:

```python
friction_model=BamFrictionCfg(model="m5")                     # m5 maths, default params
friction_model=BamFrictionCfg(model="m6", quadratic=False)    # ablate one term
friction_model=BamFrictionCfg(stribeck=True)                  # or set flags by hand
```

Precedence is: defaults (m1) → variant from `model` / the file → flags you set
explicitly. So a file always decides its own variant, and individual flags can
still ablate one term on top of it.

### What `compute()` does

Per environment and joint, every step:

```
q_physics    = joint_pos + q_offset                                        # rig offset: physics only
volts        = motor.control(q_target, joint_pos, dq, dt)                   # per-motor firmware
motor_torque = kt * volts / R - kt^2 * dq / R                               # DC motor + back-EMF
net_torque   = motor_torque + external_torque
tau_stop     = net_torque + (inertia / dt) * dq                             # torque to stop in dt
tau_friction = -sign(tau_stop) * min(|tau_stop|, frictionloss + damping*|dq|)
applied      = clip(net_torque + tau_friction, -effort_limit, +effort_limit)
```

Note the `q_offset` asymmetry: it follows BAM, where the firmware sees the raw
joint angle but the physics sees the offset one. The identified value is only
meaningful under that convention.

The control law is `Motor.control` from the selected module (see
`bam_actuators/motors/`); the DC motor torque is shared by every voltage-controlled
servo; `frictionloss` / `damping` come from
`bam_actuators.friction.BamFrictionModel`, a vectorized torch port of
`bam.model.Model.compute_frictions`. The stopping-torque clip is BAM's Algorithm 1,
which is what produces realistic stiction: a joint at rest does not drift under a
load smaller than the friction budget.

## Caveats

- **Set the solver's joint friction to zero.** `friction`, `dynamic_friction` and
  `viscous_friction` in `ActuatorBaseCfg` are applied by PhysX *in addition* to
  this model, so leaving them at their USD defaults double-counts friction.
- **Load-dependent models need the external torque.** Isaac Lab does not hand the
  actuator the gravity/contact torque, so the external torque defaults to zero and
  the load-dependent terms (`m3`–`m6`) are inactive. Call
  `actuator.set_external_torque(tau)` from your environment each step to feed it.
- **Stateful control laws need `physics_dt`.** The STS3215 slews an internal target
  at `max_velocity * dt`, and Isaac Lab never passes the timestep to an actuator,
  so `physics_dt` must be set or construction raises.
- **`command_delay` is loaded but not applied.** BAM shifts the goal sequence
  offline; there is no transport-delay buffer here yet, and the actuator prints a
  note when the identified value is non-zero.
- **The stopping-torque term is off by default.** Set
  `use_joint_armature_as_inertia=True` (with `physics_dt`) to enable it. Note it
  then uses only the *rotor* armature, whereas the stopping torque is defined with
  total joint inertia, so it is a lower bound.
- **The params file decides the motor and the model variant.** `params_file="sts3215/m5"`
  runs the `sts3215` control law and the `m5` maths — you never set either by hand.
  `motor` / `friction_model.model` are only used when there is no file, and an
  invalid combination (e.g. `quadratic` without `stribeck`) raises.

## Tests

The suite is pure `torch` (+ `numpy` for the parity comparisons) — no Isaac Lab:

```bash
python -m pytest tests/
```

| File | What it covers |
| --- | --- |
| `test_friction.py` | the m1–m6 budget maths |
| `test_motors.py` | each control law vs BAM's actuator class |
| `test_sts3215_m5.py` | the six bundled models, end to end |
| `test_offline_rollout.py` | **full composition**: BAM's pendulum loop driven by our motor + friction, step-by-step against `bam.simulate.Simulator` on a recorded log |

`tests/offline_rollout.py` is the harness behind that last one, and is also a
usable reference for the external-torque hook: it feeds the testbench's gravity
torque into `BamFrictionModel` exactly as an Isaac env must via
`set_external_torque()` for the load-dependent terms to do anything.

The parity tests load BAM from its source checkout (`bam` cannot be pip-installed
here: its `requires-python` is `>=3.12`) and skip cleanly when it is absent.

