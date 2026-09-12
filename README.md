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
│   ├── trajectory.py           # port of BAM's identification motions
│   ├── testbench.py            # the URDF realising BAM's pendulum testbench
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
    ├── test_trajectory.py      # the trajectory port vs bam.trajectory
    ├── test_testbench.py       # the rig geometry vs bam.testbench.Pendulum
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

## Getting started on an Isaac Lab machine

Prerequisites: Python 3.11 with Isaac Sim 5.x and Isaac Lab installed (Isaac Sim
requires an NVIDIA RTX GPU — there is no CPU-only mode). Run `nvidia-smi` first.

```bash
conda activate <your-isaaclab-env>          # or: ./isaaclab.sh -p <cmd>

# 1. Install this extension (editable). The bundled identified models travel with it.
python -m pip install -e /path/to/isaac_bam_actuators

# 2. Run the offline suite. No simulator, no GPU needed - and it must be green
#    before you debug anything about the scene.
python -m pip install pytest
BAM_ROOT=/path/to/BAM python -m pytest tests/ -q
# -> 97 passed. Without BAM_ROOT, 71 pass and the 26 parity tests skip.

# 3. Check the Isaac Lab integration without building a scene.
python scripts/check_isaac_actuator.py

# 4. Drive a pendulum whose dynamics match BAM's testbench, through BAM's own
#    identification trajectories - one model, or several side by side.
python scripts/pendulum_scene.py --visual --loop             # watch it, over and over
python scripts/pendulum_scene.py                             # headless, paced in real time
python scripts/pendulum_scene.py --visual --model1 m1 --model2 m6
python scripts/pendulum_scene.py --visual --models m1 m3 m6  # any number
python scripts/pendulum_scene.py --visual --models m1 m6 --overlay  # same pivot
python scripts/pendulum_scene.py --visual --command lift_and_drop --speed 0.25
python scripts/pendulum_scene.py --command nothing --initial-angle 1.2 --visual
python scripts/pendulum_scene.py --list-motors                # valid --motor values
python scripts/pendulum_scene.py --csv /tmp/traj.csv
```

**Step 2 matters more than it looks.** It verifies the friction maths, the control
laws and the motor+friction composition against BAM on your recorded logs, which is
everything except Isaac Lab's plumbing and PhysX. If a trajectory then disagrees in
Isaac, the cause is almost certainly the scene, the timestep or the effort limits —
not the model. The parity tests need `BAM_ROOT` because `bam` cannot be
pip-installed (its `requires-python` is `>=3.12`) and is imported from source.

**Step 3** is a direct actuator instantiation: no articulation, no asset. It covers
the `@configclass` definitions, `ActuatorBase` construction, `class_type`
resolution, the `ArticulationActions` contract, effort clipping and a stateful
control law over a few steps.

**Step 4** builds a single-joint pendulum whose rigid-body dynamics are
*analytically* identical to `bam.testbench.Pendulum`, by deriving the URDF's
`<inertial>` block rather than tuning geometry:

```
I_pivot = mass*L^2 + (arm_mass/3)*L^2                 # swing inertia about the pivot
d       = (mass + arm_mass/2) * L / (mass + arm_mass) # COM distance
I_com   = I_pivot - (mass + arm_mass) * d^2           # parallel-axis shift
```

With gravity `-9.80665`, that reproduces BAM's `compute_mass` and `compute_bias`
exactly, so the only remaining difference from the offline harness is the
integrator.

**A recording is optional.** Without one, the rig is built from
`--mass/--arm-mass/--length/--dt` (the defaults are the values every recorded log
carries) and driven by one of **BAM's own identification trajectories** via
`--command`. That needs neither BAM nor any recorded data. What the recording
supplies is worth being clear about, because none of it is "measured data the
motor needs to run":

| From the log | Used for |
| --- | --- |
| `mass`, `arm_mass`, `length` | the **rig** - so the pendulum's inertia and gravity torque match the real testbench |
| `goal_position` | the **command sequence** replayed as the joint target |
| `position`/`speed` at step 0 | the **initial condition** |
| `kp`, `vin` | **firmware constants** - params files do not contain them |
| `position`, all steps | **comparison only** - the simulator never consumes these |

So the recorded positions are the answer key, not an input. The actuator's dynamics
come entirely from the params file (bundled, works with `--log none`) plus those
firmware constants.

When a recording *is* replayed, two comparisons are reported:

- **Isaac vs the offline harness** — the same model in PhysX vs in BAM's loop.
  A gap here means the *scene* differs (timestep, inertia, armature).
- **Isaac vs the recording** — the model against the physical servo. This is the
  one that answers "does it react like it's supposed to".

`--motor` takes the name BAM's params files write (`sts3215`, `md01`), not the
module filename; `--list-motors` prints the valid values. The params file's
`"actuator"` key overrides it, so `--params sts3215/m5` alone is enough.

The script also shows the external-torque hook in action: it computes
`(mass + arm_mass/2) * g * L * sin(q)` each step and pushes it in via
`set_external_torque()`, which is what makes the load-dependent (`m3`–`m6`) terms
do anything.

### Driving a model through BAM's identification trajectories

`bam_actuators/trajectory.py` is a port of `bam.trajectory` — same names, same
durations, same arithmetic, pinned sample-by-sample against BAM by
`tests/test_trajectory.py`. These are the motions the friction models were *fitted
from*, so running one is the closest thing to re-running the identification.

| `--command` | Motion | Torque |
| --- | --- | --- |
| `sin_time_square` | `sin(t²)` — BAM's recommended primary trajectory, sweeping the widest velocity range in a single 6 s run | on |
| `up_and_down` | slow cubic `0 → π/2 → 0.8·π/2` | on |
| `steps` | staircase `0 → 0.75 → 1.55 → 0.75 → 0` | on |
| `half_sine` | slow half-sine `0 → π/2` | on |
| `sin_sin` | multi-frequency `sin(t)·π/2 + sin(5t)·0.5·sin(2t)` | on |
| `lift_and_drop` | cubic to `−π/2` over 2 s, **then released** | off after 2 s |
| `nothing` | no torque — the pure gravity response | off |

The two unpowered entries are the interesting ones: `torque_enable = False` is
*not* the same as commanding zero torque. The servo contributes nothing, but
friction and gravity still act, so the arm back-drives and falls. That is a
distinct code path — `BamActuator.set_torque_enable()` — and it is precisely what
those two trajectories exist to measure.

`nothing` starts at 0 rad, which is the arm hanging straight down and also the
gravity equilibrium, so it genuinely does not move. Pass `--initial-angle 1.2` to
start it off-equilibrium if you want to watch it fall under gravity alone.

### Comparing models side by side

`--model1`/`--model2` put two models in the scene at once; `--models` takes any
number:

```bash
python scripts/pendulum_scene.py --visual --model1 m1 --model2 m6
python scripts/pendulum_scene.py --visual --models m1 m3 m5
```

A bare variant name is expanded against `--motor`, so `--model1 m1` means
`<motor>/m1.json`; a `motor/model` reference or a path also works. `--params` is
ignored (with a note) once any `--model*` is given.

Every model gets its own pendulum, colour-coded by variant — `m1` red, `m2`
orange, `m3` yellow, `m4` green, `m5` blue, `m6` violet — and printed as a legend at
startup. All of them receive the *same* command on the same physics step, so any
difference on screen is the friction model and nothing else.

Structurally the arms are the joints of **one** articulation rather than separate
assets. That is what makes per-model parameters possible: an articulation can carry
several actuator groups, each covering a different subset of joints, and each group
is its own `BamActuator` reading its own params file. Being a single articulation
also means the arms share a physics step and a `robot.update()`, which is what makes
the comparison fair to begin with. The report runs the offline harness once per
model, so you get a per-model agreement table rather than a single figure.

**Overlaying them (`--overlay`).** By default the arms are spread out so they read as
a row. `--overlay` instead puts every arm on the *same* pivot, so their arcs coincide
exactly and the only thing that can separate them is the friction model.

That is exact rather than approximate. The arms are offset only along **Y**, their
own rotation axis, and rotating about the Y line through `(0, y, 0)` is the same
rotation as about the Y line through the origin; the COM keeps its X and Z, so the
gravity torque is unchanged too. The `<inertial>` block is byte-identical between the
two layouts, which `tests/test_testbench.py` asserts.

Overlapping the arcs is safe because the rig emits **no collision geometry at all** —
no `<collision>` elements anywhere, plus `self_collision=False` and
`collision_from_visuals=False`. Nothing can touch anything, so two arms may disagree
wildly without the solver caring. There is a test for that too.

Because the arms are separated along the view axis in this mode, the camera is yawed
off it: looking straight down the rotation axis would hide every arm but the nearest.
When the models agree you see a single rod; the fan that opens up is the difference.

Worth knowing before you look: the six models do not diverge dramatically. On
`steps` with the STS3215 params, m1 and m2 land within 0.003 rad of each other, while
m3–m6 sit 0.04–0.08 rad from m1 — a couple of centimetres at the tip. BAM's
refinements are real but modest on this rig, so expect subtle differences rather than
dramatically different motion.

### Watching it move

A 6 s trajectory simulates in well under a second, so an unpaced run is over
before the viewport draws its first frame — the window then shows a pendulum that
never moves. Two things fix that, both on by default when `--visual` is passed:

- **Pacing.** The loop sleeps to hold `--speed` (default `1.0`, real time). `0.25`
is quarter speed, `0` runs as fast as possible.
- **Framing.** `sim.set_camera_view()` is pointed at the whole row of arms. They all
rotate about the same axis, so looking along it shows every swing face-on, and they
are spaced along the perpendicular direction so they read as a row instead of hiding
behind one another. Without this, 15 cm arms are specks near the world origin.

Rendering is decimated to about 60 wall-clock frames per second
(`SimulationCfg.render_interval`, scaling with `--speed` so slow motion stays
smooth) while the actuator still sees the true `dt`. `--loop` replays the sequence
so the motion stays on screen for as long as you want; without it the window is
left open on the final pose.

Three things to get right when wiring the actuator into a task of your own:

- `physics_dt` must equal your task's `env.sim.dt`. The actuator never receives a
  timestep from Isaac Lab, and a mismatch silently distorts dynamics.
- Set `effort_limit` explicitly, or PhysX's USD joint limit does your clipping.
- Start with `m1` or `m2`, not `m5`. Load-dependent models need
  `set_external_torque()`; until an env calls it those terms are inert.
  `scripts/pendulum_scene.py` and `tests/offline_rollout.py` both show how.


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
  this model, so leaving them at their USD defaults double-counts friction. The
  scripts set all three to `0.0` explicitly.
- **Don't shadow an `ActuatorBase` attribute.** `ActuatorBase` already defines
  `friction`, `dynamic_friction`, `viscous_friction`, `armature`, `stiffness` and
  `damping` as tensors. Ours is called `friction_model` for that reason: naming it
  `friction` makes `Articulation` init fail with
  `TypeError: can't assign a BamFrictionModel to a torch.cuda.FloatTensor`, from
  `write_joint_friction_coefficient_to_sim(actuator.friction, ...)`.
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
| `test_trajectory.py` | the trajectory port vs `bam.trajectory`, sample-by-sample, plus the `torque_enable` boundaries |
| `test_testbench.py` | the rig geometry - the physics is read back out of the generated URDF and checked against `bam.testbench.Pendulum` |
| `test_sts3215_m5.py` | the six bundled models, end to end |
| `test_offline_rollout.py` | **full composition**: BAM's pendulum loop driven by our motor + friction, step-by-step against `bam.simulate.Simulator` on a recorded log |

`tests/offline_rollout.py` is the harness behind that last one, and is also a
usable reference for the external-torque hook: it feeds the testbench's gravity
torque into `BamFrictionModel` exactly as an Isaac env must via
`set_external_torque()` for the load-dependent terms to do anything.

The parity tests load BAM from its source checkout (`bam` cannot be pip-installed
here: its `requires-python` is `>=3.12`) and skip cleanly when it is absent.

