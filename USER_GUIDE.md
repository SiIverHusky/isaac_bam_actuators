# User guide

How to drive the two scripts in `scripts/`. For what the extension *is* and how it is
built, see [`README.md`](README.md); for the maths, see the docstrings in
`bam_actuators/`.

Both scripts are **standalone Isaac Sim applications**. They need an NVIDIA RTX GPU and
a Python environment with Isaac Sim 5.x and Isaac Lab installed — there is no CPU mode.
They also need the extension itself importable:

```bash
conda activate <your-isaaclab-env>
python -m pip install -e /path/to/isaac_bam_actuators
```

The offline test suite is the exception: it is pure torch, so it runs anywhere, with or
without a GPU.

```bash
BAM_ROOT=/path/to/BAM python -m pytest tests/ -q    # 135 passed, 4 skipped
python -m pytest tests/ -q                          # 92 passed, 47 skipped
```

The skips are always missing *data*, never missing code: without a BAM checkout the
parity tests skip, and with one the four STS3215 rollout cases skip if their Feetech
recording is absent. See [Troubleshooting](#troubleshooting).

---

## Which script do I want?

| | `check_isaac_actuator.py` | `pendulum_scene.py` |
| --- | --- | --- |
| Answers | "Is the extension wired into Isaac Lab correctly?" | "How does this model actually behave?" |
| Builds | one actuator, no scene, no asset | a full pendulum rig |
| Takes | a few seconds | a few seconds, plus playback |
| Run it when | anything in `bam_actuators/actuators/` changed, or a task of yours fails during `Articulation` init | you want to see, or compare, dynamics |

If a task of your own blows up deep inside `Articulation` initialisation, run
`check_isaac_actuator.py` **first**. It exercises the same Isaac-facing code path with
nothing else in the way, which separates "the extension is broken" from "my scene is
wrong".

---

## 1. `check_isaac_actuator.py` — the smoke test

Instantiates `BamActuator` directly, with no scene, no articulation and no asset. That is
enough to cover the entire Isaac-facing surface, which is exactly the part the offline
tests cannot reach.

### Running it

```bash
python scripts/check_isaac_actuator.py            # headless
python scripts/check_isaac_actuator.py --visual   # with a viewport
```

Exit code `0` on success, `1` on the first failure (with a full traceback).

### What it prints

```
--- md01 (voltage law, module defaults) ---
<BamActuator ...>                                  <- ActuatorBase.__str__
  motor=md01  friction model=None terms=[]
    step: applied_effort=+0.049840 Nm (computed=+0.099840)
    step: applied_effort=+0.049840 Nm (computed=+0.099840)
    step: applied_effort=+0.049840 Nm (computed=+0.099840)
  effort_limit respected (peak 0.150000 <= 0.15)

--- md01i/m3 (bundled params file) ---
  motor=md01i  friction model=m3 terms=['load_dependent']
    ...

--- md01c (loop law, module defaults) ---
  motor=md01c  friction model=None terms=[]
    ...

--- sts3215/m5 (bundled params file) ---
  motor=sts3215  friction model=m5 terms=['load_dependent', 'directional', 'stribeck']
    step: applied_effort=+0.150000 Nm (computed=+0.537778)
    step: applied_effort=+0.150000 Nm (computed=+1.075557)
    step: applied_effort=+0.150000 Nm (computed=+1.613335)
  effort_limit respected (peak 0.150000 <= 0.15)

OK: BamActuator works inside Isaac Lab.
```

*(the `...` rows print theirs too; the numbers above come from the model, not a simulator)*

Reading it:

- **`friction model=… terms=…`** — which variant the params file selected. `None` with
  empty `terms` means no params file, so only the base variant applies.
- **the three `step:` lines** — `compute()` handed back finite efforts of the right shape.
  `md01` is stateless, so its three `computed` values are identical, while the STS3215's
  climb (0.54 → 1.08 → 1.61 Nm) as its slew-limited internal target travels towards the
  command. That growth is exactly why there are three steps rather than one.
- **`effort_limit respected`** — a deliberately huge command is still clipped. The
  limit is deliberately low (0.15 Nm) so the clip actually binds in every case.

### Why those cases

1. **`md01`** — the voltage law with no params file: the module's own seeds are the
   model, the way BAM's `MD01Actuator.initialize()` seeds it. (No `md01` fits are
   bundled; the campaign-2 fits are the `md01i` ones below.)
2. **`md01i/m3`** — a bundled identified model of the *new* campaign. The params file
   alone decides the motor (`"actuator": "md01i"`), the friction maths (`"model":
   "m3"`) and every identified value, including the current limit that no cfg field
   carries.
3. **`md01c`** — the measured AT32 loops, whose firmware constants (`kp_current`,
   `kff_current`, `cap_ma`) are module defaults rather than params-file values.
4. **`sts3215/m5`** — a stateful control law, so stepping the same instance three
   times is what exercises the slew limiter.

### What a failure means

Almost always a change in `bam_actuators/actuators/bam_actuator.py` broke the
`ActuatorBase` contract. The offline suite cannot catch that, which is the whole reason
this script exists. See [Troubleshooting](#troubleshooting).

---

## 2. `pendulum_scene.py` — the pendulum rig

### The 60-second tour

```bash
python scripts/pendulum_scene.py --visual --loop
```

One arm, 15 cm long, hanging from a fixed pivot and driven through BAM's `steps`
trajectory on repeat. Close the window (or Ctrl+C) to exit.

### What you are looking at

A single revolute joint whose rigid-body dynamics are **analytically identical** to
`bam.testbench.Pendulum`: the URDF's `<inertial>` block is derived, not tuned, so the
swing inertia and gravity torque match BAM exactly. The only remaining difference from a
BAM simulation is the integrator.

Two consequences worth knowing:

- **There is no ground plane.** The pivot sits at the origin and the arm hangs below it,
  so a floor would intersect it and introduce contacts BAM's model does not have.
- **The rig carries no collision geometry at all.** Nothing can touch anything, which is
  what makes `--overlay` safe.

`python scripts/pendulum_scene.py --list-motors` prints valid `--motor` values and exits
without starting Isaac Sim, so it works even when the environment is misconfigured.

### Cookbook

```bash
# Watch one model move, in slow motion, forever
python scripts/pendulum_scene.py --visual --loop --command sin_time_square --speed 0.25

# Compare two models side by side, spread out
python scripts/pendulum_scene.py --visual --model1 m1 --model2 m6

# Compare with the arcs exactly coincident - the closest possible comparison
python scripts/pendulum_scene.py --visual --models m1 m6 --overlay

# All six at once
python scripts/pendulum_scene.py --visual --models m1 m2 m3 m4 m5 m6 --overlay

# A pure gravity drop: unpowered from 1.2 rad, no params file needed
python scripts/pendulum_scene.py --command nothing --initial-angle 1.2 --visual

# Lift to -pi/2, then release the motor at 2 s and let it fall
python scripts/pendulum_scene.py --command lift_and_drop --visual --speed 0.5

# The new MD01 campaign: the current law, on a real MD01 recording
BAM_ROOT=/path/to/BAM python scripts/pendulum_scene.py --motor md01i --params md01i/m3 \
    --log $BAM_ROOT/data_md01-6v2/2026-09-21_18h22m26.json

# Headless, no BAM checkout required, write the curves out
python scripts/pendulum_scene.py --log none --command steps --csv /tmp/traj.csv

# Replay a real recording: the model against the physical servo
BAM_ROOT=/path/to/BAM_PROJECT python scripts/pendulum_scene.py --log /path/to/recording.json --params sts3215/m5
```

`--command` and the `--model1`/`--model2`/`--models` family are the two independent
axes: `--command` picks *what motion*, the model flags pick *which models*. Neither needs
BAM or a recording.

### Flags

**What to simulate**

| Flag | Default | Meaning |
| --- | --- | --- |
| `--command NAME` | `steps` | BAM identification trajectory to drive the rig with. See the table below. |
| `--model1 NAME`, `--model2 NAME` | none | Add a pendulum driven by that model, e.g. `--model1 m1 --model2 m6`. |
| `--models NAME [NAME ...]` | none | General form of the above; any number. |
| `--params FILE` | `sts3215/m5` | Params file for the single-pendulum case, or `none`. Ignored when `--model1`/`--model2`/`--models` is used. |
| `--motor NAME` | `sts3215` | Servo family. A params file's `"actuator"` key overrides it. |
| `--steps N` | whole trajectory | Simulate only the first `N` steps. |
| `--initial-angle RAD` | `0.0` | Starting angle for synthetic runs. `0` is hanging straight down, which is also the equilibrium — so `--command nothing` at `0` does not move at all. |
| `--log FILE` | `$BAM_ROOT/data_raw/…` | Recording to replay, or `none`. Gives the rig, the command sequence, the initial state and the firmware constants. |

**The rig** (used when there is no recording)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mass KG` | `0.5` | Tip mass. |
| `--arm-mass KG` | `0.02` | Arm mass. |
| `--length M` | `0.15` | Arm length. |
| `--dt S` | `0.005` | Timestep. Becomes the actuator's `physics_dt`, which the slew limiter depends on. |

The defaults are the values every recorded log carries, so they *are* BAM's testbench.

**Playback**

| Flag | Default | Meaning |
| --- | --- | --- |
| `--visual` | off | Show the viewport. Without it the run is headless. |
| `--speed X` | `1.0` | Wall-clock playback speed: `1.0` real time, `0.25` quarter speed, `0` as fast as possible. |
| `--loop` | off | Keep replaying after the report instead of exiting. |
| `--overlay` | off | Coincide all pivots so the arcs overlap exactly. See below. |

**Output**

| Flag | Default | Meaning |
| --- | --- | --- |
| `--csv FILE` | off | Write goal, torque-enable and every model's Isaac/offline position per step. |

Both scripts also accept Isaac Sim's own launcher options (`--device cuda:0`, …). Run with
`-h` to see the list for your installed version.

### The trajectories

All of BAM's identification motions, ported from `bam.trajectory` and pinned against it
sample-by-sample. Every one runs for 6 s = 1200 steps at the default `dt`.

| `--command` | Motion | Torque |
| --- | --- | --- |
| `sin_time_square` | `sin(t²)` — sweeps the widest velocity range in one run; BAM's recommended primary trajectory | on |
| `up_and_down` | slow cubic `0 → π/2 → 0.8·π/2` | on |
| `steps` | staircase `0 → 0.75 → 1.55 → 0.75 → 0` | on |
| `half_sine` | slow half-sine `0 → π/2` | on |
| `sin_sin` | multi-frequency `sin(t)·π/2 + sin(5t)·0.5·sin(2t)` | on |
| `lift_and_drop` | cubic to `−π/2` over 2 s, **then released** | off for the last 800 steps |
| `nothing` | no torque — the pure gravity response | off for all 1200 steps |

**"Torque off" is not "command zero torque".** The servo contributes nothing, but friction
and gravity still act, so the arm back-drives and falls. That is a distinct code path
(`BamActuator.set_torque_enable()`) and it is what those two trajectories exist to measure.
`--initial-angle` is what makes `nothing` observable, since its target is always 0.

Note that `steps` gives each breakpoint to the **previous** stage: at `t = 1.0 s` the
target is still `0.0`, not `0.75`. That is what BAM does, and the port matches it exactly.

### Comparing models

Every model gets its own pendulum, colour-coded by variant, and all of them receive the
**same command on the same physics step** — so any difference you see is the friction
model and nothing else.

| Model | Colour |
| --- | --- |
| `m1` | <kbd>#e63333</kbd> red |
| `m2` | <kbd>#f28c1a</kbd> orange |
| `m3` | <kbd>#d9cc26</kbd> yellow |
| `m4` | <kbd>#40bf4c</kbd> green |
| `m5` | <kbd>#338cf2</kbd> blue |
| `m6` | <kbd>#a659d9</kbd> violet |
| anything else | <kbd>#bfbfc7</kbd> grey |

The colours are printed as a legend on startup, so you do not have to remember them.

**`--overlay`** puts every arm on the *same* pivot, so their arcs coincide exactly. This is
exact, not approximate: the arms are offset only along **Y**, their own rotation axis, and
rotating about the Y line through `(0, y, 0)` is the same rotation as about the Y line
through the origin. The COM keeps its X and Z, so the gravity torque is unchanged and the
`<inertial>` block is byte-identical between layouts. Overlapping the arcs is safe because
the rig has no collision geometry.

Without `--overlay` the arms are spread along X so they read as a row. With it they stack
along the view axis, so the camera yaws off it — otherwise all but the nearest arm would
hide. Either way, when the models agree you see a single rod; the fan that opens up is the
difference.

### Reading the report

```
[scene] source    : BAM trajectory 'steps' (6s)  (1200 steps, dt=0.005000s)
[scene] testbench : mass=0.5 arm_mass=0.02 length=0.15  start=+0.000 rad
[scene] motor     : sts3215
[scene] firmware  : (motor module defaults)
[scene] pendulums : 2
    m1         #e63333  params=/…/bam_actuators/params/sts3215/m1.json
    m6         #a659d9  params=/…/bam_actuators/params/sts3215/m6.json
[scene] layout    : spread along X
[scene] playback  : speed=1x, rendering every 3 physics step(s)

[scene] 1200 steps simulated, 2 pendulum(s)
  m1       motor=sts3215  final isaac=+0.0011  offline=+0.0011
           Isaac vs offline harness: max|dq|=0.000123  rms=0.000045 rad
  m6       motor=sts3215  final isaac=+0.0012  offline=+0.0012
           Isaac vs offline harness: max|dq|=0.000456  rms=0.000078 rad
```

*(positions illustrative; the shape is real)*

The header line is worth reading properly — `source`, `testbench`, `motor`, `firmware`,
`pendulums` and `layout` together are the complete description of what was simulated. If
the numbers look wrong, check those first.

The per-model block compares the **same model** simulated two ways:

- **Isaac** — PhysX, through `BamActuator`.
- **offline harness** — BAM's own loop, driven by the same motor and friction code.

So a gap here is *integrator and solver*, not model. Small and non-zero is expected: PhysX
is implicit, BAM's loop is semi-implicit Euler. A **large** gap means the scene does not
match BAM's testbench — a timestep, inertia or armature difference — and that is the first
thing to investigate.

When you replay a recording with `--log`, a second line appears per model:

```
           Isaac vs recording:       max|dq|=0.041  rms=0.012 rad
```

That one compares the model against the **physical servo**. It is the number that answers
"does it react like it's supposed to", and with several models side by side it tells you
which model fits that particular servo best.

### How different are the models, really?

The six models genuinely differ, but not dramatically. On `steps` with the STS3215 params:

| | divergence from m1 |
| --- | --- |
| m2 | ~0.003 rad |
| m3–m6 | 0.04–0.08 rad |

That is a couple of centimetres at the tip. Read a subtle difference as data, not as a bug.

### The CSV

`--csv` writes one column per quantity per step: `goal`, `torque_enable`, and then
`<model>_isaac_pos`, `<model>_isaac_vel`, `<model>_offline_pos` for each model — plus
`recorded_pos` when you replayed a recording. One file, every model, ready to plot.

---

## Workflows

### "Which model fits my servo best?"

1. Record a log with BAM's testbench.
2. Replay it against every model:
   ```bash
   python scripts/pendulum_scene.py --log /path/to/recording.json --models m1 m2 m3 m4 m5 m6
   ```
3. Read the `Isaac vs recording` lines. Lowest `rms` wins *for that servo, on that motion*.
4. Watch the winner: `--visual --overlay` on the top two.

Note the whole `--model1`/`--model2`/`--models` family ignores `--params`, and each
model is resolved against `--motor`, so `--model1 m3` means `<motor>/m3.json`. For the
MD01 campaign, add `--motor md01i` so the variants resolve to `md01i/*.json`:

### "I added a motor and want to check it"

1. Offline: `BAM_ROOT=… python -m pytest tests/ -q`
2. Isaac wiring: `python scripts/check_isaac_actuator.py`
3. In a rig: `python scripts/pendulum_scene.py --motor <name> --params <motor>/m1 --visual`

### "The model diverges from the recording and I want to know why"

Two comparisons, two different diagnoses:

- **Isaac vs offline harness is large** → the *scene* is wrong. Check that `--dt` matches
  your rig, and that `mass`/`arm_mass`/`length` are the real ones. BAM's params were
  identified with the armature included, so a double-counted armature shows up here.
- **Isaac vs recording is large, but the two harnesses agree** → the *model* disagrees with
  the physical servo. That is the honest measurement, and it is what BAM's fitting targets.
  It is not something the extension can fix.

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `ModuleNotFoundError: bam` | Only needed for `--log`. Either set `BAM_ROOT`, or pass `--log none`. |
| `no recording at …` then `falling back to a synthetic run` | `BAM_ROOT` is unset or points elsewhere. Harmless; pass `--log none` to silence it. |
| `FileNotFoundError: No bundled params for …` | The model name is not bundled. `available_bundled()` in the traceback lists what is; `--list-motors` lists motors. |
| `No bundled params for 'md01/m3'` but the JSON exists | The directory is named after the **actuator**, not the product. The campaign-2 MD01 fits were identified with the current law, so they are `md01i/m3`, not `md01/m3`. |
| `TypeError: can't assign a BamFrictionModel to a torch…FloatTensor` | An attribute on `BamActuator` is shadowing one of `ActuatorBase`'s (`friction`, `armature`, `stiffness`, …). Ours is called `friction_model` for exactly this reason. |
| `PDGainsCfg.stiffness is MISSING` | The URDF spawner needs explicit drive gains. The scene passes `stiffness=0.0, damping=0.0`. |
| The window opens and closes instantly | You are on a `main()` that returned before entering the render loop. Use `--loop`, or drop `--visual` and read the report. |
| The pendulum never moves | Either nothing was paced (use `--speed 1`), or the trajectory is `nothing` at `--initial-angle 0`, which is the equilibrium. |
| Nothing is visible at all | Arms are 15 cm at the world origin; the camera is framed automatically, but check `--visual` was actually passed. |
| Arms all look the same colour | The per-arm material bind failed — a warning will say so. The URDF's own `<material>` is the fallback. |
| `available motors:` but Isaac Sim never starts | That is `--list-motors` working as intended: it exits before the Isaac Sim import. |

---

## Where to go next

- [`README.md`](README.md) — the extension itself, its layout, and how to add a motor.
- `bam_actuators/testbench.py` — how the URDF is derived from BAM's testbench, and why.
- `bam_actuators/trajectory.py` — the identification motions.
- `tests/` — the offline parity suite, and `offline_rollout.py`, which is a usable
  reference for how `set_external_torque()` must be driven.
