# Lolo Agent

This repository is a research harness for an agent that learns puzzle-game
mechanics from pixels, controller actions, and their consequences. The agent is
not given object labels, rules, demonstrations, solutions, or ROM memory.

The initial milestone is a small but complete experimental loop:

- a pixel-only emulator contract;
- controller actions and save-state branching;
- a persistent learned transition model;
- temporary episodic memory and count-based visual novelty;
- receding-horizon planning over real and predicted consequences;
- explicit training and frozen-evaluation modes;
- a deterministic, evaluator-hidden puzzle environment for tests.

It is infrastructure and a baseline, not yet a claim that all 50 rooms are
solved. The model boundary is designed so a neural latent world model can
replace the baseline without changing the information available to the agent.

## Run

The package has no third-party runtime dependencies:

```bash
python3 -m unittest discover -s tests -v
python3 -m lolo_agent --steps 40 --depth 2
```

## Local NES integration

Keep legally obtained ROM files outside version control. With an ARM64 libretro
NES core installed, verify raw framebuffer capture and save-state determinism:

```bash
source .venv/bin/activate
lolo-emulator-smoke \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --output /tmp/lolo-frame.ppm
```

`LibretroEnv` binds video, controller, lifecycle, and serialization functions
only. It does not bind the libretro memory-inspection API. The current Python
loader remains an integration baseline; use the native host below for research
runs so save states remain server-side behind opaque handles.

Build and use the isolated native host:

```bash
make -C native
```

`NativeLibretroEnv` runs that host as a separate process. ROM data and serialized
state buffers remain inside the native process; Python receives RGB frames and
session-scoped opaque state capabilities only.

Train the first convolutional visual dynamics smoke model on MPS:

```bash
lolo-train-smoke \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --checkpoint checkpoints/smoke.pt
```

The collector tries alternative controller actions from native save-state
handles. The model learns an image encoder, action-conditioned latent dynamics,
and a pixel decoder. This is a training-pipeline baseline rather than the final
object-centric architecture.

Train and validate the multi-step uncertainty ensemble, then exercise its
verified neural rollout planner:

```bash
lolo-ensemble-smoke \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --checkpoint checkpoints/ensemble-smoke.pt
```

Train the gated unlabeled spatial-token successor from persisted strict branch
experience:

```bash
lolo-spatial-train \
  --dataset experiments/lolo1-medium/dataset \
  --checkpoint experiments/lolo1-spatial-smoke/checkpoints/spatial.pt \
  --reward-track strict \
  --validation-split run \
  --max-groups 250 \
  --minimum-multistep-groups 250 \
  --grid-size 16 \
  --renderer flow_residual \
  --renderer-rollout anchored \
  --changed-region-loss-weight 1
```

This model predicts spatial effects and uncertainty without object labels. Its
metrics distinguish the effect-learning gate from the stricter planner-
integration gate. A spatial checkpoint can also be measured against live
save-state branches with `--spatial-shadow-checkpoint`; selection weight is
zero by default. A promoted checkpoint can be tested as an explicit, logged
branch-verification-priority ablation with `--spatial-selection-weight`. The
score compares each predicted action with a duration-matched predicted NOOP;
real verified outcomes alone decide which branch is committed. Zero remains
the safe default.

Train the observational returnability sidecar from pixel-state transition
graphs while keeping the spatial world model frozen:

```bash
lolo-spatial-returnability-train \
  --dataset experiments/lolo1-medium/dataset \
  --spatial-checkpoint experiments/lolo1-spatial-v10/checkpoints/spatial-v10-native-adapt-e5.pt \
  --checkpoint experiments/lolo1-spatial-v12/checkpoints/spatial-v12-returnability-grid4-e5.pt \
  --reward-track strict \
  --maximum-return-steps 3 \
  --minimum-endpoint-actions 5 \
  --spatial-bins 4
```

Positive targets require an observed visual-state return path. A transition is
negative only after its endpoint has been probed with at least five distinct
controls; inconclusive transitions remain unlabeled. This sidecar currently
fails the native generalization gate and must remain telemetry-only.

Collect explicit short-horizon return evidence from verified save-state
branches without changing the policy:

```bash
lolo-neural-run \
  ... \
  --returnability-probe-depth 2 \
  --returnability-probe-beam-width 4 \
  --returnability-probe-pixel-l1-threshold 0.002
```

Each probe restores the branch endpoint, tests every configured controller
action, and compares the result with a NOOP branch advanced for the same total
number of emulator frames. Probe actions are logged under a separate phase,
excluded from normal experience import, and cannot affect planning. Results are
flattened to `returnability_probes.csv`.

Train an observational relation head only after assigning complete strict runs
to disjoint training and validation partitions:

```bash
lolo-spatial-probe-returnability-train \
  --training-run runs/strict-train-a \
  --training-run runs/strict-train-b \
  --validation-run runs/strict-heldout \
  --spatial-checkpoint experiments/lolo1-spatial/checkpoints/spatial.pt \
  --checkpoint experiments/lolo1-spatial/checkpoints/probe-returnability.pt
```

The importer verifies probe/branch lifecycles and stored pixel digests, rejects
reward-track or probe-configuration mixing, removes exact transition overlap
and training-source pixel overlap from validation, and requires both labels in
each partition. Training is class balanced; held-out evaluation preserves its
natural prevalence. Probe-trained checkpoints remain telemetry-only until they
pass broader native calibration.

Run the saved model in frozen evaluation mode:

```bash
lolo-neural-run \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --checkpoint checkpoints/ensemble-smoke.pt \
  --spatial-shadow-checkpoint experiments/lolo1-spatial-v10/checkpoints/spatial-v10-native-adapt-e5.pt \
  --spatial-returnability-checkpoint experiments/lolo1-spatial-v12/checkpoints/spatial-v12-returnability-grid4-e5.pt \
  --spatial-selection-weight 0 \
  --decisions 20 \
  --log-root runs
```

Strict evaluation starts from NES power-on and makes the agent discover every
input. For puzzle-learning experiments, the evaluator can instead perform the
known title/story transition and hand control to the frozen agent at the exact
first-room pixels:

```bash
lolo-neural-run \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --checkpoint checkpoints/ensemble-smoke.pt \
  --bootstrap lolo1-first-room \
  --decisions 20 \
  --log-root runs
```

This is evaluator initialization, not an agent skill or training example. The
fixture is ROM-hash and final-pixel checked, remains disabled by default, and
its inputs are tagged `phase=bootstrap` and excluded from agent action and
attempt statistics.

Every neural run now creates a self-contained telemetry directory. It contains
an append-only JSONL event stream, deduplicated PNG observations, a manifest
with input hashes and frozen-model audit data, a decision CSV, and an aggregated
transition graph. See [docs/telemetry.md](docs/telemetry.md) for the schema and
evaluator-only level annotations.

Turn any completed telemetry run into high-speed, frame-accurate browser
playback. `committed` shows the chosen gameplay trajectory; `full` also shows
every rejected planning branch and save-state jump:

```bash
lolo-replay \
  --run runs/<run-id> \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --mode both \
  --speed 120
```

Open `runs/<run-id>/replays/committed/index.html` or
`runs/<run-id>/replays/full/index.html`. Players support scrubbing, stepping,
keyboard control, and playback from 5 to 240 frames per second.

A later local session can continue from a decision the agent reached itself by
passing `--resume-run runs/<parent-run> --resume-decision <n>` to
`lolo-neural-run`. The child manifest hashes that parent telemetry, replay
reconstructs both logs as one provenance chain, and gameplay resumes exclude
the title-screen `START`/`SELECT` controls.

## Durable learning experiments

Run repeated collection, MPS training, and frozen evaluation with explicit
variable-duration controller actions:

```bash
lolo-experiment \
  --experiment-dir experiments/lolo1-main \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --cycles 10 \
  --durations 1,2,4,8,16
```

Running the same command again completes ten additional cycles. Atomic dataset
segments, per-cycle checkpoints, phase state, metrics, collection telemetry,
and frozen-evaluation telemetry are retained below the experiment directory.
See [docs/experiments.md](docs/experiments.md).

The first 10-cycle duration-aware audit and its observed failure modes are
documented in
[docs/medium-experiment-2026-08-08.md](docs/medium-experiment-2026-08-08.md).

To connect an emulator, implement `PixelSaveStateEnv` from
`lolo_agent/environment.py`. Keep success, room number, sprite identity, and RAM
outside that interface. An evaluation harness may inspect success to score a
run, but must never return it to the agent.

See [docs/protocol.md](docs/protocol.md) for the split and freeze protocol and
[docs/architecture.md](docs/architecture.md) for the component boundaries.
