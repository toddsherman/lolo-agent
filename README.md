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

Run the saved model in frozen evaluation mode:

```bash
lolo-neural-run \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --checkpoint checkpoints/ensemble-smoke.pt \
  --decisions 20 \
  --log-root runs
```

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

To connect an emulator, implement `PixelSaveStateEnv` from
`lolo_agent/environment.py`. Keep success, room number, sprite identity, and RAM
outside that interface. An evaluation harness may inspect success to score a
run, but must never return it to the agent.

See [docs/protocol.md](docs/protocol.md) for the split and freeze protocol and
[docs/architecture.md](docs/architecture.md) for the component boundaries.
