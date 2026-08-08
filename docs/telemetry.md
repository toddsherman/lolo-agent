# Run telemetry

Each `lolo-neural-run` writes a new directory below `runs/` (ignored by Git):

```text
runs/run-<UTC timestamp>/
├── manifest.json
├── events.jsonl
├── frames/<frame SHA-256>.png
├── decisions.csv
├── transitions.json
└── summary.json
```

The event log is the source of truth. Derived CSV, graph, and summary files can
be rebuilt at any time:

```bash
lolo-log summarize --run runs/run-20260808T120000.000000Z
```

## Captured events

Telemetry includes:

- run, emulator, and frozen-parameter audit lifecycle events;
- environment resets and attempt numbers;
- every real controller action, duration, source frame, target frame, and pixel
  change, including actions explored on rejected save-state branches;
- anonymous save, load, and release events for every opaque state capability;
- all final latent planner candidates and their model scores and uncertainty;
- each real verified branch with a stable branch ID, candidate rank, action,
  plan, novelty, prediction error, visual change, penalty, total score, and the
  exact `env_step` event that produced it;
- archive insertions, restorations, and pruning;
- committed decisions, temporary action counts, scene streaks, archive size,
  and restored-branch status.

Frame pixels are stored once under their digest, even if thousands of events
refer to the same screen. Save-state bytes and native state tokens are never
serialized. State lifecycle records use run-local aliases such as
`state-00000042`.

`decisions.csv` is the convenient per-decision view. `transitions.json` contains
nodes for visual states and counted directed edges for all investigated
controller transitions. `summary.json` contains total and per-attempt counts,
committed versus investigated action distributions, archive restores, branches,
states, unique frames, and unique coarse scenes.

## Attempts and level labels

An attempt begins whenever the environment is reset. Because room number,
completion, death, and object identity are deliberately excluded from the agent
interface, the logger does not invent semantic room boundaries. An evaluator
can add labels after the run without exposing them to the agent:

```bash
lolo-log annotate-level \
  --run runs/run-20260808T120000.000000Z \
  --label lolo1-withheld-room-07 \
  --start-seq 500 \
  --end-seq 1900 \
  --attempt 1

lolo-log summarize --run runs/run-20260808T120000.000000Z
```

These labels live in `evaluator_annotations.jsonl`, separate from the immutable
agent event stream. Rebuilding the summary adds the matching label to each
decision row. This separation lets later visualizations group attempts and
states by a human or evaluator-known room while preserving the pixel-only
research boundary during play.

Use `--no-frame-images` for digest-only profiling runs. The default keeps PNGs
because they make state timelines and transition-graph inspection much easier.

## Deterministic high-speed playback

The event stream contains enough information to reconstruct the entire native
emulator session, including anonymous save, load, release, and controller-step
operations. Render both replay views with:

```bash
lolo-replay \
  --run runs/<run-id> \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --mode both \
  --speed 120
```

The renderer verifies the ROM, core, and host binaries against the run manifest,
recreates state handles in event order, and expands every multi-frame controller
press into individual NES frames. It checks every reconstructed endpoint
against the original telemetry and fails on the first divergence.

Two standalone local players are generated:

- `replays/committed/index.html` follows only actions ultimately chosen by the
  agent, with explicit frames for resets and archive-restoration jumps;
- `replays/full/index.html` shows every planning branch, state restoration,
  action frame, and committed-decision marker in chronological order.

Both players are scrub-capable and offer 5, 15, 30, 60, 120, and 240 fps. Once
rendered, playback requires only the HTML and its content-addressed PNG folder;
the emulator and ROM are not read by the browser. `replay_manifest.json` records
the verification result and exact input hashes.
