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
- archive insertions, restorations, pruning, and restoration reasons;
- delayed visual-return detections, loop length, prior visit, every credited
  decision and scene/action/duration choice, plus unavailable recoveries;
- autonomous-motion detections and neutral grace waits across short pauses;
- temporal-option starts, every passive continuation, controllable endpoints,
  credited initiating state/action/duration choices, same-duration
  counterfactual counts and pixel contrast, novelty/scene-span/duration/return
  score components, returns to the source or another previously known
  behavioral endpoint, exact-choice and action-level learned values, and traces
  discarded at save-state jumps or run boundaries;
- delayed temporal-counterfactual reservation, every matched neutral step with
  source and target state aliases, factual-versus-counterfactual pixel
  contrast, endpoint contrast, and explicit state-release reasons;
- persistent-frontier successor-novelty updates, completed samples, loop
  penalties, provisional traces discarded at save-state jumps, and the value
  used to rank each archived branch, including state/action/duration samples;
- frozen-encoder abstraction assignments, latent distance, cluster creation,
  running cluster size, and the abstract signature used by each decision;
- interaction-derived behavioral-cluster assignments, matched controller
  probes, per-probe successor-latent distances, provisional-state deferrals,
  frontier-signature migrations, active-probe selection reasons, and prior
  probe-observation counts;
- committed decisions, temporary action counts, scene streaks, archive size,
  exact-visual stagnation streaks, and restored-branch status.

Frame pixels are stored once under their digest, even if thousands of events
refer to the same screen. Save-state bytes and native state tokens are never
serialized. State lifecycle records use run-local aliases such as
`state-00000042`.

## Evaluator bootstrap boundary

`lolo-neural-run --bootstrap lolo1-first-room` runs a minimal deterministic
controller macro before attempt 1. The fixture is accepted only for its known
ROM SHA-256 and only when its endpoint exactly matches the expected first-room
frame and coarse visual signature. `--bootstrap none` is the default, preserving
strict power-on evaluation.

The event stream records `bootstrap_started`, every `env_step` with
`phase=bootstrap`, `bootstrap_action_committed`, and `bootstrap_completed` under
attempt 0. `attempt_started` and `env_attached` then mark the pixel-exact handoff
to the agent as attempt 1. Summary fields report the fixture, action counts,
durations, and total emulator frames separately; bootstrap actions are excluded
from `investigated_actions`, `investigated_durations`, and per-attempt agent
statistics. The transition graph retains them with their phase-bearing source
events so a visualization can show or filter the complete session.

Both replay modes include the power-on-to-room bootstrap. The committed player
then follows only choices made by the agent, while the full player additionally
shows its rejected save-state branches.

`decisions.csv` is the convenient per-decision view. `transitions.json` contains
nodes for visual states and counted directed edges for all investigated
controller transitions. `summary.json` contains total and per-attempt counts,
committed versus investigated action distributions, archive restores, delayed
returns and recoveries, branches, states, unique frames, and unique coarse
scenes. The decision CSV exposes both the restore reason and whether a delayed
return recovery was pending after each committed emulator action. It also
records the successor-novelty reward and learned persistent-frontier value.
It also exposes the learned temporal-option value used for each committed
choice, whether that estimate came from the exact choice, an action-level
prior, or no evidence, and whether a passive option trace remained active after
the decision. Summary counts distinguish started, completed, discarded, and
credited option samples.

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
