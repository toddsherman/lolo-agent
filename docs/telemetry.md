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
  exact `env_step` event that produced it, plus matched-neutral action-effect
  contrast, learned estimate, sample count, and planning bonus;
- archive insertions, restorations, pruning, and restoration reasons;
- delayed visual-return detections, loop length, prior visit, every credited
  decision and scene/action/duration choice, plus unavailable recoveries;
- autonomous-motion detections and neutral grace waits across short pauses;
- rejected autonomous-motion hypotheses when the behavioral state already has
  a learned controlling action;
- learned-hazard filtering at commit and archive restoration, plus rejected
  archive insertions and the exact temporal-option evidence responsible;
- temporal-option starts, every passive continuation, controllable endpoints,
  credited initiating state/action/duration choices, same-duration
  counterfactual counts and pixel contrast, novelty/scene-span/duration/return
  score components, returns to the source or another previously known
  behavioral endpoint, exact-choice and action-level learned values, whether a
  negative sample met the global action-hazard criteria, and traces
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
- committed decisions, temporary action, duration, and action-duration pair
  counts, scene streaks, archive size, exact-visual stagnation streaks, and
restored-branch status.

`summary.json` aggregates matched action-effect observations by controller
action, branches with a known effect estimate, hazard-filter events and choices,
archive hazard rejections, and negative samples that met the global hazard
criteria. The same fields remain available at branch/decision granularity in
`events.jsonl`; the most useful action-effect and coverage fields are also
materialized in `decisions.csv`.

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

Matched-neutral exploration adds the following raw events and decision fields:

- `matched_neutral_verified` identifies evaluator-generated `NOOP` endpoints
  paired by root state and duration;
- `causal_observation_wait` records the neutral observation committed before a
  new intervention;
- `causal_spatial_signature`, `causal_changed_pixels`,
  `causal_change_centroid`, `causal_spatial_novelty`, and
  `causal_spatial_bonus` describe action-dependent screen changes;
- `causal_cell_coverage`, `causal_cell_unvisited`, `causal_cell_count`, and
  `causal_cell_coverage_bonus` describe global attempt-level coverage of the
  coarse cells changed relative to matched `NOOP`; the bonus is disabled by
  default and enabled with `--causal-cell-coverage-weight`;
- `causal_spatial_archive_bonus` records the live rarity value used when an
  archived branch is selected;
- `causal_cell_coverage_archive_bonus` records the corresponding live global
  coverage value used to rank an archived branch;
- `persistent_change_evidence_updated` records coarse cells whose visual value
  stayed different from their learned pre-change modal value for the configured
  number of committed observations, or later ceased to do so; baselines adapt
  before activation so moving sprites do not permanently imprint the first
  frame; `--persistent-change-minimum-value-drop` can restrict evidence to
  persistent disappearance of visually salient content without naming it;
- `persistent_change_archives_filtered` records temporary preference for
  archived states that preserve all active persistent changes, while
  `persistent_change_preservation_unavailable` makes the fallback to older
  alternatives explicit; this rule-free mechanism is disabled by default and
  enabled with `--persistent-change-stability-decisions`;
- `archive_branch_rejected` distinguishes covered causal frontiers, exhausted
  causal outcomes, non-causal alternatives, and learned hazards;
- `archive_causal_outcome_added` records a retained persistent visual
  transition, while `causal_outcome_exhausted` is the rejection reason for an
  already-restored coarse-pixel and directional-pose key.

`summary.json` aggregates matched-neutral verifications, total and unique
causal-spatial observations, unique committed causal signatures, first-visited
causal cells, and global coverage score/bonus totals.
`decisions.csv` includes the corresponding per-commit fields so a visualization
can map where action-dependent changes occurred without adding semantic object
labels.

On the explicitly labelled assisted reward track, semantic archive search adds:

- `human_prior_graph_source_signature` and
  `human_prior_graph_target_signature`, stable keys containing the visible
  goal state, detected player tile, HUD life glyph, and reversible world
  context;
- `human_prior_world_source_context`,
  `human_prior_world_target_context`, and
  `human_prior_world_effect_signature` on verified, archived, restored, and
  committed transitions;
- `human_prior_world_effect_confirmation`, which records the candidate coarse
  effect, immediate-control acceptance, future outcome spread, confirmation
  observations, action, duration, and endpoint frame;
- `human_prior_semantic_frontier_override`, which explains when a new assisted
  graph state is retained despite coarse causal-frontier deduplication; and
- `human_prior_best_first_archives_filtered`,
  `human_prior_best_first_frontier_exhausted`, and
  `human_prior_graph_stagnation_detected`, which make every semantic
  backtracking decision reconstructable;
- `human_prior_option_search_started`,
  `human_prior_option_branch_verified`, and
  `human_prior_option_search_completed`, which preserve every exact
  save-state sequence rollout, its action/duration path, parent and endpoint
  state IDs, pixel-derived goal analysis, novelty counts, score, and selection;
- `human_prior_option_search_deferred` and
  `human_prior_option_search_skipped`, which distinguish a cheaper unseen
  local archive endpoint from an already exhausted sequence-search source;
  and
- `human_prior_option_archive_added`, plus
  `human_prior_verified_option`, `human_prior_option_depth`, and
  `human_prior_option_path_visits_before` on restore/commit events, which make
  whole-sequence selection and replay explicit.

These fields are experimental assisted-track telemetry. They do not expose ROM
memory and are not present in the strict policy state. `summary.json` rolls up
confirmation/acceptance counts, unique effect signatures, committed world
contexts, graph states, player positions, semantic overrides, best-first
filter/exhaustion events, graph-stagnation events, sequence searches,
deferrals, cached skips, verified option branches, archived endpoints, and
committed options. `decisions.csv` carries the graph/world-context keys and
selected-option fields on every commit.

## Frozen spatial-model telemetry

Supplying `--spatial-shadow-checkpoint` adds predictions from the spatial model
without changing the existing planner by default. Every `planner_candidates`
row includes the counterfactual-usefulness score, raw predicted-activity score,
predicted action-versus-NOOP pixel and effect differences, predicted pixel
change, predicted effect, uncertainty, mode, selection weight, and weighted
priority bonus. Each real save-state branch also emits
`spatial_shadow_branch_evaluated`, linked by decision, branch ID, candidate
rank, action, and duration. It records:

- pixel and effect-weighted prediction error;
- the corresponding frame-persistence baselines;
- whether the prediction beat persistence;
- spatial-effect L1 and F1;
- predicted versus observed effect mass; and
- predicted pixel-change magnitude;
- predicted action-versus-duration-matched-NOOP pixel and effect differences;
- the raw activity and counterfactual-usefulness scores;
- actual action-effect contrast against the matched verified NOOP branch; and
- ensemble uncertainty.

`spatial_shadow.csv` provides one flat row per verified branch for plotting.
`summary.json` reports evaluation count, persistence wins, mean metrics, and
whether the separate spatial parameter-hash audit passed. It also records
whether selection was enabled, the configured weight, and the mean selection
bonus. The run manifest stores checkpoint file and parameter hashes plus the
mode and weight.

`--spatial-selection-weight N` enables a controlled frozen-model ablation. The
counterfactual-usefulness score is multiplied by `N` to prioritize which
save-state candidates are verified first. It is not added to the final
verified-branch commit score: once real outcomes are available, the existing
outcome-based objective decides which branch is committed. No model parameters
are updated, and verified-branch error is never fed back into the current
decision. Candidate, branch, and committed-decision telemetry retain the raw
score, weight, bonus, mode, and `spatial_selection_applied_to_commit` flag.
Omitting the option, or setting it to zero, preserves observational mode
exactly.

Supplying `--spatial-returnability-checkpoint` with the exact spatial
checkpoint it was trained against adds two observational fields to candidate
and verified-branch rows:

- `spatial_shadow_predicted_returnability` is the ensemble mean probability of
  an observed visual-state return path within the training horizon;
- `spatial_shadow_returnability_uncertainty` is the variance across relation
  heads.

The sidecar is parameter-hash bound to its frozen spatial encoder and cannot be
loaded with another encoder. The manifest records both checkpoint hashes and
the pixel-graph target configuration. `summary.json` reports mean probability,
mean uncertainty, and whether `spatial_returnability_parameter_audit` proved
that evaluation left the sidecar unchanged. Returnability has no selection or
commit weight; the current checkpoint is research telemetry, not a reward.

## Bidirectional save-state probes

`--returnability-probe-depth N` enables explicit, telemetry-only endpoint
experiments. For every verified branch, the collector restores its save state,
tests all configured controller actions at the branch duration, and retains the
closest `--returnability-probe-beam-width` endpoints for the next depth. A
candidate is compared with the frame produced by restoring the original root
and applying NOOP for the same total emulator time. This matched control keeps
ordinary animation from masquerading as irreversibility.

The append-only event stream includes:

- `bidirectional_probe_started` with the actions, depth, beam, threshold, state
  aliases, and source pixels;
- `bidirectional_probe_reference` for every duration-matched NOOP frame;
- `bidirectional_probe_step` for every tested path, including state aliases,
  emulator event links, pixel distance, exact-match status, and return result;
- `bidirectional_probe_completed` with path coverage, shortest observed return,
  best distance, and `no_return_within_probe_budget`.

All generated emulator steps use phase `returnability_probe`. The standard
experience importer excludes them, and branch scores, attempt memory, and
persistent parameters never see their outcomes. `returnability_probes.csv`
provides one row per path. `summary.json` reports probed branches and paths,
return counts, budget-scoped non-returns, mean best distance, and the artifact
location. “No return” always means within the logged action/depth/beam budget;
it is not asserted as a universal property.

When a probe run resumes from assisted-policy telemetry, its manifest inherits
assisted provenance as `human_prior_resume_observational`. Such a run can be
used for evaluator validation but cannot enter the strict dataset.

### Explicit-probe import

`lolo-spatial-probe-returnability-train` takes separate repeated
`--training-run` and `--validation-run` arguments. Before decoding examples it
requires a complete manifest, content-addressed pixel frames, the requested
reward track, matching probe settings, one start/completion/verified-branch
lifecycle per label, internally consistent return evidence, and valid source
and endpoint digests. Conflicting labels abort the import.

Transitions are deduplicated within each partition. Exact source, endpoint,
action, and duration overlap is removed from validation first; any remaining
example whose source pixels occur in training is removed as well. Both counts
are reported in the metrics provenance. Both remaining partitions must contain
positive and budget-scoped negative labels. Training is balanced by label;
validation keeps the natural prevalence so ROC AUC, Brier error, majority
accuracy, and calibration are not distorted by a tiny balanced sample.
Manifests and event logs are hashed into the resulting metrics file.

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

## Episodic resume

Use a state the agent previously reached without replaying a hand-authored
controller sequence:

```bash
lolo-neural-run \
  --host build/lolo-libretro-host \
  --rom "Adventures of Lolo.nes" \
  --core "$HOME/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --checkpoint checkpoints/cycle-000010.pt \
  --resume-run runs/<parent-run> \
  --resume-decision 879 \
  --decisions 500
```

The child run records `episodic_resume_completed` and stores the parent run ID,
decision, source location, and source-event SHA-256 in its manifest. Replay
validates that hash and reconstructs the parent decision before applying the
child event stream. Gameplay-only resumes exclude `START` and `SELECT` from the
agent action set.

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
For an episodically resumed run it first verifies and replays the content-hashed
parent log, so the same integrity check covers the provenance boundary.

Two standalone local players are generated:

- `replays/committed/index.html` follows only actions ultimately chosen by the
  agent, with explicit frames for resets and archive-restoration jumps;
- `replays/full/index.html` shows every planning branch, state restoration,
  action frame, and committed-decision marker in chronological order.

Both players are scrub-capable and offer 5, 15, 30, 60, 120, and 240 fps. Once
rendered, playback requires only the HTML and its content-addressed PNG folder;
the emulator and ROM are not read by the browser. `replay_manifest.json` records
the verification result and exact input hashes.
