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

## Current status

Acceptance gates are defined in [docs/roadmap.md](docs/roadmap.md) §12. Their
current standing, with the evidence for each:

- **Gate 3 — accessibility consequence: CLOSED on the assisted track.**
  Removing the entity at cell `(7,6)` — two button transformations and one
  push of the transformed object — opens 24 certified configuration-held
  cells against a 7-cell baseline, including the milestone-bearing `(12,11)`
  cell; a rerun from a fresh restore reproduced that envelope at Jaccard 1.0
  (learnings §4.30; roadmap §17 item 4). The companion probe certified an
  earlier confirmed push as accessibility-*neutral* (§4.28), so manipulation
  value is a measurable property of configurations rather than of
  manipulations as such.
- **Gate 4 — deliberate preparation: OPEN.** Four interventions have been
  measured and each failed with a distinct, named mechanism: a
  verified-accessibility restore preference was behaviorally redundant,
  because the incumbent novelty scorer already chose the same branch
  (§4.43); a hypothesis-directed search reserve had no searches to ride,
  because the planner commits without searching (§4.45, §4.46); commit-time
  steering toward the target starved the exploration that deposits the
  archives later restores consume (§4.50); and the target-aware restore key
  alone could not hold the position the agent was already standing on, which
  is never an archive candidate (§4.51). Roadmap §19 and §20 record the
  resulting plan changes.
- **WP5 — learned controllable-region tracker: campaign complete, promoted
  to shadow.** The detector-free learned masking convention passes the
  functional promotion gate on every axis and every corpus, exceeding the
  assisted incumbent on stability, preservation, and in-place detection
  (§4.42). It is not yet wired into the planner.
- **WP9 step 1 — milestone discovery: falsified three times.**
  Delayed-divergence valence survived; the event unit itself did not. The
  representation rebuilds on object-level tracks after WP2/WP3 integration
  (§4.33, §4.36, §4.41).

Two boundaries constrain how those results may be read. Every native run in
the certified-probe and Gate 4 series (`v322`–`v337`) was collected with
assisted-track instruments, so none of it is strict-track evidence; the
strict re-measurement is gated on WP5. And gate design is treated as a
first-class deliverable (roadmap §18 item 4): several failures above were
caught — and a few caused — by measurement quality rather than capability,
including one operational measurement error by the orchestrator itself
(§4.52). Every gate is expected to ship an instrument that can contradict
it.

## Research record

The active research plan, ordered implementation work packages, acceptance
gates, and coding-agent handoffs are maintained in
[docs/roadmap.md](docs/roadmap.md); its amendments §17–§20 carry the current
plan changes and their evidence. Prior experiments, negative results, and the
plan changes they caused are synthesized in
[docs/learnings.md](docs/learnings.md), which is the source of truth for
every claim above.

Individual experiments keep dated notes alongside those two files: each one
records its preregistration before any arm runs and appends the scored result
afterwards. Entry points into the current series:

- [docs/object-removed-probe-2026-08-16.md](docs/object-removed-probe-2026-08-16.md)
  and
  [docs/paired-accessibility-probe-2026-08-16.md](docs/paired-accessibility-probe-2026-08-16.md)
  — the certified paired probes that closed Gate 3.
- [docs/wp5-tracker-training-2026-08-16.md](docs/wp5-tracker-training-2026-08-16.md)
  — the six-iteration WP5 perception chronicle, plus
  [docs/tracker-ood-eval-2026-08-16.md](docs/tracker-ood-eval-2026-08-16.md)
  and [docs/strict-collection-recon-2026-08-16.md](docs/strict-collection-recon-2026-08-16.md).
- [docs/wp8-lite-ablation-design-2026-08-16.md](docs/wp8-lite-ablation-design-2026-08-16.md),
  [docs/wp8-relational-planner-design-2026-08-17.md](docs/wp8-relational-planner-design-2026-08-17.md),
  and
  [docs/wp8-search-scheduling-design-2026-08-17.md](docs/wp8-search-scheduling-design-2026-08-17.md)
  — the four measured Gate 4 interventions.
- [docs/milestone-event-census-2026-08-16.md](docs/milestone-event-census-2026-08-16.md)
  and the `milestone-scoring` v1/v2/v3 notes — the WP9a falsifications.
- [docs/direction-review-2026-08-16.md](docs/direction-review-2026-08-16.md)
  — the adversarial review that set the current work order.

Paid or unattended experiments should use the mandatory evidence and cost
gates in [docs/research-loop.md](docs/research-loop.md). See
[docs/runpod.md](docs/runpod.md) for the Linux/CUDA image, deterministic
Mac-versus-Pod benchmark, private-asset setup, and automatic Pod shutdown.

## Run

The package has no third-party runtime dependencies. Modules that train or
evaluate learned models additionally need the `ml` extra (`torch`,
`torchvision`):

```bash
python3 -m unittest discover -s tests -v
python3 -m lolo_agent --steps 40 --depth 2
```

## Package map

`lolo_agent/` is flat. The long-standing core is `agent.py`,
`world_model.py`, `memory.py`, `pixels.py`, `environment.py`, `libretro.py`
and `native_env.py` (emulator binding), `neural_planner.py` (the search
monolith), `neural_run.py` (the native run driver), the `spatial_*` modules
(the unlabeled spatial world model and its sidecars), `entity_behavior.py`
(persistent anonymous appearance types), `run_logging.py` /
`log_summary.py` / `replay.py` (telemetry), and `experiment.py` /
`research_cycle.py` (campaign drivers).

The object-centric work added the following. None of it reads ROM memory,
sprite tables, or supplied object labels; where a module scores assisted-era
telemetry it does so as evaluation input only, and `strict_lineage.py` is the
mechanical check on that boundary.

Object representation:

- `object_tracks.py` — anonymous object tracks and transitions, extracted
  from `neural_planner.py` with no behavior change.
- `object_correspondence.py` — endpoint-relative correspondence across up to
  four simultaneous tracks, abstaining and freezing on ambiguity instead of
  swapping identity. Current cells derive from present-frame evidence;
  accumulated history is provenance, never correspondence input (§4.29).

Detector-free perception (WP5):

- `counterfactual_labels.py` — controllable-region pseudo-labels from branch
  structure alone: factual endpoints against duration-matched `NOOP` controls
  from the same causal root, with leave-one-action-out corroboration.
- `controllable_tracker.py`, `controllable_tracker_train.py` — a per-cell
  controllable-region mask head over the frozen spatial encoder.
- `pixel_mask_head.py`, `pixel_mask_train.py` — pixel-resolution silhouette
  refinement over the frozen tracker, supervised by the same counterfactual
  differences before they are pooled to cells.

Measurement and gates:

- `accessibility.py` — the certified configuration-held accessibility
  instrument: certification windows, coverage, footprint-excluded deltas,
  scored target bits, repetition agreement. It has no policy authority.
- `accessibility_preference.py` — the verified-accessibility preference term
  for archive/restore ranking. A record whose provenance is not
  `certified_hold` scores exactly zero, with the refusal exposed; predicted
  accessibility can gate measurement, never preference.
- `mask_sensitive_gate.py`, `functional_mask_gate.py`,
  `tracker_substitution_replay.py`, `tracker_ood_eval.py` — the WP5
  promotion gates and out-of-distribution evaluation, all judged against
  detector-free counterfactual ground truth.
- `milestone_discovery.py`, `milestone_discovery_run.py` — the offline
  milestone-discovery scoring skeleton and its one-shot scoring pass (WP9a).
- `conflict_root_mining.py` — read-only mining of stored telemetry for
  ablation roots that exhibit *score conflict*, after §4.43 showed that a
  root without conflict cannot discriminate deliberate from incidental
  choice.

Planning:

- `relational_planner.py` — the WP8 hypothesis planner:
  `establish_configuration`, `hold_configuration`, and
  `exploit_configuration` hypotheses over a deterministic bounded queue,
  driven by verified-event summaries, scoring with every component exposed,
  and emitting declarative realization objectives that the monolith's option
  search interprets. It never searches and never touches an emulator.

Protocol and provenance:

- `partitions.py` — loader and audit for the pre-registered room partition in
  `configs/evaluation-partitions.json`; see
  [docs/protocol.md](docs/protocol.md) for the allocation and its
  immutability rules.
- `strict_lineage.py` — a static linter answering mechanically whether a
  module's or a checkpoint's derivation could have touched assisted
  perception.

`accessibility.py`, `accessibility_preference.py`, `relational_planner.py`,
and `milestone_discovery.py` are pure: they map telemetry-shaped values to
frozen result dataclasses and never touch an emulator, a file, or planner
state. The preference term and the relational planner are wired into
`lolo-neural-run` but default to inert (weight `0.0`, authority `off`), so
default ranking stays bit-identical to the incumbent. The learned masking
convention is not imported by the planner at all.

Manifests classify collection provenance per run. A strict policy resumed
from an assisted-era save state records `strict_from_assisted_state` rather
than `strict_rule_free`, so state-source ancestry cannot be laundered
(`docs/strict-collection-recon-2026-08-16.md`).

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

Learn persistent anonymous appearance types and conditional behavior
distributions from passive intervals and controlled local effects:

```bash
lolo-neural-run \
  ... \
  --human-prior-hearts \
  --human-prior-option-effect-stability-steps 3 \
  --human-prior-option-effect-phase-offsets 3 \
  --human-prior-option-effect-local-controls \
  --human-prior-option-entity-frontier \
  --human-prior-option-entity-curiosity-weight 2.0 \
  --human-prior-option-entity-curiosity-reserve 4 \
  --human-prior-option-entity-inert-penalty-weight 3.0 \
  --human-prior-proactive-entity-probe-limit 1 \
  --anonymous-entity-behavior-checkpoint experiments/lolo1-entity/anonymous-behavior.json \
  --anonymous-entity-behavior-mode learn \
  --anonymous-entity-passive-horizons 16,32,64,224 \
  --anonymous-entity-causal-horizons 16,32,64,224
```

The checkpoint contains no sprite names or supplied mechanics. Use
`--anonymous-entity-behavior-mode frozen` for parameter-immutable held-out or
sequel evaluation. Optional passive horizons restore the decision-root save
state and advance an equal-action `NOOP` branch, exposing delayed visual
dynamics without changing the committed controller action. The optional inert
penalty uses only learned factual/control pixel matches to down-rank supported
interventions that previously had no measured effect; it is not an object rule
or label. Causal horizons additionally compare each verified controller
endpoint with an equal-duration neutral endpoint, then assign a
later differential life loss only to rare patches that first showed a local
differential outcome at an earlier nonterminal horizon. Use
`--anonymous-entity-shadow-horizons 16,32,64,224` for policy-neutral endpoint
predictions. After validation, `--anonymous-entity-hazard-veto` can filter only
context-matched, provenance-qualified causal hazards; it fails open if every
verified endpoint is marked. See `docs/anonymous-entity-behavior.md` and the
dated policy-gate report.

The optional entity-curiosity weight and reserve make under-tested anonymous
appearance/action pairs first-class exploration candidates. Their matched
controls also learn reusable no-effect outcomes, while persistent visual
changes retain the existing phase, player-mask, and causal promotion gates.
Both settings default to zero.

Rank restore candidates by the certified accessibility of the configuration
they hold, and let an object-level hypothesis direct where the search goes.
Both mechanisms are default-off; the flags below are the shape the
preregistered Gate 4 ablations used, with the preference weight left at its
inert default:

```bash
lolo-neural-run \
  ... \
  --human-prior-accessibility-records experiments/lolo1-wp5/wp8lite-accessibility-records.json \
  --human-prior-accessibility-preference-weight 0.0 \
  --relational-planner-authority telemetry \
  --relational-navigation-seams restore_only \
  --relational-decision-budget 12
```

Records are imported through a provenance check and only `certified_hold`
records can score; loading them is independent of scoring them, so an
ablation's two arms can load an identical store and differ in the weight
alone. `--relational-planner-authority off` (the default) keeps planner
ranking and restore selection bit-identical to today, `telemetry` proposes
and logs hypothesis plans with zero selection influence, and `selection`
additionally lets the active hypothesis direct restore selection and
option-search reserve slots. `--relational-navigation-seams` chooses which
navigation seams selection authority may use — `both`, `restore_only` or
`off` — where `restore_only` disables commit-time steering so exploration
runs identically to the control and only the closing restore is contested
(§4.50). The seams are inert at any authority other than `selection`. A
fourth choice, `restore_plus_deposit`, adds a seam that deposits a committed
position adjacent to a certified milestone cell as an archive so the restore
key has a candidate to reach (§4.51); it lands with the E6 build and is
**not yet committed**, so it is not a valid choice at this commit. The
decision budget
is hypothesis scope, not beam width, and consumes no beam slots. All four
measured Gate 4 interventions are recorded arm by arm in the WP8 design
notes; none of these settings is promoted.

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
`lolo-neural-run`. Each committed decision has a content-addressed, opaque
emulator snapshot, so a child can validate the parent telemetry and restore in
constant time. The agent never observes the snapshot bytes. Legacy logs still
use deterministic full-event replay, and gameplay resumes exclude the
title-screen `START`/`SELECT` controls.

Assisted exact-search resumes also retain evaluator-owned snapshots for the
small bounded set of promoted, unconsumed option alternatives. Their opaque
bytes never enter policy observations; pixels and snapshot hashes are verified
on import, and the child copies active alternatives into its own telemetry so
later descendants remain self-contained.

## Offline research tools

The modules added by the object-centric work are run in module form rather
than as installed console scripts. They read stored telemetry, datasets, and
checkpoints; none of them starts an emulator. `mask_sensitive_gate`,
`tracker_substitution_replay` and `tracker_ood_eval` carry their
preregistered corpus, checkpoint, and report paths as defaults, so a bare
invocation reruns that gate's recorded configuration. `functional_mask_gate`
does not: bare, it runs detection quantity v1 against the v1 pixel head and
writes the v1 report, and no invocation of it reproduces the WP5-final PASS
— see below.

Ask whether a module or checkpoint could have touched assisted perception:

```bash
python3 -m lolo_agent.strict_lineage lolo_agent/controllable_tracker.py
```

Generate detector-free controllable-region labels, then train the cell-level
tracker and the pixel-level silhouette head over a frozen spatial backbone.
The label writer emits a sidecar manifest with a content digest and refuses
to overwrite an existing pair:

```bash
python3 -m lolo_agent.counterfactual_labels \
  --dataset experiments/lolo1-medium/dataset \
  --maximum-roots 32768 \
  --destination experiments/lolo1-wp5/wp5-labels-full.jsonl

python3 -m lolo_agent.controllable_tracker_train \
  --labels experiments/lolo1-wp5/wp5-labels-full.jsonl \
  --dataset experiments/lolo1-medium/dataset \
  --spatial-checkpoint experiments/lolo1-spatial-v10/checkpoints/spatial-v10-native-adapt-e5.pt \
  --checkpoint experiments/lolo1-wp5/controllable-tracker-v4.pt

python3 -m lolo_agent.pixel_mask_train train \
  --labels experiments/lolo1-wp5/wp5-labels-full.jsonl \
  --dataset experiments/lolo1-medium/dataset \
  --tracker-checkpoint experiments/lolo1-wp5/controllable-tracker-v4.pt \
  --spatial-checkpoint experiments/lolo1-spatial-v10/checkpoints/spatial-v10-native-adapt-e5.pt \
  --checkpoint experiments/lolo1-wp5/pixel-mask-head-v3.pt \
  --target-semantics occupied-v2
```

Rerun the WP5 promotion gates:

```bash
python3 -m lolo_agent.functional_mask_gate \
  --detection-quantity v2 \
  --head-checkpoint experiments/lolo1-wp5/pixel-mask-head-v3.pt \
  --report /tmp/functional-gate-rerun.json

python3 -m lolo_agent.mask_sensitive_gate
python3 -m lolo_agent.tracker_substitution_replay
python3 -m lolo_agent.tracker_ood_eval
```

That functional-gate invocation reruns the **v4-convention** gate — the
§4.40 NO-PROMOTE — not the WP5-final PASS. The PASS
(`functional-gate-v5-report.json`, payload digest `ac4bd00f…`, §4.42) was
produced by a preregistered scratchpad driver, quoted verbatim for audit in
[docs/wp5-tracker-training-2026-08-16.md](docs/wp5-tracker-training-2026-08-16.md)
§"Run (fixed now)": that spike's ownership terms deliberately kept
`functional_mask_gate.py` import-only, and the module CLI still exposes no
flag for the switch that selects the ensemble-agreement-anchor-v3
reconstruction convention — `anchor_uncertainty_bound` is a
`PixelSilhouettePredictor` constructor argument
(`ANCHOR_CELL_UNCERTAINTY_BOUND_V3`), not a property carried by the head
checkpoint. Reproducing the PASS means composing that predictor, not calling
this CLI. Both of the gate's report defaults point at recorded campaign
artifacts, so always send `--report` to a scratch path unless you intend to
overwrite one.

Each of these writes a report carrying its thresholds, its provenance
(backbone and checkpoint parameter digests, the label manifest digest, and
the reward track), and a content digest, so a rerun is byte-comparable
against the recorded result.

Run the offline milestone-discovery scoring pass, and mine stored telemetry
for ablation roots that exhibit score conflict:

```bash
python3 -m lolo_agent.milestone_discovery_run --v3
python3 -m lolo_agent.conflict_root_mining
```

`milestone_discovery_run` scores the census-qualified corpora; `--v2` and
`--v3` select the successive preregistered redesigns. All three passes are
engineering artifacts: every corpus available to feed them was collected with
assisted instruments, so none of their output is strict-track evidence.

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

Cost-gated campaigns with mandatory reflection run through
`lolo-research-cycle`:

```bash
lolo-research-cycle run \
  --plan configs/research-cycle.example.json \
  --campaign-dir experiments/lolo1-campaign
```

A cycle plan may declare an `evaluation_partition` section (manifest path,
game, room, intent, reward track, and one audited artifact path per
persistent class). When it does, the cycle loads the pre-registered manifest
before any side effect, authorizes or refuses the write against the room's
partition and reward track, and records a digest audit over every persistent
artifact class at cycle start and cycle end — with frozen-evaluation cycles
additionally verifying that no digest moved.

To connect an emulator, implement `PixelSaveStateEnv` from
`lolo_agent/environment.py`. Keep success, room number, sprite identity, and RAM
outside that interface. An evaluation harness may inspect success to score a
run, but must never return it to the agent.

See [docs/protocol.md](docs/protocol.md) for the split and freeze protocol and
[docs/architecture.md](docs/architecture.md) for the component boundaries.
