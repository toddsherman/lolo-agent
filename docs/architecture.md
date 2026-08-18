# Architecture

The core deliberately separates the agent's observations from evaluator-only
ground truth.

```text
                 evaluator-only success / room accounting
                                  ^
                                  |
Nestopia <-> native host <-> PixelSaveStateEnv <-> branching planner
             RGB/actions      opaque handles       |       |
                                                   |       +-> temporary memory
                                                   +----------> world model
```

`PixelSaveStateEnv` returns raw images. Save-state blobs are opaque: the agent
may store and restore them but cannot decode them. Controller button names are
hardware affordances, not game semantics.

`EmpiricalWorldModel` is the dependency-free planning baseline. It predicts distributions
over coarse, object-agnostic visual signatures for each action. It combines a
state-conditioned table with an action-conditioned fallback, enabling limited
transfer while remaining dependency-free. It is intentionally replaceable by
a convolutional encoder plus recurrent latent dynamics model.

`VisualDynamicsModel` is the first neural training baseline. It learns a
convolutional image encoder, action-conditioned latent transition, and pixel
decoder on branched emulator transitions. It currently validates the neural
data and freeze pipeline; planner integration follows after multi-step model
validation.

`BranchingAgent` performs receding-horizon beam search. It restores a state,
tries actions, scores visual novelty, prediction surprise, uncertainty, and
screen change, and commits only the first action of the best path. During
training all observed branches can update the model. In evaluation, model
updates raise an error and only temporary per-run memory changes.

The mock puzzle is not imported by the agent package. Its symbolic state and
success predicate are evaluator-side test fixtures; the agent receives only
the rendered byte array.

## Spatial causal successor

The first persistent spatial successor is implemented in
`spatial_world_model.py`. It preserves a learned 2-D token map, predicts
action-localized pixel effects with an uncertainty ensemble, and uses sparse
rendering so unchanged pixels are copied rather than regenerated. Strict and
assisted telemetry are now bound to separate dataset tracks.

Its local flow/residual renderer passes a trajectory-balanced run-held-out
offline and native prediction gate. Its counterfactual usefulness score remains
disabled by default because paired planning ablations were mixed.
See
`spatial-causal-model-2026-08-10.md` for the reproducible MPS result.

`spatial_returnability.py` adds a separate ensemble relation head over frozen
spatial tokens. Its targets come only from the observed pixel-transition graph:
positive transitions have an observed return path, well-probed non-returns are
negative, and uncertain cases are censored. The sidecar passes one run-held-out
development fold but fails native branch calibration, so it is attached only
to telemetry and cannot affect the planner.

`bidirectional_probe.py` collects stronger relation targets directly from
opaque save-state branches. It performs a bounded pixel-only search from each
endpoint toward a duration-matched NOOP reference from the original root. The
collector has a separate logging phase and no reference to planner scores,
objects, rewards, or evaluator labels. Its output is intended to replace the
policy-dependent graph negatives used by the first returnability sidecar.

`probe_returnability_import.py` converts those explicit outcomes into a
provenance-audited relation corpus. Complete training and validation runs are
named separately; strict and assisted telemetry cannot mix, exact transition
and source-image overlap are removed from validation, and both partitions must
retain return and budget-scoped non-return examples. The relation head is
trained from frozen spatial encodings of the observed source and verified
endpoint. During live planning it can inspect only predicted endpoints; after
branch execution its telemetry uses the observed endpoint. Neither path
currently affects control.

`entity_behavior.py` adds the first persistent object-centric sidecar. It
clusters pooled RGB patches into anonymous appearance types and learns
conditional empirical outcome distributions from matched controls and passive
intervals. Outcomes are position-relative pixel-effect hashes; types have no
sprite names or supplied mechanics. Context-specific distributions can differ
from the cross-room fallback. The primary context is a translation-invariant,
coarsely binned distance and alignment between the anonymous patch and the
action-correlated controllable patch; an anonymous whole-scene hash is used
only when that patch cannot be localized. Optional save-state `NOOP` horizons
collect duration-conditioned passive dynamics. One appearance can therefore
remain stationary in one relation and transform or precede a terminal visual
transition in another. Exact save-state evidence is deduplicated,
contradictory observations reduce confidence, and a frozen mode supports
parameter-immutable evaluation. Empirical terminal correlation is stored
separately from the localized causal hazard posterior. The sidecar has additive
selection weight zero; after native shadow calibration, an optional conservative
filter can veto a provenance-qualified hazardous endpoint while failing open if
all verified endpoints are hazardous.

The causal behavior collector pairs every verified non-neutral endpoint with
an equal-duration neutral endpoint, then advances both with the same `NOOP`
horizons. A rare patch is localized only when the pre-wait appearances match,
the controllable-patch relation differs, its local position-relative outcome
differs, and both branches are nonterminal. A later life-loss contrast can be
assigned only to a patch localized at an earlier horizon. This turns global
terminal association into an intervention-linked local chain while preserving
the raw anonymous appearance identity. Ordinary passive terminal rows remain
in telemetry but are ineligible for checkpoint updates.

## Module layering and dependency rules

The 2026-08-16/17 campaign added modules on both sides of the planner. What
keeps them separable is a one-way import rule: pure modules never import the
monolith, and the monolith adapts its own state into their narrow read-only
input views.

```text
     process entry            neural_run
                                  |  native env, checkpoints, run logging
                                  v
     planner monolith        neural_planner ------> goal_prior (assisted surface)
                              |     |     |
              +---------------+     |     +----------------------+
              v                     v                            v
     relational_planner    accessibility_preference        object_tracks
              |                     ^                            ^
              +---------------------+                            |
                    (its only intra-package import)      object_correspondence

     pure: stdlib only, no torch, no emulator, no files, no telemetry streams

     offline instruments, imported by tests and drivers, never by the planner:
       accessibility            certified paired-probe scoring
       conflict_root_mining     score-conflict root mining over events.jsonl
       milestone_discovery      WP9a offline scorer  <- milestone_discovery_run
       strict_lineage           lineage linter over modules and checkpoints
       partitions               frozen evaluation split  <- research_cycle

     WP5 perception chain, trained and gated outside the planner
     (pipeline order; each stage imports the stage before it):
       counterfactual_labels --> controllable_tracker --> pixel_mask_head
       gates over those stages: tracker_ood_eval, tracker_substitution_replay,
                                mask_sensitive_gate, functional_mask_gate
```

Arrows point from importer to imported, except in the WP5 row, which is
written in pipeline order. Nothing below the planner imports upward: no
instrument, gate, or pure module imports `neural_planner`.
`relational_planner` never imports it either; `neural_planner` imports the
relational names under `RELATIONAL_*` aliases and adapts planner state into
`RelationalStateView` / `ArchiveCandidateView` / `TransitionRuleView` values.
`relational_planner`, `accessibility_preference`, `accessibility` and
`object_tracks` are all emulator-free and file-free: they hold no environment
handle and open nothing.

## Strict and assisted lineage

The strict and assisted tracks were previously discipline. `strict_lineage.py`
turns the boundary into tooling. The assisted surface is named explicitly: the
`goal_prior` module plus the symbols `detect_player`, `player_pixel_mask`,
`PixelHeartGoalPrior`, `HeartGoalAnalysis` and the heart/chest prototypes.

Two instruments share one entry point. An `ast` walker builds the
intra-package import/attribute graph and reports, for an entry module, every
transitive chain to that surface — every analyzed file is only ever read, never
imported or executed. A checkpoint auditor checks declared `reward_track`,
`persistent_inputs` and `excluded_inputs` against a strict allowlist, with
`RAM`, `rewards`, `object_labels`, `level_annotations`, `solutions`,
`planner_scores` and `evaluator_annotations` as explicit violations.
`lint_strict_lineage(paths)` combines both into one deterministic,
content-signed report and a `strict_lineage_linted` telemetry event.

The question is "could this derivation have touched assisted perception", so
the answer over-approximates on purpose. Naming an assisted symbol at all —
import, attribute, definition, parameter, or dynamic-access string — is
decisive; `human_prior`-prefixed telemetry names are advisory and never flip a
verdict alone. Two visible consequences: `object_tracks` reports
`assisted: true` on its injected `player_pixel_mask` *parameter* although it
imports nothing assisted, and `object_correspondence` inherits that verdict
through it; the linter also reports itself assisted, because its allowlist
constants are the symbol strings. Current verdicts elsewhere are
`assisted: false` for `relational_planner`, `accessibility`,
`accessibility_preference`, `counterfactual_labels`, `controllable_tracker`,
`pixel_mask_head`, `milestone_discovery`, `conflict_root_mining` and
`partitions`, and `assisted: true` for `neural_planner`, which imports
`goal_prior` directly.

The linter is not advisory. Unit tests invoke `lint_strict_lineage` on the
pure modules and assert `assisted: false` with an empty finding list, and on
the coupled training entry points to assert that the coupling runs only
through their evaluation imports — so a new import into `relational_planner`
or `pixel_mask_head` fails the suite rather than a review.

Lineage is also enforced at the data level.
`experience_import.classify_reward_track` maps a run manifest's declared
`reward_track` onto strict or assisted and refuses to guess on an unrecognized
value; `strict_from_assisted_state` — a strict policy branched from an
assisted-era save state with the ancestry disclosed in the manifest — counts
strict, and legacy `human_prior*` manifests keep their assisted
classification. `partitions.py` holds the frozen evaluation split and raises
`PartitionUpdateError`, carrying the exact `partition_update_rejected`
payload, when an update is attempted from a frozen partition.

The boundary matters for reading results: every `v3xx` run of this campaign is
assisted-lineage, so results measured from them — Gate 3 included — are
assisted-track claims. Strict-track re-measurement is gated on WP5 (roadmap
§17 item 4).

## Certified-accessibility instrument

`accessibility.py` productizes the methodology that closed Gate 3. It is pure
and has no policy authority: every function maps telemetry record dicts to
frozen result dataclasses, so a paired probe is scriptable instead of
hand-scored.

The certification predicate is `certify_branch`: a branch is
configuration-held only when its `anonymous_object_track_cells` equal the
probe root's tracked cells *and* its tracked world-state signature *and* its
confirmed world-effect signature both match the root's. The third outcome is
the load-bearing one — censored, which is neither held nor departed. Branches
outside the causal-restore validity window are censored, because
`archive_branch_restored` events carry no track fields and silently reset the
tracker (learnings §4.29), as are branches missing the track keys entirely
(pre-instrument-fix telemetry): the instrument cannot observe their
configuration, so it declines to classify them.

The same discipline runs through the rest of the module. `delta` takes the
declared footprint set as an explicit argument, because the vacated cell is
trivially nonzero and proves nothing. `ProbeBudget` records the budgets a
coverage claim is scoped to, so a non-reach is censored scoped evidence rather
than "unreachable". `repetition_agreement` encodes the Gate 3 repetition
criterion of ≥0.8 Jaccard between the certified coverage sets of two runs of
the same configuration.

Gate 3 is closed on the assisted track by this instrument: removing the `(7,6)`
entity opened 24 certified cells against 7 for both baselines, including a
milestone-bearing cell, while the earlier confirmed eastward push measured
certifiably neutral (learnings §4.28, §4.30). The repetition from a fresh
restore reproduced the envelope at Jaccard 1.0 (roadmap §17 item 4).

`accessibility_preference.py` is the planner-facing half, and the only
accessibility module the planner imports. It scores one candidate
configuration's `CertifiedAccessibilityRecord` against the current
configuration's, exposing every component separately. Three rules are
structural rather than conventional: a record whose provenance is not
`certified_hold` scores exactly zero with the refusal exposed, so prediction
can gate measurement but never preference; only previously-unreachable cells,
frontiers and milestone-bearing cells score, while raw new-affordance counts
are logged at weight zero because configuration churn can mint affordances at
already-reachable cells; and absence is censored, never negative.

## Object-level perception

`object_tracks.py` is the WP1 extraction from the monolith. It is pure and
emulator-free — it imports only `Action` and `Frame` — and its masking
convention is a *parameter*: `player_pixel_mask` arrives as an injected
callable. That injection seam is what lets the substitution gates run the same
functions over recorded frames under two different masks without any planner
code participating.

`object_correspondence.py` is the descoped WP2 correspondence engine: at most
four simultaneous tracks, greedy minimum-cost assignment, abstain-and-freeze on
ambiguity, no split/merge events. Its contract is endpoint-relative by
construction (learnings §4.29, where five of six accumulated cells had relaxed
to baseline while the set still listed them): `EndpointRelativeTrackState`
separates still-changed, ever-changed, relaxed-to-baseline and
not-observed-here, and accumulated history is provenance, never correspondence
input. HUD regions and autonomous patrol are excluded by caller-supplied
predicates and reported with their pruning reason. The module never *confirms*
a transformation — an appearance beyond the match threshold at a persistent
locus freezes the track as a candidate for matched-control evidence.

## The learned masking convention (WP5)

The perception chain is detector-free end to end. `counterfactual_labels.py`
derives per-cell controllable pseudo-labels from branch structure alone:
factual action endpoints against duration-matched `NOOP` control endpoints from
the same causal root, 4-connected components of the changed-cell set surviving
leave-one-action-out corroboration, and explicit censor reasons
(`absent_control`, `ambiguous_endpoint`, `ambiguous_control`,
`no_sibling_corroboration`). `controllable_tracker.py` distills those labels
into a small per-cell head over the frozen spatial encoder, with checkpoint
provenance declared in the `strict_lineage` auditor's fields and the backbone
digest pinned at save time. `pixel_mask_head.py` refines the frozen tracker's
cell map to a per-pixel silhouette; its v2 target is the *occupied* silhouette
only, with the occupied/vacated split derived from counterfactual structure
alone, and its reconstruction convention (anchor rule, 0.5 operating point,
halo radius) is pinned in code before any gate runs against it.

The gates are separate modules so the scored quantity is auditable:
`tracker_ood_eval.py` (report-only out-of-distribution metrics against
detector-free ground truth), `tracker_substitution_replay.py`,
`mask_sensitive_gate.py` (scores only frames where masking demonstrably
changes the downstream quantity), and `functional_mask_gate.py`. The last one
carries the campaign's instrument correction: it scores *function* against
detector-free counterfactual ground truth rather than replication of the
assisted mask's bytes, with the assisted convention scored alongside as the
incumbent reference rather than as the referee (learnings §4.35).

Because they run the incumbent, the comparison gates import `goal_prior` and
therefore lint `assisted: true` — an evaluation-only coupling, and the reason
the training entry point `pixel_mask_train.py` is separately asserted to reach
the assisted surface only through those evaluation imports. The learned side of
the chain stays clean: `counterfactual_labels`, `controllable_tracker` and
`pixel_mask_head` all lint `assisted: false`.

Promotion status is **shadow, not wired** (learnings §4.42). The learned
convention — tracker v4, pixel head v3 with occupied-v2 targets,
reconstruction v3, detection quantity v2 — passes the functional gate on every
axis and every corpus and exceeds the assisted incumbent on stability,
preservation and in-place detection. It is not imported by `neural_planner` or
`neural_run`: the planner's tracking quantities still run under the assisted
mask, and WP5's acceptance clause — strict tracking without a
`PixelHeartGoalPrior` import — becomes reachable only once shadow telemetry
accumulates native evidence.

## Relational hypothesis planner

`relational_planner.py` implements the smallest hypothesis-planning slice that
can demonstrate chained deliberate preparation: `establish_configuration`,
`hold_configuration`, `exploit_configuration`. `propose` yields a
deterministic bounded queue; `advance` is driven exclusively by verified-event
summaries, so a verified transition contradicting the active hypothesis forces
a replan rather than a silent retry. Its accessibility term is computed by the
pure `verified_accessibility_preference`, inheriting that function's churn
exclusion, censoring discipline and refusal to score uncertified provenance.

Two structural boundaries are worth naming. First, the module never searches
and never touches an emulator: it emits declarative `RealizationObjective`
values that the monolith's option search interprets, which is why the whole
capability layer can be exercised at telemetry authority with zero behavioral
effect. Second, stored options are relational by construction — a
`RealizedOption` carries initiation and termination conditions plus
transfer-evidence counts, `from_payload` rejects unknown fields, and there is
no field that could hold a controller sequence or an absolute coordinate, so no
universal macro can be minted from one room-specific trajectory. Room-scoped
cells live only in in-run objective payloads and the episodic record store.
`navigation_preference` is likewise a tie-break inside an already-filtered
candidate set rather than a distance reward, and `target_cell_distance`
returns `None` whenever either side is unavailable so the caller falls open to
its incumbent ordering.

## Authority gating

Every new planner seam follows the same three-state pattern, config-gated and
defaulting to `off`:

```text
  off        no hypothesis is proposed or logged; the incumbent path stands
  telemetry  proposes, logs, and tracks state with zero selection influence
  selection  the active objective may additionally direct selection
```

`telemetry` is the mandatory shadow stage before any capability claim: the
relational shadow run confirmed non-interference, with the committed trajectory
identical to the incidental runs (learnings §4.44). Control arms run at `off`
and are checked for state-for-state invariance against the prior run before a
treatment is read (§4.50). At `selection` each seam still fails open to the
incumbent ordering when its inputs are unavailable.

The accessibility preference term follows the same discipline through a
separate weight rather than an authority: `verified_accessibility_weight`
defaults to `0.0`, so its ablation is a single-variable change against an
otherwise byte-identical configuration (§4.43).

Seams are individually selectable so a single lever can be ablated in
isolation — `relational_navigation_seams` takes `both`, `restore_only` or
`off`, where S1 is the commit-ladder tier and S2 the target-aware restore
key. A third seam, S3, the certified-adjacent archive deposit, adds the
choice `restore_plus_deposit`; it lands with the E6 build and is **not yet
committed**, so at this commit the selector rejects it. S3 is a gate, not a
weight: it only makes a candidate exist for the already-working S2 key to
re-rank.

That granularity exists because Gate 4 is being closed one measured lever at a
time, and it remains **open** after four failures, each with a distinct named
mechanism: the restore preference term was behaviorally redundant, because
novelty and certified value preferred the same branch (§4.43); the
hypothesis search reserve had no searches to ride, since the planner's own
archive growth suppressed its search gate (§4.45, §4.46); commit steering
worked as designed and made the outcome worse, because the excursions it
removed were what deposited the archive ladder later restores climb (§4.50);
and the restore key alone could not hold the position the agent was standing
on, because the current position is never an archive candidate (§4.51). The
`restore_only` mode is the direct encoding of the third finding, and the
uncommitted `restore_plus_deposit` mode of the fourth.

## Offline analysis instruments

`conflict_root_mining.py` walks recorded `events.jsonl` telemetry, reconstructs
the archive candidate set at every restore-selection and committed-decision
instant, and re-scores each candidate under both the baseline frontier score
recorded in telemetry and the real `verified_accessibility_preference`. A
conflict root is a decision point where the two argmaxes disagree. It exists
because of learnings §4.43: an ablation root without score conflict cannot
discriminate deliberate from incidental choice. When no organic conflict exists
in a corpus, that is a disclosed result and any constructed design is marked
`constructed: true` rather than silently substituted. It never touches an
emulator and writes only the manifest the caller asks for.

`milestone_discovery.py` and its telemetry-reduction runner
`milestone_discovery_run.py` are the WP9a offline scorer, engineering-only and
carrying an explicit assisted-footprint caveat: every corpus available to feed
them is assisted-track. WP9 step 1 has now been falsified three times (§4.33,
§4.36, §4.41). The delayed-divergence valence survives; what is wrong is the
event unit itself, which rebuilds on object-level tracks after WP2/WP3
integration.

`partitions.py` and `research_cycle.py` bind a run to a declared evaluation
partition and drive the evidence-gated cycle, with frozen partitions failing
closed.

## Instruments are components

Several of the campaign's failures were instrument defects rather than
capability defects — mask-irrelevant gate bits, a replication-versus-function
confusion, an ablation root without score conflict, and one operational
misdiagnosis where two independent-looking health signals were both wrong
because both were the orchestrator's own (§4.52). Gate and probe modules
therefore live inside the package and are held to the same explicit contracts
as the code they score: read-only over telemetry, no emulator, deterministic
content-digested reports, and a linted lineage verdict — clean where the
instrument is strict, and openly assisted where it deliberately runs the
incumbent. Roadmap §18 item 4 states the resulting policy —
budget design effort for a gate equal to the capability it gates, and require
of every gate an instrument that can contradict it.

The remaining successor milestones are:

1. changed-region rendering that reliably beats persistence on native branches;
2. multi-fold room-held-out validation without semantic inputs;
3. promote calibrated causal anonymous type-conditioned predictions from observation
   to planning, then replace the assisted action-ray locator with a learned
   controlled-entity tracker — the tracker now exists and passed its functional
   gate (learnings §4.42), so what remains here is shadow wiring and native
   evidence, not training;
4. a native-generalizing reachability/reversibility representation, followed
   by terminal-risk estimates learned from trajectories rather than hand-authored
   death or object rules — the certified-accessibility half is instrumented
   (`accessibility.py`, Gate 3 closed on the assisted track), the reversibility
   and terminal-risk halves are not.
