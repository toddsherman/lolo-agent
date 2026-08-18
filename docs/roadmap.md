# Lolo agent research and implementation roadmap

Status: active plan of record  
Last updated: 2026-08-16  
Primary execution platform: Apple Silicon MacBook Pro (M5)  
Repository: private `toddsherman/lolo-agent`

Companion evidence and negative-results record: `docs/learnings.md`

## 1. Purpose

Build an agent that learns to play and beat the original NES *Adventures of
Lolo* through interaction, using only rendered game images, NES controller
actions, outcomes produced by those actions, and opaque emulator save states.
The final system must not receive game rules, object names, level solutions,
ROM-specific RAM, demonstrations, or solution videos.

The research claim is stronger than completing the 50 training rooms. The
agent must learn reusable visual and causal concepts that support:

1. solving held-out *Lolo 1* rooms governed by familiar mechanics; and
2. attempting *Adventures of Lolo 2* with persistent learned parameters frozen.

During frozen evaluation, the agent may inspect the current room, branch from
opaque save states, plan, explore, and maintain temporary room memory. It may
not update persistent model parameters, anonymous behavior statistics, type
prototypes, learned rewards, or cross-room memory.

This document is both the research roadmap and the implementation handoff for
coding agents. Each work package states its dependencies, expected code
ownership, required telemetry, tests, acceptance gate, and stopping rule.

## 2. Tracks and claim boundaries

### 2.1 Strict interaction-only track

This is the final research track. The agent receives:

- RGB frames;
- controller actions and durations;
- opaque save/load/release capabilities;
- pixel-observed consequences of actions; and
- temporary memory during evaluation.

It does not receive supplied heart, player, enemy, block, egg, chest, life,
hazard, or room-solution labels. Room completion and game completion may be
measured by evaluator-only code but cannot be fed into the policy as a
hand-authored semantic rule.

### 2.2 Assisted development track

The current heart-aware planner uses pixel detectors for the player, hearts,
treasure, and life glyphs. It is useful scaffolding for isolating failures in
search, delayed credit, object tracking, phase transitions, and archive
restoration. It is not sufficient for the strict final claim.

Assisted results must always be labeled `human_prior_v2` or an equivalent
assisted reward track in manifests and datasets. They may guide engineering
decisions, but assisted transitions must not silently enter strict training or
strict evaluation corpora.

### 2.3 Evaluator-only fixtures

The deterministic start-screen bootstrap, known ROM digest checks, room
completion accounting, telemetry summarization, and replay rendering are
evaluator infrastructure. Bootstrap actions must remain separately attributed
and excluded from agent action statistics.

## 3. Current evidence and baseline

The current architecture already provides:

- a native headless Nestopia/libretro environment;
- raw RGB observations and opaque save-state branching;
- convolutional visual encoder/decoder and action-conditioned ensemble
  dynamics;
- exact emulator verification of candidate actions;
- episodic transition memory and archive restoration;
- extensive event, frame, state, decision, and transition telemetry;
- accelerated committed-action and full-branch replay;
- an anonymous appearance and behavior sidecar;
- matched-action versus matched-`NOOP` counterfactuals;
- persistent checkpoint freeze audits;
- bounded research-cycle and cost controls; and
- separate strict and assisted data provenance.

The most important recent evidence is recorded in
`docs/object-state-gate-2026-08-16.md`:

- Broad search did not solve the representation problem. Thousands of branches
  could not compensate for unreliable manipulation detection.
- Repeated visual appearances must remain eligible; rarity is not identity.
- Player masking must preserve disconnected adjacent objects.
- A one-cell anonymous displacement can be discovered and independently
  confirmed from pixels and save-state counterfactuals.
- The pushed-object state can now survive archive restoration and a new search
  cycle.
- In `entity-v320-room3-restored-object-track-d6`, all 2,497 verified
  descendants retained the manipulated cell.
- In `entity-v321-room3-confirmed-identity-d2`, all 132 descendants retained
  the confirmed interaction identity: action `RIGHT`, source `(7, 6)`,
  destination `(8, 6)`, effect distance one, and persistence evidence.
- Retaining a manipulation did not by itself collect another heart. Object
  memory is necessary but not sufficient; the planner must learn how
  arrangements change future accessibility.

The present limiting factor is therefore representation and long-horizon
causal planning, not a larger heart reward, a wider beam, or more unbounded
runtime.

## 4. Plan-of-record architecture

```text
RGB frames + controller actions + opaque save states
                         |
                         v
         learned controllable-region tracker
                         |
                         v
     anonymous multi-object tracker and phase encoder
                         |
            +------------+-------------+
            |                          |
            v                          v
 relational behavior model     visual world model
            |                          |
            +------------+-------------+
                         |
                         v
      accessibility / reversibility transition model
                         |
                         v
           object-level hypothesis planner
                         |
                         v
       exact emulator branch search and verification
                         |
                         v
        episodic archive, telemetry, and replay
```

The neural visual world model remains useful for representation, uncertainty,
and predicted consequences. It is not expected to replace exact emulator
verification for irreversible puzzle decisions. The planner should use learned
models to propose and rank experiments; the emulator remains the acceptance
oracle during interaction.

The next architecture is object-centric but not object-labeled. Internally,
entities may be called `track-17` or `type-42`; supplied names such as “skull”
or “egg” must never appear in model inputs, checkpoints, planning rules, or
strict policy logic.

## 5. Non-negotiable engineering invariants

Every implementation must preserve these constraints:

1. Pixels are authoritative. Save-state bytes are opaque capabilities and are
   never decoded by the agent.
2. Every persistent parameter update has explicit data provenance and a
   content digest.
3. Frozen evaluation must fail if any persistent parameter, type prototype,
   behavior count, or checkpoint digest changes.
4. Directly observed emulator outcomes outrank learned predictions.
5. Counterfactual claims require matched roots, matched durations, and logged
   controls.
6. Animation, player motion, and global scene changes must be separated from
   local anonymous-object changes before causal credit is assigned.
7. Unknown behavior fails open to experimentation; it is not treated as safe,
   dangerous, useful, or inert without evidence.
8. Raw novelty is not equivalent to progress.
9. No experiment runs without a falsifiable hypothesis, bounded budget, and
   declared stopping condition.
10. Every selected action, rejected alternative, archive restore, learned
    prediction, and persistent update must be reconstructable from telemetry.
11. Training, development, withheld, sequel, assisted, and strict data never
    mix implicitly.
12. Code changes must pass the full unit suite and a relevant native gate
    before being promoted.

## 6. Target data contracts

These are conceptual contracts. Coding agents may adjust exact class names,
but the semantics and provenance must remain stable.

### 6.1 Anonymous object track

```python
@dataclass(frozen=True)
class AnonymousObjectTrack:
    track_id: str
    appearance_fingerprint: str
    anonymous_type_id: int | None
    current_cell: tuple[int, int]
    previous_cell: tuple[int, int] | None
    source_cell: tuple[int, int] | None
    destination_cell: tuple[int, int] | None
    displacement: tuple[int, int] | None
    appearance_state_signature: str
    previous_appearance_state_signature: str | None
    local_context_signature: str
    phase_signature: str
    first_observed_frame: str
    latest_observed_frame: str
    persistence_steps: int
    confidence: float
    controlled_change_evidence: tuple[str, ...]
```

Requirements:

- `track_id` is temporary and content-derived within a room episode. It is not
  a semantic object class.
- `anonymous_type_id` is a learned appearance/behavior cluster and may be
  absent in frozen evaluation for unfamiliar appearances.
- Track matching must tolerate animation and partial player occlusion.
- Track splitting and merging must be explicit events, never silent mutation.
- Multiple visually identical tracks must remain distinct by spatial and
  temporal correspondence.
- A track may survive an appearance transformation while retaining a link to
  its previous state.

### 6.2 Object transition

```python
@dataclass(frozen=True)
class AnonymousObjectTransition:
    source_track_id: str
    action: Action
    duration: int
    source_cell: tuple[int, int]
    destination_cell: tuple[int, int] | None
    displacement: tuple[int, int] | None
    source_appearance: str
    target_appearance: str
    source_phase: str
    target_phase: str
    factual_frame: str
    control_frame: str
    effect_cells: tuple[tuple[int, int], ...]
    persistence_horizons: tuple[int, ...]
    causal_confidence: float
    reversible_status: str  # observed_return, budget_nonreturn, unknown
```

No “push,” “shoot,” “enemy,” or “egg” field is allowed. Those concepts may
emerge as clusters of measured displacement and appearance-transition
outcomes.

### 6.3 Room relational state

```python
@dataclass(frozen=True)
class RelationalRoomState:
    phase_signature: str
    controllable_region: str
    tracks: tuple[AnonymousObjectTrack, ...]
    reachable_player_cells: tuple[tuple[int, int], ...]
    interaction_frontier: tuple[tuple[str, Action, int], ...]
    observed_goal_regions: tuple[str, ...]
    reversible_frontier: tuple[str, ...]
    state_signature: str
```

Assisted runs may attach detected-heart diagnostics outside this strict state.
The strict state must not encode supplied goal identities.

### 6.4 Learned behavior query

```python
prediction = behavior_model.predict(
    appearance=track.appearance_fingerprint,
    anonymous_type_id=track.anonymous_type_id,
    action=action,
    duration=duration,
    local_context=track.local_context_signature,
    phase=room.phase_signature,
    controllable_relation=relation,
)
```

The result must expose a distribution, not a single rule:

- no measured effect probability;
- controlled displacement distribution;
- appearance-transition probability;
- autonomous-motion distribution;
- global-phase-change probability;
- terminal correlation probability;
- causally attributed terminal probability;
- entropy, evidence count, and provenance level; and
- exact-context versus fallback source.

### 6.5 Accessibility delta

```python
@dataclass(frozen=True)
class AccessibilityDelta:
    source_state_signature: str
    target_state_signature: str
    newly_reachable_cells: tuple[tuple[int, int], ...]
    newly_unreachable_cells: tuple[tuple[int, int], ...]
    newly_reachable_tracks: tuple[str, ...]
    lost_reachable_tracks: tuple[str, ...]
    goal_region_distance_delta: tuple[tuple[str, float], ...]
    return_observed: bool
    return_search_exhausted: bool
    verification_budget: tuple[int, int]
```

This structure contains measured graph consequences, not a declaration that a
configuration is globally solvable.

## 7. Ordered implementation program

### WP0 — Freeze the experimental baseline and dataset split

Priority: immediate  
Dependencies: none  
Primary files: `docs/protocol.md`, `docs/experiments.md`, new split manifest
under `configs/`, `lolo_agent/research_cycle.py`, related tests

Build:

1. Record the current checkpoint digests, behavior checkpoint digest, native
   host digest, core digest, ROM digest, and planning configuration used by
   `v318` through `v321`.
2. Pre-register rooms already touched as training/development rooms. Room 3 is
   not eligible as withheld because save states and targeted decisions have
   already influenced engineering.
3. Create an immutable room-allocation manifest with categories:
   `training`, `development`, `withheld_lolo1`, and `sequel`.
4. Add a loader that rejects writing training artifacts from withheld or sequel
   runs.
5. Add manifest fields for strict versus assisted tracks.
6. Add a frozen-parameter audit covering every persistent artifact, not only
   the neural checkpoint.

Required telemetry:

- `evaluation_partition_loaded`;
- `persistent_artifact_digest_audited`;
- explicit partition, reward track, and update authority in `manifest.json`;
- rejection event when an update is attempted from a frozen partition.

Tests:

- partition parsing and immutability;
- strict/assisted dataset separation;
- update rejection for withheld and sequel partitions;
- digest audit across neural, spatial, entity, and future relational models.

Acceptance gate:

- A frozen evaluation smoke run cannot change any persistent digest.
- A deliberately attempted update from a withheld fixture fails loudly.
- Training and assisted events cannot be imported into a strict withheld
  corpus without an explicit test-only override.

Stopping rule:

- Do not begin broad room training until the split is committed.

### WP1 — Extract object tracking from the monolithic planner

Priority: immediate  
Dependencies: WP0 may proceed in parallel  
Primary files: new `lolo_agent/object_tracks.py`,
`lolo_agent/unlabeled_entities.py`, `lolo_agent/neural_planner.py`, new
`tests/test_object_tracks.py`

Build:

1. Move track-state construction, serialization, legacy reconstruction, and
   player-masked appearance matching out of `neural_planner.py`.
2. Introduce `AnonymousObjectTrack`, `AnonymousObjectTransition`, and a
   deterministic `ObjectTrackSet` signature.
3. Convert the current single root object state into a tuple of tracks.
4. Keep backward compatibility with `v318` through `v321` telemetry.
5. Preserve the current confirmed-manipulation identity separately from
   transient interaction candidates.
6. Make all conversion functions pure where possible so they can be tested
   without an emulator.

Required interfaces:

- `observe_frame(frame, player_mask, phase) -> TrackObservation`;
- `match(previous_tracks, observation) -> TrackMatchResult`;
- `apply_verified_transition(...) -> ObjectTrackSet`;
- `to_telemetry()` and `from_archive_metadata(...)`;
- `signature` for deduplication and archive comparison.

Required telemetry:

- `anonymous_track_created`;
- `anonymous_track_matched`;
- `anonymous_track_occluded`;
- `anonymous_track_reacquired`;
- `anonymous_track_split`;
- `anonymous_track_merge_ambiguous`;
- `anonymous_track_expired`;
- `anonymous_track_set_restored`.

Tests:

- exact persistence;
- one-cell displacement;
- repeated identical appearances;
- player overlaps source cell;
- player overlaps destination cell;
- appearance animation within match threshold;
- appearance transformation beyond match threshold;
- temporary disappearance and reacquisition;
- archive round trip;
- legacy archive reconstruction;
- deterministic ordering and signatures.

Acceptance gate:

- Existing `v318` and `v321` archive metadata reconstruct equivalent tracks.
- No behavior change in existing planner tests.
- The full suite passes.

Stopping rule:

- Do not add multiple-track planning inside `neural_planner.py` before this
  extraction; otherwise the current monolith will become harder to validate.

### WP2 — Multiple simultaneous anonymous tracks

Priority: immediate after WP1  
Dependencies: WP1  
Primary files: `lolo_agent/object_tracks.py`,
`lolo_agent/unlabeled_entities.py`, `lolo_agent/neural_planner.py`, tests

Build:

1. Detect candidate stable regions across matched neutral frames.
2. Match tracks using appearance distance, displacement bounds, local context,
   phase, and temporal continuity.
3. Use minimum-cost correspondence for repeated appearances rather than a
   unique-appearance assumption.
4. Retain an ambiguity set when two assignments are equally plausible.
5. Propagate track hypotheses through exact option-search nodes.
6. Include the complete track-set signature in world-state beam diversity and
   archive deduplication.
7. Bound track count and hypothesis count to avoid combinatorial explosion.

Initial bounds:

- maximum active tracks per room state: 32;
- maximum ambiguous correspondence hypotheses: 4;
- maximum displacement considered during one verified edge: configurable,
  initially four coarse cells;
- track expiration only after configurable matched-neutral evidence, not one
  missing frame.

Required telemetry:

- candidate and retained track counts per branch;
- correspondence cost and ambiguity margin;
- track-set signature;
- pruning reason;
- per-track source, target, displacement, and appearance transition.

Tests:

- two identical objects moving independently;
- one object moving while another animates;
- two objects crossing or becoming adjacent;
- one object transforming while retaining continuity;
- track count and ambiguity bounds;
- no dependence on absolute room coordinates for type sharing.

Native acceptance gate:

- From a targeted Room 3 state, preserve at least two anonymous tracks through
  two consecutive verified manipulations.
- Restore the resulting archive in a new process and reproduce the same
  track-set signature.
- At least 95% of stable descendant branches retain the correct track set;
  every rejected or ambiguous match is explained in telemetry.

Stopping rule:

- If correspondence errors are dominated by coarse cell resolution, improve
  localized patch geometry before increasing search depth.

### WP3 — Explicit displacement and transformation transitions

Priority: high  
Dependencies: WP2  
Primary files: `lolo_agent/object_tracks.py`,
`lolo_agent/entity_behavior.py`, `lolo_agent/neural_planner.py`, tests

Build:

1. Convert confirmed source/destination evidence into explicit displacement
   vectors.
2. Represent an appearance change at a persistent locus or continuing track as
   a transformation transition.
3. Distinguish local transformation, displacement, disappearance, global phase
   change, and animation using matched controls and future neutral horizons.
4. Record blocked or no-effect interventions as first-class observations.
5. Deduplicate evidence by root state, action, duration, source track, control,
   and horizon.
6. Store descriptors in the anonymous behavior checkpoint without supplied
   semantics.

Required telemetry:

- `anonymous_object_transition_candidate`;
- `anonymous_object_transition_confirmed`;
- `anonymous_object_transition_rejected`;
- `anonymous_displacement_observed`;
- `anonymous_transformation_observed`;
- `anonymous_intervention_no_effect_observed`;
- source/control/target frames and state aliases;
- persistence horizons and causal confidence.

Tests:

- directional displacement in all four directions;
- blocked directional action;
- button-conditioned appearance transition;
- autonomous transition reproduced by matched `NOOP` and therefore denied
  controller credit;
- local change coincident with global phase change;
- replay deduplication;
- checkpoint migration and frozen-mode behavior.

Native acceptance gate:

- The known Room 3 displacement is encoded as `(1, 0)` without a supplied
  object label.
- A second matching appearance receives a transferable displacement prior.
- A mismatched context remains uncertain rather than inheriting a false rule.

### WP4 — Relational behavior prediction

Priority: high  
Dependencies: WP3  
Primary files: `lolo_agent/entity_behavior.py`, optional new
`lolo_agent/relational_behavior.py`, training/import utilities, tests

Build:

1. Condition predictions on appearance family, action, duration, local
   geometry, controllable-region relation, stable phase signature, and
   neighborhood signature.
2. Preserve exact-context, factored-context, predictive-family, and global
   fallback provenance.
3. Add calibrated outcome probabilities for displacement vector,
   transformation, autonomous motion, no effect, global phase change, and
   causally supported terminal outcomes.
4. Add held-out likelihood, calibration, and abstention metrics.
5. Keep the empirical distribution as the initial implementation; introduce a
   learned relational network only after the empirical baseline and data split
   are reliable.
6. A learned network must consume anonymous embeddings and relations, never
   supplied object IDs.

Required metrics:

- negative log likelihood;
- Brier score per outcome family;
- expected calibration error;
- exact-context coverage;
- fallback coverage;
- abstention rate;
- transfer accuracy across positions, rooms, and animation variants;
- appearance-agnostic and context-agnostic baseline comparisons.

Tests:

- contradictory evidence increases entropy;
- phase-conditioned evidence overrides cross-phase fallback only with support;
- unseen context uses the correct fallback and reports provenance;
- frozen inference creates no new type or count;
- checkpoint content digest is deterministic.

Acceptance gate:

- Held-out prediction beats action-only and appearance-only baselines.
- Known no-effect rules reduce wasted probes without suppressing novel visible
  effects.
- Direct emulator evidence still overrides every prior.

Promotion rule:

- Predictions remain telemetry-only until a native shadow gate passes.
- Then enable bounded curiosity and continuous inert penalties.
- Hazard veto remains disabled until causal precision and fail-open behavior
  pass their separate gate.

### WP5 — Learned controllable-region tracker

Priority: high for strict evaluation; may proceed alongside WP3/WP4  
Dependencies: existing transition data  
Primary files: new `lolo_agent/controllable_tracker.py`,
`lolo_agent/goal_prior.py`, `lolo_agent/unlabeled_entities.py`, training code,
tests

Build:

1. Learn which visual region is consistently controlled by directional
   actions using action-correlated displacement across save-state branches.
2. Track that region through animation, temporary occlusion, disappearance,
   room transitions, and reset.
3. Produce a pixel mask, coarse cell distribution, facing/pose uncertainty,
   and confidence.
4. Replace assisted blue/white player masking in strict object tracking.
5. Keep the current assisted detector as a development baseline and telemetry
   comparator only.

Training data:

- matched directional action and `NOOP` branches;
- no semantic player labels in the strict corpus;
- pseudo-targets from consistent action-correlated optical displacement;
- held-out runs and held-out rooms.

Tests:

- mock sprite with color changes;
- controlled sprite adjacent to same-colored object;
- blocked action changes pose but not position;
- temporary disappearance;
- two independently moving regions where only one is action-correlated;
- frozen inference audit.

Acceptance gate:

- Native held-out tracking agrees with action-counterfactual localization more
  reliably than the present color heuristic.
- Adjacent white objects remain outside the learned controllable mask.
- Strict object tracking no longer imports `PixelHeartGoalPrior` player masks.

### WP6 — Accessibility and reversibility measurement

Priority: high  
Dependencies: WP2 and preferably WP5  
Primary files: new `lolo_agent/accessibility.py`,
`lolo_agent/bidirectional_probe.py`, `lolo_agent/spatial_returnability.py`,
`lolo_agent/neural_planner.py`, tests

Build:

1. From an exact state, run bounded directional reachability search while
   holding the anonymous track configuration fixed where possible.
2. Compute player-region connectivity before and after a verified object
   transition.
3. Record newly reachable and newly unreachable cells and interaction
   frontiers.
4. Measure observed returns with explicit backward search.
5. Treat budget-exhausted non-return as scoped evidence, not proof of
   irreversibility.
6. Learn an accessibility/reversibility predictor only from these verified or
   censored relations.
7. Add track-set and phase conditioning to reachability predictions.

Search constraints:

- use opaque save states only;
- separate source and target branch ownership;
- cap depth, beam, events, and wall time;
- avoid actions predicted to manipulate another track during a pure
  accessibility probe, but log and fail open when classification is unknown;
- never call a state unsolvable solely because the bounded search failed.

Required telemetry:

- `accessibility_probe_started`;
- `accessibility_cell_reached`;
- `accessibility_interaction_frontier_reached`;
- `accessibility_delta_measured`;
- `accessibility_probe_censored`;
- `return_path_observed`;
- `return_budget_exhausted`;
- source/target track-set signatures and phase signatures.

Tests:

- mock corridor opened by displacement;
- corridor closed by displacement;
- reversible detour;
- budget-censored return;
- unrelated animation does not change accessibility;
- a second manipulation invalidates the fixed-layout probe and is reported.

Native acceptance gate:

- Identify one Room 3 manipulation whose post-state changes reachable player
  space or reachable interaction frontiers.
- Reproduce that delta from an archived state.
- Prefer an empirically useful configuration over an equally novel but
  accessibility-neutral configuration in a shadow-policy comparison.

### WP7 — Phase-conditioned mechanics

Priority: high after basic object transitions  
Dependencies: WP3, WP4  
Primary files: new `lolo_agent/phase_model.py`,
`lolo_agent/entity_behavior.py`, `lolo_agent/neural_planner.py`, tests

Build:

1. Learn a stable global phase embedding from regions whose appearance changes
   persistently across multiple disconnected screen areas.
2. Separate local object effects, HUD-like counters, room transitions,
   animation phase, and global behavior phase.
3. Condition anonymous behavior rules on phase embeddings.
4. Link phase transitions to the actions and prior relational states that
   preceded them without naming the trigger.
5. Preserve phase history across save-state archives and episodic resumes.
6. Allow the planner to branch before and after a phase transition to compare
   behavior changes.

Assisted development gate:

- Use a targeted last-heart state to verify that the same anonymous appearance
  can have different posteriors before and after the global visual transition.

Strict gate:

- Recover the phase distinction without using heart or chest detectors.
- Predict a changed anonymous behavior on a separate exact branch.
- Use the prediction in shadow mode to rank a safer or more informative test.

Tests:

- local change is not a phase transition;
- disconnected stable changes qualify only with persistence;
- phase-conditioned contradictory rules remain separate;
- archive restoration preserves the correct phase;
- unseen phase abstains or uses an explicitly reported fallback.

### WP8 — Object-level hypothesis planner

Priority: high after WP6  
Dependencies: WP2, WP4, WP6, WP7  
Primary files: new `lolo_agent/relational_planner.py`,
`lolo_agent/neural_planner.py`, `lolo_agent/memory.py`, tests

Build:

1. Generate abstract hypotheses from observed affordances and learned
   distributions rather than flat controller sequences.
2. Candidate hypotheses include:
   - approach an under-tested track;
   - test an action/context combination;
   - reproduce a known displacement;
   - reproduce a known appearance transition;
   - move toward a newly reachable interaction frontier;
   - preserve or restore a valuable configuration;
   - investigate a phase-conditioned contradiction;
   - attempt an observed return path.
3. Translate a hypothesis into an exact controller-search objective.
4. Use exact save-state search to realize and verify the hypothesis.
5. Replan after every verified transition.
6. Treat model predictions as priors with uncertainty and information value.
7. Store successful hypothesis realizations as learned options with initiation
   and termination conditions derived from relational state.
8. Never store a room-specific controller sequence as a universal mechanic.

Suggested hypothesis score:

```text
verified goal/milestone evidence
+ expected accessibility improvement
+ expected information gain
+ option transfer evidence
+ reversibility confidence
- causal terminal risk
- predicted inert probability
- search cost
- repeated experiment count
```

Every component must be separately logged. Unverified predicted accessibility
must not be scored as if it were an observed improvement.

Tests:

- hypothesis generation from anonymous tracks;
- uncertainty-driven experiment selection;
- known inert action down-ranking;
- exact outcome overriding the prior;
- option applicability across translated layouts;
- no universal macro from one room-specific trajectory;
- bounded queue and deterministic tie-breaking.

Acceptance gate:

1. The agent proposes that changing an anonymous object configuration may
   improve future access.
2. It finds the required controller sequence through exact search.
3. It verifies the accessibility delta.
4. It retains the new configuration across a planning cycle.
5. It uses the result to reach a subsequent positive visual milestone.

This is the decisive milestone for the next phase of the project.

### WP9 — Reward and value learning

Priority: after accessibility measurement exists  
Dependencies: WP6, WP8  
Primary files: `lolo_agent/neural_planner.py`, new value module if warranted,
`lolo_agent/reward_audit.py`, tests

Assisted development reward remains:

- detected heart collection: positive;
- detected all-hearts transition: larger positive;
- detected treasure/room completion: largest positive;
- confirmed life loss: negative.

Add only evidence-backed shaping:

- information gain from a discriminating counterfactual;
- newly verified reachable cells or interaction frontiers;
- newly verified access to a previously observed positive-milestone region;
- accurate behavior prediction followed by exact verification;
- preservation of valuable reversible alternatives;
- negative value for causally verified terminal risk;
- bounded negative value for empirically verified loss of accessibility.

Do not add:

- rewards for a supplied object identity;
- unconditional reward for moving any object;
- raw screen-change reward as progress;
- room-specific coordinates;
- hand-authored safe paths;
- penalties for bounded non-return presented as certain irreversibility.

Strict-track replacement:

1. Learn visual milestone representations from repeated transition structure,
   control loss/recovery, scene changes, and delayed consequences.
2. Use evaluator room completion only for evaluation metrics, not as an object
   rule.
3. Compare strict intrinsic/value learning against the assisted reward track as
   an ablation.

Acceptance gate:

- A preparation action with delayed verified benefit receives credit.
- An equally novel but accessibility-neutral action does not.
- A harmful accessibility reduction is penalized only after sufficient
  evidence.
- Reward audit attributes every nonzero component to telemetry evidence.

### WP10 — Unrecoverable-state and hazard learning

Priority: after later rooms expose meaningful deaths  
Dependencies: WP4, WP6, WP7  
Primary files: `lolo_agent/entity_behavior.py`,
`lolo_agent/spatial_returnability.py`, planner and tests

Build:

1. Distinguish empirical terminal correlation from intervention-attributed
   hazard evidence.
2. Learn delayed hazards from matched factual/control horizons.
3. Learn budget-scoped returnability and accessibility-loss predictions.
4. Preserve uncertainty and censoring.
5. Promote predictions through stages:
   telemetry only, shadow ranking, soft penalty, then conservative veto.
6. A veto must require context-matched causal support and fail open if all
   verified choices are rejected.

Acceptance gate:

- High precision on native held-out causal hazard contrasts.
- No authority from passive terminal correlation alone.
- No false certainty from search-budget exhaustion.
- Correct frozen-evaluation audit.

### WP11 — Room curriculum and continuous research loop

Priority: begins after WP0; broadens after WP8 gate  
Dependencies: WP0 and relevant capability gates  
Primary files: `lolo_agent/research_cycle.py`, experiment configs, docs

For each training/development room:

1. Run a bounded inspection cycle.
2. Record new appearances, contexts, phases, transitions, and failure modes.
3. State the smallest falsifiable capability hypothesis.
4. Create a targeted save-state gate.
5. Run a cheap smoke test.
6. Run one bounded native measurement.
7. Reflect with `continue`, `revise`, or `stop`.
8. Implement only evidence-supported changes.
9. Run regressions and the targeted native gate.
10. Commit and push before starting the next capability cycle.

Room progression should not be a blind sequence of longer searches. A room may
be paused while a reusable missing capability is developed. Once the
object-level planning gate passes, resume the current Room 3 development state,
collect the next milestone, and continue until the room clears. Repeat the
capability audit before advancing.

Campaign stopping rules:

- stop after a declared wall-clock or telemetry ceiling;
- stop when the target evidence appears;
- stop when the leading hypothesis is falsified;
- stop after two bounded runs reproduce the same failure without new evidence;
- do not increase beam or depth until telemetry identifies search coverage,
  rather than representation, as the active bottleneck.

### WP12 — Withheld and sequel evaluation

Priority: after training curriculum and strict-component gates  
Dependencies: WP0 through strict WP9 gates  
Primary files: evaluation configs, frozen runner, reports

Before evaluation:

1. Freeze neural parameters, anonymous appearance prototypes, behavior rules,
   relational model, phase model, controllable tracker, value model, and any
   learned option library.
2. Record all digests.
3. Disable persistent writes.
4. Start with empty room-local temporary memory.
5. Confirm no withheld-room save states or telemetry are present in training
   corpora.

Withheld *Lolo 1* metrics:

- rooms solved;
- completion rate by fixed branch and wall-clock budget;
- attempts and life losses;
- exact emulator branches;
- learned-option reuse;
- behavior prediction calibration;
- track continuity;
- phase prediction accuracy;
- accessibility prediction accuracy;
- parameter digest audit.

*Lolo 2* metrics:

- rooms entered and solved in order;
- zero-shot behavior prediction coverage;
- unfamiliar appearance abstention;
- reused versus novel temporary hypotheses;
- branch and wall-clock efficiency;
- failures classified by perception, tracking, model, planning, search, or
  truly unfamiliar mechanics;
- unchanged persistent digests.

Required ablations:

- flat exact search versus relational planner;
- no persistent behavior model;
- no accessibility model;
- no phase conditioning;
- strict versus assisted reward track;
- frozen model versus illegal-update sentinel run that must fail.

## 8. Immediate build backlog for coding agents

The following task cards are intended to be independently assignable. Agents
must not edit overlapping planner regions concurrently without coordination.

### Task A — Evaluation partition manifest

Owner surface:

- new config schema and loader;
- `research_cycle.py` integration;
- focused tests and protocol documentation.

Deliverables:

- immutable partition manifest;
- update-authority check;
- strict/assisted provenance fields;
- frozen artifact inventory and digest audit;
- tests for prohibited updates.

May run in parallel with Task B.

### Task B — Object-track module extraction

Owner surface:

- new `object_tracks.py`;
- migration of root object-state parsing and serialization;
- focused track tests.

Deliverables:

- pure dataclasses and signatures;
- archive conversion including legacy metadata;
- no planner behavior change;
- compatibility tests for `v318` and `v321` metadata shapes.

Must land before Tasks C and D.

### Task C — Multi-track correspondence engine

Owner surface:

- `object_tracks.py` correspondence and ambiguity logic;
- `unlabeled_entities.py` feature access;
- new focused tests.

Deliverables:

- repeated-appearance matching;
- occlusion/reacquisition;
- bounded ambiguity hypotheses;
- deterministic track-set signatures;
- telemetry payload helpers.

Depends on Task B.

### Task D — Planner node/archive integration

Owner surface:

- `_HumanPriorOptionNode` integration;
- archive construction, restoration, and beam diversity;
- planner tests.

Deliverables:

- tuple of tracks on search roots and descendants;
- track-set propagation;
- archive round trips;
- world-state reserve keyed by relational track state;
- no loss of confirmed identity during transient interactions.

Depends on Tasks B and C. Coordinate carefully because
`neural_planner.py` is a high-conflict file.

### Task E — Transition descriptor extension

Owner surface:

- `entity_behavior.py` schema migration;
- object-transition descriptors;
- checkpoint and behavior tests.

Deliverables:

- explicit displacement vectors;
- appearance transitions;
- blocked/no-effect evidence;
- provenance and deduplication;
- backward-compatible checkpoint loading.

Can begin after Task B and proceed mostly independently of Task D.

### Task F — Object-track telemetry and summaries

Owner surface:

- `run_logging.py`, `log_summary.py`, `replay.py`, telemetry docs, tests.

Deliverables:

- track lifecycle events;
- transition events;
- summary counts and per-track histories;
- replay annotations sufficient to visualize track identities and confidence;
- no duplicate frame storage.

Depends on the data contracts from Task B, but can use fixture payloads before
planner integration.

### Task G — Accessibility probe prototype

Owner surface:

- new `accessibility.py`;
- extensions to `bidirectional_probe.py`;
- mock environments and tests.

Deliverables:

- fixed-layout directional reachability;
- before/after deltas;
- budget-censored return evidence;
- no policy authority initially.

May begin with mock track sets while Tasks C and D are in progress.

### Task H — Learned controllable tracker research spike

Owner surface:

- new tracker module and offline dataset builder;
- no planner authority.

Deliverables:

- action-correlated localization dataset;
- simple baseline model;
- held-out comparison with assisted detector;
- failure analysis around same-colored adjacent objects.

May proceed in parallel. It must remain telemetry-only until its gate passes.

### Task I — Native two-manipulation gate

Owner surface:

- experiment config, bounded run, telemetry audit, reflection document.

Deliverables:

- one falsifiable hypothesis;
- fixed save-state source and content digests;
- branch/event/time ceilings;
- track continuity report;
- `continue`, `revise`, or `stop` decision.

Depends on Tasks C, D, E, and F.

## 9. Recommended agent coordination

When multiple coding agents work concurrently:

1. Assign file ownership explicitly.
2. Avoid concurrent edits to `neural_planner.py`.
3. Land pure modules and tests before planner integration.
4. Require each task to report:
   - files changed;
   - public interfaces added;
   - tests run;
   - unresolved assumptions;
   - telemetry changes;
   - migration concerns.
5. Rebase or merge only after checking the user's existing uncommitted work.
6. Never modify or delete `tmp/`; it contains user-owned, untracked material.
7. Use `apply_patch` for hand edits and preserve unrelated changes.
8. Run focused tests first, then:

```bash
.venv/bin/python -m unittest discover -s tests
git diff --check
```

9. A planner integration agent owns the final full-suite run and native gate.
10. Commit messages should describe the learned capability, not merely the
    refactor.

## 10. Telemetry expansion plan

Add enough data to reconstruct not only what action occurred, but what the
agent believed and why it selected that experiment.

Per frame/state:

- phase signature and confidence;
- controllable-region mask digest, cell distribution, and confidence;
- track-set signature;
- every active track's position, appearance, type, context, and confidence;
- ambiguity hypotheses and pruning.

Per verified branch:

- parent and target track sets;
- matched correspondence;
- displacement and appearance transitions;
- factual and control state aliases;
- model prediction before execution;
- observed outcome;
- prediction error and calibration bucket;
- accessibility delta;
- returnability status;
- hypothesis ID and score components.

Per archive:

- full relational state;
- parent archive/state ancestry;
- confirmed versus transient interaction identity;
- persistence evidence;
- source of every restored field: serialized, reconstructed, or newly
  observed;
- exact state and frame digests.

Per decision:

- considered object-level hypotheses;
- exact-search realization result;
- selected and rejected reasons;
- predicted information gain;
- verified accessibility contribution;
- terminal-risk provenance;
- learned option initiation/termination match;
- persistent parameter update authority.

Per run summary:

- tracks created, retained, lost, split, merged, and reacquired;
- confirmed displacements and transformations;
- prediction calibration by provenance;
- accessibility gains and losses;
- observed returns and censored non-returns;
- hypotheses proposed, realized, confirmed, and falsified;
- learned options created and reused;
- attempts, milestones, room transitions, life changes, branch count,
  wall-clock time, and estimated cost;
- strict/assisted partition and before/after persistent digests.

Replay visualization must eventually support:

- accelerated committed trajectory;
- full rejected-branch tree;
- track overlays with stable colors per temporary track ID;
- displacement arrows and transformation markers;
- controllable-region and accessibility overlays;
- model prediction versus observed outcome;
- phase timeline;
- archive restore ancestry;
- reward/value component timeline;
- per-state attempts and visit counts.

## 11. Testing pyramid

### 11.1 Pure unit tests

Use synthetic frames and small deterministic fixtures for:

- feature extraction;
- player-mask exclusion;
- track correspondence;
- ambiguity;
- displacement;
- transformations;
- phase signatures;
- behavior distributions;
- accessibility graph deltas;
- serialization and migrations;
- frozen-update rejection.

### 11.2 Mock pixel environments

Extend mock environments to test only pixel-observed consequences:

- movable anonymous patches;
- repeated identical patches;
- blocked motion;
- appearance transformation;
- phase-dependent autonomous motion;
- delayed terminal transitions;
- reversible and irreversible-looking layouts;
- corridors opened or closed by manipulation.

The agent must not import mock symbolic state or success predicates.

### 11.3 Logged-frame regression tests

Use content-digested frame fixtures derived from legal local runs for:

- known push-state masking;
- legacy archive reconstruction;
- animation correspondence;
- source/destination appearance matching;
- phase-change candidates.

Fixtures must not encode or expose ROM RAM.

### 11.4 Native bounded gates

Every promoted capability requires:

- exact run ID;
- input digests;
- fixed source state;
- declared depth/beam/event/time ceilings;
- raw telemetry audit;
- success and falsification conditions;
- reflection document;
- no unbounded follow-up run.

### 11.5 Frozen evaluation tests

Test that:

- every persistent artifact is read-only;
- unknown appearances do not create types;
- temporary tracks and room memory remain allowed;
- process exit preserves all before/after digests;
- telemetry clearly distinguishes temporary adaptation from persistent
  learning.

## 12. Experiment sequence after implementation

### Gate 1 — Multi-track persistence

Hypothesis:

> Multiple repeated anonymous appearances can be tracked through player motion,
> animation, one controlled displacement, archive persistence, and process
> restoration without supplied identities.

Success:

- at least two tracks retained;
- known displacement correct;
- archive round-trip signature stable;
- ambiguity reported rather than silently misassigned.

Falsification:

- correspondence follows appearance alone and swaps identical tracks;
- player pixels become an object track;
- restoration changes track identity or state.

### Gate 2 — Two consecutive manipulations

Hypothesis:

> The agent can preserve the first verified object transition while discovering
> and representing a second transition.

Success:

- two transitions have distinct source/target track evidence;
- first transition survives descendants and resume;
- complete configuration is replayable.

### Gate 3 — Accessibility consequence

Hypothesis:

> At least one verified manipulation changes bounded controllable-region
> accessibility in a reproducible way.

Success:

- before/after delta;
- repeatable from archive;
- neutral control lacks the same delta;
- uncertainty remains scoped to probe budget.

### Gate 4 — Deliberate preparation

Hypothesis:

> Relational planning prefers a verified accessibility-improving configuration
> over an equally novel neutral configuration and uses it to reach a later
> milestone.

Success:

- hypothesis logged before execution;
- controller realization discovered, not supplied;
- accessibility verified;
- subsequent positive milestone reached;
- counterfactual neutral configuration performs worse within the same budget.

### Gate 5 — Phase-conditioned behavior

Hypothesis:

> A recurring anonymous appearance has a measurably different outcome
> distribution after a learned global visual phase transition.

Success:

- phase learned from pixels;
- context-specific posterior supported;
- changed behavior predicted on a separate branch;
- shadow policy ranking improves without a supplied rule.

### Gate 6 — Room 3 completion

Only after Gates 1 through 4 pass, run bounded room-level cycles. Completion is
not the only metric. Report which learned tracks, transitions, accessibility
deltas, phase contexts, and options contributed.

### Gate 7 — Training-room transfer

Test recurring anonymous behavior and options in a different training room.
Reject a claim of transfer if the behavior depends on an absolute coordinate
or exact full-screen hash.

### Gate 8 — Withheld and sequel freeze

Run only after partition and frozen-artifact audits pass. No implementation
changes may be justified using hidden sequel results and then reported as the
same untouched evaluation.

## 13. Cost and compute policy

Stay on the M5 for current emulator branching, tests, telemetry analysis, and
small models. Existing measurements found no justification for RunPod for the
current bottleneck.

Use RunPod only when a benchmark shows a compute-bound training workload with
a lower measured cost per useful update or per validated experiment. Before
paid execution:

1. Run a local correctness smoke test.
2. Declare hourly price and storage costs.
3. Set wall-clock, event, per-cycle, and campaign-dollar ceilings.
4. Use `lolo-research-cycle` and the outer RunPod watchdog.
5. Stop the Pod after the cycle.
6. Verify durable artifacts and provider state.

Default research behavior:

- one targeted native run at a time;
- no automatic depth escalation;
- no repeated run after identical negative evidence;
- reflect before another expensive measurement;
- prefer offline telemetry analysis and mock gates when they answer the same
  question.

## 14. Risk register

### Risk: object tracker becomes a collection of game-specific heuristics

Mitigation:

- anonymous fields only;
- generic pixel correspondence tests;
- cross-room and translated-layout gates;
- strict review for names, coordinates, or sprite-specific colors.

### Risk: identical appearances swap identity

Mitigation:

- spatial-temporal correspondence;
- ambiguity hypotheses;
- no forced unique assignment;
- archive ancestry and exact transition evidence.

### Risk: phase model memorizes rooms

Mitigation:

- factored stable-region changes;
- translation-tolerant representation;
- held-out rooms;
- compare against full-screen hash baseline.

### Risk: bounded non-return is mistaken for irreversibility

Mitigation:

- censored labels;
- recorded budgets;
- observed returns as the only certain positive;
- no hard veto from budget exhaustion.

### Risk: assisted labels leak into strict claims

Mitigation:

- separate manifests and datasets;
- loader rejection;
- artifact provenance audits;
- strict replacement of assisted player/goal/life components.

### Risk: planner complexity grows inside one file

Mitigation:

- extract object tracks, accessibility, phase, and relational planning into
  modules before integration;
- pure contracts and focused tests;
- explicit file ownership for coding agents.

### Risk: broader search hides representation failures

Mitigation:

- targeted gates;
- compare detector acceptance and prediction calibration before depth;
- increase budget only after evidence of a search-coverage bottleneck.

### Risk: world model is treated as an exact simulator

Mitigation:

- uncertainty-aware proposals;
- exact emulator verification;
- measured calibration;
- direct outcome override.

### Risk: demonstration or solution contamination

Mitigation:

- no YouTube or solution data;
- no demonstrations in the strict project;
- immutable room split;
- provenance checks for every training transition.

## 15. Definition of completion

### Near-term capability completion

The next architecture milestone is complete when the agent:

1. tracks multiple anonymous objects;
2. learns an explicit transition for at least one recurring appearance;
3. predicts that a manipulation may alter future accessibility;
4. discovers and executes the required controller sequence;
5. verifies the accessibility change through save-state branches;
6. preserves the new configuration across a planning cycle; and
7. reaches a subsequent positive visual milestone because of the preparation.

### *Lolo 1* completion

- All required rooms completed under a documented training protocol.
- Held-out rooms evaluated separately with frozen persistent parameters.
- Complete action, branch, attempt, archive, learning, and replay telemetry.
- No RAM inspection, supplied solution, object rule, or demonstration.

### Generalization completion

- Persistent artifacts frozen before first *Lolo 2* exposure.
- No persistent updates during sequel evaluation.
- Temporary planning and save-state exploration allowed and logged.
- Results reported by room, budget, attempt, failure class, and unchanged
  artifact digest.

## 16. Next action

Superseded in part by the 2026-08-16 amendment (§17). WP0 and WP1 are
landed (commits `b95b68f`, `236ea65`). The original sequence for reference:

1. Land WP0's immutable evaluation partition and artifact inventory. [DONE]
2. Land WP1's `object_tracks.py` extraction without behavior changes. [DONE]
3. Implement WP2 multi-track correspondence.
4. Extend WP3 displacement/transformation descriptors.
5. Integrate relational track sets into planner nodes, archives, and telemetry.
6. Run the bounded two-manipulation native gate.
7. Reflect on the evidence before building policy authority.
8. Implement the WP6 accessibility prototype using mock environments while
   native tracking work proceeds.

## 17. Amendment — 2026-08-16 (evidence: learnings §4.26–§4.30)

Basis: the certified paired-probe series v322–v325
(`docs/paired-accessibility-probe-2026-08-16.md`,
`docs/object-removed-probe-2026-08-16.md`) and the adversarial direction
review (`docs/direction-review-2026-08-16.md`). Adopted under Todd's
delegated authority of 2026-08-16; every change is reversible and each
carries its evidence.

1. **Bottleneck reframed: valuation, not discovery.** Ordinary search
   discovers useful manipulations spontaneously (11+ confirmed across
   v322–v325, including the band-opening removal chain), while the planner
   preserved a certified-neutral configuration for four run-generations.
   **WP8-lite precedes WP8:** a verified-accessibility preference term in
   the existing archive/restore-selection seams, promoted only through a
   preregistered matched-budget paired ablation (mixed result = FAIL).
   `relational_planner.py` follows the ablation's outcome. The concrete
   Gate 4 chain is named: entity removal → east region → `(12,11)`-class
   hearts (the chain completed non-deliberately in v325).
2. **WP3 priority shift: transformation/removal chains.** The Room 3
   door-opener is transform-in-place → displaced-transformed-object →
   expulsion. WP3 gains a removal-chain native gate alongside the
   displacement gate; button-conditioned transformation posteriors are the
   highest-value rule family (the type-7 result shows the behavior model
   supports them).
3. **WP2 contract hardened:** track state must be endpoint-relative, never
   accumulated history (five of six accumulated cells were stale at v324
   d7 — learnings §4.29). HUD regions and autonomous patrol must be
   excluded from manipulation credit by measurement. Every archive class
   that can reseed a search root must carry the track block (causal-archive
   fix in progress).
4. **WP6 reframed as productization.** Certified-hold paired configuration
   probes are a proven ~25-minute instrument; `accessibility.py` and the
   §6.5 `AccessibilityDelta` contract are built around certified holds.
   Gate 3 is substantially met on Room 3 (7 → 24 certified cells);
   formal closure requires one repetition from a fresh restore.
   [GATE 3 CLOSED 2026-08-16 on the assisted track: v326 reproduced the
   24-cell envelope at Jaccard 1.0; strict-track re-measurement still
   gated on WP5.]
5. **Risk register addition:** a confirmed manipulation is not progress —
   the v318 push was preserved across four generations and is certified
   accessibility-neutral. Preservation priority must be coupled to
   measured consequence.
6. **§13 economics update:** preregistered native paired probes cost
   ~25–30 minutes on the M5. Prefer smaller, more frequent certified
   measurements over large campaigns.
7. **Direction-review amendments A–E adopted** as the operating plan
   (measure-early, WP5 mechanized in parallel, strict-lineage linter +
   preregistration addendum, WP9a offline spike, WP7 off Gate 4's critical
   path). WP5 remains required for the strict headline claim — all
   v322–v325 evidence is assisted-track. The `configs/`
   evaluation-partition allocation stands as shipped.

## 18. Amendment — 2026-08-17 (evidence: learnings §4.43, mining §11)

1. **Accessibility preference is near-redundant as a restore ranker.**
   Zero score-conflicts across 74 decision points in the v322–v328
   corpus: wherever a certified-improving candidate existed, the
   baseline novelty/coverage scorer already preferred it. The §17 item-1
   framing ("the planner preserved a neutral configuration while the
   valuable one sat unvalued") is corrected — the existing machinery
   found the removal twice unaided. Accessibility's load-bearing role is
   therefore as a **hypothesis generator and chain justifier**, not a
   single-decision ranker.
2. **Gate 4 re-centred on chaining.** The discriminating question is not
   "does it prefer the right configuration" (satisfied incidentally) but
   "can it sustain a preparation across decisions to reach a milestone
   incidental behavior never reaches". E1's discriminator is the
   `(12,11)` heart: inside the certified envelope, never collected
   in-window by v324/v327/v328.
3. **Gate 6 requires a second manipulation, merging with Gate 2.** The
   remaining Room 3 hearts `(8,4)` and `(9,12)` lie OUTSIDE the 24-cell
   envelope the removal opens. Room 3 completion therefore needs a
   chained second configuration change — exactly Gate 2's "two
   consecutive manipulations". Plan them as one experiment line.
4. **Gate design is a first-class deliverable.** Every failure of
   2026-08-16/17 was caught by measurement quality, and several were
   measurement defects: mask-irrelevant bits, a class-mix-sensitive
   false-positive bound falsified at design time, replication-vs-function
   confusion, ablation roots without conflict, a window semantics that
   would have self-voided. Budget design effort for a gate equal to the
   capability it gates; require of every gate an instrument that can
   contradict it.

## 19. Amendment — 2026-08-17 evening (evidence: learnings §4.45)

**WP8 needs schedule authority, not just scoring authority.** E1 failed
with an architectural cause: the relational planner can rank what an
option search offers, but cannot cause a search, and has no lever on the
decisions that commit without searching (which is most of them — the
exploit held authority across five consecutive search-free decisions).
This supersedes §17 item 1's "wire a preference term" framing entirely:
preference wiring is now twice-measured as insufficient (§4.43 restore
scalar, §4.45 hypothesis reserve).

Consequences for the plan:

1. WP8's next increment is **objective-driven search scheduling** — an
   active hypothesis may request an exact search, and/or express its
   target through the existing control-frontier/navigation machinery —
   preregistered on its own bits, with the same off/telemetry/selection
   gating and invariance discipline.
2. Gate 4's remaining open criteria are unchanged; what changed is the
   mechanism believed capable of closing them.
3. Roadmap §7 WP8's implicit assumption — that hypotheses realize
   through exact search — must state that the planner controls *when*
   search happens, or the hypothesis layer is inert by construction.

## 20. Amendment — 2026-08-17 night (evidence: learnings §4.50)

**Steering is the wrong shape of intervention; closing is the right one.**
E3 showed the navigation mechanism working exactly as designed (7 of 20
instants changed behavior) and making the outcome *worse*: the treatment's
best approach was distance 3 against the control's 1. Cause: the
excursions away from a target are what deposit the archive ladder later
progress climbs — steering narrowed archive geography from columns 6–12
to 6–8 and every subsequent restore ratcheted westward.

Consequences:

1. **§4.7 generalizes and hardens.** It is not "distance rewards fail" but
   "any consistent proximity preference fails" — the tie-break-not-reward
   care taken in the E3 mechanism did not save it. Add to the §5
   invariants: an intervention that narrows exploration must prove it does
   not starve the supply that later progress consumes.
2. **WP8's intervention model changes**: hypotheses should not steer
   trajectories. They should (a) leave exploration untouched and (b)
   intervene at *closing instants* — the moments when exploration has
   already delivered the opportunity and the incumbent scorer would
   discard it. E5 tests exactly this (S2-only ablation).
3. Three levers are now measured: restore preference (redundant),
   search-time reserve (no searches to ride), commit steering (starves
   supply). The surviving hypothesis is that the planner needs *no new
   preference at all* — only a veto on discarding a certified-adjacent
   position.

## 21. Amendment — 2026-08-17 late (evidence: learnings §4.53)

**Gate 4's obstacle is clock misalignment, not scoring.** Four measured
failures now share one class: the hypothesis layer's lifecycle is driven
by its own budget counters, while the moments that decide outcomes are
generated by the incumbent's archive and stagnation machinery. Alive but
no search to ride (§4.45); steering when it should not (§4.50); able only
to rank candidates that exist (§4.51); dead three events before the
decisive instant (§4.53).

Consequences:

1. **Stop adding seams.** The next WP8 step is a design pass on
   hypothesis lifecycle: liveness driven by incumbent events (approach,
   contested restore, milestone adjacency) rather than decision-count
   budgets; and an explicit ruling on whether a mechanism may read
   certified cells from the record store rather than from a live
   hypothesis. Both are intervention-class questions, not tunings.
2. **The narrowing is real.** Record it as progress: the failure point
   moved from "no lever exists" to a three-event gap, with supply
   preserved, geography intact, and the closing refusal firing correctly
   at the contested restore. The remaining defect is small and named.
3. **Invariant added to §5**: a capability layer bolted onto the
   incumbent planner must state which of the incumbent's events drive its
   state transitions. A layer with an independent clock will
   misalign — four times measured.

## 22. Amendment — 2026-08-18 (evidence: learnings §4.54) — REWRITE TRIGGER A FIRED

E7 fired the preregistered rewrite trigger: bits 1–3 PASS, bit 4 FAIL,
`void: false`. The deposit fired at the decisive instant, the restore key
selected it, the agent deliberately returned itself to one cell from the
certified milestone — and never stepped onto it. The failure is now
*inside* the P5 commit ladder, which has no tier referencing a target
cell.

Consequences, binding:

1. **The bolt-on program is closed.** Five levers, five distinct named
   mechanisms (§4.43 redundancy, §4.45/§4.46 no opportunity, §4.50 supply
   starvation, §4.51 candidate absent, §4.53 objective absent), and now a
   missing actuator. No sixth seam.
2. **WP8's next deliverable is a planner rewrite**, scoped as narrowly as
   the evidence licenses: a commit-ladder tier able to take the final step
   onto an adjacent certified milestone under hold, with the same
   authority gating, invariance discipline, and preregistration the seams
   used. E8 (R3) is superseded as a priority.
3. **Q3 ruling scope-corrected.** The unscored attribution arm was
   trajectory-identical to the treatment (`deposit_events_without_a_
   hypothesis: 14`, identical 85,601 event counts). The ruling's
   principle stands — a standing rule cannot decline, so it cannot
   evidence choosing — but its empirical support at this root does not.
   No Gate 4 claim may cite E7 as evidence that hypothesis-driven
   planning produced the behavior.
4. **§5 invariant added**: before building a capability layer, identify
   the actuator that will execute its final step, and verify that
   actuator exists. Four levers were built above an actuator that was
   never checked for.

## 23. Amendment — 2026-08-18 — §22 consequence 2 SUPERSEDED (evidence: learnings §4.55)

§22 fired trigger A correctly but attached the wrong cause. Recon found
the commit ladder was **never entered** at the decisive instant
(`branches_examined: 0`, zero verified branches at v341 d17/d18 —
`decide()` returns early via `_restore_if_stagnant()` before
`planner.plan(...)`), that the ladder already *has* a target-cell tier
(refuted separately by §4.50), and that a new adjacency-gated tier would
have **zero opportunities** across ten runs — i.e. trigger C, not a fix.

Superseding consequence 2:

1. **No ladder rewrite.** The next change is **R-A**, outside the ladder:
   decline a stagnation restore when the current position already
   satisfies the S3 deposit predicate at distance exactly 1, falling
   through to expansion. No new tier, term, or weight; it reuses the
   deposit view E7 verified.
2. The decisive defect is now named precisely: at d17 the restore
   selected the state the agent already occupied — a positional no-op
   that consumed the decision and then removed the `(12,10)` candidate
   from the archive. The closing mechanism won its contest and spent the
   win standing still.
3. **§5 invariant added** (replacing §22 item 4's weaker form): before
   building where a failure "must" live, verify the code path was
   reached. One recon pass falsified an interpretation that would have
   cost an experiment.

## 24. Amendment — 2026-08-18 — E8 PASS; Gate 4 outcome criterion met (evidence: learnings §4.56)

E8 passed all five preregistered bits with `void: false`. At d17 the
treatment stepped `(12,10) → (12,11)` and collected `(192,176)` — a cell
nineteen prior runs at this root never reached. The fix was R-A alone:
decline a stagnation restore when the agent already stands adjacent to a
certified milestone, so expansion generates a candidate set. The missing
primitive was one 16-frame `down`; it was never absent and never
mis-ranked, only never generated.

Standing:

1. **Gate 4's outcome criterion is met on the assisted track.** A
   prepared configuration was established, held, and converted into a
   subsequent milestone no incidental run reaches. Its *deliberateness*
   criterion is **NOT met**: the attribution arm (v345) came back
   `trajectory_identical_to_treatment: true` and collected at d17 as
   well, so a standing rule reproduces the behaviour exactly. Attribution
   has now failed in both experiments that tested it (E7, E8). **No Gate
   4 claim may cite E8 as evidence of hypothesis-driven choice.** The
   defensible claim is capability only: the agent can complete a step it
   prepared for once the restore bifurcation stops consuming that
   decision.
2. **`(12,11)` retires as a discriminator**; Gate 4 work moves to
   `(8,4)`/`(9,12)`, which require a **second manipulation** — merging
   with Gate 2 per §18 item 3. New preregistration and fresh
   control-never-does-it evidence required.
3. **R-B is unshipped-in-effect and should be REMOVED**: it changed
   nothing at 15 of 15 expansion decisions. An R-A-only confirmation
   costs nothing (`decline_restore` already exists). Carrying a lever
   with no measured effect violates the counterfactual discipline that
   caught it.
5. **The Q3 ruling now has two failed attribution tests.** The principle
   (a standing rule cannot decline, so it cannot evidence choosing)
   stands as reasoning, but nothing measured supports preferring
   hypothesis-scoped liveness at this root. Any future design that pays
   complexity for hypothesis scoping must first show an instance where
   the standing rule and the hypothesis-scoped version *diverge* — that
   divergence is now a precondition, not an assumption.
4. **§5 invariant added**: every lever ships with a counterfactual
   instrument recording what the incumbent would have done. R-B would
   have been wrongly credited without one.

## 25. Amendment — 2026-08-18 — §18 item 3 FALSIFIED (evidence: learnings §4.57)

**"The remaining hearts require a second manipulation" was a budget
artifact, not a fact about the room.** Three independent methods put both
remaining hearts inside a 67-cell region reachable from `(12,11)` in the
post-first-manipulation configuration; v303 already holds an
emulator-verified 24-action branch that collects `(9,12)` with zero life
loss; and v344's own d19 search reached five steps from `(8,4)` at branch
depth exactly 12 — the configured ceiling — before a stagnation restore
pulled it back. The certified 24-cell envelope measures what a
depth-12/beam-128 search retained from one root. It is a lower bound, not
a wall.

Binding consequences:

1. **§18 item 3 and §24 item 2 are superseded.** Gate 6 **un-merges**
   from Gate 2. No second-manipulation experiment is justified until a
   reachability re-measurement says one is needed.
2. **Next experiment is R1** — reachability at declared larger depth with
   a depth-12 control. It is cheap and can void everything downstream.
   Depth 12 has been silently load-bearing across the whole Gate-4
   family; the runs that solved this region used depth 24–36.
3. **Discriminator is `(9,12)`** (2 of 353 runs, never on a committed
   decision) if one is still needed — never `(8,4)`, which 44 of 353 runs
   collected.
4. **New §5 invariant**: "outside the certified envelope" never means
   "unreachable". A certified record is a lower bound from one budget and
   one root. Before claiming X needs a new capability, rule out that X
   needs a larger budget — and populate `certified_open_frontiers`, which
   exists to record exactly that distinction and has been empty in every
   record written so far.
5. **Room 3 completion is not just hearts**: `human_prior_chest_obtained`
   is `False` in every run examined, including one resumed from an
   all-hearts state. Unmeasured, and required for Gate 6.

### §5 invariant consolidation (housekeeping)

Amendments §20–§25 each declared a §5 invariant that was never merged
into §5's numbered list. They are, in force and by amendment:

- §20 — an intervention that narrows exploration must prove it does not
  starve the supply later progress consumes.
- §21 — a capability layer bolted onto the incumbent planner must state
  which of the incumbent's events drive its state transitions.
- §23 — before building where a failure "must" live, verify the code
  path was reached.
- §24 — every lever ships with a counterfactual instrument recording
  what the incumbent would have done; and measured divergence is a
  precondition for paying complexity for hypothesis scoping.
- §25 — "outside the certified envelope" never means "unreachable".

Amended next sequence: (a) Gate 3 repetition run; (b) WP8-lite preference
ablation; (c) WP2 multi-track correspondence with the endpoint-relative
contract; (d) WP3 including the removal-chain gate; (e) WP5 spike and
strict-lineage linter in parallel; (f) reflect and amend again.

Status updates (2026-08-16 evening): (a) DONE — Gate 3 closed on the
assisted track (learnings §4.28-note in the probe doc). (c) WP2-lite
correspondence engine landed (`befd629`). (e) WP5 pipeline landed and
iterating: labels → tracker v2 (gates passed) → substitution replay
no-promote (§4.31) → OOD gap quantified (§4.32) → ratified
strict_from_assisted_state collection (§ recon doc) → tier-2 corpus →
tracker v3 training. **WP5 CAMPAIGN COMPLETE (2026-08-17): PROMOTE-to-shadow** — the learned
masking convention passes the functional gate on every axis and every
corpus, exceeding the incumbent on stability, preservation, and in-place
detection (learnings §4.42; six-gate chronicle in the WP5 tracker doc).
Shadow wiring queues behind the planner-file release. **WP9 step 1
demoted to representation rethink after three falsifications** (§4.33,
§4.36→§4.41): delayed-divergence valence validated; the event unit
rebuilds on object-level tracks after WP2/WP3 integration. Original
note: **WP9 step 1 FALSIFIED as written** (§4.33) —
redesign requirements: censoring semantics for mixed changed-cell sets,
multi-decision successor windows, delayed-divergence valence in place of
reversion (terminal commits are action-independent: matched NOOP controls
also die). WP8-lite module + preregistered ablation design landed
(`25a8eda`); the planner seam patch is prepared and waits on file
ownership. WP6a instrument productized (`33bbeb7`).

Do not begin with a larger beam, a new reward weight, or a long room-clearing
run. The next decisive result must demonstrate that the agent can represent
multiple object changes and measure how one of those changes affects future
access.
