# Lolo agent research learnings and negative-results log

Status: living research record  
Last updated: 2026-08-16  
Companion plan: `docs/roadmap.md`

## 1. Purpose

This document records what the project has tried, what did not work well, what
was falsified, what remains merely unproven, and how each result changed the
plan. Its purpose is to prevent future coding agents from repeating expensive
experiments or treating a previously rejected idea as established progress.

The raw event logs and experiment-specific documents remain the source of
truth. This file is the durable synthesis and decision history.

## 2. How to interpret this file

Conclusions use four evidence levels:

- **Falsified at the measured gate**: the stated hypothesis failed under the
  documented state and budget. It may be reconsidered only with a materially
  different mechanism or falsifiable hypothesis.
- **Negative result**: the mechanism ran correctly but did not improve the
  target outcome.
- **Engineering defect**: the experiment exposed an implementation or
  telemetry error; conclusions drawn before the correction are invalid.
- **Not yet demonstrated**: a component passed a narrower gate but has not
  shown room completion or generalization.

A finite search failure is never described as proof that a puzzle state is
globally unsolvable. It is evidence scoped to its exact state, model, search
budget, and controller-edge set.

## 3. Executive summary

The strongest conclusions so far are:

1. The title and story sequence consumed large exploration budgets without
   exercising puzzle mechanics. A deterministic, evaluator-owned bootstrap is
   appropriate and separately logged.
2. Raw visual novelty, screen change, and broader action coverage are useful
   for exploration but are not reliable measures of puzzle progress.
3. Sparse positive milestones were too weak for early navigation. Explicit
   heart-aware shaping greatly accelerated the assisted development track, but
   reward is no longer the primary Room 3 bottleneck.
4. Straight-line distance to a visible goal cannot represent obstacle
   preparation, block placement, delayed transformations, or post-milestone
   behavior.
5. Wider or deeper flat exact search does not compensate for a missing
   representation. Several searches verified thousands or tens of thousands
   of branches without discovering a useful preparation.
6. Coarse persistent pixel changes frequently confused player animation,
   enemy animation, remote displays, and nearby objects. Matched controls,
   phase alignment, connected player masking, and object correspondence are
   mandatory.
7. A bounded search failure is not causal evidence that collecting a milestone
   was wrong. Earlier goal-exhaustion logic learned false negative value from
   insufficient search.
8. Save-state restoration must preserve semantic and relational state as well
   as pixels. Lost pose, world context, archive ownership, or object identity
   produced incorrect conclusions and wasted repeated search.
9. The anonymous behavior model can learn reusable outcome distributions, but
   policy authority requires strict provenance, calibration, and fail-open
   behavior. Broad hazard association produced a false positive.
10. The current world-model and returnability sidecars have not passed all
    native generalization gates and therefore remain selection-disabled or
    telemetry-only where documented.
11. RunPod is not cost-effective for the current sequential emulator-search or
    small real-data training workloads. The M5 remains the default platform.
12. The latest work proved that a confirmed anonymous displacement can survive
    planning descendants and process restoration. It did not prove that the
    agent can choose a useful placement. The next bottleneck is relational
    object state and accessibility consequences.

## 4. Chronological learning record

### 4.1 Initial neural world-model training

What was tried:

- A convolutional visual encoder/decoder with action-conditioned ensemble
  dynamics was trained on branched emulator transitions.
- Held-out RGB prediction was measured over multiple horizons.
- A frozen planner used model error, uncertainty, novelty, and verified
  emulator outcomes.

What worked:

- The data path, neural checkpoint, training/freeze split, and multi-horizon
  prediction pipeline were validated.
- Held-out RGB error improved across modeled decisions in the first medium
  experiment.

What did not work well:

- Better pixel prediction did not translate directly into entering or solving
  the first puzzle.
- Longer-horizon training later worsened horizon-three L1 from `0.1307` to
  `0.1377` in one measured variant.
- Frozen planning spent large budgets in title and story animation.

Learning:

- Pixel reconstruction quality is necessary infrastructure, not a sufficient
  planning metric.
- Native policy gates override narrow offline loss improvements.
- Do not promote a model into action selection solely because held-out L1
  improves.

Plan change:

- Keep exact emulator verification as the decision oracle.
- Use learned models to propose, rank, and estimate uncertainty.
- Require paired native planning gates before adding policy weight.

Evidence:

- `docs/medium-experiment-2026-08-08.md`
- `docs/spatial-causal-model-2026-08-10.md`

### 4.2 Raw novelty and delayed-return exploration

What was tried:

- Visual novelty, scene novelty, action coverage, delayed-return penalties,
  archive restoration, and passive-sequence memory.
- Neutral grace windows after autonomous motion.
- Persistent-frontier values and behavioral abstraction.

What worked:

- Loops and returns became auditable.
- The planner preserved and restored more diverse visual and behavioral
  frontiers.
- Autonomous transitions and timer-driven sequences were distinguished more
  clearly.

What did not work well:

- The agent continued spending most of its budget in intro animation.
- Improved scene coverage did not produce puzzle entry.
- Fixed and active probe variants reached story tableaus but not Floor 1.
- Broader visual exploration was not equivalent to controllable progress.

Learning:

- Novelty is a useful exploration ingredient but an unreliable objective.
- Autonomous animation must not receive controller credit.
- Exploration metrics must be tied to controllability, persistent causal
  effects, or later verified outcomes.

Plan change:

- Add matched neutral controls.
- Add action-dependent causal signatures.
- Separate autonomous grace from intervention selection.
- Treat raw novelty as bounded secondary value.

Evidence:

- `docs/medium-experiment-2026-08-08.md`

### 4.3 Temporal-option credit assigned to timer-driven events

What was tried:

- Long passive sequences credited the initiating action when a later visual
  transition occurred.

Failure:

- A preliminary run assigned positive value to `START@4` before a 51-decision,
  28-scene timer-driven transition.
- Reusing that value caused repeated `START` selection on the same static
  pixels.
- The initiating action preceded the transition but did not demonstrably
  cause it.

Classification:

- **Falsified at the measured gate**: temporal precedence alone is not adequate
  causal credit.

Learning:

- Long-delay value must compare a factual trajectory with a matched-duration
  or delayed counterfactual from the same root.
- Correlated passive transitions cannot establish controller authority.

Plan change:

- Reserve and advance delayed counterfactual states.
- Credit an initiating choice only when the factual/counterfactual contrast
  persists.
- Keep negative samples scoped to exact state/action/duration unless broader
  causal evidence exists.

Evidence:

- `docs/medium-experiment-2026-08-08.md`, temporal-option and
  delayed-counterfactual sections

### 4.4 Learning from the title and story sequence

What was tried:

- Let the agent discover how to leave the start screen and traverse the story
  sequence through ordinary exploration.

Failure:

- Multiple frozen runs exhausted hundreds of decisions before reaching puzzle
  mechanics.
- The sequence primarily exercised timers, fades, and menus rather than the
  research target.

Classification:

- **Negative result** for research efficiency, not a claim that autonomous
  menu discovery is impossible.

Learning:

- Spending the dominant budget on deterministic initialization produces little
  evidence about puzzle learning.

Plan change:

- Introduce an opt-in evaluator-owned `lolo1-first-room` bootstrap.
- Bind it to the legal local ROM digest and an expected pixel endpoint.
- Log all bootstrap actions under attempt zero and exclude them from agent
  statistics.

Boundary:

- The bootstrap is infrastructure, not learned game knowledge, and it must not
  expand into room-specific solution macros.

Evidence:

- `docs/medium-experiment-2026-08-08.md`, evaluator bootstrap section

### 4.5 `SELECT` and broad hazard generalization

What was tried:

- Learn negative temporal value after `SELECT` caused a delayed fade/reset.
- Generalize harmful action evidence more broadly.

What worked:

- Exact delayed evidence identified the causal `SELECT@1` initiation.
- Pair-scoped temporal values reduced repetition of the known failure.

What did not work well:

- Broad action-level hazard generalization risked suppressing legitimate
  actions in unrelated contexts.
- A locally returning rightward move acquired negative exact-choice evidence
  but did not justify disabling `RIGHT` globally.

Learning:

- Hazard evidence must retain state, action, duration, context, and causal
  lineage.
- Global action hazards require much stronger evidence than local failure.

Plan change:

- Use exact-choice and context-specific values first.
- Generalize only causally matched terminal evidence.
- Fail open when all actions would otherwise be filtered.

Evidence:

- `docs/medium-experiment-2026-08-08.md`
- `docs/anonymous-entity-policy-gate-2026-08-13.md`

### 4.6 Sparse reward and early heart discovery

What was tried:

- Strict novelty and causal exploration without explicitly weighting hearts.
- Later, an assisted pixel-heart reward with distance shaping and life loss.

What worked:

- The strict agent eventually collected its first heart through causal
  frontier restoration.
- Assisted navigation shaping dramatically accelerated early Room 2 heart
  collection. A sequence that previously required hundreds of decisions was
  reproduced in tens of decisions.
- Explicit life-loss evidence provided a useful negative milestone.

What did not work well:

- Sparse milestones alone were too slow for repeated development.
- After all Room 2 hearts were collected, the agent did not reliably navigate
  back to treasure or clear the room.
- Larger or longer heart-aware runs continued exploring without learning the
  required preparation relationship.

Learning:

- Explicit positive milestones are effective assisted debugging tools.
- Reward can expose planning failures, but cannot substitute for object state,
  obstacle reasoning, timing, or delayed consequences.
- “Hearts are good” helped early movement but did not solve configuration
  planning.

Plan change:

- Retain the heart-aware track as assisted development scaffolding.
- Keep strict and assisted data separate.
- Stop treating reward weight as the leading Room 3 intervention.
- Move next to relational object and accessibility representations.

Evidence:

- `docs/human-prior-reward-experiment-2026-08-10.md`
- `docs/control-preserving-search-2026-08-10.md`
- `docs/roadmap.md`, strict versus assisted boundary

### 4.7 Straight-line goal distance

What was tried:

- Reward movement that reduced Manhattan-like pixel distance to a visible
  heart.
- Add a short detour grace window so temporary regression was allowed.

What worked:

- It accelerated uncomplicated routes and allowed short paths around simple
  obstacles.

What did not work well:

- It stalled at obstacles requiring preparation.
- It could not value moving an object away from the goal, arranging a safe
  path, triggering a transformation, or preparing for post-heart behavior.
- Hundreds of preparation decisions could leave the relevant blocker
  unchanged even when the player repeatedly approached the goal.

Classification:

- **Negative result** as the principal long-horizon planning representation.

Learning:

- Goal proximity must be coupled to world configuration and verified
  accessibility.
- Temporary distance regression can be necessary and should not be treated as
  failure by itself.

Plan change:

- Add phase-conditioned player-position novelty.
- Preserve world configurations alongside goal state.
- Plan over accessibility deltas and object-level hypotheses rather than only
  distance.

Evidence:

- `docs/human-prior-reward-experiment-2026-08-10.md`
- `docs/control-preserving-search-2026-08-10.md`

### 4.8 Causal cell coverage and persistent disappearance

What was tried:

- Reward globally under-visited action-caused coarse cells.
- Preserve stable changes from a learned modal baseline, especially persistent
  disappearance.

What worked:

- Lower Room 2 hearts were collected and their disappearance was preserved
  much longer.
- Persistent-change filtering prevented immediate rollback to some
  pre-collection states.

What did not work well:

- Coverage alone eventually restored pre-collection states.
- Persistence preserved milestones but did not collect upper hearts, open
  treasure, or clear the room.
- A transient player overlay could retire a persistent marker even though the
  underlying game state remained changed.
- A current-context causal-outcome preference over-fragmented movement and was
  stopped as invalid.

Learning:

- Preservation and acquisition are separate problems.
- Coarse pixel persistence is valuable evidence but not an object model.
- Player occlusion and animation can corrupt modal cell evidence.

Plan change:

- Use player-masked object tracks rather than only persistent coarse cells.
- Preserve configurations by relational state.
- Treat coverage as exploration telemetry or bounded secondary value.

Evidence:

- `docs/spatial-coverage-persistence-2026-08-10.md`

### 4.9 Save-state lifecycle and archive ownership

Engineering defects discovered:

- A full archive could add and immediately evict a branch, release its state
  during pruning, then release it again during decision cleanup.
- Short-lived archive handles disappeared when a process ended.
- Resuming a decision by replaying earlier rejected branches was costly and
  could diverge from the intended exact physical state.
- Semantic memory and restored pixels could become decoupled without explicit
  provenance.

Learning:

- Save-state capabilities require explicit ownership, cloning, release, and
  persistence semantics.
- Cross-process continuation needs content-addressed snapshots for selected
  decisions and promoted alternatives.
- Pixels remain authoritative when later semantic memory is paired with an
  older physical state.

Plan change:

- Track same-decision pruned handles.
- Persist decision snapshots and promoted option archives.
- Verify state and frame digests during restoration.
- Record memory-source and state-source runs separately.
- Skip incompatible live archive imports when state and memory sources are
  deliberately decoupled.

Evidence:

- `docs/medium-experiment-2026-08-08.md`
- `docs/control-preserving-search-2026-08-10.md`
- `docs/object-state-gate-2026-08-16.md`

### 4.10 Reversible world context and semantic state

What was tried:

- Extend the heart/player graph with reversible anonymous world-effect
  signatures.
- Preserve action-confirmed visual changes across archive restores.

What worked:

- The assisted graph distinguished more states than position-only memory.
- It reduced some animation-sensitive restoration and retained persistent
  transformations.

What did not work well:

- More semantic graph states did not produce more reachable player positions
  in some long runs.
- A longer continuation added no position beyond a plateau.
- Coarse world effects still confused animation and local sprite residue.

Learning:

- A different pixel signature is not necessarily a strategically different
  puzzle configuration.
- World context needs object correspondence, persistence, and relational
  consequences.

Plan change:

- Require matched controls and phase alignment.
- Separate transient interaction identity from confirmed manipulation.
- Move from one cumulative effect signature to multi-object track sets.

Evidence:

- `docs/human-prior-reward-experiment-2026-08-10.md`
- `docs/control-preserving-search-2026-08-10.md`

### 4.11 Physical frontiers and modest exact-search expansion

What was tried:

- Prefer unvisited player positions and verified action-sequence frontiers.
- Expand depth-five and later deeper exact option searches with larger beams.

What worked:

- More distinct player positions and controller edges became reachable.
- Verified intermediate prefixes improved graph coverage and archive recovery.
- Some topology objectives and previously unchosen forks were reached.

What did not work well:

- A depth-five/beam-16 search verified 1,656 paths at one plateau without
  opening treasure or changing rooms.
- Later Room 3 searches verified tens of thousands of branches without finding
  a simple route to the remaining heart.
- A deeper search could repeat states or pursue tracker artifacts.

Classification:

- **Falsified at the measured gates**: modestly increasing movement horizon is
  not enough at those bottlenecks.

Learning:

- Search organization improved, but the missing information was not simply an
  untried short controller sequence.
- A flat tree cannot efficiently reason about preparations whose value appears
  only after object and phase changes.

Plan change:

- Do not automatically increase depth or beam after failure.
- Add object-level hypotheses and learned options.
- Couple search diversity to relational configurations and accessibility.

Evidence:

- `docs/control-preserving-search-2026-08-10.md`
- `docs/object-state-gate-2026-08-16.md`

### 4.12 Coarse persistent-effect false positives

What was tried:

- Treat stable, localized, non-player pixel changes as manipulation frontiers.

Failures:

- Apparent effects failed stricter action-control checks.
- Enemy animation, player pose, blocked presses, and a remote display could
  look like persistent local transformations.
- An apparent push hypothesis from repeated directional actions was falsified.
- One preserved `[7,6]` state did not open the hypothesized lower-right route.
- Delaying a heart did not reveal a simple preparation within a depth-20,
  22,759-branch search.

Learning:

- Persistence alone does not establish causality or object correspondence.
- A changed cell near the controlled sprite is especially vulnerable to mask
  and animation errors.
- A verified visual state change may be real but strategically irrelevant.

Plan change:

- Compare action endpoints with equal-duration root and local `NOOP` controls.
- Search nearby neutral phase offsets.
- Mask the controlled sprite in both source and target independently.
- Require appearance correspondence at the predicted destination for
  directional displacement.
- Measure downstream accessibility before assigning strategic value.

Evidence:

- `docs/control-preserving-search-2026-08-10.md`
- `docs/object-state-gate-2026-08-16.md`

### 4.13 Player tracking and pose artifacts

Engineering defects discovered:

- Episodic reconstruction did not initially restore action-derived facing.
- Blocked movement animation could be interpreted as spatial progress.
- Half-tile snapping around negative offsets created false topology and
  waypoint conclusions.
- A player mask based on nearby blue and white pixels absorbed an adjacent
  disconnected white object.

Invalid conclusions caused by defects:

- Some apparent unreachable pockets were tracker artifacts.
- Some push correspondence failed because the object was erased with the
  player.

Learning:

- Player localization, facing, and masking are foundational dependencies for
  causal object learning.
- Tracker uncertainty must be visible in telemetry and must not silently become
  graph truth.

Plan change:

- Persist target pose through commits and resumes.
- Use non-negative half-up tile snapping.
- Anchor the assisted player mask on a connected blue/white component and keep
  disconnected adjacent objects.
- Build a learned action-correlated controllable-region tracker for the final
  strict track.

Evidence:

- `docs/control-preserving-search-2026-08-10.md`
- `docs/object-state-gate-2026-08-16.md`

### 4.14 Goal-exhaustion rollback learned false negative value

What was tried:

- If a post-heart search found no new endpoint, assign negative value to that
  collection ordering and restore the pre-milestone state.

Failure:

- In Room 3, bounded search exhaustion after too few evidence steps was treated
  as if the milestone caused an unrecoverable failure.
- The learned negative then filtered the same legitimate collection.
- A corrected replay found a new endpoint, directly falsifying the earlier
  “frontier exhausted” conclusion.

Classification:

- **Engineering and inference defect**: finite search failure was promoted to
  causal negative value.

Learning:

- Failure to find progress is not evidence that the preceding milestone caused
  failure.
- Exhaustion needs minimum post-milestone evidence, no intervening progress,
  known transitions, scoped context, and explicit non-hazard provenance.

Plan change:

- Require at least 16 consecutive committed post-milestone decisions by
  default.
- Reset exhaustion evidence on graph or player-position progress.
- Defer rather than learn negative value when the evidence threshold is not
  met.
- Mark ordering hints separately from hazard evidence.
- Fail open when filtering would remove every alternative.

Evidence:

- `docs/room3-milestone-credit-correction-2026-08-13.md`

### 4.15 Treasure detector missed an actual success

What happened:

- An audit initially concluded that Room 2 preparation had failed.
- Frame-by-frame inspection showed that Lolo had already contacted the open
  treasure.
- The semantic detector failed to persist and credit the acquired-treasure
  phase.

Classification:

- **Engineering defect** that invalidated the earlier negative conclusion.

Learning:

- Evaluator and semantic detectors can undercount success even when emulator
  behavior is correct.
- Negative policy conclusions must be checked against stored frames and scene
  transitions.

Plan change:

- Persist acquired-treasure state across restores and resumes.
- Use stable novel-scene transition detection as evaluator confirmation.
- Keep raw frames and replay sufficient for independent audit.

Evidence:

- `docs/human-prior-reward-experiment-2026-08-10.md`
- `docs/control-preserving-search-2026-08-10.md`

### 4.16 Spatial causal world model promotion

What was tried:

- Several changed-region renderers: redraw, directly supervised flow,
  recursive flow, and anchored flow.
- Offline held-out evaluation and native shadow comparisons.
- Optional counterfactual usefulness in planner ranking.

What worked:

- The anchored local flow/residual renderer passed a trajectory-balanced
  offline gate and improved many native branches.
- Uncertainty/error correlation became positive in a measured configuration.

What did not work well:

- Several renderer variants beat persistence at one step but failed later
  horizons.
- One offline-passing renderer still lost the native mean comparison.
- Paired planner ablations produced mixed exploration results.

Classification:

- **Not yet demonstrated** for policy authority.

Learning:

- Persistence is a strong baseline for sparse-change video.
- Offline averages can hide native failure on decision-relevant branches.
- A predictor can be useful for telemetry before it is reliable enough for
  selection.

Plan change:

- Keep spatial selection weight zero by default.
- Continue native shadow evaluation.
- Promote only after run-held-out and native branch gates both pass.

Evidence:

- `docs/spatial-causal-model-2026-08-10.md`

### 4.17 Observed-returnability sidecar

What was tried:

- Learn whether an endpoint had an observed return path using frozen spatial
  tokens.
- Train from observed transition graphs and later explicit bidirectional
  probes.

Failures:

- Graph-derived negative labels were policy-dependent and sparse.
- Native positive and negative probabilities were poorly separated.
- A source-disjoint aggregate gate produced AUC `0.510`, worse than a useful
  discriminator, and a negative mean probability above the positive mean.
- The model fit a tiny training set but did not generalize.

Classification:

- **Failed promotion gate**; remains telemetry-only.

Learning:

- Returnability needs explicit, balanced, source-disjoint probes and censored
  unknowns.
- Tiny negative sets encourage memorization.
- Native integration success does not establish model generalization.

Plan change:

- Use explicit bidirectional branch collectors.
- Label observed returns positively, budget-scoped non-returns negatively, and
  censor unresolved cases.
- Expand source-diverse data before retraining.
- Do not use the sidecar as reward, hazard, or policy authority.

Evidence:

- `docs/spatial-causal-model-2026-08-10.md`

### 4.18 Anonymous entity behavior and curiosity

What was tried:

- Cluster anonymous patch appearances.
- Learn context-, action-, duration-, and phase-conditioned outcome
  distributions.
- Reserve curiosity probes for rare or uncertain interactions.
- Learn inert/no-effect probabilities.

What worked:

- Matched controls rejected animation false positives.
- The model learned reusable no-effect and measured-change descriptors without
  supplied object names.
- Frozen guarded planning converted some evidence into a bounded advantage
  while retaining exact verification.

What remains unproven:

- Early native gates did not demonstrate a successful push or transformation.
- Curiosity runs mainly provided evidence about exploration coverage and
  false-positive control.
- Appearance types and local behavior do not yet form a persistent multi-object
  relational state.

Learning:

- Distributions with uncertainty and provenance are preferable to one
  unconditional rule per appearance.
- No-effect outcomes are valuable learned mechanics.
- Curiosity should distinguish an unseen context from a globally familiar
  appearance.

Plan change:

- Add explicit displacement and transformation descriptors.
- Preserve object identity across interactions and restores.
- Condition behavior on stable phase and local geometry.
- Keep unfamiliar or weakly supported cases experimental rather than
  authoritative.

Evidence:

- `docs/anonymous-entity-behavior.md`
- `docs/relational-manipulation-milestone-2026-08-13.md`
- `docs/anonymous-entity-semantics-gate-2026-08-13.md`

### 4.19 Anonymous hazard veto false positive

What was tried:

- Simulate vetoing branches using predicted anonymous-entity hazards.

Failure:

- A broad terminal association marked one genuinely dangerous result but also
  marked safe `DOWN` as a simulated veto.
- The false positive came from insufficient causal provenance.

Classification:

- **Failed broad-authority hypothesis**.

Learning:

- Passive terminal correlation cannot grant policy authority to every rare
  patch visible before a reset.
- Hazard transfer needs locally attributed intervention/control evidence.

Plan change:

- Separate empirical terminal correlation from causal hazard posterior.
- Require context-matched causal support and sufficient samples.
- Keep veto disabled by default.
- Fail open if all verified endpoints would be rejected.

Evidence:

- `docs/anonymous-entity-policy-gate-2026-08-13.md`

### 4.20 Pose-only and changed-cell preparation

What was tried:

- Give delayed preparation credit to player poses near a learned future goal.
- Couple preparation to cumulative anonymous changed cells.

Failures:

- `entity-v311-room3-previsited-future-goal-d36x18` visited the later chest
  location before the final heart but still stalled afterward.
- `entity-v312-room3-layout-aware-preparation-d24x18` produced apparent layout
  variants that mapped to an animated blue entity rather than useful object
  arrangements.

Classification:

- **Falsified at the measured gates**: pose-only preparation and unqualified
  cumulative changed cells are insufficient.

Learning:

- Preparation value requires persistent causal object state and future
  consequences.
- Spatial coincidence with a later goal does not establish a useful setup.

Plan change:

- Require confirmed causal manipulation or learned future-goal evidence.
- Introduce object correspondence and player-masked state signatures.
- Plan next around verified accessibility changes.

Evidence:

- `docs/object-state-gate-2026-08-16.md`

### 4.21 Broad pristine-room search before reliable detection

What was tried:

- Search broadly from a pristine Room 3 state for preparation configurations.

Failures:

- `v313` produced 1,756 raw changed-layout branches and accepted zero reliable
  manipulations through 3,202 verified branches.
- `v314` completed 3,333 exact branches and 139 causal probes without a
  confirmed directional displacement.

Classification:

- **Negative result** that isolated a detector/contact bottleneck.

Learning:

- More search was generating candidates faster than the representation could
  validate them.
- A targeted historical pre-interaction state was a better scientific gate
  than another larger pristine search.

Plan change:

- Use targeted save states to test one primitive at a time.
- Audit each detector gate quantitatively.
- Increase broad search only after the primitive passes.

Evidence:

- `docs/object-state-gate-2026-08-16.md`

### 4.22 Directional displacement detector failures

What was tried:

- Detect a known Room 3 displacement at a historical pre-interaction state.

Defects found:

- Directional action effects were not allowed the same persistence treatment
  as button actions.
- Repeated appearances were excluded by a rarity gate.
- The nonlocal effect mask discarded the adjacent destination cell.
- Source/destination appearance distance exceeded threshold because player
  pixels contaminated the source patch.
- The player mask erased the adjacent disconnected white object.

Learning:

- Repeated appearance is exactly the reusable class evidence the model needs.
- Directional manipulation destinations are often adjacent to the player and
  cannot be filtered as ordinary local sprite spill without correspondence.
- Source and target features need independent player masking.

Plan change:

- Allow phase-stable one-cell directional displacement to bootstrap a mechanic.
- Use raw player-masked matched-counterfactual effects at the expected
  destination.
- Remove rarity as a prerequisite for correspondence.
- Keep only the connected player-color component in the assisted mask.

Result after correction:

- `v318` found six tracked push branches, two persistent `RIGHT -> NOOP`
  branches, independent causal displacement evidence, and replayable archives.

Evidence:

- `docs/object-state-gate-2026-08-16.md`

### 4.23 Object state lost across planning cycles

What was tried:

- Resume from the confirmed pushed-object archive and continue deeper search.

Failure:

- `v319` restored correct pixels but not tracked source, destination,
  interaction, appearance, or persistence metadata at the new search root.
- It explored 869 branches through depth nine without another displacement or
  heart.

Classification:

- **Engineering representation defect**.

Learning:

- Pixel restoration is not enough for long-term reasoning when learned latent
  relational state is omitted.
- Archive telemetry is part of the persistent planning contract.

Plan change:

- Serialize tracked cells, object appearance, interaction direction, effect
  distance, phase/context, and persistence.
- Seed restored exact-search roots with that state.
- Reconstruct legacy metadata conservatively from verified effect signatures
  and destination pixels.

Result after correction:

- `v320`: 2,497 of 2,497 verified descendants retained the manipulated cell.
- `v321`: 132 of 132 descendants retained the confirmed `RIGHT`, source
  `(7,6)`, destination `(8,6)`, distance, and persistence evidence.

Evidence:

- `docs/object-state-gate-2026-08-16.md`

### 4.24 Confirmed identity overwritten by a later interaction

Defect:

- A descendant archive retained the confirmed pushed-world signature but
  paired it with a later unrelated `UP` interaction near another patch.

Learning:

- “Current interaction candidate” and “interaction that causally produced the
  confirmed world state” are different concepts.

Plan change:

- Store confirmed manipulation identity separately from transient interaction
  probes.
- Propagate confirmed fields through descendants.
- Serialize the confirmed source when archiving the corresponding world state.

Validation:

- `v321` retained the original `RIGHT` identity after a later two-action
  continuation.

Evidence:

- `docs/object-state-gate-2026-08-16.md`

### 4.25 Retaining a push did not solve the next objective

What happened:

- After cross-cycle track restoration, `v320` preserved the changed
  configuration across all verified descendants and archived a five-action
  continuation.
- It did not collect another heart.

Classification:

- **Not yet demonstrated**: persistent single-object state is necessary but
  does not establish strategic usefulness.

Learning:

- The planner needs multiple simultaneous tracks, explicit transformations,
  and a model of how configurations affect reachable space.
- Reward and search depth should not be changed before those representations
  exist.

Plan change:

- Adopt the object-centric and accessibility roadmap in `docs/roadmap.md`.
- The next decisive gate is deliberate preparation with verified downstream
  access, not merely detecting another changed cell.

### 4.26 Offline accessibility diff of the confirmed push

What was tried:

- A preregistered, read-only telemetry diff (no emulator cost) comparing
  player-cell coverage of the pushed-configuration searches (v319/v320/v321,
  3,498 verified branches) against pre-push 4-heart-era Room 3 searches
  (v313/v314/v316/v317/v318, 6,899 branches), excluding the pushed object's
  own footprint cells, with pose-level distinctions not counted (v56 lesson).

Result:

- The beyond-footprint coverage envelopes are identical, element for element
  (8 cells each). Every apparent per-run difference reduced to a
  budget/reserve artifact, not a configuration effect.
- v319 self-exhausted at depth 9 / beam 128 with zero novel endpoints at
  depth 10, so the pushed configuration's exhausted search found nothing
  outside the pre-push envelope.
- No player in either era ever occupied the pushed destination `(8,6)` or
  any cell right of column x=128 above the bottom row — the room's right
  side is unexplored in both eras. Pre-push probe telemetry is absent, so
  interaction-frontier comparison is censored.

Classification:

- **Negative result**, scoped to offline telemetry at explored budgets
  (depth ≤9, beam ≤128). Censored evidence that the single push is
  accessibility-neutral at those depths; not proof of neutrality.

Learning:

- The confirmed push's strategic value, if any, is not visible in existing
  search coverage; it can only live in unexplored territory (right side of
  the room, the untested `(8,11)` A-interaction, or walkability of `(8,6)`
  itself). This further supports 4.25: object persistence alone does not
  establish usefulness.
- Offline telemetry diffs are cheap and sharpen native experiments, but
  probe-level questions need probe telemetry in both arms — record probe
  events in every future paired experiment.

Plan change:

- The native paired accessibility probe (direction-review Amendment A) is
  now the decisive instrument, with three directed targets and a clean
  matched-lineage arm design recorded in
  `docs/offline-accessibility-diff-2026-08-16.md`.
- The Room 3 single-push Gate 4 vehicle is downweighted pending the probe's
  outcome; representation work (WP1/WP2) continues regardless.

Evidence:

- `docs/offline-accessibility-diff-2026-08-16.md`
- `docs/direction-review-2026-08-16.md`
- runs under `experiments/lolo1-entity-v10/evaluations/`

### 4.27 Paired native accessibility probe of the confirmed push

What was tried:

- The preregistered paired probe (`docs/paired-accessibility-probe-2026-08-16.md`):
  two arms from the same v318 lineage one push apart, byte-identical
  planning flags (depth 12, beam 128, v320 reserve profile, probes enabled,
  8 decisions), Arm A from the pushed archive `state-00000117`, Arm B from
  the `33addc6c` pre-push rollback checkpoint. Runs
  `entity-v322-…-arm-a-pushed-d12` (4,061 endpoints, 776 s) and
  `entity-v323-…-arm-b-prepush-d12` (12,267 endpoints, 1,903 s), both
  complete within ceilings.

What worked:

- First causally paired accessibility fact in the project: the object
  demonstrably blocks the tile it occupies — `(8,6)` positively probed
  blocked in Arm A (object present) and committed-walkable in Arm B
  (empty). Footprint-scoped, but verified in both directions.
- The corrected v318-generation detector now confirms manipulations during
  ordinary search: 11 new confirmed manipulations across the two arms
  (2 in A, 9 in B, including a spontaneous new westward push
  `(7,6)→(6,6)` in B). Contrast v313's 0/1,756.
- Arm B reached column-8 cells `(8,7)`,`(8,8)`,`(9,8)` that Arm A never
  touched; Arm A's frontier into that band is positively closed at every
  tested edge, with `(8,6)` the sole differing edge — a sharp candidate
  mechanism: the confirmed push parked the object in the only doorway.

Failure:

- Configuration-hold certification failed at the instrument level: the
  player-masked world signature was identical across all 12,232 Arm B
  branches even though the run demonstrably displaced the object
  (`world_effects_accepted=0` while tracked/pixel evidence shows footprint
  disturbance in most branches). The coarse signature is blind to object
  displacement, so Arm B's beyond-footprint coverage cannot be certified
  configuration-held.

Classification:

- **Engineering defect** (hold-certification instrument) plus **censored
  evidence** on the preregistered delta: verdict "no beyond-footprint delta
  at budget (censored)", with directional evidence favoring the hypothesis
  recorded as directional only.

Learning:

- A paired design is only as strong as its configuration-hold instrument;
  world signatures must incorporate tracked object cells before
  accessibility claims can be certified.
- Walkability of the object's own tile is now a verified mechanic, and the
  candidate strategic reading of the confirmed push is inverted: it may
  have *closed* the sole entrance to the column-8 band rather than opened
  anything — preparation value can be negative.

Plan change:

- Extend `human_prior_option_branch_verified` telemetry (or the
  player-masked signature) to carry tracked object cells — a small change
  now feasible cleanly via `object_tracks.py` (WP1).
- Rerun Arm B alone at identical settings scoring one preregistered bit:
  ≥1 certified configuration-held branch reaching column ≥8, rows 5–7.
  Yes → beyond-footprint delta confirmed, Gate 4 vehicle promoted; no →
  censored-negative, vehicle downweighted.

Evidence:

- `docs/paired-accessibility-probe-2026-08-16.md` (full preregistration and
  results)
- runs `entity-v322-room3-paired-probe-arm-a-pushed-d12`,
  `entity-v323-room3-paired-probe-arm-b-prepush-d12`

### 4.28 Certified rerun: the confirmed push is accessibility-neutral; the object is the door

What was tried:

- After the 4.27 instrument fix (tracked object cells on every branch
  event, commit `ddae223`), the pre-push arm was rerun once at identical
  settings (`entity-v324-room3-paired-probe-arm-b-rerun-certified-d12`,
  12,232 branches, 30 min, within all ceilings), scoring one preregistered
  bit: does ≥1 certified configuration-held branch reach column ≥8,
  rows 5–7?

Result:

- **Bit = 0.** Certified-held coverage (1,756 branches, object undisturbed
  at `(7,6)`) is identical, element for element, to the pushed arm's
  envelope: `(6,6),(6,7),(6,8),(6,9),(6,10),(7,10),(8,10)`.
- All 54 branches that entered the `(8,7)/(8,8)/(9,8)` band carry `(7,6)`
  in their effect set — object disturbance is a necessary condition for
  band entry across the entire dataset.

Classification:

- **Negative result (certified, budget-scoped)** for the v318 eastward
  push as preparation: with the object undisturbed, pushed and pre-push
  configurations reach exactly the same space. The 4.27 directional
  evidence favoring a delta is resolved: it was configuration-departure,
  not configuration difference.

Learning:

- The first fully certified accessibility conclusion of the project: a
  confirmed, persistent, causally verified manipulation can be
  strategically neutral. Object persistence (4.25) and now even verified
  displacement are insufficient without measured accessibility
  consequences — the roadmap's WP6/WP8 thesis, demonstrated natively.
- The constructive inversion: the object itself is the door to the
  column-8 band. The useful preparation is displacing it out of the
  corridor (westward, discovered spontaneously in both pre-push runs),
  not eastward into the corridor's far cell.

Plan change:

- Room 3 Gate 4 vehicle redirected from "preserve the v318 push" to
  "westward displacement opens the column-8 band": commit and archive the
  westward push, then run a certified paired probe of displaced vs
  pre-push scoring certified-hold band entry — the roadmap's own WP6
  native gate with the correct manipulation. Preregistration to follow in
  the experiment note before execution.

Evidence:

- `docs/paired-accessibility-probe-2026-08-16.md` §7–§8
- run `entity-v324-room3-paired-probe-arm-b-rerun-certified-d12`

### 4.29 Track decomposition of the band-opening configuration

What was tried:

- An offline recon over v324's new track telemetry (358 distinct track-set
  signatures, per-cell button correlation over 12,232 branches,
  decision-boundary frame diffs) to locate the displaced object and design
  the next probe root.

Result — three corrections to the record:

- **The "westward push" inference in 4.27/4.28 is falsified.** The
  manipulation that opens the column-8 band is *removal*: the `(7,6)`
  entity was transformed in place by a button press, pushed one cell east
  in transformed state, then expelled east off along row 6. A westward-
  displaced state exists only in 71 released, unrestorable branch
  endpoints.
- **`anonymous_object_track_cells` is accumulated history, not endpoint
  configuration.** Five of the committed six cells had physically relaxed
  to baseline by v324 d7 while the set still listed them. Decomposition:
  `(7,6)` real vacated home (action-caused); `(8,6)/(11,6)/(12,6)`
  transient transit cells; `(2,6)/(3,7)` autonomous patroller leak that
  registers only in button branches; `(14,5)` the HUD shot counter —
  outside the room entirely.
- **Second instrument gap:** causal-archive `archive_branch_restored`
  events carry no track fields, so a mid-run causal-archive restore
  silently resets accumulated hold evidence (v324 d8). Option-archive
  restores are covered (WP1 `archived_track_fields`); the causal archive
  is not.

Classification:

- **Engineering defect** (4.27/4.28's westward inference — drawn from
  branch-level evidence without track telemetry; conclusions corrected) —
  plus a **validated decomposition method**: accumulated-set + button
  correlation + frame-diff separates real manipulation, transit,
  autonomous leak, and HUD echo without any supplied labels.

Learning:

- Carried track state must distinguish "changed at some point" from
  "still changed"; endpoint-relative track state is the WP2/WP3 contract
  this confirms as necessary.
- HUD regions and autonomous patrol must be excluded from manipulation
  credit by measurement (matched controls catch them in movement-only
  branches; they leak only after a real anomaly exists in the branch).
- Every archive class that can reseed a search root must carry the track
  block — instrument fix queued.

Plan change:

- The next probe is re-titled the **object-removed configuration probe**,
  preregistered in `docs/object-removed-probe-2026-08-16.md` with root =
  v324 d7 snapshot (player at `(7,6)`, entity removed, world relaxed),
  certification `cells == []`, and a new analysis rule voiding
  certification after any causal-archive restore.

Evidence:

- `docs/object-removed-probe-2026-08-16.md` (premise correction section)
- v324 telemetry; recon preserved in session transcripts

### 4.30 First verified accessibility-improving manipulation

What was tried:

- The preregistered object-removed probe
  (`docs/object-removed-probe-2026-08-16.md`): one bounded run
  (`entity-v325-room3-object-removed-probe-d12`, 9,691 branches, 24 min)
  from v324's decision-7 snapshot — player at `(7,6)`, the resident entity
  removed — scoring certified configuration-held band entry against the
  two certified 7-cell baselines (pushed v322, pre-push v324).

Result:

- **Bit = YES: 135 certified configuration-held branches reached the
  column-8 band.** Certified coverage totaled 24 cells vs 7 in both
  baselines — the former footprint, the band, and the whole eastern
  region through column 12, including known-heart cell `(12,11)`. Zero
  life losses; every claimed cell was reached with the configuration
  certifiably intact (pre-causal-restore window per the preregistered
  rule).

Classification:

- **Milestone result.** The project's first verified
  accessibility-improving manipulation: removal of the `(7,6)` entity
  (two button transformations + one transformed-object push) more than
  triples certified reachable space and exposes a milestone-bearing cell.

Learning:

- The arrangement→accessibility thesis (roadmap §3) is now demonstrated
  natively in both directions: the v318 push was certified neutral
  (4.28), the removal is certified enabling. Manipulation value is a
  measurable property of configurations, not of manipulations per se.
- The full causal chain — detect manipulation, preserve configuration,
  measure accessibility consequence — now runs end-to-end on real
  emulator state with anonymous instruments only.

Plan change:

- Repeat the delta from a fresh restore to close Gate 3 formally
  (same-run neutral control included).
- Gate 4 becomes the active target: make the planner *prefer* the removal
  because of its measured consequence (verified-delta term in
  restore-selection/hypothesis preference per the direction-review
  Amendment A ablation design), then test preparation → `(12,11)` heart
  within budget.

Evidence:

- `docs/object-removed-probe-2026-08-16.md`
- run `entity-v325-room3-object-removed-probe-d12`

### 4.31 Substitution replay: letter-pass, substantive no-promote

What was tried:

- The preregistered WP5 substitution replay
  (`docs/wp5-tracker-training-2026-08-16.md`): reconstruct v318/v321
  tracked state with the learned mask (tracker v2, held-out AUC 0.9997 on
  its training distribution) substituted for the assisted mask, scoring
  fixed bits; report at
  `experiments/lolo1-wp5/substitution-replay-report.json` (digest
  `6061b45e…`, reproducible).

Result:

- Bits 1–2 PASS on both archives — but analysis shows those computations
  are mask-irrelevant for these archive shapes (identity fields derive
  from metadata/bitmask; the recorded destination signature was computed
  unclipped), so the pass does not demonstrate learned≈assisted
  equivalence.
- Bit 3 divergence telemetry: endpoint IoU 0.0 / 0.0217; full-sweep mean
  IoU ≈ 0.0002–0.0009 over 178 frames; the player's cell receives
  probability 0.25 and ~3e-10. **The tracker does not localize on Room 3
  frames** — its training corpus (`lolo1-medium`, early-room legacy
  segments) does not cover Room 3's visual distribution.
- Genuine replay finding: the v318 assisted replay itself deviates from
  the recorded signature (L1 0.1347 > 0.08) — the original run computed
  the destination signature with different slot/clipping than a fresh
  replay produces.

Classification:

- **Failed promotion in substance** (formal letter-pass recorded and
  disqualified); plus an **instrument-design lesson**: preregistered bits
  must be sensitive to the mechanism they claim to test — the informative
  channel here was the report-only divergence, not the gated bits.

Learning:

- Offline gate metrics on the training distribution (AUC 0.9997) say
  nothing about out-of-room generalization; this repeats the project's
  recorded offline-pass/native-fail pattern (spatial v2/v9,
  returnability) at the perception layer.
- Assisted-era telemetry may be used as EVALUATION ground truth without
  contaminating strict corpora; the counterfactual displacement
  components in the v322–v326 probe events are detector-free localization
  ground truth for exactly this measurement.

Plan change:

- Tracker remains telemetry-only. Next: (a) quantify the OOD gap by
  evaluating tracker v2 against counterfactual localization ground truth
  from v322–v326 telemetry (evaluation-only use of assisted-collected
  data); (b) broaden the strict corpus with strict-track counterfactual
  collection beyond `lolo1-medium` coverage, retrain, and (c) redesign
  the substitution gate with mask-sensitive bits before any promotion
  claim.

Evidence:

- `docs/wp5-tracker-training-2026-08-16.md`
- `experiments/lolo1-wp5/substitution-replay-report.json`

### 4.32 Tracker OOD gap quantified: state-dependent, not uniform

What was tried:

- The §4.31(a) evaluation (`docs/tracker-ood-eval-2026-08-16.md`,
  report digest `ecc5336b…`, byte-identical on rerun): tracker v2 scored
  against dense detector-free counterfactual ground truth from v322–v326
  telemetry (24,538 arms, 16 censored, 100% validity cross-check against
  the assisted detector's cell), with a 400-arm held-in reference.

Result:

- Held-in: hit 1.000, AUC 0.9997, IoU 0.775 — the instrument reproduces
  training-time numbers exactly.
- Room 3 pooled: hit 0.478, AUC 0.679, IoU 0.022. **State-dependent
  failure:** object-present states fail totally (AUC < 0.5 — mass ranked
  away from the player); object-removed states partially transfer (hit
  0.833, but ~28-cell blobs vs ~2 true cells).

Classification:

- **Measurement result** completing the §4.31 diagnosis: the gap is
  corpus coverage (Room 3 palette, object-present configurations,
  duration diversity), not architecture.

Plan change:

- Broaden the strict corpus with a bounded strict-track counterfactual
  collection run in Room 3 (development partition; both object-present
  and object-removed states; duration diversity; hard negatives at
  movable-object tiles), import via the provenance-checked path,
  regenerate labels, retrain, then the §4.31(c) mask-sensitive gate.

Evidence:

- `docs/tracker-ood-eval-2026-08-16.md`
- `experiments/lolo1-wp5/tracker-ood-report.json`

### 4.33 WP9 step 1 falsified as written; the mechanism survives in parts

What was tried:

- The preregistered WP9a scoring pass (`docs/milestone-scoring-2026-08-16.md`,
  report digest `424bb775…`): thresholds fixed from the event census
  before scoring, then one pass over ~1.33M matched factual/NOOP endpoint
  pairs across three corpora.

Result:

- **Falsified via heart-inseparability**: 7/47 collection instances
  scored positive against the fixed 0.80 recall threshold. Timer/animation
  domination did NOT occur (matched differencing is phase-aligned by
  construction) — the failure is subtler than the preregistered worry.
- **Reversion-based negative valence failed 0/14**: at the fatal commit
  the matched NOOP control also dies, so life loss is action-independent
  exactly where the valence rule required dependence; losses present as
  large persistent novel changes, not reversions.
- Named failure mechanisms, each verified on an instance: event-level
  dependence-censoring on mixed changed-cell sets; return-censoring by
  restore-heavy assisted timelines; rarity non-separation in the biggest
  corpus. Clean collections in corpus B still rank 7/14/48 — the
  matched-NOOP core and score shape survive.

Classification:

- **Falsified at the measured gate** for WP9 step 1 as written; the early
  falsification is precisely the value Amendment D promised — this would
  otherwise have surfaced at WP12.

Learning:

- Milestone valence needs a signal that works when death is
  action-independent at the terminal commit (e.g., delayed divergence
  before the terminal horizon, as the causal hazard machinery already
  does) — reversion/control-collapse alone cannot carry it.
- Milestone extraction must handle mixed changed-cell sets and
  restore-heavy timelines; single-decision successor windows are too
  short.

Plan change:

- WP9a redesign requirements recorded (censoring semantics, multi-decision
  windows, hazard-style delayed-divergence valence); WP9b unchanged.
  Rescoring only after a redesign with these fixes, preregistered afresh.

Evidence:

- `docs/milestone-scoring-2026-08-16.md`
- `experiments/lolo1-wp5/milestone-scoring-report.json`

### 4.34 Mask-sensitive gate: localization closed, resolution is the remaining gap

What was tried:

- The §4.31(c) redesigned promotion gate (preregistered in
  `docs/wp5-tracker-training-2026-08-16.md`; report digest `7bb95c5e…`,
  byte-identical rerun): learned-vs-assisted agreement scored only on
  mask-mattering frames (6,474/6,474 — the instrument cannot pass
  vacuously, fixing the §4.31 lesson).

Result:

- **FAIL — NO-PROMOTE.** Signature agreement 0.000 on all three corpora
  despite 100%/100%/98.8% mask overlap. The failure is mask
  resolution/extent, not localization: the learned mask erases whole
  16×16 cell blocks while the assisted mask erases a pixel silhouette
  plus a 3px halo across partial cells, so pooled features differ by
  construction.

Classification:

- **Failed promotion gate** with a completed diagnosis chain: v2 failed
  localization (§4.31–4.32) → v3/v4 closed it state-by-state → the
  remaining gap is representational resolution, not learning.

Learning:

- Cell-resolution perception cannot silently substitute into a
  pixel-resolution masking convention; promotion requires either a
  pixel-mask reconstruction head anchored at learned cells, or an
  explicitly gated convention change with its own regression evidence.
- The three-cycle collection recipe (counterfactual labels + targeted
  strict collection) is validated as an iterative pipeline — state-local
  closure was immediate every time coverage was supplied.

Plan change:

- Next WP5 step: pixel-mask reconstruction spike (learned-cell-anchored
  silhouette recovery), then re-run this same gate unchanged. Tracker v4
  remains telemetry-only with divergence telemetry.

Evidence:

- `docs/wp5-tracker-training-2026-08-16.md` (preregistration + results)
- `experiments/lolo1-wp5/mask-sensitive-gate-report.json`

### 4.35 Pixel reconstruction closes the resolution gap; replication is the wrong gate

What was tried:

- The §4.34 pixel-mask reconstruction spike (preregistered; report digest
  `1052c9ea…`, byte-identical rerun): a 19.7k-param pixel head trained on
  pixel-granularity counterfactual silhouettes (detector-free; pixel AUC
  0.99843 held-out), reconstruction anchored on tracker v4 cells, and the
  mask-sensitive gate rerun unchanged.

Result:

- **FAIL / NO-PROMOTE** on the preregistered bits — but every measurable
  axis moved (mask IoU mean 0.33→0.40, per-frame max 0.90–0.95; the L1
  bound now holds on high-IoU frames, which v1 never achieved once).
- Two residual causes, one of them an instrument insight: (1) silhouette
  extent diverges because the assisted mask is strongly multi-modal
  (~340/1,100/1,640 px modes — occlusion shrinkage plus the same
  white-component leakage that produced the v316/v317 defects) while the
  learned mask is stable (~760 px); (2) the signature bit requires
  byte-identical masks per pooled cell — an equality-class criterion no
  learned reconstruction can satisfy short of replication.

Classification:

- **Failed promotion gate** plus a **gate-design finding**: the
  replication criterion conflates "masks correctly" with "reproduces the
  assisted mask including its defects." A learned mask that is more
  stable than the assisted one can never pass a byte-equality bit.

Learning:

- Substitution gates must ultimately be FUNCTIONAL: does tracking built on
  the learned mask produce correct, ground-truth-verifiable outcomes
  (correspondence, fingerprints, confirmed manipulations) — not identical
  intermediate bytes. §4.34's "explicitly gated convention change" path is
  the scientifically correct one and is hereby chosen (engineering-internal,
  reversible, claim boundary unmoved): the strict pipeline defines its own
  masking convention around the learned mask and gates it on functional
  track-reconstruction correctness against counterfactual ground truth.

Plan change:

- Design the functional gate (WP5-final): learned-convention tracking vs
  assisted-convention tracking scored on outcome agreement over the probe
  corpora ground truth; preregister before running. Replication gates
  retired for perception promotion.

Evidence:

- `docs/wp5-tracker-training-2026-08-16.md` (spike preregistration +
  results)
- `experiments/lolo1-wp5/mask-sensitive-gate-v2-report.json`

### 4.36 WP9a v2: valence solved, separation falsified again — rethink

What was tried:

- The §4.33 redesign (per-component censoring, restore-robust
  lineage-filtered windows, delayed-divergence valence), preregistered in
  `docs/milestone-scoring-v2-2026-08-16.md` and rescored once (report
  digest `898676b5…`, byte-identical rerun; v1 still reproduces
  `424bb775…`).

Result:

- **Life-loss negative valence 14/14 PASS** (v1: 0/14) — all via delayed
  divergence, 10 of 14 from runs never inspected at design time. Both v1
  failure mechanisms individually repaired (dependence rates 1.00;
  fallback windows engage).
- **Heart separation fails again**: 15/47 = 0.319 vs the 0.80 gate
  (v1: 0.149). New dominant mechanism, instance-verified: reset
  bleed-through — window-scoped rewind marks any occurrence whose window
  crosses a later terminal reset, and per-signature valence over merged
  component classes lets a handful of rewound windows flip whole
  collection classes negative (28 of 32 residual failures; one class of
  176 occurrences decided by a single evaluable window). Not
  threshold-tunable: per-signature valence trades the negative gate
  against the positive gate on the same axis.

Classification:

- **Falsified at the measured gate, second time** — WP9 step 1 demotes
  from redesign to RETHINK. The valence mechanism is validated; the
  aggregation structure is the falsified part.

Learning:

- Delayed factual-vs-control divergence is the correct terminal-valence
  signal (generalizes out of design sample).
- Rewind evidence must anchor to the event's own component cells, and
  valence must be occurrence-scoped with signature aggregation used only
  for ranking — class-level valence is structurally unable to satisfy
  both gates.

Plan change:

- WP9 step 1 paused pending the rethink (occurrence-scoped valence +
  component-anchored rewind); no third rescore until that redesign is
  preregistered. WP9's schedule position is unchanged — this work remains
  ahead of need.

Evidence:

- `docs/milestone-scoring-v2-2026-08-16.md`
- `experiments/lolo1-wp5/milestone-scoring-v2-report.json`

### 4.37 Functional gate: learned beats incumbent on absorption; symmetric erasure blocks promotion

What was tried:

- The WP5-final functional gate (preregistered; report digest `414c6576…`,
  byte-identical rerun; GT = detector-free counterfactual components as
  referee): manipulation detection, fingerprint stability, and
  absorption-regression bits over the three probe corpora, learned vs
  assisted conventions.

Result:

- **FAIL / NO-PROMOTE**: detection 0.30–0.45 vs gate 0.95 (mechanism,
  instance-verified on 200/200 misses: symmetric erasure — the
  vacated/occupied label blur plus anchor dilation covers the entire GT
  component in BOTH endpoints, zeroing the factual/control difference
  exactly where the effect lives); stability 0.77–0.82 vs 0.95
  (all-or-nothing extent swings produce phantom world-state changes).
- **Learned strictly beats assisted on bit (c)**: preservation 0.97–0.98
  vs 0.72–0.77 — the incumbent erases 23–28% of player-adjacent
  player-free object cells, the exact v316/v317 defect class. First
  measured functional axis where the learned convention exceeds the
  incumbent.
- Report-only: the assisted convention itself misses 6–12% of GT
  manipulations — a defect structurally invisible to replication gates.

Classification:

- **Failed promotion gate** with the correct instrument at last: the gate
  can now see both conventions' defects against neutral ground truth.

Learning:

- The incumbent is not a gold standard; it is merely incumbent. Promotion
  gates refereed by ground truth measure both sides.
- Silhouette supervision must disambiguate occupied vs vacated pixels —
  symmetric union targets teach symmetric erasure.

Plan change:

- Next spike: occupied/vacated disambiguation in the pixel-label path
  (separate channels or vacated-only targets at the factual endpoint),
  reconstruction convention v2, then THIS gate rerun unchanged. Tracker
  remains telemetry-only.

Evidence:

- `docs/wp5-tracker-training-2026-08-16.md`
- `experiments/lolo1-wp5/functional-gate-report.json`

### 4.38 Occupied/vacated disambiguation kills displacement erasure; in-place erasure remains

What was tried:

- The §4.37 spike (preregistered; report digest `7d1e5703…`,
  byte-identical rerun): occupied-only silhouette targets via a
  detector-free sibling-agreement rule (two different displacements
  agreeing on local content certifies revealed scene), vacated pixels as
  weighted negatives, convention v2 with undilated anchor, functional
  gate rerun unchanged.

Result:

- **FAIL / NO-PROMOTE overall — hypothesis confirmed where it applied**:
  detection roughly doubled (0.30–0.45 → 0.68–0.78), masks sprite-sized,
  first-ever learned-only detections (58/34/36 per corpus that the
  incumbent misses); bit (b) passes at v322 (0.975) and reaches
  mean-parity elsewhere; bit (c) widened to 0.985–0.998 vs assisted
  0.72–0.77.
- Remaining failure class, instance-verified on 750 misses: **in-place
  erasure** — blocked/contact arms where nothing vacates, so occupied
  masking still covers the pose-change evidence in both endpoints
  (factual/control mask IoU median 0.95).

Classification:

- **Failed promotion gate**, second mechanism isolated. The label-semantics
  program is exhausted: in-place changes are inside the region a correct
  controllable mask must cover, so no silhouette refinement can expose
  them under the current detection quantity.

Learning:

- Detection currently asks "does the world outside the mask differ?" —
  for in-place effects the answer is structurally no under any correct
  mask. The quantity itself must change (e.g., compare masked-region
  content across matched endpoints as a separate channel), and that is a
  conventions change requiring its own preregistered gate, not a tuning
  step.

Plan change:

- Next: preregister the detection-quantity v2 design (masked-region
  differential channel alongside the outside-mask signature), rerun the
  functional gate with the new quantity scored separately so the change
  is auditable. Tracker/head remain telemetry-only.

Evidence:

- `docs/wp5-tracker-training-2026-08-16.md`
- `experiments/lolo1-wp5/functional-gate-v2-report.json`

### 4.39 Two-channel detection closes bit (a); promotion hangs on a data-scale tail

What was tried:

- The §4.38 detection-quantity change (preregistered; report digest
  `01a9b128…`, byte-identical rerun): outside-mask signature OR
  masked-region differential, applied symmetrically to both conventions,
  with a hard false-positive bit designed from training-corpus
  measurement after the drafted flat bound was falsified at design time.

Result:

- **Bit (a) 1.000 on all three corpora** (0 misses / 4,488 measurements);
  in-place erasure eliminated as predicted. Bit (d) passes everywhere
  (0 fires on identical pairs; exact parity with the incumbent on
  uncorroborated pairs). **v322 passes all four bits — first corpus
  ever.** The incumbent's own 6–12% GT misses are explained by the same
  in-place blindness and are fixed by the same quantity.
- **NO-PROMOTE** solely on the bit-(b) placement-flip tail: v323 0.936,
  v325 0.943 vs 0.95.

Classification:

- **Failed promotion gate by tail margin**, with every detection
  mechanism now closed. The perception program's remaining gap is
  quantitative (pose-diverse data at two states), not conceptual.

Learning:

- Class-mix-sensitive false-positive bounds must be per-class; a flat
  bound was falsifiable from training data alone before spending the
  gate run.
- The masked-region differential dominates detection (~93–100% of GT
  rows); any planner use outside GT-anchored evaluation must carry the
  bit-(d) discipline.

Plan change:

- Tier-4 probe-distribution strict collection at the v323/v325 states
  (pose diversity for the stability tail), pixel-head retrain, functional
  gate v3 rerun unchanged. If bit (b) clears 0.95, recommend
  shadow-promotion with divergence telemetry.

Evidence:

- `docs/wp5-tracker-training-2026-08-16.md`
- `experiments/lolo1-wp5/functional-gate-v3-report.json`

### 4.40 Stability tail resists the data lever; WP5 promotion campaign paused

What was tried:

- Tier-4 pose-diversity strict collection (10 sub-runs at the two tail
  states, cycle 20, labels v5: 21,633 roots / 96,682 arms), pixel head v3
  retrained on occupied-v2 semantics, functional gate rerun unchanged
  (report digest `99285632…`).

Result:

- **v322 passes all four bits with learned stability 1.000** (first
  full-pass corpus, now with a perfect stability score); detection 1.000
  everywhere; preservation 0.982–0.998 vs incumbent 0.72–0.77.
- **NO-PROMOTE**: the stability tail at v323 (0.936→0.929) and v325
  (0.943→0.933) moved slightly against the data lever. Two bounded
  attempts (v3 gate, v4 gate) reproduce the same failure with different
  data — the roadmap §11 stopping rule fires for this lever.

Classification:

- **Negative result for the data lever** on the stability tail;
  campaign paused per stopping rule rather than escalated.

Learning:

- The placement-flip tail is not a coverage phenomenon — pose-diverse
  data improved nothing and slightly worsened it. The flip is plausibly a
  property of single-forward-pass extent prediction across adjacent
  poses; the next lever is a design change (reconstruction hysteresis
  across matched frames, ensemble extent averaging, or extent
  regularization), which must be preregistered fresh, not tuned into the
  current head.

Plan change:

- WP5 promotion campaign PAUSED at the tail (perception is at
  parity-plus for every functional purpose except it; tracker/heads
  remain telemetry-only with the divergence discipline). Queue priority
  shifts to: WP8-lite ablation (awaiting planner-file release), WP9a
  rethink, and the stability-design lever when prioritized.

Evidence:

- `docs/wp5-tracker-training-2026-08-16.md`
- `experiments/lolo1-wp5/functional-gate-v4-report.json`

### 4.41 Third falsification: the event unit itself is wrong

What was tried:

- The §4.36 rethink (component-anchored rewind + occurrence-scoped
  valence), preregistered in `docs/milestone-scoring-v3-2026-08-16.md`
  and rescored once (digest `e2c3434c…`, byte-identical rerun; v1/v2
  reports re-verified byte-identical first).

Result:

- Reset bleed-through repaired exactly as designed (the flagship v2
  class goes 6/9 rewound → 0/9). Heart recall improves a third time
  (0.149 → 0.319 → **0.574**) yet fails the 0.80 gate; and the negative
  gate regresses to 2/14 for a mirror-image structural reason: fatal
  commits' own windows are empty — nothing descends from a death
  endpoint — so occurrence scoping starves precisely the terminal class
  that class scoping over-reached.

Classification:

- **Falsified at the measured gate, third time — the unit is wrong.**
  Class-scoped valence bleeds; occurrence-scoped valence starves; both
  defects live inside the matched-endpoint-pair + successor-window event
  representation itself. No fourth rescore of this unit.

Learning:

- Monotone metric improvement across redesigns can coexist with a
  structurally unsatisfiable representation — the paired gates exposed
  it where either gate alone would have invited more tuning.
- Milestone events should be represented at the OBJECT level — the
  track/transition units WP2/WP3 already model (collection = a tracked
  appearance ceasing at a heart cell; death = control-loss divergence on
  the tracked controllable region) — with replay/reset-stable identity,
  rather than as pixel-pair windows. The strict-objective path and the
  object-centric roadmap converge on the same machinery.

Plan change:

- WP9 step 1 paused at the representation level; its next incarnation is
  scheduled AFTER WP2/WP3 integration provides object-level event
  streams, and will be preregistered against the same three gates.

Evidence:

- `docs/milestone-scoring-v3-2026-08-16.md`
- `experiments/lolo1-wp5/milestone-scoring-v3-report.json`

### 4.42 WP5 campaign complete: PROMOTE-to-shadow on the ensemble-agreement anchor

What was tried:

- The §4.40 design lever, chosen from flip-structure measurement (the
  v4 bit-b failures reproduced to the last digit; the flip lives at the
  tracker anchor — marginal cells straddling 0.5 across adjacent poses —
  not at the head threshold; hysteresis excluded structurally, morphology
  excluded by blob shape). Reconstruction convention v3: anchor requires
  mean p ≥ 0.5 AND cell-ensemble variance ≤ 0.004, calibrated from the
  training corpus only. No retraining; single-variable change; functional
  gate rerun unchanged (report digest `ac4bd00f…`, byte-identical rerun).

Result:

- **PASS / PROMOTE-to-shadow — first full pass of the campaign.**
  Stability 1.000 / 0.9995 / 0.9728, exceeding the incumbent on all
  three corpora for the first time; detection 1.000 everywhere;
  preservation 0.994–0.999; false-positive discipline at exact v4 parity.
- Watch-items for shadow telemetry (not re-tuning): v325's residual 2.7%
  tail, empty-mask rates, and the detection-channel shift toward the
  differential.

Classification:

- **Promotion gate PASSED.** The WP5 evidence chain is complete: six gate
  iterations, each isolating one mechanism (mask-irrelevance → coverage →
  resolution → symmetric erasure → in-place erasure → anchor
  marginality), each fix verified by the next unchanged gate. The learned
  masking convention (tracker v4 + pixel head v3 occupied-v2 +
  reconstruction v3 + detection quantity v2) is detector-free,
  lineage-clean, and functionally superior to the assisted incumbent on
  stability, preservation, and in-place detection.

Learning:

- Measure the failure before choosing the lever: all three candidate
  designs were decidable from data before spending the gate run.
- Ensemble variance is the reusable uncertainty instrument the roadmap
  hoped for — here it converts pose-marginality into an abstention rule.

Plan change:

- Next: shadow wiring in the planner (learned convention as telemetry
  alongside assisted, divergence + empty-mask + tail dashboards) — this
  touches `neural_planner.py` and queues with WP8-lite behind the
  worktree release. WP5's acceptance clause (strict tracking without
  `PixelHeartGoalPrior` imports) becomes reachable once shadow telemetry
  accumulates native evidence.

Evidence:

- `docs/wp5-tracker-training-2026-08-16.md` (full campaign chronicle)
- `experiments/lolo1-wp5/functional-gate-v5-report.json`

### 4.43 WP8-lite ablation: mechanism validated, behavior unchanged — FAIL as preregistered

What was tried:

- The preregistered paired ablation (design doc §3/§6, report digest
  `19f4092f…`, scorer byte-identical across three reruns): control
  (weight 0.0) vs treatment (weight 1.0), byte-identical otherwise
  (278-field config equality verified), from the v318 pre-push checkpoint
  root with the certified record store loaded in both arms.

Result:

- **Bit 1 PASS**: the treatment's decision-2 restore selected the
  removal-class archive on its certified value — bonus 25.0 (17 new
  cells + milestone ×8), `current_source: baseline`, full component
  attribution; later restores correctly show `mapped/0.0` once the root
  acquired the removal-class signature.
- **Bit 2 FAIL**: the arms' committed trajectories are identical — the
  control's plain frontier score already restored the *same*
  removal-class branch non-deliberately (29.578 vs 54.578 on the same
  winner). Both arms reach beyond the baseline envelope at the same
  decision; neither collects the `(12,11)` milestone within 8 decisions.
- **Bit 3 PASS**: zero life losses both arms. First behavioral divergence
  is post-window reserve-order permutation that fully reconverges.
- Record correction: v324's committed trajectory DID collect hearts at
  d1/d3 (the §4.27-era "zero hearts from this root" framing was wrong);
  the uncollected object is specifically the `(12,11)` milestone heart.

Classification:

- **FAIL (mixed) per the preregistration** — no weight tuning, no rerun;
  the declared Amendment E fallback (relational planner extraction)
  engages. The mechanism itself is validated at the selection level; what
  failed is *discrimination*: at this root, novelty/coverage and
  certified accessibility prefer the same branch, so deliberateness is
  behaviorally redundant.

Learning:

- An ablation root must exhibit *score conflict* — a configuration the
  baseline scorer disprefers but certified accessibility prefers —
  or bit-2-style consequence bits cannot discriminate deliberate from
  incidental choice. Root selection is part of experimental power, not
  just provenance.
- The existing novelty machinery is better at finding valuable
  configurations than the §4.28-era narrative suggested: it chose the
  removal twice (v324's committed line, and here) without any
  accessibility term. The planner's gap is sustaining and *chaining*
  preparations, not the single restore choice.

Plan change:

- WP8 proceeds on the declared fallback: relational planner extraction
  (Amendment E), whose hypothesis-level planning is where deliberateness
  can matter beyond a single restore — with ablation roots chosen for
  score conflict, preregistered.

Evidence:

- `docs/wp8-lite-ablation-design-2026-08-16.md` §7
- `experiments/lolo1-wp5/wp8lite-ablation-report.json`
- runs `entity-v327-…-control-w0-d12`, `entity-v328-…-treatment-w1-d12`

### 4.44 Relational shadow run: the chain forms; non-interference confirmed

What was tried:

- The design's mandatory telemetry-only shadow run
  (`entity-v329-room3-relational-shadow-d12`, 79,493 events, complete):
  the relational planner proposes, logs, and tracks hypotheses with zero
  selection influence.

Result:

- **The full chain formed with correct linkage**: `establish` proposed and
  logged at decision 1 *before any realization* (Gate 4's open criterion,
  mechanically satisfied), achieved at d2 on the removal-class
  configuration `85fd9014…`; `hold` chained to it, achieved d3;
  `exploit` chained to the hold, targeting the certified milestone cell
  `(12,11)`, correctly gated on
  `requires_uncollected_certified_milestone`.
- **Options stored and reused**: a hold option was re-instantiated at d8
  from its relational initiation conditions — the first observed option
  transfer.
- **Non-interference confirmed**: the committed trajectory is identical
  to the incidental runs (hearts at d1/d3, same endpoints), as
  telemetry authority requires.
- The exploit terminated `budget_exhausted` at d7.

Interpretive caution (recorded to prevent a false inference):

- That termination is the EXPECTED null in telemetry mode, not evidence
  about budget sizing. With zero steering authority the exploit's
  realization budget drains across decisions it does not control — and
  indeed the agent moved west/up, away from the eastern target region,
  throughout. Budget adequacy is only measurable under selection
  authority.

Classification:

- **Instrument validation** for the chain machinery; the capability claim
  remains untested pending E1.

Learning:

- Hypothesis chaining, chain-parent linkage, option storage and reuse,
  and pre-execution logging all work on real native state — the parts
  §4.43 showed a restore scalar could not express.
- A telemetry-mode null must not be read as a parameter finding; the same
  discipline that made the shadow run mandatory applies to reading it.

Plan change:

- Proceed to E1 under selection authority with the exploit realization
  budget justified from v325's actual traverse cost (the certified run
  that did reach `(12,11)`), disclosed in the E1 preregistration.

Evidence:

- run `entity-v329-room3-relational-shadow-d12`
- `docs/wp8-relational-planner-design-2026-08-17.md`

### 4.45 E1 FAIL: hypotheses are passengers — they cannot request the search they need

What was tried:

- Experiment E1 (preregistered, design doc §12; report digest
  `6b6708db…`, scorer byte-identical across three runs and validated
  against v329 first): control (authority `off`) vs treatment
  (`selection`), matched budgets, from the v318 pre-push root — does the
  treatment collect the certified `(12,11)` milestone that four
  incidental runs never collect in-window, with its chain logged first?

Result:

- **FAIL.** Bit 1 fails (the exploit never achieved; terminated
  `budget_exhausted` at d7), bit 2 fails (neither arm collected
  `(12,11)`), bit 3 passes (zero life losses). No VOID condition fired;
  the arms' committed trajectories are identical state-id by state-id and
  the treatment's telemetry differs by exactly its 16 relational events.

Mechanism (measured, two parts):

1. **Seam-opportunity, not budget size.** The exploit held selection
   authority across decisions 3–7, which contained **zero option
   searches** — and its reserve seam fires only inside an option search.
   Its budget was never read. The planner commits most decisions from
   archives and direct selection without searching, so a hypothesis
   wanting to steer has no lever on those decisions at all.
2. **Redundancy one level up.** The only seam that did fire — the d2
   restore preference — was redundant exactly as §4.43 found: the
   candidate already carried the removal signature.

Classification:

- **Falsified at the measured gate**, with an architectural cause: the
  relational planner is a *passenger*. It can rank what the search
  offers, but it cannot cause the search that would realize its
  objective, nor act on non-search decisions.

Learning:

- Hypothesis-level planning requires **schedule authority**, not just
  scoring authority. An objective that cannot request an exact search, or
  express itself through the non-search commit path, cannot steer
  regardless of how well it is scored or budgeted.
- Another units lesson: `realization_branch_budget` is a per-depth
  parent-reserve slot count, not a verified-branch count. Sizing it from
  traverse cost (1,389 branches) would have been a unit error; the
  derivation caught it, and the run then proved the budget inert (supply
  ceiling 2–3 at this root vs 48 allocated).

Plan change:

- Next: design and preregister **search-schedule coupling** — can an
  active objective request an option search (and/or express a navigation
  target through the existing control-frontier machinery) — scored on its
  own bits. Explicitly NOT a budget re-size (proved inert) and NOT a
  rerun of E1. E2 (conflict root) is unaffected by this failure mode and
  remains queued.

Evidence:

- `docs/wp8-relational-planner-design-2026-08-17.md` §12
- `experiments/lolo1-wp5/e1-gate4-report.json`
- runs `entity-v330-…-e1-control-off-d12`,
  `entity-v331-…-e1-treatment-selection-d12`

### 4.46 The planner had stopped searching entirely — archive growth self-suppresses the gate

What was found:

- Read-only telemetry analysis for the search-scheduling design
  (`docs/wp8-search-scheduling-design-2026-08-17.md`): across
  **v327–v331 the planner's own machinery ran ZERO option searches**. The
  single search in each run is the resume audit at decision 0 — which
  executes *before* the removal configuration exists.
- Cause, measured: the stagnation path's deferral gate defers whenever
  any archive branch carries a frontier flag. All three stagnation
  instants (d2/d5/d8) deferred with 9/4/3 global archive frontiers. The
  planner's own archive growth suppresses its search gate. The remaining
  decisions never reach the block at all (navigation-recovery grace).
- Contrast: v325 — the run that reached `(12,11)` — ran **two**
  planner-initiated searches (d5, d8), and its final approach steps were
  ordinary commits, not searches.
- Related structural find: commit-time archive constructions never set
  `tracked_world_state_signature`, so hold-gated restore supply at the E1
  root is capped at the four decision-0 audit branches (max cell
  `(9,8)`).

Classification:

- **Engineering/behavioral defect of the incumbent planner**, discovered
  while designing around it. §4.45's "hypotheses are passengers" has a
  deeper cause: in these runs there was almost nothing to be a passenger
  ON.

Learning:

- A capability layer can be starved by a gate that looks unrelated. The
  E1 result would have been mysterious without this enumeration — and no
  amount of hypothesis-side tuning could have fixed it.
- Search frequency is now a first-class planner-health metric to report
  per run, not an implicit assumption.

Plan change:

- E3 targets the navigation path (budget-neutral, steers the commit
  ladder that actually moves the agent) rather than the search path;
  a search-request mechanism is deferred to E4 with a forced-search
  control arm, since granting searches breaks matched budgets.
- **E3-pre first** (one 16-decision authority-off run): confirm the
  `(12,11)` discriminator still discriminates in the extended window. If
  the control collects the heart at 16 decisions, E3 as designed is
  cancelled and the discriminator must be re-chosen — a cheap check that
  can void an expensive experiment.
- Report search counts in every future native run summary.

Evidence:

- `docs/wp8-search-scheduling-design-2026-08-17.md` §Q1
- runs v325 (contrast), v327–v331

### 4.47 E3-pre: the discriminator survives at 16 decisions but is visibly dying

What was tried:

- The design's cheap pre-check (`entity-v332-…-e3-pre-control-off-d16`,
  complete): one 16-decision authority-`off` run to confirm the
  `(12,11)` discriminator still discriminates in the extended window
  E3 requires.

Result:

- The control does **not** collect `(12,11)` in 16 decisions — so the
  discriminator is technically alive. But the trajectory shows it
  closing: after re-entering the east region at d9 it descends column 12
  — `(12,7) → (12,8) → (12,9)` — ending **2 cells** from the target and
  moving directly at it (Chebyshev distance 3 → 2 across d9–d16).
- Also confirmed: exactly one search, the decision-0 resume audit —
  §4.46's suppression finding reproduces at 16 decisions.

Classification:

- **Measurement-validity finding.** A matched-budget E3 at 16 decisions
  would remain formally sound, but a treatment-collects/control-doesn't
  result would evidence a 2–3 decision *speedup*, not a capability
  difference — the redundancy pattern of §4.43/§4.45 for a third time,
  dressed as a Gate 4 pass.

Learning:

- Discriminator validity is a *trend* property, not a binary at one
  budget. "Control never does X in N steps" must be checked against
  whether the control is converging on X; a near-miss is a warning, not
  a margin.
- The stronger discriminator per roadmap §18 item 3 is the `(8,4)` /
  `(9,12)` hearts, which lie OUTSIDE the certified envelope and require a
  **second manipulation** — a capability claim incidental behavior has
  never demonstrated, rather than a speed claim.

Plan change:

- Definitive extension launched (`entity-v333-…-off-d24`): if the control
  collects `(12,11)` by 24 decisions the discriminator is dead and E3
  re-targets to the second-manipulation discriminator; if not, E3 runs at
  24 decisions with real headroom, and the speedup-vs-capability caveat
  is preregistered either way.

Evidence:

- run `entity-v332-room3-e3-pre-control-off-d16`
- `docs/wp8-search-scheduling-design-2026-08-17.md` §Q4

## 5. Platform and cost learnings

### 5.1 RunPod for emulator branching

Hypothesis:

> A paid RunPod GPU worker would reduce cost or time for the current search
> loop.

Result:

- The dominant emulator path is sequential and CPU-bound.
- The GPU did not accelerate it.

Decision:

- **Do not migrate current emulator branching to RunPod.**
- Keep the M5 as the default execution platform.

### 5.2 Synthetic GPU training benchmark

What happened:

- A small synthetic workload showed almost no useful advantage.
- Model-level final loss differed by only about `0.15%` in one comparison.

Learning:

- Synthetic throughput alone does not justify routine paid launches.

### 5.3 Real-data GPU training

What happened:

- The paid real-data gate failed by a wide margin.
- The synthetic pass did not transfer because it preallocated random one-step
  batches and did not represent the real storage/decode/training pipeline.

Decision:

- Use RunPod only after a representative local benchmark identifies a
  compute-bound model workload.
- Every paid cycle needs explicit dollar and wall-clock ceilings plus automatic
  Pod shutdown.

Evidence:

- `docs/runpod-platform-gate-2026-08-15.md`
- `docs/runpod.md`
- `docs/research-loop.md`

## 6. Directions discussed and deliberately rejected

### 6.1 Training from YouTube solutions

Proposal:

- Train a winning policy by watching people solve the game on YouTube.

Why it was rejected for this project:

- It changes the central research question from autonomous rule discovery to
  demonstration-assisted imitation.
- It contaminates held-out Room 1 evaluation if solutions are shown.
- It risks room-specific trajectory memorization.
- Raw video omits synchronized controller actions and introduces compression,
  cuts, overlays, timing differences, and uncertain emulator alignment.
- It would make a later *Lolo 2* generalization claim harder to interpret.

Decision:

- No YouTube or solution demonstrations in the plan of record.
- If demonstration learning is ever studied, it must be a separately labeled
  baseline with independent data and claims, preferably using synchronized
  emulator action telemetry.

### 6.2 Hard-coding complete game mechanics

Discussion:

- Hard-code known object behaviors, safe paths, or room solutions.

Decision:

- Rejected. The final model must discover reusable behavior from pixels and
  interaction.
- The deterministic start bootstrap is a narrow evaluator fixture, not
  authority to add room-specific macros.

### 6.3 Treating hearts and lives as the final strict solution

Discussion:

- Explicitly weight hearts positively and life loss negatively.

Learning:

- This was valuable for the assisted development track and isolated planning
  failures.
- It violates the strict final claim if supplied semantic detectors remain in
  the final agent.

Decision:

- Keep the assisted track for debugging and ablation.
- Replace supplied player/heart/life semantics with learned visual milestones,
  controllability, and terminal evidence before strict evaluation.

## 7. Plans formulated from the accumulated evidence

### 7.1 Evidence-gated research cycles

Formulated because:

- Long unattended runs repeatedly spent thousands of branches after the
  decisive failure mode was already visible.
- Paid-compute uncertainty needed hard ceilings.

Plan:

1. State one falsifiable hypothesis.
2. Declare wall-clock, telemetry, cycle-cost, and campaign-cost limits.
3. Run one bounded gate.
4. Audit raw telemetry.
5. Record an immutable reflection.
6. Continue only with the exact next hypothesis justified by evidence.

### 7.2 Strict and assisted separation

Formulated because:

- Heart-aware development produced useful progress but does not satisfy the
  original no-object-definition claim.
- Mixing transitions would make evaluation uninterpretable.

Plan:

- Explicit reward-track and partition manifests.
- Dataset loaders reject incompatible provenance.
- Freeze every persistent artifact during withheld and sequel evaluation.

### 7.3 Object-centric relational representation

Formulated because:

- Pose, changed cells, and cumulative world hashes could not represent
  preparation.
- Single-object persistence succeeded but did not establish utility.

Plan:

- Multiple anonymous tracks.
- Explicit displacement and appearance transitions.
- Phase and local-context conditioning.
- Track-set signatures in planner nodes and archives.

### 7.4 Accessibility and reversibility model

Formulated because:

- Raw object movement is not inherently good.
- Goal distance cannot express opened or closed routes.
- Bounded non-return must remain censored evidence.

Plan:

- Measure player reachability before and after verified manipulations.
- Track newly reachable cells and interaction frontiers.
- Use explicit bidirectional probes.
- Value verified accessibility changes, not generic pixel change.

### 7.5 Phase-conditioned mechanics

Formulated because:

- The same appearance may behave differently after a global visual milestone.
- Final-heart and treasure phases change room dynamics.

Plan:

- Learn stable global phase embeddings from pixels.
- Condition anonymous outcome distributions on phase.
- Compare matched branches before and after phase transitions.

### 7.6 Hierarchical object-level planning

Formulated because:

- Flat exact search verified many paths without capturing delayed preparation.

Plan:

- Generate anonymous hypotheses such as testing or reproducing a displacement,
  approaching an interaction frontier, preserving a configuration, or
  investigating a phase contradiction.
- Use exact emulator search to realize and verify each hypothesis.
- Learn reusable options from relational initiation and termination states,
  not room-specific action strings.

### 7.7 Learned controllable-region tracker

Formulated because:

- Assisted color and shape masks caused false object conclusions.
- The final strict system cannot depend on supplied player identity.

Plan:

- Learn the action-correlated controllable visual region from matched branches.
- Produce masks and position distributions with uncertainty.
- Keep the assisted detector only as a development comparator.

### 7.8 Immutable evaluation split

Formulated because:

- Repeated targeted Room 3 work makes it a development room, not a legitimate
  withheld room.
- Generalization claims require untouched states and rooms.

Plan:

- Pre-register training, development, withheld *Lolo 1*, and sequel
  partitions.
- Reject persistent updates from frozen partitions.
- Audit all artifact digests before and after evaluation.

## 8. Do-not-repeat checklist

Before proposing a new experiment, confirm it does not repeat one of these
mistakes:

- Do not increase beam or depth merely because a run failed.
- Do not treat better pixel L1 as evidence of better planning.
- Do not treat raw novelty or screen change as puzzle progress.
- Do not credit an action for a later timer-driven transition without a matched
  counterfactual.
- Do not generalize one local action failure into a global action hazard.
- Do not treat a bounded search failure as causal proof that a milestone was
  bad or a state unrecoverable.
- Do not infer an object transformation from persistence without action
  controls and phase matching.
- Do not require visual rarity for reusable object identity.
- Do not remove adjacent same-colored objects with the player mask.
- Do not let a transient interaction overwrite the interaction that produced a
  confirmed world state.
- Do not restore pixels without the associated pose, phase, track, and archive
  metadata needed by planning.
- Do not promote passive terminal correlation to hazard authority.
- Do not train a returnability classifier on tiny policy-dependent negatives
  and report integration as generalization.
- Do not run paid GPU infrastructure for sequential CPU-bound emulator search.
- Do not use assisted heart/player/life labels in the final strict claim.
- Do not use YouTube solutions or demonstrations in the interaction-only
  project.
- Do not run another expensive experiment before recording what result would
  change the plan.

## 9. Current open hypotheses

These are not yet established facts:

1. Multiple anonymous tracks can remain stable through repeated identical
   appearances, occlusion, transformations, and several manipulations.
2. The behavior model can transfer displacement or transformation predictions
   across rooms without memorizing absolute locations.
3. A verified accessibility delta is a better preparation signal than goal
   distance or raw object movement.
4. Object-level hypothesis planning can find useful preparations within lower
   branch budgets than flat search.
5. Stable visual phase embeddings can predict post-milestone behavior changes.
6. A learned action-correlated controllable tracker can replace assisted
   player masks on native held-out rooms.
7. Learned visual milestones and controllability can replace the assisted
   heart/life reward while retaining useful exploration.
8. The persistent learned mechanics will transfer to withheld *Lolo 1* rooms.
9. The frozen persistent system will make meaningful progress in *Lolo 2*.

Each hypothesis requires its own bounded gate. None should be reported as
achieved merely because the required code exists.

## 10. Current decision and next experiment

Decision:

- Continue the interaction-only approach.
- Do not pivot to demonstrations.
- Do not change heart reward or launch a larger flat search now.
- Build the object-centric and accessibility architecture in
  `docs/roadmap.md`.

Next implementation sequence:

1. Freeze the experimental partition and persistent artifact inventory.
2. Extract object tracking from `neural_planner.py` into a tested module.
3. Support multiple simultaneous anonymous tracks.
4. Encode explicit displacement and transformation transitions.
5. Propagate track sets through search nodes, archives, resume, and telemetry.
6. Run the bounded two-manipulation native gate.
7. Reflect before granting any new policy authority.
8. Prototype accessibility deltas in mock environments and then at a targeted
   native state.

The next decisive evidence is:

> The agent preserves multiple anonymous object changes, measures that one
> verified manipulation changes future accessibility, deliberately chooses the
> useful configuration, and reaches a later milestone because of it.

## 11. Evidence index

- `docs/medium-experiment-2026-08-08.md`
- `docs/human-prior-reward-experiment-2026-08-10.md`
- `docs/spatial-causal-model-2026-08-10.md`
- `docs/spatial-coverage-persistence-2026-08-10.md`
- `docs/control-preserving-search-2026-08-10.md`
- `docs/room3-milestone-credit-correction-2026-08-13.md`
- `docs/relational-manipulation-milestone-2026-08-13.md`
- `docs/anonymous-entity-semantics-gate-2026-08-13.md`
- `docs/anonymous-entity-policy-gate-2026-08-13.md`
- `docs/anonymous-entity-behavior.md`
- `docs/runpod-platform-gate-2026-08-15.md`
- `docs/research-loop.md`
- `docs/object-state-gate-2026-08-16.md`
- `docs/roadmap.md`

When a new gate changes the plan, update this file with:

1. the hypothesis;
2. the exact measured evidence;
3. whether the result was falsified, negative, defective, or unproven;
4. the plan change;
5. the run IDs and source document; and
6. the condition under which the rejected direction may be reconsidered.
