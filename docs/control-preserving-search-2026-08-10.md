# Control-preserving Room 2 search

Date: 2026-08-10

## Result

The strict pixel-only track collected all four Room 2 hearts for the first
time, retained the resulting persistent screen changes, and reached both the
upper and lower halves of the room. A later assisted positive-control run
collected the treasure and cleared Room 2 using only pixels, controller
outcomes, and agent-discovered save-state branches. The strict rule-free track
has not yet reproduced that completion.

The run also exposed the first repeatable death mechanism. A controller action
could produce a distinct immediate frame while every action from its endpoint
led to the same future frame. Treating that condition as a learned
unrecoverable state, rather than as novelty or autonomous animation, now lets
the agent roll back before the life-loss sequence.

No object label, enemy definition, RAM value, level solution, or demonstration
is used by these changes. Heart and player positions cited below were measured
offline by the labeled evaluator and were not visible to the strict policy.

## Four-heart frontier

Behavioral best-first recovery fixed an archive-starvation failure on the
right bridge. In
`spatial-v14-room2-global-edge-first-second-bridge-from-d28-d100`, the agent
restored an unexpanded `UP` branch, traversed the right bridge, and collected
the fourth heart at decision 38. The accepted persistent-change configuration
kept all four heart disappearances through decision 100.

The first continuation from that state produced a clean death trace:

- the player sprite disappeared;
- the screen dimmed and then became black;
- the room returned with the HUD life value changed from 5 to 4;
- all four hearts returned.

The trace also showed that action-independent animation had been allowed to
create false persistent-change evidence. Persistence updates are now causal:
a changed cell advances only when a non-`NOOP` endpoint differs from its
duration-matched neutral endpoint.

## Counterfactual control collapse

When a causal observation wait is proposed, the agent can now branch one step
from the proposed passive endpoint and measure the maximum pairwise pixel
spread across controller actions. If the spread is below the existing action
equivalence threshold, it probes one step beyond every immediately
action-dependent alternative.

Three outcomes are distinguished:

1. passive waiting retains future control, so normal causal observation
   continues;
2. passive waiting collapses control but another endpoint retains it, so the
   viable endpoint is selected;
3. every endpoint collapses future control, so the initiating choice receives
   an exact negative temporal-option sample and the pre-action save state is
   restored.

Native validation from the upper route learned and restored the following
long-press hazards from the same behavioral state:

| Initiating action | Duration | Learned value | Outcome |
|---|---:|---:|---|
| `DOWN` | 16 | -2.0 | checkpoint restored |
| `UP` | 16 | -2.0 | checkpoint restored |
| `RIGHT` | 16 | -2.0 | checkpoint restored |

Control-collapse rollback now removes only archive descendants newer than the
causal checkpoint. Same-decision sibling alternatives remain available.
Visual-return and transition history produced by the rejected descendant are
also removed, preventing a shorter later trial from being mistaken for a loop
through an abandoned branch. A confirmed negative archive branch is never
restored merely because all other interventions have been exhausted.

## Time-sensitive action refinement

The initial controller probes used only 16-frame directional holds. After a
duration is learned unsafe, verification now reserves slots for untried shorter
presses of the same controller action. Required probes are keyed by both action
and duration, so they coexist instead of replacing one another.

A second timing bug was then isolated: a 2- or 4-frame movement was followed by
a mandatory 16-frame `NOOP`, causing the observation itself to wait into the
death. Causal observation now reserves and selects a neutral action whose
duration matches the initiating press.

The native trace
`spatial-v14-room2-matched-observation-from-junction-d46-d45` contains explicit
pairs such as `DOWN 4` followed by `NOOP 4`. That pair retained control and
created a new intermediate state; search then began a fresh duration sweep
from that state. This is the first evidence that the agent can refine an
unsafe macro action into a controllable micro-action without a rule or
demonstration.

## Rule-free dark-transition rollback

The 120-decision timed continuation eventually exhausted its local alternatives
and lost a life. The failure frames were:

- decision 91: mean intensity 0.0288;
- decision 92: fully black;
- decision 93: the room returned with life 4 and all hearts present.

The strict agent now keeps a bounded memory of diverse coarse pixel layouts.
After a dark transition resolves, it compares the bright result with that
temporary visual memory. A visually novel layout is retained as possible
progress. A near return to a previously observed layout schedules recovery.

Resume evaluations seed this temporary memory from every pixel frame in the
save-state ancestry. This does not update the world model or any frozen
parameter. It restores episodic context that checkpoint-only continuation had
previously discarded.

Recovery excludes every archive branch created on or after the dark-transition
start. A retained pre-transition archive state is preferred; a saved run-entry
state is the last-resort fallback.

The five-decision native proof
`spatial-v14-room2-native-pre-dark-only-rollback-from-timed-d90-d5` observed a
dark frame, a black frame, and a known-scene return at coarse distance
0.001667. It filtered two post-dark archive branches and restored the pre-dark
state at decision 4. The restored frame retained HUD life 5; the rejected reset
frame showed life 4. This resume point was already inside the death animation,
so it is a mechanism proof rather than a useful gameplay frontier.

## Temporal confirmation of lost control

The first control-collapse detector made a terminal judgment after one future
action horizon. That was unsafe for generalization because a temporary room
animation can also make every controller action produce the same pixels.

The detector now advances disposable save-state branches for four matched
action horizons. It retains the branch if action-dependent outcomes return or
if darkness resolves to a novel bright layout. It confirms a collapse only if
control remains absent and no novel transition appears. These observations are
counterfactual; the live agent state is restored after every probe.

The 60-decision native A/B run
`spatial-v14-room2-temporal-control-confirmation-from-d100-d60` recorded 24
temporal confirmations. Eleven regained action-dependent control and were
retained. Thirteen remained invariant across all four horizons; twelve were
rolled back to their causal checkpoints. The first confirmed endpoint was a
bright, stable room state at known-scene distance 0.0075, not a dark/reset
transition. Visual inspection showed the controllable character boxed by the
local layout. No life was lost, and the frozen-parameter audit passed.

The preceding 150-decision run
`spatial-v14-room2-control-preserving-lower-search-from-d100-d150` verified
1,026 branches and 364 unique frames. It preserved the zero-remaining-item HUD
and lost no life, but did not reach a stable scene transition. This establishes
that the current bottleneck is multi-action reachability after the persistent
frontier, rather than collecting or retaining that frontier.

## Behavioral breadth and delayed-outcome audit

The 200-decision strict A/B run
`spatial-v14-room2-behavioral-edge-w4-from-d100-d200` enabled behavioral-edge
coverage at weight 4.0. It committed 91 unique behavioral edges, verified 1,323
branches, and produced 565 unique frames without a life loss during the live
trajectory. This was a modest breadth improvement over the zero-weight run,
not a room-clear result.

Pixel-only post-run localization found that the agent repeatedly reached the
lower-right region and also verified branches near the lower central flashing
element. The trace exposed a recovery-order bug: a matched causal NOOP could
set delayed-return recovery, causing a global archive restore before any
outgoing intervention from the observed endpoint. The agent now guarantees one
non-NOOP intervention after a matched causal observation and logs recovery
suppression and intervention selection explicitly.

A direct non-training diagnostic from committed decision 194 applied one
verified action and then passive NOOPs. The sequence became dark at passive
step 10, black at step 11, and returned to a dim Room 2 layout with life 4 at
step 12. This corrected an initial visual interpretation: the branch was a
death/reset, not Room 3.

The planner can now perform an optional, rule-free delayed-transition probe on
verified save-state branches. It passively rolls a branch forward, selects it
only when darkness resolves to a visually novel bright layout, and schedules
the same passive observations on the live trajectory. The known-scene coarse
distance threshold was widened from 0.01 to 0.04 after the dim reset measured
0.0276 from remembered Room 2; the evaluator's distinct-room threshold remains
0.05.

The ten-decision native validation
`spatial-v14-room2-delayed-transition-known-reset-from-d194-d10` recorded 14
delayed probes that reached a dim known-scene return, zero novel-scene probes,
and zero delayed-transition branch selections. The frozen-parameter audit
passed. A known return under passive waiting is not treated as proof that the
first action is fatal, because continued intervention can still be required to
escape a dynamic threat.

## Assisted semantic save-state frontier

The explicitly labelled positive-control track now performs best-first search
over a stable pixel goal key: remaining heart slots, detected player tile,
detected open-chest tile, and the visible HUD life glyph. This key is used only
for archive ordering on the assisted track. The strict track remains unchanged.

The first 300-decision semantic-frontier run reached 45 detected player
positions and chest distance 4, but restored 206 branches while verifying only
819. Extending the same configuration was stopped at decision 572 after no new
position had appeared since decision 211; the archive had fallen to 35 and the
chest had not opened. Offline inspection found that the position-only key
merged screens with materially different lower-room layouts. It also archived
same-state actions during autonomous flashing merely because the controller
edge had not yet been tried.

The assisted graph now retains an alternative only when its stable semantic
state changes. A matched action-versus-`NOOP` coarse effect is allowed to make
a new world variant only after:

1. source and target player cells are removed from the effect;
2. directional effects are local to the detected player, while learned
   interaction-button effects may be non-local; and
3. counterfactual probing shows that action-dependent control is present on
   the very next horizon.

Accepted non-player cells are accumulated as a reversible parity state rather
than an action-history hash. Applying the same coarse transformation twice
returns to the prior context. Save-state archive entries and life/scene
checkpoints preserve this context explicitly.

The final native comparison,
`spatial-v14-room2-assisted-reversible-world-context-from-d100-d300`, completed
300 decisions with unchanged frozen parameters:

| Metric at decision 300 | Position-only graph | Reversible world context |
|---|---:|---:|
| Detected player positions | 45 | 36 |
| Distinct assisted graph states | 45 | 101 |
| Archive restores | 206 | 123 |
| Verified branches | 819 | 1,503 |
| Unique frames | 457 | 1,116 |
| Best chest distance | 4 | 4 |
| Confirmed life losses | 0 | 0 |

The new run probed 204 candidate non-player effects, rejected 86 without
immediate control, accepted 118, and visited 36 world-context values. It did
not open the chest or clear Room 2. The result is nevertheless a stronger
search direction: it spends fewer decisions restoring animation-sensitive
branches and investigates more action-conditioned puzzle states, at the cost
of deferring nine positions in the far upper-right corridor within this fixed
budget.

## Physical frontier and verified action sequences

The reversible-context run was extended to decision 503. It reached 181
goal-plus-world graph states and retained 419 archived alternatives, but it
still had only the same 36 player positions seen at decision 300. It produced
no goal milestone, life loss, or scene transition. The extra graph growth was
therefore context churn rather than physical reachability progress.

Archive recovery now gives first priority to any unexpanded endpoint at a
globally unseen detected player position. This is still an assisted-track
pixel key; it does not assign names or traversability rules to room objects.
When the current semantic source has no such one-step endpoint, an optional
bounded search restores the exact source save state and verifies short action
sequences in the emulator. The sequence is tracked as one graph edge so an
already tried first action does not suppress an untried multi-action option.
Only a goal milestone or a globally unseen player endpoint can be archived.
Search is deferred while a cheaper unseen endpoint exists locally, and an
exhausted source is cached to prevent identical rollout work.

Matched 120-decision frozen runs from the same strict Room 2 decision-100
state isolate the effects:

| Treatment | Player positions | Graph states | Restores | One-step branches | Option branches | Options committed |
|---|---:|---:|---:|---:|---:|---:|
| Reversible-context baseline | 22 | 40 | 48 | 612 | 0 | 0 |
| Physical frontier only | 45 | 47 | 20 | 855 | 0 | 0 |
| Unbounded depth-3/beam-8 options | 46 | 48 | 20 | 855 | 1,092 | 1 |
| Local-gated depth-3/beam-8 options | 46 | 48 | 20 | 855 | 666 | 1 |

The selected exact option was `UP, UP` from detected `(96,112)` to the new
endpoint `(96,96)`. The physical-only run reached every other position in the
option run, so unseen-position archive ordering accounts for 23 of the 24
additional positions and verified sequence search accounts for one. Local
gating preserved that endpoint while reducing sequence rollouts by 39% from
the unbounded treatment. All four runs reached chest distance 4; none opened
the chest or changed rooms within 120 decisions.

The local-gated configuration was then extended and stopped at decision 187
after no new position had appeared since decision 127. It reached 49 player
positions and 62 assisted graph states, committed four exact options, restored
70 branches, and verified 2,004 option paths. It still reached only chest
distance 4 and produced no chest, life-loss, or scene-transition event. This
is a substantially broader physical frontier than the prior 36-position
reversible-context run, but depth-3 reachability has now exposed its own clean
plateau. Longer search should be targeted at the distance-4 boundary or score
persistent non-player transformations; repeating depth-3 search elsewhere is
not justified by these data.

A bounded depth-5/beam-16 diagnostic resumed the exact distance-4 state at
local decision 89. Across nine searches it verified 1,656 exact option paths
and committed seven endpoints. None was closer than distance 4, none completed
the chest, and no life or scene event occurred. The frozen audit passed. This
falsifies a modest increase in movement horizon at that boundary; the next
search signal must distinguish a lasting non-player room transformation from
ordinary reachability and animation.

## Persistent-effect falsification

The option verifier now records raw action-versus-duration-matched-`NOOP`
effects, replays them across future horizons, limits accepted effects to four
coarse cells, checks nearby animation phases, and performs leave-one-action-out
controls. An opt-in assisted world-effect frontier was also tested. When it was
allowed to compete directly with ordinary endpoints, it reduced physical
exploration; ordinary unseen endpoints now always win and a confirmed effect is
only a fallback.

The apparent local signal did not survive stricter controls. A first
memory-faithful audit from the decision-127 endurance plateau recorded 48
stability probes, three compact local candidates, eight action ablations, and
three same-tile confirmations. Stored factual/control frames showed Lolo in a
different sub-tile position or facing pose. The evaluator now removes the
union of the factual/control blue-and-white player footprints plus a small
outline halo, and requires 12 remaining changed pixels in a coarse cell. The
exact matched rerun
`spatial-v14-room2-player-footprint-local-effect-matched-audit-with-memory-d127-d5`
recorded the same 48 probes, three candidates, and eight controls, but zero
confirmations. The sole planning-configuration difference from the prior audit
was the new 12-pixel support threshold. No effect telemetry influenced action
selection, and the frozen audit passed.

This falsifies the current coarse-pixel transformation signal at the measured
all-heart plateau. Broad effects were animation; compact adjacent effects were
player pose. More option depth or promotion of those cells would manufacture
state rather than discover it.

## Resume fidelity and goal-exhaustion credit

Two temporary-memory bugs were found while reproducing the plateau. Emulator
and scene pixels were restored across runs, but assisted graph visits, position
visits, edge coverage, and option coverage were initially reset. Recursive
episodic seeding now restored 51 graph states and exactly 49 player positions
for the decision-127 audit. A later strict-to-assisted resume exposed a second
case: stale semantic fields from an older assisted ancestor overwrote a heart
visibly present in the current save state. Current hearts, player, and life are
now anchored to the resumed pixels unless the latest source decision itself
contains assisted state. Exact learned goal-exhaustion values are also carried
as temporary episodic memory.

The all-heart frontier map explains why preparation must be tested. The agent
reached 49 positions, including `(32,176)` four tiles below the chest. At
`(48,160)` the stored frame shows Lolo pressed against the pink moving entity;
from that endpoint every controller action was pixel-equivalent. The lower-left
corridor to the chest was blocked in this particular post-heart state. This is
an observed controllability failure, not an encoded entity label or rule.

An opt-in assisted rollback now treats a goal state as boundedly exhausted only
when all three conditions hold: its semantic graph state is repeated, no
recoverable archive frontier remains, and exact option search adds no endpoint.
It gives the exact milestone choice a small negative temporal-option sample and
restores the pre-milestone save state. In
`spatial-v14-room2-goal-exhaustion-rollback-v3-from-pre-final-heart-d37-d150`,
the final heart was collected at decision 1. Search continued through decision
146, reaching 64 graph states, 53 player positions, and 12 exhausted option
sources. At decision 147 the proof fired, learned a negative value for that
exact `LEFT 16` context, invalidated 358 post-heart descendants, and restored
the visible final heart at `(176,48)`. All 2,194 save states were released and
the frozen audit passed.

The next resumed run restored that negative value, filtered the original
choice, tried a different collection context, and rolled that context back
after its known post-heart frontier exhausted. Subsequent preparation rounds
sustained hundreds of decisions with the final heart visible, reached 74
cumulative detected positions, and verified more than a thousand exact options
in a single 100-decision round. No life was lost.

The result is a real improvement in credit assignment, not a room-clear result.
Later preparation rounds stopped adding option endpoints and cycled near
`(80,64)`/`(96,64)`. Visual comparison with the original pre-heart state showed
the lower-left pink entity in the same location. Exact-context rejection can
defer an irreversible choice, but the current graph still lacks an
object-centric objective for changing the dynamic blocker before retrying it.

## Telemetry

New events preserve every diagnostic and recovery decision:

- `counterfactual_control_probe`
- `counterfactual_control_confirmation`
- `counterfactual_control_escape_probe`
- `counterfactual_control_collapse_learned`
- `control_collapse_state_restored`
- `causal_observation_recovery_suppressed`
- `causal_observation_intervention_selected`
- `delayed_transition_probe`
- `delayed_transition_branch_selected`
- `anticipated_transition_observation`
- `post_dark_archive_branches_filtered`
- `generic_dark_transition_started`
- `generic_dark_transition_resolved`
- `known_scene_recovery_checkpoint_restored`
- `episodic_scene_memory_seeded`

`behavior_probe_selected` marks `control_collapse_recovery_probe` and
`matched_causal_observation` for each reserved action-duration pair. Run
summaries count these probes, collapses, checkpoint restores, descendant
invalidations, dark transitions, known-scene returns, and post-dark filters.
All frames, save/load operations, branch parentage, action durations, and
restores remain available for replay and visualization.

## Assisted Room 2 completion

An offline, unlabelled 16×15 visual-patch audit changed the diagnosis of the
all-heart frontier. It found that the lower-left pink patch disappeared after
the agent contacted the treasure at decision 74 of
`spatial-v14-room2-goal-exhaustion-rollback-v3-from-pre-final-heart-d37-d150`.
Stored frames confirmed Lolo entering the open treasure tile, the treasure
emptying, and the moving patch disappearing. The assisted detector had missed
the event because it required a dark or scene-changing target frame.

Treasure acquisition now also accepts the directly observed contact
transition: the source contains the open treasure, the target does not, and
the target player location equals the source treasure location. The acquired
state persists across archive restores and episodic resume. A 120-decision
frozen validation recognized the event immediately and retained
`human_prior_chest_obtained=true` throughout.

Patch comparison then isolated a persistent post-treasure change at coarse
cell `(11,1)`, the upper-right room boundary. The agent had already discovered
a controllable branch on the right ladder. Resuming that opaque committed save
state at `(176,96)`—without forcing any action—let fresh best-first branching
climb to `(176,32)`. At decision 23 the selected `UP 16` entered the changed
boundary. A dark transition began at decision 24 and resolved to the visibly
novel Room 3 at decision 27. The evaluator confirmed a stable scene distance
of `0.1455488` at decision 31, above its `0.05` room-change threshold.

The proof run is
`spatial-v14-room2-right-ladder-local-branch-from-d17-d40`: 31 committed
decisions, six archive restores, no life loss, balanced save-state ownership,
and a passing frozen-parameter audit. A matched phase-conditioned position
novelty ablation was identical to control for 40 decisions, so that optional
signal remains disabled by default and is not credited for the result.

## Assessment and next experiment

The approach is still viable, but Room 2 is no longer mainly a reward problem.
The agent has learned the hearts frontier and can preserve it. The remaining
failure is control-preserving multi-action search around a dynamic hazard and
blocked routes.

The next native experiment should begin from a live, all-heart state before the
hazard corridor—not from an already collapsing intermediate—and retain the new
duration-matched observation, exact hazard values, lineage cleanup, and
known-scene rollback for the entire search. The success criteria are:

1. no HUD life decrement;
2. no regression of the four persistent disappearances;
3. at least one new controllable state beyond the current lower or upper
   junction;
4. a stable scene transition only after a visually novel post-dark layout.

If that search still exhausts every control-preserving micro-action, the next
architectural change should be explicit multi-action option search over
verified save states, with controllability as a hard constraint and persistent
pixel changes as the progress value. Increasing a semantic reward would not
address the demonstrated timing and lineage failures.

The semantic-frontier comparison and goal-exhaustion preparation runs have now
passed those gates. Short exact options work, but physical archive ordering is
the dominant reachability gain, and coarse persistent-effect detection does not
survive player-pose controls. Exhaustion credit successfully defers the final
heart, yet hundreds of preparation decisions leave the visible blocker in the
same place. The next architectural experiment should therefore learn and track
unlabelled local sprite/tile entities through time, score verified changes to
their position or controllability, and use that object state in the reversible
graph. It should retain exact options and milestone rollback as safety
mechanisms, without increasing reward or adding a room-specific action
sequence.

Room 3 exposed one additional symmetry bug in this mechanism. An immediately
selected action branch used its duration-matched `NOOP` contrast to arm a
follow-through observation, but the same branch lost that evidence when first
stored in the archive. Restoring a short interaction-button effect could
therefore move away before its delayed visual consequence resolved. Archived
branches now retain the same matched-neutral option eligibility, contrast, and
counterfactual count as immediately selected branches.

A subsequent controlled effect-frontier ablation showed that persistence and
action causality are still insufficient: decrementing the visible shot counter
survived stability, phase, player-footprint, and leave-one-action-out controls.
That is a real persistent effect but not evidence of puzzle progress. Effect
frontier admission now additionally requires bounded controllability gain.
From factual and action-ablated endpoints with the same detected player tile,
the evaluator branches every directional action sequence through a configured
depth and requires the factual state to reach at least one player position
unavailable from control. Both reachable sets, exact action paths, and
pixel-outcome spreads are logged. The default remains one step; Room 3 uses two
steps to test effects whose navigational consequence is not immediately
adjacent.

The native Room 3 depth-two audit verified 180 option branches and tested
three persistent effects. In every case the factual and action-ablated states
had the same bounded reachable player tiles: `(96,96)`, `(96,112)`, and
`(96,128)`. No effect entered the frontier. The run released all 198 save
states and passed the frozen-checkpoint audit. This rules out both a deeper
option horizon and a delayed two-move navigation benefit at the tested
post-interaction state; the next representation must distinguish recurring
unlabelled entity state rather than treating every compact screen change as an
undifferentiated effect.

The next opt-in frontier implements that distinction without object names or
room rules. It learns quantized local appearance prototypes from pixels during
the room, reconstructs the facing direction before each controlled option
action, and intersects the persistent factual/control change with the forward
interaction ray. Admission requires endpoint-safe persistence, exact
leave-one-action-out causality, a changed anonymous patch appearance, and
failure to match a nearby neutral animation phase. The anonymous appearance
hash becomes part of the reversible graph state even before movement changes.
A synthetic control confirms that a two-action local transformation is
archived, while an equally persistent remote display change is rejected.
Native telemetry then exposed a planning-state alias: same-tile option nodes
were deduplicated without their action-derived facing, and effect candidates
were later deduplicated by changed cells without player endpoint or facing.
Both stages now preserve facing. This keeps a direction-setting action followed
by an interaction button distinct across orientations without supplying an
object label or a room solution.

The next native audit showed that facing alone was not enough: the first
interaction changed a remote visible state, but option deduplication collapsed
that state before a later interaction could be tested. Temporary option nodes
now include an anonymous absolute appearance hash over action-changed cells,
and the entity-search track admits neutral waits. With four-frame actions and
depth four, the frozen Room 3 agent discovered and archived
`RIGHT, A, RIGHT, A`. The final `A` was causally necessary for persistent
appearance changes at coarse cells `(7,6)` and `(8,6)`; both changed anonymous
prototype assignments, failed the neutral phase match, and created an unseen
reversible graph context. The audit verified 273 option branches, released all
285 save states, and passed the frozen-checkpoint audit.

An exhaustive 64-candidate audit then exposed a second loss of information.
The previously selected `RIGHT, A, RIGHT, A` state reappeared at rank 17, but
ranks 46--51 revealed a different persistent local appearance after only one
interaction, represented by `RIGHT, A, RIGHT`. Both states affected the same
anonymous cells but had different appearance signatures. The earlier search
retained only its highest-scoring endpoint and released the other verified save
state. Option search now keeps one capacity-bounded representative for every
distinct verified anonymous entity context. A synthetic two-stage transform
confirms that both intermediate and final states remain branchable without
assigning either a name or a game-specific value.

The first longer native continuation exposed an independent save-state
ownership defect: a causal-affordance archive entry and a pending life-hazard
recovery checkpoint could both reference the same emulator handle. Restoring
another archive correctly released the hazard checkpoint, but left the
affordance entry dangling. Affordance checkpoints now clone the source save
state and restore the live endpoint immediately, so each long-lived owner has
one handle. A release-after-clone regression verifies that releasing the
original owner cannot invalidate the archived clone.

## Verification

The complete test suite passes: 199 tests, with 3 expected skips. Every
completed native run reported `frozen_evaluation_audit=pass`.
