# Control-preserving Room 2 search

Date: 2026-08-10

## Result

The strict pixel-only agent collected all four Room 2 hearts for the first
time, retained the resulting persistent screen changes, and reached both the
upper and lower halves of the room. It has not yet opened the chest or cleared
Room 2.

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

The semantic-frontier comparison has now reached that gate. The next experiment
should plan short, verified multi-action options inside each assisted world
context and prefer options that reach a new context or a new player position
without losing immediate control. It should not increase the heart/chest reward
or add a room-specific action sequence.

## Verification

The complete test suite passes: 168 tests, with 3 expected skips. Every
completed native run reported `frozen_evaluation_audit=pass`.
