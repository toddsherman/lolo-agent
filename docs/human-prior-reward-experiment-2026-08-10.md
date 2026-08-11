# Pixel-heart reward experiment (2026-08-10)

## Question

Does a small, explicitly labelled human prior—visible hearts are desirable—make
the frozen cycle-16 agent's Room 2 exploration more effective without supplying
a room solution or reading emulator memory?

The heart-only runs below used the separate `human_prior_v1` evaluation track.
The chest/life extension is labelled `human_prior_v2`. The strict rule-free
track remains the default and is unchanged. Both tracks receive only pixels,
controller outcomes, and save states.

## Reward design

The semantic track uses a fixed 16×16 pixel prototype to discover heart slots.
It also locates the player from visible sprite colours, rejecting animated water
colours. All detections and rewards are recorded on every verified branch.

- A collected heart gives `+25`.
- Collecting the final heart gives an additional `+75`.
- Moving one tile closer to the nearest visible heart gives `+1`; moving one
  tile farther gives `-1`.
- Navigation shaping is a state-potential difference, so an out-and-back pair
  has zero semantic return.
- Only heart collection clips intrinsic score before adding the milestone.
  Navigation shaping does not clip or force an action.
- Best collected-heart count persists across save-state restores. Distance is
  not monotonic because obstacle detours can require moving farther away.
- For the semantic track only, distance contributes softly to the score of
  causal archives. It does not create an archive by itself.
- A distance-changing action receives a two-decision grace window before
  delayed-return recovery may abandon that new frontier.
- A bounded temporal-option observation window now takes precedence over
  visual-stagnation recovery. This prevents archive restoration from cutting
  off a delayed animation before the configured passive observations and the
  subsequent intervention can occur.
- When autonomous-animation grace expires, recovery is held for one additional
  decision and the planner must commit a non-NOOP controller probe. This keeps
  a repeated passive-animation classification from preempting the intervention
  that tests whether control has returned. Both the reserved turn and selected
  probe are recorded in telemetry.
- After the final heart, either of the two pixel-observed open-treasure
  animation frames becomes the navigation target using the same symmetric
  `+1`/`-1` potential. Contact followed by a dark or room-changing frame gives
  `+100`.
- The treasure prototype is the pink/white object at `(32,112)` in Room 2. The
  green lower-right object was explicitly rejected after inspecting the stored
  frames; it is not treated as the chest.
- Player localization rejects water and chest colours, then tracks the selected
  visual candidate continuously across adjacent positions. A distant blue
  entity cannot create semantic progress by being mistaken for a teleported
  player; archive states retain the tracked player slot explicitly.
- A life loss gives `-100` only when a dark transition is followed by a changed
  8×8 life glyph in the visible HUD. The pre-transition pixel context, action,
  duration, and trajectory endpoints are logged and remembered as a hazard.

## Matched Room 2 results

Every run below used the same frozen `cycle-000016.pt` checkpoint, reconstructed
Room 2 state, action durations, nine verified branches per decision, and 2,048
state archive. Runs were intentionally stopped at clean comparison points; the
event streams, frame store, CSV, graph artifacts, and evaluator annotations are
preserved in each run directory.

| Run | Decisions | Heart decisions | Result |
| --- | ---: | --- | --- |
| `cycle-000016-floor2-human-prior-hearts-v2-monotonic-5000` | 802 | 38, 548 | Sparse heart reward works, but slowly |
| `cycle-000016-floor2-human-prior-hearts-v3-navigation1-5000` | 100 | 6, 44 | Navigation gives a large improvement; unrestricted recovery regresses distance 2→5+ |
| `cycle-000016-floor2-human-prior-hearts-v4-navigation1-distanceguard-5000` | 210 | 6 | Hard distance monotonicity traps above the obstacle-protected heart |
| `cycle-000016-floor2-human-prior-hearts-v5-navigation1-softarchive-5000` | 115 | 6 | Archiving every closer step floods recovery with semantic checkpoints |
| `cycle-000016-floor2-human-prior-hearts-v6-navigation1-softcausalarchive-5000` | 105 | 6, 44 | Minimal shaping replicates the obstacle detour; oldest-first causal restore still regresses the third-heart frontier |
| `cycle-000016-floor2-human-prior-hearts-v7-navigation1-semanticarchive-5000` | 302 | 6, 44, 97 | First run to reach one remaining heart; final detour expands to distance 14 but recovery interrupts most steps |
| `cycle-000016-floor2-human-prior-hearts-v8-navigation1-detourgrace2-5000` | 5,000 | 6, 50, 72, 459 | Collected all four hearts; did not navigate back to the chest or clear Room 2 |

The second heart is the important positive control. It is protected by rocks and
bushes, so the successful route is not greedy: observed distance goes 3→4, then
3→3 around the obstacle, then 2→1→collection. The 44-decision result replicated
in v3, v6, and v7, versus decision 548 with sparse milestones alone.

## Negative results that changed the design

Hard distance-monotonic archive filtering is invalid for this domain. It makes
Manhattan local minima absorbing even though the action planner itself may
choose a `-1` detour. Creating an archive for every `+1` step is also harmful:
it expands the restore pool faster than it adds useful causal alternatives.

The first completed autonomous-animation trace also exposed a real death
sequence in Room 2: an active `RIGHT` option at decision 29 was followed by an
enemy attack, a black frame at decision 40, and a same-room reset whose visible
life glyph changed from 5 to 4 at decision 41. The pixel detector recognized
both frames, but simultaneous four-heart re-calibration initially replaced the
confirmed loss with a neutral re-analysis. That result is preserved as the v13
ablation. Re-calibration now preserves the original `-100` outcome, and the
hazard is credited to the causally supported active temporal option rather than
the final passive `NOOP`. Exact initiation decision and frame are retained for
telemetry.

The v14 baseline confirmed that credit alone is insufficient for efficient
save-state exploration: it learned `-100` for the correct option at decision
41, then spent over 100 additional decisions replaying the room and still had
two hearts remaining. In the assisted life-loss track, a causally supported
temporal option now retains its pre-action emulator state as a temporary safety
checkpoint. Normal completion releases it. A confirmed life loss instead
restores it on the next decision while keeping the learned hazard value, so the
exact lethal choice can be filtered without replaying the entire room or
spending another life. Checkpoint creation, release, confirmation, and restore
all have dedicated telemetry events and state IDs.

v15 proved the short causal rollback path, but also showed why it is not enough
for puzzle dead ends. It restored the first pre-`RIGHT` state and filtered that
exact choice, then encountered two distinct lethal `RIGHT` contexts later. It
never crossed x=80. The source v8 trace shows the final heart was collected at
`(48,48)` while Lolo remained above the water; the shooter activated before the
chest below the water was reachable. A longer-lived checkpoint is now retained
before each positive semantic milestone. If a later death occurs before room
completion, the death penalty is also credited to that milestone choice and the
agent rewinds before it. Positive milestone branches no longer override an
exact learned hazard. This lets the agent treat “collecting this now” as a
contingent, potentially unrecoverable decision and experiment with preparation
first, without encoding what preparation the room requires.

v16 validated the longer rollback twice, then found a branching-consistency
bug: an archive state created after the abandoned final-heart timeline could be
restored after rewinding before the heart. Milestone rollback now invalidates
and releases every archived descendant created after the milestone decision,
while preserving alternatives verified at the milestone decision itself. It
also truncates committed transition history from the abandoned future and
restarts the frontier trace at the restored state. The number of invalidated
descendants is logged on the restore event and in the run summary.

The second v17 rollback exposed duplicate/stale native handle ownership during
bulk invalidation. Cleanup now deduplicates releases by telemetry state ID (or
object identity when no ID remains), never releases a state still referenced by
a retained branch, and records stale-handle failures without aborting the
agent. This changes cleanup robustness only; stale descendants are removed from
the archive regardless.

The bounded v18 validation completed all 130 decisions normally. Two
pre-final-heart rollbacks invalidated 63 and 71 descendants (134 total). The
second cleanup encountered one already-released native handle, recorded it,
continued restoring, and did not reintroduce a zero-heart descendant. State
ownership balanced exactly at run close: 1,120 saves and 1,120 releases. The
agent tried multiple final-heart contexts while retaining the delayed hazards,
which is the intended behavior for autonomous preparation search.

The resulting direction is deliberately narrower: potential shaping guides
immediate experiments; causal novelty decides what deserves persistent state;
heart milestones preserve irreversible semantic progress; and a short temporal
grace lets newly observed detours branch before loop recovery intervenes.

## Telemetry

Relevant branch and commit fields include:

- `human_prior_known_heart_slots`
- `human_prior_source_hearts`, `human_prior_target_hearts`
- `human_prior_collected_heart_slots`, `human_prior_remaining_hearts`
- `human_prior_source_player_slot`, `human_prior_target_player_slot`
- `human_prior_source_heart_distance`, `human_prior_target_heart_distance`
- `human_prior_heart_reward`, `human_prior_all_hearts_reward`
- `human_prior_navigation_reward`, `human_prior_milestone_reward`
- `human_prior_goal_reward`, `human_prior_best_remaining_hearts`
- `human_prior_goal_phase`, `human_prior_source_chest_slot`,
  `human_prior_target_chest_slot`
- `human_prior_source_chest_distance`, `human_prior_target_chest_distance`,
  `human_prior_chest_completed`, `human_prior_chest_reward`
- `human_prior_source_life_signature`, `human_prior_target_life_signature`,
  `human_prior_life_counter_changed`, `human_prior_life_loss_confirmed`,
  `human_prior_life_loss_penalty`

Dedicated events include `human_prior_calibrated`,
`human_prior_goal_choice`, `human_prior_regressive_archives_filtered`, and
`human_prior_navigation_recovery_suppressed`. Version 2 adds
`human_prior_chest_completed`, `human_prior_dark_transition_observed`,
`human_prior_dark_transition_cleared`, and
`human_prior_life_loss_confirmed`. Temporal control handoff is visible through
`temporal_option_recovery_suppressed`, `autonomous_intervention_started`, and
`autonomous_intervention_selected`. Life rollback is recorded by
`life_hazard_checkpoint_created`, `life_hazard_checkpoint_released`, and
`life_hazard_state_restored`. Longer-lived semantic rollback adds
`goal_milestone_checkpoint_created` and
`goal_milestone_checkpoint_released`. Every committed and rejected
branch still retains its action, duration, source/target frames, intrinsic
components, archive state IDs, and replay ordering.

## Current assessment

The explicit heart prior is a productive experimental direction, but it should
not replace the strict evaluation track. It demonstrated a repeatable order of
magnitude improvement on a nontrivial Room 2 milestone and ultimately collected
all hearts while preserving the need for learned obstacle detours. The next
focused gate is returning to the now pixel-detected open chest and completing
Room 2 without repeating the now learned enemy hazard. The life-loss detector
has been validated against the first real 5→4 death sequence rather than only
synthetic frames.
After this assisted positive control, the same fixed configuration must be
tested without room-specific tuning, and the strict track still needs a learned
object-centric substitute for these semantic prototypes.

## Final v19 preparation-search baseline

`cycle-000016-floor2-human-prior-v19-preparation-search-1000` completed all
1,000 requested decisions with the cycle-16 neural parameters unchanged. It
did not complete the chest or clear Room 2.

The run investigated 8,463 actions, verified 6,525 branches, restored 271
archive branches, and produced 57,693 telemetry events. It detected four life
losses at decisions 48, 91, 132, and 492 and restored the pre-final-heart
milestone after each one. Rollback invalidated 487 descendant archive branches
in total. State ownership balanced exactly: 7,273 states were saved and 7,273
were released.

The final heart was nevertheless selected as a positive goal 11 times from
several player positions and approach actions. The agent learned hazards for
specific behavioral contexts, but it did not learn the reusable relationship
“prepare the room before causing the last-heart transition.” This is the final
baseline for the assisted exact-context architecture. Further room-specific
reward or rollback tuning is deferred in favor of the persistent spatial
causal model described in `spatial-causal-model-2026-08-10.md`.

## Reversible assisted world context

After the all-heart frontier was recovered, a stable assisted best-first graph
was added as a positive-control search mechanism. The first key used only the
visible goal state and player tile. It reached 45 player positions and chest
distance 4, but a longer continuation added no position after decision 211 and
did not complete the room.

The key now includes confirmed action-conditioned non-player coarse cells. It
does not label those cells as enemies, blocks, hazards, or any other object.
Directional effects are masked to remove player motion and remote animation;
the endpoint must retain counterfactual control immediately. Accepted cells are
represented by reversible parity so repeated inverse-like effects can revisit
an existing context instead of extending a path-history string.

In the 300-decision frozen comparison
`spatial-v14-room2-assisted-reversible-world-context-from-d100-d300`, the agent
visited 101 goal-plus-world graph states across 36 player positions, verified
1,503 branches, restored 123 archives, and produced 1,116 unique frames. The
position-only comparison visited 45 graph states, verified 819 branches,
restored 206 archives, and produced 457 unique frames. Neither run opened the
chest; both reached chest distance 4 and lost no life.

This supports the current plan of record: keep the reward weights fixed and
improve verified multi-action planning over the richer context. The strict
rule-free evaluation track remains separate and receives none of the heart,
chest, player, or life prototypes used by this assisted diagnostic.

## Physical-frontier and exact-option ablation

The reversible-context policy was extended to decision 503. Although it grew
from 101 to 181 assisted graph states after decision 300, it never exceeded 36
detected player positions and produced no chest, life, or scene milestone. This
confirmed that the remaining growth was dominated by dynamic context variants.

The assisted archive now ranks globally unseen detected player positions ahead
of context-only edges. A bounded exact option search is available only after
the current source has no cheaper unseen one-step endpoint. It verifies real
emulator action sequences from opaque save states, caches exhausted sources,
and archives only a positive milestone or a globally unseen endpoint.

At 120 matched frozen decisions from the same strict decision-100 state:

| Treatment | Positions | Restores | Option branches | Option commits |
|---|---:|---:|---:|---:|
| Prior reversible-context policy | 22 | 48 | 0 | 0 |
| Unseen-position priority only | 45 | 20 | 0 | 0 |
| Unbounded depth-3/beam-8 options | 46 | 20 | 1,092 | 1 |
| Local-gated depth-3/beam-8 options | 46 | 20 | 666 | 1 |

The option commit was the exact sequence `UP, UP`, which reached detected
`(96,96)` from `(96,112)`. The position-only ablation reached the other 45
positions. Thus archive ordering produced nearly all of the improvement, while
multi-action verification added one real endpoint and local gating cut its
rollout cost by 39%. None of these runs improved on chest distance 4 or cleared
Room 2, so further reward increases remain unsupported.

An endurance extension was stopped at decision 187 after 60 decisions without
a new player position. It reached 49 positions and 62 graph states, committed
four exact options, and verified 2,004 option paths, but still produced no
chest, life-loss, or room-transition event. The next assisted experiment should
therefore target longer options or persistent non-player changes from the
distance-4 boundary rather than repeat broad depth-3 search.

The targeted longer-option hypothesis was tested directly from the local
decision-89 distance-4 state. A depth-5/beam-16 run verified 1,656 exact paths
across nine searches and committed seven endpoints, but found zero endpoints
closer than distance 4 and zero chest completions. The frozen audit passed.
The next assisted mechanism should therefore value stable, matched-NOOP
non-player transformations rather than extend the movement horizon again.
