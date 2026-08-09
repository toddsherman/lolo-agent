# Pixel-heart reward experiment (2026-08-10)

## Question

Does a small, explicitly labelled human prior—visible hearts are desirable—make
the frozen cycle-16 agent's Room 2 exploration more effective without supplying
a room solution or reading emulator memory?

This is a separate `human_prior_v1` evaluation track. The strict rule-free
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
| `cycle-000016-floor2-human-prior-hearts-v8-navigation1-detourgrace2-5000` | active | — | Current overnight run with bounded detour grace |

The second heart is the important positive control. It is protected by rocks and
bushes, so the successful route is not greedy: observed distance goes 3→4, then
3→3 around the obstacle, then 2→1→collection. The 44-decision result replicated
in v3, v6, and v7, versus decision 548 with sparse milestones alone.

## Negative results that changed the design

Hard distance-monotonic archive filtering is invalid for this domain. It makes
Manhattan local minima absorbing even though the action planner itself may
choose a `-1` detour. Creating an archive for every `+1` step is also harmful:
it expands the restore pool faster than it adds useful causal alternatives.

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

Dedicated events include `human_prior_calibrated`,
`human_prior_goal_choice`, `human_prior_regressive_archives_filtered`, and
`human_prior_navigation_recovery_suppressed`. Every committed and rejected
branch still retains its action, duration, source/target frames, intrinsic
components, archive state IDs, and replay ordering.

## Current assessment

The explicit heart prior is a productive experimental direction, but it should
not replace the strict evaluation track. It has demonstrated a repeatable order
of magnitude improvement on a nontrivial Room 2 milestone while preserving the
need for learned obstacle detours. The next gates are collecting the final heart,
discovering/opening the chest from pixels, completing Room 2, and then testing
whether the same fixed semantic configuration improves withheld rooms without
additional tuning.
