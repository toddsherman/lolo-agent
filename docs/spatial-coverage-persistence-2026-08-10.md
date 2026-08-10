# Rule-free spatial coverage and persistent-change experiment

Date: 2026-08-10

## Question

Room 2 exposed two separate strict-track failures:

1. the agent repeatedly explored the lower corridor after collecting one heart;
2. longer runs could collect two hearts, but later save-state recovery restored
   worlds in which those hearts were present again.

The experiment asked whether both failures could be reduced without object
labels, ROM memory, demonstrations, or a semantic reward.

## Changes

### Global controlled-cell coverage

`--causal-cell-coverage-weight` scores the 16x15 coarse screen cells whose
pixels differ between an action endpoint and a duration-matched `NOOP`
endpoint. A cell contributes `1 / sqrt(global visits + 1)`, averaged across
the action's changed cells. The score is attempt-global rather than local to a
behavioral context, and the same live score participates in archive ranking.

The default is zero, preserving prior behavior.

### Modal persistent-disappearance evidence

`--persistent-change-stability-decisions` learns a modal 4-bit intensity value
for each coarse cell. A non-modal value becomes active only after the required
number of committed observations. The modal baseline adapts before activation,
so a moving sprite does not permanently imprint its position from the first
frame.

`--persistent-change-minimum-value-drop` can restrict evidence to persistent
decreases in coarse intensity. While evidence is active, archive recovery
prefers states that preserve every active cell, but only if at least one such
state is available. Exhausted preserving branches therefore do not make older
alternatives unreachable. Evidence retires only after the learned baseline
itself repeatedly returns; an unrelated temporary overlay is ambiguous and
does not erase it.

Both controls default to zero. No sprite prototype, object location, game
rule, or semantic outcome is visible to this mechanism.

## Native evaluation

All runs used the frozen cycle-16 model, MPS, the same Room 2 entry save state,
durations `1,2,4,8,16`, nine verified actions, and evaluator-only stable scene
change detection. Heart and player measurements below were computed offline by
the labeled pixel evaluator and were never exposed to the strict policy.

| Run | Decisions | Unique frames | Player tiles | Highest row | Best hearts remaining | Final hearts remaining |
|---|---:|---:|---:|---:|---:|---:|
| uninterrupted baseline | 100 | 243 | 11 | 176 | 3 | 3 |
| coverage weight 4 | 100 | 389 | 23 | 144 | 2 | 2 |
| coverage weight 4 | 200 | 626 | 32 | 112 | 2 | 4 |
| modal persistence 4 + coverage 4 | 100 | 314 | 22 | 144 | 2 | 3 |
| unrestricted persistence 3 + coverage 4 | 100 | 312 | 18 | 144 | 3 | 3 |
| drop 4, persistence 3 + coverage 4 | 100 | 384 | 22 | 144 | 2 | 2 |
| drop 4, persistence 3 + coverage 4 | 250 | 703 | 30 | 112 | 2 | 2 |

The coverage-only 100-decision run collected the two lower hearts at decisions
18 and 73. The 200-decision extension eventually restored pre-collection
states; all four hearts were present at the final decision. That failure ruled
out treating exploration breadth alone as progress memory.

The accepted persistence configuration was:

```text
--causal-cell-coverage-weight 4.0
--persistent-change-stability-decisions 3
--persistent-change-minimum-value-drop 4
```

Its 100-decision run collected the lower hearts at decisions 18 and 63 and
ended with both still absent. Exactly two persistent-disappearance cells
activated; none retired and preservation never lacked an alternative.

The uninterrupted 250-decision run reproduced those milestones and never
regressed them. It produced 703 unique frames, 1,611 verified branches, 71
archive restores, 53 committed causal signatures, and 31 first-visited causal
cells. Persistent preservation filtered 829 archive branches across 17
recovery events, with zero unavailable fallbacks. It reached player row 112
but did not collect either upper heart, open the chest, or transition rooms.

An evaluator-only scan of all 1,611 verified endpoints found 151 with four
hearts, 351 with three, and 1,109 with two. No endpoint had fewer than two.
Therefore the planner did not merely score or retain a third-heart branch
incorrectly; its one-step branch set never reached one. The intervention trace
contained 49 behavioral source abstractions and 90 distinct
state/action/duration edges, with 35 repeated edge commits. Raw frame identity
showed only six repeats because animation fragments equivalent states, while
scene identity over-collapsed the trace. Behavioral abstraction is therefore
the supported key for the next search frontier.

One marker retired at decision 247 because Lolo temporarily overlaid a cleared
cell. The implementation was corrected after this run so only repeated return
to the learned baseline retires evidence. This did not change the run's game
state: both lower hearts remained absent through decision 250.

## Rejected variants

- Prioritizing causal outcomes from the current context did not prevent the
  decision-27 rollback. Ordinary movement over-fragments the current causal
  context graph, so no matching successor was available.
- A fixed initial-frame persistence baseline activated on Lolo's starting
  footprint before any collectible changed. The run was stopped as invalid.
- Stability 3 without a minimum value drop activated transient neighboring
  appearances and reduced exploration to one collected heart.
- Stability 4 preserved the first disappearance but reacted one observation
  too late to protect the second before archive recovery.

## Assessment and next step

The result is positive but not a room solution. The agent now demonstrates two
reusable, pixel-derived concepts: globally novel controlled regions and
temporarily preserved persistent disappearance. These improve exploration and
prevent a concrete class of save-state regression without naming hearts.

The remaining failure is search organization. After reaching row 112, the
one-step verified policy and stagnation-triggered archive recovery cycle among
middle/lower states instead of systematically expanding multi-action routes to
the upper corridor. The next change should turn the save-state archive into an
explicit best-first frontier: expand each pixel-distinct state/action/duration
edge once using the behavioral abstraction key, retain parent links, and rank
unexpanded endpoints by global causal coverage plus preserved-change
compatibility. This is preferable to adding more reward weight because it
directly addresses repeated expansion and should produce auditable paths,
attempt counts, and replayable solution traces.

## Artifacts

- Baseline:
  `experiments/lolo1-spatial-v14/baselines/spatial-v14-room2-uninterrupted-cycle16-d100`
- Coverage comparison:
  `experiments/lolo1-spatial-v14/coverage_evaluations/spatial-v14-room2-global-causal-coverage-w4-d100`
- Coverage-only long run:
  `experiments/lolo1-spatial-v14/coverage_evaluations/spatial-v14-room2-global-causal-coverage-w4-d200`
- Accepted 100-decision run:
  `experiments/lolo1-spatial-v14/persistence_evaluations/spatial-v14-room2-persistent-drop4-s3-coverage-w4-d100`
- Accepted 250-decision run:
  `experiments/lolo1-spatial-v14/persistence_evaluations/spatial-v14-room2-persistent-drop4-s3-coverage-w4-d250`

All native runs reported `frozen_evaluation_audit=pass`.

Subsequent behavioral best-first, control-collapse, duration-refinement, and
dark-transition results are documented in
[`control-preserving-search-2026-08-10.md`](control-preserving-search-2026-08-10.md).
