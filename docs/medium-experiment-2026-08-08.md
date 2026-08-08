# Medium experiment: 2026-08-08

This experiment tested whether the duration-aware durable loop could move the
dataset beyond boot imagery and whether a frozen planner could autonomously
reach interactive gameplay.

## Configuration

- 10 collect–train–evaluate cycles on MPS
- 20 roots and 3 branches per root per cycle
- planning and training horizon 3
- controller durations 1, 2, 4, 8, and 16 frames
- one training epoch per cycle
- 600 persisted visual sequences
- 1,800 real branched controller transitions

Artifacts are stored locally in `experiments/lolo1-medium/` and intentionally
excluded from Git because they contain checkpoint data and game imagery.

## Dataset result

The collector stored 257 unique RGB observations spanning 23 coarse visual
scenes. Its exploration was balanced without supplied game semantics:

| Dimension | Counts |
| --- | --- |
| Actions | 168–220 uses per controller action |
| Durations | 344–379 uses per duration |
| Nonzero visual transitions | 623 / 1,800 |

Evaluator-side visual inspection confirmed that collection reached:

- the title and start/password interface;
- the castle sequence;
- the opening story sequence;
- Floor 1 with Lolo at multiple positions;
- a game-over/password state.

These labels were assigned only after collection and were never exposed to the
agent or model.

## Model result

Held-out RGB L1 prediction error improved across all three modeled decisions:

| Cycle | Horizon 1 | Horizon 2 | Horizon 3 |
| ---: | ---: | ---: | ---: |
| 1 | 0.363579 | 0.376114 | 0.375117 |
| 3 | 0.209504 | 0.206669 | 0.205028 |
| 5 | 0.162166 | 0.163127 | 0.160852 |
| 8 | 0.132716 | 0.133896 | 0.133228 |
| 10 | 0.110654 | 0.112996 | 0.113037 |

Ensemble disagreement continued to increase with horizon, but its correlation
with prediction error was `-0.0820` at cycle 10. Deep imagined rollouts must
therefore remain advisory and real first-action verification remains mandatory.

## Frozen-planner findings

The original 20-decision evaluation was too short and remained in intro
animation. Longer frozen audits exposed three general exploration failures:

1. action-independent animations produced timestamp branches that could restore
   the agent backward through a cutscene;
2. password-character edits generated effectively unlimited exact-frame
   novelty inside one coarse scene;
3. the duration-expanded verification budget could test several durations of
   one button while omitting other buttons.

The planner was changed without introducing game-specific concepts:

- matched equal-duration control probes detect autonomous visual dynamics;
- autonomous dynamics select the longest neutral wait and do not archive
  equivalent timestamp alternatives;
- novelty is hierarchical, reducing exact-frame reward as one coarse scene is
  repeatedly explored;
- archive pruning preserves minority scenes instead of allowing one scene to
  consume all memory;
- least-tested distinct controller buttons are verified before extra durations.

With these changes, the frozen cycle-10 checkpoint progressed through menus and
castle into the opening story sequence within 210 decisions. It did not yet
reach Floor 1 in frozen planning, so this is a navigation milestone rather than
a puzzle-solving result.

## Next research target

The next target is a frozen trajectory that reliably reaches Floor 1 and then
spends most of its interaction budget there. The remaining issue is delayed
progress: a locally novel action can skip or reset a sequence and only reveal
that it returned to a known region several decisions later.

The next model/planner addition should learn multi-step return-to-known-state
risk and use it to prefer persistent visual frontiers. It should be learned
from the transition graph and save-state experiments, not from menu labels or a
supplied definition of progress.
