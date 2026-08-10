# Unlabeled spatial causal model — 2026-08-10

## Decision

The whole-screen latent model remains the frozen baseline. The selected
predictive candidate is now v10: a 16×16 action-conditioned spatial-token
ensemble with anchored cumulative flow rendering. It is trained only from RGB
frames, controller actions, action durations, and branch grouping. It beats
frame persistence on all three held-out horizons and on two consecutive unseen
native Room 1 continuations.

The spatial model is still **selection-disabled by default**. A 0.25 frozen
tie-break A/B test changed behavior but did not clearly improve 60-decision
exploration, so promotion as an action objective is not yet justified.

This is deliberately separate from `human_prior_v1` and `human_prior_v2`.
Assisted-policy runs cannot be imported into a strict dataset: the sequence
store is permanently bound to either `strict` or `assisted` provenance.

## Model slice

`SpatialTokenDynamicsModel` keeps a learned spatial feature map rather than
compressing the whole frame to one vector. V10 uses 16×16 tokens; the earlier
8×8 grid could locate affected regions but smeared sub-tile motion.
Convolutional dynamics share the same transition rule across locations.
Independent heads receive the hardware action and duration and predict:

- successor spatial tokens;
- an unlabeled map of where pixels will change;
- ensemble disagreement for epistemic uncertainty.

No grid cell is named. The effect target is adaptive pooling over raw pixel
differences, so the model receives no object, room, player, hazard, reward, or
completion label. A sparse-change renderer copies pixels predicted to remain
stable. The selected renderer warps local source pixels with learned bounded
flow and adds a bounded RGB residual instead of regenerating the whole screen.
Its auxiliary changed-region objective bypasses the stable background, and its
anchored rollout predicts each trajectory endpoint from the initial frame plus
cumulative learned effect masks. This avoids recursive pixel smearing. Token
prediction uses a scale-invariant cosine objective because
learned-token magnitude has no fixed semantic meaning.

Complete save-state branch groups are sampled together. This retains exact
counterfactual alternatives and matched NOOP controls while avoiding the
memory cost of decoding the entire 393 MB compressed sequence store.

## Reproducible v1 MPS result

The best fixed-development run is:

```text
experiments/lolo1-spatial-smoke/checkpoints/spatial-v1-context250-e15.pt
experiments/lolo1-spatial-smoke/metrics/spatial-v1-context250-e15.json
```

It ran locally on Apple MPS with seed 17, 250 sampled causal groups, 15 epochs,
and an ensemble of three heads. The sample contained 2,277 sequences, 245
counterfactual roots, and 245 roots with both NOOP and active controls. Group
splitting produced 1,854 training and 423 held-out sequences.

| Measurement | Before | After / baseline |
| --- | ---: | ---: |
| Mean spatial-effect F1 | 0.177 | 0.699 |
| Mean effect L1 | 0.480 | 0.051 |
| Mean balanced effect L1 | ~0.358 | 0.124 |
| Always-no-change balanced effect L1 | — | 0.292 |
| Mean within-horizon uncertainty/error correlation | negative | +0.567 |
| Effect-weighted pixel L1 | ~0.052 | 0.00890 |
| Frame-persistence effect-weighted L1 | — | 0.00877 |

The effect gate passes: localization beats both the untrained model and the
class-balanced no-change baseline, and uncertainty is positively calibrated
at every horizon. The planner gate remains false because rollout error is
about 1.6% above persistence. Extending the identical configuration to 20
epochs worsened the mean to 0.00906, so the threshold was not relaxed and the
longer checkpoint was not selected.

This is a development split over causal root groups, not the final withheld-
room evaluation. It demonstrates cross-context prediction within collected
strict experience; it does not yet demonstrate room or sequel generalization.

## V2 run-held-out result and native shadow check

The selected v2 checkpoint is:

```text
experiments/lolo1-spatial-v2/checkpoints/spatial-v2-runheldout-flow-long250-e15.pt
experiments/lolo1-spatial-v2/metrics/spatial-v2-runheldout-flow-long250-e15.json
```

It ran on Apple MPS with seed 17, 250 complete causal groups, 15 epochs,
and 250 reserved groups containing multi-action trajectories. Source-run
provenance is part of group identity. The size-balanced split contained 1,920
training sequences from 11 runs and 510 validation sequences from one entirely
held-out run, with no run appearing in both partitions.

| Offline measurement | V2 held-out value |
| --- | ---: |
| Mean spatial-effect F1 | 0.5944 |
| Mean effect L1 | 0.07876 |
| Mean balanced effect L1 | 0.12388 |
| Always-no-change balanced effect L1 | 0.28753 |
| Uncertainty/effect-error correlation | +0.32275 |
| Effect-weighted pixel L1 | 0.00954488 |
| Frame-persistence effect-weighted L1 | 0.00954870 |

The offline planner gate passes, but only by about 0.04%. Horizon steps one
and two beat persistence; step three is slightly worse. This is evidence that
the local renderer can learn real motion, not enough evidence to let it steer.

The checkpoint was therefore attached to the existing frozen planner with a
hard-coded selection weight of zero for five native Room 1 decisions:

```text
experiments/lolo1-spatial-v2/shadow_evaluations/spatial-v2-shadow-room1-5
```

The run evaluated 30 real save-state branches. Mean effect F1 was 0.577, but
only 3/30 branches (3/24 branches with nonzero observed visual effect) beat
persistence. Mean effect-weighted error was 0.00457026 versus 0.00456985 for
persistence. Both persistent models passed before/after parameter-hash audits,
and the manifest records `selection_weight: 0.0`.

This native result overrides the narrow offline pass for promotion purposes.
The checkpoint remains a shadow model and cannot affect selected actions.

## Renderer audit and v10 promotion result

The native failures exposed a persistence-collapse problem: the 8×8 effect
head learned where change occurred while the renderer copied almost every
pixel. V5 predicted only `0.0000156` mean pixel change on native branches and
won 0/102 comparisons, despite passing its aggregate offline gate.

Controlled variants isolated the bottleneck:

| Variant | Held-out result | Native result |
| --- | --- | --- |
| V6, 8×8 changed-patch redraw | 0.01184 vs 0.00955 persistence; fail | Not run |
| V7, 8×8 directly supervised flow | 0.01167 vs 0.00955; fail | Not run |
| V8, 16×16 recursive flow | One-step 0.01002 vs 0.01079; later horizons fail | Not run |
| V9, 16×16 anchored flow | 0.01399 vs 0.01424; pass | 51/102 wins, but mean 0.00672 vs 0.00638; fail |

V10 warm-started v9 and imported four strict sequential native branch logs.
Source-run splitting placed three native runs in training while keeping both
`legacy-segment:cycle-000016` and the complete
`spatial-v5-shadow-room1-future20` run held out. The 621-sequence validation set
had no source-run overlap with training.

```text
experiments/lolo1-spatial-v10/checkpoints/spatial-v10-native-adapt-e5.pt
experiments/lolo1-spatial-v10/metrics/spatial-v10-native-adapt-e5.json
```

| V10 held-out horizon | Model | Persistence |
| --- | ---: | ---: |
| 1 | 0.009141 | 0.010091 |
| 2 | 0.013444 | 0.013602 |
| 3 | 0.015590 | 0.016440 |
| Mean | 0.012725 | 0.013378 |

The predicted/actual visual-change ratio was 1.160, mean spatial-effect F1 was
0.617, and uncertainty/effect-error correlation was +0.428. Both the effect
gate and the stricter planner-integration prediction gate passed.

V10 was then frozen and evaluated with selection weight zero on two later,
unseen continuations:

| Native run | Branch wins | Model mean | Persistence mean |
| --- | ---: | ---: | ---: |
| `spatial-v10-shadow-room1-future20` | 47/108 | 0.005022 | 0.005338 |
| `spatial-v10-shadow-room1-confirm20` | 60/114 | 0.005555 | 0.006362 |
| Combined | 107/222 | 0.005295 | 0.005864 |

The combined native improvement is 9.69%. Both the spatial and baseline model
parameter-hash audits passed in both runs.

## Planner tie-break ablation

The frozen spatial score can now be enabled explicitly with
`--spatial-selection-weight`; it remains zero by default. Two 60-decision runs
started from the exact same save-state provenance:

| Measurement | Weight 0 | Weight 0.25 |
| --- | ---: | ---: |
| Unique frames | 93 | 84 |
| Unique scenes | 3 | 3 |
| Unique causal signatures | 51 | 47 |
| Maximum persistent-frontier value | 6.956 | 7.184 |
| Archive restores | 7 | 8 |
| Delayed-return recoveries | 2 | 3 |
| Room transition | No | No |

The weighted run reached a slightly higher peak frontier but explored fewer
states and looped/restored more. This is not a clear win, so weight 0 remains
the plan of record.

## Run locally

```bash
source .venv/bin/activate
python -m lolo_agent.spatial_train \
  --dataset experiments/lolo1-medium/dataset \
  --checkpoint experiments/lolo1-spatial-v9/checkpoints/spatial-v9-grid16-anchored-flow-e15.pt \
  --metrics experiments/lolo1-spatial-v9/metrics/spatial-v9-grid16-anchored-flow-e15.json \
  --reward-track strict \
  --validation-split run \
  --max-groups 250 \
  --minimum-multistep-groups 250 \
  --epochs 15 \
  --batch-size 16 \
  --token-size 32 \
  --action-size 8 \
  --ensemble-size 3 \
  --grid-size 16 \
  --renderer flow_residual \
  --renderer-rollout anchored \
  --max-flow-pixels 32 \
  --effect-mask-power 1 \
  --pixel-loss-weight 5 \
  --changed-region-loss-weight 1 \
  --planning-horizon 3 \
  --seed 17
```

## Next gate

1. Suppress false predicted change for NOOP, A, B, and blocked movement without
   weakening the large gains on effective directional movement.
2. Replace raw predicted-effect curiosity as the selection bonus with a
   calibrated usefulness or reachability objective, then repeat paired runs.
3. Repeat frozen prediction and planning ablations in later rooms and across
   multiple source-run folds before any default-on promotion.
4. Add learned reachability, reversibility, and reset-risk heads over spatial
   tokens. Those outcomes must be derived from visual trajectories rather than
   heart, enemy, life, or room labels.
