# Unlabeled spatial causal model — 2026-08-10

## Decision

The whole-screen latent model remains the frozen baseline. The next persistent
model is an action-conditioned spatial-token ensemble trained only from RGB
frames, controller actions, action durations, and branch grouping. Its first
effect-prediction gate passes. It is **not yet connected to the planner**
because its multi-step changed-region rendering has not beaten frame
persistence on the held-out development split.

This is deliberately separate from `human_prior_v1` and `human_prior_v2`.
Assisted-policy runs cannot be imported into a strict dataset: the sequence
store is permanently bound to either `strict` or `assisted` provenance.

## Model slice

`SpatialTokenDynamicsModel` keeps an 8×8 learned feature map rather than
compressing the whole frame to one vector. Convolutional dynamics share the
same transition rule across locations. Independent heads receive the hardware
action and duration and predict:

- successor spatial tokens;
- an unlabeled map of where pixels will change;
- ensemble disagreement for epistemic uncertainty.

No grid cell is named. The effect target is adaptive pooling over raw pixel
differences, so the model receives no object, room, player, hazard, reward, or
completion label. A sparse-change renderer copies pixels predicted to remain
stable and renders only the predicted effect region. Token prediction uses a
scale-invariant cosine objective because learned-token magnitude has no fixed
semantic meaning.

Complete save-state branch groups are sampled together. This retains exact
counterfactual alternatives and matched NOOP controls while avoiding the
memory cost of decoding the entire 393 MB compressed sequence store.

## Reproducible MPS result

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

## Run locally

```bash
source .venv/bin/activate
python -m lolo_agent.spatial_train \
  --dataset experiments/lolo1-medium/dataset \
  --checkpoint experiments/lolo1-spatial-smoke/checkpoints/spatial-v1-context250-e15.pt \
  --metrics experiments/lolo1-spatial-smoke/metrics/spatial-v1-context250-e15.json \
  --reward-track strict \
  --max-groups 250 \
  --epochs 15 \
  --batch-size 16 \
  --token-size 32 \
  --action-size 8 \
  --ensemble-size 3 \
  --grid-size 8 \
  --planning-horizon 3 \
  --seed 17
```

## Next gate

1. Replace full-color changed-region synthesis with a learned local residual or
   patch-copy transition so true movement can beat persistence without
   regenerating the background.
2. Split validation by run and then by room, not only by branch root, to measure
   progressively stronger cross-context transfer.
3. Scale strict causal-group training only after those splits are fixed.
4. Connect effect and uncertainty predictions to planning only after all
   planner-integration checks pass unchanged.
5. Add learned reachability, reversibility, and reset-risk heads over spatial
   tokens. Those outcomes must be derived from visual trajectories rather than
   heart, enemy, life, or room labels.
