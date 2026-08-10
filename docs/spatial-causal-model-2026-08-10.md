# Unlabeled spatial causal model — 2026-08-10

## Decision

The whole-screen latent model remains the frozen baseline. The next persistent
model is an action-conditioned spatial-token ensemble trained only from RGB
frames, controller actions, action durations, and branch grouping. The v2
local flow/residual renderer passes its offline run-held-out gate by a narrow
margin. It remains **observational only** because native Room 1 shadow
evaluation did not reliably beat frame persistence on verified emulator
branches.

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
stable. Its v2 changed-region candidate warps local source pixels with learned
bounded flow and adds a bounded RGB residual instead of regenerating the whole
screen. Token prediction uses a scale-invariant cosine objective because
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

## Run locally

```bash
source .venv/bin/activate
python -m lolo_agent.spatial_train \
  --dataset experiments/lolo1-medium/dataset \
  --checkpoint experiments/lolo1-spatial-v2/checkpoints/spatial-v2-runheldout-flow-long250-e15.pt \
  --metrics experiments/lolo1-spatial-v2/metrics/spatial-v2-runheldout-flow-long250-e15.json \
  --reward-track strict \
  --validation-split run \
  --max-groups 250 \
  --minimum-multistep-groups 250 \
  --epochs 15 \
  --batch-size 16 \
  --token-size 32 \
  --action-size 8 \
  --ensemble-size 3 \
  --grid-size 8 \
  --renderer flow_residual \
  --planning-horizon 3 \
  --seed 17
```

## Next gate

1. Repeat run-held-out validation across multiple source-run folds rather than
   relying on one held-out run and a 0.04% margin.
2. Improve the native verified-branch persistence win rate materially, with
   special attention to suppressing false change on visually neutral actions.
3. Extend zero-weight shadow evaluation across longer Room 1 trajectories and
   later rooms while preserving frozen parameter hashes.
4. Give the model nonzero selection weight only after both offline and native
   promotion gates pass unchanged.
5. Add learned reachability, reversibility, and reset-risk heads over spatial
   tokens. Those outcomes must be derived from visual trajectories rather than
   heart, enemy, life, or room labels.
