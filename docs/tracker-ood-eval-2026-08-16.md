# WP5 tracker out-of-distribution evaluation (2026-08-16)

Status: REPORT-ONLY (no gate); learnings §4.31 plan-change item (a)
Report: `experiments/lolo1-wp5/tracker-ood-report.json`
(content digest `ecc5336b…`, deterministic; regenerating reproduces it)
Code: `lolo_agent/tracker_ood_eval.py`; tests
`tests/test_tracker_ood_eval.py`

## Method

Tracker v2 (`controllable-tracker-v2-uncapped.pt`, held-out AUC 0.9997 on
its training distribution) is scored against DETECTOR-FREE counterfactual
localization ground truth extracted from the Room 3 paired-probe runs
v322–v326. The option search in those runs records, for every saved
parent state, each action edge's endpoint frame
(`human_prior_option_branch_verified`) plus a duration-matched NOOP
endpoint from the same state (`human_prior_option_local_neutral_verified`,
root-level `…_neutral_verified` at depth 1). Each such record is a
first-step counterfactual edge in exactly the sense of
`counterfactual_labels`; the edges are regrouped into causal roots
(keyed by saved state, never reused within a run) and labeled by
`label_counterfactual_root` itself — the same endpoint-difference,
4-connected-component, and leave-one-action-out corroboration rule that
produced the training labels, with the same explicit censoring. Labeled
arms with an empty controllable mask carry no localization evidence and
are excluded, mirroring training-example construction. This is an
evaluation-only use of assisted-collected archives; no training artifact
is produced.

Per scored arm the tracker predicts on the factual endpoint frame (the
training input convention), thresholded at the pinned validation
operating point 0.5. Metrics: hit rate (the true component's argmax cell
falls inside the ≥0.5 mask), pooled per-cell ROC AUC, per-arm IoU
distribution, and the probability at the true component's argmax cell
(detector-free ground truth localizes the vacated∪occupied component,
not the single player cell, so this upper-bounds the true-player-cell
probability). The in-distribution reference is 400 arms sampled
(seed 18) from the tracker's own hash-stable run-held-out validation
split (modulus 5; run `legacy-segment:cycle-000015`) of the
`lolo1-medium` label corpus.

## Ground-truth volume and censoring

Ground truth is dense, not sparse: 366–1,112 causal roots per run,
zero ambiguous endpoints or controls, and only 16 censored arms in
total (all `absent_control`, pooled across runs). Empty-mask labeled
arms (blocked / no-effect actions, excluded as in training): 732 of
3,652 labeled arms in v322, 7,355 of 11,120 in v323/v324, 1,762 of
8,806 in v325/v326. Scored arms: 2,920 / 3,765 / 3,765 / 7,044 / 7,044
(pooled 24,538; 13,729 distinct — v324 and v326 are certified
deterministic reruns of v323 and v325 and reproduce their metrics
bit-for-bit, so the pooled sample covers three distinct room states).

Ground-truth validity cross-check: on every one of the 24,538 scored
arms, the player cell recorded by the assisted detector lies inside the
detector-free counterfactual component (`assisted_cell_in_true_cells_rate`
1.0 on all five runs). The two conventions agree about where the player
is; only the tracker disagrees.

## Numbers

| corpus | arms | hit rate | cell AUC | IoU mean (max) | P(true argmax cell) mean | mask cells |
|---|---|---|---|---|---|---|
| v322 arm-a-pushed | 2,920 | 0.000 | 0.440 | 0.000 (0.000) | 0.041 | 27.7 |
| v323 arm-b-prepush | 3,765 | 0.001 | 0.446 | 0.000 (0.036) | 0.101 | 28.5 |
| v324 (certified rerun of v323) | 3,765 | 0.001 | 0.446 | 0.000 (0.036) | 0.101 | 28.5 |
| v325 object-removed | 7,044 | 0.833 | 0.865 | 0.038 (0.121) | 0.707 | 28.3 |
| v326 (certified rerun of v325) | 7,044 | 0.833 | 0.865 | 0.038 (0.121) | 0.707 | 28.3 |
| Room 3 pooled | 24,538 | 0.478 | 0.679 | 0.022 (0.121) | 0.442 | 28.3 |
| held-in (lolo1-medium val) | 400 | 1.000 | 0.9997 | 0.775 (1.0) | 0.9995 | 2.8 |

Gap (held-in minus Room 3 pooled): hit rate **−0.52**, cell AUC
**−0.32**, IoU mean **−0.75**, true-cell probability **−0.56**. The
held-in harness reproduces the training-time validation AUC (0.9997)
and IoU regime (0.78 vs the gated 0.75), so the gap is measured on a
calibrated instrument.

Secondary assisted reference (evaluation-only): mean probability at the
assisted-recorded player cell is 0.039 (v322), 0.095 (v323/v324), 0.690
(v325/v326) — matching the detector-free component numbers, as expected
given the 100% containment above.

## Honest interpretation

- **The failure is state-dependent, not uniformly total.** With the
  pushable object present (v322–v324) localization fails completely:
  hit rate ≈ 0 and cell AUC 0.44–0.45 — *below* 0.5, i.e. the tracker
  systematically ranks the true player region below average background.
  With the object removed (v325/v326) it partially transfers: the ≥0.5
  mask touches the true component on 83% of arms and AUC is 0.865.
- **The tracker over-fires by an order of magnitude out of
  distribution.** ~28 predicted mask cells per frame versus 2.8
  held-in (true component ≈ 1.6–2.2 cells); mean background probability
  0.127 versus 0.004. Even where it "hits", IoU never reaches 0.13, and
  the global argmax cell is in the true component on exactly 0 of
  24,538 arms. The hit-rate numbers are therefore the *charitable*
  reading; the mask is a broad blob, not a localization.
- This confirms and sharpens §4.31: offline training-distribution
  metrics said nothing about Room 3, and the earlier sweep's IoU ≈ 0
  telemetry is now quantified against proper counterfactual ground
  truth instead of the assisted mask.

Caveats:

- Certified reruns double-count v323/v325 arm populations in the pooled
  row; per-run rows and the three distinct states carry the evidence.
- Ground truth is the controllable *component* (vacated∪occupied cells,
  the training-label convention with its one-displacement-step blur),
  so "true player cell" probabilities are upper bounds.
- All training arms have duration 16; Room 3 arms are durations 8 and
  16. Duration 8 is a mild extrapolation, but the object-present runs
  fail at every duration (v322 hit rate is exactly 0), so duration
  alone cannot explain the gap.
- Room 3 pairs come from planner-selected states, not a uniform state
  distribution; metrics describe the states the agent actually probes.

## Retraining target (recommendation)

The gap is corpus coverage, not label quality — the same labeling rule
produces clean, dense ground truth on Room 3 telemetry. To close it
(§4.31 item b), the strict-track counterfactual collection should add:

1. **Room 3-palette coverage, object present and absent.** The
  object-present states are the catastrophic case; collection must
  include player-adjacent-to-object configurations (the
  `controlled-relative` distance-1 contexts), not just open-floor
  frames.
2. **Hard negatives for movable-object tiles.** The below-chance AUC
  with the object present indicates probability mass parked on
  non-player appearance; corroborated residual cells from multi-action
  roots in these states supply exactly those negatives.
3. **Duration diversity** (8 alongside 16), matching the probe search's
  actual edge durations.
4. Scale comparable to the existing corpus per new room state
  (thousands of roots are already available per probe run at zero new
  emulator cost if strict-track probe collection replays this
  extraction path natively).

Any promotion claim afterwards still requires the redesigned
mask-sensitive substitution gate (§4.31 item c); this evaluation is
report-only and grants nothing.
