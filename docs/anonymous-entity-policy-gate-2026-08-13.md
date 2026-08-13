# Anonymous entity policy gate — 2026-08-13

## Outcome

The anonymous behavior sidecar now has an optional, conservative endpoint
hazard veto. It is disabled by default. When enabled, it can filter a verified
controller endpoint only when all of the following are true:

1. the endpoint contains a known anonymous appearance;
2. its translation-invariant relation to the pixel-detected controllable patch
   matches a learned context;
3. the exact future wait horizon has at least the configured minimum number of
   locally attributed causal hazard samples; and
4. the causal hazard posterior meets the configured threshold.

Ordinary terminal correlation is still retained as scientific telemetry, but
it has no veto authority. If every verified endpoint meets the veto, the gate
fails open and records that fact rather than leaving the controller without an
action.

## Why provenance was required

The first observational native replay detected the true delayed hazard, but it
also marked safe `DOWN` as a simulated veto. The false positive came from an
unrelated rare appearance whose earlier passive observation happened to share
a whole-screen terminal reset. The old empirical posterior scored 23/24
matched causal outcomes: one true positive, 22 true negatives, one false
positive, and no false negatives.

Schema v5 therefore stores a second hazard posterior containing only localized
intervention/control evidence. A migration backfilled this provenance from the
two immutable Room 2 causal-learning runs. It added no appearance type, rule,
outcome, or ordinary observation:

| Field | Value |
| --- | ---: |
| Anonymous types | 22 |
| Rules | 236 |
| Ordinary observations | 600 |
| Causal hazard observations backfilled | 12 |
| Causally hazardous rows | 2 |
| Causally safe rows | 10 |
| v9 parameter digest | `cdfb36a6c862c959b0b889ad2fca6549935c6997c68b0e2c1038d7002a9016d5` |
| v9 file digest | `055b5eff042ee82d6d302a260c5834c3295cfdb171d2b634131660d1fa91a100` |

`lolo-entity-causal-backfill` accepts only evidence IDs already present in the
input checkpoint. This prevents an evaluator from manufacturing new behavior
samples during provenance migration.

## Frozen native results

All runs used threshold `0.9`, minimum support `2`, horizons
`16,32,64,224`, and a frozen v9 checkpoint.

| Evaluation | Matched causal outcomes with supported predictions | TP | TN | FP | FN | Simulated vetoes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Room 2 validation, v11 decision 40 | 9 | 1 | 8 | 0 | 0 | 1 |
| Room 2 held-out historical state, v16 decision 34 | 9 | 1 | 8 | 0 | 0 | 1 |
| Room 3 screen, v10 p8 decision 1 | 0 | 0 | 0 | 0 | 0 | 0 |

In both Room 2 states, only `RIGHT` was marked. The implicated anonymous
appearance was type 7 (`cce8d09a9ec5ef55`) at coarse cell `(5,2)`, one cell
above the controlled patch. Its 224-frame causal hazard probability was `1.0`
from two samples. The emulator branch lost a life; the duration-matched neutral
control did not. The other eight supported action/horizon outcomes were safe
and predicted safe.

Room 3 had 24 causal contrasts, no life-loss contrast, no newly localized
candidate at that state, and no provenance-qualified prediction. The legacy
correlation-only posterior reached `1.0` for several appearances there; the new
gate correctly gave all of them zero authority.

## Policy-authority check

From the Room 2 v11 source, the observational planner committed `RIGHT`. With
the veto enabled, the gate filtered exactly that endpoint, retained six
alternatives, and the unchanged downstream planner committed `DOWN`:

| Field | Value |
| --- | --- |
| Hazards detected / filtered | `1 / 1` |
| Alternatives remaining | `6` |
| Fail-open | `false` |
| Filtered causal probability | `1.0` |
| Filtered support | `2` |
| Committed action | `DOWN`, 16 frames |
| Committed predicted hazard | `0.0` |

The base neural checkpoint and anonymous behavior checkpoint both passed their
before/after frozen-parameter audits.

## Reproducible artifacts

```text
experiments/lolo1-entity-v9/anonymous-behavior.json
experiments/lolo1-entity-v9/evaluations/entity-v9-room2-causal-provenance-shadow-frozen-v11-d40
experiments/lolo1-entity-v9/evaluations/entity-v9-room2-causal-provenance-shadow-heldout-v16-d34
experiments/lolo1-entity-v9/evaluations/entity-v9-room3-causal-provenance-shadow-frozen-v10-p8-d1
experiments/lolo1-entity-v9/evaluations/entity-v9-room2-hazard-veto-frozen-v11-d40
```

Each run includes `entity_behavior_shadow.csv` for patch/horizon predictions
and `entity_behavior_shadow_branches.csv` for endpoint verdicts. The full
intervention/control result remains in `events.jsonl`; confusion counts and
veto counts are in `summary.json`.

## Scope

This gate demonstrates same-mechanic transfer between separately captured Room
2 states and a no-veto screen in Room 3. It is not yet positive cross-room
hazard transfer, a preregistered held-out-room result, or evidence that the
current system can beat the whole game. The controllable-patch and life-loss
detectors are still from the assisted pixel track. Final strict evaluation
requires learned replacements and must keep persistent parameters frozen.
