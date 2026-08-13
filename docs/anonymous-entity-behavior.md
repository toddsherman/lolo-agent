# Anonymous entity behavior learning

## Purpose

The agent can now accumulate reusable evidence that recurring visual entities
behave similarly without being told that an entity is a skull, enemy, block,
heart, or any other game object. Internally, a recurring pooled RGB appearance
receives an anonymous integer such as `type 17`.

Each type stores empirical distributions conditioned on:

- the NES hardware action and duration;
- passive versus action-controlled observation;
- an anonymous, order-independent visual-context signature;
- the observed position-relative pixel outcome;
- a pixel-derived life-loss observation when it is causally isolated.

There are no hand-authored mechanics in the checkpoint. A passive outcome can,
for example, encode that the same appearance recurred one grid cell to the
left. The model is not told that this means an enemy moved. An action-controlled
outcome can encode that replacing one intervention with an equal-duration NOOP
changed a local appearance and the controlled sprite's relative endpoint.

## Why the rule is a distribution

An appearance is not assigned one unconditional behavior. The model retains
all observed outcomes and reports probability, entropy, evidence count, and
hazard probability. If the same appearance is stationary in one visual context
and mobile in another, sufficiently supported context-specific distributions
override the cross-context fallback. Conflicting evidence lowers confidence
instead of being silently overwritten.

This is the mechanism needed for activation rules: the trigger need not be
named. The room's anonymous visual context before and after the trigger is
enough to support different predictions while sharing the same appearance
type.

## Persistence and evaluation

Learning mode creates or updates a content-digested JSON checkpoint only after
a run completes cleanly:

```bash
lolo-neural-run \
  ... \
  --human-prior-hearts \
  --human-prior-option-effect-stability-steps 3 \
  --human-prior-option-effect-phase-offsets 3 \
  --human-prior-option-effect-local-controls \
  --human-prior-option-entity-frontier \
  --anonymous-entity-behavior-checkpoint \
    experiments/lolo1-entity/anonymous-behavior.json \
  --anonymous-entity-behavior-mode learn
```

For withheld rooms or *Lolo 2*, load the same checkpoint frozen:

```bash
lolo-neural-run \
  ... \
  --human-prior-hearts \
  --human-prior-option-effect-stability-steps 3 \
  --human-prior-option-effect-phase-offsets 3 \
  --human-prior-option-effect-local-controls \
  --human-prior-option-entity-frontier \
  --anonymous-entity-behavior-checkpoint \
    experiments/lolo1-entity/anonymous-behavior.json \
  --anonymous-entity-behavior-mode frozen
```

Frozen runs record a before/after digest audit. Predicting an unfamiliar
appearance does not create a new type. Exact replay of the same save-state
evidence is deduplicated and cannot inflate confidence.

## Current research boundary

The sidecar is observational and has selection weight zero. It logs whether a
frozen prediction matched a later emulator observation, but it cannot yet
change the selected controller action. Promotion requires held-out native
evidence that type-conditioned predictions beat appearance-agnostic baselines
and that confidence is calibrated.

The current action-controlled collector uses the assisted track's pixel-derived
controlled-sprite locator to define an interaction ray. Object types and their
outcomes remain unlabeled, but this locator is not yet part of the strict
rule-free architecture. Passive rare-patch tracking does not need the locator
except to mask the controlled sprite. A learned action-correlated tracker must
replace that assisted component before the final strict evaluation.

## Initial native sanity result

The schema-v3 passive representation was learned for three decisions from a
self-discovered Room 3 state and evaluated frozen for three decisions from a
different Room 2 run. The sidecar had selection weight zero in both cases.

| Measurement | Room 3 learn | Room 2 frozen |
| --- | ---: | ---: |
| Passive observations | 75 | 96 |
| Anonymous types in checkpoint / recognized | 21 | 17 |
| Known frozen predictions | — | 81 |
| Exact known-prediction matches | — | 81/81 |
| Base-model digest audit | Pass | Pass |
| Behavior-sidecar digest audit | Updated once | Pass, unchanged |

The result validates cross-room appearance reuse and removes an earlier
room-layout leak: only the nearest recurrence of an appearance now represents
local persistence or motion. It is not yet an enemy-behavior result. These
particular four-frame passive intervals were dominated by stationary outcomes,
so native movement, activation, and hazard transfer still need separate
held-out evidence before planning promotion.

Reproducible artifacts:

```text
experiments/lolo1-entity-v3/anonymous-behavior.json
experiments/lolo1-entity-v3/evaluations/entity-v3-room3-passive-learn-d3
experiments/lolo1-entity-v3/evaluations/entity-v3-room2-heldout-frozen-d3
```

## Telemetry

Every prediction and observation is written as
`anonymous_entity_behavior_observed`. It includes the anonymous type, appearance
fingerprint, action, duration, context, prediction probability, observed
outcome, surprise, confidence, entropy, hazard probability, evidence ID, and
whether learning accepted the evidence. `entity_behaviors.csv` provides the
same records as a flat visualization-ready artifact.

`anonymous_entity_passive_scan_completed` describes each passive scan.
Learning runs end with `anonymous_entity_behavior_checkpoint_updated`; frozen
runs end with `anonymous_entity_behavior_parameter_audit`.
