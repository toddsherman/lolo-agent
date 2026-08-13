# Anonymous entity behavior learning

## Purpose

The agent can now accumulate reusable evidence that recurring visual entities
behave similarly without being told that an entity is a skull, enemy, block,
heart, or any other game object. Internally, a recurring pooled RGB appearance
receives an anonymous integer such as `type 17`.

Each type stores empirical distributions conditioned on:

- the NES hardware action and duration;
- passive versus action-controlled observation;
- a translation-invariant pixel relation to the action-correlated controllable
  patch, with an anonymous scene signature as a localization fallback;
- the observed position-relative pixel outcome;
- a pixel-derived life-loss observation.

Terminal behavior has two separate posteriors. The empirical posterior retains
all pixel-observed correlations for audit and representation learning. The
causal hazard posterior counts only locally attributed intervention/control
rows, preventing a whole-screen reset from granting policy authority to every
rare patch on screen.

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
named. Coarsely binned distance, row/column/diagonal alignment, and relative
direction can support different predictions while sharing the same appearance
type. These relations transfer under translation instead of memorizing an
absolute room location.

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
  --anonymous-entity-behavior-mode learn \
  --anonymous-entity-passive-horizons 16,32,64,224 \
  --anonymous-entity-causal-horizons 16,32,64,224
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
  --anonymous-entity-behavior-mode frozen \
  --anonymous-entity-shadow-horizons 16,32,64,224
```

Frozen runs record a before/after digest audit. Predicting an unfamiliar
appearance does not create a new type. Exact replay of the same save-state
evidence is deduplicated and cannot inflate confidence.

The optional passive horizons restore the decision-root save state separately
for each duration and advance only `NOOP`. They do not change action selection
or the live trajectory. This makes delayed transformations and terminal visual
changes observable without forcing a short action interval to stand in for an
entity's full dynamics.

Causal horizons are more expensive and stricter. For each verified
non-neutral controller endpoint they construct an equal-duration `NOOP`
endpoint from the same root, then wait identically from both endpoints. A rare
patch becomes eligible only when:

1. its pre-wait appearance matches between intervention and control;
2. its relation to the detected controllable patch differs;
3. its local position-relative pixel outcome differs; and
4. neither branch is terminal at that localization horizon.

If a later horizon loses a life in only one branch, terminal credit is limited
to candidates localized earlier in that same matched contrast. A terminal
contrast with no prior local differential produces no entity behavior sample.

After frozen shadow evaluation passes its promotion gate, enable the optional
commit filter explicitly:

```bash
lolo-neural-run \
  ... \
  --anonymous-entity-behavior-mode frozen \
  --anonymous-entity-shadow-horizons 16,32,64,224 \
  --anonymous-entity-shadow-hazard-threshold 0.9 \
  --anonymous-entity-hazard-veto
```

The veto trusts only context-matched causal hazard rules with sufficient
support. It does not add a score bonus or penalty. If every verified endpoint
is hazardous it fails open, emits `anonymous_entity_hazard_veto_evaluated`, and
keeps the original alternatives.

## Current research boundary

The sidecar remains observational by default and always has additive selection
weight zero. The optional provenance-qualified hazard veto can now remove a
verified endpoint after passing the native shadow gate documented in
`anonymous-entity-policy-gate-2026-08-13.md`. It cannot rank safe alternatives,
and unsupported or context-fallback predictions cannot veto.

The current action-controlled collector uses the assisted track's pixel-derived
controlled-sprite locator to define an interaction ray. Object types and their
outcomes remain unlabeled, but this locator is not yet part of the strict
rule-free architecture. Passive rare-patch tracking does not need the locator
except to mask the controlled sprite. A learned action-correlated tracker must
replace that assisted component before the final strict evaluation.

A life-signature change during an ordinary passive horizon is still logged for
every rare candidate tracked across that interval, but those rows are marked
`evidence_eligible=false` and cannot update the checkpoint. Only the causal
collector can accept terminal entity evidence. It uses the assisted track's
pixel-derived player and life detectors, so cross-room validation,
appearance-agnostic and context-agnostic baselines, and a learned
controllable-entity tracker are still required before hazard evidence can
qualify for the final strict evaluation.

## Native causal-attribution milestone

The causal collector was trained from two independently recorded Room 2
episodes, checked on one development-validation episode, and then evaluated
frozen on a previously unused historical episode. All began before a
controller move changed the relation to anonymous type 7
(`cce8d09a9ec5ef55`). Each run tested six non-neutral actions at four matched
wait horizons, for 24 causal contrasts. Only `RIGHT` produced the relevant
chain.

| Stage | Training episode 1 | Training episode 2 | Frozen validation |
| --- | --- | --- | --- |
| Source | v19 decision 65 | v15 decision 58 | v11 decision 40 |
| First localized cell | `(5,2)` at 32 frames | `(5,2)` at 32 frames | `(5,2)` at 32 frames |
| Anonymous identity | type 7 | type 7 | type 7 |
| Activated relation | distance 1, same column | same | same |
| 32/64-frame outcome | transformed, safe | transformed, safe | all four predictions matched |
| 224-frame intervention | life loss | life loss | predicted hazard `1.0`, life loss |
| 224-frame neutral control | safe | safe | predicted hazard `0.0`, safe |

After the two training episodes, the activated type-7 rules contain two exact
samples at each of 32, 64, and 224 frames. In both frozen validation runs, v11
and the earlier v16 development fold, all six causal intervention/control
outcome predictions matched. Each also matched 33 of 34 known predictions and
all 33 known hazard classifications across every anonymous patch. The behavior
checkpoint had 600 observations and 236 rules; its digest remained
`3b227c5543b8b6cd32c966c55d3ae283e131e15187d12b6f715b4e97fb696977`
before and after frozen evaluation. The base neural checkpoint audit also
passed, and selection weight remained zero.

The paired native negative is equally important. From a later aligned state,
all one-action endpoints retained the same detected controllable position and
both intervention and neutral branches lost a life. The collector recorded 24
contrasts but zero localized candidates and zero causal attributions.

Reproducible artifacts:

```text
experiments/lolo1-entity-v8/anonymous-behavior.json
experiments/lolo1-entity-v8/evaluations/entity-v8-room2-safe-causal-learn-v19-d65
experiments/lolo1-entity-v8/evaluations/entity-v8-room2-causal-learn-v15-d58
experiments/lolo1-entity-v8/evaluations/entity-v8-room2-causal-validation-frozen-v11-d40
experiments/lolo1-entity-v8/evaluations/entity-v8-room2-causal-heldout-frozen-v16-d34
experiments/lolo1-entity-v8/evaluations/entity-v8-room2-aligned-causal-frozen-v19-d79
```

This is episode/state transfer within Room 2, not a preregistered split or
withheld-room generalization. The v11 fold was not used to design or train the
behavior representation, but it comes from the historical development corpus.
The result establishes a causal local credit mechanism and a reusable
anonymous rule; the next gate is recurrence in other rooms and prospective
folds.

An exact pixel-only screen of the available Room 3 source found 25 rare
appearance fingerprints. Nineteen matched existing checkpoint types, but none
matched type 7. That state can measure generic stationary appearance reuse,
not transfer of this learned dynamic rule, so it was not presented as the
cross-room gate.

## Native relational-dynamics milestone

Room 2 development branches produced an anonymous appearance with fingerprint
`cce8d09a9ec5ef55`, assigned checkpoint type 7. The system was not supplied a
sprite name, object class, activation rule, or death rule. It learned two
supported relational contexts and was then run frozen from separately captured
Room 2 states.

| Frozen target | Relation at type 7 | 16 frames | 32 frames | 64 frames | 224 frames |
| --- | --- | --- | --- | --- | --- |
| Safe heldout | distance 2, diagonal | stationary, safe | stationary, safe | stationary, safe | stationary, safe |
| Hazard heldout | distance 2, same column | stationary, safe | transformed, safe | transformed, safe | terminal life loss |

For type 7, all eight duration-conditioned predictions matched the native
outcomes. At 224 frames the safe context predicted hazard probability `0.0`
and observed no life loss; the aligned context predicted `1.0` and observed a
life loss. Each contextual rule had two independent training observations.
Both heldout runs rejected all evidence updates and passed the behavior-model
before/after digest audit. Across all anonymous patches, the safe run matched
88 of 112 known outcome predictions and the hazard run matched 108 of 112.
Hazard classification across every tracked patch was 94/112 in the safe run
and 112/112 in the hazardous run. The safe-run false positives are direct
evidence of the global-credit limitation described above; the type-7
contextual result is promising, but the aggregate model is not ready to steer
the policy.

An earlier frozen development run had predicted a hazard from an unconditional
rule where waiting was safe. That failure exposed the context alias and led to
the relational representation plus separately supported safe and hazardous
examples. It remains in the audit trail rather than being counted as a final
heldout result.

Reproducible artifacts:

```text
experiments/lolo1-entity-v7/anonymous-behavior.json
experiments/lolo1-entity-v7/evaluations/entity-v7-room2-safe-heldout-frozen-v19-d65
experiments/lolo1-entity-v7/evaluations/entity-v7-room2-aligned-hazard-heldout-frozen-v19-d79
experiments/lolo1-entity-v7/evaluations/entity-v7-room2-hazard-heldout-frozen-v16-d34
```

This is a same-room, separately captured state/episode milestone, not yet a
withheld-room or sequel generalization result. Its value is that a recurring
anonymous visual type now retains reusable, context-dependent delayed behavior
instead of being rediscovered as unrelated pixels.

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
`anonymous_entity_passive_horizon_verified` records each additional neutral
duration branch from the root save state.
`anonymous_entity_causal_horizon_verified` and
`anonymous_entity_causal_contrast_completed` record matched intervention and
neutral waits plus localization and hazard-attribution counts.
Learning runs end with `anonymous_entity_behavior_checkpoint_updated`; frozen
runs end with `anonymous_entity_behavior_parameter_audit`.
