# Relational manipulation milestone — 2026-08-13

## Outcome

The anonymous behavior sidecar now represents the three future interaction
families that motivated this milestone without receiving their game-specific
names:

- translation of a recurring anonymous appearance relative to an action,
  suitable for learning push-like effects;
- a causal appearance transition relative to an equal-time neutral control,
  suitable for learning transformations and their later behavior;
- a distributed stable-cell phase change, suitable for learning limited visual
  resources and post-transition behavior without treating one animated cell as
  a room-wide transition.

Predictions are conditioned on pixel-derived relation, local neighborhood, and
global phase. The phase signature uses only cells supported as stable by
repeated neutral observations, so ordinary animation is excluded while a
formerly stable visual variable can establish a new phase. Distinct animation
variants can share evidence only when their measured semantic profiles agree.

No supplied object name, level solution, or mechanic rule is stored in the
model. The existing assisted reward track still labels hearts, the chest, and
life loss; that is separate from these anonymous manipulation descriptors.

## Native matched-state validation

All native runs resumed decision 4 of
`entity-v84-room3-post-heart-frontier-chain-frozen-d4`, at the same Room 3
pixels, with depth-2 exact search and four bounded adjacent probes.

| Run | Mode | Result |
| --- | --- | --- |
| `entity-v88-room3-causal-gate-frozen-d2` | canonical frozen | Three clean inert probes remained eligible. One apparent transformation was rejected because its matched control did not confirm causality. Both frozen audits passed. |
| `entity-v89-room3-relational-candidate-learn-d2` | candidate learn | Six inert observations were accepted across adjacent and beam probes. Three apparent appearance transitions and one apparent global-phase change were withheld. Four distinct stable phase contexts were logged. |
| `entity-v90-room3-relational-candidate-frozen-reuse-d2` | candidate frozen | Reloaded 9,663 observations and 1,154 rules. The three adjacent inert outcomes returned posterior probability 1.0 with one stored sample each. No evidence was accepted and both frozen audits passed. |

Follow-up native audits `entity-v91` through `entity-v98` exposed and corrected
three exploration aliases: fallback evidence was closing unseen exact
contexts, action variants at one locus consumed diversity slots, and phase
context was unresolved while probes were ranked. The revised policy calibrates
phase with neutral branches, ranks exact-context novelty, covers appearances
before action/locus variants, and then applies the causal gate. In the
eight-decision Room 3 continuation (`entity-v98`), it ran 48 curiosity probes
across 16 cells and withheld 11 unconfirmed effects. The run also revealed
that single stable-cell differences were being called global phase changes;
the label now requires at least three cells in two disconnected regions.
The native audit still contains no confirmed push or transformation frontier,
so it is evidence about exploration and false-positive control rather than a
claim that manipulation transfer is solved.

The canonical schema-6 checkpoint remained byte-identical throughout:
`ec645d68e9f708e6c01df479adb03795bec1f9872833ee89ca8e23d3d9c1b09d`.
The separate candidate (schema 7 at the original milestone, schema 8 after the
distributed-phase audit) is
`experiments/lolo1-entity-v10/anonymous-behavior-relational-v1.json` and is an
ignored experiment artifact, not a replacement for the canonical checkpoint.

## Verification

- The complete unit/integration suite is run after each policy refinement;
  platform-specific skips remain documented in the test output.
- Synthetic controls prove that an action-caused appearance transformation is
  learned and promoted before beam search.
- A matched synthetic false-positive proves that an unconfirmed appearance
  transformation cannot update the model.
- Phase tests prove that an animating cell is excluded while a persistent
  formerly stable visual change produces a new phase.
- Distributed-phase tests reject both one-cell residue and one connected local
  change, while accepting a three-cell, two-region transition.
- Schema 3 through 7 checkpoints load through the schema-8 migration.
- `entity_behaviors.csv` and `summary.json` expose relation, neighborhood,
  phase, displacement, transformation, global-phase, manipulation, and
  predictive-family telemetry.

## Research boundary and next gate

This validates representation, causal gating, persistence, and frozen reuse;
it does not yet demonstrate a successful native push or transformation. The
next evidence gate should resume or reach a state containing a genuinely
available manipulation, let the candidate discover it through the bounded
save-state probes, and then verify frozen transfer when the same anonymous
appearance occurs in a different layout or phase.

Before final rule-free evaluation, the assisted pixel-derived controlled-sprite
locator must also be replaced with a learned action-correlated controllability
tracker. Until then, this is a strong development architecture for the planned
mechanics, not yet proof that the full game will be beaten.
