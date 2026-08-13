# Anonymous entity outcome-semantics gate — 2026-08-13

## Question

Can persistent anonymous appearance rules improve exact planning after the
agent learns, from factual and equal-duration control images, that a recurring
appearance/action outcome had no measured effect?

## Implementation under test

- schema-6 behavior checkpoints preserve a structured pixel-derived descriptor
  beside each opaque outcome hash;
- a prediction reports semantic coverage plus posterior probabilities for no
  measured effect, controlled-patch displacement, local visual change, and
  terminal visual change;
- an opt-in planner weight subtracts the confidence-weighted learned no-effect
  probability;
- current verified emulator evidence has precedence: displacement, a world
  effect, a milestone, a terminal transition, or missing localization suppresses
  the learned penalty;
- all semantics remain anonymous. No sprite name, object rule, room solution,
  or mechanic label is stored.

## Native evidence

### Learning

Run: `entity-v12-room3-semantic-learn-d1`

- resumed independently from decision 4 of the previously discovered central
  room-3 state;
- searched 1,012 exact branches to depth 5;
- ran four reserved matched controls with four correct interaction-cell
  attributions;
- accepted 12 unique behavior observations and persisted seven unique outcome
  descriptors;
- grew the cloned sidecar from 7,702 to 7,714 observations and from 1,029 to
  1,037 rules;
- kept the frozen neural parameter digest unchanged at
  `5fd31f73f6b0e6e652a36d34232fd1ccab91e281b00d57ade547a518de6abe28`.

Because the descriptor key is the existing deterministic outcome hash, one
newly measured descriptor also makes older occurrences of that exact outcome
interpretable. The recurring anonymous type 26, `LEFT` for 16 frames, now has:

- two supporting outcomes, both `5d87d6bdd885774ab247d985`;
- semantic coverage `2/2`;
- learned no-effect probability `1.0`;
- evidence confidence `0.6321205588`;
- measured-effect probability `0.0`.

At the tested weight of 3.0, that rule contributes a predicted subtraction of
`1.8963616765` when its current branch has no stronger observed effect.

### Frozen guarded planning

Run: `entity-v12-room3-semantic-frozen-guarded-d1`

- loaded the learned sidecar frozen and searched 308 native branches to depth
  3;
- applied a learned no-effect penalty to 53 branches, totaling
  `121.9339712503` score points;
- suppressed 156 predicted penalties, totaling the difference between
  `421.2011619532` predicted and `121.9339712503` applied points, because the
  current branch supplied stronger evidence;
- applied `2.9877396857` to a supported 4-frame `DOWN` branch that left the
  controlled patch at `(128, 128)` with no measured effect;
- suppressed a predicted `0.9999986290` penalty on 16-frame `RIGHT` because the
  current branch moved the controlled patch from `(128, 128)` to `(144, 128)`;
- committed `UP` for 16 frames to `(128, 112)`;
- passed both frozen audits: neural digest unchanged and behavior digest
  unchanged at
  `4b53b8bea7bf92884f85718dc87e48a2bae3c4aee2bd221dd2acef2b9a0052be`.

An earlier diagnostic frozen run, `entity-v12-room3-semantic-frozen-d1`, used
the learned penalty before the verified-effect precedence guard. It is retained
for audit but is not policy-qualified; its over-broad penalties directly led to
the guard and regression test.

## Promotion

The validated schema-6 sidecar was promoted to
`experiments/lolo1-entity-v10/anonymous-behavior.json` with file SHA-256
`911b287cbeb7e7551ce94a08de1955e11bddbbcb2fe61d511fa3832e76f3cd48`.
The previous canonical file remains recoverable as
`anonymous-behavior-pre-semantics-v12.json`, SHA-256
`4ac6c9b4623f4fc13abcd66639948b79cde05dd7b55202b6fedf713aebee360c`.

## Gate decision

Pass for opt-in continued room exploration. The mechanism now turns recurring
visual interaction evidence into a bounded planning advantage while failing
open on unknown legacy outcomes and yielding to current verified effects.

This is not evidence that room 3 or the game is solved. The sidecar still uses
the assisted track's pixel-derived controlled-patch locator, and the current
training lineage contains exploratory observations that must be rebuilt from a
clean provenance ledger before formal held-out evaluation.
