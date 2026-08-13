# Room 3 milestone-credit correction (2026-08-13)

## Diagnosis

The first strong Room 3 search exposed a reward-credit error, not evidence
that the room was unsolvable. The agent collected one of the two
pixel-detected hearts, then a depth-4 exact option search found no globally
novel endpoint. Goal-exhaustion rollback treated that bounded search result as
if the collection action were an unrecoverable temporal hazard. It assigned
the exact `DOWN 16` choice a value of `-2`, restored the pre-heart state, and
subsequent policy filtering rejected the same legitimate collection. The next
run oscillated between two player positions.

That inference was invalid for two independent reasons:

1. the Room 2 safeguard had originally fired after 146 committed
   post-milestone decisions, whereas Room 3 fired after one;
2. failure to find a new endpoint within a finite search budget is not causal
   evidence of life loss or loss of control.

The checkpoint also represented both "target heart set is unknown" and the
valid "no hearts remain" target as an empty tuple. That prevented reliable
transition-scoped credit for final-heart milestones.

## Correction

Goal-exhaustion rollback now has three separate semantics:

- the default minimum is 16 committed post-milestone decisions, configured by
  `--human-prior-goal-exhaustion-minimum-steps`;
- an earlier exact-search failure emits
  `goal_milestone_exhaustion_deferred` and leaves the checkpoint and policy
  intact;
- a later bounded exhaustion records only a soft heart-set ordering hint. It
  does not write a negative temporal-option value and therefore cannot enter
  the learned-hazard hard filter.

Goal checkpoint metadata now carries
`goal_target_heart_slots_known` separately from the target tuple. A known empty
tuple is therefore a valid final-heart transition. The compatibility event
`goal_milestone_exhaustion_learned` explicitly reports
`hazard_evidence=false`, `policy_effect=milestone_priority_only`, and whether a
transition hint was recorded.

## Native matched replay

The corrected agent resumed from decision 1 of
`entity-v10-room3-causal-learning-hazard-veto-v10-p8-d1-d4`, the identical
self-discovered save state immediately after the first Room 3 heart. The run
is:

`entity-v10-room3-soft-exhaustion-min16-from-heart-d1-d8`

Results:

- 8 committed decisions and 414 unique frames;
- 3 exact option searches and 1,221 verified option branches;
- 2 correctly deferred exhaustion results at evidence steps 1 and 3;
- 0 goal-exhaustion hazard samples and 0 rollback events;
- 0 life losses;
- 4 anonymous-entity hazard-veto evaluations, with 0 detections and 0
  fail-opens;
- 128 causal anonymous-entity contrasts and 36 passing frozen shadow audits;
- the persistent anonymous behavior memory grew from 35 to 36 appearance
  types, 464 to 525 conditional rules, and 1,054 to 1,477 unique observations;
  causal hazard provenance remained unchanged at 12 observations.

Most importantly, the agent preserved the collected heart and expanded the
post-heart reachable frontier from `(128,48)` to `(128,32)`, `(144,32)`,
`(160,32)`, and `(176,32)`. At decision 5, an exact search added a genuinely
new endpoint. This directly falsifies the earlier conclusion that the
post-heart state was exhausted.

## Next experiment

Continue from decision 8 with the corrected credit semantics and the updated
anonymous behavior checkpoint. Success is either collection of the remaining
heart, a persistent new object/world state, or expansion toward the lower
route. If 16 committed post-heart decisions pass without progress, rollback
may record a soft collection-order hint, but it still cannot label the
collection action hazardous. Any future hard veto must be supported by actual
causal hazard provenance.
