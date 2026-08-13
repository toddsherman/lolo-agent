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

- the default minimum is 16 consecutive committed post-milestone decisions
  without verified reachable progress, configured by
  `--human-prior-goal-exhaustion-minimum-steps`; a new graph state, player
  position, or world context resets this clock;
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
transition hint was recorded. Rollback now requires that transition metadata;
older checkpoints that cannot distinguish an unknown target from a known empty
heart set remain usable after observed life loss but cannot trigger bounded
exhaustion. Old unqualified goal-exhaustion values are ignored during episodic
seeding.

For compatible legacy checkpoints, the loader can now recover the missing
known target flag from immutable ancestral telemetry. Recovery requires an
exact match on the checkpoint's source frame, behavioral source, action,
duration, and source heart set; it then copies the pixel-detected target heart
set from that committed transition. The Room 3 chain resolves to the original
self-discovered decision 1 transition, from `[(128,64),(144,192)]` to
`[(144,192)]`, and records `legacy_decision_telemetry` provenance. No visual
object identity or game rule is introduced.

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

A second eight-decision continuation verified that the exploration counter
survived resume, then expanded the same post-heart frontier leftward through
`(112,32)`, `(96,32)`, `(80,32)`, and `(64,32)`. This exposed a subtler issue:
the first correction still counted total elapsed decisions, so it reached 16
and restored the pre-heart checkpoint despite those new states. That replay is
retained as negative evidence in
`entity-v10-room3-soft-exhaustion-continue-from-d8-d8`. The progress-reset and
known-transition requirements above are the resulting correction; progress
can no longer be accumulated as exhaustion evidence.

The matched native validation
`entity-v10-room3-progress-reset-from-d8-d2` resumed the same decision-8 state
with the final semantics. It imported the counter at 8 and no unqualified
hazard values. The first exact search deferred at step 9, then reaching the new
`(112,32)` graph state emitted a progress reset from 9 to 0. The next decision
incremented only to 1 and reaching `(96,32)` reset it again. Across two
decisions it verified 869 exact option branches, recorded two deferrals and
two progress resets, and produced zero exhaustion rollbacks, hazard samples,
life losses, entity-hazard detections, or fail-opens. The frozen neural audit
passed. This is the intended behavior on real emulator state, not only a unit
test.

## Next experiment

Continue from decision 2 of the matched progress-reset validation toward the
remaining heart or a downward route. If 16 consecutive decisions pass without
a new reachable state, rollback may record a soft collection-order hint, but
it still cannot label the collection action hazardous. Any future hard veto
must be supported by actual causal hazard provenance.
