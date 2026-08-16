# Object-state and delayed-credit gate — 2026-08-16

## Research question

Can the assisted pixel planner carry a rule-free preparation hypothesis across
a delayed milestone without confusing sprite animation, player motion, and a
persistent object displacement?

## Findings

- `entity-v311-room3-previsited-future-goal-d36x18` falsified pose-only
  preparation. The agent learned the post-heart chest location, visited that
  tile before collecting the final heart, then still stalled four tiles from
  the chest after 12,063 verified branches.
- `entity-v312-room3-layout-aware-preparation-d24x18` showed that cumulative
  pixel-change cells were not sufficient object state. Its apparent layout
  variants all mapped to the animated blue enemy at coarse cell `(7, 6)`.
- `entity-v313-room3-pristine-reliable-preparation-d18` rejected 1,756 raw
  changed-layout branches and accepted zero as reliable manipulation evidence
  through 3,202 verified branches. This established a detector bottleneck,
  not a reward or search-depth bottleneck.
- `entity-v314-room3-pristine-directional-persistence-d8` completed 3,333
  exact branches and 139 causal probes. It confirmed that broad pristine-room
  search did not contact a movable object within the bounded horizon.
- `entity-v316` and `entity-v317` used a historical pre-push Room 3 state.
  Independent save-state probing verified a one-cell anonymous displacement,
  but exact-search tracking initially lost it because repeated appearances
  were excluded and a nearby white object was included in Lolo's mask.
- `entity-v318-room3-known-push-connected-mask-d2` passed the targeted gate.
  Six exact branches tracked the pushed object at destination cell `(8, 6)`,
  two matched `RIGHT -> NOOP` branches preserved that state for one persistence
  step, independent probing confirmed the displacement, and three replayable
  archive states were stored.
- `entity-v319-room3-pushed-object-continuation-d12` resumed the confirmed
  push but lost the object track at the new search root. It explored 869
  branches through depth nine without another displacement or a heart,
  isolating cross-cycle state continuity as the next representation gap.
- `entity-v320-room3-restored-object-track-d6` reconstructed the older
  archive's anonymous track from its verified pixel-effect signature. All
  2,497 exact-search descendants retained destination cell `(8, 6)`, and the
  planner archived a safe continuation five actions beyond the push. It did
  not collect a heart, so this is a representation result rather than a room
  solution.
- `entity-v321-room3-confirmed-identity-d2` verified corrected serialization:
  all 132 descendants retained the track, and its archive kept the confirmed
  `RIGHT`, source `(7, 6)`, destination `(8, 6)`, and one-step persistence
  instead of pairing the push state with a later unrelated interaction.

## Implemented changes

- A future goal revealed after an exhausted milestone is stored in episodic
  telemetry and can provide temporary spatial credit after rollback.
- Delayed future-goal credit propagates to compatible earlier states whose
  remaining-goal sets contain the failed late-stage set.
- Raw appearance changes cannot reopen an exhausted milestone; preparation
  requires confirmed causal state or a learned future-goal hypothesis.
- Future-goal beam reserves couple player reachability to anonymous persistent
  world configurations.
- Directional displacement uses the exact player-masked counterfactual effect
  at the contacted object's destination instead of the nonlocal-effect mask.
- Repeated anonymous appearances are eligible for displacement correspondence;
  rarity is not treated as object identity.
- Source and destination object features are compared after independently
  masking the controlled sprite.
- The assisted player mask now retains only the connected blue/white component
  anchored on Lolo's blue pixels, preventing adjacent white objects from being
  erased as part of the player.
- A phase-stable one-cell displacement can bootstrap a new mechanic before an
  empirical behavior probability is already known, then gains stronger
  evidence through matched neutral persistence and causal probing.
- Promoted option archives now serialize tracked cells, anonymous appearance
  and context, interaction direction, effect distance, and persistence.
  Exact archive resumes seed these fields into the new search root; legacy
  archives conservatively reconstruct the track from their learned spatial
  effect signature and recover a reusable appearance fingerprint from the
  player-masked destination pixels.
- Confirmed manipulation identity is stored separately from later transient
  interactions, preventing navigation near another anonymous patch from
  relabeling the manipulation that produced the retained world state.

## Roadmap consequence

The immediate priority is no longer a larger beam or a different heart reward.
The validated primitive should be extended into persistent anonymous object
tracks and relational state:

1. derive explicit source-to-destination displacement vectors from the now
   persistent track and maintain multiple simultaneous anonymous tracks;
2. aggregate the same learned displacement across recurring appearances and
   rooms without assigning supplied object names;
3. represent transformations as appearance-state transitions on an existing
   track;
4. distinguish milestone phases through learned global transition evidence;
5. rank room-level preparation configurations by verified post-milestone
   reachability; and
6. use exact emulator branches as the acceptance test for every proposed
   object-level plan.

This gate validates one necessary object-state primitive. It does not yet show
that the agent can choose the correct preparation, solve Room 3, or generalize
to later rooms.
