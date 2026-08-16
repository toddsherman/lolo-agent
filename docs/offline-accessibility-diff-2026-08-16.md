# Offline accessibility diff of the confirmed push — 2026-08-16

Method: preregistered, read-only, offline telemetry analysis (roadmap §13
offline-first rule); zero emulator cost. Direction-review Amendment A step 0
(`docs/direction-review-2026-08-16.md` §3.A).

## Research question

Did the v320/v321 pushed-object configuration's verified search descendants
reach (or lose) player cells or interaction frontiers beyond the pushed
object's footprint — source `(7,6)`/pixel `(112,96)`, destination
`(8,6)`/pixel `(128,96)` — relative to pre-push Room 3 exact search from
comparable states? Footprint cells excluded from all deltas.

## Findings

### Beyond-footprint deltas

Pooled comparison (pushed union v319+v320+v321, 3,498 verified branches, vs
pre-push 4-heart-era lineage union v313/v314/v316/v317/v318, 6,899 branches —
the only pre-push sets in the pushed state's heart era and detector
generation):

- Pushed-era union (n=8): `(96,96)`, `(96,112)`, `(96,128)`, `(96,144)`,
  `(96,160)`, `(112,96)`, `(112,160)`, `(128,160)`.
- Pre-push 4-heart union (n=8): **identical set, element for element.**
- Cells in pushed-era but never pre-push: none. The one candidate signal —
  v319's `(128,160)`/cell `(8,10)` — is present in pre-push v313/v314. Not a
  delta.
- Cells in pre-push but never pushed-era: none.
- Footprint check: the pushed destination `(128,96)`/`(8,6)` appears in
  NEITHER era's player coverage — no player ever stood on the destination
  tile in any of the 8 compared runs, and no player ever occupied any cell
  right of column x=128 above the bottom row. The room's right side is
  unexplored in both eras.

Interaction frontiers could not be diffed: pushed-era telemetry has
adjacent-entity probes (all unconfirmed, incl. `A`/`RIGHT`/`B` at `(8,6)`)
and curiosity probes, plus v319's unique A-press probe at `(8,11)` from
`(8,10)`; the pre-push runs contain no probe events at all, so probe-level
comparison is censored on the pre-push side. The `(8,11)` A-interaction is an
untested candidate, not a delta.

v311/v312 and the entity-v10-room3-* series are different heart eras /
detector generations and were excluded as lineage-mismatched comparators.

### Matched-lineage comparison (preferred)

Closest match: v316/v317/v318 (shared `33addc6c` pre-push checkpoint digest,
identical resume state) vs the pushed runs resuming from v318's
`state-00000117` one push later.

- Depth-matched (d2): pre-push n=4 cells vs pushed v321 n=3 — v321 missing
  `(96,128)` despite double beam. Not a configuration loss: v320 (d6) and
  v319 (d9) reach it from the same pushed state; v321's zeroed reserves make
  this a beam-composition artifact. Per the v56 lesson, pose-level
  distinctions were not counted anywhere.
- Depth-unmatched: the four extra pushed-era cells over v316–318 are all
  recovered pre-push by v313/v314 (same physical lineage, one step earlier,
  deeper budgets). Deeper search on either side converges on the same 8-tile
  envelope.

### Budget honesty

Severe asymmetry in both directions; no single pair is budget-clean. Any cell
absent from the shallow v316–318 runs is censored, and pushed-era "gains"
over them vanish against v313/v314. The strongest fact: v319 self-exhausted
at depth 9, beam 128 (zero novel endpoints at d10), so its 8-position set is
a genuine frontier bound for the pushed configuration — **the pushed
configuration's exhausted search found nothing outside the pre-push
envelope.** All three pushed runs share one root state (correlated, not
replicates) and committed exactly one decision each, so this is search-reach,
not trajectory-reach.

## Verdict

**No beyond-footprint delta in available telemetry** — the preregistered
null result. Scoped strictly: evidence that the single push is
accessibility-neutral at explored depths (≤9, beam ≤128, left-corridor /
bottom-row envelope), NOT proof of neutrality. Unresolvable offline:

1. whether the object at `(8,6)` blocks or enables rightward passage — the
   entire right side above the bottom row is unexplored in both eras;
2. whether the `(8,11)` A-interaction (probed only in v319, matched but
   unconfirmed) is state-dependent on the push;
3. all interaction-frontier comparisons (pre-push probe telemetry absent).

The native paired probe (Amendment A week 1) is the decisive instrument for
all three.

## Native probe design (informed by this diff)

- Arms: (A) pushed — resume `state-00000117` (v318 archive); (B) pre-push —
  resume the `33addc6c` goal-milestone rollback checkpoint stored by
  v316/v317/v318. Same physical lineage, same 4-heart era, one push apart:
  the clean pair this offline diff lacked.
- Identical budgets both arms: depth 12 (headroom over v319's exhaustion at
  9), beam 128, v320's reserve profile (position_reserve 16, curiosity
  32/8.0, goal_world_state_reserve 12, milestone_reserve 32,
  stationary_history 2). Do NOT reuse v321's zeroed reserves.
- requested_decisions ≥ 8 with the warmup-4 evaluator stop disabled/raised,
  so committed trajectories exist and coverage is walk-reach, not only
  search-reach.
- Directed targets: (1) attempt right traversal past the destination toward
  column ≥8, rows 5–7, both arms; (2) reach `(8,10)` and execute the
  A-interaction at `(8,11)`, both arms; (3) attempt to occupy `(8,6)` itself
  in arm B (empty pre-push) vs arm A (object present) — a direct
  walkability test of the footprint.
- Log adjacent-entity and curiosity probe events in BOTH arms so the frontier
  diff becomes possible.
- Amendment A analysis rules apply: success requires beyond-footprint
  connectivity change; v57 matched-player-footprint discipline; per-branch
  configuration-hold verification with censoring; duration-matched NOOP
  control; empty delta = censored evidence, never proof.

## Runs cited

Pushed: v319 (869 branches), v320 (2,497), v321 (132). Pre-push: v313
(3,270), v314 (3,333), v316 (99), v317 (99), v318 (110). Excluded as
lineage-mismatched: v311, v312, entity-v10-room3-* (22 dirs). Paths under
`experiments/lolo1-entity-v10/evaluations/`.
