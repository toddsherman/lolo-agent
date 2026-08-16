# Object-removed configuration probe — preregistration (2026-08-16)

Status: PREREGISTERED before execution; results to be appended
Run: `entity-v325-room3-object-removed-probe-d12`
Predecessors: `docs/paired-accessibility-probe-2026-08-16.md` (§8 proposed a
"westward-displacement" probe), corrected by the v324 track-telemetry recon
recorded in `docs/learnings.md` §4.29.

## Premise correction

The §8/§4.28 inference of a "spontaneous westward push `(7,6)→(6,6)`" is
**falsified** by v324's track telemetry. The manipulation that opens the
column-8 band is **removal**: the `(7,6)` anonymous entity was shot
(`a` → in-place transformation), the transformed object pushed one cell
east, then shot again and expelled east along row 6. A genuinely
westward-displaced state exists only in 71 unarchived (released) branch
endpoints and is not restorable. The v324 committed six-cell effect set
`[[2,6],[3,7],[7,6],[11,6],[12,6],[14,5]]` is accumulated history, not
endpoint configuration: by decision 7 five of the six cells had physically
relaxed to baseline (frame-diff verified); `(14,5)` is the HUD shot
counter, `(2,6)/(3,7)` are autonomous patroller leak (register only in
button branches), `(8,6)/(11,6)/(12,6)` are transient transit cells.

## Hypothesis

> With the `(7,6)` entity removed and the world otherwise at baseline, the
> column-8 band `(8,7)/(8,8)/(9,8)` is reachable under certified
> configuration-hold — whereas both certified baselines (pushed
> configuration, Arm A v322; pre-push configuration, v324) could not reach
> it: identical envelopes
> `{(6,6),(6,7),(6,8),(6,9),(6,10),(7,10),(8,10)}`.

A YES is the project's first verified accessibility-improving manipulation
(Gate-3 shape, WP6's Room 3 native gate). A certified NO at completed
depth is censored-negative and would indicate the removal alone does not
open the band either.

## Root and configuration

- Physical state: v324 decision-7 snapshot, sha
  `bdb5bbde46acbd44dde775dec02d24a2ac9b1efe329967c0251e436c9e6b0d49` —
  player at `(7,6)` (the entity's vacated home), object removed, world
  relaxed. Decision-8 restore resolved to the byte-identical state, and
  `human_prior_life_loss_confirmed=False` throughout v324, so the
  configuration survived to run end.
- Maximal symmetry with Arm A (v322): identical root player cell `(7,6)`,
  one manipulation apart (entity parked at `(8,6)` vs entity removed).
  Band entry requires ≥2 real moves — the bit is not trivialized by root
  placement (archive roots inside the band were considered and rejected
  for exactly that reason).
- Resume: memory from v318 decision 1 (as all prior arms);
  `--resume-state-run <v324> --resume-state-decision 7`. The decision
  event carries no track keys, so the root track seeds empty
  (`legacy_track_reconstructed=true`) and branch effects measure against
  the new run's root frame — correct-by-construction for hold
  certification (code-verified path).
- All other flags byte-identical to v322/v323/v324; ceiling 10,800 s,
  ≤200k events; ~30 min expected.

## Certification predicate and scored bit (fixed before execution)

- Configuration-held iff `anonymous_object_track_cells == []` and tracked/
  confirmed state signatures remain empty/null. No autonomous modulo-set:
  v324 shows patroller motion and the HUD cell never register in
  movement-only branches, and band entry from this root requires no
  buttons. Conservative self-exclusions disclosed: heart pickups register
  the heart cell (remaining hearts off the scored route); any `a` press
  counts as departure.
- **New rule (instrument gap found in recon):** `archive_branch_restored`
  events from the causal archive carry no track fields and silently reset
  the tracker. Certification is valid only for branches from decisions
  before the first such restore in this run; later decisions are reported
  but not certified. (Instrument fix — extending the causal-archive events
  with the track block — is queued separately and intentionally not
  blocking this probe.)
- **Scored bit:** does ≥1 certified configuration-held branch reach
  `(8,7)`, `(8,8)`, or `(9,8)`?
- Disclosed risk: an off-screen respawn/return of the removed entity would
  re-close the corridor mid-run; the predicate catches it as `(7,6)`
  departure (conservative). v324 saw no return within its horizon.

## Results (appended 2026-08-16)

Run complete, no errors, 1,462 s (ceiling 10,800), 9,691 verified branches
with endpoints, zero life-loss confirmations. First causal-archive restore
at seq 15,054; per the preregistered rule, certification uses only
pre-restore branches (the decision-1 root search).

**The preregistered bit scores YES: 135 certified configuration-held
branches reached the band** (`(8,7)` from seq 248 onward, `(8,8)`,
`(9,8)`).

Certified configuration-held coverage (cells == `[]`, pre-restore):

```text
(6,6) (6,7) (6,8) (6,9) (6,10) (7,6) (7,10) (8,6) (8,7) (8,8) (8,10)
(9,8) (10,6) (10,7) (10,8) (11,6) (11,7) (11,8) (12,6) (12,7) (12,8)
(12,9) (12,10) (12,11)
```

24 cells — versus the two certified baselines' identical 7-cell envelopes
(pushed Arm A v322; pre-push v324). Removing the `(7,6)` entity opened the
former footprint cells `(7,6)/(8,6)`, the scored band, and the entire
eastern region through column 12 — including `(12,11)`, a known remaining
heart cell. Every cell reached by non-certified branches was also reached
certified (non-certified-only set: empty).

**Conclusion: this is the project's first verified
accessibility-improving manipulation.** The removal of the `(7,6)` entity
(two shots + one transformed-object push, discovered by ordinary search)
changes bounded certified accessibility from 7 to 24 cells including a
milestone-bearing cell. Gate 3's success criteria are substantially met:
before/after delta (7 → 24 certified cells), measured against two matched
baseline configurations that both lack the delta, from an archived
restorable state, with uncertainty scoped to the declared budgets. Formal
Gate 3 closure wants one clean repetition of the delta from a fresh
restore, and the neutral-control comparison is currently cross-run
(v322/v324 baselines) rather than same-run — both noted for the
repetition run.

Next (per roadmap §12 Gate 3 → Gate 4): repeat the delta from a fresh
restore to close Gate 3 formally; then the Gate 4 question becomes
concrete — can the planner *choose* the removal because of its measured
accessibility consequence, rather than stumbling into it: wire the
verified delta into hypothesis preference (direction-review Amendment A's
restore-selection ablation, WP8 seams) and test whether preparation →
`(12,11)` heart follows within budget.

Post-run note: v325's committed trajectory itself collected the heart at
`(12,11)` (decisions 2–4; `human_prior_collected_heart_slots` gained
`[192,176]` at d4) — the first Room 3 heart since v311-era stalls. The
manipulation → accessibility → milestone chain has now occurred end-to-end
in one run; Gate 4 requires making the preparation step deliberate.

### Repetition preregistration (Gate 3 closure; added before execution)

Run `entity-v326-room3-object-removed-repetition-d12`: identical to v325
in every setting and root (v324 d7 snapshot `bdb5bbde…`), fresh process,
same ceilings, same certification predicate and causal-restore rule.
Scored bits, fixed now: (1) ≥1 certified configuration-held branch reaches
the band — must repeat; (2) the certified coverage set is compared to
v325's 24-cell set — substantial agreement (≥80% Jaccard) closes Gate 3's
"repeatable from an archived state" criterion; divergence is reported and
scoped, not hidden.

### Repetition results — GATE 3 CLOSED (appended 2026-08-16)

`entity-v326-room3-object-removed-repetition-d12`: complete, 1,470 s,
9,691 branches. **Bit 1: 135 certified band branches — repeats. Bit 2:
identical 24-cell certified envelope, Jaccard 1.0 — closes.** The heart at
`(12,11)` was collected again at decision 4. Honest caveat: on a
deterministic emulator an identical-settings repetition primarily
certifies the restore/serialization path (fresh process, digest-verified
inputs) rather than sampling variability; the informative contrast remains
cross-configuration (24 vs 7 certified cells against v322/v324). Gate 3's
roadmap criteria — before/after delta, reproduced from an archived state,
matched baselines lacking the delta, budget-scoped uncertainty — are all
met on the assisted development track. Strict-track Gate 3 still requires
a WP5-clean re-measurement (linter now enforces the boundary).
