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

## Results

(to be appended after the run completes)
