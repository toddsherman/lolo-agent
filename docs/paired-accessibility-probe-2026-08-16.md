# Paired native accessibility probe — preregistration (2026-08-16)

Status: PREREGISTERED before execution; results section to be appended
Design source: `docs/offline-accessibility-diff-2026-08-16.md` (Amendment A,
`docs/direction-review-2026-08-16.md` §3.A)
Runs: `entity-v322-room3-paired-probe-arm-a-pushed-d12` (Arm A),
`entity-v323-room3-paired-probe-arm-b-prepush-d12` (Arm B)

## Hypothesis

> The confirmed one-cell push (source `(7,6)` → destination `(8,6)`) changes
> bounded player accessibility or interaction frontiers in Room 3 beyond the
> pushed object's own footprint, measurable as a difference between two
> matched search arms launched from the same v318 lineage one push apart.

Falsification (scoped): identical beyond-footprint coverage and frontier
sets across both arms at the declared budgets — recorded as censored
evidence further downweighting the Room 3 single-push Gate 4 vehicle
(never as proof of neutrality; learnings §2, §4.14).

## Arms

Both arms: memory from v318 decision 1 (`--resume-run` + `--resume-decision
1`), byte-identical planning flags — depth 12, beam 128, v319/v320's
reserve profile (position 16, milestone 32, world-state 32, goal-proximity
12, goal-world-state 12, curiosity 32 @ 8.0, stationary-history 2,
missing-player 4/2), probe logging enabled (adjacent-entity, curiosity,
effect probes, proactive limit 16), frozen entity-behavior checkpoint,
`--decisions 8`, evaluator scene-change stop omitted (v321 precedent).

- **Arm A (pushed):** physical state = v318 archived `state-00000117`
  (`--resume-state-archive-id`), state digest `5394ec3f…039e`.
- **Arm B (pre-push):** physical state = v318's goal-milestone rollback
  checkpoint (`--resume-state-checkpoint-event-seq 2026`), state digest
  `33addc6c…fb92` — the same lineage one push earlier.

Input digests verified pre-launch (host `c03694c5…`, core `a3450a09…`, ROM
`914c6769…`, neural ckpt `bb7a7a37…`, entity ckpt `984b8334…`); v318
`events.jsonl` bit-identical to what v319–v321 resumed from
(`0bbe1d15…`). Exact full command lines are recorded in the session
recon transcript and reproduced by the two run manifests at launch.

## Budgets and stopping conditions

- Arm A wall-clock ceiling 7,200 s; Arm B 10,800 s — enforced by an
  external watchdog (no ceiling flags exist in `lolo-neural-run`; a hard
  branch ceiling would need a code change and is out of scope).
- Event expectation ≤200k/arm, branch expectation ≤25k/arm — monitored,
  with overrun noted in results.
- One native run at a time (Arm A first — cheaper by v319 precedent, and
  it shakes down the harness before the longer pre-push arm).
- No rerun on identical negative evidence; no depth/beam escalation.

## Analysis rules (fixed before execution)

1. Success requires a beyond-footprint delta: cells `(7,6)`/`(8,6)` and
   pixel equivalents excluded from all claimed deltas.
2. v57 discipline: pose/sub-tile differences at the same coarse cell are
   not deltas.
3. Directed analysis targets (planner is autonomous; these are scored
   post-hoc, not forced): (a) traversal into column ≥8, rows 5–7; (b)
   reach `(8,10)` and the A-interaction at `(8,11)` in both arms; (c)
   occupancy of `(8,6)` in Arm B (empty tile) vs Arm A (object present).
4. Probe/frontier events compared across arms (both arms log identically —
   the censoring failure of the offline diff cannot recur).
5. Configuration-hold: branches whose player-masked world signature departs
   the arm's expected configuration are analyzed separately (a second
   manipulation invalidates the fixed-layout comparison for that branch).
6. Budget-exhausted non-reach is censored, never "unreachable."

## Known asymmetries (accepted, disclosed)

- Arm A auto-imports the 33addc6c rollback capability (its rollback state
  IS Arm B's root); Arm B starts with no rollback checkpoint. Harmless at
  8 decisions (< exhaustion minimum 16); disclosed here.
- Both arms share v318-decision-1 memory, which includes learning from the
  push search — the price of memory-matched pairing; Arm B's world is
  pre-push while its memory is not.
- First native run on the WP1-refactored planner (`236ea65`); the
  extraction was verified byte-identical on telemetry, so any divergence
  from v319-era behavior observed here would itself be a finding.

## Results (appended 2026-08-16; paired analysis per preregistered rules 1–6)

### 1. Coverage deltas

**SEARCH-REACH (verified branch endpoints).** Arm A: 4,061 endpoints (4,026
option + 35 generic), 8 coarse cells. Arm B: 12,267 endpoints (12,232 option
+ 35 top-level), 12 coarse cells. After excluding footprint cells
(7,6)/(8,6) (rule 1; v57 pose collapse changed nothing in either arm —
rule 2):

- Shared beyond-footprint coverage: (6,6),(6,7),(6,8),(6,9),(6,10),(7,10),(8,10).
- **B-only: (8,7) n=30, (8,8) n=22, (9,8) n=2.** A-only: none.

Budget-honesty caveats: the arms differ ~3x in branch count (4,061 vs
12,267) and in root cell (A rooted at (7,6), B at (6,9) — a consequence of
the one-push-apart physical states). Both arms completed 12/12 search depths
with zero budget truncation (A: 776 s of 7,200 s, 2 completed option
searches; B: 1,902 s of 10,800 s, 1 completed search, beam-saturated at
1,408 candidates/depth from depth 5). A's smaller branch yield is endogenous
to its root's reachable graph, not a budget cut — but per the censoring
discipline A's absences are treated as censored, not absent.

**WALK-REACH (committed trajectories).** Beyond-footprint committed cells —
A: {(6,6),(7,10),(8,10)}; B: {(6,8),(8,7),(8,8),(9,8)} (start cells
excluded). The sets are disjoint. B's decisions d6–d8 additionally committed
occupancy of (8,6) then (7,6) (footprint; see §2, §4).

### 2. Directed targets (scored post-hoc per rule 3)

- **(a) Column ≥8, rows 5–7:** Arm A **NO** — no endpoint, commit, or probe
  endpoint at x≥128 in y 80–112 despite 2,269 endpoints at adjacent (7,6);
  searches completed, non-reach censored. Arm B **YES** — (8,7) reached in
  search (n=30) and committed at d5; beyond-footprint-admissible evidence is
  (8,7) only ((8,6) is excluded). Row 5 was reached in neither arm (B
  probes: up→(8,5) blocked).
- **(b) (8,10) and the (8,11) A-interaction:** Arm A reached (8,10) by
  search (195) and walk (d3–d4) and executed the A-interaction toward (8,11)
  repeatedly (curiosity a at depth 9; facing-button down+a/b at d4,d5; 7 WEC
  audits): **all null**. Arm B reached (8,10) by search only (109 endpoints,
  no commit) — and **every one of those 109 branches carries an
  endpoint-persistent (7,6) change**, i.e., all of B's (8,10) coverage is
  configuration-departed (rule 5). B's sole down-facing 'a' branch at
  (8,10): null; no probes at (8,11). Outcome concordant (null/null); access
  asymmetric (A clean-config, B departed-config only).
- **(c) Occupancy of (8,6) — the direct walkability signal: confirmed in
  the predicted directions.** Arm A: zero occupancy across 4,061 endpoints
  + 8 commits, with the tile positively probed blocked (8 adjacent probes,
  2 facing-button, 5 curiosity, 16 WEC audits directed in — no movement, no
  effect). Arm B: 188 search endpoints, committed occupancy d6→d7,
  pre-commit adjacent probe up→(8,6) movement confirmed. The object
  demonstrably blocks the tile it occupies. Within-footprint, so excluded
  from the delta claim per rule 1.

### 3. Probe / frontier comparison

Interaction outcomes are null-concordant everywhere probed in both arms (0
entity effects confirmed, 0 evidence accepted run-wide in each). Arm A's
frontier toward the column-8 rows 5–8 band is **positively closed at every
edge its reachable set touches**: (7,6)→(8,6) blocked (object), (7,6)→(7,7)
blocked, (8,10)→(8,9) blocked, (9,10) blocked. B's blocked probes agree on
the shared edges ((9,6),(9,7),(8,5),(7,7) blocked) — the sole edge that
differs is (8,6) itself. B-only interactable frontier: confirmed 'a'-button
world-effect controls at (5,8),(7,8),(6,7),(7,7),(6,8) — cells/actions A
never audited (censored on A's side, root-position artifact). A-only: the
(8,11)-directed interaction battery (B's absence censored). Nothing
interactable-with-effect exists in one arm's tested set and not the
other's.

### 4. Configuration-hold and new confirmed manipulations

**Arm A: clean hold.** World hash dd9de862 on all 4,061 endpoints and 8
commits; zero partitioned branches. Reported separately: 10 stability
probes with a persistent within-footprint (8,6) delta. **New confirmed
manipulations: 2** — d4/d5 (7,10)-left, moving_directional_ray, entity
effect at (6,10).

**Arm B: hold certification failed at instrument level.** The player-masked
signature is identical across all 12,232 branches and 8 commits — yet the
run demonstrably displaced the object (committed d7–d8 occupancy of (7,6),
d8 left-probe movement at (6,6), consistent with a new westward push
(7,6)→(6,6)) while `world_effects_accepted=0`. The coarse signature is
insensitive to object displacement. Partition by tracked/pixel evidence:
endpoint-persistent (7,6) effects in 140 branches — including **all**
(8,10)×109, (8,8)×22, (9,8)×2 — plus (6,6)×20, (8,6)×6; in-branch tracked
footprint disturbance in the majority of branches ((7,6): 6,659; (8,6):
3,748). Consequently the "clean" status of B's (8,7) n=30 endpoints cannot
be certified: routes into column 8 plausibly transit the footprint. **New
confirmed manipulations: 9**, including d3 up@(8,7),
moving_directional_ray, effect cell (8,6), displacement [0,-1]
(footprint-touching).

### 5. VERDICT

**No beyond-footprint delta at budget (censored).** The falsification
condition does not fire — coverage sets are not identical (B-only
(8,7),(8,8),(9,8); disjoint walk sets) — but the confirmatory claim fails
on both legs: (i) A's absences are censored per the discipline (~3x branch
disparity, different roots); (ii) B's beyond-footprint column-8 coverage
cannot be certified configuration-held, because the signature instrument
was shown blind to object displacement and (8,8)/(9,8)/(8,10) coverage is
affirmatively configuration-departed. The only fully certified paired
contrast — (8,6) blocked in A, walkable in B — is the preregistered direct
walkability signal but is footprint-excluded from the delta. Directional
evidence favors the hypothesis (target (a) B-yes/A-no; A's measured closed
frontier; endogenous branch-count contraction) and is recorded as such, not
as confirmation.

### 6. Implications and next experiment

The Room 3 single-push Gate 4 vehicle is neither confirmed nor
downweighted-for-neutrality: coverage differed, so the scoped falsification
clause does not apply, but the confirmatory read is blocked by one
identified instrument defect, not by the world. The candidate mechanism is
now sharp: (8,6) is the sole open edge into the (8,7)+ band from the
object's neighborhood; every alternative tested edge is blocked in both
arms. **Next experiment (bounded, falsifiable):** fix hold certification —
extend branch telemetry so each `human_prior_option_branch_verified`
carries the tracked object cell (or fold tracked object cells into the
player-masked signature) — then rerun Arm B alone at identical settings
(depth 12, beam 128, seq-2026 checkpoint, ≤10,800 s, ≤200k events, no
escalation). Score one preregistered bit: does ≥1 certified
configuration-held branch (object at (7,6) throughout) reach column ≥8,
rows 5–7? Yes → beyond-footprint delta confirmed against Arm A's positively
closed frontier, vehicle promoted. No, at completed depth → the
accessibility difference collapses to footprint-only; record as
censored-negative and downweight the vehicle per learnings §2/§4.14.

### 7. Rerun preregistration (added 2026-08-16, before execution)

Instrument fix landed as commit `ddae223`: every
`human_prior_option_branch_verified` event now carries a self-contained
anonymous-track block. Rerun:

- Run ID: `entity-v324-room3-paired-probe-arm-b-rerun-certified-d12`,
  identical to Arm B in every setting (seq-2026 checkpoint root, depth 12,
  beam 128, same reserves/probes/decisions, ceiling 10,800 s, ≤200k
  events, no escalation), on the telemetry-extended code.
- Certification predicate (fixed now): a branch is configuration-held iff
  `anonymous_object_track_cells` equals the root's tracked cells and the
  tracked state signature matches the root's — i.e. the object remained at
  `(7,6)` throughout the branch.
- Single scored bit: does ≥1 certified configuration-held branch reach
  column ≥8, rows 5–7 (pixel x ≥ 128, y 80–112)?
- Arm A is not rerun: its result (positively closed frontier, clean hold,
  zero footprint occupancy) is already certified by its world-hash
  uniformity and stands as the comparison.
