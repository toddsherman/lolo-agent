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

## Results

(to be appended after both arms complete)
