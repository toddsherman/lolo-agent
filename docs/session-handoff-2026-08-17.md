# Session handoff — 2026-08-17 (Gate 3 closure, WP5 completion, the Gate 4 lever series)

Status: recovery point for a cold agent or a returning session
Operator: Claude (Opus 5) sessions driven by Todd, running the standing loop
Companion docs: `docs/roadmap.md` (plan of record, amendments §17–§20),
`docs/learnings.md` (negative-results log, §4.26–§4.52)
Predecessor: `docs/session-handoff-2026-08-16.md` — still the authority on the
WP0/WP1 build, the direction-review panel, and the session-artifact paths. It
is not superseded; this file continues it.

## 1. Purpose

Todd's standing goal is unchanged: run the research loop — build, bounded
experiment, record learnings, improve — until a model beats *Adventures of
Lolo* under the strict interaction-only claim. This file records where every
work package stands after the 2026-08-16/17 campaign, what is committed and
what is not, what is waiting on Todd, and what the next agent should do first.

Read this file, then `docs/learnings.md` §4.42–§4.52 (the most recent results
and the ones most likely to be misread), then `docs/roadmap.md` §17–§20.

## 2. How this loop works

Every capability step is a bounded cycle. Skipping a step is how the project
produces results it later has to retract.

1. **Preregister.** Write the hypothesis, the arms, the exact command lines,
   the fixed bits, the VOID conditions, and the reading that will be given to
   each possible outcome — *before* any arm runs, in the dated experiment
   note. Any mixed outcome is FAIL; that is declared up front so it cannot be
   renegotiated afterwards.
2. **Run bounded.** Declared wall-clock, event, and dollar ceilings; the
   supervisor terminates the process group at a ceiling
   (`docs/research-loop.md`). A crashed arm is VOID, not FAIL.
3. **Score against the fixed bits.** A separate scorer, byte-identical across
   reruns, validated first against a prior run whose answer is already known.
   No weight tuning, no budget re-sizing, no rerun on an identical negative.
4. **Record in `docs/learnings.md`** as a numbered §4 entry: hypothesis,
   measured evidence, evidence class (falsified / negative / engineering
   defect / not yet demonstrated), plan change, run IDs, source document, and
   the condition under which the direction may be reconsidered.
5. **Amend `docs/roadmap.md`** when the result changes the plan — as a dated
   amendment section that cites the learnings entries, never by silently
   editing the earlier text.
6. **Commit** with a capability-oriented message describing what the work
   established, not what files moved.

Two rules earned the hard way this campaign: budget as much design effort for
a gate as for the capability it gates, and require of every gate an instrument
that can contradict it (roadmap §18 item 4). The same discipline applies to
operational monitoring — §4.52 records the orchestrator calling a healthy run
dead from two instrument artifacts that agreed only because both were its own.

## 3. Where every work package stands

### WP0 — evaluation partitions — LANDED (`b95b68f`)

`configs/evaluation-partitions.json` (immutable manifest), `partitions.py`
loader with rejection of training writes from withheld/sequel partitions,
digest audit over every persistent artifact class, partition telemetry.
The withheld allocation shipped as a default and is **still unratified** —
see §7.

### WP1 — object-track extraction — LANDED (`236ea65`)

`lolo_agent/object_tracks.py`, zero planner behavior change, telemetry
verified byte-identical.

### WP2-lite — multi-track correspondence — LANDED (`befd629`)

`lolo_agent/object_correspondence.py` with endpoint-relative track state,
the contract roadmap §17 item 3 hardened after §4.29 showed five of six
accumulated cells were stale at v324 d7. Full WP2 planner/archive
integration (backlog Task D) is **not** done.

### WP3 — transition descriptors — LANDED (`e535148`)

Transformation and removal chains represented in behavior descriptors. The
removal-chain **native gate** added by roadmap §17 item 2 has not been run.

### WP4 — relational behavior prediction — untouched this campaign

The frozen `anonymous-behavior-relational-v2-clean.json` checkpoint remains
the incumbent in every native run.

### WP5 — learned controllable-region tracker — COMPLETE, PROMOTED TO SHADOW, NOT WIRED

The campaign closed with **PASS / PROMOTE-to-shadow** (§4.42): the learned
detector-free masking convention (tracker v4 + pixel head v3 occupied-v2 +
reconstruction v3 + detection quantity v2) passes the functional gate on
every axis and every corpus, and **exceeds the assisted incumbent** on
stability (1.000 / 0.9995 / 0.9728), preservation (0.994–0.999 vs
0.72–0.77) and in-place detection. Six gate iterations, each isolating one
mechanism: mask-irrelevance (§4.31) → coverage (§4.32) → resolution (§4.34,
§4.35) → symmetric erasure (§4.37) → in-place erasure (§4.38) → anchor
marginality (§4.39, §4.40, §4.42).

**Not yet wired.** Shadow wiring (learned convention as telemetry alongside
assisted, with divergence / empty-mask / tail dashboards) touches
`neural_planner.py` and has queued behind the E-series all campaign. Watch
items for shadow telemetry — to observe, **not** to re-tune: v325's residual
2.7% stability tail, empty-mask rates, and the detection-channel shift
toward the masked-region differential.

Claim boundary: WP5 is what a strict-track re-measurement of Gate 3 needs.
Until it is wired and has native evidence, every accessibility result below
is assisted-lineage.

### WP6 / WP6a — accessibility measurement — PRODUCTIZED, GATE 3 CLOSED (assisted)

`lolo_agent/accessibility.py` and `accessibility_preference.py` landed;
the certified-hold paired probe is productized (`33bbeb7`) as a ~25-minute
instrument. **Gate 3 is CLOSED on the assisted track** (roadmap §17 item 4):
removing the `(7,6)` entity opens **24 certified cells against 7**,
including the milestone-bearing cell `(12,11)` (§4.30), reproduced from a
fresh restore at Jaccard 1.0 (v326). Strict-track Gate 3 remains open and is
gated on WP5.

Do not overstate this: the paired v318 push was certified **neutral**
(§4.28) before the removal was certified **enabling** (§4.30). Manipulation
value is a property of configurations, not of manipulations.

### WP7 — phase-conditioned mechanics — deliberately off Gate 4's critical path

Direction-review amendment E, adopted in roadmap §17 item 7.

### WP8 — object-level hypothesis planner — OPEN after four measured failures; E6 in flight

`lolo_agent/relational_planner.py` and `conflict_root_mining.py` landed. The
chain machinery is validated (§4.44: `establish` → `hold` → `exploit` formed
with correct linkage and pre-execution logging, option reuse observed at d8,
non-interference confirmed). **Gate 4 is open.** Four levers are measured and
each failed for a distinct named mechanism:

| Lever | Runs | Mechanism of failure | Entry |
| --- | --- | --- | --- |
| Restore preference (accessibility scalar) | v327/v328 | **Redundant** — the baseline novelty/coverage scorer already preferred the same removal-class branch; committed trajectories identical | §4.43 |
| Hypothesis search reserve | v330/v331 | **No searches to ride** — the exploit held authority across decisions 3–7, which contained zero option searches; its budget was never read | §4.45, §4.46 |
| Commit steering (seam S1) | v334/v335 | **Starves supply** — steering worked (7 of 20 instants differed) and made the outcome worse; archive geography narrowed from columns 6–12 to 6–8 and every restore ratcheted westward | §4.50 |
| Target-aware restore key alone (seam S2) | v336/v337 | **Target not a candidate** — the cell that needed holding was the current position `(12,10)`, which was never deposited as an archive in either arm | §4.51 |

What survives all four: the discriminator, and a sharp statement of the gap.
The incumbent machinery *finds* valuable configurations unaided (§4.43) and
*wanders into* the target neighborhood (§4.48, distance 1 at d17) but has no
mechanism to **close** — standing adjacent to a certified milestone is worth
zero to the incumbent scorer, and a stagnation restore valued at 60.35 wins.
E5 additionally proved the intervention *shape* implementable: supply
preserved, columns 6–12 in both arms, `hold_matching_candidates` ratcheting
eastward (§4.51 bit 1).

**E6 is in flight** — seam S3, "deposit the certified-adjacent position as an
archive so the existing restore key has a candidate to reach". Status in §9.

### WP9 — reward and value learning — PAUSED AT THE REPRESENTATION

Step 1 falsified three times, each with the paired gates catching what a
single gate would have invited tuning on:

- §4.33 — heart-inseparability; reversion-based negative valence 0/14
  because at the fatal commit the matched NOOP control also dies.
- §4.36 — delayed-divergence valence **solved** (14/14), separation falsified
  again by reset bleed-through; class-scoped valence bleeds.
- §4.41 — occurrence-scoped valence **starves** the terminal class; recall
  improved monotonically (0.149 → 0.319 → 0.574) and still failed 0.80.
  **The event unit itself is wrong.** No fourth rescore of the
  matched-endpoint-pair + successor-window unit.

Next incarnation is object-level (collection = a tracked appearance ceasing
at a heart cell; death = control-loss divergence on the tracked controllable
region), scheduled **after** WP2/WP3 integration provides object-level event
streams, preregistered against the same three gates. Landed meanwhile:
`milestone_discovery.py`, `milestone_discovery_run.py`, and the event census
in `docs/milestone-event-census-2026-08-16.md`.

### WP10 / WP11 / WP12 — untouched

### Cross-cutting modules landed this campaign

`partitions`, `object_tracks`, `object_correspondence`, `counterfactual_labels`,
`strict_lineage` (static linter enforcing the assisted/strict boundary,
`8b74c3b`), `controllable_tracker(+_train)`, `pixel_mask_head(+_train)`,
`mask_sensitive_gate`, `functional_mask_gate`, `tracker_substitution_replay`,
`tracker_ood_eval`, `accessibility`, `accessibility_preference`,
`milestone_discovery(+_run)`, `relational_planner`, `conflict_root_mining`,
and the ratified `strict_from_assisted_state` reward track (`a3ac7d1`,
`786200d`).

Suite at time of writing: **1,085 tests, 4 skipped, OK in ~17 s**
(`.venv/bin/python -m unittest discover -s tests`), with the uncommitted E6
seam present in the working tree.

## 4. Run inventory v322–v339

All runs live under
`experiments/lolo1-entity-v10/evaluations/<run-id>/` (gitignored). **Every one
is assisted-lineage** — no result below may be cited as strict-track evidence.

The live-state lines in this section and in §8 are perishable: re-verify with
`ps` and `ls experiments/lolo1-entity-v10/evaluations/` before trusting them.

| Run | What it established | Entry |
| --- | --- | --- |
| `entity-v322-room3-paired-probe-arm-a-pushed-d12` | Arm A of the first paired probe (4,061 endpoints, 776 s). The object blocks the tile it occupies: `(8,6)` positively probed blocked with the object present. Arm A's frontier into the column-8 band is closed at every tested edge. | §4.27 |
| `entity-v323-room3-paired-probe-arm-b-prepush-d12` | Arm B pre-push (12,267 endpoints, 1,903 s). Reached `(8,7)`,`(8,8)`,`(9,8)`; 11 confirmed manipulations across the two arms against v313's 0/1,756. Configuration-hold certification **failed at the instrument level** — the player-masked signature was blind to object displacement. | §4.27 |
| `entity-v324-room3-paired-probe-arm-b-rerun-certified-d12` | Rerun after the tracked-object-cells instrument fix (`ddae223`), 12,232 branches, ~30 min. **Bit = 0**: certified-held coverage identical to the pushed arm, 7 cells each. All 54 band-entering branches carry `(7,6)` in their effect set — object disturbance is necessary for band entry. The v318 push is certified accessibility-**neutral**. Its d7 snapshot is the root of every removal-configuration run. | §4.28, §4.29 |
| `entity-v325-room3-object-removed-probe-d12` | 9,691 branches, 24 min. **Bit = YES: 135 certified configuration-held branches reached the band; 24 certified cells against 7.** First verified accessibility-**improving** manipulation. Collected `(12,11)` at d4. Zero life losses. | §4.30 |
| `entity-v326-room3-object-removed-repetition-d12` | 9,691 branches, 1,470 s, fresh process. 135 certified band branches; identical 24-cell envelope, **Jaccard 1.0**; `(12,11)` collected again at d4. **Gate 3 CLOSED on the assisted track.** Honest caveat recorded: on a deterministic emulator this certifies the restore/serialization path, not sampling variability. | probe doc §repetition; roadmap §17 item 4 |
| `entity-v327-room3-wp8lite-control-w0-d12` / `entity-v328-…-treatment-w1-d12` | WP8-lite paired ablation, 278-field config equality verified. Bit 1 PASS (treatment's d2 restore selected the removal-class archive on certified value, bonus 25.0, full component attribution); **Bit 2 FAIL** (identical committed trajectories — the control's plain frontier score restored the same branch non-deliberately, 29.578 vs 54.578 on the same winner); Bit 3 PASS. Also corrected the record: v324's committed line *did* collect hearts at d1/d3; the uncollected object is specifically the `(12,11)` milestone. | §4.43 |
| `entity-v329-room3-relational-shadow-d12` | Mandatory telemetry-only shadow, 79,493 events. The full hypothesis chain formed with correct linkage and was logged **before any realization**; a hold option was re-instantiated at d8 (first observed option transfer); non-interference confirmed. The exploit's `budget_exhausted` termination is the **expected null in telemetry mode**, not a budget finding. | §4.44 |
| `entity-v330-…-e1-control-off-d12` / `entity-v331-…-e1-treatment-selection-d12` | E1 **FAIL**. Committed trajectories identical state-id by state-id; the treatment's telemetry differs by exactly its 16 relational events. Mechanism: seam-opportunity, not budget size — decisions 3–7 contained zero option searches, so the reserve seam never fired. Units lesson: `realization_branch_budget` is a per-depth parent-reserve slot count, not a verified-branch count. | §4.45 |
| (offline over v327–v331) | **The planner had stopped searching entirely.** The only search in each run is the decision-0 resume audit, which executes before the removal configuration exists. Cause: the stagnation deferral gate defers whenever any archive branch carries a frontier flag — the planner's own archive growth suppresses its search gate. Contrast: v325, the run that reached `(12,11)`, ran two planner-initiated searches. | §4.46 |
| `entity-v332-room3-e3-pre-control-off-d16` | Cheap pre-check. The control does not collect `(12,11)` in 16 decisions, so the discriminator is technically alive — but the trajectory is closing (Chebyshev 3 → 2), which would have made an E3 pass read as a 2–3 decision speedup rather than a capability difference. One search, the d0 resume audit, reproducing §4.46 at 16 decisions. | §4.47 |
| `entity-v333-room3-e3-pre-control-off-d24` | The definitive extension, and the opposite of "converging". The control reached `(12,10)` — **distance 1** — at d17, then a `human_prior_graph_stagnation` restore valued at `persistent_frontier_value` 60.35 moved it to `(10,7)` at d18 and it never returned (d18–d24 distances 4,5,5,5,4,5,6). **Discriminator VALIDATED and the capability gap named**: the gap is not speed, it is closing. Also the offline source for §4.49's supply measurement (52 of 56 commit-time archives carry the signature — 14×, not the 4 a stale precondition claimed). | §4.48, §4.49 |
| `entity-v334-…-e3-control-off-d24` / `entity-v335-…-e3-treatment-selection-d24` | E3 **FAIL** (bit 1 PASS, bit 2 FAIL, bit 3 PASS). The mechanism fired — 20 navigation instants, 7 with `differs: true` — and the outcome went backwards: treatment minimum distance 3 against the control's 1. First divergence at d6 was locally correct and causally decisive; the control's d6–d8 excursion *away from* the target is what deposited the archives it later restored into. Archive geography 6–12 (control, 44 deposits) vs 6–8 (treatment, 33); `hold_matching_candidates` decayed 1,4,3,2,1,1,1,1,1 westward. The control reproduced v333 state-for-state. | §4.50 |
| `entity-v336-…-e5-control-off-d24` / `entity-v337-…-e5-treatment-restore-only-d24` | E5 **FAIL** on outcome, and a far better failure. Bits 1, 2, 4 PASS: arms identical d1–d7, archive geography columns 6–12 in **both** arms (44 vs 43), `hold_matching_candidates` ratcheting **eastward** (1,3,3,6,6,5,4,3), and at d17 the objective **refused** the `(10,7)` baseline — cell-for-cell §4.48's failure instant — taking `(12,7)` instead. Reached distance 1 at d16, one decision earlier than the control. Bit 3 OUTCOME FAIL: `(12,11)` not collected. Mechanism: `(12,10)` was never deposited as an archive in either arm; the whole column-12 deposit set is `(12,6)`–`(12,9)`. The restore key re-ranks archives that exist; **the required candidate did not exist.** | §4.51 |
| `entity-v338-room3-e6-control-off-d24` | **IN PROGRESS.** E6 control arm, launched 21:45, 24 decisions, `--relational-navigation-seams restore_plus_deposit --relational-planner-authority off`. Its directory exists and `events.jsonl` is growing; health-check it that way, never by process pattern (§4.52). | §9 |
| `entity-v339-…` | **RESERVED for the E6 treatment arm. NOT STARTED.** No directory under `experiments/lolo1-entity-v10/evaluations/`. | §9 |

Discriminator status: no run from the **v318 pre-push root** has collected
`(12,11)` in-window — thirteen runs and counting (design §14.7). v325/v326
collected it at d4, but from the **object-removed root**, which is the point:
the configuration makes it reachable and the planner still cannot close from
the pre-push root.

Standing instruction from §4.46: **report option-search counts in every future
native run summary.** Search frequency is a first-class planner-health metric,
not an implicit assumption.

## 5. Artifacts and digests

Everything under `experiments/` and `runs/` is gitignored — local-only, not
recoverable from the repo.

**Gate reports** live in `experiments/lolo1-wp5/`. Each carries a digest over
its canonical payload, reproduced by rerunning the scorer, but the key is not
uniform — check the shape before scripting against it. The three Gate 4
reports (`e1`/`e3`/`e5`) use `digest_sha256`, a plain string. The twelve WP5
and WP9 gate reports use `content_digest`, also a plain string.
`wp8lite-ablation-report.json` uses `content_digest` as an *object* with
`algorithm` and `value` (its value is `19f4092f…`). In every shape the digest
is over the canonical payload and is **not** `shasum` of the file (e.g. the
E5 report's payload digest is `3c6b1e18…` while the file's sha256 is
`43d5b891…`). Verified byte-identical on rerun unless noted.

| Report | Payload digest | Result |
| --- | --- | --- |
| `substitution-replay-report.json` | `6061b45e…` | letter-pass, substantive no-promote (§4.31) |
| `tracker-ood-report.json` | `ecc5336b…` | OOD gap state-dependent, not uniform (§4.32) |
| `milestone-scoring-report.json` | `424bb775…` | WP9 step 1 falsified (§4.33) |
| `mask-sensitive-gate-report.json` | `7bb95c5e…` | fail on resolution (§4.34) |
| `mask-sensitive-gate-v2-report.json` | `1052c9ea…` | pixel reconstruction; replication is the wrong gate (§4.35) |
| `milestone-scoring-v2-report.json` | `898676b5…` | valence solved, separation falsified (§4.36) |
| `functional-gate-report.json` | `414c6576…` | symmetric erasure (§4.37) |
| `functional-gate-v2-report.json` | `7d1e5703…` | in-place erasure (§4.38) |
| `functional-gate-v3-report.json` | `01a9b128…` | bit (a) closed; v322 first full-pass corpus (§4.39) |
| `functional-gate-v4-report.json` | `99285632…` | stability tail resists the data lever (§4.40) |
| `functional-gate-v5-report.json` | `ac4bd00f…` | **PASS / PROMOTE-to-shadow** (§4.42) |
| `milestone-scoring-v3-report.json` | `e2c3434c…` | third falsification; the unit is wrong (§4.41) |
| `wp8lite-ablation-report.json` | `19f4092f…` | mechanism validated, behavior unchanged (§4.43) |
| `e1-gate4-report.json` | `6b6708db…` | E1 FAIL (§4.45) |
| `e3-gate4-report.json` | `26a3cc22…` | E3 FAIL (§4.50) |
| `e5-gate4-report.json` | `3c6b1e18…` | E5 FAIL, narrowed (§4.51) |

**Model and data artifacts**, also in `experiments/lolo1-wp5/`:
`controllable-tracker-v1…v4.pt` + per-version metrics JSON (v4 is the
promoted tracker); `pixel-mask-head-v1…v3.pt` + metrics (v3 is the promoted
head, occupied-v2 semantics); `wp5-labels-full{,-v2,-v3,-v4,-v5}.jsonl` each
with a `.manifest.json` (v5 = 21,633 roots / 96,682 arms, the tier-4
pose-diversity corpus); `collection_runs/`;
`wp8lite-accessibility-records.json` (the certified record store both E-series
arms load); `conflict-root-manifest.json` (E2's mined corpus — no organic
conflict roots, primary seeded root's record content signature `4d6377f4…`).

**Frozen baseline digests** are pinned in `configs/evaluation-partitions.json`
under `frozen_baseline`: the training checkpoint (`bb7a7a37…` file /
`0622f3c8…` parameters), the behavior checkpoint (`984b83c3…` / `ac1c5667…`),
the native host (`c03694c5…`), the Nestopia core (`a3450a09…`), the ROM
(`914c6769…`), and `planning_config_sha256` for v318–v321.

## 6. Committed versus local-only

- **Branch:** `main` at `ffb58b2` ("Vindicate the closing shape; find the
  lever attached to the wrong object"). **265 commits total; 64 unpushed.**
  `origin/main` sits at `e91cdc1` — the last commit before the campaign — so
  **nothing from this campaign has been pushed**. The campaign starts at
  `236ea65`.
- **Uncommitted working tree (the E6 build, in progress by another session) —
  five files:** `lolo_agent/neural_planner.py` (+277 lines: seam S3, the
  certified-adjacent archive deposit), `lolo_agent/neural_run.py` (the
  `--relational-navigation-seams` choice list gains `restore_plus_deposit`;
  at HEAD it is still `both | restore_only | off`),
  `tests/test_ensemble_planner.py`, `tests/test_relational_planner.py`, and
  `docs/wp8-search-scheduling-design-2026-08-17.md` (the E6 preregistration,
  §15 — see §9 item 1). A separate doc-sync lane also has `README.md`,
  `docs/architecture.md`, `docs/telemetry.md` modified and this file plus
  `docs/findings-index-2026-08-17.md` untracked. Also untracked: `tmp/` —
  leave it alone.
- **Worktrees:** `.claude/worktrees/{amazing-fermat-73b137, brave-allen-b00cf2,
  serene-allen-854961}`. All three branch heads (`6a8488a`, `786200d`,
  `9d2889e`) are **contained in `main`** — no unmerged work is stranded there.
  The 08-16 handoff's "worktree hold" is resolved (`e437bb1`).
- **Never in git:** `experiments/`, `runs/`, `build/`, `checkpoints/`,
  `artifacts/`, the ROM. Every run, report, tracker and label corpus in §4–§5
  exists only on this machine.

## 7. Decisions pending for Todd

1. **Push to remote.** 64 commits, the entire campaign, sit unpushed on
   `main` against `origin/main` at `e91cdc1`
   (`https://github.com/toddsherman/lolo-agent.git`). Local commits are part
   of the loop discipline; pushing awaits Todd's confirmation. This has been
   pending since the 08-16 handoff.
2. **Withheld-room allocation in `configs/evaluation-partitions.json`.** The
   manifest shipped as a default and declares itself `immutable: true`:
   development = Lolo 1 rooms 1–3; withheld = rooms 25/30/35/40/45/50 (the
   final room of floors 5–10, strict-track only, frozen); training = the
   remaining Lolo 1 rooms; sequel = all of Lolo 2. This binds all future
   evaluation and needs Todd's **explicit ratification before broad room
   training begins** (roadmap WP0 stopping rule). Also pending since 08-16.

Open questions that are not yet decisions, recorded so they are not lost:
whether Gate 4 continues on the assisted track through E6/E2/E4 or pauses for
WP5 shadow wiring first (every Gate 4 result to date is assisted-lineage and
cannot support the headline claim); and whether the frozen-signature risk
disclosed in §4.49 — the carried configuration signature is assigned at five
sites and never advanced by an ordinary committed decision — gets its own gate
before any experiment runs at accessibility weight > 0.

## 8. Active sessions and file ownership

At the time of writing (`ListAgents`): one running general-purpose subagent
(`a2a773d115971213d`), and three interactive peer sessions —
`brave-allen-b00cf2-a1`, `serene-allen-854961-b6`, `amazing-fermat-73b137-5b`.
**A native run is in flight:** `entity-v338-room3-e6-control-off-d24`, the
E6 control arm (§4 inventory, §9 item 1). Health-check it with the run's own
telemetry — growth of
`experiments/lolo1-entity-v10/evaluations/entity-v338-room3-e6-control-off-d24/events.jsonl`
and expected seq milestones — never with `pgrep -f "lolo-neural-run"`, which
can never match the module-form process; that is exactly the §4.52 error.
This line is perishable; confirm with `ps` before acting on it.

Files owned by other sessions during this campaign, not to be edited without
checking first: `lolo_agent/neural_planner.py`, `lolo_agent/neural_run.py`,
everything under `tests/`, `docs/wp8-search-scheduling-design-2026-08-17.md`,
`docs/learnings.md`, `docs/roadmap.md`. Read them freely. The general rule
that held all campaign: one agent at a time in `neural_planner.py`, disjoint
file ownership declared before parallel work starts, full suite as the
acceptance net.

## 9. Next steps and their preregistration status

1. **E6 — the certified-adjacent archive deposit (seam S3). PREREGISTERED;
   CONTROL ARM RUNNING, TREATMENT ARM PENDING.** When the agent occupies a
   cell on or adjacent to a certified milestone under hold, deposit that
   position as an archive so the already-working restore key (S2) has a
   candidate to reach.
   Strictly smaller than a veto, reuses the seam that works, adds no new
   preference weight (roadmap §20 constraint honored). *Status:* the seam code
   is in the working tree uncommitted, gated behind
   `--relational-navigation-seams restore_plus_deposit` so `both`,
   `restore_only` and `off` all keep their exact pre-E6 behavior. The code
   comments reference design §15 — **that section has now landed (also
   uncommitted, in the working tree) in
   `docs/wp8-search-scheduling-design-2026-08-17.md`**, written before either
   arm ran, carrying the seam, arms, exact command lines, the four fixed bits
   with ANY-mixed-outcome = FAIL, the pre-declared reading, the declared
   caveats, VOID conditions and scoring (§15.1–§15.8). The blocker is
   therefore cleared. Arms are `entity-v338-room3-e6-control-off-d24`
   — **launched 21:45 and still running**, telemetry accumulating under
   `experiments/lolo1-entity-v10/evaluations/` — and
   `entity-v339-room3-e6-treatment-deposit-d24`, **not yet started**. Do not
   launch heavy CPU work — full-suite runs, gate reruns, a second native arm
   — while an arm is in flight: design §15.7 VOID condition V5 voids an arm
   that exceeds the 10,800 s wall ceiling before `run_finished`, and a VOID
   is not evidence. The command block is design §11.3 verbatim with the
   §13.3 substitutions (24 decisions,
   `--relational-decision-budget 12`,
   `--human-prior-accessibility-preference-weight 0.0`, authority
   `off | selection`). Declared fallback if the deposit proves insufficient:
   a veto on stagnation-restoring away from a certified-adjacent position
   (§4.51, design §14.7).
2. **E4 — search request. DEFERRED behind E6, NOT PREREGISTERED.** Can an
   active objective cause an option search? Strengthened as a fallback by
   §4.50's measurement: with 1 distance-reducing branch of 7 hold-eligible at
   nearly every instant, re-ranking cannot help — only changing the candidate
   set can. Requires a forced-search control arm, since granting searches
   breaks matched budgets.
3. **E2 — deliberate selection under score conflict. ROOT DESIGNED, ADDENDUM
   NOT WRITTEN.** Corpus mining found **no organic conflict roots** across 74
   decision points in v322–v328, so E2 runs against the seeded novelty-decoy
   root staged in `docs/wp8-relational-planner-design-2026-08-17.md` §11.3
   (disclosed as constructed). Unaffected by the E1/E3/E5 failures — its
   discriminator is a restore-instant disagreement at decision 2, which the
   seam does reach.
4. **WP5 shadow wiring.** No preregistration needed (telemetry-only,
   non-interfering by construction), but it touches `neural_planner.py` and
   must not collide with the E6 build. Ship the divergence, empty-mask and
   tail dashboards with it; treat §4.42's watch items as observations.
5. **WP2 full planner/archive integration (Task D) and the WP3 removal-chain
   native gate.** Both are prerequisites for WP9's next incarnation.
6. **WP9 step 1, object-level.** Not before WP2/WP3 deliver object-level event
   streams. No third rescore of the pixel-pair unit under any circumstances
   (§4.41). Preregister against the same three gates so the comparison holds.

## 10. Standing constraints

- `docs/learnings.md` §8 (do-not-repeat) is binding. In particular: no beam or
  depth increase because a run failed; no reward weight added to rescue a
  result; no bounded search failure reported as proof of unsolvability.
- Roadmap §20 adds an invariant this campaign paid for: **an intervention that
  narrows exploration must prove it does not starve the supply that later
  progress consumes.** §4.50 generalizes §4.7 — it is not "distance rewards
  fail" but "any consistent proximity preference fails"; the
  tie-break-not-reward care taken in E3 did not save it.
- Preference wiring is **twice-measured as insufficient** (§4.43 restore
  scalar, §4.45 hypothesis reserve). Roadmap §19: WP8 needs schedule
  authority, not scoring authority.
- The strict/assisted boundary is enforced by `strict_lineage.py` and must be
  respected in prose too: **every v3xx run is assisted-lineage.** Gate 3 is
  closed on the assisted track only; strict-track Gate 3 needs a WP5-clean
  re-measurement.
- Mixed outcome = FAIL. No weight tuning, no budget re-sizing, no rerun on an
  identical negative result. A crashed arm is VOID, not FAIL.

## 11. If this session dies mid-stream

1. `git -C /Users/toddsherman/Projects/lolo status` — expect the five
   uncommitted files of the E6 build listed in §6, plus the doc-sync lane's
   three modified and two untracked files, plus untracked `tmp/`. Do not
   commit `tmp/`.
2. `.venv/bin/python -m unittest discover -s tests` — expect a clean OK with
   4 skipped (1,085 tests as of this edit; the count grows as lanes land, so
   trust the OK, not the number). A failure means the E6 build is mid-edit;
   find its owner (§8) before touching `neural_planner.py`. Run step 3 first:
   defer the suite while a native arm is in flight (§9 item 1).
3. `ps aux | grep neural_run` — check whether a native run is in flight or
   orphaned. If one is running, read its `events.jsonl` growth rather than
   trusting a process pattern (§4.52).
4. `ls experiments/lolo1-entity-v10/evaluations/ | grep -E "entity-v3[0-9][0-9]"`
   — the last completed run tells you where the E-series stopped.
5. Read `docs/learnings.md` from §4.42 to the end, then roadmap §17–§20, then
   `docs/wp8-search-scheduling-design-2026-08-17.md` §13–§15.
6. Resume at §9 item 1. Do not restart finished work; the acceptance list for
   anything incomplete is its own preregistration.

## 12. Document index for this campaign

- `docs/direction-review-2026-08-16.md` — adversarial direction panel; amendments A–E.
- `docs/offline-accessibility-diff-2026-08-16.md` — the cheap null that sharpened the probe (§4.26).
- `docs/paired-accessibility-probe-2026-08-16.md` — v322–v324 (§4.27, §4.28).
- `docs/object-removed-probe-2026-08-16.md` — v325/v326, Gate 3 closure (§4.29, §4.30).
- `docs/wp5-tracker-training-2026-08-16.md` — the six-gate WP5 chronicle (§4.31–§4.42).
- `docs/tracker-ood-eval-2026-08-16.md`, `docs/strict-collection-recon-2026-08-16.md` — WP5 corpus work.
- `docs/milestone-event-census-2026-08-16.md`, `docs/milestone-scoring{,-v2,-v3}-2026-08-16.md` — WP9 step 1 (§4.33, §4.36, §4.41).
- `docs/wp8-lite-ablation-design-2026-08-16.md` — v327/v328 (§4.43).
- `docs/wp8-relational-planner-design-2026-08-17.md` — v329–v331, E2's seeded root (§4.44, §4.45).
- `docs/wp8-search-scheduling-design-2026-08-17.md` — v332–v337, E3/E5, and E6's §15 when it lands (§4.46–§4.51).
- `docs/session-handoff-2026-08-16.md` — the predecessor; WP0/WP1 build history and session-artifact paths.
