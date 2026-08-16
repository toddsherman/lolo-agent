# Research-direction review — 2026-08-16

Status: adversarially reviewed recommendations; amendments pending Todd's
ratification
Method: multi-agent panel — 3 grounding readers (docs corpus, planner core,
learned-model stack), 5 independent direction proposers with distinct lenses
(learning-first, objective-design, search-and-oracle, science-and-claims,
pragmatic-delivery), 1 adversarial critic per proposal checking against
recorded evidence, strict-track constraints, compute limits, and overlap with
the existing roadmap. All five verdicts: **adopt_modified** — no proposal
survived intact, and none justified inverting the plan of record.
Companion docs: `docs/roadmap.md`, `docs/learnings.md`,
`docs/session-handoff-2026-08-16.md`

## 1. Headline conclusion

**The plan of record survives adversarial review.** Every proposed
"redirect" turned out to be largely a restatement, resequencing, or
mechanization of work the roadmap already contains — the critics repeatedly
cited the exact WP sections. The representation-first ordering (multi-track →
transitions → accessibility → hypothesis planning) is supported, not
undermined, by the recorded evidence. What the panel produced instead is a
set of **sharpenings**: one high-value measurement to run immediately, one
work package to mechanize in parallel, two cheap hygiene tools, one early
falsification spike, and several descope corrections.

## 2. Verified findings the record should absorb

The critics checked these directly against code and artifacts (line numbers
should be re-verified at implementation time; they will drift as WP1 lands):

1. **The assisted player mask is load-bearing inside tracked-state
   signatures** used by the object-track machinery
   (`neural_planner.py` ~:2392–2435 `player_pixel_mask` inside
   appearance-state signatures; ~:5829–5833 `detect_player` in transition
   analysis). This is sanctioned for the assisted development track (roadmap
   §2.2), and in strict runs `goal_prior` is `None` — but it means the
   v318–v321 object-state lineage is assisted-provenance, and WP5's
   acceptance gate ("strict object tracking no longer imports
   `PixelHeartGoalPrior` player masks") is the bar the strict claim depends
   on. Nothing currently *enforces* that boundary mechanically.
2. **The strict track has cleared a room.** `docs/medium-experiment-2026-08-08.md`
   records a strict first heart (decision 374) and a strict Floor 1
   completion (`cycle-000010-floor1-resume-d879-finite-causal-bfs-1000`)
   with evaluator-only accounting. Any narrative that "strict has never
   cleared a room" is false; the accurate debt statement is: *no strict
   milestone mechanism has been validated, and no strict corpora exist from
   the object-centric era or Rooms 2–3.*
3. **`experiments/lolo1-medium/dataset` is strict-bound**
   (`reward-track.json`: `{"reward_track":"strict"}`) — 16 segments,
   ~155k sequences, ~16.9k counterfactual-paired causal groups, ~393 MB.
   A large strict training corpus already exists; the confirmed Room 3 push
   evidence, by contrast, lives in assisted entity runs and is permanently
   barred from strict corpora.
4. **v313 archived zero save states** (its `states/` directory is empty) —
   the 1,756 changed-layout branches are not restorable and that candidate
   set is the falsified raw-changed-layout detector regime; no future plan
   may "sweep" it.
5. **Probe budgets are trivial on the M5** (verified 586 branches/s): a
   25k-branch bounded reachability probe is ~43 seconds of emulator time.
   The panel's highest-value recommendation costs essentially nothing.

## 3. Adopted amendments (recommended; pending ratification)

Ordered by value-to-cost. None changes WP0/WP1, which are landing now.

### A. Run the accessibility measurement on the confirmed push NOW (WP6-early)

Two lenses (search-and-oracle, pragmatic-delivery) independently converged
on this; it directly tests the roadmap §3 thesis ("the missing piece is how
arrangements change future accessibility") on the project's only confirmed
manipulation. Moves exactly one dependency edge: the WP6 *native
measurement* runs before WP2, at Task G scope (no policy authority).

- Day 0 (free): offline telemetry diff of v320's 2,497-descendant player
  positions/frontiers against pre-push-era search telemetry, per §13's
  offline-first rule — bounds what the native probe can add.
- Week 1: standalone `accessibility.py` probe from the digest-verified
  pre-push root and pushed archive (v320/v321), bounded directional
  reachability run to closure, plus duration-matched NOOP control.
- **Mandatory corrections the critics imposed:**
  - Success requires a connectivity change **beyond the object's own
    source/destination footprint** (new connected region, new interaction
    frontier, or goal-region distance change per §6.5). The vacated-cell
    delta is trivially nonzero and proves nothing.
  - Apply the v57 matched-player-footprint discipline (the v56 lesson:
    reachable-set diffs from player pose, not world change, are a recorded
    false-positive mode).
  - Per-branch configuration-hold verification via player-masked non-player
    signatures; branches that disturb a second object are censored
    (approximates WP6's "second manipulation invalidates the probe").
  - Node termination/dedup/contact classification uses matched-NOOP,
    phase-tolerant comparison — never raw player-masked signature diff
    (v312/v313: ~55% of raw changed-layout signatures were animation noise).
  - An empty delta at budget is **censored scoped evidence** that
    downweights the Room 3 single-push vehicle and triggers vehicle
    re-evaluation. It falsifies nothing (learnings §2, §4.14).
- Provenance: probe outputs are assisted-lineage until WP5 replaces the
  player anchor; nothing enters `entity_behavior.py` or any strict corpus
  except through the provenance-checked importer.
- Restore-selection authority for measured deltas comes only later, via a
  preregistered paired fixed-budget ablation (mixed result = FAIL, per the
  spatial-v10 lesson), and only after Task B lands so the edit stays out of
  the monolith's high-conflict region. Success metric hardened to
  previously-unreachable cells/frontiers or milestones — not raw
  new-affordance counts, which configuration churn can inflate.

### B. Mechanize WP5 in parallel (not blocking): counterfactual distillation

Upgrade Task H from a vague spike into a concrete WP5 implementation running
alongside WP1–WP3 — exactly where the roadmap already schedules WP5, so no
resequencing:

- `lolo_agent/counterfactual_labels.py` over the strict-bound
  `lolo1-medium` store: controllable-region pseudo-labels from
  leave-one-action-out factual-vs-matched-NOOP counterfactual consistency;
  **no `goal_prior` anywhere in the label path**.
- Train the controllable-region head warm-started on the frozen spatial v10
  encoder (the only learned component to pass offline gates *and* show a
  native shadow improvement) at the adopted lr ~1e-5, run-held-out splits,
  on the M5 (overnight-scale; RunPod stays out per the 0.196× gate).
- Land the v316/v317 failure modes as permanent regression fixtures
  (adjacent same-appearance object outside the mask; rarity distractor;
  blocked action changes pose not position).
- **Promotion gate (novel, adopted):** the v320/v321 archive-restore
  substitution replay — recompute tracked-state signatures with the learned
  mask substituted at the mask choke points; the pushed-object track must
  survive restore at descendant retention comparable to 2,497/132. The
  assisted mask stays as a shadow comparator emitting divergence telemetry.
- WP1 scope addition: a provenance-tagged mask interface at the identified
  choke points, so the learned mask can swap in without re-plumbing.
- **Dropped from the original proposal:** the stage-2 learned
  correspondence head (data-starved toward the recorded 18-negative
  memorization failure, and its labels would launder assisted-mask
  provenance into a strict artifact); any re-targeting of Gate 1; any
  gating of WP2 on WP5.

### C. Enforce the strict/assisted boundary mechanically (1–2 days)

- **Strict-lineage linter:** fail any artifact whose derivation graph
  touched `goal_prior.detect_player`, `player_pixel_mask`, or heart
  templates. The leak surface is verified in code (finding §2.1); today the
  boundary is discipline, not tooling.
- **Preregistration addendum to `docs/protocol.md`** (days, not weeks):
  claim tiers — Tier 1 (powered headline): frozen cross-room
  concept-transfer battery (displacement-confirmation precision,
  behavior-model Brier/AUC vs unconditional baseline, track persistence
  across restore) reported at every gate on **development rooms only**;
  Tier 2: withheld-room solves (honest low-n flagship); Tier 3: frozen
  Lolo 2 as exploratory stress test with the WP12 failure taxonomy.
  Withheld rooms are probed at **two pre-declared checkpoints only**
  (post-WP8 gate, Gate 8) — a continuously-run withheld battery burns the
  test set through researcher-mediated leakage and is rejected (§4).
- **Hardened WP0 stopping rule:** no native run in any unpartitioned room.
- Optional cheap canary: a pixel-only PPO probe at matched budget on the
  M5 quantifies exploration hardness and doubles as a contamination
  detector for withheld rooms. Dreamer-class baselines and a procgen
  puzzle universe are **deferred** (§4).

### D. WP9a offline milestone-discovery spike (bounded, after WP1)

The strict-objective replacement (WP9 step 1) is the least-specified item on
the WP12 critical path and currently has zero scheduled tests before WP8.
Adopt a 1–2 week, telemetry-only offline spike, in WP1 review-latency gaps:

- Precondition: an evaluator-only event census over candidate corpora
  (assisted-era entity/spatial telemetry for Rooms 2–3; the strict
  `lolo1-medium` extended-evaluation replays containing the strict first
  heart and Floor 1 clear) — fix precision/recall thresholds only after the
  census confirms enough positive events, and before the scoring run.
- `lolo_agent/milestone_discovery.py` as a pure module: matched-NOOP
  endpoint differencing, existing pooled features, rarity ×
  action-dependence × censored-non-return × novelty-margin scoring,
  reversion/control-collapse valence. Labeled engineering-only until WP5
  supplies a strict-legitimate controllable footprint.
- Run once against preregistered evaluator-only fixtures. Timer/animation
  domination or heart-inseparability **falsifies WP9 step 1 as written** —
  learning that now costs weeks; learning it at WP12 costs the program.
- Gate 4 amended permissively: a strict-discovered milestone MAY satisfy
  "subsequent positive milestone"; the assisted path remains valid; WP8
  never depends on WP9a.

### E. Descope corrections to the existing sequence

- **WP7 (phase model) off Gate 4's critical path** — for the correct
  reason: Gate 4's success criteria never mention phase and no phase model
  exists yet. (The claim "no global phase transition has ever been
  observed" is false — the all-hearts/chest transition is recorded — so
  WP7 returns immediately if Room 3's payoff proves post-all-hearts.)
- **WP2 first cut at K≤4 tracks** with greedy correspondence +
  abstain-and-freeze on ambiguity; Gate 1 requires only two tracks. The
  32-track/min-cost/4-hypothesis machinery is the target contract, not the
  first implementation.
- **Gate 4 preference wiring** may first target the existing
  reserve/archive-frontier scoring seams (each component separately
  logged, preregistered matched-budget counterfactual), with
  `relational_planner.py` extraction as the declared fallback if monolith
  integration thrashes.
- WP1/WP2 proceed **regardless of the probe outcome** in amendment A —
  object tracking is needed for any room vehicle; a Room-3-neutral result
  changes the Gate 4 vehicle, not the necessity of representation work.

## 4. Explicitly rejected (with reasons)

- **Resequencing WP5 ahead of WP2 / re-targeting Gate 1 at learned-tracker
  outputs** — serializes the decisive gates behind a training success; the
  assisted-mask pipeline demonstrably works (2,497/2,497, 132/132); WP5
  runs in parallel per plan.
- **Continuous frozen battery on withheld rooms** — researcher-mediated
  leakage burns the test set; development-room battery + two pre-declared
  withheld checkpoints instead.
- **Immediate strict-provenance collection loop with a milestone/bridge
  objective** — requires WP6 artifacts that don't exist, grants reward
  authority to unvalidated terms against the telemetry-first promotion
  discipline, and re-runs the v313/v314 failure mode.
- **Stage-2 learned correspondence head distilled from engineered-gate
  firings** — data-starved toward the recorded 18-negative memorization
  failure; labels carry assisted-mask provenance into a strict artifact.
- **Sweeping v313's 1,756 changed-layout branches** as a fallback — states
  were never archived; the candidate set is the falsified detector regime.
- **Treating a budget-censored empty accessibility delta as falsifying
  Room 3** — violates the project's own censoring discipline (learnings §2,
  §4.14).
- **DreamerV3 baseline and procgen puzzle universe now** — integration cost
  and fairness methodology exceed solo bandwidth; revisit at the WP8 gate
  (Dreamer, with a preregistered budget-currency definition) and at
  write-up time (procgen; §11.2 mock environments already cover the test
  role). A misconfigured baseline is attacked harder than a missing one.
- **YouTube demonstrations / hard-coded mechanics** — not proposed by any
  lens; remain rejected per learnings §6.

## 5. Amended near-term sequence (reconciled with work in flight)

1. **WP0 + WP1** — already building (see
   `docs/session-handoff-2026-08-16.md` §3.2). Follow-up if not already in
   scope: the provenance-tagged mask interface (B) and the hardened
   stopping rule + preregistration addendum (C).
2. **Amendment A** — day-0 offline diff, then the corrected week-1
   accessibility probe from the v320/v321 archives. This is the next new
   *research learning*, at ~minutes of emulator time.
3. **WP2-lite** (K≤4, greedy+abstain) → Task C/D per roadmap.
4. **Parallel:** WP5 mechanized spike (B); strict-lineage linter (C);
   WP9a offline spike (D) in review-latency gaps.
5. **Gate 4 wiring** per amendment E, with the preregistered matched-budget
   counterfactual.

## 6. Provenance of this review

Panel run: 13 agents, ~1.32M tokens, 2026-08-16. Full proposals and
critiques preserved in the session transcript
(`~/.claude/projects/-Users-toddsherman-Projects-lolo/354d515c-2be6-4001-be73-90cf19acae16/subagents/workflows/wf_f106dc0b-fdd/journal.jsonl`);
this document is the durable synthesis. Line-number citations were verified
by critics against the working tree of commit 99f6f92 and must be
re-verified after the WP1 extraction lands.
