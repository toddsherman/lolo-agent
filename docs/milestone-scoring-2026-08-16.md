# WP9a milestone-discovery scoring run — 2026-08-16

Status: preregistration written and frozen BEFORE the scoring pass; results and
verdict appended after the single corpus run. Direction-review Amendment D step
3 (`docs/direction-review-2026-08-16.md` §3.D), roadmap §17 item 7. Census
precondition: `docs/milestone-event-census-2026-08-16.md`.

Assisted-footprint caveat (inherited, load-bearing): both label-bearing corpora
are assisted-track. Every number below is an ENGINEERING artifact of the
offline spike. Nothing here is strict-track milestone-discovery evidence until
WP5 supplies a strict-legitimate controllable footprint.

Anonymous-terminology note: "heart-type collection", "life loss", and
"coarse-scene transition" are the census's evaluator-side labels for committed
telemetry field changes (`human_prior_collected_hearts` increases,
`human_prior_life_loss_confirmed`, `scene_signature` changes). They never
enter the scorer; the scorer sees only pooled pixel arrays, matched controls,
and measured outcome categories (reversion, persistence, novelty).

## 1. Preregistration (fixed before any corpus scoring)

### 1.1 Corpora in and out

- Corpus A (assisted Room 3, `experiments/lolo1-entity-v10/evaluations/`,
  345 run directories) — IN, with instance-level recall gates.
- Corpus B (assisted Room 2, `experiments/lolo1-medium/extended_evaluations/`
  `*human-prior*`, 19 runs) — IN, with instance-level recall gates.
- Corpus C (strict `lolo1-medium` extended evaluations, the 104 non-human-prior
  runs) — EXCLUDED from all precision/recall thresholds. The census (§5) found
  exactly one machine-checkable positive instance and zero structured
  negatives; no threshold is meaningful there. Corpus C is retained ONLY for
  Amendment D's two label-free falsification bits (timer/animation domination;
  separability structure of the ranking), plus one preregistered anchor
  report: the committed Floor 1 clear transition of
  `cycle-000010-floor1-resume-d879-finite-causal-bfs-1000` (committed
  decisions 506–508 per the evaluator annotation) is looked up in the ranking
  and reported, not gated.

### 1.2 Thresholds (from census §6, transcribed verbatim and frozen)

1. Positive recall gate, pooled over corpora A+B: at least 40 of the 50
   committed heart-type collection instances (13 in A, 37 in B) must map to a
   signature scored positive-valence with non-zero m(sigma). Justification:
   the census confirmed 50 structured instances — enough for an instance-level
   0.80 gate — while the ~dozen distinct transition classes make any
   class-level threshold coarser than 0.08, so class-level recovery is
   reported alongside but NOT gated.
2. Negative recall gate, corpus B only (corpus A has zero losses): at least 10
   of the 14 confirmed life-loss instances must map to a signature classified
   negative-valence (reversion/control-collapse basis). Justification: n=14 is
   thin; 10/14 tolerates individual censoring without letting the valence
   mechanism fail silently.
3. Precision over the top-10 signatures per corpus (fraction whose signature
   matches any verified candidate event instance: collection, life-loss, or
   committed coarse-scene transition) is REPORTED, NOT gated — the census
   cannot bound the denominator of true discoverable events.
4. `MilestoneScoreConfig` stays at its preregistered defaults
   (novelty baseline 0.0, negative reversion threshold 0.5, positive
   persistence threshold 0.5, positive novelty threshold 0.25). The census
   found no evidence forcing a different prior, and tuning on the census
   corpora before the run would spend the ground truth twice.

### 1.3 Falsification bits (from Amendment D, operationalized before the run)

- Timer/animation domination (per corpus): the corpus is
  animation-dominated when, among the top-10 ranked signatures, at least half
  of those with non-zero score are members of the corpus's neutral-drift
  signature set (defined §1.4), OR fewer than 10 signatures score non-zero at
  all (a degenerate ranking in which censoring/autonomy zeroed the corpus).
- Heart inseparability: failure of the positive recall gate (§1.2.1).
- Either bit firing on a qualifying corpus FALSIFIES WP9 step 1 as written.
- Life-loss valence (§1.2.2) is a gate of this scoring run and is reported
  with the same prominence; per the direction review it is not by itself one
  of the two named falsification bits.

### 1.4 Pair construction (fixed rules; deterministic; telemetry-only)

All pairs are root-relative matched factual/NOOP endpoint pairs reduced from
stored `events.jsonl` streams only. The pooled array is the existing 8x8
`visual_signature` (64 cells, mean intensity // 16, values 0–15) decoded from
its hex form; no frames are decoded and no emulator is touched.

- Strict-planner branch pairs (all corpora): root = the decision's
  `decision_started` array; factual = each `branch_verified` endpoint with a
  non-NOOP action (the planner verifies exactly the first action at its first
  duration); control = the NOOP endpoint from the same decision root at the
  identical duration (`branch_verified` with action NOOP, else
  `matched_neutral_verified`), exact duration equality only.
- Assisted option-search pairs (corpora A/B): root = the search root
  (`source_state_id` resolved to its array via `state_saved` frame joins);
  factual = each `human_prior_option_branch_verified` endpoint whose path is
  not all-NOOP; control = `human_prior_option_neutral_verified` from the same
  root with `elapsed_frames` equal to the branch's total duration
  (single-step branches may fall back to the duration-matched
  `human_prior_option_local_neutral_verified` from the same state). Missing
  or partially reproducing controls are dependence-censored, never assumed
  dependent.
- Committed transitions: a committed decision whose exact transition (same
  root array source, same endpoint frame digest) is already represented by a
  verified branch pair contributes successors to that pair rather than a
  duplicate pair; otherwise one committed pair is added (control matched as
  above where resolvable, else censored). Commits with
  `restored_archive=true` are excluded as pairs — a restore is not a
  controller action — and NOOP commits are excluded as factual pairs (NOOP is
  the control arm by definition).
- Successors: for every factual pair whose endpoint frame digest and root
  frame digest equal the committed transition of its own decision, successors
  are the next committed `visual_signature` arrays in the same attempt,
  truncated at the first `restored_archive` commit (a restore jump is not the
  factual arm's future) and capped at a window of W=8 committed observations.
  All other pairs are return-censored, which the scorer treats as
  non-evidence.
- Event signatures, scoring, and valence come exclusively from the
  preregistered pure functions in `lolo_agent/milestone_discovery.py`;
  provenance rows carry `source="telemetry"`.

### 1.5 Seen-signature pool (fixed before the run, with justification)

The pool for novelty/collapse is the PRE-INTERVENTION pool per corpus: content
signatures of arrays observed before each run's first scored decision —
`env_reset` arrays, bootstrap-phase `env_step` arrays, and every attempt's
first `decision_started` root array — unioned over the corpus's runs.

Justification (this is a deviation from the module's fixture helper
`seen_pool_from_pairs`, whose own docstring scopes it to fixtures): with
chained committed pairs, every successor of decision d is the root of decision
d+1, so a roots+controls pool contains essentially every successor and zeroes
successor novelty STRUCTURALLY — positive valence would be impossible by
construction and the recall gate would fail vacuously. Post-event controls
inherit the event's outcome the same way. The pre-intervention pool is the
only single-pool choice that is causally prior to every scored event; its
known cost — collapse-to-seen can only fire on exact returns to pre-era
configurations — is recorded here BEFORE the run and is treated as an
instrument limitation in interpretation, never as a post-hoc threshold
adjustment.

### 1.6 Neutral-drift signature set (for the domination bit)

For every resolved control, the control arm is itself differenced against the
root with the same event-signature function (`extract_event` on a root->control
pair). The set of resulting signatures is the corpus's neutral-drift set: the
changes that equal-duration neutral timelines produce spontaneously. A
top-ranked signature that is also a neutral-drift member is a timer/animation
class signature by measurement, with no game semantics involved.

### 1.7 Instance mapping (recall bookkeeping)

Ground-truth instances are recomputed from `decision_committed` events
(collection = increase of the committed collected-count field over the last
non-null committed value within the attempt; loss = confirmed-loss flag true;
scene transition = committed `scene_signature` change), cross-checked against
the census counts (A: 13 collections; B: 37 collections, 14 losses) and any
discrepancy reported. Each instance maps to the event signature of its exact
committed transition pair; instances with no pair (restore commit, unresolved
root, or no change at the pooled granularity) FAIL their gate and are reported
with the reason — censoring never supports a claim.

### 1.8 Run discipline

- The scorer runs ONCE per corpus over the corpora above, after this section
  was written. A single-run plumbing smoke (pair-join resolution on
  `entity-v325-room3-object-removed-probe-d12`, whose collection at committed
  decision 4 is documented in the census) validated event-stream joins before
  this preregistration was frozen; no corpus scoring preceded it.
- Output: deterministic, content-digested JSON at
  `experiments/lolo1-wp5/milestone-scoring-report.json`; findings and verdict
  appended below.
- Runner: `lolo_agent/milestone_discovery_run.py` (stdlib-only; reuses the
  module's pure functions; append-only unit tests in
  `tests/test_milestone_discovery.py`).

## 2. Results

Single pass executed 2026-08-16 by `lolo_agent/milestone_discovery_run.py`
(100 s wall, 1.26 GB peak, M5, telemetry-only). Report:
`experiments/lolo1-wp5/milestone-scoring-report.json`, content digest
`424bb775fa1e26047c699d8cffe75ed8b52052c6ed08ae101033fd8344340d31`.

### 2.1 Corpus volumes and plumbing cross-checks

| Quantity | A | B | C |
| --- | ---: | ---: | ---: |
| Runs scanned | 345 | 19 | 104 |
| Matched pairs scored | 914,988 | 52,497 | 360,518 |
| — strict branch pairs | 8,112 | 52,497 | 360,518 |
| — assisted option pairs | 906,876 | 0 | 0 |
| Controls resolved / unresolved | 914,988 / 0 | 52,497 / 0 | 327,612 / 32,906 |
| Pairs dropped (unresolved root) | 63,948 | 0 | 0 |
| Events extracted (nonzero diffs) | 862,324 | 29,979 | 133,266 |
| Distinct event signatures | 7,159 | 2,800 | 6,739 |
| Signatures with non-zero score | 29 | 80 | 265 |
| Committed decisions | 1,495 | 9,445 | 66,552 |
| Committed coarse-scene transitions | 504 | 4,726 | 22,773 |

Cross-checks against the census: coarse-scene transition counts match
EXACTLY in all three corpora (504 / 4,726 / 22,773) and corpus B life losses
match exactly (14), validating the committed-chain plumbing. Collection
instances recovered: A 11 (census 13), B 36 (census 37). The gap mechanism
was identified directly: the census scanned `decisions.csv`, which
materializes the collected-count on restored rows, while restored
`decision_committed` events report a null count; a restore-reset
re-collection whose pre-restore value equals the re-collected value (the
v323/v324 paired-probe arms: committed values 1, null-restore, 1) is
invisible to the events-only carry-forward preregistered in §1.7. Committed
decisions: A 1,495 vs census 1,491 (events-vs-CSV row accounting). All
discrepancies are on the census side of correlated replays; with 7 of 47
recovered instances passing the gate (§2.3), no accounting choice changes
any verdict. Corpus A's 63,948 dropped option pairs are searches rooted at
episodically imported archive states whose root frame never appears with a
pooled signature in the child run's stream (6.6%, counted, censored).

### 2.2 Timer/animation domination — NOT observed

- Top-10 neutral-drift fractions: A 0.2, B 0.0, C 0.0 — all below the 0.5
  domination criterion, and every corpus has ≥ 10 non-zero-score signatures
  (29 / 80 / 265), so the degenerate clause did not fire either.
- The top ranks of all three corpora are rare, action-dependent, persistent,
  novel committed transitions (occurrences 1–5 at the very top). Two of
  corpus C's top-10 are census-class coarse-scene-transition candidates.
- Mechanism: exact equal-duration matched-NOOP differencing is phase-aligned
  by construction (deterministic emulator, same frame count from the same
  root), so pure animation cancels in the dependence check instead of
  flooding the ranking. The v312/v313 raw-signature failure mode
  (~55% animation) did not recur at the 8x8, quantized pooling.
- Top-10 candidate-event precision (reported, not gated): A 0/10, B 1/10,
  C 2/10.

### 2.3 Heart-type collection recall — FAILED (7/47 = 0.149 vs gate 0.80)

7 of 47 recovered collection instances land in a positive-valence signature
with non-zero score (all 7 in corpus B; ranks 7, 14, 48 — when the event is
clean, the preregistered score DOES lift collections into the top-10).
Class-level (distinct-signature) recovery, reported not gated: A 0 of 8
classes, B 3 of 18. The 40 failures decompose into three measured
mechanisms, each verified on a named instance:

1. Dependence-censoring of mixed changed-cell sets (dominant in A, common in
   B). Example: `entity-v141-room3-alternative-order-d16` d7 — a 9-cell
   committed event whose matched control reproduced part of the diff
   (concurrent animation cells inside the event's own changed set), so every
   occurrence is dependence-censored and the multiplicative score is zero.
   The module's event-level censoring is working as designed — and the
   design zeroes real collections whenever any autonomous cell lands in the
   same 8x8 diff.
2. Return-censoring by restore-heavy assisted timelines. Example:
   `entity-v325-room3-object-removed-probe-d12` d4 (the Gate 3 lineage
   collection): a clean 2-cell, action-dependent event whose successor
   window is empty because the very next commit is an archive restore. Both
   census-starred v325/v326 collections fail this way. The assisted
   planner's restore-after-progress habit censors exactly the events the
   spike wants to rank.
3. Rarity non-separation in corpus A: with novelty near-saturated against
   the pre-intervention pool (§1.5), separation rests on rarity ×
   dependence, and at 8x8 granularity collection diffs are not rarer than
   the long tail of one-off movement/manipulation transitions that top the
   A ranking (precision 0/10).

### 2.4 Life-loss negative valence — FAILED (0/14 vs gate 10/14)

All 14 losses classify positive (novel-and-persistent), e.g.
`…human-prior-v14-life-hazard-credit-1000` d41: a 60-of-64-cell change that
the equal-duration NOOP control REPRODUCES (`action_dependent=False` — the
hazard was already inevitable at the root, so the control arm dies too),
followed by persistent, pool-novel successors. Two independent refutations
of the valence hypothesis as written:

- Death-reset is not a reversion: the post-loss room reset (progress fields
  reset, positions reset) is a LARGE persistent change relative to the
  event's own root, the opposite of changed-cells returning to pre-event
  values.
- Collapse-to-seen cannot fire: corpus B's pre-intervention pool has 3
  signatures (resumed mid-floor lineages have no reset/bootstrap frames),
  and post-loss states never exactly equal them. This is the §1.5
  limitation recorded before the run, now confirmed as load-bearing.

### 2.5 Corpus C falsification bits and anchor

- Not animation-dominated (drift fraction 0.0; 265 non-zero signatures).
- Preregistered anchor (the machine-checkable Floor 1 clear,
  `cycle-000010-floor1-resume-d879-finite-causal-bfs-1000` d506–d508):
  d506/d507 produce no committed-pair diff at the pooled granularity; d508
  carries the transition signature — positive valence, but score 0 at rank
  4,649 because the control arm reproduces the 60-cell change
  (`action_dependent=False`: the clear transition completes under NOOP as
  well once triggered). The one strict positive anchor is therefore
  action-independent AT THE COMMITTED DECISION where it is visible — the
  triggering action lies earlier in the chain, outside a single
  matched-endpoint window.

## 3. Verdict

**WP9 step 1 as written is FALSIFIED — by heart-inseparability, not by
timer/animation domination.** Preregistered bits: timer/animation domination
false on all three corpora; heart inseparability true (0.149 << 0.80). The
negative-valence gate also failed outright (0/14).

What survives: matched-NOOP endpoint differencing with exact duration
matching is a sound animation rejector at zero emulator cost, and the
rarity × dependence × persistence × novelty product does surface clean
collection events (corpus B ranks 7/14/48). What is refuted, with measured
mechanisms: (a) event-level dependence censoring — one concurrent autonomous
cell inside a diff zeroes a real milestone; cell-level dependence is needed;
(b) reversion/control-collapse as the negative-valence signal — death-reset
dynamics present as large persistent novel changes and are largely
action-INDEPENDENT at the fatal commit; (c) single-decision endpoint windows
— restore-heavy search timelines return-censor exactly the interesting
events, and delayed transitions (Floor 1 clear) complete autonomously after
the triggering action. Learning this now cost one offline day; learning it
at WP12 would have cost the program — which is what Amendment D bought.

Consequences (for the plan of record, not enacted here): WP9 step 1 needs a
revision before any native test — per-cell action-dependence instead of
event-level censoring, a negative-valence signal that recognizes
reset-to-known-configuration structurally rather than via exact-array
collapse, and successor windows that survive archive restores (or
branch-level follow-up observations). Gate 4 is unaffected (WP8 never
depended on WP9a). The assisted path of the amended Gate 4 remains the
operative one.

## 4. Provenance

- Runner: `lolo_agent/milestone_discovery_run.py` (stdlib-only; pure
  reduction helpers unit-tested in `tests/test_milestone_discovery.py`,
  append-only additions; synthetic fixtures only).
- Scorer: `lolo_agent/milestone_discovery.py` pure functions, config at
  preregistered defaults; provenance rows carry `source="telemetry"`.
- Inputs: stored `events.jsonl` streams of the three census corpora; no
  emulator, no frames, no `decisions.csv` in the scoring path.
- Report: `experiments/lolo1-wp5/milestone-scoring-report.json`
  (deterministic canonical JSON; the `content_digest` field is the SHA-256
  of the payload with that field absent).
- Post-run diagnostics quoted in §2.3–§2.5 re-read per-event fields
  (`action_dependent`, `reverted`, successor counts) of named instances from
  the same deterministic reduction at identical config; no re-scoring and no
  threshold was revisited after results were first observed.
- Context: `docs/direction-review-2026-08-16.md` §3.D, `docs/roadmap.md`
  §17.7, `docs/milestone-event-census-2026-08-16.md`,
  `docs/learnings.md` §4.26–§4.32.
