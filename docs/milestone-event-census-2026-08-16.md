# Milestone-event census for the WP9a offline spike — 2026-08-16

Method: read-only, evaluator-only census over stored telemetry; zero emulator
cost. This is the preregistered precondition of direction-review Amendment D
(`docs/direction-review-2026-08-16.md` §3.D, adopted in `docs/roadmap.md`
§17 item 7): fix precision/recall thresholds for the milestone-discovery
scoring run only after a census confirms enough positive events. The scorer
(`lolo_agent/milestone_discovery.py`) was NOT run against any real corpus for
this document; the numbers below come exclusively from counting fields that
already exist in stored run artifacts.

Assisted-footprint caveat: both corpora with recoverable goal-semantic event
fields are assisted-track. Nothing here is strict-track evidence; the strict
corpus is included precisely to document how little of it carries structured
event ground truth.

## 1. What was counted, and from what

- Candidate positive events
  - Heart-type collections: increases of the committed
    `human_prior_collected_hearts` field between consecutive committed
    decisions within one attempt (from `decisions.csv` where the column
    exists, otherwise from `decision_committed` events in `events.jsonl`).
  - Room/scene transitions: changes of the committed `scene_signature`
    between consecutive committed decisions, plus evaluator annotations in
    `evaluator_annotations.jsonl` where present.
- Candidate negative events: committed decisions with
  `human_prior_life_loss_confirmed` true (same sources), cross-checked
  against `human_prior_life_losses` in `summary.json`.
- Scoring-run input volume: matched factual/NOOP endpoint verifications from
  `summary.json` (`matched_neutral_verifications`,
  `human_prior_option_local_neutral_verifications`,
  `human_prior_option_neutral_verifications`) and verified branch counts.

Known limitations, stated up front:

- Counts are committed-timeline only. Branch-level (non-committed) events are
  not counted here; the scoring run will operate on branch-level matched
  endpoint pairs, which are far more numerous.
- The collected-count field is timeline-relative: archive/checkpoint restores
  reset it, so the same physical heart slot can be re-collected and recounted
  across a run's committed timeline. Increases and decreases are reported
  separately; re-collections dominate in heavily replayed corpora.
- Committed `scene_signature` changes are coarse visual-scene transitions,
  not room transitions. In title/story-era strict runs they are dominated by
  animation; in single-room runs they include palette and animation shifts.
  They are an upper bound on room-transition-class positives.
- The strict corpus records no goal-semantic fields by design; its heart and
  life events exist only in documentation plus stored frames, and would need
  a separate evaluator-only frame pass to become machine-checkable ground
  truth.

## 2. Corpus A — assisted Room 3 (`experiments/lolo1-entity-v10/evaluations/`)

345 run directories; 343 scanned via `decisions.csv`, 2 via `events.jsonl`
fallback (`entity-v284-…` has no committed decisions in its event stream;
`entity-v326-room3-object-removed-repetition-d12` has an event stream but no
`decisions.csv`/`summary.json` — partial artifacts).

| Quantity | Count |
| --- | ---: |
| Committed decisions | 1,491 |
| Committed heart-collection events (field increases) | 13 |
| Runs containing at least one collection | 12 |
| Committed collected-count decreases (restore resets) | 13 |
| Confirmed life losses (committed) | 0 |
| Committed coarse-scene transitions | 504 |
| Matched local-neutral verifications (branch level) | 153,754 |
| Matched full-neutral verifications (branch level) | 11,064 |
| Verified option branches | 950,135 |

Notable recoveries, consistent with the run documentation:

- `entity-v325-room3-object-removed-probe-d12`: collection at committed
  decision 4 (the (12,11)-class heart behind the removed entity —
  `docs/learnings.md` §4.30).
- `entity-v326-room3-object-removed-repetition-d12`: collection at committed
  decision 4 recovered from the event stream despite missing summary
  artifacts (the Gate 3 repetition lineage).
- `entity-v323`/`entity-v324` paired-probe arms: collection at decision 3 in
  both, matching the preregistered probe notes.
- Zero committed life losses across the whole corpus: Room 3 provides no
  negative-event ground truth at all.

The 13 collection events cover only a handful of distinct heart-slot
transitions; most instances are matched replays re-collecting the same slot
(e.g., the `[(128,64),(144,192)] -> [(144,192)]` chain of
`docs/room3-milestone-credit-correction-2026-08-13.md`).

## 3. Corpus B — assisted Room 2 (`experiments/lolo1-medium/extended_evaluations/`, `*human-prior*` runs)

19 runs; 5 scanned via `decisions.csv` (v15–v19), 14 via `decision_committed`
events (earlier schema without the committed CSV columns).

| Quantity | Count |
| --- | ---: |
| Committed decisions | 9,445 |
| Committed heart-collection events (field increases) | 37 |
| Runs containing at least one collection | 15 |
| Committed collected-count decreases (restore resets) | 41 |
| Confirmed life losses (committed) | 14 (13 also aggregated in summaries) |
| Runs containing at least one life loss | 6 |
| Committed coarse-scene transitions | 4,726 |
| Matched-neutral verifications (branch level) | 17,393 |
| Verified branches | 59,060 |

Notable recoveries, consistent with the annotations:

- `…hearts-v1-5000`: first collection at decision 38, matching the
  evaluator annotation ("first heart collected at decision 38").
- `…v15-life-hazard-rollback-1000`: 3 confirmed losses at decisions 41, 184,
  197 — exactly the corrected annotation ("Total confirmed losses: 3").
- `…hearts-v8-…detourgrace2-5000`: 4 collection events including decision
  459 (the final-heart upper-slot collection cited in the v15 annotation).
- The committed collected-count high-water is 1 in every run that contains a
  collection: restores kept resetting committed progress, which is why
  increases (37) and decreases (41) nearly match. Distinct heart slots/orderings underlying the 37
  instances number roughly half a dozen.

## 4. Corpus C — strict `lolo1-medium` extended evaluations (non-human-prior)

104 runs; 95 scanned via `decisions.csv`, 9 via event-stream fallback (these
9, including `cycle-000010-floor1-explicit-pose-3000` and
`cycle-000010-first-heart-search-1000`, retain only
`events.jsonl`/`frames`/`manifest.json`).

| Quantity | Count |
| --- | ---: |
| Committed decisions | 66,552 |
| Structured heart-collection events | 0 (no goal-semantic fields exist) |
| Structured life-loss events | 0 (same reason) |
| Committed coarse-scene transitions | 22,773 |
| Evaluator annotations | 3 (across 2 runs) |
| Matched-neutral verifications (branch level) | 97,587 |
| Verified branches | 383,323 |

The only machine-checkable positive-event ground truth in this corpus is the
annotated Floor 1 clear boundary in
`cycle-000010-floor1-resume-d879-finite-causal-bfs-1000`
(`stage-1-floor-1` -> `stage-1-floor-2` at seq 28,891; committed transition
at decisions 506–508).

Documented positive/negative events that are NOT structurally recoverable
(they live in `docs/medium-experiment-2026-08-08.md` prose plus stored
frames, and would need an evaluator-only frame pass to become ground truth):

- strict first heart at committed decision 374
  (`cycle-000010-first-heart-causal-dedup-500`; 575 stored frames);
- right heart ~d135, persistent transformation ~d777, upper heart d879
  (`cycle-000010-floor1-explicit-pose-3000`; event stream + frames only);
- Floor 2 entries implied by the resume roots of the `cycle-000012-floor2-*`
  fresh-frozen lineage (d849 dark transition, d2893);
- one life loss (5 -> 4) caused by `SELECT@1` in the first-room bootstrap
  audit lineage.

That is at most ~4 positive heart-type instances, ~3 room-transition
instances, and ~1 negative instance for the entire strict corpus, of which
exactly one positive (the annotated clear) is machine-checkable today.

## 5. Verdict: are precision/recall thresholds meaningful?

Per corpus:

- Assisted corpora (A + B combined): YES for instance-level thresholds.
  Ground truth recoverable from structured telemetry: 50 committed
  heart-collection instances (13 + 37) and 14 confirmed life-loss instances,
  against ~10.9k committed decisions and >180k branch-level matched-neutral
  verifications as scorer input. Caveats: the positives collapse to roughly
  a dozen distinct transition classes (replays are correlated evidence), and
  ALL negative instances come from Room 2 — Room 3 contributes none.
- Strict corpus (C): NO. One machine-checkable positive instance and zero
  structured negatives cannot support any precision/recall threshold.
  The strict corpus can still serve Amendment D's falsification bits, which
  need no labels: whether the score ranking over its 97,587 matched-neutral
  endpoint pairs is dominated by timer/animation signatures, and whether
  heart-class events are separable from them at all.

## 6. Threshold recommendation for the scoring-run preregistration

1. Run the scorer on corpora A and B with instance-level gates:
   - positive recall ≥ 0.80: at least 40 of the 50 committed collection
     instances must fall in a signature scored positive-valence with
     non-zero m(sigma);
   - negative recall ≥ 10/14: life-loss instances classified
     negative-valence (reversion-to-seen);
   - precision reported over the top-10 signatures per corpus (fraction
     matching a verified candidate event), reported but NOT gated at this
     spike stage — the census cannot bound the denominator of true
     discoverable events.
   Report class-level (distinct-transition) recovery alongside, without
   gating on it: with ~a dozen classes the granularity (~0.08) is too coarse
   for a fine threshold.
2. On corpus C, preregister only the two falsification bits of Amendment D
   (timer/animation domination; heart-inseparability) over the
   matched-neutral endpoint pairs. Do not preregister precision/recall
   there. If strict recall is later wanted, first run a separate
   evaluator-only frame pass to convert the ~4 documented strict instances
   into machine-checkable labels — and even then n≤4 supports a sanity
   check, not a threshold.
3. Keep `MilestoneScoreConfig` defaults (novelty baseline 0.0, reversion
   threshold 0.5, persistence threshold 0.5, novelty threshold 0.25) frozen
   until that preregistration; the census found no evidence forcing a
   different prior, and tuning them on the census corpora before the scoring
   run would spend the ground truth twice.

## 7. Provenance

- Census script: stdlib-only, read-only scan of `decisions.csv`,
  `events.jsonl` (committed events only), `summary.json`, and
  `evaluator_annotations.jsonl`; run 2026-08-16 in a session scratchpad
  (deliberately not committed — this document records the method and every
  number needed to re-derive it).
- Artifact gaps found in passing: `entity-v284-…-d18` (no committed
  decisions in its event stream), `entity-v326-…-repetition-d12` (event
  stream only, no summary — collection at decision 4 recovered), and 9
  strict runs retaining only event streams and frames.
- Scorer skeleton under test-only fixtures: `lolo_agent/milestone_discovery.py`,
  `tests/test_milestone_discovery.py` (synthetic arrays; no telemetry).
- Context: `docs/direction-review-2026-08-16.md` §3.D, `docs/roadmap.md` §17,
  `docs/learnings.md` §4.26–§4.30, `docs/medium-experiment-2026-08-08.md`,
  `docs/room3-milestone-credit-correction-2026-08-13.md`.
