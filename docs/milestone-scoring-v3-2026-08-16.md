# WP9a milestone-discovery scoring run v3 (section-4.36 rethink) — 2026-08-16

Status: preregistration written and frozen BEFORE the v3 scoring pass; results
and verdict appended after the single corpus run. This is the rethink-and-
rescore step recorded in `docs/learnings.md` §4.36 after the v2 pass
(`docs/milestone-scoring-v2-2026-08-16.md`, report digest `898676b5…`) was
falsified a second time via heart-inseparability (15/47 = 0.319 against the
0.80 gate) while the delayed-divergence negative-valence gate passed 14/14.
The falsification frame here is IDENTICAL to v1 and v2: same corpora, same
gates, same two falsification bits. Only the two mechanisms §4.36 named are
redesigned; every numeric constant is inherited unchanged.

Stakes, fixed before scoring: v1 and v2 each failed the heart-separation gate
(0.149, then 0.319). Per the §4.36 plan of record, a THIRD heart-separation
failure demotes WP9 step 1 from "rethink" to a fundamental rethink of the
event representation itself; a pass on both main gates validates WP9 step 1
and unblocks the strict-objective path.

Assisted-footprint caveat (inherited, load-bearing): both label-bearing
corpora are assisted-track. Every number below is an ENGINEERING artifact of
the offline spike. Nothing here is strict-track milestone-discovery evidence
until WP5 supplies a strict-legitimate controllable footprint.

Anonymous-terminology note: as in v1/v2, "heart-type collection", "life
loss", and "coarse-scene transition" are evaluator-side labels for committed
telemetry field changes; they never enter the scorer. The scorer sees only
pooled pixel arrays, matched controls, observed-array histories, and measured
outcome categories.

## 1. Preregistration (fixed before any v3 corpus scoring)

### 1.1 Corpora in and out

Unchanged from v1/v2 (census `docs/milestone-event-census-2026-08-16.md`):

- Corpus A (assisted Room 3, `experiments/lolo1-entity-v10/evaluations/`,
  345 run directories) — IN, instance-level recall gates.
- Corpus B (assisted Room 2, `experiments/lolo1-medium/extended_evaluations/`
  `*human-prior*`, 19 runs) — IN, instance-level recall gates.
- Corpus C (strict extended evaluations, 104 non-human-prior runs) —
  EXCLUDED from precision/recall thresholds (census §5); retained for the two
  label-free falsification bits plus the same preregistered anchor report
  (the committed Floor 1 clear of
  `cycle-000010-floor1-resume-d879-finite-causal-bfs-1000`, decisions
  506–508), looked up and reported, not gated.

### 1.2 The two redesigned mechanisms (fixed by §4.36)

Everything not named here is carried over from v2 verbatim: per-component
censoring, lineage-filtered successor windows with branch-followup fallback,
escape-divergence lookback, pair construction, the seen pool, and the score
product m(sigma) = log_rarity x dependence rate x non-return factor x
novelty margin.

1. **Component-anchored rewind**
   (`lolo_agent/milestone_discovery.py::extract_component_event_v3`). The v2
   rewind test was window-scoped: any successor at least
   `rewind_transient_floor` cells from the event root while within
   `rewind_proximity_ceiling` cells of a pre-event history array marked the
   occurrence rewound, regardless of what the event's own cells did — so a
   terminal reset up to 8 commits AFTER a genuine milestone poisoned that
   milestone's occurrence (the §4.36 reset bleed-through mechanism, 28 of 32
   residual v2 failures). V3 keeps the v2 structural reset recognizer
   UNCHANGED (same two constants) and adds a component anchor: a successor
   marks the occurrence rewound only when, against the SAME history array
   that satisfied the proximity test, the event's own component cells revert
   toward the pre-event configuration — strictly more component cells hold a
   pre-event-consistent value (the cell's root value, or the matched history
   array's value at that cell) than hold their post-event value. Component
   cells at third values support neither side (censoring never supports a
   claim). The anchor introduces NO new numeric constant: it is a structural
   plurality over the event's own attributable cells, answering §4.36's
   question "did THIS change reset?".

   Precedence rule, fixed here: a component cell whose post-event value
   equals the matched history array's value at that cell counts as
   pre-event-consistent, not as retained. Rationale: such a change moved the
   cell ONTO a known pre-event configuration — reset-shaped by construction
   — and this keeps a terminal commit's own occurrence rewound whether its
   stored endpoint is the mid-transient array or the settled post-reset
   array, decided structurally rather than by inspecting instances. The
   symmetric residual risk is disclosed: an event whose post-event component
   values coincide with a pre-event reference remains classifiable as
   rewound; that is the intended semantics (the change itself is a
   reset-shaped move), not bleed-through.

2. **Occurrence-scoped valence, signature aggregation only for ranking**
   (`lolo_agent/milestone_discovery.py::occurrence_valence`,
   `::score_events_v3`). In v2, valence was assigned per signature over
   merged component classes, so a handful of rewound windows flipped whole
   collection classes negative (one 176-occurrence class decided by a single
   evaluable window) and no threshold could trade that against the 14/14
   negative gate. In v3 each occurrence carries its own valence from its own
   evidence, and nothing overwrites it:

   - RETURN-CENSORED (no successor observations): unresolved.
   - NEGATIVE (basis `delayed_divergence`), checked first: the occurrence is
     rewound under the component-anchored test AND participates in measured
     factual-vs-control divergence structure — its own matched contrast is
     dependence-evaluable, or an escape divergence was observed within the
     preregistered lookback before its commit. Identical evidence rule to
     v2, applied at occurrence scope; dependence-censored occurrences with
     no escape evidence can never be negative.
   - POSITIVE (basis `novel_and_persistent`): the occurrence's component
     never reverts within its window, its first successor does not collapse
     onto the seen pool, and its own successor novelty fraction is at least
     `positive_novelty_threshold`. This is the v1/v2 positive rule with the
     signature-level rates replaced by the occurrence's own binary
     persistence and own novelty fraction.
   - UNRESOLVED otherwise.

   The signature score m(sigma) is UNCHANGED and remains the only
   signature-level aggregate with decision power (ranking, and the
   non-zero-score requirement of the positive gate). The per-signature
   valence label in the v3 report is a REPORTING-ONLY plurality of
   occurrence valences (negative when negative occurrences strictly exceed
   positive, positive when the reverse, unresolved otherwise or when fully
   return-censored); no gate reads it and it overwrites no occurrence.
   `negative_divergence_threshold` remains in the config but no longer
   classifies valence (the occurrence rule is binary), exactly as v2
   retired the v1 reversion threshold without removing it.

### 1.3 Thresholds (fixed here, before scoring)

Shared with v1/v2, all unchanged, all census-derived; v3 adds NO new
constant:

- Positive recall gate 0.80 pooled over recovered A+B collection instances;
  negative recall gate 10/14 corpus-B life-loss instances; timer/animation
  domination bits per corpus (top-10 neutral-drift fraction >= 0.5, or fewer
  than 10 non-zero-score signatures); precision over top-10 reported, not
  gated.
- `MilestoneScoreConfig` at v1/v2 defaults: novelty baseline 0.0,
  persistence 0.5, novelty 0.25, reversion threshold 0.5 (inert since v2),
  `negative_divergence_threshold` 0.5 (inert from v3 on, per §1.2.2),
  `rewind_transient_floor` 16, `rewind_proximity_ceiling` 8,
  `escape_cell_minimum` 8, `divergence_lookback` 8; successor window W=8;
  branch-followup window 8; pre-intervention seen pool.

Gate reading under occurrence scope, restated and frozen before scoring
(same thresholds, occurrence semantics per §1.2.2):

- A collection instance counts toward the positive gate when ITS OWN
  committed occurrence classifies positive AND its signature's aggregated
  score is non-zero.
- A life-loss instance counts toward the negative gate when its own
  committed occurrence classifies negative. (Consequence, disclosed: an
  instance whose committed transition never entered the signature ranking —
  e.g., a NOOP-commit loss whose full-diff signature is shared by no scored
  arm — can now satisfy the negative gate on its own occurrence evidence,
  where v1/v2 could not; the positive gate still requires a ranked
  signature with non-zero score.)

### 1.4 Pair construction deltas

NONE. The v2 assembly (`assemble_run_pairs_v2`) is reused verbatim — same
pairs, windows, histories, and escape flags. The only additions are
bookkeeping: the assembler records, per committed transition, the exact pair
whose component signature v2 already used for instance mapping, and v3
extracts that pair's occurrence event (component-anchored rewind included)
so instance rows can carry occurrence valence. Committed-transition
signatures are unchanged from v2 (the component anchor changes only the
`rewound` flag, never the signature).

### 1.5 Design-time evidence disclosure

This preregistration was designed against ALREADY-PUBLISHED evidence only:
the v2 report's §2.4 per-signature diagnostics (signatures `a03abd13…`,
`1dde50a3…`, `55daa1a2…` and their rewound-window counts) and the v1/v2/
census-published rewind geometry of the v14/v15 loss runs. No run directory,
telemetry stream, instance, or scorer ranking was newly inspected before
this section was frozen; no corpus was scored with any v3 code before the
single pass of §1.6. The endpoint-shape question for terminal commits
(mid-transient vs settled endpoint arrays) was resolved by the §1.2.1
precedence rule structurally, precisely so that it did NOT require opening
any instance. The v3 unit fixtures are synthetic. No collection or loss
instance beyond those already spent by v1/v2 (disclosed there) is consumed
at design time; the v2 design-time spend of 4 of the 14 loss instances
still stands and is inherited, so the negative gate's evidential weight
continues to rest on the 10 uninspected instances.

### 1.6 Run discipline

- The v3 scorer runs ONCE per corpus, after this section was written, via
  `python3 -m lolo_agent.milestone_discovery_run --v3` under the project
  venv interpreter (pinned as in v2 provenance).
- Output: deterministic, content-digested JSON at
  `experiments/lolo1-wp5/milestone-scoring-v3-report.json`
  (schema `milestone-scoring-v3-report/1`); findings and verdict appended
  below. A determinism rerun must reproduce the report byte-identically.
  The v1 and v2 code paths are rerun to scratch locations to verify
  byte-identical reproduction of their published reports (digests
  `424bb775…`, `898676b5…`) after the additive v3 changes.
- Tests: append-only fixtures in `tests/test_milestone_discovery.py` for
  both mechanisms: a reset bleed-through occurrence that v2 marks rewound
  and classifies negative at class level while v3 must not (component cells
  retained across the window's later terminal reset), and a two-occurrence
  signature whose v2 class valence flips a positive occurrence negative
  while v3 keeps that occurrence's own positive valence; plus
  component-anchor edge cases, occurrence-valence rules, runner-level
  committed-occurrence bookkeeping, and determinism coverage.
- Interpretation, fixed in advance: if the heart-separation gate fails a
  THIRD time, WP9 step 1 demotes from "rethink" to a fundamental rethink of
  the event representation, and this document says so plainly. If both main
  gates pass (positive recall >= 0.80 pooled A+B, negative recall >= 10/14)
  with no domination bit firing, WP9 step 1 is validated and the
  strict-objective path unblocks. Partial outcomes are reported as such
  with their mechanisms named.

## 2. Results

(Appended after the single scoring pass; nothing above this line changed
after section 1 was frozen.)

Single pass executed 2026-08-17 by
`python3 -m lolo_agent.milestone_discovery_run --v3` under the project venv
interpreter (Python 3.12.13, ~104 s wall, M5, telemetry-only). Report:
`experiments/lolo1-wp5/milestone-scoring-v3-report.json`, content digest
`e2c3434c6eac9c049382dc553c7b59e16b277c914df4685d48db51fdd228f7ad`
(byte-identical on the determinism rerun). Before the pass, the untouched v1
and v2 code paths were rerun under the same interpreter and reproduced their
published reports BYTE-IDENTICALLY (digests `424bb775…`, `898676b5…`), so
the additive v3 changes perturbed nothing.

### 2.1 Volumes, plumbing, and ranking cross-checks

Pair volumes, all counters, seen pools, and distinct/nonzero signature
counts are IDENTICAL to v2 in all three corpora (assembly reused verbatim,
as preregistered: A 914,988 / B 52,497 / C 360,518 pairs; signatures
2,363 / 1,174 / 3,112; nonzero 65 / 121 / 323), and the top-25 rankings
with their scores are IDENTICAL to v2 per corpus — the score product was
untouched, exactly as §1.2.2 required. All census cross-checks reproduce
(collections 11 + 36 = 47; losses 14; scene transitions 504 / 4,726 /
22,773).

### 2.2 Timer/animation domination — NOT observed (unchanged)

Top-10 neutral-drift fractions: A 0.1, B 0.0, C 0.0; every corpus has
>= 10 non-zero signatures. Top-10 candidate-event precision (reported, not
gated) is unchanged from v2 at A 0.5, B 0.6, C 0.2 (the ranking did not
move).

### 2.3 Reset bleed-through — REPAIRED (the v2 dominant mechanism is gone)

The component anchor does what §4.36 asked. The three v2-named
bleed-through signatures, re-read from the v3 report's own diagnostics:
corpus B `a03abd13…` (rank 25) drops from 6-of-9 rewound windows to ZERO,
classifying 9/9 evaluable occurrences positive — all 10 of its v2-failed
instances now pass; corpus A `1dde50a3…` (rank 6) and `55daa1a2…` (rank 4)
keep only the rewound windows whose component cells genuinely reverted.
Pooled positive recall rises 15/47 = 0.319 -> 27/47 = 0.574 (A 2 -> 5,
B 13 -> 22, plus one positive-valence zero-score instance in B).

### 2.4 Heart-type collection recall — FAILED A THIRD TIME (27/47 = 0.574 vs 0.80)

The 20 remaining failures decompose, from the report's per-instance rows:

1. **Component-anchored negatives (10: 4 in A, 6 in B).** The anchored
   rewind fires because the event's OWN component cells revert toward the
   pre-event configuration at a later reset-crossing successor — i.e. the
   collected state is genuinely undone on the observed timeline (corpus B
   `5db312ba…` at the replayed d53 of v16–v19, `d499e4fb…` at hearts-v7
   d44 / hearts-v8 d50; corpus A includes `55daa1a2…` and both `1dde50a3…`
   instances). This is not bleed-through: the census (§1) already noted
   the collected-count field is timeline-relative and re-collections
   recur; the evaluator label counts a commit-time increment while the
   pixel side correctly measures that the change did not survive. The
   label and the measured event live at different levels of the
   representation.
2. **Return-censored own occurrences (6: 2 in A, 4 in B).** The instance's
   own committed occurrence has no successor window (lineage empty AND no
   verified arm rooted at its endpoint), so occurrence-scoped valence is
   unresolved where v2's class valence borrowed windows from replayed
   occurrences.
3. Residual mixed occurrences (3 in B) and one positive-valence
   zero-score instance (B v19 d518).

### 2.5 Life-loss negative valence — FAILED 2/14 (v2: 14/14; REGRESSION)

Occurrence scoping starves exactly the terminal class. 12 of the 14 fatal
commits' OWN successor windows are EMPTY: after a death the assisted
planner restores (restored rows are never parent-linked, so no later
commit descends from the death endpoint), and no verified branch or option
is ever rooted at a death state, so the branch-followup fallback is empty
too. Those 12 occurrences are return-censored -> unresolved. Only v14 d41
and v15 d197 — whose own windows happen to be non-empty — classify
negative, and both come from design-time-inspected runs (§1.5 inherited
disclosure); all 10 uninspected instances are return-censored. The
REPORTING-ONLY class plurality still labels every one of the 14 mapped
signatures negative (ranks 599/810) — the delayed-divergence evidence
exists, but only at class scope, which §1.2.2 deliberately removed from
the gate.

### 2.6 Corpus C falsification bits and anchor (unchanged)

Not animation-dominated (drift fraction 0.0; 323 non-zero signatures). The
preregistered Floor 1 clear anchor: d506/d507 produce no committed-pair
diff at the pooled granularity; d508 carries the transition signature —
POSITIVE at BOTH occurrence scope and class scope (novel-and-persistent,
not rewound under the component anchor), score 0 at rank 2,208 because the
transition is action-independent at its committed decision, exactly as in
v1/v2.

## 3. Verdict

**WP9 step 1 is FALSIFIED A THIRD TIME on heart separation (0.574 << 0.80;
v1 0.149, v2 0.319), and the occurrence-scoped negative gate FAILED as
well (2/14; v2 passed 14/14 at class scope). Per the preregistered
interpretation, WP9 step 1 demotes to a FUNDAMENTAL RETHINK OF THE EVENT
REPRESENTATION.** That is the plain statement §1.6 committed to.

Preregistered bits: timer/animation domination false on all three corpora;
heart inseparability TRUE; negative recall 2/14 FAILED.

What the rethink proved: both §4.36 fixes work as specified. The component
anchor eliminates reset bleed-through completely (the dominant v2
mechanism: its flagship class goes 0-rewound, 9/9 positive, +12 instances
recovered), and occurrence scoping provably stops class valence from
overwriting occurrences. The instruments are correct; the gates still
fail.

Why this is a representation problem, not another aggregation problem: the
two gates now fail for MIRROR-IMAGE structural reasons inside the same
event representation (matched endpoint pairs + successor windows over
whole-array configurations):

1. Ten collection instances fail because their component cells genuinely
   revert — the evaluator's commit-time label and the pixel-level
   persistence question are different events. No valence rule over this
   event unit can call a change both "positive milestone" (label) and
   "did not survive its own timeline" (measurement) — the unit itself is
   wrong for the label.
2. Twelve loss instances fail because terminal events, by construction,
   have no observed future on their own timeline — their evidence lives
   only in the class of replayed occurrences. Class scope bleeds (v2);
   occurrence scope starves (v3). The window, as an attribute of ONE
   occurrence's timeline, cannot carry terminal valence.

Both point past scoring-rule surgery toward event units that are stable
across replays and resets — object/slot-level state changes rather than
whole-array endpoint diffs — which is exactly the object-centric direction
already recorded in `docs/learnings.md` §7.3. Three preregistered passes
(digests `424bb775…`, `898676b5…`, `e2c3434c…`) with named,
instance-verified mechanisms are the evidence trail; per §4.33's framing,
each falsification cost one offline day on stored telemetry.

What survives for reuse: matched-NOOP componentized differencing, lineage
windows with fallbacks, the component-anchored structural-rewind detector
(now verified to separate "this change reset" from "a reset crossed the
window"), the unchanged ranking (precision holding at its quintupled v2
level), and class-level delayed-divergence labeling of terminal signatures
as a diagnostic.

Consequences for the plan of record (not enacted here): WP9 step 1 moves
to "fundamental rethink of the event representation"; the strict-objective
path stays blocked at this step; no fourth rescore of the matched-endpoint
event unit — the next preregistration, if any, must define a different
event unit first. Gate 4 remains unaffected.

## 4. Provenance

- Runner: `lolo_agent/milestone_discovery_run.py --v3` (stdlib-only;
  additive; the v1 and v2 paths reproduce their published reports
  byte-identically under the project venv interpreter, verified before the
  v3 pass).
- Scorer: `lolo_agent/milestone_discovery.py` `*_v3` pure functions at the
  §1.3 preregistered config; provenance rows carry `source="telemetry"`.
- Inputs: stored `events.jsonl` streams of the three census corpora; no
  emulator, no frames, no `decisions.csv` in the scoring path.
- Report: `experiments/lolo1-wp5/milestone-scoring-v3-report.json`
  (deterministic canonical JSON; `content_digest` is the SHA-256 of the
  payload with that field absent; determinism rerun byte-identical).
- Tests: `tests/test_milestone_discovery.py` append-only v3 fixtures (89
  tests in file; full-suite figure recorded in the session summary).
- Post-run diagnostics in §2.3–2.5 re-read per-signature and per-instance
  fields of the v3 (and published v2) reports; no re-scoring occurred and
  no threshold was revisited after results were first observed.
- Context: `docs/learnings.md` §4.33/§4.36,
  `docs/milestone-scoring-2026-08-16.md`,
  `docs/milestone-scoring-v2-2026-08-16.md`,
  `docs/milestone-event-census-2026-08-16.md`.
