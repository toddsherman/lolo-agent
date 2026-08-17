# WP9a milestone-discovery scoring run v2 (section-4.33 redesign) — 2026-08-16

Status: preregistration written and frozen BEFORE the v2 scoring pass; results
and verdict appended after the single corpus run. This is the redesign-and-
rescore step recorded in `docs/learnings.md` §4.33 after the v1 pass
(`docs/milestone-scoring-2026-08-16.md`, report digest `424bb775…`) was
falsified via heart-inseparability (7/47 = 0.149 against the 0.80 gate) with
the reversion-based negative-valence rule failing 0/14. The falsification
frame here is IDENTICAL to v1: same corpora, same gates, same two
falsification bits. Only the three mechanisms §4.33 named are redesigned.

Assisted-footprint caveat (inherited, load-bearing): both label-bearing
corpora are assisted-track. Every number below is an ENGINEERING artifact of
the offline spike. Nothing here is strict-track milestone-discovery evidence
until WP5 supplies a strict-legitimate controllable footprint.

Anonymous-terminology note: as in v1, "heart-type collection", "life loss",
and "coarse-scene transition" are evaluator-side labels for committed
telemetry field changes; they never enter the scorer. The scorer sees only
pooled pixel arrays, matched controls, observed-array histories, and measured
outcome categories.

## 1. Preregistration (fixed before any v2 corpus scoring)

### 1.1 Corpora in and out

Unchanged from v1; the census (`docs/milestone-event-census-2026-08-16.md`)
disqualified none of the three corpora from their v1 roles:

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

### 1.2 The three redesigned mechanisms (fixed by §4.33)

1. **Per-component censoring semantics**
   (`lolo_agent/milestone_discovery.py::extract_component_event`). Changed
   cells of a matched pair are partitioned against the equal-duration
   control: DEPENDENT cells (control kept the root value), AUTONOMOUS cells
   (control reproduced the factual value), AMBIGUOUS cells (control at a
   third value). When dependent cells exist, the event IS the dependent
   component: its signature, reversion, and dependence cover the
   attributable cells only, and autonomous/ambiguous cells are recorded but
   excluded. With no dependent cells the event keeps the full changed set
   and is autonomous (fully reproduced), or dependence-censored (ambiguous
   cells present, or control missing) exactly as in v1 — censoring never
   supports a claim. This removes the v1 mechanism where one concurrent
   animation cell inside a real milestone's diff censored every occurrence
   (the entity-v141 d7 mechanism: 9 changed cells = 2 dependent + 7
   autonomous, verified during design).
2. **Restore-robust successor windows**
   (`lolo_agent/milestone_discovery_run.py::assemble_run_pairs_v2`).
   Successors of a committed transition are later committed arrays in the
   same attempt whose root frame DESCENDS from the event's endpoint frame
   (frame-level parent links: committed non-restored rows to their roots,
   verified branch/option endpoints to their search roots; restored commits
   are never parent-linked). Restored rows are skipped without truncating;
   rows on a non-descendant lineage are skipped as well, and the window
   resumes when the timeline returns to a descendant state; the window caps
   at 8 counted observations (v1's W, unchanged). When the lineage window is
   empty — the assisted planner's restore-after-progress habit, the
   v325/v326 mechanism — the fallback window is the first 8 verified
   branch/option endpoints rooted at the event's endpoint frame in seq
   order (the stored observations of that state's future; v325's post-
   collection root has 6,028 option endpoints, verified during design).
   Both windows skip restores rather than truncating at them; naive
   skip-and-continue was rejected during design because v325's restore
   rewinds to a PRE-event archive, whose committed future would falsely
   revert the event.
3. **Delayed-divergence valence replacing reversion**
   (`lolo_agent/milestone_discovery.py::score_events_v2`). The v1
   reversion-to-seen negative rule is deleted. An occurrence carries
   negative evidence when BOTH hold:
   - its successor window is structurally REWOUND: some successor lies at
     least `rewind_transient_floor` cells from the event's own root while
     lying within `rewind_proximity_ceiling` cells of a configuration
     observed before the event (the run's pre-intervention pool plus the
     attempt's committed arrays strictly before the event root) — the
     reset-to-known-configuration recognizer §4.33 requires, replacing v1's
     exact-array reversion test which could never fire (v1 §2.4); and
   - the occurrence participates in measured factual-vs-control divergence
     structure: its own matched contrast is dependence-evaluable (the
     pre-terminal root contrast — at the fatal commit the control
     reproduces the change, and that measured inevitability is itself the
     differential evidence at the horizon before the terminal endpoint), or
     an ESCAPE divergence was observed at a decision root within the
     lookback window before the commit: a verified alternative arm keeping
     at least `escape_cell_minimum` cells at root values while the
     equal-duration control changed them (the causal-hazard pattern of
     `docs/anonymous-entity-behavior.md`: a change arriving under NOOP but
     absent under an action was avoidable). Dependence-censored occurrences
     with no escape evidence can never be negative.
   A signature classifies NEGATIVE when the fraction of return-evaluable
   occurrences carrying negative evidence reaches
   `negative_divergence_threshold`; the check precedes the positive rule.
   Positive valence is UNCHANGED from v1: persistence rate and mean
   successor novelty over the (now v2) windows. The score product
   m(sigma) = log_rarity x dependence rate x non-return factor x novelty
   margin is UNCHANGED (§4.33: the matched-NOOP core and score shape
   survive), computed over component events and v2 windows.

### 1.3 Thresholds (fixed here, before scoring)

Shared with v1, unchanged: positive recall gate 0.80 pooled over recovered
A+B collection instances; negative recall gate 10/14 corpus-B life-loss
instances mapping to negative-valence signatures; timer/animation domination
bits per corpus (top-10 neutral-drift fraction >= 0.5, or fewer than 10
non-zero-score signatures); precision over top-10 reported, not gated;
`MilestoneScoreConfig` v1 fields at v1 defaults (novelty baseline 0.0,
persistence 0.5, novelty 0.25; the v1 reversion threshold 0.5 remains in the
config but no longer classifies valence); successor window W=8;
pre-intervention seen pool with the v1 §1.5 justification.

New v2 constants, justified from the census scales and a label-free
preflight over the corpora (distributions of changed-cell counts of
committed transitions, root-to-control drift, and escape cells; no
collection/loss labels were read for these measurements):

1. `rewind_transient_floor` = 16 cells (25% of the 64-cell array).
   Label-free preflight: corpus B committed-transition diffs occupy 0–15
   cells (6,515 of 6,555) with an EMPTY 16–48 band and a separated 49–64
   band (40 transitions); corpus A: all committed diffs <= 15; corpus C
   sample: 95.5% <= 15. Neutral drift is <= 12 cells outside the same
   49–64 tail (controls in which the same terminal transient fires). 16 is
   strictly above every ordinary movement/manipulation/drift scale and far
   below the observed terminal transient scale.
2. `rewind_proximity_ceiling` = 8 cells (= floor/2, 12.5% of the array).
   A rewound successor must be an order of magnitude nearer a previously
   observed configuration than the transient it crossed. Design-time
   check on the two loss-bearing runs already named by v1/census (v14 d41;
   v15 d41/d184/d197): post-loss successors sit 0–8 cells from pre-event
   attempt history while >= 58 cells from the event root; ordinary
   mid-play states are also 0–9 cells from their recent history but never
   cross the 16-cell transient floor from their own root, so the JOINT
   condition separates. Disclosure: this check spends 4 of the 14 gate
   instances at design time; the gate's evidential weight rests on the 10
   uninspected instances, and the per-instance table below reports all 14.
3. `escape_cell_minimum` = 8 cells. Label-free preflight: escape cells
   (control changed away from root, factual kept root) are <= 4 in the
   overwhelming bulk (52,471 of 52,497 corpus-B arms; all corpus-A sample
   arms), with a sparse 5–12 tail; 8 sits above the animation-phase
   coincidence scale and at half the proximity ceiling's array fraction.
   The escape term is expected to be SPARSE in these corpora — the
   design-time scan of the v14/v15 fatal decisions found zero escape cells
   at every matched horizon (death fires under every arm or none within 16
   frames) — so the operative negative signal there is the pre-terminal
   inevitability contrast plus structural rewind; escape covers the
   directly-avoidable case without being load-bearing for the gate.
4. `divergence_lookback` = 8 committed decisions (inclusive), mirroring the
   successor window scale W=8; at the corpora's dominant 16-frame action
   duration this spans 128 frames, the same order as the causal-hazard
   machinery's longest matched horizons (224 frames).
5. `negative_divergence_threshold` = 0.5, the same majority-of-evaluable-
   occurrences role the v1 reversion threshold had.
6. `branch_followup_window` = 8 (= W), the fallback window cap.

### 1.4 Pair construction deltas

The v1 §1.4 rules (root-relative matched pairs, exact duration matching,
option-commit dependence censoring, restore commits never pairs, NOOP
commits never factual arms) carry over verbatim. V2 deltas only:

- Successor windows: §1.2.2 above, replacing truncate-at-first-restore.
- Every pair carries the per-decision escape-lookback flag (`True` /
  `False` / censored `None`) computed over control-resolved verified arms
  at decisions within the lookback window.
- Pairs with successors carry the rewind history reference set
  (pre-intervention pool plus pre-root committed arrays, deduplicated).
- Committed-transition signatures for instance mapping are computed by the
  SAME control-aware component extraction the scorer uses (via the covering
  verified pair when one exists), so a recovered instance maps to the
  signature actually scored. NOOP-commit transitions are extracted
  control-less; their full-diff signature coincides with the scored
  signature of autonomous verified arms producing the same arrays, which is
  how a loss on a committed NOOP maps into the ranking (unchanged from v1
  behavior, now made explicit).
- Neutral-drift signatures for the domination bit remain full-diff
  extractions of root-to-control changes, comparable with the full-diff
  signatures of autonomous scored events.

### 1.5 Design-time evidence disclosure

The redesign was grounded, before this preregistration was frozen, in
exactly the failure instances v1 and the census had already published:
`entity-v141-room3-alternative-order-d16` d7 (component partition),
`entity-v325-room3-object-removed-probe-d12` d4 and its restore targets
(lineage windows and fallback), `…v14-life-hazard-credit-1000` d41 and
`…v15-life-hazard-rollback-1000` d41/d184/d197 (rewind structure, escape
vacuity), and the corpus C anchor's published action-independence. No other
labeled instance was examined, no scorer ranking was computed over any
corpus before this section was frozen, and the label-free preflight read no
goal-semantic fields. The v2 unit fixtures are synthetic.

### 1.6 Run discipline

- The v2 scorer runs ONCE per corpus, after this section was written, via
  `python3 -m lolo_agent.milestone_discovery_run --v2`.
- Output: deterministic, content-digested JSON at
  `experiments/lolo1-wp5/milestone-scoring-v2-report.json`
  (schema `milestone-scoring-v2-report/1`); findings and verdict appended
  below. The v1 report and its digest are left untouched; the v1 code path
  was rerun to a scratch location to verify byte-identical reproduction
  after the additive changes.
- Tests: append-only fixtures for each redesigned mechanism in
  `tests/test_milestone_discovery.py` (a mixed-cell event v1 censors and v2
  scores; a restore-skipping lineage window and a branch-followup fallback;
  a delayed-divergence negative that v1's reversion rule classifies
  positive), plus escape/config/determinism coverage.
- Interpretation: if any gate fails again, the failure and its mechanism
  are reported plainly; a second falsification demotes WP9 step 1 from
  "redesign" to "rethink" per the §4.33 plan-of-record framing.

## 2. Results

Single pass executed 2026-08-16 by
`python3 -m lolo_agent.milestone_discovery_run --v2` under the project venv
interpreter (~170 s wall, 1.33 GB peak, M5, telemetry-only). Report:
`experiments/lolo1-wp5/milestone-scoring-v2-report.json`, content digest
`898676b5b4f3ff00510d1278c559e7f004e955280fbb27eb8211687ef4cd1124`
(byte-identical on rerun). Before the pass, the untouched v1 code path was
rerun under the same interpreter and reproduced the published v1 report
BYTE-IDENTICALLY (digest `424bb775…`), so the additive changes perturbed
nothing; under the system Python 3.9 one corpus-C score differs by one ULP
(libm), which is why the interpreter is pinned in this provenance note.

### 2.1 Volumes and plumbing cross-checks

Pair volumes are IDENTICAL to v1 in all three corpora (A 914,988 / B 52,497
/ C 360,518 pairs; controls resolved 914,988 / 52,497 / 327,612; A's 63,948
dropped option roots unchanged), and every census cross-check reproduces
exactly (coarse-scene transitions 504 / 4,726 / 22,773; corpus B losses 14;
collections recovered 11 + 36 = 47 with the same v1 accounting note).
Component extraction consolidates the signature space as designed: distinct
signatures 7,159 -> 2,363 (A), 2,800 -> 1,174 (B), 6,739 -> 3,112 (C);
non-zero-score signatures rise 29 -> 65 / 80 -> 121 / 265 -> 323. The v2
windows engage where v1 censored: lineage windows 679 / 4,487 / 36,374,
branch-followup fallbacks 58 / 315 / 2,716, still-empty windows 292 / 1,753
/ 9,485. Escape divergence is as sparse as preregistered: 0 / 8 / 4
escape-true decisions among 1,294 / 6,563 / 47,759 evaluable.

### 2.2 Timer/animation domination — NOT observed (unchanged)

Top-10 neutral-drift fractions: A 0.1, B 0.0, C 0.0; every corpus has >= 10
non-zero signatures. Top-10 candidate-event precision (reported, not
gated) improves sharply: A 0.5, B 0.6, C 0.2 (v1: 0.0 / 0.1 / 0.2) — with
event-level censoring removed, committed candidate events now reach the top
of the ranking.

### 2.3 Life-loss negative valence — PASSED 14/14 (v1: 0/14)

Every corpus-B life-loss instance maps to a NEGATIVE-valence signature with
basis `delayed_divergence`: the fatal commits are action-independent at
their pre-terminal root contrast (both arms show the change), their
windows contain a structurally rewound successor (>= 58 cells from the
event root, <= 8 cells from pre-event history), and the negative rule
fires exactly as designed. 10 of the 14 instances come from runs never
inspected at design time (v16–v19), so the §1.3.2 disclosure's overfit risk
did not materialize — the rewind structure generalizes across the corpus.

### 2.4 Heart-type collection recall — FAILED AGAIN (15/47 = 0.319 vs 0.80)

15 of 47 recovered collection instances land in a positive-valence
signature with non-zero score (v1: 7). Corpus A: 2/11; corpus B: 13/36.
The v1 failure mechanisms are individually REPAIRED — the entity-v141-class
dependence-censoring is gone (top-rank dependence rates are 1.00 and 8 more
instances pass than in v1), and the v325/v326-class return-censoring is
gone (fallback windows engage 58/315/2,716 times). The 32 remaining
failures decompose, from the report's own per-signature diagnostics:

1. **Reset bleed-through onto merged component classes — the dominant new
   mechanism (28 of 32 failures: 7 in A, 21 in B).** The rewind test is
   window-scoped, not event-scoped: any occurrence whose 8-commit successor
   window crosses a LATER terminal reset (a death in corpus B; a large
   restore-era transition in corpus A) is marked rewound, regardless of
   what the event's own cells did. Because the occurrence is
   action-dependent, it counts as negative evidence; because valence is
   per-signature and component extraction merges replayed transitions into
   large classes, a few rewound windows flip entire collection classes to
   NEGATIVE. Verified instances: corpus B signature `a03abd13…` (rank 25,
   10 occurrences, 6 of 9 evaluable windows rewound -> rate 0.667, 10
   instances failed); corpus A signature `1dde50a3…` (rank 6, 701
   occurrences of which only 4 are window-evaluable, 2 rewound -> rate
   exactly 0.500, 5 instances failed); corpus A signature `55daa1a2…`
   (rank 4, 176 occurrences, ONE evaluable window, rewound -> rate 1.0,
   1 instance failed). The evaluable-window denominator can be a vanishing
   fraction of a class's occurrences, so one or two reset-crossing windows
   decide the class.
2. Residual zero-score positives (2 in B): positive valence but a zero
   score component (novelty margin or dependence rate zero for the
   signature).
3. Residual unresolved (2 in A, mixed-outcome signatures below both
   valence thresholds).

### 2.5 Corpus C falsification bits and anchor (unchanged)

Not animation-dominated (drift fraction 0.0; 323 non-zero signatures). The
preregistered Floor 1 clear anchor: d506/d507 still produce no
committed-pair diff at the pooled granularity; d508 carries the transition
signature — POSITIVE valence (novel-and-persistent, not rewound: the new
floor's successors are far from all pre-event history, confirming the
rewind test's sign discrimination), score 0 at rank 2,208 because the
transition is action-independent at its committed decision, exactly as in
v1.

## 3. Verdict

**WP9 step 1 is FALSIFIED A SECOND TIME — heart-inseparability again
(0.319 << 0.80). Per the preregistered interpretation, WP9 step 1 demotes
from "redesign" to "RETHINK".**

Preregistered bits: timer/animation domination false on all three corpora;
heart inseparability TRUE; the life-loss negative-valence gate PASSED 14/14.

What the redesign proved: all three §4.33 mechanisms are individually
repaired — per-component censoring recovers the v141-class events
(dependence rates 1.00, censoring no longer zeroes collections),
lineage/fallback windows recover the v325/v326-class events, and
delayed-divergence valence solves the negative-valence problem completely
(14/14, generalizing to 10 uninspected instances) where v1 scored 0/14.
Top-10 precision roughly quintupled. These components are validated
instruments and survive for reuse.

What failed, with a named and instance-verified mechanism: **valence is
assigned per-signature while rewind evidence is window-scoped.** A terminal
reset within 8 commits AFTER a genuine milestone marks that occurrence
rewound; component merging pools replays into large classes whose
evaluable-window subset can be tiny; the 0.5 negative threshold over that
subset then flips whole collection classes negative (28 of 32 remaining
failures, including a class where a single evaluable window decided 176
occurrences). The fix is NOT a threshold tweak — with per-signature valence
any threshold trades the 14/14 negative gate against the positive gate on
the same axis. The mechanism points at a structural revision outside this
preregistration's redesign space: rewind evidence anchored to the event's
own component cells (did THIS change reset?) rather than whole-array window
proximity, and/or occurrence-scoped valence with signature-level
aggregation only for ranking. That is a rethink of the valence unit, not a
third scoring pass of the same design — and per §4.33's framing, learning
this on stored telemetry cost one offline day.

Consequences for the plan of record (not enacted here): WP9 step 1 moves to
"rethink" status; the validated sub-instruments (matched-NOOP componentized
differencing, lineage windows, structural-rewind terminal detection) are
available to WP9b and the causal-hazard machinery independently of the
milestone-valence question. Gate 4 remains unaffected.

## 4. Provenance

- Runner: `lolo_agent/milestone_discovery_run.py --v2` (stdlib-only;
  additive; the v1 path reproduces its published report byte-identically
  under the project venv interpreter).
- Scorer: `lolo_agent/milestone_discovery.py` `*_v2` pure functions at the
  §1.3 preregistered config; provenance rows carry `source="telemetry"`.
- Inputs: stored `events.jsonl` streams of the three census corpora; no
  emulator, no frames, no `decisions.csv` in the scoring path.
- Report: `experiments/lolo1-wp5/milestone-scoring-v2-report.json`
  (deterministic canonical JSON; `content_digest` is the SHA-256 of the
  payload with that field absent; byte-identical rerun verified).
- Tests: `tests/test_milestone_discovery.py` append-only v2 fixtures (72
  tests in file; full suite 783 OK, 4 skipped).
- Post-run diagnostics in §2.4 re-read per-signature fields
  (`negative_divergence_rate`, `rewound_occurrences`, `return_evaluable`)
  of named signatures from the same report; no re-scoring occurred and no
  threshold was revisited after results were first observed.
- Context: `docs/learnings.md` §4.33, `docs/milestone-scoring-2026-08-16.md`,
  `docs/milestone-event-census-2026-08-16.md`,
  `docs/anonymous-entity-behavior.md`.
