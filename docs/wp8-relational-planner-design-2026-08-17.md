# WP8 relational planner extraction — implementation design (2026-08-17)

Status: DESIGN ONLY — no code changed by this document; preregistration-ready
structure for the extraction, the shadow run, the conflict-root mining, and
the Gate 4 paired ablation. Nothing here is evidence.
Authority: roadmap §7 WP8 and §17 item 1 ("`relational_planner.py` follows
the ablation's outcome"), direction-review Amendment E
(`docs/direction-review-2026-08-16.md` §3.E: "`relational_planner.py`
extraction as the declared fallback if monolith integration thrashes"),
engaged by the preregistered WP8-lite FAIL (learnings §4.43,
`docs/wp8-lite-ablation-design-2026-08-16.md` §7.6).
Evidence base: certified probe series v322–v326 (learnings §4.26–§4.30,
Gate 3 closed on the assisted track), the WP8-lite paired ablation
v327/v328 (§4.43), WP5 PROMOTE-to-shadow (§4.42).

## 1. What this is, and what the FAIL taught

WP8-lite put a verified-accessibility preference term into the two existing
archive/restore-selection seams and ran the preregistered paired ablation.
The verdict (design doc §7.6): **bit 1 PASS** — the term deliberately ranked
the removal-class restore +25.0 for the hardened reason; **bit 2 FAIL** —
the arms' committed trajectories are *identical*, because the baseline
frontier score's stagnation-driven restore already selects the same
removal-class branch non-deliberately (29.578 vs 54.578 on the same winner,
identical selection score 30.7). Two consequences drive this design:

1. **Deliberateness at a single restore is behaviorally redundant at roots
   where novelty and certified value agree.** An ablation that can
   discriminate needs *score conflict* — a configuration the baseline
   scorer disprefers but certified accessibility prefers (§4.43 learning).
   Root selection is part of experimental power.
2. **The planner's real gap is sustaining and chaining preparations, not
   the single restore choice** (§4.43, verbatim). v325/v326 completed the
   full chain — removal → east traversal → `(12,11)` heart at decision 4 —
   *non-deliberately*; v327/v328, started from the earlier pre-push root,
   reached the removal configuration at decision 2 but never converted it
   into the milestone within 8 decisions. What no run has ever shown is a
   hypothesis held across decisions and exploited on purpose.

WP8 proper therefore extracts the smallest hypothesis-planning slice that
can demonstrate **chained deliberate preparation** — something the
single-restore WP8-lite provably could not test.

## 2. Scope: the chained-preparation slice

### 2.1 The Room 3 chain, in certified terms

Room 3 has five heart slots (root telemetry `goal_heart_slots`
`[[96,128],[128,64],[128,128],[144,192],[192,176]]` — cells
`(6,8) (8,4) (8,8) (9,12) (12,11)`). The certified facts:

- Pre-removal certified envelope: 7 cells
  (`(6,6)…(6,10),(7,10),(8,10)`), identical for pushed and pre-push
  configurations (§4.28 — the confirmed v318 push is certified neutral;
  "the object is the door").
- Removal of the `(7,6)` entity (transform-in-place by button → push the
  transformed object one cell east → expulsion east along row 6; §4.29)
  certifies 24 cells including milestone-bearing `(12,11)` (§4.30, Gate 3
  closed at Jaccard 1.0 by v326).
- Hearts `(6,8)` and `(8,8)` are collectible from the baseline+removal
  configurations (v327/v328 collected both, d1/d3). Heart `(12,11)` is
  inside the certified 24-cell envelope but was **never** collected from
  the v318 root within 8 decisions by any arm. Hearts `(8,4)` and `(9,12)`
  lie **outside** the certified 24-cell envelope — reaching them requires
  further, currently unmeasured preparation.

The chain the slice must support, entirely inside existing certified
evidence: **establish** the removal configuration → **hold** it → **exploit**
it by navigating east to the certified milestone-bearing cell `(12,11)`.
The `(8,4)`/`(9,12)`-class hearts are the declared *continuation* target
(they force a second establish hypothesis later) but are not required for
Gate 4 and are not scored in this design's ablation.

### 2.2 Hypothesis types in scope (first cut)

Of roadmap §7 WP8's eight candidate hypothesis kinds, three matter first,
because each maps onto machinery and evidence that already exist:

| Kind | Roadmap WP8 item | Existing support |
| --- | --- | --- |
| `establish_configuration` — realize a configuration with a certified accessibility record (restore an archived branch carrying its signature, or reproduce the manipulation via behavior-model rules) | "preserve or restore a valuable configuration"; "reproduce a known displacement / appearance transition" | Certified record store + preference term (WP8-lite seams, `neural_planner.py:19130–19223`); removal-chain descriptors already first-class in the behavior model: `controlled_appearance_transition` / `controlled_entity_displacement` / `controlled_entity_removal` / `controlled_entity_expulsion` (`lolo_agent/entity_behavior.py:281–320`); §17 item 2 names button-conditioned transformation posteriors as the highest-value rule family (type-7 result) |
| `hold_configuration` — keep the established configuration intact across subsequent decisions | "preserve or restore a valuable configuration" (the sustain half) | Certified-hold predicate productized in `lolo_agent/accessibility.py` (`CertificationWindow:174`, `certify_branch:294`); endpoint-relative track state (`object_correspondence.EndpointRelativeTrackState:204`) is the §17 item 3 contract for "still changed" vs "changed at some point" |
| `exploit_configuration` — move toward a certified newly-reachable milestone cell / interaction frontier while the hold predicate is satisfied | "move toward a newly reachable interaction frontier" | Certified records carry `certified_milestone_cells`; the option search already has goal-proximity and milestone reserve families to direct expansion (`neural_planner.py:10938–11120`) |

Deliberately out of the first cut (each returns on its own evidence):
under-tested-track approach and action/context testing (already served by
the entity-curiosity reserve, `neural_planner.py:11008–11064` and config
`:174–176`); phase-contradiction investigation (WP7 is off Gate 4's
critical path per Amendment E); return-path attempts; any learned-model
proposal ranking beyond the certified records and existing behavior-model
posteriors (the §4.16 lesson: telemetry before selection authority).

**Chained** means: at least two hypotheses executed in sequence where the
successor's initiation condition references the predecessor's *verified*
outcome (exploit initiates only on an established-and-held configuration),
each logged before its execution, with replanning after every verified
transition. This is precisely what a single restore-selection scalar
cannot express, and what §4.43 licenses WP8 to test.

## 3. Module contract — `lolo_agent/relational_planner.py`

Pure module, stdlib + project dataclasses only. No emulator, no torch, no
file I/O, no planner state, no imports from `neural_planner` (one-way
dependency, like `accessibility_preference`). Strict-lineage clean
(`python -m lolo_agent.strict_lineage` must report `assisted: false`; the
*records* feeding it remain assisted-lineage, as in WP8-lite §2).

### 3.1 Inputs (narrow views, never planner objects)

```python
@dataclass(frozen=True)
class RelationalStateView:
    """What the hypothesis layer may know about the current root."""
    configuration_signature: str          # tracked world-state signature
    track_set_signature: str              # ObjectTrackSet.signature
    player_cell: Optional[Cell]
    remaining_milestone_cells: Tuple[Cell, ...]   # uncollected, from goal telemetry
    decision_index: int

@dataclass(frozen=True)
class ArchiveCandidateView:
    """One archived branch, as restore-selection sees it."""
    state_id: str
    configuration_signature: str
    baseline_score: float                 # the frontier score, pre-bonus
    verified_option: bool

@dataclass(frozen=True)
class TransitionRuleView:
    """One behavior-model rule summary (posterior, not authority)."""
    interaction_signature: str
    transition_kind: str                  # displacement/transformation/removal/expulsion
    posterior: float
    samples: int
    inert_probability: float
```

Certified accessibility enters exclusively as
`accessibility_preference.CertifiedAccessibilityRecord` (`:171`) via the
existing provenance-checked store
(`neural_planner.VerifiedAccessibilityRecordStore:27361`,
`load_verified_accessibility_records:27391`). The structural refusal rule
is inherited: a record whose provenance is not `certified_hold` scores
zero, with the refusal exposed.

### 3.2 Core types

```python
class HypothesisKind(Enum):
    ESTABLISH_CONFIGURATION = "establish_configuration"
    HOLD_CONFIGURATION = "hold_configuration"
    EXPLOIT_CONFIGURATION = "exploit_configuration"

@dataclass(frozen=True)
class RelationalHypothesis:
    kind: HypothesisKind
    hypothesis_id: str                    # deterministic content digest
    target_configuration_signature: str
    initiation: InitiationCondition       # relational predicate over RelationalStateView
    termination: TerminationCondition     # achieved / violated / budget-exhausted
    realization: RealizationObjective     # see 3.3
    score: HypothesisScore                # see 3.4
    chain_parent_id: Optional[str]        # predecessor whose verified outcome gates this

@dataclass(frozen=True)
class HypothesisPlan:
    """Bounded, deterministically ordered queue (WP8 test requirement)."""
    hypotheses: Tuple[RelationalHypothesis, ...]   # max_queue enforced
    active_id: Optional[str]
```

`InitiationCondition` and `TerminationCondition` are relational: predicates
over configuration signature, track-set signature, certified-record
availability, and milestone-cell membership *relative to the record* —
never absolute coordinates in any persisted option (roadmap WP8 item 8; the
room-scoped cells live only in the episodic record store, exactly as
today). Serialized options carry initiation/termination conditions, option
transfer evidence counts, and NO controller sequences.

### 3.3 Realization objectives (hypothesis → exact-search objective)

The module never searches. It emits a declarative objective the monolith's
option search interprets:

```python
@dataclass(frozen=True)
class RealizationObjective:
    kind: str          # "restore_archive" | "reach_cells_under_hold" | "reproduce_transition"
    # restore_archive: preferred configuration signature (restore seam realizes it)
    # reach_cells_under_hold: target cell set + hold configuration signature
    # reproduce_transition: interaction signature + expected transition kind
    payload: Mapping[str, Any]
    branch_budget: int                    # per-hypothesis slice of the search budget
```

`reach_cells_under_hold` maps onto the existing reserve pattern: a new
reserve family ranked like the milestone-continuation and world-state
reserves (`_human_prior_milestone_continuation_candidates:18398`,
`_human_prior_world_state_reserve_candidates:18250` with the
`verified_accessibility_rank` injection already at `:10967`), plus the
goal-proximity machinery (`_human_prior_active_preparation_goal_slot:17941`
shows the precedent for a preparation-scoped pixel goal). Exact save-state
search remains the acceptance oracle; model posteriors only rank.

### 3.4 Hypothesis score (roadmap WP8 rule, every component logged)

```python
@dataclass(frozen=True)
class HypothesisScore:
    verified_milestone_evidence: float        # from certified_milestone_cells minus collected
    expected_accessibility_improvement: float # AccessibilityPreferenceComponents total (certified only)
    information_gain: float                   # 0.0 in the first cut; field exists, logged
    option_transfer_evidence: float           # realized-option reuse count
    reversibility_confidence: float           # 0.0/unknown in the first cut; logged
    causal_terminal_risk: float               # from causal hazard evidence, subtractive
    predicted_inert_probability: float        # subtractive, from TransitionRuleView
    search_cost: float                        # subtractive, branch_budget-scaled
    repeated_experiment_count: float          # subtractive
```

The accessibility term is computed by the existing
`verified_accessibility_preference` (`accessibility_preference.py:394`)
and inherits its churn exclusion, censoring discipline, and
predicted-provenance refusal (`AccessibilityPreferenceComponents:282`).
Unverified predicted accessibility must not be scored as observed —
enforced structurally, tested again here. `log_fields()` emits the full
flat decomposition under a `relational_hypothesis_` prefix.

### 3.5 The chain state machine (replan per verified transition)

```python
def propose(state: RelationalStateView,
            records: Mapping[str, CertifiedAccessibilityRecord],
            archive: Sequence[ArchiveCandidateView],
            rules: Sequence[TransitionRuleView],
            realized_options: Sequence[RealizedOption],
            config: RelationalPlannerConfig) -> HypothesisPlan

def advance(plan: HypothesisPlan,
            verified: VerifiedTransitionSummary) -> HypothesisAdvance
    # -> continue | hypothesis_achieved | hold_violated | budget_exhausted | replan
```

Both pure. `advance` consumes only verified-event summaries (committed
decision endpoints, restore selections, configuration signatures after the
step). Exact outcomes override priors: a verified transition contradicting
the active hypothesis's expectation forces `replan`, never a silent
retry. Hold violation (configuration signature no longer maps to the held
record and no `mapped` equivalence per the §6.8 baseline-designation rule)
terminates the exploit hypothesis and logs the reason.

### 3.6 Telemetry (all new, additive)

- `relational_hypothesis_proposed` — full queue with per-hypothesis score
  decomposition, BEFORE any realization step executes (Gate 4 criterion 1);
- `relational_hypothesis_activated` / `_realized` / `_achieved` /
  `_terminated` (reason-coded: achieved, hold_violated, budget_exhausted,
  replanned, contradicted);
- `relational_option_stored` / `_reused` with initiation/termination
  conditions and transfer evidence;
- every event carries `hypothesis_id`, `chain_parent_id`, and the
  `relational_hypothesis_*` score fields, so the paired analysis can
  attribute every behavioral difference to a named hypothesis, mirroring
  §3.5 of the WP8-lite design.

## 4. Extraction steps (ordered; what moves, what stays)

Monolith context: `neural_planner.py` is 27,497 lines / 275 methods /
1.18 MB — the §14 "planner complexity grows inside one file" risk is
already realized. The extraction adds NO new scoring inside the monolith.

1. **Land the module** (`relational_planner.py` + `tests/
   test_relational_planner.py`): types, `propose`, `advance`, scoring,
   serialization. Pure; zero planner imports; strict-lineage lint clean.
2. **Views assembly seam** (read-only): a small
   `_relational_state_view()` on the planner assembling
   `RelationalStateView` from existing fields — root object state
   (`object_tracks.HumanPriorRootObjectState:119`, already imported at
   `neural_planner.py:44–56`), `tracked_world_state_signature` on
   `_ArchivedBranch:219`/`_HumanPriorOptionNode:296`, goal telemetry heart
   slots. `ArchiveCandidateView` from the archive with each branch's
   pre-bonus `_archive_frontier_score` (`:19225`). `TransitionRuleView`
   from `AnonymousEntityBehaviorModel` summaries
   (`entity_behavior.py:518`, `transition_for:616`).
3. **Propose-and-log seam** in `decide()` (`:19910`), after life-loss
   recovery and before restore/stagnation handling: call `propose`, emit
   `relational_hypothesis_proposed`. In **shadow mode**
   (`relational_planner_authority = "telemetry"`), stop here — behavior
   is byte-identical by construction (the §4.6-style invariance argument,
   re-proven by test).
4. **Realization seams** (authority mode only, each ≤ ~20 lines):
   - `restore_archive`: pass the active hypothesis's preferred
     configuration signature into restore selection alongside the
     existing verified-accessibility bonus (`_restore_if_stagnant:25058`
     candidate ranking; the bonus plumbing at `:19262–:19328` stays
     untouched);
   - `reach_cells_under_hold`: one additional reserve family in the
     option-search reserve assembly (`:10938–11120`), budgeted like the
     milestone-continuation slots, plus the hold predicate as a
     certification check on candidate endpoints (reusing
     `world_effect_cells_state_signature` / root-track match as in
     `accessibility.certify_branch:294`);
   - `reproduce_transition`: rank existing entity-frontier candidates by
     the hypothesis's interaction signature (the entity-curiosity
     representative machinery at `:11008` already indexes candidates by
     `entity_interaction_signature`).
5. **Feedback seam**: after each committed decision / restore, build
   `VerifiedTransitionSummary` and call `advance`; store realized options
   through `memory.py`/archive metadata using `object_tracks`
   serialization conventions (`archived_track_fields:537`,
   `object_track_telemetry:690`, `ObjectTrackSet.from_archive_metadata:
   1118`).
6. **Config**: `relational_planner_enabled: bool = False`,
   `relational_planner_authority: str = "off"` (`off|telemetry|
   selection`), `relational_max_queue`, per-kind budgets — all on
   `NeuralPlanningConfig` (`:63`), validated like
   `verified_accessibility_weight` (`:759`), default-off.

What deliberately stays in the monolith: all existing scoring
(`_archive_frontier_score`, reserves, goal-exhaustion recovery
`:146–148`/`:1256`), the WP8-lite seams verbatim, search execution,
archive/restore mechanics, telemetry transport. What is deliberately NOT
built: any change to `object_correspondence.py` wiring
(`CorrespondenceResult:506` is consumed if WP2 planner integration lands
first, via the same views — this design does not gate on it), any learned
proposal model, any new reward weight.

## 5. Test plan (unit, no emulator; mirrors roadmap WP8 test list)

1. Hypothesis generation from anonymous-track fixtures: removal-class
   record + matching archive candidate ⇒ establish hypothesis proposed;
   no certified record ⇒ no establish hypothesis (fail open to nothing).
2. Bounded queue and deterministic tie-breaking: same inputs ⇒
   byte-identical `HypothesisPlan`; queue never exceeds `relational_max_queue`.
3. Predicted-provenance refusal: a `predicted` record contributes exactly
   zero with the refusal exposed (regression twin of the
   `accessibility_preference` rule).
4. Known-inert down-ranking: `predicted_inert_probability` from
   `TransitionRuleView` strictly lowers the score, separately logged.
5. Exact outcome overrides the prior: `advance` with a contradicting
   verified transition ⇒ `replan`, never `continue`.
6. Chain mechanics: establish→hold→exploit ordering; exploit refuses to
   initiate without a verified established parent; hold violation aborts
   the chain with reason-coded telemetry; milestone achievement
   terminates.
7. Option storage: realized option round-trips serialization; initiation
   condition is relational (translated-layout fixture matches; an
   absolute-coordinate initiation is a test failure); no controller
   sequence appears in the persisted option (no universal macro from one
   room-specific trajectory).
8. Score decomposition: every nonzero component appears in `log_fields()`;
   totals equal the sum of parts.
9. Seam invariance (in `tests/test_ensemble_planner.py`, alongside the
   250 existing planner tests): `relational_planner_authority != "selection"`
   ⇒ planner ranking and restore selection bit-identical to today
   (the §4.6 argument, as an executable test).

## 6. Conflict-root mining procedure (preregistered, offline, ~zero emulator cost)

Purpose: find or construct ablation roots exhibiting *score conflict* —
the §4.43 requirement that the baseline scorer and hypothesis-level
accessibility preference genuinely disagree — before either Gate 4 arm
runs. All mining is read-only over stored telemetry in
`experiments/lolo1-entity-v10/evaluations/` (v322–v328; 61–179 MB per run;
`states/` archived and restorable, e.g. 116 states in v325).

Procedure (a small offline tool, `lolo_agent/conflict_root_mining.py` or a
scored notebook committed as a report artifact — decided at implementation
time; the procedure below is fixed now):

1. **Candidate instants.** For each run, walk `events.jsonl` for every
   restore-selection instant (`archive_branch_restored` and the ranked
   alternatives recorded around it) and every
   `human_prior_option_archive_added` event, collecting per-candidate:
   `state_id`, `tracked_world_state_signature`, baseline score
   (`score` / `persistent_frontier_value`), decision index, and depth.
2. **Offline re-scoring.** For every candidate, compute the would-be
   verified-accessibility bonus against the v322–v326 record store
   (`wp8lite-accessibility-records.json`, sha `cf01a67a…`) using the pure
   `verified_accessibility_preference` — the same §6.8
   baseline-designation rule for the root side.
3. **Conflict predicate.** An instant is a conflict candidate iff
   `argmax(baseline)` ≠ `argmax(baseline + bonus)` over its candidate
   set, i.e. the baseline's top-ranked branch maps to a certified-neutral
   (or unmapped) configuration while a certified-improving branch exists
   at lower baseline rank. Record the **conflict margin**: the minimum
   bonus that flips the argmax, and the baseline gap it must overcome.
4. **Root families to mine**, in priority order:
   - *Novelty-decoy roots*: instants where fresh unexplored signatures
     (high frontier novelty — e.g. northern-region endpoints) outrank the
     removal-class branch on the baseline score. v327/v328's d5/d8
     restores and mid-window archive states are the first place to look,
     since §7.7 shows the reserve-order permutation already brushed this
     boundary.
   - *Post-exploit roots*: v325/v326 post-d4 states (milestone `(12,11)`
     collected) where the removal record's milestone component is spent —
     these exercise the score's milestone-vs-cells decomposition and
     are the natural staging ground for the `(8,4)`/`(9,12)` continuation.
   - *Exhaustion roots*: v324 d7-class states under goal-exhaustion
     recovery, where the recovery machinery biases toward stale goals.
5. **Manifest.** Emit a preregistered conflict-root manifest: state
   digest, source run/decision, candidate table with both scores and the
   conflict margin, and the declared VOID condition for each root. The
   manifest is appended to THIS document before any Gate 4 arm launches
   (mirroring §6.1 of the WP8-lite design). If mining finds **no**
   natural conflict instant, that is itself a disclosed result; the
   fallback construction is a *seeded* root (archive seeded with both a
   certified-improving branch and a strictly higher-baseline neutral
   branch from sibling runs), disclosed as constructed, never silently
   substituted.

Bias control: roots are chosen from already-recorded telemetry by a fixed
predicate, before any relational-planner run exists; the mining tool's
output is deterministic and its report digest is recorded. No new native
runs may be launched to hunt for roots.

## 7. Gate 4 closure path

Gate 4 (roadmap §12) with today's ledger:

| Criterion | Status | Evidence / gap |
| --- | --- | --- |
| Hypothesis logged before execution | **OPEN — needs the relational planner** | WP8-lite logged valuation components *at the restore*; nothing proposes a hypothesis before execution. |
| Controller realization discovered, not supplied | Component satisfied; deliberate form OPEN | Ordinary search discovered the removal chain spontaneously (v324/v325/v327/v328); restores realized it. Must recur under hypothesis direction, not incidentally. |
| Accessibility verified | **CLOSED (assisted track)** | 7 → 24 certified cells, repeated at Jaccard 1.0 (§4.30, Gate 3 closed by v326). Strict re-measurement remains gated on WP5 shadow evidence (§4.42). |
| Configuration retained across a planning cycle | Component satisfied | v318 push retained four generations (§4.28); removal-class signature carried across d2–d8 committed decisions in v327/v328. Retention *because of the hypothesis* is what remains. |
| Subsequent positive milestone reached | Component satisfied, non-deliberately | v325/v326 collected `(12,11)` at d4 from the post-removal root. From the v318 root, no arm collected it within 8 decisions — the chain has never been completed deliberately from before the preparation. |
| Counterfactual neutral configuration worse at matched budget | **OPEN — needs conflict roots** | Configuration-level counterfactual is certified (7-cell envelopes both sides, §4.28); the *planner-level* counterfactual failed to discriminate at the v318 root (v327 ≡ v328). |

### 7.1 The Gate 4 experiment sketch (paired, preregistered before launch)

Two experiments, run in this order, each with its own bits:

**E1 — chain completion at the non-conflict root (the consequence gap).**
Root: the exact v318-lineage resume of §6.1 of the WP8-lite design
(digests already preregistered there). Arms: control = today's planner
(relational authority off); treatment = relational planner in `selection`
authority. Matched budget: the v327/v328 envelope (12,232 branches
observed; window fixed at the first 10,000 verified branches,
branch-budget semantics per the §7.2 ruling; 8 decisions; wall ceiling
10,800 s/arm). Bits, all preregistered, ANY mixed outcome = FAIL:

1. *Deliberate chain*: treatment emits `relational_hypothesis_proposed`
   (establish → exploit with `chain_parent_id` linkage) before decision 2,
   `_realized` on the removal-class restore, and `_achieved` on the
   exploit — the full decomposition present at each step.
2. *Chained consequence*: treatment collects the milestone heart at
   `(12,11)` (slot `(192,176)`) within the window; control does not (the
   recorded precedent: neither v327 nor v328 did). Strictly-earlier
   collection also passes if both collect. Metric is the milestone cell
   only — never affordance counts.
3. *No safety regression*: treatment life-loss confirmations ≤ control's
   within the window.

E1 attacks the exact bit that failed in WP8-lite: at this root incidental
behavior demonstrably does NOT complete the chain, so completing it is
attributable to the hypothesis layer — while sharing the root keeps every
digest and precedent from §6.1/§7 of the WP8-lite design reusable.

**E2 — deliberate selection under score conflict (the discrimination
gap).** Root: the top-ranked mined conflict root (§6 manifest). Same arm
structure and budget discipline. Bits:

1. *Exercised conflict*: telemetry records the disagreement at the
   selection instant — baseline argmax ≠ hypothesis-preferred candidate —
   and the treatment follows the hypothesis while the control follows the
   baseline (both logged with full decompositions).
2. *Consequence*: treatment reaches a certified previously-unreachable
   cell or milestone strictly earlier than control (control never doing
   so within the window also passes).
3. *No safety regression* (as E1).

VOID (either experiment): the declared choice never materializes in-run
(no removal-class candidate in the archive within the window; or, E2, the
mined conflict does not reproduce at run time). Fix the staging, disclose,
rerun once — a VOID is not evidence, per WP8-lite §3.4.

Outcome rules: PASS on E1+E2 closes Gate 4 on the assisted track and
unblocks the roadmap's "decisive milestone" claim at §7 WP8's acceptance
gate (all five clauses map onto E1 bits 1–2 + the standing Gate 3
record + E2 bit 1). FAIL on either stays engineering-only, recorded in
learnings, and the next move is a *representation* question (WP2/WP3
integration depth), not weight tuning and not a rerun — the same
discipline §3.4 of the WP8-lite design enforced.

### 7.2 Shadow run precondition (the §4.16 lesson)

Before E1: one native shadow run (authority = `telemetry`) at the E1 root,
budget identical, scoring nothing but confirming (a) byte-identical
behavior vs the recorded v327 control stream (invariance in vivo), and
(b) well-formed hypothesis telemetry at every decision. Offline-passing
components have lost native comparisons before (§4.16: "offline averages
can hide native failure"; "a predictor can be useful for telemetry before
it is reliable enough for selection") — the shadow run is the cheap
insurance that pattern demands, and doubles as the logging-overhead
control.

## 8. Budgets (bounded, declared now)

- Module + tests + seams: code-review-scale work, no emulator time.
- Mining: offline over ~7 stored runs (~560k events total), zero emulator
  cost; one committed manifest.
- Shadow run: ~30 min M5 native.
- E1 + E2: 2 arms × 2 experiments × ~30 min = ~2 h M5 native, run one at
  a time, external watchdog, event ceiling 200k/arm.
- No depth/beam escalation anywhere; no rerun on identical negative
  evidence; nothing on RunPod (§13: emulator branching is M5-bound).

## 9. Risks and sequencing

1. **§14 planner-complexity risk (realized).** The monolith is 27.5k
   lines; WP8-lite's own design had to be applied by a different lane
   owner (§4.0 ownership note). Mitigations here: the module is pure with
   one-way imports; monolith touch is limited to the enumerated seams
   (§4 steps 2–5), each anchored by grep-able names, none inside the
   high-conflict scoring bodies; authority is config-gated off by
   default with an executable invariance test.
2. **Unmerged worktree branches.** `claude/amazing-fermat-73b137`
   (commit `6a8488a`, "Carry the root object track through the causal
   archive") is one commit ahead of main and touches
   `neural_planner.py` (97 lines), `docs/telemetry.md`, and
   `tests/test_ensemble_planner.py` (+288). It is the learnings §4.29
   instrument fix — without it a mid-run causal-archive restore silently
   resets configuration-hold evidence, which would corrupt the
   `hold_configuration` hypothesis's verification. **It must merge before
   the seam work starts** (its merge base is `510db51`, 38 main commits
   behind — including the applied WP8-lite seam patch; expect a real
   rebase, starting around the `:1334` region). `claude/brave-allen-b00cf2`
   has no commits ahead — confirm and prune. WP5 shadow wiring also
   queues behind the planner-file release (§17 status note); sequence:
   merge `6a8488a` → WP8 seams → WP5 shadow wiring, with explicit file
   ownership declared per §14, so the three planner-file consumers never
   interleave.
3. **Native-integration failure pattern.** §4.16's record: offline gates
   passed, native mean comparisons lost, paired ablations mixed.
   Mitigations: telemetry-first shadow stage (§7.2) before any selection
   authority; every hypothesis effect decomposed into named logged
   components; fail-open to byte-identical baseline behavior whenever no
   hypothesis initiates; exact-outcome override forced by the state
   machine.
4. **Root-selection bias.** Mining is preregistered, deterministic, and
   restricted to existing telemetry (§6); the manifest lands in this doc
   before any arm runs; constructed/seeded roots are disclosed as such.
5. **Provenance.** Everything here is assisted-track (the records derive
   from the player-anchored hold instrument). No strict claim is made or
   implied; the WP5 shadow campaign (§4.42) remains the strict path, and
   the strict-lineage linter guards the module boundary.

## 10. What this document does not claim

No hypothesis planner exists yet; no bit here is scored; Gate 4 remains
open. The next durable artifacts, in order: the module + tests commit, the
merged `6a8488a`, the conflict-root manifest appended here, the shadow-run
disclosure, and only then the E1/E2 preregistration addendum with exact
digests and command lines — each before the step it governs executes.
