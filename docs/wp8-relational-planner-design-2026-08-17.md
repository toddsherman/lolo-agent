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

## 11. Conflict-root mining manifest (appended 2026-08-17, §6 executed)

The §6 procedure was executed offline by `lolo_agent/conflict_root_mining.py`
(new pure analysis module, stdlib + `accessibility_preference` only; tests in
`tests/test_conflict_root_mining.py`, 41 tests; full suite 953 OK, 4
skipped). Read-only over stored telemetry; zero emulator cost; no native
runs. Output is deterministic — two full reruns produced byte-identical
manifests. Regenerate with:

```
.venv/bin/python -m lolo_agent.conflict_root_mining \
  --evaluations-dir experiments/lolo1-entity-v10/evaluations \
  --records experiments/lolo1-wp5/wp8lite-accessibility-records.json \
  --output experiments/lolo1-wp5/conflict-root-manifest.json
```

Manifest artifact: `experiments/lolo1-wp5/conflict-root-manifest.json`
(gitignored, like the run telemetry it derives from). Manifest content
digest (`digest_sha256`, canonical-JSON over the manifest body):
`f3d240384044242599599c39c72134562fb29a9e94fd4555ee90d8fc3652e1a7`; file
sha256 `67af4a669201c0dc0de9e6437cfac654452223ffd075ff75bd327e7f57fc9bfa`.
Record store re-verified at mining time: sha `cf01a67a…` (§6.8 value),
3 records, content signatures byte-equal to the values every v327/v328 run
logged at load (`15604cb5…`/`37ea410d…`/`47975c94…`), root designation
`prepush-root-empty-track-unmatchable`.

### 11.1 Corpus and verification

v322–v328 (7 runs): 489,521 events walked; 401 archived candidates (option
+ frontier archives; causal-outcome entries tracked but excluded from the
scored universe per the recorded `archive_size` evidence); 16 candidates
map to certified records (the four removal-class `85fd9014…` states in
each of v323/v324/v327/v328); 19 restore-selection instants; 74 decision
points evaluated (every restore instant + every committed-decision
boundary). Instrument checks all pass: the offline re-scorer reproduces
v328's recorded treatment bonuses exactly at all three restores (+25.0 /
`baseline` at d2; 0.0 / `mapped` at d5 and d8), and `archive_size`
reconciles exactly at every instant that decides the result below
(including the decision-2 seam: 13 = 13, 4 bonus-positive candidates);
late d5/d8 instants show ±1–2 entry discrepancies from unnamed add
streams, disclosed in the manifest and not load-bearing.

### 11.2 Result: NO organic conflict roots

**Zero conflicts, all three families** (novelty-decoy 0, post-exploit 0,
exhaustion 0). The §4.43 learning now quantified over every recorded
decision point: wherever a certified-improving candidate coexisted with a
scoring current side (v323/v324/v327/v328, d1–d2), the baseline argmax was
*itself* the certified removal-class branch (`state-00012257`, 53.65); at
every other instant no live candidate mapped to any record (v322: pushed
root, all candidate signatures uncertified; v325/v326: post-removal
accumulated-track signatures, none certified) or the current side had
already acquired the removal signature, zeroing every bonus. Four
*near-conflict observations* are recorded: at the d8 restores of
v323/v324/v327/v328 the in-run baseline selected the unmapped causal
branch `state-00012322` (restore-time 59.243) over the live certified
`state-00012258` — but the root was already mapped to the removal record,
so the bonus was structurally zero. This is the §7.7 boundary, and it is
exactly the shape the primary seeded root reconstructs at a scoring-live
root.

### 11.3 Seeded designs (§6 step 5 fallback — DISCLOSED AS CONSTRUCTED)

**Primary — archive-seeded root (novelty-decoy; E2's top-ranked root).**
No fabricated certification: real record store, real restorable states.
Root: the §6.1 pre-push staging
(`episodic_resume:entity-v318-room3-known-push-connected-mask-d2:decision-4`,
empty root signature ⇒ §6.8 baseline designation live). Archive seeded
with, from sibling runs:

| Role | State | Source | Signature | Baseline value | State sha256 (digest verified) |
| --- | --- | --- | --- | --- | --- |
| Decoy (neutral, higher baseline) | `state-00001391` | v325 (restore winner, seq 15054) | none (unmapped) | 54.6125 (restore-time) | `b9283ab1…` ✓ |
| Certified-improving | `state-00012257` | v323 (≡ v324/v327/v328) | `85fd9014d58deb42` | 52.3094 (restore-time; 53.65 add) | `5a3cf71c…` ✓ |

Arithmetic: baseline gap 2.3031 = minimum flipping bonus; provided
certified bonus +25.0 (17 cells + milestone ×8 vs the designated pre-push
baseline); flip margin 22.6969. Selection rule (fixed in the module):
maximize `min(gap, flip margin)` over all valid restorable pairs;
deterministic tie-break. VOID: at the staged root's first restore instant
the weight-0 argmax must be the decoy and the certified branch must score
exactly +25.0, else fix staging, disclose, rerun once (§7.1 VOID rule).
Cross-run restore-time values are a disclosed staging assumption the VOID
condition covers.

**Alternate — records-file variant** (the §6 step 5 "certify a coverage
record for a configuration the baseline underranks"), for use only if the
archive-seeding mechanism is unavailable: seeded record = the certified
removal envelope re-keyed to an underranked configuration, provenance
force-marked `SEEDED-CONSTRUCT` (it can never read as a measured
certification; discrimination staging only, never accessibility
evidence). Selected instants, both at existing restorable states:

- *novelty-decoy*: v325 d2 (seq 15054) — decoy `state-00001951` (27.8547,
  unmapped, sha `9d56adcb…` ✓) vs challenger `state-00001475` (13.7, sig
  `d532f20cbec33347`, sha `55864657…` ✓); gap 14.1547, flip margin
  10.8453; seeded-record content signature `4d6377f4…`; variant-file
  recipe sha `2155ed25…`.
- *post-exploit*: v325 d5 (seq 56852; milestone `(12,11)` collected at
  d4, so the seeded record's milestone component is spent-but-scoring —
  the milestone-vs-cells decomposition §6 step 4 asks for) — decoy
  `state-00001076` (15.5104) vs the same challenger; gap 1.8104, flip
  margin 23.1896.

**Exhaustion family: not constructible** from existing certified
evidence, honestly recorded per-instant in the manifest: the
goal-exhaustion instants (v325/v326 d8) carry a non-empty *unmapped*
current signature (structural refusal ⇒ every bonus zero), and the mapped
d8 instants resolve to the removal record itself, so any positive
challenger would require inventing cells beyond every certified envelope.
Deferred to the `(8,4)`/`(9,12)` continuation measurements.

### 11.4 Consequence for §7.1 E2

E2's root is the primary seeded design above (mining found no organic
root to rank ahead of it); the disclosure obligation of §6 step 5 is
hereby met before any arm launches. Everything else in §7.1 stands. The
E1/E2 preregistration addendum with exact digests and command lines
remains a separate, later artifact (§10 ordering), and §9.2's `6a8488a`
merge remains a precondition — note the mining also shows why: the only
argmax flips the predicate ever produced on raw telemetry were
instrument-gap artifacts of the §4.29 signature reset, which the module
detects, flags, and disqualifies (they vanish under the corrected
candidate universe).

## 12. E1 preregistration — Gate 4 chain completion (appended 2026-08-17, BEFORE either arm runs)

Governs §7.1 E1 only. Written and committed to this document before
`entity-v330-…` or `entity-v331-…` executed a single emulator step. Nothing
in §12.1–§12.7 may be revised after launch; §12.8 is the results section,
written after both arms complete.

### 12.1 Budget derivation from evidence (§7.1's matched-budget requirement)

The design left `realization_branch_budget` declared but unjustified (§8
gives wall/event ceilings only). Before preregistering, the parameter was
traced to its consumption site and sized from the run that actually reached
the discriminator cell. All measurements below are read-only over stored
telemetry; zero emulator cost.

**(a) What the parameter actually is.** `realization_branch_budget` is
consumed at exactly one place in the monolith:

```
neural_planner.py:11180   relational_hold_slots = min(
                              self._relational_reach_cells_slot_budget(),   # = branch_budget
                              max(0, beam_width - len(retained_parent_ids)),
                              len(relational_hold_candidates))
```

It is a **per-depth-level parent-reserve slot count** inside the option
search's beam assembly — *not* a count of verified branches. It applies only
when (i) `relational_planner_authority == "selection"`
(`_relational_reach_cells_slot_budget:19801` returns 0 otherwise) and (ii)
the active hypothesis's realization kind is `reach_cells_under_hold` (hold
or exploit). The candidate set is built by
`_relational_hold_reserve_candidates:19812`: option nodes whose
`tracked_world_state_signature` equals the held signature, no life change,
no dark transition, **one representative per distinct target player slot**,
ranked by grid distance to the objective's target cells.

The other two realizations consume no branch budget at all:
`restore_archive` acts through `_relational_restore_preference:19767` (a 0/1
key that *leads* the untouched restore key at `:27079`), and
`reproduce_transition` through a stable partition of entity-frontier
candidates (`:11134`). `branch_budget` additionally enters the hypothesis
score only as the subtractive `search_cost = search_cost_per_branch ×
branch_budget = 0.001 × budget` (`relational_planner.py:762`).

Consequence: the effective per-level slot count is
`min(budget, beam residual, candidate supply)`. The budget can only bind
when it is *below* the candidate supply.

**(b) What the traverse that reached `(12,11)` actually cost.** Measured over
`entity-v325-room3-object-removed-probe-d12` (9,691 verified branches,
69,809 events):

| Fact | Value |
| --- | --- |
| Decisions that ran an option search | d1 (1,947 branches), d5 (6,028), d8 (1,716) — no others |
| First branch reaching pixel `(192,176)` = cell `(12,11)` | global branch **1,389**, `branch_index` 1,389, **depth 10** (max depth 12), seq 6,463, already carrying `human_prior_collected_heart_slots: [[192,176]]` |
| `human_prior_option_milestone_settled` for that slot | 2 events, both decision 1 (seqs 12,079 / 12,098), action paths of length 11 and 12 |
| Committed traverse | d2 = `archive_branch_restored` (`branches_examined: 0`) landing at `(192,160)` = `(12,10)`; d3 = commit at `(192,176)`, `branches_examined: 7`; d4 = collect `[[192,176]]`, `branches_examined: 7` |
| Verified branches consumed by d2–d4 | **0** — `total branches` is 1,947 at d1's commit and still 1,947 at d4's |

So the traverse cost **1,389 verified branches to first reach the region**,
inside a 1,947-branch decision-1 search, and **zero** additional branches
across the three committed decisions that executed it — the committed
traverse was a replay of options the decision-1 search had already found.

**(c) The supply ceiling that actually binds.** `len(representatives)` in
`_relational_hold_reserve_candidates` equals the number of distinct
non-fatal target player slots at a beam level sharing the held signature.
Measured directly:

| Corpus | Per-level distinct hold-matching representatives |
| --- | --- |
| v325 d1 (the search that reached `(12,11)`), depths 1→12 | 3, 5, 7, 9, 11, 10, 11, 13, 13, 11, 7, 5 — **max 13** |
| v327 (E1 root), removal signature `85fd9014d58deb42` | present only at d1 depth 11 (4 branches / **2** reps) and depth 12 (19 / **3**) |
| v329 (E1 root, shadow) | d1 depth 12: 19 branches / **3** reps |

**(d) Derivation, and the values fixed for E1.**

- **`relational_exploit_budget = 48`** (the `NeuralPlanningConfig` default,
  the value v329 logged). Derivation: the maximum hold-matching
  representative supply ever observed at a beam level is 13 (v325's
  reaching search); 48 is **3.69× that ceiling**, and 16× the supply
  observed at the E1 root itself. Because the effective slot count is
  `min(budget, residual, supply)`, 48 is already non-binding everywhere in
  the corpus and any larger value is provably inert. Sizing it as a
  multiple of the 1,389-branch traverse cost would be a **unit error** —
  branches are not beam slots, and the term is hard-capped by
  `beam_width = 128` regardless. The evidence therefore *confirms* the
  shadow value rather than changing it; it is fixed here so the
  confirmation is on the record before the run.
- **`relational_hold_budget = 8`.** Same unit; ≥2× the 3-representative
  supply at the E1 root. The hold objective carries an empty `target_cells`
  tuple (`relational_planner.py:1109`), so its ranking is distance-free and
  extra slots buy nothing beyond supply coverage.
- **`relational_establish_budget = 48`.** Structurally inert: the establish
  realization is `restore_archive`, which consumes no slots. Its only
  effect is `search_cost = 0.048` inside the hypothesis score. Left at the
  shadow value so establish and exploit carry an identical search-cost
  term and the chain ordering is unchanged.
- **`relational_decision_budget = 4`.** Positive evidence from v325: the
  committed traverse from the removal configuration to collection spanned
  **3** committed decisions (d2 restore → d3 reach → d4 collect); 4 covers
  it with one decision of margin. Explicitly **not** derived from v329's
  `budget_exhausted` at d7 — per learnings §4.44 that termination is the
  expected telemetry-mode null and carries no information about budget
  sizing.
- **`relational_max_queue = 4`** — unchanged; v329 proposed 2 chains total.

None of these are CLI-settable (`neural_run.py` exposes only
`--relational-planner-authority`); they are the module defaults, so the run
commands in §12.3 realize exactly the values derived here. Both arms carry
identical values; the control never reads them (`authority = off`).

**(e) A preregistered power limitation, disclosed now, not after scoring.**
The reserve family that consumes `exploit_branch_budget` executes only
inside an option search. At the E1 root every conforming run to date —
v327, v328, v329 — ran the option search **exactly once, at decision 1**
(12,232 verified branches), then emitted `human_prior_option_search_deferred`
at decisions 2, 5 and 8 and committed d2–d8 by restore/replay with
`branches_examined` of 0 or 7. In v329 the exploit hypothesis was active
d3–d7, a span containing **zero** option searches. If the treatment
reproduces that search geometry, the exploit's realization seam has no
execution opportunity and the only relational lever with authority is the
d2 restore preference (active only while an `establish` hypothesis with
`restore_archive` realization is live). This is recorded as the named
mechanism most likely to produce a FAIL, and as the reason the budget
derivation above cannot be rescued by a larger number. Selection authority
may still change the d2 restore and hence the downstream search geometry —
that is precisely the untested question — so this is a power caveat, not a
prediction, and it does not alter any bit below.

### 12.2 Arms, root, and matched budget

- **Control** — `--relational-planner-authority off`; run id
  `entity-v330-room3-e1-control-off-d12`.
- **Treatment** — `--relational-planner-authority selection`; run id
  `entity-v331-room3-e1-treatment-selection-d12`.
- The arms differ in **exactly two** things: that flag and `--run-id`. Every
  other flag, file, and digest is byte-identical, and both arms load the
  certified record store (`--human-prior-accessibility-records`) with
  `--human-prior-accessibility-preference-weight 0.0`, so the WP8-lite
  preference term is **off in both arms** and any behavioural difference is
  attributable to the relational layer alone.
- **Root** (identical to §6.1 of the WP8-lite design; all digests
  re-verified 2026-08-17 against the files on disk):
  memory `entity-v318-room3-known-push-connected-mask-d2` decision 1 with
  `--resume-option-search`; physical state the same run's **seq-2026**
  checkpoint (`goal_milestone_checkpoint_snapshot_stored`, decision 1,
  `state-00000002`, state sha256
  `33addc6c7c6828bf13d35ed0666ce7712647a8b614a12e343e96ff87ddcbfb92`;
  source `events.jsonl` sha256
  `0bbe1d1571d2d9d02b03e51816acc07a7945ba97256ec6e710ff88c7179b6f83`).
- **Input digests** (re-verified 2026-08-17, all equal to the v322–v329
  manifests): host `c03694c5…3e891f3`, core `a3450a09…5a40024886`, ROM
  `914c6769…3efd059e01`, neural checkpoint `bb7a7a37…284f678b9`,
  entity-behavior checkpoint `984b83c3…25c7c6aa`, record store
  `cf01a67aca2b6e8feeab38c0c85520dec2470cba2a5f2257cd817912c204d1fe`.
- **Matched budget**: `--decisions 8`; wall ceiling **10,800 s per arm**
  under an external watchdog; one native run at a time, control first; no
  depth/beam escalation; no rerun on an identical negative result. Observed
  envelope from this root: 12,232 verified branches, ~30 min, ~79.5k events
  per run (v327/v328/v329); event expectation ≤ 200k/arm, overrun reported.
- **Scoring window**: each arm's first **10,000**
  `human_prior_option_branch_verified` events, applied under the §7.2
  ruling of the WP8-lite design, restated and inherited verbatim:
  branch-level quantities are hard-truncated at branch 10,000, and the
  run's decision-level restore/commit events are in scope **provided the
  branch in question was verified within the first 10,000 branches**. The
  strict wall-order reading is rejected for the reason given there — at
  this root every restore postdates branch 10,000, so it would make the
  design self-voiding by construction.

### 12.3 Exact command lines (both arms)

```
.venv/bin/python -m lolo_agent.neural_run \
  --host build/lolo-libretro-host \
  --core "/Users/toddsherman/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --rom "Adventures of Lolo.nes" \
  --checkpoint experiments/platform-benchmarks/m5-real-data-training-sample.pt \
  --log-root experiments/lolo1-entity-v10/evaluations \
  --run-id <ARM RUN ID> \
  --decisions 8 \
  --action-durations 1,2,4,8,16 \
  --verify-actions 7 \
  --archive-capacity 1024 \
  --archive-max-age 2048 \
  --behavioral-best-first-archive \
  --behavioral-edge-coverage-weight 4.0 \
  --human-prior-hearts \
  --human-prior-heart-reward 25.0 \
  --human-prior-all-hearts-reward 75.0 \
  --human-prior-chest-reward 100.0 \
  --human-prior-life-loss-penalty 100.0 \
  --human-prior-best-first-archive \
  --human-prior-episodic-graph-guidance \
  --human-prior-goal-exhaustion-frontier-budget 32 \
  --human-prior-goal-exhaustion-rollback \
  --human-prior-graph-stagnation-visits 1 \
  --human-prior-navigation-recovery-grace 2 \
  --human-prior-option-archive-representatives 80 \
  --human-prior-option-causal-effect-frontier \
  --human-prior-option-effect-controllability-depth 2 \
  --human-prior-option-effect-frontier \
  --human-prior-option-effect-local-controls \
  --human-prior-option-effect-phase-offsets 3 \
  --human-prior-option-effect-probe-limit 16 \
  --human-prior-option-effect-stability-steps 3 \
  --human-prior-option-entity-curiosity-reserve 32 \
  --human-prior-option-entity-curiosity-weight 8.0 \
  --human-prior-option-entity-frontier \
  --human-prior-option-entity-inert-penalty-weight 1.0 \
  --human-prior-option-search-action-frames 16 \
  --human-prior-option-search-beam-width 128 \
  --human-prior-option-search-depth 12 \
  --human-prior-option-search-goal-proximity-reserve 12 \
  --human-prior-option-search-goal-world-state-reserve 12 \
  --human-prior-option-search-long-direction-frames 8 \
  --human-prior-option-search-milestone-reserve 32 \
  --human-prior-option-search-missing-player-reserve 4 \
  --human-prior-option-search-position-reserve 16 \
  --human-prior-option-search-stationary-history 2 \
  --human-prior-option-search-world-state-reserve 32 \
  --human-prior-phase-position-novelty \
  --human-prior-proactive-entity-probe-limit 16 \
  --anonymous-entity-behavior-checkpoint experiments/lolo1-entity-v10/anonymous-behavior-relational-v2-clean.json \
  --anonymous-entity-behavior-mode frozen \
  --resume-run experiments/lolo1-entity-v10/evaluations/entity-v318-room3-known-push-connected-mask-d2 \
  --resume-decision 1 \
  --resume-option-search \
  --resume-state-run experiments/lolo1-entity-v10/evaluations/entity-v318-room3-known-push-connected-mask-d2 \
  --resume-state-checkpoint-event-seq 2026 \
  --human-prior-accessibility-records experiments/lolo1-wp5/wp8lite-accessibility-records.json \
  --human-prior-accessibility-preference-weight 0.0 \
  --relational-planner-authority <off | selection>
```

This is v329's flag profile with `telemetry` replaced by the arm value.

### 12.4 The three preregistered bits (fixed; ANY mixed outcome = FAIL)

1. **Deliberate chain.** The treatment emits, in order and with the full
   `relational_hypothesis_*` score decomposition present at each step:
   `relational_hypothesis_proposed` carrying an `establish` → `hold` →
   `exploit` chain with `chain_parent_id` linkage, **before decision 2**;
   `relational_hypothesis_realized` on the removal-class restore; and
   `relational_hypothesis_achieved` on the **exploit** hypothesis. The
   proposal must be logged **before** the realization that collects — i.e.
   the `relational_hypothesis_proposed` event carrying the exploit's
   `hypothesis_id` must precede, in sequence order, the
   `decision_committed` event that collects the milestone.
2. **Chained consequence.** Within the window, the treatment's committed
   trajectory collects the milestone heart at cell `(12,11)` / slot
   `(192,176)` — evidenced by `[192,176]` entering
   `human_prior_collected_heart_slots` on a `decision_committed` event —
   and the control does not. If both collect it, the treatment must do so
   at a **strictly earlier** decision index. The metric is the milestone
   cell only, never affordance counts. Recorded precedent: neither v324,
   v327, v328 nor v329 collected it in-window from this root.
3. **No safety regression.** The treatment records no more
   `human_prior_life_loss_confirmed` committed decisions than the control
   within the window.

All three must pass. **ANY mixed outcome = FAIL.** No weight tuning, no
budget re-sizing, no rerun on an identical negative result. Per §7.1, a
FAIL keeps WP8 engineering-only and makes the next move a representation
question (WP2/WP3 integration depth), not a parameter question.

### 12.5 VOID conditions (WP8-lite precedent; a VOID is not evidence)

VOID — disclosed defect, fix the staging, disclose, rerun once:

1. **Config inequality.** Either arm's manifest `planning_config` differs
   from the other's in any field except `relational_planner_authority`
   (and the derived `relational_planner_enabled`), or either differs from
   v329's `planning_config` in any field except those two.
2. **Records inequality.** The two arms' `verified_accessibility_records_loaded`
   events do not both report `record_count: 3` with content signatures
   `15604cb5…` / `37ea410d…` / `47975c94…` and store digest
   `cf01a67a…`, or the two arms' record digests differ from each other.
3. **Seeding defect.** No archived branch carrying the removal-class
   signature `85fd9014d58deb42` exists within the window in **either**
   arm — the choice the chain depends on never materialises.
4. **Root defect.** Either manifest's `episodic_resume` block does not
   record source run `entity-v318-room3-known-push-connected-mask-d2`,
   `source_decision: 1`, `state_source_checkpoint_event_seq: 2026`, and
   `state_source_events_sha256: 0bbe1d15…`.
5. **Budget defect.** Either arm exceeds the 10,800 s wall ceiling and is
   killed by the watchdog before `run_finished`, or either arm's window is
   starved (fewer than 10,000 verified branches).

Budget-exhausted non-reach is **censored**, never reported as
"unreachable" (learnings §2, §4.14).

### 12.6 Scoring procedure and determinism

A single deterministic scorer walks each arm's `events.jsonl` once,
truncates branch-level quantities at branch 10,000, applies §12.4 verbatim,
and writes `experiments/lolo1-wp5/e1-gate4-report.json` with a
canonical-JSON `digest_sha256` over the report body. The scorer is run
**twice** end to end; both runs must produce byte-identical reports, and
the digest is recorded in §12.8. The scorer reads only telemetry; it
contains no arm-specific branches beyond the run ids.

### 12.7 Ownership and what this preregistration does not claim

This section changes no code. `neural_planner.py`, `relational_planner.py`
and `tmp/` are untouched by the E1 lane. Nothing here claims a strict-track
result: the certified records are assisted-lineage (§9.5), so a PASS would
close Gate 4 **on the assisted track only**. E2 remains a separate,
later experiment against the §11.3 seeded root and is not scored here.

### 12.8 E1 results — **FAIL** (scored 2026-08-17, after both arms completed)

Both arms ran to completion, one at a time, control first, under the §12.2
watchdog. Neither approached the wall ceiling.

| | Control `entity-v330-room3-e1-control-off-d12` | Treatment `entity-v331-room3-e1-treatment-selection-d12` |
| --- | --- | --- |
| Authority | `off` | `selection` |
| Wall clock | 1,812 s (ceiling 10,800 s) | 1,822 s |
| Events | 79,477 (= v327/v328 exactly) | 79,493 (= v329 exactly) |
| Verified branches | 12,232, all in the decision-1 search | 12,232, all in the decision-1 search |
| Option searches | started d1, completed d1; **deferred at d2, d5, d8** | identical |
| `frozen_evaluation_audit` | pass | pass |

Report: `experiments/lolo1-wp5/e1-gate4-report.json`, body digest
`6b6708dbdb53e9d20e1d9d823689049edcc4f036a08d1968b88a1cff630f138e`, file
sha256 `a2295221ef746714b983590cac25bc162063e8b1e4f294f26b53e492105b0c65`.
The scorer was run three times end to end; all three reports are
byte-identical (§12.6 satisfied).

**No VOID condition fired.** V1: the arms' `planning_config` differ in
exactly `relational_planner_authority` and `relational_planner_enabled`
(125 fields each), and each differs from v329's only within that same set.
V2: both arms loaded `record_count: 3` with content signatures
`15604cb5…`/`37ea410d…`/`47975c94…` at
`verified_accessibility_weight: 0.0`. V3: 4 removal-signature branches
verified in-window in each arm, first at branch index **9,419**, and 4
matching `human_prior_option_archive_added` events in each — the choice
materialised. V4: both manifests record the v318 source run, decision 1,
checkpoint seq 2026, events sha `0bbe1d15…`. V5: both runs finished; 12,232
branches ≥ the 10,000 window; 79.5k events ≤ 200k.

**Bit 1 — deliberate chain: FAIL** (two clauses pass, one fails).

- *Passes*: `relational_hypothesis_proposed` at seq 75,243, **decision 1**
  (before decision 2), queue `establish → hold → exploit` with correct
  `chain_parent_id` linkage and the complete
  `relational_hypothesis_*` decomposition on all three;
  `relational_hypothesis_realized` on the removal-class restore at seq
  76,564, decision 2, decomposition complete.
- *Fails*: **no `relational_hypothesis_achieved` for the exploit
  hypothesis.** The exploit activated at decision 3 (seq 77,195) and
  terminated `budget_exhausted` at decision 7 (seq 79,005). A second
  hold/exploit pair was proposed at decision 8 (option reuse at seq 79,013)
  and did not resolve before the run ended.

**Bit 2 — chained consequence: FAIL.** Neither arm collected the `(12,11)` /
`(192,176)` milestone heart within the window. Both arms collected exactly
the same two hearts at the same decisions: `(96,128)` at d1 and `(128,128)`
at d3.

**Bit 3 — no safety regression: PASS.** Zero
`human_prior_life_loss_confirmed` committed decisions in both arms.

**Verdict: FAIL** (§12.4: any mixed outcome = FAIL). No tuning, no rerun.

### 12.9 The mechanism, named

The two arms' committed trajectories are **identical**, state id by state
id: d1 `state-00012280`, d2 `state-00012256`, d3 `state-00012294`, d4
`state-00012305`, d5 `state-00012257`, d6 `state-00012317`, d7/d8
`state-00012322`; restores at d2/d5/d8 to `12256`/`12257`/`12322` in both
arms; identical actions, plans and committed scores in both console logs.
The treatment's telemetry differs from the control's by exactly **16
events** — precisely its 16 `relational_*` events — and its total event
count equals the telemetry-mode shadow run's to the event. Selection
authority changed nothing.

Two independent reasons, both now measured rather than inferred:

1. **The exploit's realization seam never had an execution opportunity.**
   `reach_cells_under_hold` consumes its budget only inside the
   option-search beam assembly (`neural_planner.py:11180`). The treatment
   ran the option search exactly once, at decision 1, and deferred it at
   decisions 2, 5 and 8. The exploit was active from decision 3 to decision
   7 — a span containing **zero** option searches. Its 48-slot budget was
   never consulted once. This is exactly the power limitation preregistered
   in §12.1(e), and it means the `budget_exhausted` termination is a
   *seam-opportunity* fact, not a budget-size fact: no value of
   `relational_exploit_budget`, and no value of
   `relational_decision_budget` within an 8-decision run, could have
   changed this outcome.
2. **The one seam that did fire was redundant.** The establish
   hypothesis's `restore_archive` realization is the only relational lever
   with authority at a restore, and it is active only while an establish
   hypothesis is live — i.e. at decision 2 alone (at d5 and d8 the active
   hypothesis is exploit/hold, so `_relational_restore_preference_active`
   is false and restore selection is bit-identical by construction). At
   decision 2 the preference was exercised and the baseline picked the
   same branch anyway: `state-00012256` carries the removal signature
   `85fd9014d58deb42`, so the hypothesis preference and the plain frontier
   key agree. This is learnings §4.43's redundancy finding reproduced one
   level up — the chain layer inherits it because its only authoritative
   restore is the same restore WP8-lite already showed to be
   non-discriminating.

This is a genuinely new negative, not a repeat of §4.44. The shadow run's
`budget_exhausted` was uninformative by construction (zero authority). Here
the exploit *held* selection authority for five decisions and still could
not act, because the authority it holds is expressed through a seam the
planner's own search schedule never opened. The chain machinery is
confirmed correct a second time on native state (proposal before execution,
linkage, realization on the restore, hold achieved at d3, option storage
and reuse at d8) — Gate 4's "hypothesis logged before execution" criterion
is mechanically satisfied — but Gate 4's *consequence* criteria remain
open, and the gap is not in the hypothesis layer's scoring.

### 12.10 Consequence for the plan (no tuning, no rerun)

Per §7.1's outcome rule, a FAIL keeps WP8 engineering-only and makes the
next move a representation/integration question, not a parameter question.
The specific, evidence-named next questions, in the order the evidence
supports:

1. **Search-schedule coupling, not budget size.** A hypothesis with
   selection authority cannot steer a planner that does not search while it
   is active. Whether an active `reach_cells_under_hold` objective should
   be able to *request* an option search — rather than only re-rank the
   parents of a search the stagnation machinery independently decides to
   run — is a seam-design question for §4 step 4, and it is the single
   change this result identifies. It must be designed and preregistered on
   its own bits, not slipped in as a rerun of E1.
2. **E2 is unaffected and remains the right next experiment.** Its
   discriminator is a *restore-instant* disagreement at the §11.3 seeded
   conflict root, which the decision-2 seam does reach; E1's failure mode
   is about decisions 3–7, where E2 places no weight.
3. **Do not re-size `relational_exploit_budget`.** §12.1 fixed it from
   evidence and §12.9 shows it was never read. Changing it would be tuning
   against a parameter the run proved inert.
