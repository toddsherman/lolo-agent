# WP8 objective-driven search scheduling — design (2026-08-17 evening)

Status: DESIGN ONLY — no code changed by this document. It answers the
question roadmap §19 item 1 raises and enumerates the seams, the mechanism
choice, the test plan, and an E3 preregistration skeleton. Nothing here is
evidence; the measurements below are read-only re-derivations from stored
telemetry (zero emulator cost) and are labelled as such.
Authority: roadmap §19 (Amendment — 2026-08-17 evening), learnings §4.45,
`docs/wp8-relational-planner-design-2026-08-17.md` §12.10 item 1.
Evidence base: runs v323–v331 in
`experiments/lolo1-entity-v10/evaluations/`; certified record store
`experiments/lolo1-wp5/wp8lite-accessibility-records.json` (sha
`cf01a67a…`); the E1 report `experiments/lolo1-wp5/e1-gate4-report.json`
(body digest `6b6708db…`).
Ownership: this document creates nothing else. `neural_planner.py`,
`relational_planner.py`, `tests/` and `tmp/` are untouched by this lane.

---

## 1. The problem, restated in the terms the evidence supports

E1 failed with an architectural cause (learnings §4.45): the relational
planner **can rank what an option search offers, but cannot cause a search,
and has no lever on the decisions that commit without searching.** The
exploit hypothesis held selection authority across decisions 3–7 and those
five decisions contained zero option searches, so its only seam
(`_relational_hold_reserve_candidates`, consumed at
`neural_planner.py:11180`) never executed.

Two prior increments are now twice-measured as insufficient, and neither
may be re-tried:

- §4.43 — a verified-accessibility **restore scalar** was behaviorally
  redundant (novelty already preferred the same branch);
- §4.45 — a hypothesis-level **reserve preference** was inert because the
  seam never opened.

So the next increment must supply *schedule or direction* authority, not a
third preference weight on a candidate the baseline already ranks. §5 makes
that requirement testable rather than assumed.

---

## 2. Q1 — When does an option search actually happen today?

### 2.1 Every trigger and gate, enumerated

There are exactly **two** entry points into
`_search_human_prior_options` (`neural_planner.py:8950`).

**Entry A — the stagnation path (the only planner-initiated one).**
Inside `decide()` (`:20612`), reached in this order:

| # | Gate | Site | Effect when it holds |
| --- | --- | --- | --- |
| A0 | life-loss recovery returns a decision | `:20636` (`_restore_after_life_loss:25400`) | `decide()` returns; no search, no relational propose |
| A1 | relational propose-and-log | `:20641` (`_relational_propose_and_log:19599`) | logs only; never searches |
| A2 | proactive entity probe added archive branches | `:20642–20672` | arms `_restore_if_stagnant`; may return before any search |
| A3 | `human_prior_graph_repeated` = best-first archive on **and** `human_prior_graph_stagnation_visits > 0` **and** a non-empty goal-graph signature **and** visits ≥ limit | `:20691` | necessary precondition |
| A4 | `navigation_recovery_grace_active` (`_human_prior_navigation_recovery_grace_active`, config `--human-prior-navigation-recovery-grace 2`) | `:20697`, suppression event emitted at `:20775` | **cancels A3**: `human_prior_graph_stagnant = False` (`:20746`), so the entire search block is skipped |
| A5 | `if human_prior_graph_stagnant:` | `:20807` | the only block containing a search call |
| A6 | goal-exhaustion rollback branch: `milestone_frontier_budget_exhausted` → either `goal_milestone_frontier_recovery_deferred` (sets `milestone_verified_route_search_due`) or `_restore_goal_milestone_after_exhaustion` returns a decision | `:20823–20903` | can return before the search gate |
| A7 | **the deferral gate**: `option_search_depth >= 2` **and not** `milestone_verified_route_search_due` **and** (`unvisited_archive_endpoints > 0` **or** `global_archive_frontiers > 0`) | `:20943–20950` | emits `human_prior_option_search_deferred` (`:20951`) and **does not search** |
| A8 | else | `:20969–20974` | `self._search_human_prior_options()` — **the only in-`decide()` search call** (`:20971`) |
| A9 | `option_search_exhausted = (added == 0)` | `:20975` | feeds the goal-exhaustion rollback at `:21046` and the egress filter at `:22616` |

`unvisited_archive_endpoints` / `global_archive_frontiers` come from
`_human_prior_unvisited_archive_endpoints` and
`_human_prior_archive_frontier_flags` (`:4302`) over the non-regressive
archive (`:20904–20942`).

**Entry B — the explicit forced audit.**
`search_human_prior_options_from_current_state(*, reason)` (`:8938`) is a
public method that calls `_search_human_prior_options(force_reopen_reason=
reason)`. Its only caller today is `neural_run.py:2666` under
`--resume-option-search`, emitting `episodic_resume_option_search_completed`
at decision 0. **This is the seam mechanism (a) would use — it already
exists and is already exercised in every run in the corpus.**

**Gates inside `_search_human_prior_options` itself:**

| Gate | Site | Effect |
| --- | --- | --- |
| `goal_prior is None` or `option_search_depth < 2` | `:8961` | return 0 |
| empty action set | `:8987` | return 0 |
| empty `source_signature` | `:9083` | return 0 |
| **exhaustion cache**: `exhausted_key = (source_signature, search_budget)` in `human_prior_option_exhausted_sources` **and** `force_reopen_reason is None` | `:9236–9242` | emits `human_prior_option_search_skipped` reason `source_already_exhausted`, return 0 |
| same key, **with** `force_reopen_reason` | `:9258` | emits `human_prior_option_search_reopened` with the caller's reason, and proceeds |
| same signature, different budget tuple | `:9279` | `human_prior_option_search_reopened` reason `search_budget_changed` |

Cache writes: `:12378`, `:12445`, `:12483`, `:12577`; resets at `:13895`,
`:14417`, `:25373`. `search_budget` is a 30-field tuple (`:9013–9075`) with
a sha recorded per search — so a config change reopens a source, but nothing
in the *planner* can reopen one except `force_reopen_reason`.

### 2.2 Quantified from telemetry (read-only; zero emulator cost)

Searches per run, with the wall cost of each phase
(`elapsed_ms` deltas from `events.jsonl`):

| Run | Root | Verified branches | Resume audit (entry B) | Planner-initiated searches (entry A) | Deferrals |
| --- | --- | --- | --- | --- | --- |
| v325 | post-removal probe | 9,691 | d1: 1,947 br, **320 s**, +80 archive | **d5** (6,028 br, **730 s**, +32), **d8** (**200 s**, +0, `no_unexpanded_endpoint`) | d2 (`global_semantic_archive_frontier_available`, 79 frontiers) |
| v326 | ≡ v325 | 9,691 | identical | identical | identical |
| v327 | v318 pre-push | 12,232 | d1: 12,232 br, **1,594 s**, +13 archive | **none** | d2 (9), d5 (4), d8 (3) — all `global_semantic_archive_frontier_available` |
| v328 | ≡ v327 | 12,232 | identical | **none** | identical |
| v329 (shadow) | ≡ v327 | 12,232 | identical | **none** | identical |
| v330 (E1 control) | ≡ v327 | 12,232 | identical | **none** | identical |
| v331 (E1 treatment) | ≡ v327 | 12,232 | identical | **none** | identical |

Decisions **d3, d4, d6, d7** in every one of these runs never reach the
search block at all: they emit `human_prior_graph_recovery_suppressed`
reason `navigation_recovery_grace` — gate A4 cancels A3.

**The sharpest statement of the E1 mechanism, measured:** at the E1 root, in
five conforming runs (v327–v331), the planner's own stagnation machinery
decided to search **zero times**. The single search in each run is a fixture
of the resume protocol (entry B, decision 0) and it runs *before the
enabling configuration exists* — the removal-class configuration is only
established at d2, by restore.

**Why the deferral gate is effectively self-suppressing.** Every committed
decision adds 2–4 archive branches (measured in v331 at d3/d4/d6/d7), and
gate A7 defers whenever *any* archive branch carries *any* frontier flag.
`global_archive_frontiers` was 9 / 4 / 3 at the three stagnation instants —
never 0. In a short run the planner's own archive growth guarantees the
deferral. This is a structural property of the schedule, not a tuning
accident.

**Contrast: how v325 actually reached `(12,11)`.** v325's committed
traverse, cell by cell (`human_prior_source/target_player_slot` on
`decision_committed`, ÷16):

```
d1 (7,6)→(6,6)   d2 restore →(12,10)   d3 (12,10)→(12,11)   d4 collect (192,176)
```

The d2 restore *jumped* into the eastern region (its archive contained a
branch there because that run's resume audit ran from the **post-removal**
configuration and had depth to spare). The last two steps to the milestone
were **ordinary committed decisions, not searches** — `branches_examined: 7`
at d3 and d4. That fact is the entire argument for mechanism (b).

---

## 3. Q2 — The non-search commit path, and what an objective may steer

### 3.1 Enumeration of commit paths, with seams

| # | Path | Entry / seam | Steerable by an active objective? |
| --- | --- | --- | --- |
| P1 | **Life-loss recovery restore** | `_restore_after_life_loss:25400`, called at `:20636` | **No.** Safety recovery; must stay hypothesis-blind. |
| P2 | **Proactive entity-probe restore** | `:20642–20672` → `_restore_if_stagnant` | Only via P3's key. |
| P3 | **Stagnation restore** | `_restore_if_stagnant:25773`; eligibility/keys built `:27020–27059`; **the existing WP8 relational seam** at `:27079–27084`; `branch = max(restore_eligible, key=restore_key)` at `:27085` | **Yes** — the seam already exists and today expresses only a 0/1 configuration preference (`_relational_restore_preference:19767`). |
| P4 | **Goal-exhaustion rollback restore** | `_restore_goal_milestone_after_exhaustion:25217`, called `:20896` / `:21072` | In principle; deliberately out of scope (it is the §4.14 machinery). |
| P5 | **Direct verification commit** (the 7-branch path; `branches_examined: 7`) | verify loop `:21260–21304`; candidate families `:22656–22790`; the **choice ladder** `:23180–23500` | **Yes — this is the untouched axis.** See §3.2. |
| P6 | **Learned-route replay / control-frontier waypoints** | `_human_prior_episodic_control_frontier_plan:3673`, `_human_prior_episodic_graph_plan:3784`; route controls consumed at `:9157–9176` and as replay parents at `:11138–11151` | **Search-only.** See §3.3 — this is the important negative. |

### 3.2 P5's ladder is where the agent actually moves, and it has no target

The commit ladder, in priority order, with construction sites:

1. `anticipated_observation_choice` (`:23180`)
2. `delayed_transition_choice` (`:22760`)
3. **`human_prior_goal_choice`** (`:22830`) — max over `positive_goal_branches` by `milestone_reward`
4. `human_prior_navigation_detour_choice` (`:22841`)
5. **`human_prior_semantic_frontier_choice`** (`:22850`) — max over `unvisited_semantic_frontiers` keyed on *position-visit novelty, graph-visit count, and unexpanded control actions*; emitted at `:23303` with reason `player_endpoint_needs_expansion`
6. **`known_goal_fallback_choice`** (`:22890`) — emitted at `:23342` as `human_prior_known_milestone_fallback`, reason `no_unvisited_semantic_frontier`
7. dynamic-control / control-intervention / causal-observation-wait / autonomous / autonomous-grace tiers (`:23352–23483`)
8. generic `max(selection_verified, key=(score, path))` (`:23497`)

**Not one tier in this ladder references a target cell.** The only
distance-to-goal terms in the planner live either inside the option search
(`_human_prior_visible_goal_distance:18111`,
`_human_prior_preparation_goal_distance:18167`, consumed at `:11044–11064`)
or in `_archive_frontier_score`'s `goal_navigation_bonus` (`:19967`), which
is gated on `goal_prior.navigation_reward > 0.0` — and
`human_prior_navigation_reward` defaults to **0.0** (`:178`) and is **not
set** by the E1 command line (§12.3 of the relational-planner design). It is
disabled deliberately, by the §4.7 decision.

Measured, in v331 (E1 treatment), the exploit's authority window:

| Decision | Selector (from telemetry) | Cell move | Grid distance to `(12,11)` |
| --- | --- | --- | --- |
| d3 | `human_prior_known_milestone_fallback` (`no_unvisited_semantic_frontier`) | (8,8)→(8,8), collects `(128,128)` | 7 |
| d4 | `human_prior_semantic_frontier_choice` (`player_endpoint_needs_expansion`) | (8,8)→**(9,8)** | **6** |
| d5 | `archive_branch_restored` (stagnation) → `state-00012257` | →(8,7) | 8 |
| d6 | `human_prior_semantic_frontier_choice` | (8,7)→(8,6) | 9 |
| d7 | `human_prior_semantic_frontier_choice` | (8,6)→(7,6) | 10 |
| d8 | `archive_branch_restored` → `state-00012322` | →(7,6) | 10 |

The exploit **had** a lever available on four of its five authority
decisions — it simply had no way to touch it. Its own d4 commit took the
first step of the certified path east, by novelty accident, and the d5
stagnation restore threw it back.

The certified path exists and is 4-connected inside the 24-cell envelope
(record `85fd9014d58deb42`):

```
(8,8) → (9,8) → (10,8) → (11,8) → (12,8) → (12,9) → (12,10) → (12,11)
```

Seven cells, all certified, at one cell per committed decision.

### 3.3 The control-frontier / waypoint machinery does **not** do the work

The prompt's hypothesis — that a `reach_cells_under_hold` objective might
map onto the existing control-frontier/waypoint machinery and need no search
— was checked and is **false as stated**:

- `_human_prior_episodic_graph_plan:3784` and
  `_human_prior_episodic_control_frontier_plan:3673` produce a
  `waypoint_signature` plus `route_controls`.
- Those `route_controls` are consumed **only** inside
  `_search_human_prior_options` (`:9157–9176`, replay-prefix parents at
  `:11138–11151`).
- At restore time the plan contributes only a *progress scalar*
  (`live_archive_episodic_graph_metrics:25898` → `_human_prior_episodic_graph_progress:3957`) — it ranks, it does not route.
- The learned preparation-goal slot (`_human_prior_active_preparation_goal_slot:18127`)
  is likewise consumed only at `:11042`, `:12696`, `:18797` and one
  milestone-retry gate at `:17647`.

So the waypoint machinery is a **search-side** router. On non-search
decisions nothing carries the agent anywhere; the semantic-frontier tier
wanders by novelty. A navigation objective therefore needs a *new tier in
the P5 ladder plus a target-aware P3 restore key* — not the waypoint
machinery, and not a search.

### 3.4 An instrument gap that caps any restore-side steering (new finding)

`_relational_archive_candidate_views:19478` skips any archive branch with an
empty `tracked_world_state_signature` (`:19485`), and so would any
hold-gated restore key. But the three **commit-time** `_ArchivedBranch`
constructions (`:24066`, `:24538`, `:24783`) never pass that field, so it
defaults to `""` (`_ArchivedBranch:257`, field at `:310`). Only the
option-search path (`:8543`, sets it at `:8610`) and the causal-archive
paths (`:12928` / `:16154`) carry it.

Measured consequence at the E1 root: the **only** archived branches carrying
`85fd9014d58deb42` are the four the d1 resume audit produced —

| state | committed cell |
| --- | --- |
| `state-00012257` | (8,7) |
| `state-00012256` | (8,8) |
| `state-00012258` | **(9,8)** |
| `state-00012255` | (8,8) |

— so hold-gated restore supply is capped at **(9,8)**, distance 6 from the
target, and it does not grow as the agent advances even though every
committed decision deposits 2–3 fresh branches one cell behind the agent
(measured at v331 d3/d4/d6/d7). This is the same *supply-ceiling-binds*
shape as §12.1(c), one level down, and it is a §4.29-class instrument gap.
Fixing it is step 0 of §4 below.

---

## 4. Q3 — Two candidate mechanisms, and the recommendation

### 4.1 Mechanism (a) — SEARCH REQUEST

An active hypothesis with an unrealized realization objective may request an
option search at the next decision.

- **Seam**: already exists —
  `search_human_prior_options_from_current_state(reason=…)` (`:8938`) with
  `force_reopen_reason`, which is precisely the exhaustion-cache override at
  `:9242`/`:9258`. In `decide()` the request would be consulted at the
  deferral gate (`:20943`) and, if the gate would defer, override it with a
  reason-coded `relational_search_requested` / `_granted` / `_denied` event.
- **Bounds needed**: max requests per run, cooldown in decisions, a
  cumulative branch/wall ceiling, and per-request grant/denial logging.
- **What it directly attacks**: §4.45 mechanism 1's *first* half — the
  planner never searches while a hypothesis is active.
- **Positive evidence**: v325's resume audit, run from the *post-removal*
  configuration, first reached `(12,11)` at global branch **1,389**, depth
  10, inside a 1,947-branch search. A search requested at v331's d3 would
  have the same geometry.

**Why it is not the first experiment.** It is **not budget-neutral**, and
the cost is measured: an ordinary non-search decision costs ~**12–17 s** and
~2,000 events; a search costs **320 s** (v325 d1), **730 s** (v325 d5),
**200 s** (v325 d8, exhausted), or **1,594 s** (v327–v331 d1). One granted
request is 15–120× a decision. Consequences:

1. A matched-budget paired ablation is broken by construction: the treatment
   would consume roughly twice the emulator work of the control, and no
   reviewer should accept a consequence bit under that asymmetry.
2. E1's scoring window ("first 10,000 verified branches", §12.2) inherits
   from a world where all branches come from one search. A second search
   redistributes branch indices across arms and makes the truncation
   non-comparable.
3. It changes the *schedule*, so archive contents, novelty counters and
   every later decision diverge — maximal blast radius, and squarely the
   roadmap §14 "broader search hides representation failures" risk.

**The honest fix, for when (a) is run (see §6.4, E4):** give the control a
**forced-search control arm** — an equal, un-requestable search at the same
decision index — so both arms pay identical emulator cost and the bit tests
*whether the objective directs the search*, not whether searching helps.

### 4.2 Mechanism (b) — NAVIGATION TARGET

An exploit hypothesis with a non-empty certified `target_cells` set
publishes those cells into the *non-search* decision path, so ordinary
navigation carries the agent there while the hold predicate holds.

Concretely, **two consumers of one already-existing objective** — no new
config flag, no new score term in the WP8-lite sense:

- **C1 — a new commit-ladder tier.** `relational_navigation_choice`,
  inserted between `human_prior_goal_choice` (`:22830`) and
  `human_prior_navigation_detour_choice` (`:22841`): select from
  `selection_verified` the non-fatal branch that **strictly reduces** grid
  distance to the objective's `target_cells`, ordered by (distance, then the
  existing keys). Empty ⇒ the ladder is unchanged (fail-open).
  Placement rationale: never override an actual milestone collection (d3's
  `(128,128)` must still be taken), but outrank pure position novelty.
- **C2 — a target-aware restore key**, inside the restore seam that already
  exists at `:27079`: when the active hypothesis's realization is
  `reach_cells_under_hold`, lead the key on
  `(hold-signature match, −distance to target cells)` instead of the current
  0/1 configuration preference. C2 is what makes progress **monotone** —
  without it, the d5/d8-class restores reset the ratchet exactly as measured
  in §3.2.

**Why (b) is smaller and more testable**:

- **Budget-neutral by construction**: it reorders branches that are verified
  anyway. Zero extra emulator steps, so matched budget is trivially
  satisfied and E1's config-equality VOID machinery is reusable verbatim.
- **Smaller seam**: one new ladder tier + one extension of an existing key,
  both `max(…, key=…)` sites, both gated off outside selection authority,
  both covered by the existing invariance test (§5 item 9 of the
  relational-planner design).
- **It attacks the half of §4.45 that covered 4 of the exploit's 5
  authority decisions.**
- **It has a graded observable** — cell distance to the target set, per
  committed decision — where E1's bits were all-or-nothing. That is a real
  power gain (§6.3).

### 4.3 Recommendation

**Do (b) first, as E3, with (a) designed now and preregistered later as
E4.** Rationale, in the order the evidence supports it:

1. v325 reached the milestone with **ordinary committed decisions** for its
   last two cells (§2.2). The non-search path demonstrably can carry the
   agent to `(12,11)`.
2. v331's own d4 took the first step of the certified path and the
   restore threw it back (§3.2) — the failure is *direction and retention*,
   not search coverage.
3. (b) is budget-neutral; (a) is not, and (a) run without a forced-search
   control would produce an uninterpretable result.
4. (b) is the smaller monolith touch under a realized §14 risk.

**But state the limits honestly**: (b) addresses only the *non-search
decisions* half of §4.45. It does **not** give the planner the ability to
cause a search, and it is capped by §3.4's restore supply until step 0
lands. If E3 fails on a *supply* rather than a *direction* ground, that is a
FAIL and the next move is (a) with a forced-search control — not a retune of
(b).

---

## 5. Seam enumeration and test plan

### 5.1 Seams (each ≤ ~20 lines, each grep-able, all authority-gated)

| Seam | Site | Change | Default-off proof |
| --- | --- | --- | --- |
| **S0** (instrument, step 0) | `:24066`, `:24538`, `:24783` | pass `self.current_human_prior_root_object_state.tracked_world_state_signature` into the commit-time `_ArchivedBranch` constructions | Behaviourally inert while `verified_accessibility_weight <= 0.0` (short-circuit at `:19374`) and while relational authority ≠ `selection`; **must** be proved by an executable invariance test at *both* weight settings, because `_archive_verified_accessibility_bonus` keys on this field |
| **S1** | new tier in the ladder between `:22830` and `:22841`; emit `relational_navigation_choice` | select the distance-reducing branch under the active objective | returns `None` unless `_relational_selection_authority()` and the active realization is `reach_cells_under_hold` with non-empty `target_cells` |
| **S2** | `:27079–27084` (existing seam) | extend the lead key to `(hold match, −distance)` for `reach_cells_under_hold` | same guard; `restore_key` untouched otherwise |
| **S3** | `relational_planner.py` | pure `navigation_preference(hypothesis, cell) -> tuple` + `objective_target_cells(hypothesis)`; no new state, no planner imports | pure-module test |
| **S4** | telemetry | `relational_navigation_choice` / `relational_navigation_declined` carrying: target cells, current cell, baseline-argmax candidate, objective-preferred candidate, **whether they differ**, and the full `relational_hypothesis_*` decomposition | additive only |

Deliberately **not** built: any new config weight; any re-enabling of
`human_prior_navigation_reward` (§7.3); any change to P1 (life-loss) or P4
(goal-exhaustion rollback); any change to the search schedule (that is E4).

### 5.2 Unit tests (no emulator)

1. `objective_target_cells` returns `()` for establish/hold objectives and
   the certified cells for exploit — regression twin of
   `_exploit_target_cells:991`.
2. Distance ranking is deterministic and grid-based; ties broken by the
   existing keys; identical inputs ⇒ byte-identical ordering.
3. Fail-open: no active hypothesis, non-`reach_cells_under_hold`
   realization, or empty `target_cells` ⇒ the tier returns `None` and the
   ladder is unchanged.
4. Hold gating: a branch flagged `life_counter_changed` or
   `dark_transition_started` is never selected, even if it reduces distance.
5. S2 key: with a hold-matching branch at distance 3 and a higher-baseline
   unmapped branch, the hold-matching branch wins **only** under selection
   authority.
6. **Exercised-difference telemetry**: when the baseline argmax and the
   objective-preferred candidate coincide, the event records
   `differs: false` — this is the redundancy detector §5.3 depends on.
7. S0: a commit-time archive branch now carries the root signature;
   `_relational_archive_candidate_views` sees it; and at
   `verified_accessibility_weight == 0.0` the frontier score is unchanged to
   the bit.
8. Seam invariance (in `tests/test_ensemble_planner.py`, alongside the
   existing planner tests): `relational_planner_authority != "selection"`
   ⇒ planner ranking, restore selection **and** commit selection are
   bit-identical to today.

### 5.3 The redundancy detector (mandatory, not optional)

§4.43 and §4.45 both failed because a lever agreed with the baseline. E3
must be able to detect that a third time, in-run, before scoring:
every S1/S2 instant logs the baseline argmax and the objective-preferred
candidate. If they **never differ** across the exploit's authority window,
the result is recorded as a third redundancy finding of the same family and
the plan moves to representation (WP2/WP3 integration depth), not to a
fourth lever.

---

## 6. Q4 — E3 preregistration skeleton

Nothing below is scored. The full preregistration addendum (exact digests,
command lines, scorer) lands in this document *before* any arm runs, per the
§10 ordering discipline of the relational-planner design.

### 6.1 E3-pre — precedent extension (run this first, it can cancel E3)

The `(12,11)` discriminator is strong but **only at 8 decisions**. Measured
over every run rooted at the v318 pre-push staging:

| Run | Collected hearts (decision, slot) |
| --- | --- |
| v323, v324, v327, v328, v329, v330, v331 | `(d1, (96,128))`, `(d3, (128,128))` — **seven runs, identical, none collect `(192,176)`** |

E3 needs more than 8 decisions (§6.3), which extends the *control's*
opportunity too. So: **one run, authority `off`, same root, same flags,
`--decisions 16`, scoring nothing** except whether incidental behavior
collects `(192,176)` within 16 decisions.

- If it does, the discriminator is dead at 16 decisions and E3 must be
  redesigned before it runs (this is the cheapest possible way to learn it).
- If it does not, the precedent is extended to the window E3 needs and the
  E3 control's job is confirmed.
- Cost: ~32 min M5 (§8). It doubles as the S0 invariance check in vivo — its
  first 8 committed decisions must reproduce v330's state ids exactly.

### 6.2 Arms, root, budget

- **Control** — `--relational-planner-authority off`, run id
  `entity-v33X-room3-e3-control-off-d16`.
- **Treatment** — `--relational-planner-authority selection`, run id
  `entity-v33Y-room3-e3-treatment-selection-d16`.
- Arms differ in **exactly two** things: that flag and `--run-id`. Both load
  the certified record store at
  `--human-prior-accessibility-preference-weight 0.0`, so the WP8-lite term
  is off in both arms and any difference is attributable to the navigation
  objective alone (E1 §12.2 structure, inherited verbatim).
- **Root**: identical to E1 §12.2 — memory
  `entity-v318-room3-known-push-connected-mask-d2` decision 1 with
  `--resume-option-search`, physical state seq-2026
  (`state-00000002`, sha `33addc6c…`, source events sha `0bbe1d15…`).
  All input digests re-verified at preregistration time.
- **Flags**: v329/v330/v331's profile verbatim, with `--decisions 16` and
  `relational_decision_budget` raised (§6.3). Because the mechanism carries
  no new config field, both arms' `planning_config` differ only in
  `relational_planner_authority` / `relational_planner_enabled`.
- **Matched budget: exact by construction.** (b) verifies no additional
  branches. Both arms should record 12,232 verified branches, all in the d1
  resume audit. Wall ceiling 10,800 s/arm under the external watchdog;
  event expectation ≤ 200k/arm.
- **Scoring window**: the first **16 committed decisions**, not a branch
  count. Rationale, disclosed: after d1 no branches are verified at this
  root, so a branch window would be decided entirely by the resume audit and
  would carry no information about the decisions under test. E1's §7.2
  branch-truncation ruling does not apply because there is nothing to
  truncate; the equivalent guard becomes VOID V5 below.

### 6.3 The three preregistered bits (fixed; ANY mixed outcome = FAIL)

1. **Directed navigation (the mechanism bit).** While an exploit hypothesis
   with non-empty certified `target_cells` holds authority, the treatment's
   grid distance from the committed player cell to the nearest target cell
   is **non-increasing across every steerable committed decision** (a
   steerable decision being one committed through the P5 ladder, not a
   restore), for at least **4** such decisions; and the treatment's minimum
   in-window distance is **strictly smaller** than the control's. Recorded
   precedent: v330/v331 went 7 → 6 → 8 → 9 → 10 across d3–d7, i.e.
   monotonically *away* after d4, minimum 6.
2. **Chained consequence (the E1 discriminator, unchanged).** Within the
   window, the treatment's committed trajectory collects `(192,176)` —
   evidenced by `[192,176]` entering `human_prior_collected_heart_slots` on
   a `decision_committed` event — and the control does not. If both collect
   it, the treatment must do so at a strictly earlier decision index. The
   metric is the milestone cell only, never affordance counts.
3. **No safety regression.** The treatment records no more
   `human_prior_life_loss_confirmed` committed decisions than the control
   within the window.

All three must pass. **ANY mixed outcome = FAIL.** No weight tuning, no
budget re-sizing, no rerun on an identical negative result. Explicitly:
`budget_exhausted` or `hold_violated` termination of the exploit is a
**FAIL, not a VOID** — otherwise the design is unfalsifiable.

Reported invariants (not bits): hold integrity (every committed decision in
the window carries `85fd9014d58deb42`); the S1/S2 exercised-difference
counts of §5.3; per-arm verified-branch counts.

### 6.4 Honest power analysis, §12.1(e)-style, **before** the run

**(a) Steerable-commit supply.** Measured at the E1 root, the restore
cadence is 1-in-3 (d2, d5, d8 restores; d1, d3, d4, d6, d7 commits). The
exploit activates *after* d3's commit, at cell (8,8), 7 certified cells from
`(12,11)`. In an 8-decision run it gets 3 steerable commits — **arithmetically
incapable** of arriving. This is why the decision budget must rise, and it is
the reason E3 is not a rerun of E1 with a new key.

**(b) Derivation of `--decisions 16`.** 3 decisions of establish prologue +
7 cells of travel + the 1-in-3 restore cadence (≈4 restores, each neutral
*if* C2 ratchets) + 2 decisions margin ≈ 16.

**(c) The named most-likely FAIL mode — restore drag.** Measured: archive
deposits during the exploit window trail the agent by exactly one cell
(v331 d7 deposits at (8,6)/(8,7) while committed at (7,6)). If C2 ratchets
against the *easternmost ever-visited* hold-matching branch, each restore
costs ≤1 cell and 16 decisions suffice. If instead it can only see the four
d1-audit branches (i.e. **S0 does not land, or does not work**), the restore
supply is capped at (9,8) and every restore drags the agent back to distance
6 — under which **no decision count closes bit 2**, and the required budget
balloons past 21 decisions. S0 is therefore a hard precondition, not a
nicety, and its in-vivo check is E3-pre.

**(d) Discriminator erosion.** The seven-run precedent covers 8 decisions.
Extending to 16 extends the control's opportunity. If the control collects
`(192,176)` the discriminator is destroyed and the result is a **FAIL**, not
a VOID. E3-pre exists precisely to learn this for ~32 min instead of ~64.

**(e) `relational_decision_budget`.** At 4 the exploit died at d7 (measured).
It must cover the travel window; derive it from (b) as **≥ 12**, and record
that this is a *scope* change (how long a hypothesis may remain active), not
a re-size of `relational_exploit_budget` — which §12.10 item 3 forbids and
which this design does not touch (it remains 48 and remains unread, since
(b) consumes no beam slots).

**(f) What (b) cannot show.** A PASS on E3 demonstrates non-search steering
under a held configuration. It does **not** show the planner can cause a
search, and it does not close §4.45's mechanism 1 in full. That remains E4.

### 6.5 VOID conditions (a VOID is not evidence)

1. **Config inequality** — either arm's manifest `planning_config` differs
   from the other's in any field except `relational_planner_authority` and
   `relational_planner_enabled`.
2. **Records inequality** — both arms must report `record_count: 3`, content
   signatures `15604cb5…`/`37ea410d…`/`47975c94…`, store digest
   `cf01a67a…`, at `verified_accessibility_weight: 0.0`.
3. **Seeding defect** — no archived branch carrying `85fd9014d58deb42`
   within the window in **either** arm (the establish never has a target).
4. **Root defect** — either manifest's `episodic_resume` block does not
   record source run `entity-v318-room3-known-push-connected-mask-d2`,
   `source_decision: 1`, `state_source_checkpoint_event_seq: 2026`,
   `state_source_events_sha256: 0bbe1d15…`.
5. **Budget defect** — either arm exceeds the 10,800 s wall ceiling and is
   killed before `run_finished`; **or** the two arms' verified-branch counts
   differ by more than 1% (unmatched emulator work, which (b) must not
   produce).
6. **Invariance defect** — the control's first 8 committed decisions do not
   reproduce v330's state ids exactly (S0/S1/S2 leaked outside selection
   authority).

Budget-exhausted non-reach is **censored**, never reported as
"unreachable" (learnings §2, §4.14).

### 6.6 Scoring

A single deterministic scorer walks each arm's `events.jsonl` once, applies
§6.3 verbatim, and writes `experiments/lolo1-wp5/e3-navigation-report.json`
with a canonical-JSON `digest_sha256` over the body. Run end-to-end twice;
both reports must be byte-identical, and the digest recorded in the results
section. The scorer reads only telemetry and is validated against v330/v331
first (it must reproduce E1's distances of §3.2 exactly).

---

## 7. Q5 — Risks

### 7.1 §14 monolith risk (realized)

`neural_planner.py` is ~28.2k lines. Mitigations, unchanged in spirit from
the relational-planner design §9.1 and tightened here:

- all new ranking logic is **pure**, in `relational_planner.py`, one-way
  imports;
- the monolith touch is four enumerated, grep-able seams (§5.1), none inside
  the high-conflict scoring bodies; S2 extends a seam that already exists;
- authority-gated off by default with an executable invariance test **and**
  an in-vivo control-arm reproduction of v330 (VOID V6);
- **file ownership**: `6a8488a` is already merged to main (`5b64a9b`), so
  §9.2's precondition is discharged. WP5 shadow wiring still queues behind
  the planner file; declare the sequence WP8 search-scheduling seams → WP5
  shadow wiring, one owner at a time.

### 7.2 The twice-measured redundancy pattern

This design deliberately does **not** add a third preference weight over
candidates the baseline already ranks. The axis it occupies — *distance to a
certified target cell on a non-search decision* — is currently occupied by
nothing (§3.2). But "unoccupied" is a claim, so it is instrumented: §5.3's
exercised-difference telemetry makes redundancy an in-run measurement, and
§6.3's bit 1 fails if the treatment's committed trajectory does not diverge
from the control's. If the levers never differ, that is a third redundancy
finding and the plan moves to representation, not to a fourth lever.

### 7.3 §4.7 straight-line goal distance

Learnings §4.7 recorded straight-line distance-to-heart as a **negative
result as the principal long-horizon planning representation**: it stalled
at obstacles requiring preparation and could not value moving an object away
from a goal. The failure mode is real and the corresponding machinery is
still in the tree and still disabled (`human_prior_navigation_reward = 0.0`,
`:178`; `goal_navigation_bonus` at `:19967`).

**E3 must not re-enable it**, and the mechanism here is a different object
for five reasons that are all structural rather than parametric:

1. It is a **tie-break inside an already-filtered candidate set**, not a
   reward added to every branch's score.
2. Its targets come **only from certified records** (`certified_milestone_cells`
   / `certified_cells`), never from visible sprites — so it cannot pull the
   agent at a configuration where the cells are not certified reachable.
3. It is **gated on a held configuration**: the objective exists only while
   an exploit hypothesis is active under a verified hold, and terminates on
   `hold_violated`.
4. It is **time-boxed** by `relational_decision_budget` and dies with the
   hypothesis.
5. It **never outranks a milestone collection** (placement below
   `human_prior_goal_choice`), and never overrides the safety paths P1/P4.

That is precisely §4.7's own stated learning — *"goal proximity must be
coupled to world configuration and verified accessibility"* — implemented
rather than restated. The preregistered failure signature is nonetheless
named: if the treatment's distance is non-increasing (bit 1 passes) but it
stalls at a fixed distance without collecting (bit 2 fails), that is §4.7
recurring inside a certified envelope, must be recorded as such, and makes
the next question a *preparation* question (the `(8,4)`/`(9,12)`
continuation, roadmap §18 item 3), not a tuning question.

### 7.4 §14 "broader search hides representation failures"

(b) does not broaden search at all — it is branch-neutral. (a)/E4 does, and
carries this risk directly; the forced-search control arm of §4.1 is its
mitigation, and E4 must not be started before E3 is scored.

### 7.5 Provenance

Everything here is assisted-track: the certified records derive from the
player-anchored hold instrument. No strict claim is made or implied. The
WP5 shadow campaign (§4.42) remains the strict path and the strict-lineage
linter guards the module boundary (`python -m lolo_agent.strict_lineage`
must report `assisted: false` for `relational_planner.py`).

---

## 8. Budgets (bounded, declared now)

Derived from measured wall costs (`elapsed_ms` deltas, §2.2): startup ~98 s;
the d1 resume audit at this root **1,594 s**; each subsequent non-search
decision **12–17 s**; a restore-commit ~0 s. Decisions are therefore nearly
free at this root, and the search dominates.

| Item | Cost |
| --- | --- |
| S0–S4 + unit tests | code-review scale, zero emulator |
| E3-pre (precedent extension, 16 decisions, authority off) | 98 + 1,594 + 16×13 ≈ **1,900 s ≈ 32 min** |
| E3 control + treatment | 2 × ~32 min ≈ **64 min** |
| **E3 total M5 native** | **≈ 1.6 h**, one run at a time, external watchdog, event ceiling 200k/arm |
| E4 (search request, later, with forced-search control) | 2 arms × (1,900 + one granted search 320–1,594 s) ≈ **2 h**; preregistered separately |

No depth/beam escalation anywhere; no rerun on identical negative evidence;
nothing on RunPod (§13: emulator branching is M5-bound); no re-size of
`relational_exploit_budget` (§12.10 item 3).

---

## 9. Sequencing

1. S0 instrument fix + invariance tests (behaviour-neutral, provable).
2. S3 pure module helper + tests; S1/S2/S4 seams; full suite green.
3. **E3-pre** precedent extension — may cancel or redesign E3.
4. Append the E3 preregistration addendum (digests, command lines, scorer)
   to this document.
5. **E3**: control, then treatment, one at a time. Score. Record in
   learnings either way.
6. Only then: E2 (the §11.3 seeded conflict root, unaffected by all of this
   and still queued), and E4 (search request with forced-search control) if
   E3's mechanism bit passes but its consequence bit does not.

---

## 10. What this document does not claim

No navigation tier exists yet; no search-request mechanism exists yet; no
bit here is scored; Gate 4's consequence criteria remain open. The §2 and §3
tables are re-derivations from stored telemetry, not new measurements, and
they are reproducible read-only from
`experiments/lolo1-entity-v10/evaluations/`. The claim that the
distance-to-certified-cell axis is currently unoccupied is a code reading
(§3.2, §3.3) that §5.3's telemetry is designed to falsify in run.
