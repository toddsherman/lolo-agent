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

---

## 11. E3 preregistration addendum (2026-08-18, **written before either arm ran**)

This section supersedes §6.2/§6.3/§6.4's *provisional* numbers where they
conflict. It is the scored preregistration. Two rows of §5.1/§6.4 are
**stale** and are corrected here rather than silently used: §5.1's S0 row
describes a change that was already landed by `6a8488a` before this design
was written, and §6.4(c)'s "restore drag" FAIL mode is retired because the
supply it feared was measured and is 52/56, not 4 (learnings §4.49).

### 11.1 Preconditions, re-verified at preregistration time

| Item | Status |
| --- | --- |
| S0 (commit-time archives carry the tracked signature) | landed `6a8488a`; measured supply on v333 = 52/56 commit-time archives |
| Mechanism (S1 ladder tier, S2 restore key, S3 pure module, S4 telemetry) | landed `7a232a5` |
| `--relational-decision-budget` | landed; `neural_run.py:981`, default 4, E3 uses 12 |
| Discriminator | VALIDATED over v323/324/327/328/329/330/331/332/333 — no run has ever collected `(12,11)`/pixel `(192,176)` |
| HEAD at preregistration | `9d2889e`, working tree clean except untracked `tmp/` (unrelated) |

Input digests, re-verified on disk today (all equal to the v322–v333
manifests): host `c03694c5…3e891f3`, core `a3450a09…5a40024886`, ROM
`914c6769…3efd059e01`, neural checkpoint `bb7a7a37…284f678b9`,
entity-behavior checkpoint `984b83c3…25c7c6aa`, record store
`cf01a67aca2b6e8feeab38c0c85520dec2470cba2a5f2257cd817912c204d1fe`.

### 11.2 Arms

- **Control** — `--relational-planner-authority off`, run id
  `entity-v334-room3-e3-control-off-d24`. Runs **first**.
- **Treatment** — `--relational-planner-authority selection`, run id
  `entity-v335-room3-e3-treatment-selection-d24`.
- The arms differ in **exactly two** things: that flag and `--run-id`.
  Consequently their manifest `planning_config` may differ only in
  `relational_planner_authority` and `relational_planner_enabled`.
- Both arms: `--decisions 24`, `--relational-decision-budget 12`,
  `--human-prior-accessibility-records experiments/lolo1-wp5/wp8lite-accessibility-records.json`,
  `--human-prior-accessibility-preference-weight 0.0` (records loaded in
  both arms; the WP8-lite preference term is off in both, so any difference
  is attributable to the navigation objective alone),
  `--log-root experiments/lolo1-entity-v10/evaluations`.
- **Root**: memory `entity-v318-room3-known-push-connected-mask-d2`
  decision 1 with `--resume-option-search`; physical state the same run's
  **seq-2026** checkpoint (`state-00000002`, source `events.jsonl` sha256
  `0bbe1d15…9b6f83`). Identical to v333.
- **Flag profile**: v333's verbatim (itself v329/v330/v331's profile), i.e.
  the command line of the relational-planner design §12.3 with
  `--decisions 24` and `--relational-decision-budget 12` added.
- **Ceilings**: 10,800 s wall per arm under an external watchdog; one
  native run at a time. Observed envelope at 24 decisions (v333): ~33 min,
  85,594 events, 12,232 verified branches.
- **Scoring window**: the first **24 committed decisions**. Rationale
  unchanged from §6.2: after d1 no branches are verified at this root, so a
  branch window would be decided entirely by the resume audit.

### 11.3 Exact command lines

```
.venv/bin/python -m lolo_agent.neural_run \
  --host build/lolo-libretro-host \
  --core "/Users/toddsherman/Library/Application Support/RetroArch/cores/nestopia_libretro.dylib" \
  --rom "Adventures of Lolo.nes" \
  --checkpoint experiments/platform-benchmarks/m5-real-data-training-sample.pt \
  --log-root experiments/lolo1-entity-v10/evaluations \
  --run-id <ARM RUN ID> \
  --decisions 24 \
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
  --relational-decision-budget 12 \
  --relational-planner-authority <off | selection>
```

### 11.4 The three preregistered bits (fixed; **ANY mixed outcome = FAIL**)

**Bit 1 — MECHANISM.** At the d18-class stagnation restore — defined as any
`archive_branch_restored` instant at which a certified-adjacent candidate
competes with a higher-novelty one, i.e. any restore whose
`relational_navigation_restore_selected` event reports
`hold_matching_candidates >= 2` and at least one candidate strictly nearer
the target than the baseline — the treatment does **not** abandon the
certified-adjacent position. Evidence required: at least one
`relational_navigation_restore_selected` with **`differs: true`**, and at
that instant `selected_distance < baseline_distance`.

This bit replaces §6.3 bit 1 (the "non-increasing distance for ≥4 steerable
decisions" formulation), which was written before §4.48 identified the
restore key — not the commit ladder — as the measured failure site. The
substitution is recorded here, before the run, as a deliberate change of the
mechanism bit, not a post-hoc relaxation: it is *narrower* than the old bit
in that it demands a demonstrated divergence from the incumbent choice
rather than a distance trend that novelty could produce incidentally.

**Bit 2 — OUTCOME.** Within the 24-decision window the treatment's committed
trajectory collects `(12,11)` / pixel `(192,176)` — evidenced by
`[192,176]` entering `human_prior_collected_heart_slots` on a
`decision_committed` event — and the control does **not**. If both collect
it, bit 2 FAILS (the discriminator would be dead; a speedup is not the
claim). The metric is the milestone cell only, never affordance counts.

**Bit 3 — SAFETY.** The treatment records no more
`human_prior_life_loss_confirmed` committed decisions than the control
within the window.

All three must pass. **ANY mixed outcome = FAIL.** No weight tuning, no
budget re-sizing, no rerun on an identical negative result. Explicitly:
`budget_exhausted` or `hold_violated` termination of the exploit is a
**FAIL, not a VOID**.

Reported invariants (not bits): hold integrity (committed decisions in the
window carrying `85fd9014d58deb42`); the S1/S2 exercised-difference counts
of §5.3; per-arm verified-branch counts; per-arm option-search counts
(§4.46's planner-health metric).

### 11.5 Declared caveats and blind spots (all three, before the run)

**(a) Frozen-signature caveat (learnings §4.49).** The configuration
signature an archive carries is *frozen at deposit*, not recomputed: the
root track state is assigned at only five sites and never advanced by an
ordinary committed decision, so an archive claims the configuration as of
the last restore. For E3 specifically the risk is low — the held
configuration is the *removal*, where the manipulated object no longer
exists to move, so a mid-run configuration change is implausible — but a
bit-1 PASS that depended on a stale signature would be an artifact. It is
declared, not assumed away. Changing the freeze is score-bearing at
accessibility weight > 0 and needs its own gate; E3 runs at 0.0.

**(b) Decision-1 empty-seed blind spot.** At the root the tracked
configuration is `prepush-root-empty-track-unmatchable`; the removal
configuration does not exist until after the first restore onto a removal
branch. No objective can publish target cells before that. Decision 1 is
therefore outside the mechanism's reach in *both* arms by construction, and
neither bit may be scored on it.

**(c) §5.3 redundancy instrument — the pre-declared third-null reading.**
Every S1/S2 instant logs the incumbent argmax alongside the
objective-preferred candidate and whether they `differ`. **If `differs` is
`false` (or `null`) at every instant across the exploit's authority window,
that is a THIRD redundancy finding of the §4.43/§4.45 family and must be
reported as such** — "the lever agreed with the incumbent everywhere, so
the mechanism was never exercised" — and explicitly **not** as a near-miss,
not as under-powering, and not as grounds for a fourth lever. Under that
reading the plan moves to representation (WP2/WP3 integration depth). This
is fixed now precisely so it cannot be re-narrated after seeing the result.

**(d) Speedup-vs-capability caveat, carried from §4.47.** Even a clean PASS
demonstrates *finishing* under a held configuration at this root; it does
not demonstrate a second manipulation. The stronger discriminator remains
the `(8,4)`/`(9,12)` hearts outside the certified envelope (roadmap §18
item 3).

**(e) What a PASS does not show (§6.4(f), unchanged).** It does not show
the planner can cause a search, and does not close §4.45's mechanism 1.
That remains E4.

### 11.6 VOID conditions (a VOID is not evidence)

1. **Config inequality** — either arm's manifest `planning_config` differs
   from the other's in any field except `relational_planner_authority` and
   `relational_planner_enabled`.
2. **Records inequality** — both arms must report `record_count: 3`,
   content signatures `15604cb5…`/`37ea410d…`/`47975c94…`, store digest
   `cf01a67a…`, at `verified_accessibility_weight: 0.0`.
3. **Seeding defect** — no archived branch carrying `85fd9014d58deb42`
   within the window in **either** arm.
4. **Root defect** — either manifest's `episodic_resume` block does not
   record source run `entity-v318-room3-known-push-connected-mask-d2`,
   `source_decision: 1`, `state_source_checkpoint_event_seq: 2026`,
   `state_source_events_sha256: 0bbe1d15…`.
5. **Budget defect** — either arm exceeds the 10,800 s wall ceiling and is
   killed before `run_finished`; **or** the arms' verified-branch counts
   differ by more than 1%.
6. **Invariance defect** — the control's committed state ids do not
   reproduce v333's exactly for all 24 decisions (S0/S1/S2 leaked outside
   selection authority). v333 is the same flag profile plus
   `--relational-decision-budget 12`, which is inert at authority `off`.

Budget-exhausted non-reach is **censored**, never reported as
"unreachable" (learnings §2, §4.14).

### 11.7 Scoring

A single deterministic scorer walks each arm's `events.jsonl` once, applies
§11.4 verbatim, and writes `experiments/lolo1-wp5/e3-gate4-report.json`
with a canonical-JSON `digest_sha256` over the body (digest field excluded).
It is run end-to-end twice; both reports must be byte-identical and the
digest is recorded in the results section below.

Two distance metrics are reported side by side and their difference is
declared now to prevent a later mix-up:

- **Chebyshev** (`max(|dx|,|dy|)`) — the metric of the learnings §4.47/§4.48
  traces; reported so the treatment's trace is directly comparable to
  v333's `3,3,4,3,4,4,3,2,1,4,5,5,5,4,5,6` (d9–d24).
- **Manhattan** (`|dx|+|dy|`) — the metric the mechanism itself uses
  (`relational_planner.target_cell_distance`), and therefore the metric of
  `baseline_distance` / `selected_distance` inside bit 1.

Bit 1 is scored on the mechanism's own (Manhattan) figures as emitted in
telemetry. Bit 2 is metric-free. The Chebyshev trace is reportorial only.

The scorer is validated against v333 before scoring E3: it must reproduce
§4.48's distance trace and its "never collected `(192,176)`" reading exactly.


---

## 12. E3 results (2026-08-18) — **FAIL**, with a named mechanism

Scored against §11.4 verbatim by `experiments/lolo1-wp5/e3-gate4-report.json`,
`digest_sha256` **`26a3cc22fad69bba3e61a8d299f09550f942872170cb61c4169e130d8b310452`**
(scorer run end-to-end twice; the two reports are byte-identical). The scorer
was validated against v333 first and reproduces §4.48's trace and its "never
collected `(192,176)`" reading exactly.

| Arm | Run id | Events | Branches | Searches | Result |
| --- | --- | --- | --- | --- | --- |
| Control | `entity-v334-room3-e3-control-off-d24` | 85,594 | 12,232 | 1 (+9 deferred) | reached distance 1 at d17, diverged to 6 by d24; `(12,11)` never collected |
| Treatment | `entity-v335-room3-e3-treatment-selection-d24` | 85,932 | 12,232 | 1 (+11 deferred) | minimum distance 3; `(12,11)` never collected |

### 12.1 Verdict

| Bit | Verdict | Evidence |
| --- | --- | --- |
| **1 — MECHANISM** | **PASS** | 3 contested restores (d8/d9/d10) with `differs: true` and `selected_distance` 6 < `baseline_distance` 9. The objective held ground the incumbent would have traded. |
| **2 — OUTCOME** | **FAIL** | Neither arm collected `(12,11)`/`(192,176)`. The treatment's minimum distance (Chebyshev 3 / Manhattan 6) is *worse* than the control's (1 / 1). |
| **3 — SAFETY** | **PASS** | Zero `human_prior_life_loss_confirmed` commits in both arms. |

**Mixed ⇒ FAIL**, per §11.4. **No VOID condition fired** (V1–V6 all clear).
No tuning, no re-size, no rerun.

### 12.2 The instruments that make this FAIL trustworthy

- **V6 invariance, in vivo**: the control reproduced v333 **state-id for
  state-id across all 24 decisions** (`matching_prefix_decisions: 24`, and
  the same 85,594 events). S0/S1/S2 leak nothing outside selection
  authority, and `--relational-decision-budget 12` is inert at authority
  `off` as claimed.
- **V1**: the arms' `planning_config` differ in exactly
  `relational_planner_authority` and `relational_planner_enabled`.
- **V5**: both arms verified exactly 12,232 branches — matched by
  construction, relative gap 0.0.
- **§5.3 redundancy instrument — NOT a third null.** 20 S1/S2 instants, **7
  differing**. The lever fired and it changed behavior. §11.5(c)'s
  pre-declared third-redundancy reading therefore does **not** apply; this
  is a real behavioral difference that produced a worse outcome, which is a
  strictly more informative result than another agreement null.

### 12.3 Distance-to-target traces (Chebyshev, d1–d24)

```
v333  (precedent) 6 4 4 3 4 5 5 5 3 3 4 3 4 4 3 2 1 4 5 5 5 4 5 6
v334  (control)   6 4 4 3 4 5 5 5 3 3 4 3 4 4 3 2 1 4 5 5 5 4 5 6   ← identical
v335  (treatment) 6 4 4 3 4 4 3 3 3 3 5 4 4 5 5 5 4 6 5 5 6 6 6 6
```

Manhattan (the mechanism's own metric):

```
v334  (control)   9 7 7 6 8 9 10 10 6 5 6 4 5 4 3 2 1 6 7 6 6 4 5 6
v335  (treatment) 9 7 7 6 8 7  6  6 6 6 9 8 7 10 9 9 8 11 10 9 10 9 8 8
```

The arms are identical through d5 and never re-converge after d6.

### 12.4 The named FAIL mechanism: **the objective starved its own supply**

The mechanism did exactly what §4.48 asked and lost anyway. The chain is
measured, not inferred:

1. **The fork is the commit tier, not the restore key.** First divergence is
   **d6**, a `relational_navigation_choice` with `differs: true`: the
   incumbent `human_prior_semantic_frontier_choice` would have committed
   `(8,6)` at distance 9; the objective substituted `(8,8)` at distance 7.
   Locally correct, and the entire causal fork of the experiment.
2. **What the incumbent was doing was not noise.** The control's d6–d8
   excursion *away* from the target — `(8,6) → (7,6) → (7,6)`, distance 9,
   10, 10 — is what deposited the archives it later restored into. By d12
   and d15 the control was restoring to `(11,8)` and `(12,8)`; the treatment
   never restored east of `(9,8)`.
3. **Archive geography, the decisive measurement.** Control: 44 deposits
   spanning columns 6–12, including `(12,9)` — one cell from the milestone.
   Treatment: 33 deposits spanning columns **6–8 only**. The treatment's
   archive supply never reached the target's half of the room.
4. **Then the ratchet ran backwards.** `hold_matching_candidates` across the
   treatment's nine restores: 1, 4, 3, 2, 1, 1, 1, 1, 1. The key exhausted
   its near candidates by d10, and from d11 every stagnation restore had a
   single hold-matching option — progressively *westward*: `(8,6)` d11,
   `(7,6)` d14, `(6,6)` d18, `(6,7)` d21, `(6,9)` d24. The mechanism
   ratcheted, correctly, against a supply it had itself prevented from
   growing.
5. **The commit tier had almost nothing to choose from either.**
   `distance_reducing_branches` was **1 of 7** hold-eligible branches at
   nearly every instant (2 once). The "choice" was Hobson's.

This is §6.4(c)'s restore-drag FAIL mode arriving by the opposite route from
the one predicted. §6.4(c) feared drag from a *missing* S0 capping supply at
four audit branches; S0 landed and supply was ample (§4.49). The drag came
instead from the objective suppressing the exploration that *generates*
supply. Neither the design nor the power analysis anticipated that the
mechanism could reduce its own restore supply.

It is also learnings §4.7 ("straight-line goal distance") and this design's
§7.3 reappearing in a form the design believed it had avoided. §7.3's
mitigation was to add no distance *reward* — and none was added; the key is
a tie-break inside an already-filtered set, exactly as specified. The
failure shows that the reward/tie-break distinction is not the operative
one: **any** consistent preference for target proximity, however narrowly
scoped, suppresses the excursions that build the archive supply on which
later progress depends.

### 12.5 What §4.48 got right, and what it missed

§4.48's diagnosis — "the incumbent can wander into the neighborhood but has
no mechanism to close" — is confirmed in the control (distance 1 at d17,
traded away, never recovered). What it missed is that the wandering and the
arriving are the **same mechanism**. Closing by suppressing the wandering
removes the arriving. E3 therefore refutes the specific repair §4.48
proposed while leaving its diagnosis of the gap intact.

The mechanism bit passing while the outcome bit fails is the most useful
shape this result could have taken: the seam works, is exercised, is
correctly gated, and is bit-identical when off. The lever is sound; the
*policy* on the lever is wrong.

### 12.6 Caveats, restated after the fact (none rescues the result)

- **Frozen signature (§11.5(a))**: did not bite. The held configuration was
  stable and `hold_matching_candidates` behaved consistently; no bit turned
  on a signature identity.
- **Decision-1 blind spot (§11.5(b))**: as declared, d1–d3 are outside the
  mechanism's reach in both arms; the objective activated at d3 and the
  first navigation instant is d4. Nothing is scored there.
- **`budget_exhausted` at d15**: the exploit hypothesis exhausted its
  12-decision budget at d15 and re-proposed at d16, holding authority
  through d24. Per §11.4 this is a FAIL input, not a VOID — and it is not
  the operative cause here, since the mechanism held authority for 20 of 24
  decisions and the damage was already done by d10.
- **Speedup-vs-capability (§11.5(d))**: moot. There was no speedup.
- **Not a power problem.** The treatment did not run out of decisions; it
  ran out of *room*, and was moving away from the target when the window
  closed. A larger `--decisions` would extend a diverging trajectory.
  §11.4's no-rerun clause binds.

### 12.7 Consequences for the plan

- The navigation-target mechanism as specified is **refuted** for closing on
  a certified milestone. Do not tune its weight, budget, or ranking — the
  §11.4 no-tuning clause covers exactly this temptation, and §12.4(5) shows
  there is nothing to tune: the candidate supply is one branch wide.
- **E4 (search request) is now better motivated, not worse.** §12.4(5) is a
  supply problem: at every steerable instant the objective could choose
  among one distance-reducing branch out of seven. A mechanism that can
  *cause a search* changes the candidate set rather than re-ranking a set of
  size one. E3's failure is evidence for that reading, and E3's telemetry
  gives E4 a measured supply baseline to beat.
- The §4.47/roadmap §18 point stands: the `(8,4)`/`(9,12)` second-
  manipulation discriminator remains the capability-level test.
- Retained for reuse: the discriminator (now unbroken across eleven runs),
  the v333/v334 invariance pair, and the S4 telemetry, which diagnosed this
  FAIL in one pass without a rerun.

---

## 13. E5 preregistration — the surgical closing intervention (2026-08-18, **written before either arm ran**)

E3 ran both seams together and failed (§12). §12.4 named the mechanism: seam
**S1** (the commit-ladder tier) steered the trajectory, narrowing archive
geography from columns 6–12 to 6–8, and every subsequent restore then
ratcheted westward against a supply the objective had itself prevented from
growing. Seam **S2** (the target-aware restore key) has **never been tested
in isolation** — by the time any restore mattered in E3 the geography was
already ruined.

**E5's hypothesis.** With S1 disabled and only S2 active, exploration should
run *identically to the control* — depositing the same archive ladder and
reaching distance 1 at d17 as v333/v334 both did — and the single
intervention is refusing the d18-class restore that abandons the
certified-adjacent position. This is the intervention shape roadmap §20
item 2 demands: leave exploration untouched; intervene only at the closing
instant.

### 13.1 The seam selector (the only code change)

`NeuralPlanningConfig.relational_navigation_seams`, surfaced as
`--relational-navigation-seams`, taking `both | restore_only | off`:

| Mode | S1 commit tier | S2 restore key | Meaning |
| --- | --- | --- | --- |
| `both` (**default**) | on | on | today's behavior, exactly — E3's treatment |
| `restore_only` | **off** | on | **E5's treatment**: exploration untouched, closing contested |
| `off` | off | off | both navigation seams inert under selection authority |

Implementation is two config-read guards and two helpers: the S1 entry point
`_relational_navigation_commit_view` returns `None` (the pre-existing
fail-open path, so the ladder and its S4 telemetry are both absent), and the
S2 `elif` in the restore key is additionally conjoined with the selector.
Nothing else moves. The selector is **inert at any authority other than
`selection`**, because both guarded sites already require an active
navigation objective, which only selection authority publishes.

Behavior-preservation evidence, landed before the arms ran:

- Full suite **1060 tests OK, 4 skipped** — 1045 + 4 before this change,
  plus the 15 new tests below. No pre-existing test changed.
- `tests/test_ensemble_planner.py::RelationalNavigationSeamSelectorTests`:
  the default is `both`; an arm naming `both` explicitly and an arm built by
  the untouched pre-E5 helper produce **identical event streams, identical
  commit views, identical restore decisions and identical archives**;
  `restore_only` kills S1 (commit view `None`, zero
  `relational_navigation_choice`/`_declined` events) while S2 still flips
  the restore with `differs: true`; `off` reproduces off-mode, restoring
  precisely the branch the S2 telemetry names as `baseline_state_id`; and
  the selector is byte-identically inert at authority `off` and `telemetry`
  across all three modes.
- `tests/test_relational_planner.py::NavigationSeamAblationContractTests`:
  the pure module exposes no seam selector at all — `restore_only` changes
  *who reads* the published key, never *what is published*.
- Both guards were mutation-tested: forcing either helper to `True` fails 4
  and 3 of the new tests respectively.

### 13.2 Arms

- **Control** — `--relational-planner-authority off`, run id
  `entity-v336-room3-e5-control-off-d24`. Runs **first**.
- **Treatment** — `--relational-planner-authority selection`, run id
  `entity-v337-room3-e5-treatment-restore-only-d24`.
- **Both arms pass `--relational-navigation-seams restore_only`**, so the
  arms differ in **exactly two** manifest fields —
  `relational_planner_authority` and `relational_planner_enabled` — exactly
  as in E3. This is legitimate only because the selector is provably inert
  at authority `off`; that inertness is unit-tested
  (`test_selector_is_inert_outside_selection_authority`), not assumed.
- Every other flag is **v334's manifest profile, verbatim** (§11.3's command
  line): `--decisions 24`, `--relational-decision-budget 12`,
  `--human-prior-accessibility-records
  experiments/lolo1-wp5/wp8lite-accessibility-records.json`,
  `--human-prior-accessibility-preference-weight 0.0`,
  `--log-root experiments/lolo1-entity-v10/evaluations`.
- **Root**: identical to v333/v334/v335 — memory
  `entity-v318-room3-known-push-connected-mask-d2` decision 1 with
  `--resume-option-search`; physical state the same run's seq-2026
  checkpoint, `state_source_events_sha256 0bbe1d15…9b6f83`.
- **Ceilings**: 10,800 s wall per arm under an external watchdog; one native
  run at a time.
- **Scoring window**: the first **24 committed decisions**.

Input digests, re-verified on disk today and equal to the v334 manifest:
host `c03694c5…3e891f3`, core `a3450a09…5a40024886`, ROM
`914c6769…3efd059e01`, neural checkpoint `bb7a7a37…284f678b9`,
entity-behavior checkpoint `984b83c3…25c7c6aa`, record store
`cf01a67aca2b6e8feeab38c0c85520dec2470cba2a5f2257cd817912c204d1fe`.
HEAD at preregistration `d95a8fa`, working tree clean except the four owned
files of this change and the unrelated untracked `tmp/`.

### 13.3 Exact command lines

§11.3's block verbatim, with two flags changed and one added:

```
  --run-id entity-v336-room3-e5-control-off-d24        (control)
  --run-id entity-v337-room3-e5-treatment-restore-only-d24  (treatment)
  --relational-navigation-seams restore_only            (BOTH arms)
  --relational-planner-authority <off | selection>
```

### 13.4 The four preregistered bits (fixed; **ANY mixed outcome = FAIL**)

Let **C** be the decision index of the treatment's **first contested
restore**: the first `relational_navigation_restore_selected` event carrying
`differs: true`. This is the first instant at which the intervention can
change anything at all. If no such event exists, **C is undefined** and bit
1(a) is scored over all 24 decisions.

**Bit 1 — SUPPLY PRESERVED.** Both conjuncts must hold.

- **(a) Trajectory prefix.** For every committed decision `d < C` (all 24 if
  C is undefined), the treatment's `committed_state_id` equals the
  control's. This is the direct test of "exploration runs identically to the
  control": with S1 off, nothing may fork the trajectory before the closing
  restore fires.
- **(b) Archive geography not narrowed.** The treatment's archive-deposit
  column range must not be narrower than the control's:
  `treatment_min_column <= control_min_column` **and**
  `treatment_max_column >= control_max_column`. Both raw ranges and both
  full deposit-cell histograms are reported either way.

  *Why not strict equality:* E3's failure was **narrowing** (6–12 → 6–8). A
  treatment that holds the eastern position and deposits at column 13 would
  be a better outcome, and an equality test would score it FAIL. The
  one-sided form tests the thing §12.4 measured and cannot punish
  improvement. This asymmetry is declared here, before the run, so it cannot
  be read as a post-hoc relaxation.

**Bit 2 — CLOSING.** Both conjuncts must hold.

- **(a)** At a d18-class restore the treatment does **not** abandon the
  certified-adjacent position: at least one
  `relational_navigation_restore_selected` with **`differs: true`** and
  `selected_distance < baseline_distance`. (Identical evidence shape to E3's
  bit 1, so the two experiments are directly comparable.)
- **(b)** The treatment's **minimum distance** to `(12,11)` over the window
  is `<=` the control's, on **both** metrics. E3's treatment failed exactly
  here: Chebyshev 3 against the control's 1.

**Bit 3 — OUTCOME.** Within the 24-decision window the treatment's committed
trajectory collects `(12,11)` / pixel `(192,176)` — evidenced by `[192,176]`
entering `human_prior_collected_heart_slots` on a `decision_committed` event
— and the control does **not**. If both collect it, bit 3 FAILS: the
discriminator would be dead and a speedup is not the claim. The metric is
the milestone cell only, never affordance counts.

**Bit 4 — SAFETY.** The treatment records no more
`human_prior_life_loss_confirmed` committed decisions than the control
within the window.

All four must pass. **ANY mixed outcome = FAIL.** No weight tuning, no
budget re-sizing, no rerun on an identical negative result.
`budget_exhausted` or `hold_violated` termination of the exploit is a
**FAIL, not a VOID**.

Reported invariants (not bits): hold integrity; per-arm verified-branch
counts; per-arm option-search counts; the S2 exercised-difference counts;
and the full `hold_matching_candidates` series across the treatment's
restores (E3's §12.4(4) westward ratchet, measured again).

### 13.5 The reading that is fixed now, before the result

**If bit 1 passes and bit 3 fails, that is not a null — it is a narrowing.**
It would localize the remaining gap to what happens **after** the position
is held: exploration was preserved, the closing refusal fired, the agent
kept the certified-adjacent ground, and it still did not step onto the
milestone. That result would move the question from "can the planner hold a
target through a stagnation restore?" (answered yes) to "what does the agent
do with a held adjacency?" — which is a representation/affordance question,
not a search-scheduling one, and is a strictly more informative outcome than
either a redundancy null or an E3-style supply collapse. It is written here
so it cannot be narrated as a near-miss afterwards.

Two further readings are fixed in advance:

- **No-opportunity null.** If the treatment emits zero
  `relational_navigation_restore_selected` events in the window, the lever
  never got an instant — the §4.45 seam-opportunity family, **not** the
  §4.43/§4.45 agreement family. Bits 1(a) and 2 would then be scored on a
  trajectory the mechanism never touched, and the result is reported as
  "the closing instant did not recur", with the arms' identity as the
  evidence.
- **Agreement null.** If every such event carries `differs: false`, that is
  a **third redundancy finding** of the §4.43/§4.45 family and must be
  reported as such — the lever agreed with the incumbent everywhere — and
  explicitly not as under-powering and not as grounds for a fourth lever.

### 13.6 Declared caveats and blind spots (before the run)

**(a) Frozen-signature caveat (§11.5(a), unchanged).** The configuration
signature an archive carries is frozen at deposit, not recomputed. Low risk
here for the same reason as E3 (the held configuration is the removal), and
E3's §12.6 recorded that it did not bite. Declared, not assumed away.

**(b) Decision-1 empty-seed blind spot (§11.5(b), unchanged).** At the root
the tracked configuration is `prepush-root-empty-track-unmatchable`; no
objective can publish target cells before the first restore onto a removal
branch. Nothing is scored on d1.

**(c) Other selection-authority effects are NOT ablated.** The seam selector
governs only the two WP8 navigation seams. The restore-archive preference
(`_relational_restore_preference_active`), the reach-cells reserve family
and the reproduce-transition reorder remain live in the treatment, exactly
as in E3. If bit 1(a) fails, those are the first suspects, and the failing
decision index is reported so the site can be identified — a bit-1(a)
failure is a real FAIL, not a VOID, because "S2 alone leaves exploration
untouched" is precisely what E5 claims.

**(d) The control is expected to be redundant with v334.** It is run anyway,
first, and its failure to reproduce v334 state-for-state is a **VOID** (V6),
because a control that has drifted cannot license any comparison.

**(e) Speedup-vs-capability (§11.5(d), unchanged).** Even a clean PASS
demonstrates *finishing* under a held configuration at this root; it does
not demonstrate a second manipulation. The `(8,4)`/`(9,12)` hearts remain
the capability-level discriminator.

**(f) What a PASS does not show (§6.4(f), unchanged).** It does not show the
planner can cause a search, and does not close §4.45's mechanism 1. That
remains E4.

### 13.7 VOID conditions (a VOID is not evidence)

1. **Config inequality** — the arms' manifest `planning_config` differ in
   any field except `relational_planner_authority` and
   `relational_planner_enabled`. In particular both must report
   `relational_navigation_seams: "restore_only"`.
2. **Records inequality** — both arms must report `record_count: 3`,
   content signatures `15604cb5…`/`37ea410d…`/`47975c94…`, store digest
   `cf01a67a…`, at `verified_accessibility_weight: 0.0`.
3. **Seeding defect** — no archived branch carrying `85fd9014d58deb42`
   within the window in **either** arm.
4. **Root defect** — either manifest's `episodic_resume` block does not
   record source run `entity-v318-room3-known-push-connected-mask-d2`,
   `source_decision: 1`, `state_source_checkpoint_event_seq: 2026`,
   `state_source_events_sha256: 0bbe1d15…`.
5. **Budget defect** — either arm exceeds the 10,800 s wall ceiling and is
   killed before `run_finished`; **or** the arms' verified-branch counts
   differ by more than 1%.
6. **Control-invariance defect** — the control's 24 committed state ids do
   not reproduce **v334's** exactly. v334's own equality with v333 is
   re-checked and reported alongside.

Budget-exhausted non-reach is **censored**, never reported as
"unreachable" (learnings §2, §4.14).

### 13.8 Scoring

A single deterministic scorer walks each arm's `events.jsonl` once, applies
§13.4 verbatim, and writes `experiments/lolo1-wp5/e5-gate4-report.json` with
a canonical-JSON `digest_sha256` over the body (digest field excluded). It
is run end-to-end twice; both reports must be byte-identical and the digest
is recorded in the results section below. It is validated against **v334**
before scoring E5: it must reproduce §12.3's control trace
(`6 4 4 3 4 5 5 5 3 3 4 3 4 4 3 2 1 4 5 5 5 4 5 6`) and its "never collected
`(192,176)`" reading exactly.

Distance metrics, as in §11.7: **Chebyshev** `max(|dx|,|dy|)` for the
§4.47/§4.48-comparable traces, **Manhattan** `|dx|+|dy|` for the mechanism's
own `baseline_distance`/`selected_distance`. Bit 2(a) is scored on the
mechanism's own Manhattan figures as emitted; bit 2(b) must hold on both;
bit 3 is metric-free. Distance traces for **v334, v336 and v337** are
reported side by side.

---

## 14. E5 results (2026-08-18) — **FAIL**, and the narrowing §13.5 pre-declared

Scored against §13.4 verbatim by `experiments/lolo1-wp5/e5-gate4-report.json`,
`digest_sha256` **`3c6b1e180895e148868c5ae1f6184e3548a3282d4c7c727f8e6af9ff54134017`**
(scorer run end-to-end twice; the two reports are byte-identical, and the
recorded digest recomputes over the body). The scorer was validated against
v334 first and reproduces §12.3's control trace and its "never collected
`(192,176)`" reading exactly.

| Arm | Run id | Wall | Events | Branches | Searches | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Control | `entity-v336-room3-e5-control-off-d24` | 2,010 s | 85,594 | 12,232 | 1 (+9 deferred) | reached distance 1 at d17, diverged to 6 by d24; `(12,11)` never collected |
| Treatment | `entity-v337-room3-e5-treatment-restore-only-d24` | 2,010 s | 85,881 | 12,232 | 1 (+9 deferred) | reached distance 1 at **d16**, fell to 4 at d17; `(12,11)` never collected |

### 14.1 Verdict

| Bit | Verdict | Evidence |
| --- | --- | --- |
| **1 — SUPPLY PRESERVED** | **PASS** | Arms state-for-state identical for d1–d7 (first contested restore is d8; first divergence is *also* d8 — nothing forked the trajectory before the intervention). Archive geography **columns 6–12 in both arms** (44 deposits control, 43 treatment). E3's collapse to columns 6–8 / 33 deposits does not recur. |
| **2 — CLOSING** | **PASS** | 8 navigation restore instants, **2 with `differs: true`**, both strictly nearer: d8 `(7,6)` d=10 → `(9,8)` d=6; **d17 `(10,7)` d=6 → `(12,7)` d=4**. Minimum distance 1/1 in both arms — equal, and reached one decision *earlier*. |
| **3 — OUTCOME** | **FAIL** | Neither arm collected `(12,11)`/`(192,176)`. |
| **4 — SAFETY** | **PASS** | Zero `human_prior_life_loss_confirmed` commits in both arms. |

**Mixed ⇒ FAIL**, per §13.4. **No VOID condition fired** (V1–V6 all clear).
No tuning, no re-size, no rerun.

### 14.2 The instruments that make this FAIL trustworthy

- **V6 control invariance, in vivo**: v336 reproduced v334 **state-id for
  state-id across all 24 decisions** (`matching_prefix_decisions: 24`) and
  emitted the identical 85,594 events. v334's own equality with v333 is
  re-confirmed in the same pass. The seam selector at authority `off` leaks
  nothing, exactly as the unit test claimed.
- **V1**: the arms' `planning_config` differ in exactly
  `relational_planner_authority` and `relational_planner_enabled`; **both**
  report `relational_navigation_seams: "restore_only"`.
- **V5**: both arms verified exactly 12,232 branches, relative gap 0.0; both
  reached `run_finished`; both ran 2,010 s against a 10,800 s ceiling.
- **S1 structurally absent**: the treatment emitted **zero**
  `relational_navigation_choice` and zero `relational_navigation_declined`
  events. The commit ladder was never touched. The ablation is real, not
  nominal.
- **Not a redundancy null and not a no-opportunity null** (§13.5): 8
  instants, 2 differing. Both pre-declared null readings are inapplicable.
- **Search health identical**: 1 completed + 9 deferred option searches in
  both arms (E3's treatment had 11 deferred).

### 14.3 What E5 fixed, measured against E3

E5's hypothesis about *E3's* failure is **confirmed**. Every quantity §12.4
named as the supply collapse is restored:

| Quantity | E3 treatment (v335) | E5 treatment (v337) | Control (v336) |
| --- | --- | --- | --- |
| First divergence | d6, via the S1 **commit tier** | d8, via the S2 **restore key** | — |
| Archive columns | **6–8** | **6–12** | 6–12 |
| Archive deposits | 33 | 43 | 44 |
| `hold_matching_candidates` | 1,4,3,2,1,1,1,1,1 (westward ratchet) | 1,3,3,6,6,5,4,3 | — |
| Minimum distance (Cheb/Manh) | 3 / 6 | **1 / 1** | 1 / 1 |

Steering was the whole of E3's damage. Removing S1 removed all of it: the
treatment deposits the same ladder, restores eastward rather than westward,
and reaches the same closest approach the control does — **one decision
sooner**, because the d8 refusal saved a wasted excursion.

### 14.4 The named FAIL mechanism: **the lever is attached to the wrong object**

The closing refusal fired at exactly the preregistered instant and still did
not finish. The chain is measured:

1. **The d18-class instant recurred, and the incumbent's move was
   identical.** At d17 the treatment's stagnation restore reported baseline
   `(10,7)` at distance 6 — *precisely* the control's d18 move to `(10,7)`,
   §4.48's failure instant reproduced cell-for-cell.
2. **The objective refused it and took the best available alternative**:
   `(12,7)` at distance 4, from a supply of 6 hold-matching candidates. It
   lost 3 cells instead of 5.
3. **But the position it was standing on was never a candidate.** At d16 the
   treatment stood at `(12,10)`, distance 1. `(12,10)` **was never
   deposited as an archive** — in *either* arm. The easternmost, nearest
   deposit in both arms is `(12,9)`, distance 2. The full column-12 deposit
   set is `(12,6)`, `(12,7)`, `(12,8)`, `(12,9)` and nothing beyond.
4. **Therefore no restore key could have held it.** The target-aware key
   re-ranks the archive candidate set; the cell that needed holding was the
   *current* position, which is not in that set and cannot be. The
   intervention can choose **where the agent goes** when a stagnation
   restore fires; it has no expression for **whether the restore fires at
   all**.

This is the §13.5 narrowing, arriving in a sharper form than anticipated.
The preregistration expected "the position is held and the agent still fails
to step onto the milestone". What actually happened is one level earlier:
the position was never *holdable* by this mechanism, because holding a
current position is not an operation the restore key can express. The
`differs: true` at d17 is a genuine improvement over the incumbent — and it
is an improvement in the *choice of retreat*, not a refusal to retreat.

### 14.5 What this localizes

- **Not exploration.** Bit 1 settles it: with S1 off, an S2-only
  intervention leaves the archive ladder intact (columns 6–12, 43 vs 44
  deposits) and reaches the same minimum distance. Roadmap §20's
  "intervene at closing instants, leave exploration untouched" is
  *implementable* and was implemented.
- **Not the restore ranking.** The key works, is exercised twice, is
  correctly gated, chooses strictly nearer both times, and is bit-identical
  when off. There is nothing to tune here — §13.4's no-tuning clause covers
  exactly this, and §14.4(3) shows tuning could not help: the needed
  candidate does not exist.
- **The gap is the unconditional stagnation restore.** The agent reached
  distance 1 and was moved away by `human_prior_graph_stagnation` in *both*
  arms and in *all four* runs (v333/v334/v336 at d18, v337 at d17). No
  re-ranking of restore destinations can prevent a departure; only a
  mechanism that can **decline to restore**, or that **archives the
  certified-adjacent position so it becomes a candidate**, can.

### 14.6 Caveats, restated after the fact (none rescues the result)

- **Frozen signature (§13.6(a))**: did not bite. `hold_matching_candidates`
  grew monotonically to 6 and no bit turned on a signature identity.
- **Decision-1 blind spot (§13.6(b))**: as declared; the first navigation
  restore instant is d5 and nothing is scored on d1.
- **Other selection-authority effects not ablated (§13.6(c))**: this
  mattered less than feared. Bit 1(a) passed, and the first divergence
  coincides exactly with the first contested restore (both d8), so no
  un-ablated effect forked the trajectory.
- **`budget_exhausted` at d15**: the exploit hypothesis exhausted its
  12-decision budget at d15 and re-proposed at d16, holding authority
  through d24 — and the d17 refusal fired under the re-proposed hypothesis.
  Per §13.4 this is a FAIL input, not a VOID, and it is not the operative
  cause: the lever had authority at the decisive instant and used it.
- **Speedup-vs-capability (§13.6(d))**: there *was* a one-decision speedup
  (distance 1 at d16 vs d17), which §11.5(d)/§13.6(d) already declared
  insufficient. It is reported, not claimed.
- **Not a power problem.** The treatment did not run out of decisions; it
  was moved off the target at d17 with seven decisions remaining and never
  returned. A larger `--decisions` extends a trajectory that has already
  left.

### 14.7 Consequences for the plan

- **The S2-only shape is vindicated as a *shape* and refuted as a
  *sufficient mechanism*.** Roadmap §20's amendment stands and is
  strengthened: an intervention that leaves exploration untouched is
  buildable and does not starve supply. E5 is the existence proof.
- **The next lever must act on the restore *decision*, not its
  destination** — either a veto on stagnation-restoring away from a
  certified-adjacent position, or depositing the certified-adjacent position
  as an archive so the existing key can reach it. The second is strictly
  smaller and reuses the seam that already works. Neither is a new
  preference weight.
- **E4 (search request) is unaffected by this result** and remains the
  separate question of causing a search.
- Retained for reuse: the discriminator (still unbroken — now across
  thirteen runs, none of which has collected `(12,11)`), the
  v333/v334/v336 invariance chain, the seam selector (default-preserving,
  mutation-tested), and the S4 telemetry, which localized this FAIL to a
  single missing archive cell in one pass.

---

## 15. E6 preregistration — the certified-adjacent archive deposit (2026-08-18, **written before either arm ran**)

E5 (§13–§14) vindicated the closing *shape* and failed for a named
reason. §14.4 measured it: at d16 the treatment stood at `(12,10)`,
distance 1 from the certified milestone `(12,11)`, and **`(12,10)` was
never deposited as an archive in either arm** — the whole column-12
deposit set is `(12,6)`, `(12,7)`, `(12,8)`, `(12,9)`. The target-aware
restore key re-ranks archives *that exist*; the cell that needed holding
was the current position, which is not and cannot be a restore candidate.
No tuning could have helped, and §13.4's no-tuning clause forbids trying.

**E6's hypothesis.** Deposit that position. When the agent commits to a
cell that is **on or adjacent to** a certified milestone cell **while the
held configuration matches**, archive it, so the existing — already
working, already exercised, already correctly gated — S2 restore key has
a candidate to reach. This adds **no new preference weight** (roadmap §20
constraint): the ordering is E5's, unchanged. It adds a candidate.

### 15.1 The seam (the only code change)

A third value on the existing selector,
`NeuralPlanningConfig.relational_navigation_seams` /
`--relational-navigation-seams`:

| Mode | S1 commit tier | S2 restore key | S3 deposit | Meaning |
| --- | --- | --- | --- | --- |
| `both` (**default**) | on | on | off | today's behavior, exactly — E3's treatment |
| `restore_only` | off | on | off | **E5's treatment**, byte-identical to §13 |
| `restore_plus_deposit` | off | on | **on** | **E6's treatment**: E5 plus the deposit |
| `off` | off | off | off | every navigation seam inert under selection authority |

Seam **S3** is one gate helper
(`_relational_navigation_deposit_view`) and one block in `decide()`
placed **after every incumbent archive path and before
`_prune_archive`**, so within a decision it is strictly additive and is
subject to archive capacity exactly like every other branch. The gate
conjoins, in this order, and records a `reason` at every instant either
way:

1. the seam selector is `restore_plus_deposit` (every pre-E6 mode returns
   `None` here, before anything else is read);
2. `_relational_navigation_objective()` is live — which already requires
   selection authority, an active `reach_cells_under_hold` hypothesis,
   and published certified target cells;
3. the root object state's `tracked_world_state_signature` **equals the
   objective's hold signature** (§4.7's coupling: proximity means nothing
   outside the held configuration, and a deposit outside it would sort
   below every held candidate anyway);
4. the committed cell's distance to the nearest published target is
   `<= RELATIONAL_NAVIGATION_DEPOSIT_MAX_DISTANCE = 1`, in the
   mechanism's own metric (`relational_planner.target_cell_distance`,
   Manhattan) — "**on** the milestone cell, or **orthogonally adjacent**
   to it". This is a gate, not a weight, and it is a module constant, not
   a config field, so no manifest field moves;
5. the committed position is **certified** — the same commit-time
   predicate the reach-cells family already applies: never deposit ground
   bought with a life loss or a dark transition;
6. the position is not already archived by an incumbent path (recorded as
   `already_archived`; no duplicate candidate is ever created).

Telemetry: `relational_navigation_deposit_added` /
`relational_navigation_deposit_declined`, each carrying the cell, the
distance, the gate radius, the hold signature, the current configuration
signature, the published target cells, the deposited state id, the
archive size and the reason. A certified-adjacent instant that is
*declined* is exactly as visible in the log as one that deposits.

Behavior-preservation evidence, landed before the arms ran:

- Full suite **1085 tests OK, 4 skipped** — 1060 + 4 before this change,
  plus 25 new tests. Exactly one pre-existing test changed: the
  anchor-drift guard
  `CommitTimeArchiveConfigurationSignatureTests::test_every_commit_time_construction_carries_the_root_track`
  counts commit-time `_ArchivedBranch` constructions in `decide()` and
  moves 3 → 4. **Its invariant is unchanged and still asserted for all
  four**: every commit-time construction carries
  `_root_object_track_branch_fields()`, so the deposit is visible to the
  hold-gated restore supply (learnings §4.46). No assertion was weakened.
- **Cross-version byte-identity against HEAD `ffb58b2` (the pre-E6
  planner), run as a paired harness over a real `decide()`:** for all
  three pre-E6 seam values × all three authorities (9 arms), the pre-E6
  planner and the E6 build produce **identical event streams, identical
  committed decisions and identical archives**. Digests, pre-E6 = E6 in
  every row: authority `off` `a7522ed1659278ca` (all three modes);
  `telemetry` `33e358a78d965420` (all three); `selection` `both`
  `9265f925c6f40688`, `restore_only`/`off` `b7a54aa571613b50`.
- `tests/test_ensemble_planner.py::RelationalNavigationDepositGateTests`
  (9 tests): the selector accepts the new value and still rejects junk;
  the four modes map to `(S1, S2, S3)` exactly
  (`both`→(T,T,F), `restore_only`→(F,T,F), `restore_plus_deposit`→(F,T,T),
  `off`→(F,F,F)); the gate is `None` for every pre-E6 mode, `None` at
  authority `off`/`telemetry`, and `None` with no published objective;
  distances 0 and 1 are eligible while 2 and 4 are refused; a departed
  configuration, an uncertified position and an unavailable position are
  each refused with their own reason.
- `tests/test_ensemble_planner.py::RelationalNavigationDepositSeamTests`
  (10 tests) drives a **real `decide()`**: the deposit reaches the archive
  carrying the hold signature and emits its evidence; `both`,
  `restore_only` and `off` deposit nothing and commit the identical
  decision with the identical archive (and `restore_only`/`off` agree
  event-for-event); E6 differs from E5 by **exactly one archive, one
  event, and the two archive counters those events report** — an
  exhaustive-key comparison, so any third field that moved would fail;
  the `already_archived` decline is exercised; a far position is recorded
  and not deposited; S1 stays structurally absent; S2 is bit-identical to
  `restore_only`; a deposit-shaped candidate with the **lowest** plain
  score in the archive is what the S2 key restores over the
  higher-novelty distant branch; and the selector is inert at
  `off`/`telemetry` across all four modes, compared on archives as well
  as events.
- An anchor-drift-proof AST test pins the *placement*: exactly one deposit
  gate, after all three incumbent commit-time constructions, before the
  single `_prune_archive`, with exactly one construction between.
- `tests/test_relational_planner.py::NavigationDepositContractTests`: the
  pure module exposes no deposit seam and no new preference — the key
  already ranks the position E6 makes available, so S3 changes *who
  deposits a candidate*, never *what the module publishes*.
- **Mutation-tested, six mutations, all killed**: forcing the seam
  selector `True` fails 8 tests; widening the adjacency radius fails 5;
  removing the hold-configuration check fails 1; removing the
  certification check fails 2; removing the duplicate dedupe fails 1;
  emitting the event without appending the archive fails 2.

### 15.2 Arms

- **Control** — `--relational-planner-authority off`, run id
  `entity-v338-room3-e6-control-off-d24`. Runs **first**.
- **Treatment** — `--relational-planner-authority selection`, run id
  `entity-v339-room3-e6-treatment-deposit-d24`.
- **Both arms pass `--relational-navigation-seams restore_plus_deposit`**,
  so the arms differ in **exactly two** manifest fields —
  `relational_planner_authority` and `relational_planner_enabled` —
  exactly as in E3 and E5. This is legitimate only because the selector
  is provably inert at authority `off`; that inertness is unit-tested
  (`RelationalNavigationDepositGateTests::test_gate_is_none_outside_selection_authority`
  and `RelationalNavigationDepositSeamTests::test_selector_is_inert_outside_selection_authority`,
  the latter comparing archives as well as event streams) and
  cross-checked against the pre-E6 planner, not assumed.
- Every other flag is **v336/v337's manifest profile, verbatim** (§11.3's
  command line): `--decisions 24`, `--relational-decision-budget 12`,
  `--human-prior-accessibility-records
  experiments/lolo1-wp5/wp8lite-accessibility-records.json`,
  `--human-prior-accessibility-preference-weight 0.0`,
  `--log-root experiments/lolo1-entity-v10/evaluations`.
- **Root**: identical to v333/v334/v335/v336/v337 — memory
  `entity-v318-room3-known-push-connected-mask-d2` decision 1 with
  `--resume-option-search`; physical state the same run's seq-2026
  checkpoint, `state_source_events_sha256 0bbe1d15…9b6f83`.
- **Ceilings**: 10,800 s wall per arm under an external watchdog; one
  native run at a time.
- **Scoring window**: the first **24 committed decisions**.

Input digests, re-verified on disk today and equal to the v336 manifest:
host `c03694c5…3e891f3`, core `a3450a09…5a40024886`, ROM
`914c6769…3efd059e01`, neural checkpoint `bb7a7a37…284f678b9`,
entity-behavior checkpoint `984b83c3…25c7c6aa`, record store
`cf01a67aca2b6e8feeab38c0c85520dec2470cba2a5f2257cd817912c204d1fe`.
HEAD at preregistration `ffb58b2`; working tree clean except the four
owned files of this change (`lolo_agent/neural_planner.py`,
`lolo_agent/neural_run.py`, `tests/test_ensemble_planner.py`,
`tests/test_relational_planner.py`), the unrelated untracked `tmp/`, and
concurrent documentation edits by a parallel session (`README.md`,
`docs/architecture.md`, `docs/telemetry.md`, and two new untracked
`docs/*-2026-08-17.md` files) — **none of which is an input to either
arm**, and none of which this change touches.

### 15.3 Exact command lines

§11.3's block verbatim, with two flags changed and one added:

```
  --run-id entity-v338-room3-e6-control-off-d24            (control)
  --run-id entity-v339-room3-e6-treatment-deposit-d24      (treatment)
  --relational-navigation-seams restore_plus_deposit       (BOTH arms)
  --relational-planner-authority <off | selection>
```

### 15.4 The four preregistered bits (fixed; **ANY mixed outcome = FAIL**)

Let **C** be the decision index of the treatment's **first contested
instant**: the earlier of (i) the first
`relational_navigation_restore_selected` carrying `differs: true` and
(ii) the first `relational_navigation_deposit_added`. Both are instants
at which the intervention can change something; E6 adds (ii), which E5
did not have. If neither exists, **C is undefined** and bit 1(a) is
scored over all 24 decisions.

**Bit 1 — SUPPLY.** Both conjuncts must hold.

- **(a) Trajectory prefix.** For every committed decision `d < C` (all 24
  if C is undefined), the treatment's `committed_state_id` equals the
  control's. With S1 off, nothing may fork the trajectory before the
  intervention fires.
- **(b) Archive geography not narrowed.** The treatment's archive-deposit
  column range must not be narrower than the control's:
  `treatment_min_column <= control_min_column` **and**
  `treatment_max_column >= control_max_column`, with the E5 reference
  range **columns 6–12** reported alongside. Both raw ranges and both
  full deposit-cell histograms are reported either way. The one-sided
  form is §13.4's, unchanged and for the same reason: E3's failure was
  narrowing, and a treatment that deposits further east must not be
  scored FAIL for it. The S3 deposits are counted **separately** as well
  as inside the total, so the geography claim can be read with and
  without them.

**Bit 2 — CANDIDATE EXISTS.** All three conjuncts must hold, within the
24-decision window.

- **(a)** At least one `relational_navigation_deposit_added` event.
- **(b)** That event's `deposit_distance <= 1` and its
  `hold_configuration_signature` equals both the objective's hold
  signature and the arm's `configuration_signature` at that instant — the
  deposit is certified-adjacent **and inside the held configuration**, so
  the hold-gated restore supply can see it at all.
- **(c)** The deposited branch appears as a **hold-matching candidate at
  a later restore**: some `relational_navigation_restore_selected` after
  the deposit reports the deposited cell among the arm's archive, and
  `hold_matching_candidates` at that instant is strictly greater than the
  count E5 recorded at the corresponding instant, **or** the deposit is
  itself selected. Evidence reported either way: every deposit event with
  its cell, distance, decision and hold signature; every decline with its
  reason; and the full post-deposit restore series.

**Bit 3 — OUTCOME.** Within the 24-decision window the treatment's
committed trajectory collects `(12,11)` / pixel `(192,176)` — evidenced
by `[192,176]` entering `human_prior_collected_heart_slots` on a
`decision_committed` event — and the control does **not**. If both
collect it, bit 3 FAILS: the discriminator would be dead and a speedup is
not the claim. The metric is the milestone cell only, never affordance
counts.

**Bit 4 — SAFETY.** The treatment records no more
`human_prior_life_loss_confirmed` committed decisions than the control
within the window.

All four must pass. **ANY mixed outcome = FAIL.** No weight tuning, no
budget re-sizing, no radius re-sizing, no rerun on an identical negative
result. `budget_exhausted` or `hold_violated` termination of the exploit
is a **FAIL, not a VOID**.

Reported invariants (not bits): hold integrity; per-arm verified-branch
counts; per-arm option-search counts; the S2 exercised-difference counts;
the full `hold_matching_candidates` series across the treatment's
restores; the full deposit series with reasons; and minimum distance to
`(12,11)` on both metrics for **v336, v337, v338 and v339**.

### 15.5 The reading that is fixed now, before the result

**If bits 1–2 pass and bit 3 fails, that is not a null — it is again a
narrowing, and a sharper one than E5's.** It would establish that the
candidate existed, was inside the held configuration, and was reachable
by the key that already works — and that collection still did not
follow. The question would then move off search scheduling entirely and
onto what the agent *does* with a reachable held adjacency. In that case
the report must state **what the agent did instead at the decisive
instant**: the restore that fired while the deposit was in the archive,
whether the deposit was in that restore's eligible set, whether the S2
key ranked it first, and — if it was dropped — **which filter dropped
it**, named from the arm's own telemetry. That is written here so it
cannot be narrated as a near-miss afterwards.

Three further readings are fixed in advance:

- **No-candidate null.** If the window contains zero
  `relational_navigation_deposit_added`, bit 2 FAILS and the reading is
  "the gate never saw a certified-adjacent committed position under
  hold". The declined events and their reasons are the evidence, and they
  distinguish the sub-cases (`not_certified_adjacent`,
  `held_configuration_absent`, `position_not_certified`,
  `position_unavailable`, `already_archived`) without any further run.
- **Deposited-but-ineligible finding.** If a deposit is added but never
  enters any later restore's candidate set, bit 2(c) FAILS and the result
  is reported as an **archive-eligibility** finding, naming the filter,
  not as a restore-key finding. §15.6(d) declares the specific filter
  already known to be capable of this.
- **Agreement null.** If every `relational_navigation_restore_selected`
  carries `differs: false`, that is a redundancy finding of the
  §4.43/§4.45 family and must be reported as such — and explicitly not as
  under-powering and not as grounds for a fifth lever.

### 15.6 Declared caveats and blind spots (before the run)

**(a) Frozen-signature caveat (§11.5(a)/§13.6(a), unchanged).** The
configuration signature an archive carries is frozen at deposit, not
recomputed. §14.6 recorded that it did not bite in E5. Declared, not
assumed away. It applies to the S3 deposit identically.

**(b) Decision-1 empty-seed blind spot (§13.6(b), unchanged).** At the
root the tracked configuration is
`prepush-root-empty-track-unmatchable`; no objective can publish target
cells before the first restore onto a removal branch. Nothing is scored
on d1, and S3 cannot fire there.

**(c) Other selection-authority effects are NOT ablated (§13.6(c),
unchanged).** The seam selector governs only the WP8 navigation seams.
The restore-archive preference, the reach-cells reserve family and the
reproduce-transition reorder remain live in the treatment. A bit-1(a)
failure is a real FAIL, not a VOID.

**(d) The immediate-restore blind spot — NEW, and declared now.**
`_restore_if_stagnant`'s `recovery_distinct` filter drops any archive
whose coarse frame signature equals the *current* frame's, unless the
branch also carries a frontier flag. A position deposited at decision `d`
is therefore invisible to a restore that fires at `d+1` while the agent
is still standing on it — which is a correct guard against a degenerate
self-restore, and is also exactly the geometry of E5's decisive instant
(deposit would be at d16; the stagnation restore fired at d17). **E6 may
therefore need the second restore after the deposit, not the first.**
This is a real limitation of the smallest possible change. It is written
here, before the run, so that a bit-2(c) or bit-3 failure attributable to
it is read as a measured property of the seam and not discovered
afterwards as an excuse.

**(e) Self-restore degeneracy — NEW.** If the deposit *is* restored, the
agent returns to a position it already occupied. If it then re-stagnates
from there, the same instant can repeat and consume decisions. The
committed trajectory trace will show it plainly. Either way, no tuning:
§15.4's no-tuning clause covers the radius and the gate as well as the
weights.

**(f) Archive capacity.** The deposit is subject to `_prune_archive` like
every other branch. At this profile capacity is 1024 against ~44
deposits, so pressure is nil — but the deposit is not privileged against
pruning, and that is deliberate.

**(g) Speedup-vs-capability (§11.5(d)/§13.6(e), unchanged).** Even a
clean PASS demonstrates *finishing* under a held configuration at this
root; it does not demonstrate a second manipulation. The `(8,4)`/`(9,12)`
hearts remain the capability-level discriminator.

**(h) What a PASS does not show (§6.4(f), unchanged).** It does not show
the planner can cause a search, and does not close §4.45's mechanism 1.
That remains E4.

### 15.7 VOID conditions (a VOID is not evidence)

1. **Config inequality** — the arms' manifest `planning_config` differ in
   any field except `relational_planner_authority` and
   `relational_planner_enabled`. In particular both must report
   `relational_navigation_seams: "restore_plus_deposit"`.
2. **Records inequality** — both arms must report `record_count: 3`,
   content signatures `15604cb5…`/`37ea410d…`/`47975c94…`, store digest
   `cf01a67a…`, at `verified_accessibility_weight: 0.0`.
3. **Seeding defect** — no archived branch carrying `85fd9014d58deb42`
   within the window in **either** arm.
4. **Root defect** — either manifest's `episodic_resume` block does not
   record source run `entity-v318-room3-known-push-connected-mask-d2`,
   `source_decision: 1`, `state_source_checkpoint_event_seq: 2026`,
   `state_source_events_sha256: 0bbe1d15…`.
5. **Budget defect** — either arm exceeds the 10,800 s wall ceiling and
   is killed before `run_finished`; **or** the arms' verified-branch
   counts differ by more than 1%.
6. **Control-invariance defect** — the control's 24 committed state ids
   do not reproduce **v336's** exactly. v336's own equality with
   v334/v333 is re-checked and reported alongside. A crashed arm is
   **VOID, not FAIL** (learnings §4.52).

Budget-exhausted non-reach is **censored**, never reported as
"unreachable" (learnings §2, §4.14).

**Health-checking the arms is by the run's own telemetry — event growth
and expected seq milestones — never by external process pattern matching
(learnings §4.52). The first `decision_committed` lands at seq ≈75,742 in
every arm of this family; zero commits at 6k events is on-profile, not a
death.**

### 15.8 Scoring

A single deterministic scorer walks each arm's `events.jsonl` once,
applies §15.4 verbatim, and writes
`experiments/lolo1-wp5/e6-gate4-report.json` with a canonical-JSON
`digest_sha256` over the body (digest field excluded). It is run
end-to-end twice; both reports must be byte-identical and the digest is
recorded in the results section below. It is validated against **v336 and
v337** before scoring E6: it must reproduce §14's control trace and
verdicts — including the "never collected `(192,176)`" reading, the
columns-6–12 geography with 44/43 deposits, and the 8 restore instants
with 2 differing — exactly.

Distance metrics, as in §11.7/§13.8: **Chebyshev** `max(|dx|,|dy|)` for
the §4.47/§4.48-comparable traces, **Manhattan** `|dx|+|dy|` for the
mechanism's own `baseline_distance`/`selected_distance`/`deposit_distance`.
Bit 2(b) is scored on the mechanism's own Manhattan figures as emitted;
bit 3 is metric-free. Distance traces for **v336, v337, v338 and v339**
are reported side by side.

---

## 16. E6 results (2026-08-18) — **FAIL**, on the §15.5 no-candidate null, with a mechanism §15.6 did not anticipate

Scored against §15.4 verbatim by `experiments/lolo1-wp5/e6-gate4-report.json`,
`digest_sha256` **`82aa4af8b5c18854f2e245083fb4b759a6d667a8b2afc7a63426a004998d8060`**
(scorer run end-to-end twice; the two reports are byte-identical, and the
recorded digest recomputes over the body). The scorer was validated
against v336/v337 first and reproduces §12.3's control trace, §14's
44/43 deposits over columns 6–12, its 8 restore instants with 2
differing, and its `hold_matching_candidates` series 1,3,3,6,6,5,4,3
exactly.

| Arm | Run id | Wall | Events | Branches | Searches | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Control | `entity-v338-room3-e6-control-off-d24` | 2,040 s | 85,594 | 12,232 | 1 (+9 deferred) | reached distance 1 at d17; `(12,11)` never collected |
| Treatment | `entity-v339-room3-e6-treatment-deposit-d24` | 2,021 s | 85,893 | 12,232 | 1 (+9 deferred) | reached distance 1 at d16; `(12,11)` never collected; **zero deposits** |

### 16.1 Verdict

| Bit | Verdict | Evidence |
| --- | --- | --- |
| **1 — SUPPLY** | **PASS** | Arms state-for-state identical d1–d7; first contested instant is d8 and first divergence is *also* d8 — nothing forked the trajectory before the intervention. Archive geography **columns 6–12 in both arms** (44 control, 43 treatment). E3's collapse to 6–8 does not recur. |
| **2 — CANDIDATE EXISTS** | **FAIL** | The gate ran at **12 instants and declined all 12**, every one with reason `not_certified_adjacent`. **Zero `relational_navigation_deposit_added` events.** Nearest declined position: d15 `(12,9)`, Manhattan 2. Conjunct (a) fails, so (b) and (c) are unreachable. |
| **3 — OUTCOME** | **FAIL** | Neither arm collected `(12,11)`/`(192,176)`. |
| **4 — SAFETY** | **PASS** | Zero `human_prior_life_loss_confirmed` commits in both arms. |

**Mixed ⇒ FAIL**, per §15.4. **No VOID condition fired** (V1–V6 all
clear). No tuning, no re-size, no radius change, no rerun.

This is the **no-candidate null** §15.5 fixed in advance — reported as
such, and not as under-powering.

### 16.2 The instruments that make this FAIL trustworthy

- **V6 control invariance, in vivo**: v338 reproduced v336 **state-id for
  state-id across all 24 decisions** and emitted the identical **85,594**
  events and the identical Chebyshev trace
  `6 4 4 3 4 5 5 5 3 3 4 3 4 4 3 2 1 4 5 5 5 4 5 6`. The chain is now
  four runs deep and re-verified in this pass: **v333 ≡ v334 ≡ v336 ≡
  v338**, 24/24 and 85,594 events at every link. The
  `restore_plus_deposit` selector leaks nothing at authority `off`,
  exactly as the unit test and the pre-E6 cross-version harness claimed.
- **V1**: the arms' `planning_config` differ in exactly
  `relational_planner_authority` and `relational_planner_enabled`;
  **both** report `relational_navigation_seams: "restore_plus_deposit"`.
- **V2**: both arms `record_count: 3`, signatures `15604cb5…`/
  `37ea410d…`/`47975c94…`, `verified_accessibility_weight: 0.0`.
- **V3**: 4 hold-signature archives and 23 hold-signature branches in
  **both** arms; the seeding is real.
- **V4**: both `episodic_resume` blocks record source run
  `entity-v318-room3-known-push-connected-mask-d2`, `source_decision: 1`,
  `state_source_checkpoint_event_seq: 2026`,
  `state_source_events_sha256: 0bbe1d15…`.
- **V5**: both arms verified exactly **12,232** branches, relative gap
  0.0; both reached `run_finished`; both ran ~2,030 s against a 10,800 s
  ceiling.
- **The treatment is E5's treatment plus twelve log lines.** v339
  reproduces v337 **state-for-state across all 24 decisions**: identical
  Chebyshev and Manhattan traces, identical 9 restores, identical 8
  navigation-restore instants with the same 2 differing, identical
  `hold_matching_candidates` series `1,3,3,6,6,5,4,3`, identical 43
  deposits over columns 6–12. Event count **85,893 = 85,881 + 12**, the
  12 being exactly the declined-deposit events. Seam S3 changed **no**
  behavior because it deposited nothing — which is the finding, and is
  also the cleanest possible demonstration that the seam is inert when
  its gate refuses.
- **Search health identical**: 1 completed + 9 deferred option searches in
  all of v336, v337, v338, v339.
- **Provenance note, recorded rather than tidied away.** §15.2 declared
  concurrent documentation edits by a parallel session in the working
  tree at preregistration time. Those landed as commit `727ef7c` while
  the E6 arms were running, moving HEAD from `ffb58b2` to `727ef7c`. The
  commit touches **only** `README.md`, `docs/architecture.md`,
  `docs/telemetry.md` and two new `docs/*-2026-08-17.md` files — no code
  and no run input. The code under test in both arms was `ffb58b2` plus
  the four owned files of this change, exactly as §15.2 states.
  `experiments/` is git-ignored, so `e6-gate4-report.json` lives on disk
  beside `e5-gate4-report.json` under the same convention.

### 16.3 Distance traces, four runs side by side

Chebyshev (the §4.47/§4.48 metric), 24 committed decisions:

```
v336 control   6 4 4 3 4 5 5 5 3 3 4 3 4 4 3 2 1 4 5 5 5 4 5 6
v338 control   6 4 4 3 4 5 5 5 3 3 4 3 4 4 3 2 1 4 5 5 5 4 5 6
v337 E5 treat  6 4 4 3 4 5 5 3 3 4 3 4 4 3 2 1 4 5 6 4 5 5 5 5
v339 E6 treat  6 4 4 3 4 5 5 3 3 4 3 4 4 3 2 1 4 5 6 4 5 5 5 5
```

Manhattan (the mechanism's own metric):

```
v336 control   9 7 7 6 8 9 10 10 6 5 6 4 5 4 3 2 1 6 7 6 6 4 5 6
v338 control   9 7 7 6 8 9 10 10 6 5 6 4 5 4 3 2 1 6 7 6 6 4 5 6
v337 E5 treat  9 7 7 6 8 9 10 6 5 6 4 5 4 3 2 1 4 5 6 6 7 6 6 6
v339 E6 treat  9 7 7 6 8 9 10 6 5 6 4 5 4 3 2 1 4 5 6 6 7 6 6 6
```

Minimum distance 1/1 in every arm. The controls reach it at d17, both
treatments at d16.

### 16.4 The named FAIL mechanism: **the objective is dead for exactly the decision that arrives**

The chain is measured end to end from the treatment's own telemetry:

1. **The gate ran, and refused honestly, 12 times.** Decisions 4, 6, 7,
   9, 10, 12, 13, 15, 18, 19, 21, 22 each emitted
   `relational_navigation_deposit_declined` with reason
   `not_certified_adjacent`. The nearest was **d15 at `(12,9)`,
   Manhattan 2** — one cell outside the gate.
2. **The one decision that stood on a certified-adjacent cell emitted
   nothing at all.** At **d16 the treatment committed to `(12,10)`,
   Manhattan distance 1** — and there is **no gate event of either kind**
   at d16. `_relational_navigation_deposit_view` returned `None`, which
   it does only when the seam is unselected (it was selected) or when
   `_relational_navigation_objective()` is `None`.
3. **Why the objective was `None`.** At d15, seq 82132, the exploit
   hypothesis — the one whose payload is
   `{"hold_configuration_signature": "85fd9014d58deb42", "target_cells": [[12,11]]}` —
   **terminated with `budget_exhausted`**. At d16, seq 82139, the
   re-proposal activated a **`hold_configuration`** hypothesis whose
   realization is also `reach_cells_under_hold` but whose payload is
   `"target_cells": []` — **empty by construction** (§13.6; the pure
   module's `objective_target_cells` returns `()` for it, and every
   consumer then fails open). No cells published ⇒ no objective ⇒ S3
   inert.
4. **The exploit objective came back three events too late.** It was
   re-activated at **seq 82664**; the d16 commit is at **seq 82661**. The
   hypothesis that names `(12,11)` was alive for d4–d15 and again from
   the tail of d16 onward — and dead for precisely the commit that
   reached distance 1.
5. **Every other conjunct of the gate held at d16.** The d16
   `relational_hypothesis_proposed` and `relational_hypothesis_achieved`
   events both report `configuration_signature: 85fd9014d58deb42` — the
   hold signature — so the hold check would have passed; the position was
   certified (zero life losses all run); and the distance was 1, inside
   the radius. **Objective liveness was the only failing conjunct.**
6. By d17 the stagnation restore had moved the agent to `(12,7)`, and it
   never returned to distance 1 in the remaining seven decisions.

Stated exactly: **E5 showed the lever was attached to the wrong object.
E6 attached it to the right object and found the object is not there at
the moment it is needed.**

### 16.5 What this localizes — and what it does not

- **Not exploration.** Bit 1 settles it again: columns 6–12, 43 vs 44
  deposits, first divergence coinciding exactly with the first contested
  instant.
- **Not the restore ranking.** Unchanged from E5 and re-measured
  identically: 8 instants, 2 differing, both strictly nearer, supply
  ratcheting 1,3,3,6,6,5,4,3.
- **Not §15.6(d).** The immediate-restore blind spot was declared in
  advance as the most likely spoiler and **is not the operative cause** —
  `recovery_distinct` never had anything to filter, because nothing was
  ever deposited. The declaration is scored here rather than quietly
  dropped.
- **Not the radius, and widening it is forbidden and would not help.**
  §15.4 bars re-sizing, and the measurement shows the bar costs nothing:
  a wider gate would have deposited d15's `(12,9)` (distance 2) and d13's
  `(12,7)` (distance 4) — but `(12,7)`, `(12,8)` and `(12,9)` were
  **already** archive candidates in both arms, and the d17 restore
  **already selected `(12,7)`**. A wider radius supplies nothing the
  restore key did not already have.
- **The newly localized gap is hypothesis lifetime, not archive supply.**
  The exploit objective's 12-decision budget expired one decision before
  the agent arrived, and the re-establish step that follows publishes an
  empty target set by design. Any mechanism gated on an active
  cell-publishing objective is blind at exactly that boundary — and the
  boundary is not incidental: the budget is consumed *by* the approach
  that produces the arrival.

### 16.6 Caveats, restated after the fact (none rescues the result)

- **Frozen signature (§15.6(a))**: not reached; no bit turned on a
  signature identity.
- **Decision-1 blind spot (§15.6(b))**: as declared — d1–d3 produced no
  gate event.
- **Other selection-authority effects (§15.6(c))**: bit 1(a) passed and
  the first divergence coincides with the first contested instant (both
  d8), so no un-ablated effect forked the trajectory.
- **Immediate-restore blind spot (§15.6(d))**: declared, **not reached**.
  See §16.5.
- **Self-restore degeneracy (§15.6(e))**: not reached; nothing was
  deposited, so nothing could be self-restored.
- **Archive capacity (§15.6(f))**: capacity 1024 against ~43 deposits;
  no pruning pressure; not operative.
- **`budget_exhausted` at d15**: §15.4 makes this a FAIL input, not a
  VOID — and unlike E5, where §14.6 correctly judged it was *not* the
  operative cause, **here it is exactly the operative cause**. That
  difference is the whole of E6's contribution.
- **Not a power problem.** The treatment did not run out of decisions; it
  was moved off the target at d17 with seven decisions remaining. A
  larger `--decisions` extends a trajectory that has already left.
- **Speedup-vs-capability (§15.6(g))**, **what a PASS does not show
  (§15.6(h))**: unchanged, and moot on a FAIL.

### 16.7 Consequences for the plan

- **E6 is a FAIL.** The deposit seam is correct, mutation-tested,
  default-preserving, provably inert at authority `off` and at every
  pre-E6 selector value, and it deposited nothing — because its gate is
  conditioned on an objective that was not alive at the decisive instant.
- **Five levers are now measured, each falsified for a distinct named
  reason**: restore preference (redundant), search reserve (no searches),
  commit steering (starves supply), restore key alone (target not a
  candidate), certified-adjacent deposit (no objective at the arriving
  decision).
- **The next lever is not chosen here**, and choosing it needs a decision
  this experiment is not entitled to make: whether a mechanism may read
  certified milestone cells directly from the records rather than from an
  active hypothesis's payload. That would fire without a live hypothesis
  and is a **different intervention class** — roadmap §20's "no new
  preference" constraint would still hold, but the "objective-scoped"
  property that every WP8 seam has relied on would not. The alternative
  framing is that **hypothesis lifetime**, not archive supply, is now the
  object of study.
- **E4 (search request) is unaffected** and remains the separate question
  of causing a search.
- **The discriminator remains unbroken across fifteen runs**, none of
  which has collected `(12,11)`.
- Retained for reuse: the seam selector (now four-valued,
  default-preserving, mutation-tested, cross-version byte-identity proven
  against the pre-E6 planner), the S3 decline telemetry — which localized
  this FAIL to a three-event window in a single pass — and the
  v333/v334/v336/v338 invariance chain.
