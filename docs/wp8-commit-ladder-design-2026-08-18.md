# WP8 commit-ladder rewrite — design and E8 preregistration skeleton

**Date**: 2026-08-18
**Trigger**: roadmap §22 (REWRITE TRIGGER A FIRED), learnings §4.54
**Predecessors**: `docs/wp8-search-scheduling-design-2026-08-17.md` (Q2/§3.2
enumerated the non-search commit path), `docs/wp8-lifecycle-design-2026-08-17.md`
(§7.3 stated the trigger, §10 preregistered E7)
**Planner HEAD at recon**: `fefbca7` ("Fire rewrite trigger A: the last step has
no actuator"), `lolo_agent/neural_planner.py` at 29,142 lines
**Status**: RECON + DESIGN ONLY. No code changed. No experiment run. No bit
scored. Every line anchor below was re-verified against the file at `fefbca7`
by direct read (the §4.46 stale-anchor correction is the reason this sentence
exists — the predecessor doc's `:23180–23500` ladder anchors are now **stale**
by roughly 700 lines and must not be copied forward).

---

## 0. The headline, stated before the evidence

**The answer to the central question is neither of the two options the trigger
posed.** It is not "the actuator is missing", and it is not "the actuator exists
but the ranking buried it". It is a third thing, and it changes the rewrite:

> **The P5 commit ladder was never entered at the decisive instant.** At v341's
> d17 and d18 — the only two decisions in the entire E-series at which the agent
> *began* a decision standing adjacent to `(12,11)` — `decide()` returned early
> at `neural_planner.py:21702–21704` through `_restore_if_stagnant()`, which
> executes **before** `plans = self.planner.plan(self.frame)` at `:21769`. No
> branches were generated. `branches_examined: 0`, and **zero
> `branch_verified` events** at either decision. There was no candidate set for
> any tier to rank, and therefore no tier — new or old — could have fired.

Two corollaries that the census makes exact:

1. Across the ten 24-decision runs of this family (v333–v342), there are **149
   expansion decisions and ZERO of them begin with the agent at Chebyshev
   distance ≤ 1 from `(12,11)`.** All **11** distance-≤1 decision-starts are
   restore decisions with `branches_examined: 0`. **A new commit-ladder tier
   gated on "standing adjacent to a certified milestone" has a measured
   opportunity count of exactly zero.** Adding one would be a sixth seam that
   cannot fire — the §4.45/§4.46 "no opportunity to act on" mechanism repeating,
   which §7.3 trigger C names as grounds to stop.
2. **The ladder already contains a tier that references a certified target
   cell** — Tier 4, seam S1, `:23932`, keyed on
   `-relational_target_cell_distance(...)`. Roadmap §22's "no tier referencing a
   target cell" is true of the *incumbent* ladder (S1 is off in E5/E6/E7) but
   not of the file. S1 exists, and it was measured and refuted in E3 for supply
   starvation (§4.50). "Add a target-referencing tier" is therefore both already
   built and already falsified in the only form that could fire at expansion
   decisions.

So the object of the rewrite is **not the ladder's ranking**. It is the
**restore/expansion bifurcation one level above the ladder** — the gate that
decides whether the ladder runs at all. That is a smaller change than roadmap
§22 anticipated, and it is in a different file region.

A second, *contingent* defect sits inside the ladder and is described in §3.4.
It is real, it is cheap to check offline, and whether it must be fixed is the
one open empirical question this document does not settle.

---

## 1. Q1 — The P5 commit ladder, enumerated tier by tier

### 1.1 Where the chain lives

The chain is a single `if`/`elif` cascade spanning **`:23887` (the `if`) through
`:24195`** (the end of the terminal `else`). Thirteen tiers. Its candidate
tuple is unpacked at `:24236–24244`:

```
(score, plan, state, target, _chosen_novelty, _error, _visual_change,
 target_frontier_signature) = chosen
```

so in every sort key below, `item[0]` is the planner score, `item[1]` the plan
(`.path`, `.durations`), `item[2]` the save-state (the identity key into
`branch_goal_analyses` / `branch_goal_signatures`), `item[3]` the endpoint frame.
The commit that terminates the path is at `:25868`
(`branches_examined=len(verified)`, `restored_archive=False` at `:25869`).

### 1.2 The thirteen tiers

| # | Clause | Guard | Ranked on (construction site) | Telemetry label / `reason` |
| --- | --- | --- | --- | --- |
| 1 | `:23887` | `if anticipated_observation_choice is not None` | closest NOOP duration to the scheduled observation duration, then score (`:23860–23882`) | `anticipated_transition_observation`; no `reason` |
| 2 | `:23900` | `elif delayed_transition_choice is not None` | earliest probe `resolution_step`, then score (`:23427–23436`) | `delayed_transition_branch_selected`; no `reason` |
| 3 | `:23922` | `elif human_prior_goal_choice is not None` | `milestone_reward`, then score, over `positive_goal_branches` (`:23489–23499`; set at `:23322–23331`) | `human_prior_goal_choice`; no `reason` |
| 4 | `:23932` | `elif relational_navigation_choice is not None` — **seam S1** | `(-distance_to_certified_target_cells, score, action/duration tuple)` (`_relational_navigation_commit_view:20187–20288`, key at `:20250–20266`) | none inline; emitted by S4 at `:24203–24235` |
| 5 | `:23939` | `elif human_prior_navigation_detour_choice is not None` | `(ordering_reward, unvisited target position, -graph visits, ordering_reward, score)` (`_human_prior_navigation_detour_progress_choice:2257–2332`) | `human_prior_navigation_detour_progress_selected`; `reason="continue_bounded_detour_before_optional_probe"` (`:23980`) |
| 6 | `:23983` | `elif human_prior_semantic_frontier_choice is not None` | `(target position never visited, -graph-state visits, count of unexpanded control actions, score, action/duration tuple)` (`:23529–23568`; set at `:23342–23369`) | `human_prior_semantic_frontier_choice` / `..._known_milestone_frontier_choice`; `reason="player_endpoint_needs_expansion"` (`:23999–24003`) |
| 7 | `:24025` | `elif known_goal_fallback_choice is not None` | `milestone_reward`, then score, over `known_goal_branches` (`:23569–23580`) — **gated at `:23577–23578` on `human_prior_semantic_frontier_choice is None`** | `human_prior_known_milestone_fallback`; `reason="no_unvisited_semantic_frontier"` (`:24035`) |
| 8 | `:24038` | `elif dynamic_control_choice is not None` | `(score, future control spread, action-effect contrast)` (`:23711–23724`) | `dynamic_control_selected`; `reason="counterfactual_future_control_collapse"` |
| 9 | `:24057` | `elif control_intervention_choice is not None` | `(action-effect contrast, score, action/duration tuple)` (`:23825–23845`) | `autonomous_intervention_selected` / `causal_observation_intervention_selected` |
| 10 | `:24075` | `elif causal_observation_wait is not None` | closest NOOP duration to the pending option's initiation duration, then score (`:23480–23488`) | `causal_observation_wait`; no `reason` |
| 11 | `:24096–24099` | `elif autonomous is not None and self.autonomous_grace_remaining <= 0` | largest qualifying duration's NOOP branch (`_autonomous_choice:20953–20981`) | `autonomous_dynamics_detected`; no `reason` |
| 12 | `:24112` | `elif self.autonomous_grace_remaining > 0` | three sub-branches: score+tuple (`:24124–24134`), longest NOOP duration (`:24152–24155`), score+tuple (`:24173–24182`) | `autonomous_grace_ended` / `autonomous_grace_wait` / none |
| 13 | `:24183` | bare `else` | raw planner score, then the action/duration tuple as a deterministic tie-break | **none** — the silent default commit |

### 1.3 What references a goal coordinate, definitively

Grepping `target_cell|certified|milestone_reward|target_player_slot|_position_visits|distance`
over `:23316–24196` returns only: `milestone_reward` (a scalar reward, not a
coordinate) at `:23320/23346/23493/23573`; `target_player_slot` used solely as a
**key into a visit counter** (`_human_prior_position_visits:4165–4172`) at
`:23348–23360` and `:23533–23537`; telemetry payloads at `:24009–24015`; and one
comment at `:23503`.

- **Tier 4 (S1) — yes, directly and only.** `targets` come from
  `relational_objective_target_cells` (`relational_planner.py:929–948`), which
  the module documents as sourced exclusively from certified records; the branch
  cell is `_relational_navigation_cell(analysis.target_player_slot)`
  (`:20178–20185`); the key leads on `-row[2]` where `row[2]` is
  `relational_target_cell_distance` (Manhattan, `relational_planner.py:1046–1061`).
- **Tier 5 — indirectly and conditionally.** `human_prior_navigation_ordering_reward`
  is a *progress delta* (`navigation_reward * (source_distance - target_distance)`,
  `goal_prior.py:655–674`; retarget path `:1951–1963`), against visible milestone
  slots, not certified cells — and `human_prior_navigation_reward` defaults to
  **0.0** (`:178`) by the §4.7 decision and is not set on any E-series command line.
- **Tiers 1–3, 6–13 — no.** They rank on NOOP-duration proximity, probe
  resolution step, a `milestone_reward` scalar, action-effect contrast, control
  spread, outcome spread, planner score, and a deterministic action/duration
  tie-break. **`human_prior_semantic_frontier_choice`, the tier that does the
  wandering, ranks on position novelty alone** — confirming the predecessor
  doc's §3.2 claim at the new anchors.

### 1.4 Where a "final step onto an adjacent certified milestone" tier would sit

The task's constraint — above novelty, below milestone collection and safety —
places it **between `:23922` (Tier 3) and `:23939` (Tier 5)**, i.e. exactly
where seam S1 already sits at `:23932`. The comment at `:23500–23509` states
that placement rationale verbatim: below `human_prior_goal_choice` "so an actual
milestone collection is never overridden, and above pure position novelty."

**And this is the design's first negative result: that slot is occupied, and
occupying it is not the problem.** §2 shows the slot is never reached at the
instant that matters.

---

## 2. Q2 — What the step actually requires. **This is the answer that determines scope.**

### 2.1 The primitive exists and is demonstrated, twice, at the same geometry

The actuator is a single directional action at 16 frames. Measured in v341:

- **d16** verified `path=['down','a','a']`, `durations=[16,1,2]` from `(12,9)` →
  `(12,10)`, state `state-00012381` (`seq 82630`), and committed it (`seq 82663`).
  One 16-frame `down` moves exactly one cell.
- **d1** verified `path=['up','a','a']`, `durations=[16,1,2]` from `(6,9)` →
  `(6,8)`, **collecting the heart at `(96,128)`** — `human_prior_milestone_reward:
  25.0`. **A single 16-frame step onto a heart cell collects the heart.** The
  geometry of `(12,10)` → `(12,11)` is identical, and the certified record
  `85fd9014d58deb42` lists that edge inside its 4-connected envelope
  (`(12,8)→(12,9)→(12,10)→(12,11)`, predecessor doc §3.2).

So the actuator is not missing in the sense of "no action reaches the target".
The fan that generates it (`a`, `b`, `noop`, `down`, `left`, `right`, `up` — 7
branches) is generated at **every** expansion decision, verified at d15, d16 and
d19 with exactly 7 `branch_verified` events each.

### 2.2 The branch was never generated, because the ladder was never entered

At d17 and d18, the two decisions that *begin* at `(12,10)`:

| | d17 | d18 |
| --- | --- | --- |
| `human_prior_graph_stagnation_detected` | seq 83067 | seq 83102 |
| `human_prior_option_search_deferred` | seq 83068, `reason: global_semantic_archive_frontier_available` | seq 83103, same reason |
| `relational_navigation_restore_selected` | seq 83081, `differs: true`, 7 hold-matching candidates, baseline `(10,7)` distance 6, **selected `state-00012381` distance 1** | seq 83109, `differs: true`, 6 candidates, selected `state-00012374` distance 4 |
| `archive_branch_restored` | seq 83087 | seq 83113 |
| `decision_committed` | seq 83088, `branches_examined: 0`, `restored_archive: true` | seq 83114, `branches_examined: 0`, `restored_archive: true` |
| `branch_verified` events | **0** | **0** |

Both decisions returned from `decide()` at `:21702–21704`:

```
21702        restored = self._restore_if_stagnant()
21703        if restored is not None:
21704            return restored
```

`self.planner.plan(self.frame)` at `:21769` is 67 lines further down. It never
ran. **The candidate set was empty, not mis-ranked.**

### 2.3 The direct answer to "did the incumbent already have a verified branch reaching the target?"

**No — and stronger than no.** A scan of **all 12,337** branch-verification
events in v341 (12,232 `human_prior_option_branch_verified` from the decision-0
resume audit plus 105 commit-path `branch_verified`) finds **zero** branches
whose `human_prior_target_hearts` drops `[192,176]`. Not one branch in the run,
at any depth, ever collected the certified milestone. The same scan on v340
(control) and v342 (attribution) gives 105 commit-path branches each and the
same zero.

So the question "was it ranked below something else?" has no referent at d17/d18:
there was nothing to rank.

### 2.4 The invariant that makes this a structural finding, not an incident

Census across all ten 24-decision runs (v333, v334, v335, v336, v337, v338,
v339, v340, v341, v342), classifying each committed decision by whether it began
at Chebyshev distance ≤ 1 from `(12,11)` and whether it expanded branches:

| Quantity | Count |
| --- | --- |
| Expansion decisions (`branches_examined: 7`) | **149** |
| …of which begin at distance ≤ 1 | **0** |
| Restore decisions beginning at distance ≤ 1 | **11** |
| Minimum source distance over all expansion decisions | **2** (nine runs) / 4 (v335) |

The pattern is invariant across every arm and every seam configuration: **the
decision immediately following arrival at distance 1 is always a stagnation
restore.** Controls arrive at d17 and restore at d18; E5/E6 treatments arrive at
d16 and restore at d17; E7 treatments arrive at d16, self-restore at d17, and
restore away at d18.

### 2.5 The sharpest sub-finding: E7's d17 win was spent standing still

At d17 the restore selected `state-00012381` — **the state the agent was already
in**. d16's committed state id and d17's committed state id are the same
(`state-00012381`), and d17's `parent_state_id` (`state-00012376`) is d16's
parent. Positionally, d17 was a no-op that consumed a decision.

Then `self.archive.remove(branch)` at `:28019` consumed the `(12,10)` archive:
`hold_matching_candidates` went **7 → 6** between d17 and d18, and the target-aware
key at d18 could only reach `(12,7)`.

Stated exactly: **the closing mechanism won its contest and spent the win on
standing still, then lost the candidate it had just used.** The restore path has
no expression for "hold this position *and* act from it" — `branch =
max(restore_eligible, key=restore_key)` at `:27990` always restores something.

### 2.6 The contingent second defect, inside the ladder

If the restore were declined and the ladder entered, which tier would take a
`(12,11)`-collecting branch?

- If `_human_prior_milestone_outcome_known` (`:17549–17571`) is **False** for
  that transition → the branch lands in `positive_goal_branches` (`:23322–23331`)
  → **Tier 3** at `:23922`, which outranks the novelty tier. The step is taken.
- If it is **True** → the branch lands in `known_goal_branches` (`:23332–23341`)
  → **Tier 7** at `:24025`, which is interlocked at **`:23577–23578`** to be
  computed only when `human_prior_semantic_frontier_choice is None`. At d16 that
  choice was **not** None (Tier 6 fired). **Tier 7 would have been preempted by
  position novelty.**

Which branch applies is **not determinable from E7 telemetry**, because no
branch ever collected `(192,176)`. What *is* measured, and is a warning:

- Both in-run collections used the **known** path. d1 and d3 both committed via
  `human_prior_known_milestone_fallback` with `reason:
  no_unvisited_semantic_frontier`, and both branches report
  `human_prior_milestone_outcome_known: True`. They fired only because the
  frontier choice happened to be `None` at those two decisions.
- The run seeds **12 milestone outcomes** and 5 exhausted milestone transitions
  from v318 (`episodic_human_prior_memory_seeded`, seq 14), so "known" is the
  live regime at this root, not a corner case.

**This is the real "the ranking buried it" risk, and it is one tier lower than
the trigger anticipated — Tier 7, not Tier 6.** It must be settled offline
before E8 runs (§4.1 P2).

### 2.7 Verdict on Q2

| Candidate explanation | Verdict |
| --- | --- |
| "The actuator is missing" | **False.** The primitive is generated by the standard fan and demonstrated collecting a heart at identical geometry (d1). |
| "The actuator exists but the ranking buried it" | **False at d17/d18.** There was no candidate set to rank; 0 branches verified, `branches_examined: 0`. |
| "The ladder is never entered at the decisive instant" | **True, and measured with zero ambiguity** — 0 of 149 expansion decisions across ten runs begin adjacent to the target; all 11 adjacent decision-starts are restores. |
| "…and if it were entered, the collecting branch may still be buried at Tier 7" | **Open, contingent on a memory fact, cheaply checkable offline.** |

**Therefore: this is a gating fix outside the ladder, plus a conditional
interlock fix inside it. It is not a ladder rewrite, and it is not a pure
ranking fix.**

---

## 3. Q3 — Scope, as narrowly as the evidence licenses

### 3.1 The minimal version: two changes, both authority-gated, neither a new tier

**R-A — terminal-step preemption at the restore bifurcation.** Before `decide()`
accepts a stagnation restore, if the agent's **current** position satisfies the
S3 deposit gate's own predicate *at Manhattan distance exactly 1*, decline the
restore for this decision and fall through to expansion. Concretely: a guard at
the `:21702` call site (and at the `:21329` proactive call site, which shares
`_restore_if_stagnant`) that suppresses the restore and emits a named event.

Design properties that make this the minimal correct change:

- It **adds no tier, no term, and no weight.** The ladder is byte-identical.
- It **reuses the predicate that already works** — `_relational_navigation_deposit_view`
  (`:20470–20544`) with `RELATIONAL_NAVIGATION_DEPOSIT_MAX_DISTANCE`, evaluated
  against the *current* position rather than the committed one, with all four
  existing named refusal reasons (`held_configuration_absent`,
  `position_unavailable`, `not_certified_adjacent`, `position_not_certified`)
  intact. E7 bit 2 proved this predicate fires correctly at exactly the right
  instant.
- It is **one-sided and terminal**: it can only *decline* a restore, never choose
  one, never rank one, never propose a destination.
- It **preserves the deposit.** S3 still deposits `(12,10)` at d16, so if the
  step fails the position remains restorable — the E7 machinery is not removed,
  it is unblocked.

**R-B — certified-milestone collection is not interlocked below position
novelty.** Conditional on the §4.1 P2 precondition check. If a `(12,11)`-collecting
branch is `outcome_known`, remove the `:23577–23578` interlock **only for
branches collecting an uncollected certified milestone cell** — i.e. add the
certified-cell conjunct rather than dropping the interlock wholesale. If the
precondition check shows `outcome_known` is False for that transition, **R-B is
not built at all** and E8 runs with R-A alone.

R-B's blast radius is measurable now and is very small: across all three E7 arms
there are **105 commit-path verified branches and exactly 2 with
`milestone_reward > 0`** (d1 `(96,128)`, d3 `(128,128)`), and both already fired
Tier 7 with the frontier choice `None`. **At the other 13 expansion decisions no
milestone-collecting branch existed at all**, so an interlock change cannot
reorder them. This is an offline-provable invariance argument and should be
asserted as a test, not assumed.

### 3.2 What is deliberately NOT built

- **No new ladder tier.** Opportunity count zero (§2.4); building it would be a
  sixth seam that cannot fire.
- **No re-enabling of S1.** Refuted in E3 for supply starvation (§4.50); its key
  is a monotone distance gradient over the whole eligible set at every expansion
  decision, which is the exact shape §4.50 killed.
- **No radius widening.** Forbidden by the E6 design §15.4 and independently
  useless: `(12,7)`–`(12,9)` were already candidates and the d18 restore already
  took `(12,7)`.
- **No budget re-arm, no weight, no reward term, no `--decisions` increase.**
  All four are the tunings §4.53 prohibited.
- **No change to P1 (life-loss recovery, `:21295`) or P4 (goal-exhaustion
  rollback, `:21732`).** Safety recovery stays hypothesis-blind.

### 3.3 What would justify a larger restructure

State the escalation rule now, so it is not a fatigue decision later:

1. **R-A fires and the ladder still does not take the step, for a reason other
   than the Tier-7 interlock.** Then the ladder's *ranking* is genuinely the
   object and roadmap §22's framing is vindicated — promote the ladder to an
   explicit policy object with a declared objective input, per lifecycle §7.3's
   definition of "rewrite".
2. **R-A cannot be made one-sided.** If suppressing the restore at distance 1
   requires knowing anything about distance > 1 — i.e. if the guard needs a
   gradient to be implementable — then it is a proximity preference in disguise
   and §4.50 forbids it. Abandon and escalate.
3. **R-A narrows archive geography.** If the treatment's archive columns narrow
   relative to the control (the direct instrument that failed E3 and passed
   E5/E6/E7), that is §4.50 recurring at the schedule level. **Abandon, do not
   tune.**
4. **A repeated failure reason (lifecycle §7.3 trigger C).** If R-A fails for
   redundancy, no-opportunity, supply starvation, candidate-absent, or
   objective-absent — all five already on the list — the bolt-on-plus-gate model
   is exhausted and WP8 moves to representation depth (WP2/WP3 integration).

---

## 4. Q4 — Guards against the recorded failure modes

### 4.1 §4.50 — does a terminal-step gate smuggle in a consistent proximity preference?

This is the question that killed E3, and it deserves a mechanical answer rather
than an assurance.

**§4.50's mechanism, restated precisely.** E3's S1 key was
`(-distance, score, tuple)` applied over the **whole eligible branch set at
every expansion decision**. That is a *monotone function of distance defined
everywhere*: at distance 9 it prefers 8, at 8 it prefers 7, and so on. The
measured consequence was that the d6–d8 excursions *away* from the target — the
ones that deposited the archives later progress climbed — were removed, and
archive geography collapsed from columns 6–12 to 6–8. §4.50's own words: the
tie-break-not-reward care "does not save it. Any consistent preference for
target proximity, however weakly expressed, removes the excursions."

**Why R-A is not that.** R-A is a **point predicate, not a function.** It is
defined only where Manhattan distance is exactly 1 and is identically absent
everywhere else. Four properties follow, each independently checkable:

1. **No gradient exists to follow.** There is no distance-2 behaviour, no
   distance-3 behaviour, no ordering between any two branches, ever. R-A never
   compares candidates; it declines one action (the restore) and returns control
   to an unchanged ladder.
2. **It cannot cause approach, only completion.** At Manhattan distance 1 the
   target is one primitive action away. The agent must already have arrived
   under its own novelty-driven exploration before R-A can exist. R-A is
   incapable of influencing how the agent got there, because it does not exist
   at any distance from which getting there is still in progress. The excursions
   §4.50 showed to be load-bearing run in a regime where R-A is structurally
   absent.
3. **Its effect is bounded to one decision.** R-A declines one restore. It does
   not re-arm, does not persist, does not accumulate, and does not alter the
   restore key. On the next decision the incumbent machinery is untouched.
4. **The failure mode is directly instrumented and pre-declared as fatal.**
   Archive geography (columns 6–12) is the exact instrument that failed E3 and
   passed E5/E6/E7. Bit 1(c) below re-uses it verbatim. **If R-A narrows the
   treatment's archive columns relative to the control, that is §4.50 at the
   schedule level and R-A is abandoned, not tuned.**

The honest residual: R-A does change *when* exploration happens, by giving one
decision back to expansion that would have been spent restoring. That is a
schedule effect, not a preference, but it is not nothing — which is precisely
why bit 1's prefix-equality and geography checks are scored before any outcome
bit is read.

**And the symmetric warning about R-B.** R-B *is* a ranking change, and ranking
changes are what §4.50 taught suspicion of. Its defence is not that it is
harmless in principle but that its blast radius is **empirically zero at this
root** (§3.1: 2 milestone branches in 105, both already committing via Tier 7).
That is a root-specific defence and must be stated as one. If R-B is built, the
offline invariance test must show byte-identical commits at every decision where
no certified-milestone branch exists.

### 4.2 §4.7 — no straight-line goal distance

R-A introduces no distance term, no reward, no bonus, no weight, and no
tie-break. `human_prior_navigation_reward` remains **0.0** (`:178`). The only
numeric comparison is `distance > RELATIONAL_NAVIGATION_DEPOSIT_MAX_DISTANCE`
inside the existing, already-shipped `_relational_navigation_deposit_view`
predicate — a boolean threshold at 1, not an ordering.

### 4.3 §4.43 — the redundancy detector, mandatory

Three of five prior levers failed or nearly failed because the incumbent would
have made the same choice. E8 must be able to say so *from its own telemetry*,
not from a post-hoc argument. Required instrumentation, emitted at every instant
where R-A's predicate is evaluated:

- `relational_terminal_step_gate` — `eligible` plus one of the four named
  reasons, on **every** evaluation, so declines are as visible as fires (the S3
  pattern that made E6's twelve declines readable).
- `incumbent_restore_state_id` and `incumbent_restore_distance` — **what
  `_restore_if_stagnant` would have returned had R-A not fired.** Computed
  before suppression, logged, discarded. This is the direct redundancy test: if
  the incumbent's restore was already `state-00012381` (a self-restore, as at
  d17), R-A's contribution is *not* holding the position — the incumbent was
  already holding it — but freeing the decision to act. That distinction must be
  readable in the log, not inferred.
- `ladder_tier_committed` and `ladder_tier_would_have_committed_without_R_B` —
  the tier label at every expansion decision, and, if R-B is built, the
  counterfactual tier. If they are equal at the decisive instant, R-B was
  redundant and must be reported as such regardless of the outcome bit.
- **Post-R-A branch census**: the number of `branch_verified` events at the
  decisive decision and whether any of them collects `(192,176)`. If R-A fires
  and the collecting branch still does not appear, the diagnosis is a
  reachability failure, not a ranking failure, and E8 says so.

### 4.4 §4.53 — the tuning prohibition, applied

Each element of this design must be checkable against "is this a tuning?":
R-A changes a control-flow gate (not a tuning); R-B changes a filter conjunct
(not a tuning, but conditional on a measured precondition); radius stays 1;
budget stays 12; decisions stay 24; weight stays 0.0. **Nothing in this design
adjusts a number that a previous experiment set.**

---

## 5. Q5 — E8 preregistration skeleton

*Written before any arm runs. Nothing here may be revised after a run starts.*

### 5.1 Preconditions, all discharged offline before any emulator time

| # | Precondition | How checked | If it fails |
| --- | --- | --- | --- |
| P1 | R-A's predicate would have fired at v341 d17 | Offline replay of `_relational_navigation_deposit_view` against d17's current position, hold signature `85fd9014d58deb42`, targets `[[12,11]]` | E8 is cancelled; the gate is mis-specified |
| P2 | **Is a `(12,11)`-collecting branch `outcome_known`?** | Reconstruct the seeded memory from `entity-v318-…-d2` decision 1 (the same seed E7 used, `milestone_outcomes: 12`, `exhausted_milestone_goal_slots: 1`) and evaluate `_human_prior_milestone_outcome_known` / `_human_prior_milestone_transition_exhausted` on a synthetic analysis for the `(192,176)` transition | **False** ⇒ R-B is not built, E8 runs R-A alone. **True** ⇒ R-B is required and is in scope. Either way the answer is recorded before the run. |
| P3 | R-B's blast radius is zero off the decisive instant | Offline: assert byte-identical commits at all decisions with no `milestone_reward > 0` branch (measured: 13 of 15 in v341) | R-B is abandoned; escalate per §3.3(1) |
| P4 | Cross-version byte-identity at authority `off`/`telemetry` | The E7 §10.7 row-6 harness, extended with the new gate value across all authorities | Fix before running |
| P5 | Strict-lineage linter clean; no selector token in the pure module | `python -m lolo_agent.strict_lineage lolo_agent/relational_planner.py` | Fix before running |

P2 is this design's E3-pre: a cheap check that can change the experiment, run
first, exactly as §4.46/§4.47 established.

### 5.2 Arms, matched budgets

Three arms, run **sequentially, one native run at a time**, into separate log
roots, from the same v318 pre-push root (`source_decision: 1`,
`state_source_checkpoint_event_seq: 2026`,
`state_source_events_sha256: 0bbe1d15…`):

| Arm | Authority | Gate | Scored? |
| --- | --- | --- | --- |
| Control | `off` | R-A off | Yes |
| Treatment | `selection` | R-A on (R-B on iff P2 true) | Yes |
| Attribution (unscored) | `selection` | R-A on, target cells read from the record store rather than a published chain | No — reported, never cited |

Everything else byte-identical: 24 decisions, `relational_decision_budget: 12`,
`relational_navigation_seams: restore_plus_deposit`,
`relational_lifecycle: chain_published`, `verified_accessibility_weight: 0.0`,
records `15604cb5…`/`37ea410d…`/`47975c94…`. The only permitted config
differences are `relational_planner_authority` and `relational_planner_enabled`
plus the new gate flag.

The attribution arm is retained because §4.54 recorded that E7's store-read arm
was trajectory-identical to the treatment. **E8 must not cite hypothesis-scoped
attribution unless this arm diverges**, and the report must say so on its face.

### 5.3 The preregistered bits (fixed; **ANY mixed outcome = FAIL**)

**Bit 1 — SUPPLY AND PREFIX (the §4.50 guard).**
(a) The treatment reproduces v341 state-for-state for d1–d16 (the intervention
instant is d17, so the entire approach must be untouched).
(b) The control reproduces v340/v338/v336/v334/v333 exactly, extending the
invariance chain in vivo.
(c) **Archive geography not narrowed**: treatment column range ⊇ control's
(expected 6–12 both), deposit count within one of the control's 44.
A failure of 1(c) is §4.50 recurring; **abandon, do not tune.**

**Bit 2 — THE GATE FIRES AT THE DECISIVE INSTANT.** At d17 a
`relational_terminal_step_gate` event with `eligible: true`, cell `(12,10)`,
distance 1, hold signature `85fd9014d58deb42`, and the stagnation restore
suppressed. **Plus the redundancy field**: `incumbent_restore_state_id` recorded.

**Bit 3 — THE CANDIDATE IS GENERATED.** d17 reports `branches_examined: 7` and
**≥ 1 `branch_verified` event whose `human_prior_target_hearts` drops
`[192,176]`** — i.e. the collecting branch provably exists in the verified set.
This bit is the one that has never been reached by any prior experiment and it
is scored independently of the outcome, so a failure here is diagnosable as
reachability rather than ranking.

**Bit 4 — OUTCOME.** `(12,11)` / `(192,176)` collected by the treatment and not
by the control, within 24 decisions.

**Bit 5 — SAFETY.** Zero life losses in both scored arms; no
`position_not_certified` deposit; P1 (life-loss recovery) and P4
(goal-exhaustion rollback) unmodified and un-suppressed.

**Reading fixed in advance**: bits 1–3 PASS with bit 4 FAIL means the collecting
branch existed and the ladder still discarded it — **that** is the ranking
failure roadmap §22 hypothesised, and it is then, and only then, grounds for the
ladder-as-policy-object rewrite (§3.3(1)).

### 5.4 Honest power analysis — how many opportunities does this actually get?

- **There is no sampling variance.** Runs are deterministic. "Power" means
  opportunity count.
- **Bit 2's guaranteed opportunity count is exactly ONE**: v341 d17, the single
  decision that begins at `(12,10)` under the hold signature with the deposit
  already made. v341 d18 also begins at distance 1, but R-A firing at d17 changes
  the trajectory from d17 onward, so d18 is **not** a guaranteed second
  opportunity. Report it if it occurs; do not count on it.
- **The whole ten-run corpus supplies 11 distance-≤1 decision-starts and zero
  expansion decisions at distance ≤ 1.** E8 does not increase that supply and
  must not try to: raising `--decisions`, widening the radius, or re-enabling S1
  are all excluded by §3.2.
- **Bit 3 is the first genuinely new measurement in the series.** Every prior
  experiment's central bit was about ranking or candidacy among branches that
  existed; bit 3 asks whether the branch is generated at all. Even a bit-4 FAIL
  with bit 3 PASS is a strictly more informative result than E1/E3/E5/E6/E7.
- **The dominant FAIL mode is prefix disturbance.** If bit 1(a) fails — the
  treatment's d1–d16 diverges from v341 — the reading is pre-declared: "the gate
  widened the intervention window and moved the approach", a FAIL with a named
  mechanism, **not** a VOID and **not** grounds for re-scoping and re-running.
- **What a PASS cannot show**: that the planner can cause a search (E4 remains
  untested); that hypothesis-driven planning rather than a standing rule produced
  the behaviour (the attribution arm decides that, and E7's precedent is that it
  may not); or that a second manipulation is possible.

### 5.5 VOID conditions (a VOID is not evidence)

Inherited from lifecycle §10.5, with V1 naming the new field:

1. **Config inequality** — the scored pair's `planning_config` differ in any
   field except `relational_planner_authority`, `relational_planner_enabled`,
   and the new terminal-step gate flag. Both must report
   `relational_lifecycle: chain_published`,
   `relational_navigation_seams: restore_plus_deposit`,
   `relational_decision_budget: 12`.
2. **Records inequality** — both arms `record_count: 3`, signatures
   `15604cb5…`/`37ea410d…`/`47975c94…`, `verified_accessibility_weight: 0.0`.
3. **Seeding defect** — no archived branch carrying `85fd9014d58deb42` in the
   window in either arm (E7 measured 23; a collapse invalidates the root).
4. **Root defect** — either manifest's `episodic_resume` block does not record
   `entity-v318-room3-known-push-connected-mask-d2`, `source_decision: 1`,
   `state_source_checkpoint_event_seq: 2026`,
   `state_source_events_sha256: 0bbe1d15…`.
5. **Budget defect** — either arm exceeds the 10,800 s wall ceiling and is killed
   before `run_finished`; or verified-branch counts differ by more than 1%
   *outside* the intervention decision (R-A is expected to add exactly one
   decision's worth of branches, and that delta is the mechanism, not a defect —
   the tolerance is computed on d1–d16 only).
6. **Control-invariance defect** — the control's 24 committed state ids do not
   reproduce v340's exactly. **A crashed arm is VOID, not FAIL.**
7. **Standing-rule contamination** — the attribution arm changes nothing about
   the scored pair; the scorer must show no arm-3 artefact is an input to either
   scored arm.

Budget-exhausted non-reach is **censored**, never "unreachable".

### 5.6 Health-check rule (learnings §4.52), fixed before the run

- The process is `python -m lolo_agent.neural_run` — **underscores, module
  form**. A hyphenated `pgrep` pattern can never match and its silence is not
  evidence of death.
- **Health is judged from the run's own telemetry only**: monotone growth of
  `events.jsonl` and arrival at expected seq milestones. The first
  `decision_committed` lands at seq ≈ **75,742** in every arm of this family;
  **zero committed decisions at 6k events is on-profile, not a death.**
- Two signals that share an author are not corroboration.
- **A crashed arm is VOID, not FAIL**, so no diagnosis under time pressure can
  convert an operational failure into evidence.
- Per §4.46, **report option-search counts in the run summary**. Expected at this
  root: `completed: 1, deferred: 9` in the control. R-A does not grant searches;
  a change here is a finding, not a nuisance.

### 5.7 The discriminator, and when it must change

`(12,11)` / `(192,176)` is unbroken across nineteen runs at this root and was
re-validated by §4.48 as a *capability* discriminator rather than a speed one
(the control oscillates and then diverges: `3,3,4,3,4,4,3,2,1,4,5,5,5,4,5,6`).
The E8 scorer must re-verify the count from telemetry rather than inherit it.

**Pre-declared conditions under which the discriminator retires and Gate 4 work
moves to the `(8,4)` / `(9,12)` second-manipulation targets** (both confirmed
present in v341's `relational_hypothesis_proposed`: `remaining_milestone_cells:
[[6,8],[8,4],[8,8],[9,12],[12,11]]`):

1. **E8 bit 4 PASSES.** `(12,11)` becomes a solved instance and can no longer
   discriminate. All subsequent Gate 4 claims move to `(8,4)`/`(9,12)`, which lie
   **outside** the certified envelope and require a **second manipulation** — a
   capability incidental behaviour has never demonstrated, versus the
   single-manipulation-plus-closing claim `(12,11)` supports.
2. **Any control arm collects `(12,11)` in-window.** The §4.47 rule applies:
   discriminator validity is a trend property, and a control collection kills it
   immediately.
3. **Bits 1–3 PASS with bit 4 FAIL twice.** If the branch is generated and
   discarded across two independent interventions, `(12,11)` has become a probe
   of the ladder's ranking rather than of deliberate preparation, and the
   capability question should be re-asked at a target that requires a second
   manipulation.

Retiring the discriminator is a **scope change, not a rescue**: it may not be
invoked to convert an E8 FAIL into a partial pass, and any move to
`(8,4)`/`(9,12)` requires its own preregistration with its own control-never-does-it
evidence at the new target.

### 5.8 Scoring

A single deterministic scorer walks each arm's `events.jsonl` once, applies §5.3
verbatim, and writes `experiments/lolo1-wp5/e8-gate4-report.json` with a
canonical-JSON `digest_sha256` over the body. Run end-to-end **twice**; both
reports byte-identical. **Validated against v340/v341 first**: it must reproduce
E7's verdicts exactly — the single d16 deposit at `(12,10)`, the 8 restore
instants with 3 differing, `hold_matching_series [1,3,3,6,7,6,5,4]`, columns
6–12 with 44/43 deposits, `option_searches {completed: 1, deferred: 9}`, and
"never collected `(192,176)`" — before it is trusted on E8.

Distance metrics as in E7: **Chebyshev** for §4.47/§4.48-comparable traces,
**Manhattan** for the mechanism's own gate distances (`relational_target_cell_distance`).

---

## 6. Risks

**6.1 The §14 monolith risk, still worsening.** `neural_planner.py` is 29,142
lines and this design touches `decide()` at `:21702` and, conditionally,
`:23577`. Both are inside the same 4,500-line method. The mitigations are the
existing ones: AST anchor-drift tests that pin the gate's position relative to
named neighbours rather than to line numbers, and the cross-version byte-identity
harness across all authorities. Line anchors in this document are valid at
`fefbca7` **only** and must be re-verified, not copied — the predecessor doc's
anchors decayed by ~700 lines in one working day.

**6.2 R-A may reveal a third layer.** If the gate fires, the branch is generated,
and the ladder still discards it, the failure moves inside the ladder for real.
That is a *good* outcome epistemically — it is the first time the ranking
hypothesis would be directly testable — but it means WP8 is not finished by E8.
§3.3(1) is the declared response.

**6.3 The attribution problem is unresolved and may recur.** §4.54 recorded that
E7's store-read arm was trajectory-identical, so hypothesis scoping bought
nothing observable at this root. R-A reads certified cells through the same
objective path. **E8 inherits the same limitation and must state it on the face
of the report**: unless the attribution arm diverges, no Gate 4 claim may cite
E8 as evidence that hypothesis-driven planning produced the behaviour.

**6.4 Provenance.** Everything here is assisted-track: the certified records
derive from the player-anchored hold instrument (`certified_hold` provenance,
`accessibility_preference.py:135–159`). No strict claim is made or implied. The
WP5 shadow campaign (§4.42) remains the strict path.

---

## 7. What this document does not claim

- **No code exists.** R-A and R-B are specifications, not patches. No gate, no
  flag, no test, no selector has been written.
- **No bit is scored.** E8 has not run. Gate 4's consequence criteria remain open.
- **P2 is unresolved.** Whether a `(12,11)`-collecting branch is `outcome_known`
  is not determinable from E7 telemetry (zero such branches exist in 12,337
  verification events) and must be settled by offline memory reconstruction
  before E8 is scoped.
- **The §2 census is descriptive of this root and this family.** 149 expansion
  decisions and 11 adjacent restores are the measured facts at the v318 pre-push
  root across v333–v342. They are not claimed to generalise to other roots,
  rooms, or games.
- **No claim that R-A closes Gate 4.** The most this design can establish is
  that the agent can take a final step it has demonstrably prepared for. Gate 4's
  broader criteria — and any claim about a second manipulation — remain untouched.

---

# 8. E8 PREREGISTRATION — written and committed to before any arm ran

**Date**: 2026-08-18
**Planner HEAD at implementation**: `86c085b` ("Correct the trigger-A cause: the
ladder was never entered"). `git diff --stat fefbca7 86c085b` touches
**documentation only**, so every line anchor in §1–§7 above was still valid when
the code was written; the §6.1 warning about anchor decay is nonetheless why the
tests below pin positions relative to named neighbours rather than to numbers.
**Status of this section when written**: preconditions discharged, code landed,
suite green, **no arm run, no bit scored.**

Nothing in §8.1–§8.6 may be revised after the first arm starts.

## 8.1 Preconditions — all five discharged offline, before any emulator time

| # | Precondition | Verdict | Evidence |
| --- | --- | --- | --- |
| P1 | R-A's predicate would have fired at v341 d17 | **PASS** | §8.2 |
| P2 | Is a `(12,11)`-collecting branch `outcome_known`? | **TRUE ⇒ R-B is required and in scope** | §8.3 |
| P3 | R-B's blast radius is zero off the decisive instant | **PASS** | §8.4 |
| P4 | Cross-version byte-identity at authority `off`/`telemetry` | **PASS** | §8.5 |
| P5 | Strict-lineage linter clean; no selector token in the pure module | **PASS** | §8.5 |

## 8.2 P1 — the predicate fires at the decisive instant

Replayed against v341's own telemetry rather than asserted:

| Input to `_relational_navigation_deposit_view(..., current_position=True)` | Value at d17 |
| --- | --- |
| Current position (d16's commit, seq 82663) | player `(192,160)` ⇒ cell `(12,10)` |
| Published targets | `[[12,11]]` |
| Manhattan distance | **exactly 1** |
| Hold signature | `85fd9014d58deb42`, live at d15, d16, **d17**, d18 and d19 |
| Hold clause | passes — d19's decline reads `not_certified_adjacent`, **not** `held_configuration_absent`, so the configuration signature still equalled the hold signature after d17/d18 |
| `pending_life_recovery` | `None` (the run records **zero** life losses) |
| Dark transition | none |
| Recovery reason at d17 | `human_prior_graph_stagnation` — R-A's scoped reason |
| ⇒ gate | **`eligible: true`, `restore_suppressed: true`** |

And the redundancy field P1 makes checkable in advance: the incumbent's
selection at d17 was `state-00012381`, cell `(12,10)`, distance 1 — **the state
the agent was already standing in.** `incumbent_restore_is_self_restore` will
read `true`. R-A's contribution at this instant is therefore *not* holding the
position; the incumbent was already holding it. It is freeing the decision to
act. §4.3 required that distinction be readable in the log rather than inferred,
and it will be.

## 8.3 P2 — ANSWER: **TRUE.** R-B is required and is built

The design named P2 "this design's E3-pre: a cheap check that can change the
experiment, run first". It ran first, and it changed the experiment.

**Method.** The real seeding path, not a re-implementation:
`VerifiedNeuralAgent.seed_human_prior_episodic_memory()` was run over
`iter_episodic_decision_events(entity-v318-…-d2, 1)` — the identical stream E7
resumed from — with v341's own `planning_config` reconstructed from its
manifest. **The reconstruction validates against the run's own seeding
telemetry on six independent counters**, all exact:

| Counter | Reconstructed | v341 seq 14 |
| --- | --- | --- |
| `milestone_outcomes` | 12 | 12 |
| `exhausted_milestone_transitions` | 5 | 5 |
| `exhausted_milestone_goal_slots` | 1 | 1 |
| `disproved_ordering_hypotheses` | 1 | 1 |
| `graph_states` | 374 | 374 |
| `episodic_milestone_transitions` | 65 | 65 |

**Result.** The synthetic `(192,176)`-collecting analysis at d17's actual heart
state — source `((128,64),(144,192),(192,176))`, target `((128,64),(144,192))`,
player `(192,176)`, chest flags `False`/`False` — has outcome key

```
(((128,64),(144,192),(192,176)), ((128,64),(144,192)), (192,176), False, False)
```

and **that key is already in the seeded set.** Three further seeded keys also end
with the player on `(192,176)`. Therefore:

- `_human_prior_milestone_outcome_known` ⇒ **True**
- `_human_prior_milestone_transition_exhausted` ⇒ False (with and without the
  world-context override)
- The collecting branch lands in `known_goal_branches`, **not**
  `positive_goal_branches` ⇒ **Tier 7, not Tier 3.**

§2.6's contingent second defect is therefore **real, not hypothetical**, and
§5.1 P2's rule — "**True** ⇒ R-B is required and is in scope" — binds. R-B is
built.

**One correction to §3.1's specification of R-B, recorded before the run.**
Relaxing the `:23577–23578` computation interlock *alone* would have been a
**provable no-op**, because the ladder is an `elif` cascade and Tier 6 (`:23983`)
is evaluated before Tier 7 (`:24025`): the frontier choice would still have won
regardless of whether `known_goal_fallback_choice` was computed. R-B as built
therefore adds the certified-cell conjunct in **both** places — the interlock,
and Tier 6's own guard, which stands aside only when a branch collecting an
uncollected certified milestone cell exists. This is the same conjunct in two
positions, not a wider change; shipping the one-sided version would have been
shipping a change known in advance to do nothing.

## 8.4 P3 — R-B's blast radius, measured

Across **all three** E7 arms, from telemetry:

| Quantity | v340 control | v341 treatment | v342 attribution |
| --- | --- | --- | --- |
| Commit-path `branch_verified` | 105 | 105 | 105 |
| Expansion decisions in window | 15 | 15 | 15 |
| Decisions with a `milestone_reward > 0` branch | **2** (d1, d3) | **2** (d1, d3) | **2** (d1, d3) |
| Tier committed at those two | 7, 7 | 7, 7 | 7, 7 |
| Tier committed at the other 13 | 6 (all) | 6 (all) | 6 (all) |

At the two milestone decisions Tier 7 **already** committed, with the frontier
choice `None` — R-B cannot reorder what the incumbent already chose. At the other
13 per arm the conjunct is empty by construction. **Blast radius off the decisive
instant: exactly zero.** §3.1 required this be asserted as a test rather than
assumed; it is (`RelationalCertifiedTierTests.test_blast_radius_is_zero_where_
no_milestone_branch_exists`).

This also confirms §2.6's warning empirically: **every** expansion decision in
the window except d1 and d3 commits at Tier 6, position novelty. That is the
regime R-B has to survive.

## 8.5 What was built, and what was deliberately not

**R-A** — `_relational_terminal_step_view()`, plus a guard inside
`_restore_if_stagnant` placed **after** `branch = max(restore_eligible, …)` and
**before** `self.archive.remove(branch)` / the state load. §3.1 specified guards
at the `:21702` and `:21329` call sites "which share `_restore_if_stagnant`";
implementing it inside the shared method covers both call sites by construction
and is the **only** position at which the incumbent's selection is known, which
§4.3 requires be logged. Three consequences, all intended:

- The `(12,10)` archive is **not** consumed. §2.5's loss (`hold_matching_
  candidates` 7→6, so d18 could only reach `(12,7)`) does not occur. The deposit
  E7 proved it can make survives for a later decision.
- `incumbent_restore_state_id` / `_cell` / `_distance` / `_is_self_restore` are
  measured, not argued.
- R-A is scoped to `recovery_reason == "human_prior_graph_stagnation"`. This is
  narrower than §3.1's literal text and deliberately so: it is the reason at the
  only measured opportunity, and it leaves dark-transition returns alongside P1
  and P4 untouched (§3.2).

**R-B** — `_relational_certified_milestone_branches()`, applied to Tier 7's
interlock and Tier 6's guard as §8.3 explains.

**Selector** — one new config field, `relational_terminal_step`, default `"off"`,
validated exactly like the authority: `off` | `decline_restore` |
`decline_restore_and_certified_tier`. CLI `--relational-terminal-step`. It is the
single field VOID rule 1 permits the scored pair to differ in. It is inert
outside selection authority and inert with seam S3 off.

**Not built, per §3.2**: no new ladder tier; no re-enabling of S1; no radius
widening (`RELATIONAL_NAVIGATION_DEPOSIT_MAX_DISTANCE` stays 1); no budget
re-arm, weight, reward term, or `--decisions` increase; no change to P1 or P4.
**Nothing in this change adjusts a number a previous experiment set** (§4.4).
`relational_planner.py` is **untouched**.

**Evidence that the defaults preserve today's behaviour byte-identically (P4)
and that the pure module stays clean (P5)**: 23 appended tests in
`tests/test_ensemble_planner.py`; full suite **1,150 OK, 4 skipped** (baseline
1,127 + 23). One pre-existing test changed: the §6.1 anchor-drift pin on Tier 6's
guard line, which R-B legitimately rewrote — the *ordering* invariant it exists
to protect is asserted unchanged. `python -m lolo_agent.strict_lineage
lolo_agent/relational_planner.py` reports **zero findings**.

## 8.6 Arms, matched budgets, run ids

Three arms, **sequential, one native run at a time**, watchdog **10,800 s** each,
detached so a stopped wrapper cannot orphan the emulator (§4.52 and the E7
restart incident). Same v318 pre-push root (`source_decision: 1`,
`state_source_checkpoint_event_seq: 2026`, `state_source_events_sha256:
0bbe1d15…`). Command line: `docs/wp8-search-scheduling-design-2026-08-17.md`
§11.3 **verbatim**, with `--relational-navigation-seams restore_plus_deposit` and
`--relational-lifecycle chain_published` as in E7, and one new flag.

| Arm | Run id | Authority | `--relational-terminal-step` | Scored? |
| --- | --- | --- | --- | --- |
| Control | `entity-v343-room3-e8-control-off-d24` | `off` | `off` | Yes — runs **first** |
| Treatment | `entity-v344-room3-e8-treatment-terminalstep-d24` | `selection` | `decline_restore_and_certified_tier` | Yes |
| Attribution (unscored) | `entity-v345-room3-e8-standingrule-store-d24` | `selection` + `--relational-lifecycle record_store` | `decline_restore_and_certified_tier` | **No bits.** Reported, never cited |

Everything else byte-identical: 24 decisions, `relational_decision_budget: 12`,
`verified_accessibility_weight: 0.0`, records `15604cb5…`/`37ea410d…`/`47975c94…`.

## 8.7 The preregistered bits (fixed; **ANY mixed outcome = FAIL**)

**Bit 1 — SUPPLY AND PREFIX (the §4.50 guard).** All three conjuncts.
(a) The treatment reproduces v341 state-for-state for **d1–d16**; the
intervention instant is d17, so the entire approach must be untouched.
(b) The control reproduces v340 exactly, extending the
**v333≡v334≡v336≡v338≡v340** invariance chain in vivo. *A control that does not
reproduce that chain is a **VOID**, not a FAIL* (VOID 6).
(c) **Archive geography not narrowed**: treatment column range ⊇ control's
(expected 6–12 both), deposit count within one of the control's 44.
A failure of 1(c) is §4.50 recurring; **abandon, do not tune** (§3.3(3)).

**Bit 2 — THE GATE FIRES AT THE DECISIVE INSTANT.** At d17 a
`relational_terminal_step_gate` event with `eligible: true`, `cell: [12,10]`,
`distance: 1`, `hold_configuration_signature: 85fd9014d58deb42`,
`restore_suppressed: true`, **and** the redundancy field
`incumbent_restore_state_id` recorded. Per §8.2 it is expected to read
`state-00012381` — a self-restore.

**Bit 3 — THE CANDIDATE IS GENERATED.** At the decisive decision the treatment
reports `branches_examined: 7` **and ≥ 1 `branch_verified` event whose
`human_prior_target_hearts` drops `[192,176]`** — the collecting branch provably
exists in the verified set. **Scored independently of the outcome.** This bit has
never been reached by any prior experiment: every prior central bit was about
ranking or candidacy among branches that existed, and bit 3 asks whether the
branch is generated at all. A bit-3 FAIL localizes the gap to **reachability**;
a bit-3 PASS with a bit-4 FAIL localizes it to **ranking**. Both are strictly
more informative than a null.

**Bit 4 — OUTCOME.** `(12,11)` / `(192,176)` collected by the treatment and not
by the control, within 24 decisions.

**Bit 5 — SAFETY.** Zero life losses in both scored arms; no
`position_not_certified` deposit; P1 (life-loss recovery, `:21295`) and P4
(goal-exhaustion rollback, `:21732`) unmodified and un-suppressed.

**Readings fixed in advance:**

- **Bits 1–3 PASS, bit 4 FAIL** ⇒ the collecting branch existed and the ladder
  still discarded it. *That* is the ranking failure roadmap §22 hypothesised, and
  it is then — and only then — grounds for the ladder-as-policy-object rewrite
  (§3.3(1)). It also fires **§5.7 retirement condition 3** if it recurs across a
  second independent intervention.
- **Bits 1–2 PASS, bit 3 FAIL** ⇒ reachability, not ranking. The ladder is
  exonerated and the object moves to branch generation. A valuable narrowing, not
  a null, and it must be reported as such.
- **Bit 1(a) FAIL** (treatment's d1–d16 diverges from v341) ⇒ pre-declared
  reading: "the gate widened the intervention window and moved the approach". A
  **FAIL with a named mechanism** — not a VOID, and **not** grounds for
  re-scoping and re-running.
- **`ladder_tier_committed == ladder_tier_would_have_committed_without_R_B` at
  the decisive instant** ⇒ R-B was redundant, and the report must say so
  regardless of what bit 4 reads (§4.3).

## 8.8 Honest power analysis

- **There is no sampling variance.** Runs are deterministic. "Power" means
  opportunity count.
- **The guaranteed opportunity count is exactly ONE**: v341 d17, the single
  decision that begins at `(12,10)` under the hold with the deposit already made.
  v341 d18 also begins at distance 1, but **R-A firing at d17 changes the
  trajectory from d17 onward, so d18 is not a guaranteed second opportunity.**
  Report it if it occurs; do not count on it. n = 1, stated plainly.
- **The corpus supplies 11 distance-≤1 decision-starts across ten runs and zero
  expansion decisions at distance ≤ 1.** E8 does not increase that supply and
  must not try to: raising `--decisions`, widening the radius, and re-enabling S1
  are all excluded by §3.2.
- **What a PASS cannot show**: that the planner can cause a search (E4 remains
  untested); that *hypothesis-driven* planning rather than a standing rule
  produced the behaviour (§6.3 — the attribution arm decides that, and E7's
  precedent is that it may not); or that a second manipulation is possible.

## 8.9 VOID conditions (a VOID is not evidence)

1. **Config inequality** — the scored pair's `planning_config` differ in any
   field except `relational_planner_authority`, `relational_planner_enabled`, and
   `relational_terminal_step`. Both must report `relational_lifecycle:
   chain_published`, `relational_navigation_seams: restore_plus_deposit`,
   `relational_decision_budget: 12`.
2. **Records inequality** — both arms `record_count: 3`, signatures
   `15604cb5…`/`37ea410d…`/`47975c94…`, `verified_accessibility_weight: 0.0`.
3. **Seeding defect** — no archived branch carrying `85fd9014d58deb42` in the
   window in either arm (E7 measured 23).
4. **Root defect** — either manifest's `episodic_resume` block does not record
   `entity-v318-room3-known-push-connected-mask-d2`, `source_decision: 1`,
   `state_source_checkpoint_event_seq: 2026`, `state_source_events_sha256:
   0bbe1d15…`.
5. **Budget defect** — either arm exceeds the 10,800 s wall ceiling and is killed
   before `run_finished`; or verified-branch counts differ by more than 1%
   **on d1–d16 only** (R-A is expected to add one decision's worth of branches
   from d17, and that delta is the mechanism, not a defect).
6. **Control-invariance defect** — the control's 24 committed state ids do not
   reproduce v340's exactly. **A crashed arm is VOID, not FAIL.**
7. **Standing-rule contamination** — no arm-3 artefact is an input to either
   scored arm.

Budget-exhausted non-reach is **censored**, never "unreachable".

## 8.10 Health-check rule (§4.52), fixed before the run

- The process is `python -m lolo_agent.neural_run` — **underscores, module
  form**. A hyphenated `pgrep` pattern can never match and its silence is not
  evidence of death.
- **Health is judged from the run's own telemetry**: monotone growth of
  `events.jsonl`, corroborated by `pgrep -f "lolo_agent.neural_run"`. The first
  `decision_committed` lands at seq ≈ **75,742** in every arm of this family;
  **zero committed decisions at 6k events is on-profile, not a death.**
- Two signals that share an author are not corroboration.
- **A crashed arm is VOID, not FAIL**, so no diagnosis under time pressure can
  convert an operational failure into evidence.
- Runs are launched **detached** (`setsid`-equivalent, output redirected to a
  log) so that stopping the wrapper cannot orphan or kill them.
- Per §4.46, report option-search counts. Expected at this root: `completed: 1,
  deferred: 9` in the control. **R-A does not grant searches; a change here is a
  finding, not a nuisance.**

## 8.11 Scoring

One deterministic scorer walks each arm's `events.jsonl` once, applies §8.7
verbatim, and writes `experiments/lolo1-wp5/e8-gate4-report.json` with a
canonical-JSON `digest_sha256` over the body. Run end-to-end **twice**; both
reports byte-identical. **Validated against v340/v341 first**: it must reproduce
E7's verdicts exactly — the single d16 deposit at `(12,10)`, the 8 restore
instants with 3 differing, `hold_matching_series [1,3,3,6,7,6,5,4]`, columns 6–12
with 44/43 deposits, `option_searches {completed: 1, deferred: 9}`, and "never
collected `(192,176)`" — before it is trusted on E8. Distances: **Chebyshev** for
§4.47/§4.48-comparable traces, **Manhattan** for the mechanism's own gate
distances.

---

# 9. E8 RESULTS — **PASS**, and the honest qualification that comes with it

**Date**: 2026-08-18. **Report**: `experiments/lolo1-wp5/e8-gate4-report.json`,
`digest_sha256: 9ab983b3ef6fdc694847a816e0f5298dfcfab1ed91cd4c1f7fbc81face93e71e`,
byte-identical across two end-to-end scorer runs, scorer validated against
v340/v341 first (15/15 checks, and it reproduces E7's own `void: false` + FAIL).
**`void: false`** on all seven VOID rules.

## 9.1 The headline

**At d17 the treatment stepped from `(12,10)` onto `(12,11)` and collected the
milestone at `(192,176)`.** Nineteen runs at this root had never reached
distance 0.

| Arm | Run id | Events | Chebyshev trace to `(12,11)` |
| --- | --- | --- | --- |
| Control | `entity-v343-…-control-off-d24` | 85,594 | `6,4,4,3,4,5,5,5,3,3,4,3,4,4,3,2,1,4,5,5,5,4,5,6` |
| Treatment | `entity-v344-…-treatment-terminalstep-d24` | 149,982 | `6,4,4,3,4,5,5,3,3,4,3,4,4,3,2,1,`**`0,0,0`**`,5,6,6,5,6` |
| Attribution (unscored) | `entity-v345-…-standingrule-store-d24` | 149,981 | identical to the treatment |

Treatment Manhattan: `9,7,7,6,8,9,10,6,5,6,4,5,4,3,2,1,`**`0,0,0`**`,10,11,10,10,11`.

## 9.2 Bits

| Bit | Verdict | Evidence |
| --- | --- | --- |
| **1 supply & prefix** | **PASS** | (a) treatment d1–d16 ≡ v341 state-for-state; (b) control ≡ v340 **exactly**, extending v333≡v334≡v336≡v338≡v340≡**v343**; (c) archive columns **widened** 6–12 → 6,**7**,8–12; deposits 43 vs 44 |
| **2 gate fires** | **PASS** | d17 seq 83097: `eligible: true`, cell `(12,10)`, distance 1, hold `85fd9014d58deb42`, `restore_suppressed: true` |
| **3 candidate generated** | **PASS** | d17 `branches_examined: 7`; exactly one collecting branch — **`down`, 16 frames**, `milestone_reward: 25.0` (seq 83189) |
| **4 outcome** | **PASS** | treatment collected at d17; control never |
| **5 safety** | **PASS** | zero life losses in both scored arms; no `position_not_certified` deposit; P1/P4 unmodified |

**The §2.1 prediction was exact.** Reasoning from d1 and d16, §2.1 said the
actuator was one 16-frame directional step. The branch that collected is `down`
at 16 frames. It was never missing and never mis-ranked — **it was never
generated**, because the ladder was never entered.

## 9.3 The gate's declines are as readable as its fire (§4.3 discharged)

Five stagnation restores were evaluated; **one** was declined:

| d | cell | distance | reason | suppressed | incumbent restore |
| --- | --- | --- | --- | --- | --- |
| 5 | (9,8) | 6 | `not_certified_adjacent` | no | (8,7) d8 |
| 8 | (7,6) | 10 | `not_certified_adjacent` | no | (9,8) d6 |
| 11 | (10,7) | 6 | `not_certified_adjacent` | no | (11,8) d4 |
| 14 | (12,7) | 4 | `not_certified_adjacent` | no | (12,8) d3 |
| **17** | **(12,10)** | **1** | `certified_adjacent_position` | **yes** | **`state-00012381`, (12,10), d1, `is_self_restore: TRUE`** |

**§2.5 is now measured, not inferred.** At d17 the incumbent would again have
restored to `state-00012381` — *the state the agent was already standing in*.
R-A's contribution was never "hold the position"; the incumbent was already
holding it. It was **freeing the decision to act**, which is exactly the
distinction §4.3 demanded be readable in the log.

## 9.4 R-B was REDUNDANT — reported because §4.3 requires it, not because it helps

`ladder_tier_committed == ladder_tier_would_have_committed_without_R_B` at d17
(both `human_prior_known_milestone_fallback`, Tier 7), and **R-B changed the tier
at ZERO of the 15 expansion decisions.**

Why, precisely: P2 was **right** that the collecting branch is `outcome_known`
and routes to Tier 7. §2.6's *further* inference — that Tier 7 "would have been
preempted by position novelty" — was true at **d16**, where Tier 6 fired, but
**false at d17**. Once R-A handed d17 back to expansion there was no unvisited
semantic frontier, `human_prior_semantic_frontier_choice` was `None`, and Tier 7
fired unaided exactly as it had at d1 and d3.

**R-A alone is sufficient. R-B did nothing.** Without the §4.3 counterfactual
instrument this experiment would have credited a lever that provably never
acted — the §4.43 redundancy failure mode, caught in-run for the first time.

Consequence for the record: E8's result is attributable to **R-A alone**, on the
strength of the in-run counterfactual at all 15 expansion decisions. A direct
R-A-only confirmation is available at zero design cost — the selector value
`decline_restore` already exists — and should be run before R-B is carried
forward. **R-B should not be carried forward on E8's evidence.**

## 9.5 §4.50 did not recur

The instrument that failed E3 and passed E5/E6/E7 passed again, and in the
*generous* direction: the treatment's archive columns are `6,7,8,9,10,11,12` —
a **superset** of the control's `6,8,9,10,11,12`. R-A widened archive geography
rather than narrowing it. Deposits 43 vs 44, within one. §3.3(3)'s abandon
condition is not met.

This is the mechanical prediction of §4.1 coming true: a point predicate defined
only at distance exactly 1 cannot remove the excursions, because it does not
exist at any distance from which an excursion is still in progress.

## 9.6 Option searches — a reported finding, not a nuisance (§4.46)

Control `{completed: 1, deferred: 9}`; treatment `{completed: 2, deferred: 9}`.
The extra completion is at **d19 — two decisions AFTER the collection**, and
d17's own search was still `deferred` with the unchanged reason
`global_semantic_archive_frontier_available`. **R-A granted no search.** The
second completion is downstream of the milestone collection, not a cause of it.

## 9.7 THE QUALIFICATION — attribution failed again, exactly as §6.3 warned

**`trajectory_identical_to_treatment: true`.** The standing-rule arm, which
re-derives the same certified cells from the record store with no reference to
the propose/score/advance machinery, produced the **identical trajectory** and
**also collected `(192,176)` at d17**. Event counts 149,981 vs 149,982.

Therefore, on the face of this report and binding on every downstream claim:

> **No Gate 4 claim may cite E8 as evidence that hypothesis-driven planning,
> rather than a standing rule, produced this behaviour.** §4.54 recorded the
> same outcome for E7; roadmap §22 consequence 3 already scope-corrected the Q3
> ruling on that basis. E8 does not repair it and must not be read as repairing
> it. What E8 establishes is narrower and should be stated in exactly these
> terms: **the agent can take a final step it has demonstrably prepared for,
> once the restore bifurcation stops spending that decision on standing still.**

## 9.8 TRIGGER FIRED — §5.7 retirement condition 1

**Bit 4 PASSED, so `(12,11)`/`(192,176)` is a solved instance and RETIRES as a
discriminator.** All subsequent Gate 4 claims move to `(8,4)`/`(9,12)`, both
confirmed present in v341's `relational_hypothesis_proposed`
`remaining_milestone_cells`. Those lie **outside** the certified envelope and
require a **second manipulation** — a capability incidental behaviour has never
demonstrated, and a strictly harder claim than the single-manipulation-plus-
closing one `(12,11)` supported.

Retirement is a **scope change, not a rescue**. It is invoked here on a PASS,
which is the only clean way to invoke it. Any move to `(8,4)`/`(9,12)` requires
its own preregistration with its own control-never-does-it evidence at the new
target.

## 9.9 What this result does NOT show

- **Not** that hypothesis-driven planning did it (§9.7).
- **Not** that R-B is useful — it provably did nothing (§9.4).
- **Not** that the planner can cause a search. E4 remains untested; R-A granted
  no search (§9.6).
- **Not** a second manipulation, and no claim about one.
- **n = 1.** The guaranteed opportunity count was exactly one and it was taken.
  Runs are deterministic, so this is a capability demonstration at one instant
  at one root, not an estimate with a variance. §7's caveat stands: the census
  is descriptive of this root and this family.
- **Assisted track throughout.** Certified records come from the player-anchored
  hold instrument (`certified_hold` provenance). No strict claim is made or
  implied; WP5's shadow campaign remains the strict path.

---

# 10. R-B removal and R-A-only confirmation (E8b)

**Date**: 2026-08-18
**Planner HEAD at removal**: `05b3127` ("Resolve E8's attribution against the
result"), working tree. `neural_planner.py` **29,449 lines** after the removal
(29,575 before). Line anchors in this section are valid **at that working tree
only** — §6.1's warning still binds, and the tests below pin positions relative
to named neighbours rather than to numbers.
**Trigger**: §9.4 ("**R-A alone is sufficient. R-B did nothing.** … **R-B should
not be carried forward on E8's evidence.**"), learnings §4.56, roadmap §24 item
3 ("**R-B is unshipped-in-effect and should be REMOVED**: it changed nothing at
15 of 15 expansion decisions. An R-A-only confirmation costs nothing
(`decline_restore` already exists). Carrying a lever with no measured effect
violates the counterfactual discipline that caught it.").

## 10.1 What was removed, with anchors

R-B was the certified-cell conjunct in **two** ladder positions (§8.3's
correction), plus the selector value, helpers and telemetry that fed them.
Every one is gone; nothing was left behind "in case".

| # | What | Where it was (pre-removal `neural_planner.py`) | State now |
| --- | --- | --- | --- |
| 1 | Selector value `decline_restore_and_certified_tier` and the constant `RELATIONAL_TERMINAL_STEP_DECLINE_RESTORE_AND_CERTIFIED_TIER` | `:173–180` | Gone. `RELATIONAL_TERMINAL_STEP_MODES == ("off", "decline_restore")` at `:179–182` |
| 2 | `_relational_certified_tier_enabled()` — R-B's selector predicate | `:19641–19654` | Gone |
| 3 | `_relational_certified_milestone_branches()` — the conjunct itself | `:20717–20762` | Gone |
| 4 | **Tier 7's computation interlock**, relaxed `else` branch | `:23805–23828` | Restored to the pre-E8 expression, `known_goal_fallback_choice = (… if known_goal_branches and human_prior_semantic_frontier_choice is None else None)` at `:23729–23742` |
| 5 | **Tier 6's guard conjunct**, `and not relational_certified_milestone_branches` | `:24325–24335` | Restored to the bare `elif human_prior_semantic_frontier_choice is not None:` at `:24212` |
| 6 | The `_first_tier(skip_certified)` counterfactual and the fields `ladder_tier_would_have_committed_without_R_B{,_index}`, `relational_certified_milestone_branches`, `certified_milestone_collecting_branches_present` | `:24092–24143` | Gone. The `relational_ladder_tier_committed` event survives with `ladder_tier_committed{,_index}` and the branch counts (`:24103`) |
| 7 | CLI choice `decline_restore_and_certified_tier` on `--relational-terminal-step` | `neural_run.py:1025–1029` | Gone; the flag now accepts `off` \| `decline_restore` and **rejects the retired value with exit code 2** |

**Verified mechanically, not by eye**: items 4 and 5 are byte-identical to the
same regions at `42e4bd0~1` (the last commit before E8's code landed), compared
programmatically. Item 6's event is kept deliberately — roadmap §24 item 4 made
"every lever ships with a counterfactual instrument" a standing invariant, and
R-A's counterfactual (`incumbent_restore_state_id` / `_cell` / `_distance` /
`_is_self_restore`) is untouched. What is dropped is only the counterfactual
that has lost its referent: with R-B gone,
`ladder_tier_would_have_committed_without_R_B` would be trivially equal to
`ladder_tier_committed` at every decision, which is a counterfactual in name
only and worse than none, because a later reader could cite it as evidence.

**What was NOT touched.** R-A is exactly as E8 shipped it: the
`_relational_terminal_step_view()` predicate (`:20656`), the guard inside
`_restore_if_stagnant` placed after `branch = max(restore_eligible, …)` and
**before** `self.archive.remove(branch)` (`:28252` guard, `:28326` removal), the
scoping to `recovery_reason == "human_prior_graph_stagnation"`, the four named
refusal reasons, the `relational_terminal_step_gate` event and every redundancy
field. Authority gating (`_relational_selection_authority()`), seam-S3
dependence and the `off` default are unchanged. `relational_planner.py` is still
untouched and the strict-lineage linter still reports **zero findings**.

**Test coverage was INVERTED, not deleted.** Where a test asserted R-B's
presence, its successor asserts the tier logic is the pre-E8 one:

| Removed test | Replacement | What the replacement asserts |
| --- | --- | --- |
| `RelationalCertifiedTierTests.test_the_tier_six_guard_yields_only_to_the_conjunct` | `RelationalCertifiedTierRemovedTests.test_tier_seven_interlock_is_the_pre_e8_expression` | Tier 7's fallback is `None` again, with no relaxed branch and no `certified` token in the region |
| — (same test, second half) | `…test_tier_six_guard_carries_no_conjunct` | The novelty guard is the bare `elif` and `relational_certified_milestone_branches` appears nowhere in the file |
| `…test_conjunct_is_empty_for_every_pre_e8_configuration`, `…test_conjunct_selects_only_certified_milestone_collections`, `…test_blast_radius_is_zero_where_no_milestone_branch_exists` | `…test_the_conjunct_helper_is_removed_not_merely_unused` | Both helpers are absent from the agent and `_relational_certified` appears nowhere in the source — a helper left behind is a lever that can be re-wired by accident |
| — (new) | `RelationalTerminalStepSelectorTests.test_the_removed_r_b_value_is_rejected_like_any_unknown_mode` | The retired selector value raises `ValueError`, and the constant is gone |
| — (new) | `RelationalTerminalStepDecideTests.test_the_r_b_counterfactual_field_is_gone_with_the_rule` | The four R-B telemetry fields are absent, and R-A's own counterfactual field is still present |
| — (new) | `RelationalTerminalStepCliTests.test_cli_rejects_an_unknown_terminal_step_mode` (extended) | A command line copied verbatim from E8's treatment arm now **fails at argparse** rather than quietly running R-A and being reported as R-A-plus-R-B |

**One pre-existing test changed, and it was changed back.**
`RelationalNavigationLadderPlacementTests.test_tier_sits_below_milestone_
collection_and_above_novelty` had its Tier-6 anchor weakened by E8 (§8.5 records
this) to the guard's stable head. It is restored to the full pre-E8 line
`elif human_prior_semantic_frontier_choice is not None:` — byte-identical to
`42e4bd0~1` modulo an explanatory comment. The anchor is now itself the
assertion that the guard carries no conjunct, so the test is strictly stronger
than the E8 version and identical in meaning to the pre-E8 one. **No
pre-existing test lost meaning.**

**Suite**: **1,155 OK, 4 skipped** (was 1,154 OK, 4 skipped). Net +1: four R-B
tests removed, five removal-inversion tests added. The count went *up* because
R-B's removal is asserted in more places than R-B's presence was.

## 10.2 PREREGISTRATION — written and committed to before the arm ran

*Nothing in §10.2–§10.6 may be revised after the run starts.*

### 10.2.1 One arm, and the control is deliberately NOT re-run

| Arm | Run id | Authority | `--relational-terminal-step` | Runs? |
| --- | --- | --- | --- | --- |
| Treatment | `entity-v346-room3-e8b-ra-only-d24` | `selection` | `decline_restore` | **Yes — the only run** |
| Control | `entity-v343-room3-e8-control-off-d24` | `off` | `off` | **NO — v343 is REUSED** |

**The control-reuse decision, stated explicitly rather than assumed.** v343 is
complete (`status: complete`, 85,594 events, `run_finished`), was scored under
§8's preregistration with `void: false`, and its 24 committed state ids
reproduce v340's exactly, extending the chain
**v333 ≡ v334 ≡ v336 ≡ v338 ≡ v340 ≡ v343**. R-B removal cannot change the
control by construction: at authority `off` the terminal-step selector is
`off`, `_relational_terminal_step_gate_enabled()` returns `False`, the ladder
regions are byte-identical to pre-E8, and this is asserted offline by
`RelationalTerminalStepInvarianceTests`. Re-running it would buy a re-derivation
of a chain already six runs long at the cost of 33 minutes of emulator time and
one more chance to introduce an operational defect.

**Consequently there is no control bit and no control VOID rule.** A v343
mismatch is **impossible by construction, not a bit**: v343's events are on
disk and unchanged, so the scorer reads them, not a fresh run. If the scorer's
re-read of v343 ever disagreed with §9's numbers, that would be a **scorer
defect or a corrupted artefact** — an operational fault to fix before scoring,
never evidence about R-B. It is recorded as `control_reused: true` with v343's
own digest on the face of the report.

### 10.2.2 Configuration — v344's manifest, one field changed

Command line: `docs/wp8-search-scheduling-design-2026-08-17.md` §11.3 verbatim,
plus `--relational-navigation-seams restore_plus_deposit`,
`--relational-lifecycle chain_published`, `--relational-planner-authority
selection`, and **`--relational-terminal-step decline_restore`**. Every other
field is copied from v344's manifest:

| Field | v344 (E8 treatment) | v346 (E8b) |
| --- | --- | --- |
| `relational_planner_authority` | `selection` | `selection` |
| `relational_planner_enabled` | `true` | `true` |
| `relational_navigation_seams` | `restore_plus_deposit` | `restore_plus_deposit` |
| `relational_lifecycle` | `chain_published` | `chain_published` |
| **`relational_terminal_step`** | **`decline_restore_and_certified_tier`** | **`decline_restore`** |
| `relational_decision_budget` | 12 | 12 |
| `verified_accessibility_weight` | 0.0 | 0.0 |
| `requested_decisions` | 24 | 24 |
| Root | `entity-v318-room3-known-push-connected-mask-d2`, `source_decision: 1`, `state_source_checkpoint_event_seq: 2026`, `state_source_events_sha256: 0bbe1d15…` | identical |
| Records | `record_count: 3`, `15604cb5…`/`37ea410d…`/`47975c94…` | identical |

`relational_terminal_step` is the **only** permitted difference, and after the
removal `decline_restore` is the only non-`off` value that exists.

### 10.2.3 The bits (fixed; **ANY mixed outcome = FAIL**)

**Bit 1 — THE COLLECTION STILL HAPPENS, AT THE SAME INSTANT.** The treatment
collects the milestone at `(192,176)` by stepping `(12,10) → (12,11)` at **d17**,
exactly as v344 did. Scored from the run's own telemetry: d17's
`decision_committed` reports the cell `(12,11)`, and a `branch_verified` event
at d17 drops `[192,176]` from `human_prior_target_hearts` with
`milestone_reward: 25.0`. A collection at a *different* decision is a bit-1
FAIL, not a partial pass.

**Bit 2 — BYTE-IDENTITY TO v344. THIS IS THE REAL TEST.** The treatment's 24
committed state ids reproduce v344's **exactly**, in order:

```
state-00012280 state-00012256 state-00012294 state-00012305 state-00012257
state-00012317 state-00012322 state-00012305 state-00012335 state-00012345
state-00012344 state-00012355 state-00012363 state-00012354 state-00012371
state-00012381 state-00012390 state-00012390 state-00021371 state-00020464
state-00022642 state-00022656 state-00022649 state-00022664
```

sha256 over the canonical JSON array of that sequence:
`f56bf2f73ee2c2983671bd9731e3317a07a493d60d8c469176793b2a0ebd9148`.

**Stated plainly, because it is the point of the experiment**: §9.4 measured
R-B changing the committed tier at **0 of 15** expansion decisions, so §9.4
*predicts byte-identity*. R-A-only must be trajectory-identical to E8's
treatment. **If it is NOT, that falsifies §9.4's redundancy finding and is the
headline result** — it would mean R-B was doing something the in-run
counterfactual instrument missed, that E8's attribution of the PASS to R-A
alone is unsupported, that learnings §4.56's "R-A alone is sufficient" and
roadmap §24 item 3's removal order both rest on a broken instrument, and that
the §4.43 redundancy detector this campaign built has a blind spot. A divergence
is therefore **not** a reason to restore R-B; it is a reason to reopen §9.4 and
to audit the counterfactual instrument itself. It would be reported as the
finding, above bits 1 and 3, whatever they read.

A weaker corroborating quantity is recorded but **is not a bit**: v344 emitted
149,982 events to v346's expected count. Event totals may differ by the R-B
telemetry fields that no longer exist, and by nothing else; a difference there
is expected and carries no information about the trajectory.

**Bit 3 — NO LIFE-LOSS REGRESSION.** `human_prior_life_losses == 0` in the
treatment, matching v343's and v344's zero. P1 (life-loss recovery) and P4
(goal-exhaustion rollback) unmodified and un-suppressed; no
`position_not_certified` deposit.

**Readings fixed in advance:**

- **All three PASS** ⇒ R-B's removal is confirmed harmless and E8's result is
  attributable to R-A alone on direct evidence rather than on an in-run
  counterfactual. This is the expected outcome and it is a **confirmation, not a
  new capability claim**: every §9.7/§9.9 limitation carries over unchanged, and
  in particular **no Gate 4 claim may cite E8 or E8b as evidence that
  hypothesis-driven planning produced the behaviour.**
- **Bit 2 FAIL** ⇒ the headline, per above. §9.4 is reopened; R-B is *not*
  restored on this evidence.
- **Bit 2 PASS with bit 1 FAIL** ⇒ incoherent (identical state ids imply an
  identical trajectory) and diagnosed as a scorer defect, not a result.

### 10.2.4 VOID conditions (a VOID is not evidence)

1. **Config inequality** — the treatment's `planning_config` differs from
   v344's in any field except `relational_terminal_step`.
2. **Records inequality** — not `record_count: 3` with
   `15604cb5…`/`37ea410d…`/`47975c94…`, or `verified_accessibility_weight != 0.0`.
3. **Root defect** — the manifest's `episodic_resume` block does not record
   `entity-v318-room3-known-push-connected-mask-d2`, `source_decision: 1`,
   `state_source_checkpoint_event_seq: 2026`, `state_source_events_sha256:
   0bbe1d15…`.
4. **Budget defect** — the arm exceeds the 10,800 s wall ceiling and is killed
   before `run_finished`.
5. **Selector defect** — the manifest does not report
   `relational_terminal_step: decline_restore`, or the run was launched from a
   tree still carrying R-B.
6. **Crash** — a crashed arm is **VOID, not FAIL**.

Budget-exhausted non-reach is **censored**, never "unreachable".

### 10.2.5 Health-check rule (learnings §4.52 **and its 2026-08-18 addendum**)

§4.52's addendum is the operative one here, because it records this campaign's
*own* monitor calling a finished run stalled: the monitor's `pgrep -f
"lolo_agent.neural_run"` matched **its own process** in the table, so it could
never report a run gone. The general rule beneath both incidents: **two readings
that share an author are one signal, not two.**

- The run is launched **detached** (`setsid`-equivalent, `nohup`, stdout/stderr
  to a log), so stopping the wrapper cannot orphan or kill it.
- **The launcher records the PID.** Liveness is `kill -0 $PID` against that
  recorded PID. **No `pgrep` pattern is used**, precisely because a pattern
  broad enough to match the run is broad enough to match the monitor.
- Progress is the run's own telemetry: monotone growth of `events.jsonl`. That
  and `kill -0` share no author, so they are genuinely two signals.
- The first `decision_committed` lands at seq ≈ **75,742** in every arm of this
  family; **zero committed decisions at 6k events is on-profile, not a death.**
- Watchdog **10,800 s**. Observed envelope for v344 (the same configuration):
  **57 minutes**, 149,982 events.

### 10.2.6 Scoring

One deterministic scorer walks the treatment's `events.jsonl` once and v343's
once, applies §10.2.3 verbatim, and writes
`experiments/lolo1-wp5/e8b-ra-only-report.json` with a canonical-JSON
`digest_sha256` over the body. Run end-to-end **twice**; both reports
byte-identical. Distances: **Chebyshev** for §4.47/§4.48-comparable traces,
**Manhattan** for the mechanism's own gate distances. The scorer re-derives
v344's reference state-id sequence from v344's events rather than trusting the
literal above, and reports both.

## 10.3 E8b RESULTS — **PASS**, and §9.4 is confirmed by direct evidence

**Date**: 2026-08-18. **Report**:
`experiments/lolo1-wp5/e8b-ra-only-report.json`, `digest_sha256:
087d2e5879b79be2670b867df16c3d0c3edbfa787115aa1e2f479a4948dcc3f4`,
byte-identical across two end-to-end scorer runs. Scorer **validated against
v344/v343 first — 34/34 checks**, reproducing §9's numbers exactly, including
§9.4's "R-B changed the committed tier at 0 of 15". **`void: false`** on all six
VOID rules. Run: `entity-v346-room3-e8b-ra-only-d24`, 149,976 events, 24
committed decisions, **3,464 s** wall against the 10,800 s ceiling, launched
detached and monitored by `kill -0` against the launcher-recorded PID (the
watchdog exited on the run's own exit; it never fired).

### 10.3.1 Bits

| Bit | Verdict | Evidence |
| --- | --- | --- |
| **1 collection at the same instant** | **PASS** | d17 commits cell `(12,11)`; the single collecting branch is **`['down','a','a']` @ `[16,1,2]`**, `milestone_reward: 25.0`. Collections at exactly `[17]`, as v344 |
| **2 byte-identity to v344** | **PASS** | All 24 committed state ids equal, in order. Both sha256 `f56bf2f73ee2c2983671bd9731e3317a07a493d60d8c469176793b2a0ebd9148`. `first_divergence: null` |
| **3 no life-loss regression** | **PASS** | 0 life losses (v344 0, v343 0); no `position_not_certified` deposit; P1/P4 unmodified |

**Bit 2 is the result.** §9.4 predicted byte-identity and byte-identity is what
happened, to the state id. **§9.4's redundancy finding is not falsified — it is
confirmed by a direct experiment rather than by an in-run counterfactual.** E8's
PASS is attributable to **R-A alone** on evidence that no longer depends on
trusting the instrument that produced the claim. R-B removal is therefore
behaviour-preserving in the only way that matters.

### 10.3.2 Everything else reproduced too, including the §4.50 instrument

| Quantity | v346 (R-A only) | v344 (R-A + R-B) | v343 (control) |
| --- | --- | --- | --- |
| Chebyshev trace | `6,4,4,3,4,5,5,3,3,4,3,4,4,3,2,1,`**`0,0,0`**`,5,6,6,5,6` | **identical** | `…,2,1,4,5,…` (never 0) |
| Manhattan trace | `9,7,7,6,8,9,10,6,5,6,4,5,4,3,2,1,`**`0,0,0`**`,10,11,10,10,11` | **identical** | — |
| Archive geography columns | `6,7,8,9,10,11,12` | `6,7,8,9,10,11,12` | `6,8,9,10,11,12` |
| Archive branches added | 43 | 43 | 44 |
| Seam S3 deposits | 1, d16, `(12,10)`, `state-00012381` | identical | — |
| Commit-path verified branches | 105 | 105 | 105 |
| Option searches | `{completed: 2, deferred: 9}` | identical | `{completed: 1, deferred: 9}` |
| Terminal-step gate evaluations | 5; declines at d5/d8/d11/d14, **fires at d17** | identical | none emitted |
| Tier committed at d17 | `human_prior_known_milestone_fallback` (**Tier 7**) | same | — |

**§4.50 did not recur**, again and in the generous direction: the treatment's
archive columns remain a **superset** of the control's. §3.3(3)'s abandon
condition is not met.

**§9.4's mechanism is re-observed directly.** At d17 Tier 7 fired **unaided**,
with the frontier choice `None` and with no R-B conjunct in the tree at all.
That is precisely §9.4's explanation of why R-B was redundant, now demonstrated
rather than inferred: R-A hands d17 back to expansion, there is no unvisited
semantic frontier, and Tier 7 wins on its own. The `relational_terminal_step_
gate` at d17 again reads `eligible: true`, cell `(12,10)`, distance 1,
`restore_suppressed: true`, `incumbent_restore_state_id: state-00012381`,
**`incumbent_restore_is_self_restore: true`** — the incumbent would once more
have restored to the state the agent was already standing in.

### 10.3.3 The one difference, localized rather than waved away

The treatment emitted **149,976** events against v344's **149,982** — a delta of
**6**. It is **not a bit**, and §10.2.3's prose about "R-B telemetry fields" was
imprecise, so the real cause was traced rather than assumed:

- **Localization**: the delta is *entirely* at **decision 19, depth 5** — 2
  fewer `human_prior_option_local_neutral_verified` with their matching 2
  `env_step` and 2 `state_loaded`. Every other event type and every other
  decision is equal, d1's option search included (2,224 in both).
- **Mechanism**: that option search memoizes its NOOP probes on
  `local_neutral_key = (id(parent), edge_duration)` — a **CPython object
  address**. Removing R-B's ~100 lines changes heap layout, so `id(parent)`
  values differ between the two processes and the cache hit/miss split moves. On
  a miss the probe is **recomputed identically**, and the companion set only
  suppresses duplicate telemetry, so no computed value changes. That is exactly
  why the committed trajectory is byte-identical.
- **Precedent**: §9.1 already recorded this class of jitter — v345 was
  `trajectory_identical_to_treatment: true` yet counted 149,981 events against
  v344's 149,982. Event-count jitter between trajectory-identical arms is a
  known property of this pipeline, and R-B is not on the option-search path.
- **Honest residual**: the mechanism is identified and the delta is confined to
  telemetry and recomputation, but the exact `id()` sequence was not
  reconstructed. It is reported as a residual, not as fully explained. A
  separate latent hazard is noted in passing and is **out of scope here**:
  keying a live cache on `id()` can in principle serve a stale entry if an
  address is reused within one search.

### 10.3.4 What this does and does not establish

- **Does**: R-B is removed, the ladder is the pre-E8 one, and the agent still
  takes the terminal step at d17 on a byte-identical trajectory. Roadmap §24
  item 3 is discharged. The §4.43 redundancy detector that caught R-B in-run is
  vindicated by an independent run.
- **Does NOT**: this is a **confirmation, not a new capability claim.** Every
  §9.7 and §9.9 limitation carries over verbatim. In particular, **no Gate 4
  claim may cite E8 or E8b as evidence that hypothesis-driven planning, rather
  than a standing rule, produced this behaviour** — v345 was
  `trajectory_identical_to_treatment` and also collected at d17, so attribution
  has now failed in both experiments that tested it. Assisted track throughout.
  **n = 1**, deterministic, one root, one family. `(12,11)`/`(192,176)` remains
  **retired** as a discriminator under §5.7 condition 1; E8b re-uses it only to
  confirm reproduction and it may not be cited as fresh discriminating evidence.
- **The control was not re-run**, by design (§10.2.1). v343 is reused and its
  reuse is stated on the face of the report.
