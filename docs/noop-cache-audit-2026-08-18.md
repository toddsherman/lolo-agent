# Audit: the `id()`-keyed matched-NOOP control cache

**Date**: 2026-08-18. **Scope**: `lolo_agent/neural_planner.py`,
`_search_human_prior_options`. **Trigger**: learnings §4.58 and
`docs/wp8-commit-ladder-design-2026-08-18.md` §10.3.3, which recorded the
hazard as *latent* and out of scope there.

**Verdict in one line**: the hazard is **REACHABLE, and it fires** — not
latent. A stale matched-NOOP control was observed in the very first option
search of a unit-scale fixture, roughly **once per search** at search depth
≥ 4, and it corrupts a *computed* value (`local_action_dependent`), not only
telemetry. It has been fixed by re-keying on a content identifier the node
already carries.

This audit is code-level and unit-level only. No emulator experiment was
run; nothing was committed; `docs/learnings.md` and `docs/roadmap.md` are
untouched.

---

## 1. The cache, characterised

**Site.** One dict and one companion set, both created once per option
search, immediately before the depth loop
(`lolo_agent/neural_planner.py:9707` and `:9712`; at HEAD before this change,
`:9667`–`:9671`):

```python
local_neutral_targets: Dict[K, Tuple[Frame, Optional[int], HeartGoalAnalysis]] = {}
local_neutral_events: set[K] = set()
```

**Key, before this change** (HEAD `:9750`):

```python
local_neutral_key = (id(parent), edge_duration)
```

— a CPython object address paired with the edge duration.

**What it stores.** The *matched NOOP control* for one parent: the frame
reached by `env.load_state(parent.state); env.step(Action.NOOP,
edge_duration)`, its `last_step_seq`, and the `HeartGoalAnalysis` of that
frame against `parent.frame` (`:9796`–`:9824`). This is the per-parent
baseline that the search subtracts to decide whether an action *did*
anything locally.

**Who reads it.**

- The cache itself, one lookup per `(parent, action, edge_duration)` triple
  in the expansion loop (`:9796`).
- `local_neutral_events`, keyed identically, which suppresses duplicate
  `human_prior_option_local_neutral_verified` telemetry (`:9825`).
- The values flow into `local_action_dependent_visual_difference` and
  `local_action_dependent`; `local_positive_milestone` (via
  `local_neutral_analysis.target_present` / `chest_completed` /
  `chest_obtained`); the `target_life_signature` and
  `dark_transition_started` comparisons; and through those into
  `current_branch_measured_effect` → `inert_penalty_eligible` →
  `inert_penalty` → **`score`**, and into `track_current_effect` and the
  node's stored fields. The cached value is therefore an input to beam
  ranking, not a telemetry decoration.

**Why a key can legitimately repeat.** `action_edges` pairs several actions
with the same duration, and the NOOP control does not depend on which action
shares that duration. The intended hit is exactly the repeat *inside one
parent's inner action loop*.

**Lifetime.** The dict lives for the whole search — every depth. The keyed
parents do not: `parents` is rebound at the end of each depth (`:11539`), and
a node dropped from the beam has no other referrer unless it happened to
qualify for `endpoints` / `effect_nodes` / `interaction_nodes` (`:10847` and
the two lines after it, all conditional). `_HumanPriorOptionNode` (`:474`)
stores no back-pointer to its parent — it copies parent fields by value — so
nothing in the node graph keeps ancestors alive either. **Entry lifetime
therefore strictly outlives key lifetime.** That is the defect.

---

## 2. Reachability: REACHED, demonstrated

### 2.1 The structural window

A node is a parent at **exactly one depth** (its `path` length *is* its
depth), so every legitimate hit happens inside one contiguous stretch of one
depth, during which `parents` holds a strong reference. Any hit outside that
stretch is an aliasing bug.

The window opens as follows. Let `A` be a node of depth *k* (*k* ≥ 1; the
root is pinned by `root_node` for the whole search and never dies).

1. `A` is a parent during depth *k+1* and writes its cache entries there.
2. At the end of depth *k+1*, `parents` is rebound (`:11539`). Every other
   container that held depth-*k* nodes (`depth_candidates` at `:9745`,
   `deduplicated`, `ranked_candidates`, `observed_candidates`, the reserve
   lists) was already rebound earlier in that same iteration. If `A` is not
   endpoint-retained, `A` is freed **here**.
3. The depth-*k+1* nodes were all allocated *before* that point, so they
   cannot hold `A`'s address. The first allocations that can are the
   depth-*k+2* expansions.
4. A depth-*k+2* node `B` at `A`'s address is a parent during depth *k+3*.
   Its very first lookup `(id(B), dur)` hits `A`'s entry.

So the earliest corrupted expansion is at **search depth 4**, and searches
whose maximum depth is ≤ 3 are provably immune.

Nothing holds a strong reference that would make this unreachable. The
cache stores `(Frame, seq, analysis)` and never the node; the node dataclass
has no parent field; beam retention is conditional. Contrast the *other*
`id()`-keyed structures in this file, which are safe for exactly the reason
this one was not: `retained_state_ids` / the release loop key on save-state
handles that `saved_states` holds for the whole function (`:9485`, `:9847`),
`branch_goal_analyses` and friends key on states that `verified` holds, and
`retained_parent_ids` (`:11358` onward) is consumed inside the same block
that owns the candidate lists. Those are latent-but-unreachable. Spot-checked,
not exhaustively proven.

### 2.2 The demonstration

A detector that needs no instrumentation: because a node is a parent at
exactly one depth, **every `(parent_path, parent_durations, edge_duration)`
triple the search probes must be computed exactly once.**
`human_prior_option_branch_verified` is emitted once per `(parent, action,
edge_duration)` with no early exit in the loop, and carries the child's full
control sequence, so dropping its last control recovers every probe actually
performed. Any probe with no matching
`human_prior_option_local_neutral_verified` event was served from another
parent's entry.

On an 8×8 deterministic walk fixture (`OptionSearchGridEnv`,
`PositionGoalPrior`), against HEAD's `id()` key:

| config | probes performed | controls computed | **served stale** |
| --- | --- | --- | --- |
| depth 6, beam 8 | 37 | 35–37 | **0–2, varies per process** |
| depth 7, beam 12 | 65 | 64 | **1** |
| depth 8, beam 16 | 99 | 97 | **2** |

The per-process variation is §4.58's signature reproduced at unit scale: the
same code, the same trajectory, a different heap, a different hit/miss split.

**The served control belongs to a different parent.** NOOP is a true no-op
in this fixture, so a correct control always reports the player in the
parent's own cell:

```
parent ['down','down','down','down'] at (3,7):
    control frame reports player at (2,3)   <- WRONG
```

**And it corrupts a computed value.** Diffing every
`human_prior_option_branch_verified` payload for the depth-8/beam-16 search,
pre-fix vs post-fix — same 396 branches, same paths, 8 payloads differ:

```
['down','down','down','down','down']
    pre : local_neutral_slot [3,4]  local_action_dependent True   diff 0.03125
    post: local_neutral_slot [3,7]  local_action_dependent False  diff 0.0
```

The player is against the bottom wall, so pressing DOWN does nothing and the
correct verdict is `local_action_dependent: False`. The stale control turned
a wall collision into a measured local effect. That is the exact judgement
the matched-NOOP control exists to make.

### 2.3 Frequency and depth, measured

12 searches per configuration on the same fixture, pre-fix key, counting
stale probes by the depth of the corrupted expansion:

| search max depth | corrupted expansion depth | stale probes / 12 searches |
| --- | --- | --- |
| 3 | — | 0 |
| 4 | — | 0 |
| 5 | 5 | 7 |
| 6 | 4 / 5 | 1 / 10 |
| 8 | 4 / 5 / 6 | 3 / 10 / 7 |

No stale probe was ever observed below expansion depth 4, matching §2.1's
structural bound exactly. Rate at depth 6 ≈ 0.9 stale probes per search; at
depth 8 ≈ 1.7. This is a routine occurrence, not a corner case.

---

## 3. The fix

Re-keyed on a content identifier the node already carries — no new state, no
counter, no narrowed lifetime (`:573`–`:600`):

```python
_HumanPriorOptionLocalNeutralKey = Tuple[Tuple[Action, ...], Tuple[int, ...], int]

def _human_prior_option_local_neutral_key(parent, edge_duration):
    return (parent.path, parent.durations, edge_duration)
```

and at the use site (`:9791`), with the two annotations updated. `path` and
`durations` together are the node's complete control sequence from the search
root.

### 3.1 Why a correct re-key cannot change a trajectory a correct cache would have produced

1. **Uniqueness.** `(path, durations)` identifies a node uniquely within one
   search. Induction on depth: the root is `((), ())`; at each depth
   `parents` holds pairwise-distinct nodes with distinct prefixes (the
   reserve families dedupe by `id`, and `observed_candidates` /
   `all_missing_player_candidates` are complementary filters of
   `expansion_candidates`, hence disjoint); each expands once per element of
   `action_edges`, which contains no repeated `(action, duration)` pair
   because the long-duration variant is guarded by `long_direction_duration
   != duration`; so the children's prefixes are pairwise distinct.
   Deduplication and beam selection only remove nodes.
2. **Same hit/miss set as a correct address key.** Under `id()`, a
   legitimate hit occurs iff two lookups name the same *live* object at the
   same duration. Same live object ⇒ same `(path, durations)` ⇒ same new key.
   Conversely equal new keys ⇒ same prefix ⇒ by (1) the same node ⇒ the same
   object. The two keyings agree exactly, *given* no address reuse — which is
   precisely the assumption the old key needed and did not get.
3. **Same values on a miss.** A miss recomputes
   `load_state(parent.state); step(NOOP, edge_duration)`, which is a pure
   function of `(parent.state, edge_duration)` in a deterministic emulator,
   and the state is reloaded before every probe. Recomputation is idempotent.
4. **Same telemetry semantics.** `local_neutral_events` shares the key, so it
   still emits exactly once per `(parent, duration)` — the intended dedup,
   now actually achieved.
5. **Suite.** 1,155 pre-existing tests OK, 4 skipped, before and after —
   unchanged. Total is now 1,159 OK / 4 skipped with the four tests added
   below.

### 3.2 What the fix DOES change, stated plainly

It changes behaviour wherever the bug bit — that is the point, and it must
not be glossed:

- **Event counts rise.** Every former stale suppression now emits its
  `human_prior_option_local_neutral_verified` with its matching `env_step`
  and `state_loaded`. A rerun of a recorded experiment will emit *more*
  events than the recorded run, at roughly one parent (≈ 2–4 probes) per
  search at depth ≥ 4.
- **Computed values change** at those parents, as §2.2 shows.
- Therefore **a post-fix rerun of any pre-fix run may not be
  byte-identical**, and a failure of byte-identity across this change is not
  evidence of anything except this change. Any future matched-control
  comparison must have both arms on the same side of this commit.

---

## 4. Retrospective risk

**Bound, and its basis.**

1. **Provably empty** for any option search whose maximum depth is ≤ 3, and
   for any decision where no option search ran. Basis: §2.1's structural
   argument — the earliest corrupted expansion is depth 4 — corroborated by
   §2.3's measurement, which found nothing below depth 4 in 60 searches.
2. **Not bounded away** for expansions at depth ≥ 4 in any recorded run. At
   the measured rate this means most deep searches in the recorded corpus
   probably served at least one stale control.
3. **At least one recorded run demonstrably did.** This is a strengthening of
   §4.58, which said "no run is known to be wrong". The legitimate hit count
   is a pure function of the configuration and the parent count: per parent
   it is `len(action_edges) - |{durations in action_edges}|`, both fixed. If
   two arms perform the same number of branch expansions at a depth — and
   §10.3.3 records every other event type and every other decision as equal —
   they have the same parent count and therefore the same *legitimate* miss
   count. So the 2-probe difference at d19 depth 5 between v344 and v346
   cannot be a difference in legitimate hits; it is a difference in aliasing
   collisions, and the arm with the lower count (v346,
   `entity-v346-room3-e8b-ra-only-d24`) served **at least 2 stale matched
   controls**. Note this argument depends only on *counts*, not on the two
   arms selecting the same parents, so it survives the residual §10.3.3 left
   open.
4. **What that did not do.** At d19 the stale controls did not change the
   committed decision: E8b's bit 2 passed, all 24 committed state ids equal
   v344's in order, same sha256, `first_divergence: null`. So for that pair,
   at that decision, the corruption was absorbed.

**What cannot be bounded from here.** Whether any *other* recorded result
consumed a stale control at a point where it mattered. The corrupted quantity
feeds `score`, so the mechanism to flip a beam ordering exists; the fixture
in §2.2 happened not to reorder (identical 396 branches pre and post), but
that is one fixture and is not a general argument. Establishing the bound
tighter would need the per-run event streams re-examined against the §2.2
detector, which is offline work on recorded logs and was not done here.
Stating it as unbounded rather than guessing.

**Consequence for the methodology.** Every causal claim in learnings
§4.26–§4.57 rests on matched-NOOP comparison. The controls were sound for
depths 1–3 and, at depth ≥ 4, sound except at roughly one parent per search.
The recorded conclusions are not thereby overturned — no counter-example is
in hand — but they were produced by an instrument with a known, now-measured
defect, and reruns for confirmation must be post-fix on both arms (§3.2).

---

## 5. The tests that pin it

`tests/test_ensemble_planner.py`, class
`OptionSearchLocalNeutralCacheTests`, with fixture `OptionSearchGridEnv`.

| test | pins | fails on the pre-fix key |
| --- | --- | --- |
| `test_local_neutral_key_is_the_parent_control_prefix` | the key is the control prefix, not an address | 10/10 |
| `test_local_neutral_key_never_aliases_two_control_prefixes` | stages the actual CPython address reuse (free a node, allocate until the address comes back) and asserts the two keys still differ | 10/10 |
| `test_option_search_never_serves_a_cross_parent_noop_control` | end-to-end: probes performed == controls computed, at three depth/beam settings | 1/10 |
| `test_every_served_noop_control_belongs_to_its_own_parent` | end-to-end consequence: the served control's player slot is the parent's own cell | 7/10 |

The two end-to-end tests are correctness assertions that must hold on every
run; they are *not* reliable detectors of the old defect, because whether an
address is reused depends on process heap history — which is the whole
hazard. The two key-level tests are deterministic and carry the regression
guarantee. Detection rates above were measured by temporarily restoring
`(id(parent), edge_duration)` and running the class 10 times; that edit was
reverted.

---

## 6. Files touched

- `lolo_agent/neural_planner.py` — key helper + type alias (`:573`–`:600`),
  two annotations and a comment (`:9698`–`:9712`), the key site (`:9791`).
- `tests/test_ensemble_planner.py` — appended `OptionSearchGridEnv` and
  `OptionSearchLocalNeutralCacheTests` (4 tests).
- this document.

Nothing else. Not committed.
