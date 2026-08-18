# Gate 2 / Gate 6 second-manipulation design — recon and preregistration skeleton (2026-08-18)

Status: **RECON COMPLETE, DESIGN WRITTEN, NOT PREREGISTERED.** No bit is
fixed. No emulator time was spent. No code exists. Results and bits are
fixed only when the roots are known and the R1 preconditions below are
discharged.

Commissioned by: roadmap §18 item 3, §24 item 2; learnings §4.56 plan
change. Predecessors read in full: learnings §4.28, §4.30, §4.43–§4.56;
roadmap §17–§24; `docs/object-removed-probe-2026-08-16.md`;
`docs/wp8-commit-ladder-design-2026-08-18.md`.

---

## 0. The headline, stated before the evidence

**The commission's premise does not survive measurement.** It was asked as:
`(8,4)` and `(9,12)` lie outside the certified 24-cell envelope, therefore
reaching them requires a second manipulation, therefore Gate 6 merges with
Gate 2. The middle step is false.

Three independent measurements — a pixel-derived passability map over 80
v344 frames, a 353-run corpus occupancy census, and one emulator-verified
24-action branch from v303 — agree that **the connected region reachable
from `(12,11)` in the post-first-manipulation configuration is 67 cells and
contains both remaining hearts.** The certified 24-cell envelope is a
proper subset of it. `(8,4)` is 15 steps from `(12,11)`; `(9,12)` is 36.
Every cell on both routes is floor in 60/60 sampled frames of both v325 and
v344, and every cell has been physically occupied by the player in Room 3
telemetry.

The "envelope" measures what a **depth-12, beam-128, reward-shaped exact
search** retains from one root. It was never a topological claim, and no
document in the repository ever said it was one — §1.5 lists the eight
places where the inference "outside the envelope ⇒ requires a second
manipulation" was recorded without a mechanism.

The sharpest single fact: **v344's own decision-19 option search climbed
the northern corridor to `(11,2)` — five steps from the `(8,4)` heart —
and stopped at branch depth exactly 12, the configured ceiling, with zero
life losses. At d20 a stagnation restore took the agent back to `(7,6)`.**
This is §4.48 repeating one level up: the machinery reached the
neighbourhood and the incumbent traded it away.

Consequently:

1. **This is a discovery experiment, but not the discovery that was
   scoped.** The thing to discover first is how much of Room 3 is actually
   reachable without any further manipulation. That measurement is cheap,
   uses the existing WP6a instrument, and can void the entire
   second-manipulation program for `(8,4)` — and probably for `(9,12)` too.
2. **Gate 6 should be un-merged from Gate 2** (contra roadmap §18 item 3),
   pending R1. Room 3 completion needs two hearts and the chest at `(4,6)`;
   it does not need two consecutive manipulations.
3. **Gate 2 remains a real, unmet gate** and it now has, for the first time,
   a precisely localized candidate: the single static green tile at
   `(12,12)`, which seals the 4-step route `(12,11)→(12,12)→(11,12)→
   (10,12)→(9,12)` and forces the 36-step detour. That candidate has a
   *predicted accessibility consequence stated before measurement* — 36
   steps to 4 — which is a strictly stronger epistemic step than Gate 3
   took, where the manipulation was stumbled into and measured afterwards.
4. **The agent has zero descriptor evidence about `(12,12)`, or about any
   second manipulable entity anywhere in Room 3** (§2). So if the program
   does proceed to Gate 2, it begins with discovery under a
   learning-enabled configuration, not with planning.

---

## 1. Q1 — The target geometry, established from evidence

### 1.1 The coordinate frame and the five hearts (verified, not assumed)

Cell = `(pixel_x // 16, pixel_y // 16)`, no origin offset. Grid is 16
columns × 15 rows.

- `lolo_agent/accessibility.py:153` `branch_endpoint_cell` —
  `slot[0] * 16 // frame_width, slot[1] * 15 // frame_height`
- `lolo_agent/neural_planner.py:20255` `_relational_navigation_cell` —
  `(slot[0] // 16, slot[1] // 16)`; inverse at `:20065`
- `lolo_agent/neural_planner.py:277-278` — `causal_spatial_columns = 16`,
  `causal_spatial_rows = 15`
- `lolo_agent/goal_prior.py:231` `_snap_to_tile` — half-up 16-px snap
- `tests/test_conflict_root_mining.py:477` — "slot (192, 176) -> cell (12, 11)"

**Room 3 has five hearts, not three.** From `human_prior_known_heart_slots`
in every run of the family: `[[96,128],[128,64],[128,128],[144,192],
[192,176]]` = cells `(6,8) (8,4) (8,8) (9,12) (12,11)`.

The E8 lineage's own trace, measured directly from
`entity-v344-room3-e8-treatment-terminalstep-d24/events.jsonl`
(`relational_hypothesis_proposed.remaining_milestone_cells`):

| seq | remaining milestone cells |
| --- | --- |
| 75243 (d1) | `(6,8) (8,4) (8,8) (9,12) (12,11)` |
| 82153 | `(8,4) (9,12) (12,11)` |
| 83235 (post-d17) | `(8,4) (9,12)` |

Corroborated by the d20 `goal_milestone_exhaustion_progress_reset` event
(seq 147820), whose `target_graph_signature` reads
`hearts=128,64;144,192|player=112,96`: **remaining = exactly
`{(8,4),(9,12)}`, player at `(7,6)`.**

### 1.2 What the 24-cell certified envelope actually measures

`experiments/lolo1-wp5/wp8lite-accessibility-records.json`, record
`85fd9014d58deb42`: 24 `certified_cells`, `certified_milestone_cells
[[12,11]]`, `search_depth: 12`, `search_beam: 128`, 1,530 certified of
9,691 branches. `certified_open_frontiers` is **`[]`** — the schema has a
field for exactly the question "what sat at the boundary unexpanded"
(`accessibility_preference.py:171` `CertifiedAccessibilityRecord`) and it
was never populated in any of the three records. Nothing in the record
distinguishes a wall from a budget.

I extracted every branch endpoint cell (`human_prior_target_player_slot`)
from the two certification runs and from the long-horizon runs:

| run | endpoints | distinct cells | outside the 24-cell set |
| --- | --- | --- | --- |
| v325 object-removed probe (d12) | 11,629 | 24 | none |
| v326 repetition (d12) | 11,629 | 24 | none |
| v333 e3-pre control (d24) | 14,682 | 25 | `(12,5)` ×3 |
| v344 E8 treatment (d24) | 26,853 | 29 | `(12,5)` ×115, `(12,4)` ×46, `(12,3)` ×28, `(12,2)` ×13, `(11,2)` ×1 |

v325/v326 confirm the record exactly. **v344 exceeds it by five cells, all
in a northern corridor no certified record contains.** That alone falsifies
"the envelope is the reachable set".

### 1.3 The measured passable component: 67 cells, both hearts inside

I classified every interior tile by modal 16×16 colour over 80 randomly
sampled v344 frames (dark floor `(0,0,0)`, brown floor `(86,29,0)`, heart
`(255,255,255)` passable; water `(21,95,217)`, green block `(56,135,0)`,
rock/border `(247,216,165)` blocking), then flood-filled from `(12,11)`.

```
       2  3  4  5  6  7  8  9 10 11 12
  r2   o  o  o  o  o  o  o  o  o  o  o
  r3   o  o  g  g  g  g  o  w  w  w  o
  r4   o  w  g  g  g  g  H  w  w  w  o      H = heart (8,4)
  r5   o  w  w  w  w  w  w  w  w  w  o
  r6   o  w  o  g  E  E  E  g  E  E  E      E = certified envelope cell
  r7   o  o  o  r  E  r  E  g  E  E  E      r = rock, g = green block
  r8   o  w  o  r  E  r  E  E  E  E  E      w = water, o = passable floor
  r9   o  w  o  r  E  r  g  w  w  w  E
  r10  o  w  o  g  E  E  E  w  w  w  E
  r11  o  w  o  w  w  w  w  w  w  w  E
  r12  o  o  o  o  o  o  o  H  o  o  g      H = heart (9,12)
```

- **Component containing `(12,11)`: 67 cells.**
- **Contains `(8,4)`: TRUE. Contains `(9,12)`: TRUE.**
- The 24 certified cells are a subset; **43 component cells lie outside the
  certified envelope.**

Per-tile stability, modal over 60 random frames of each of v325 and v344:

| cell | v325 | v344 | reading |
| --- | --- | --- | --- |
| `(8,2)` | floor 57 / green 3 | floor 59 / green 1 | floor; the green readings are the row-2 patroller passing through |
| `(8,3)` | floor 60 | floor 60 | floor |
| `(8,4)` | heart 60 | heart 60 | heart present |
| `(11,2)` | floor 58 / green 2 | floor 60 | floor |
| `(12,5)` | floor 49 | floor 58 | floor |
| **`(12,12)`** | **green 60** | **green 60** | **static green block** |

The transient green readings on row 2 are an autonomous entity: v344's
`entity_behaviors.csv` records types 60/61 (`24045bb75e7b6942`,
`256a846358564a63`) at `(8,2) (9,2) (10,2) (11,2) (12,2)` with
`autonomous=True` on every row and `observed_manipulation_effect=0`. That
is the §4.29 "autonomous patroller"; it patrols the corridor to `(8,4)`.
**A hazard, not a wall.**

### 1.4 Three independent measurements agree on the routes

Breadth-first over the pixel-derived passable set:

- `(12,11) → (8,4)` = **15 steps**:
  `(12,11) (12,10) (12,9) (12,8) (12,7) (12,6) (12,5) (12,4) (12,3) (12,2)
  (11,2) (10,2) (9,2) (8,2) (8,3) (8,4)`
- `(12,11) → (9,12)` = **36 steps**, via row 2 west to `(2,2)`, column 2
  south to `(2,12)`, row 12 east to `(9,12)`.
- `(11,2) → (8,4)` = **5 steps**. `(8,3) → (9,12)` = **24 steps**.

Cross-check 1 — the 353-run corpus. I extracted the distinct endpoint cells
from every `*room3*` evaluation with an `events.jsonl` (353 runs with
endpoints, 71 distinct cells). BFS over that occupancy set returns the
**identical** 15-step and 36-step paths. Per-cell support on the `(8,4)`
route ranges 99–155 runs corpus-wide and 11–27 runs within the v3xx family.

Cross-check 2 — one emulator-verified branch. `entity-v303-room3-fresh-
configuration-learning-d24x18`, seq 26910, decision 1, depth 24,
`human_prior_option_branch_verified`:

```
source (128,48)=(8,3)  ->  target (144,192)=(9,12)
path       up, left×6, down×10, right×7
durations  8, then 16 throughout
src_hearts [[144,192],[192,176]] -> tgt_hearts [[192,176]]
human_prior_milestone_reward 25.0   life_loss_confirmed False
```

That is `(8,3)` → `(2,2)` → `(2,12)` → `(9,12)`, **24 actions, the exact
length my BFS predicts for `(8,3) → (9,12)`, with the heart collected and
no life lost.** Three methods, one answer.

### 1.5 What actually blocks each heart

**`(8,4)` — nothing structural.** It sits in a two-cell pocket
`{(8,3),(8,4)}` entered from `(8,2)` above; `(8,5)` south is water,
`(7,4)` west and `(9,4)` east are green/water. The pocket is *open* to the
row-2 corridor. `(8,5)` has never been reached in 353 runs, consistent with
water. `(8,2)` has been occupied in 134 runs.

**`(9,12)` — one static green tile at `(12,12)`**, blocking the 4-step
route `(12,11)→(12,12)→(11,12)→(10,12)→(9,12)`. The 36-step western detour
around it is open. `(12,12)` has never been occupied in any of the 353
runs, and is the only cell adjacent to the certified envelope that is
blocking *and* not water.

**The operative blocker for both is search depth.** Every run in the Gate-4
family is configured `human_prior_option_search_depth = 12`,
`human_prior_option_search_beam_width = 128`,
`human_prior_option_search_milestone_extension = 0` (v344 manifest). The
runs that did reach these hearts ran deeper — v303 at depth 24 (its
collecting branch is depth 24), v305/v307 at d36.

### 1.6 The E8 treatment's own search stopped five steps short

v344 decision 19, option search from `(12,11)`, minimum branch depth at
which each northern cell was first reached:

| cell | `(12,5)` | `(12,4)` | `(12,3)` | `(12,2)` | `(11,2)` |
| --- | --- | --- | --- | --- | --- |
| min depth | 8 | 9 | 10 | 11 | **12** |

Zero life losses on any of the 203 northern endpoints. `(11,2)` was reached
at branch depth **exactly 12 — the configured ceiling** — and `(8,4)` is
five further steps. At d20, `archive_branch_restored` with reason
`human_prior_graph_stagnation`, `persistent_frontier_value 77.83`, moved
the agent to `(7,6)`. The committed trace ends d20–d24 at `(7,6) (6,6)
(6,7) (7,6) (6,6)` — back in the west.

Note also that the ascent is not a straight climb: `(12,8)` is reached at
depth 3 but `(12,7)` only at depth 7, via `(11,8) (11,7) (11,6) (12,6)`.
Column 12 is not vertically traversable between rows 6–8 in this
configuration. Any depth budget must be computed on the graph, not on
Manhattan distance.

### 1.7 Verdict on Q1

| question | answer | strength |
| --- | --- | --- |
| Where are `(8,4)`/`(9,12)` relative to the certified envelope? | Outside it, but inside the same 67-cell connected component | three independent methods agree |
| What blocks them? | Search depth 12 and a reward-shaped beam; plus one green tile at `(12,12)` on the short `(9,12)` route; plus an autonomous patroller on row 2 | pixel-modal over 80 frames; 353-run census; one verified branch |
| Is a second manipulation actually required? | **`(8,4)`: no. `(9,12)`: no — the 36-step western route is open; a manipulation at `(12,12)` would shorten it to 4** | see above |
| Is one reachable by a route no run has tried? | **Yes — `(8,4)`, five steps beyond where v344's own d19 search hit its depth ceiling** | v344 telemetry, direct |

**This is the legitimate recon outcome the commission allowed for, and it
changes the whole experiment.** The claim "outside the envelope ⇒ requires a
second manipulation" appears without a mechanism in eight places
(`roadmap.md:1614-1618`, `:1790-1793`; `learnings.md:2232-2234`, `:2866-2868`;
`wp8-commit-ladder-design-2026-08-18.md:617-623`, `:1140-1148`;
`wp8-relational-planner-design-2026-08-17.md:62-64`;
`wp8-lifecycle-design-2026-08-17.md:548`, `:587-588`;
`wp8-search-scheduling-design-2026-08-17.md:909`, `:1117`, `:1341`, `:1881`;
`findings-index-2026-08-17.md:376-378`). It should be corrected in the
record once R1 (§6.1) settles it in vivo, not before.

---

## 2. Q2 — What second manipulation is even available

### 2.1 The Room 3 entity census, from telemetry

Source: `entity_behaviors.csv` across v322–v345 — **18,031 rows, 170
distinct `(type_id, anchor_cell)` loci.** Three classes ever show
`observed_manipulation_effect = True`:

1. **The controlled sprite itself** — types 28/80/45/46/36/29, whose
   descriptors satisfy `entity_displacement == player_displacement`. Not an
   entity.
2. **The five hearts** — type 27, fingerprint `8ea4de2a516b0cbe`, at
   `(6,8) (8,4) (8,8) (9,12) (12,11)`. Appearance transition on pickup,
   never displacement. `(8,4)`: 0 manipulation rows in 470. `(9,12)`: 0 in
   470.
3. **The `(7,6)` entity already removed** — type 37, fingerprint
   `d0cf8e9bd92df1dc`, transform-in-place under `a`
   (`controlled_appearance_transition = True`, n=21), inert under `right`.

**Everything else in Room 3 is inert across all 24 runs**: type 44 at
`(7,7)(7,8)(7,9)(5,7)(5,8)(5,9)` — the rock cluster; type 31
`e9ad646c255bfc74` at `(9,7)(9,6)(8,9)(12,12)(5,6)` — the green blocks,
**including `(12,12)`**; types 30/42 at `(7,5)(9,9)(10,9)(11,9)(11,5)(8,5)`;
type 13 down column 13 — the wall; types 60/61 — the row-2 patroller. All
have `manip = 0`, `displacement = 0`, `transform = 0`.

My own read of v344's `entity_behaviors.csv` confirms `(12,12)`: type 31,
n=10, `autonomous=0`, `manip=0`. **Observed ten times, inert under
everything tried.** That is an absence of evidence produced by an absence of
probing, not evidence of inertness — see §2.3.

### 2.2 The behaviour checkpoint has no removal descriptor at all

Active checkpoint:
`experiments/lolo1-entity-v10/anonymous-behavior-relational-v2-clean.json`,
loaded `mode: frozen`, sha `984b83c3…`, in every run v322–v345. Schema 8:
97 types, 92,995 rules, 131 descriptors, 56,555 observations.

- **`types` entries carry only `{feature, observations, type_id}`.** No room
  id, no cell, no fingerprint. **No descriptor can be "Room-3-located" by
  construction** — the representation is deliberately room-agnostic.
- Of 131 descriptors, 7 carry `entity_displacement` (all unit, single-axis)
  and 4 carry `global_phase_change`.
- The schema-9 fields that make `transition_kind` return `removal` /
  `expulsion` — `entity_removed`, `removal_transit_cells`,
  `target_appearance` (`lolo_agent/entity_behavior.py:100-102`;
  `controlled_entity_removal` `:304`, `controlled_entity_expulsion` `:320`,
  `transition_kind` `:356`; `SCHEMA_VERSION = 9` at `:532`) — **occur in zero checkpoints
  and zero telemetry rows.** Their only occurrence in the repository is the
  synthetic unit test at `tests/test_entity_behavior.py:606-650`.

**The removal chain the project executed is not stored as a removal
descriptor anywhere.** Exactly one appearance-fingerprint-bound confirmed
entity interaction exists in the whole repository: v322 seq 14,
`entity_interaction_cell [7,6]`, fingerprint `bd06625b01dd444b`.

### 2.3 These runs could not have learned a second manipulation

Every v322–v345 manifest: `anonymous_entity_behavior_learning = False`,
`anonymous_entity_passive_horizons = []`,
`anonymous_entity_causal_horizons = []`,
`anonymous_entity_shadow_horizons = []`. All 18,031 behaviour rows carry
`learning_enabled = False`, `evidence_accepted = False`, and empty
`causal_attribution`.

Corroboration from the probing instruments: across all 24 runs,
`human_prior_adjacent_entity_probe_summary` reports 410 summaries and 3,578
candidates with **`confirmed_entity_effects` total = 2** — and both are the
same event in v344 and its twin v345 (decision 22), `entity_effect_cells
[[8,6]]`, the *same* object at its pushed footprint.
`human_prior_proactive_entity_probe_completed`: 408 completions, **0 with
`effect_confirmed`**.

### 2.4 Verdict on Q2, stated plainly

**No manipulable entity in Room 3 other than the already-removed `(7,6)`
object has any supporting descriptor evidence. None. The finding is an
absence, and it is a designed absence: the runs were frozen with every
entity-learning horizon empty, so they could not have produced such
evidence even if a second entity had been touched.**

Therefore, if a second manipulation is ever wanted, **the next step is
discovery, not planning** — a learning-enabled run with
`--anonymous-entity-causal-horizons` re-enabled and the existing curiosity
machinery (`human_prior_option_entity_curiosity_weight 8.0`,
`_reserve 32`, `human_prior_proactive_entity_probe_limit 16`) aimed at a
named target.

But per §1, that step is **not on Room 3's critical path**. The one thing
that would justify it on its own merits is §2.5.

### 2.5 The single candidate worth naming: `(12,12)`

If Gate 2 is pursued for its own sake rather than for Room 3 completion,
`(12,12)` is the only target the evidence supports proposing:

- It is the **only** blocking, non-water cell adjacent to the certified
  envelope (`(12,11)` is directly north of it).
- It is a **static** green tile — green in 60/60 sampled frames of both
  v325 and v344, unlike the row-2 patroller.
- It carries an observed anonymous appearance (type 31,
  `e9ad646c255bfc74`) shared with `(9,6) (9,7) (8,9) (5,6)` — so any
  transition learned here is a *type* result with four other instances in
  the same room to test transfer against, which is exactly what Gate 5/7
  want.
- **Its accessibility consequence can be stated before it is measured:**
  clearing it collapses `(12,11) → (9,12)` from 36 steps to 4, and merges
  the row-12 corridor into the envelope's short neighbourhood. Gate 3 was
  closed on a manipulation the agent stumbled into and measured afterwards;
  a *predicted-then-verified* delta is strictly stronger evidence for the
  arrangement→accessibility thesis than anything the project has.

Honest counterweight: the agent has **zero** evidence that `(12,12)` is
manipulable, ten observations of it being inert, and no mechanism by which
the `(7,6)` chain (shoot → transform → push → shoot → expel) would apply to
a static block. The row-13 boundary directly south of it means a
push-away-from-player from `(12,11)` has nowhere to go. **A discovery probe
here has a real chance of a clean negative, and should be scoped to accept
one.**

---

## 3. Q3 — The certification problem for a chained configuration

Applies only if the program reaches R3/R4 (§6.3, §6.4). Recorded now
because it is the part that cannot be improvised later.

### 3.1 What the WP6a instrument supports, exactly

`lolo_agent/accessibility.py` (854 lines, stdlib-only, pure, no CLI, no
`main()`; its only importer is `tests/test_accessibility.py`).

- `certify_branch(branch_record, root_record, *, window=None)` at `:294`.
  Precedence: causal-restore window → missing track keys → three-conjunct
  equality against the root (`anonymous_object_track_cells`,
  `human_prior_option_tracked_world_state_signature`,
  `anonymous_object_track_confirmed_world_effect_signature`).
- **The predicate is generic, not `== []`.** `RootTrackState.from_record`
  at `:253` seeds from the root record; `tests/test_accessibility.py:265`
  (`test_nonempty_root_certifies_hold_and_flags_disappearance`) proves a
  **non-empty** root track certifies holds and flags disappearance as
  departure. The `cells == []` phrasing in
  `object-removed-probe-2026-08-16.md:62` is the v325 instantiation, not
  the contract.
- `AccessibilityDelta` at `:527` is **strictly binary** —
  `delta(before, after, *, excluded_footprint_cells)` at `:621`. No
  composition, no transitivity, no lineage field. The word "chain" appears
  nowhere in the file; the only mention of a second manipulation is
  disqualifying, at `:394`: *"a second manipulation invalidates the
  fixed-layout claim for those branches."*
- `ProbeBudget` at `:351`: `search_depth, beam_width, decisions,
  wall_clock_seconds, wall_clock_ceiling_seconds, event_count,
  event_ceiling, completed_within_ceilings`. `None` means undeclared, not
  unlimited.
- `CertificationWindow` at `:173` / `certification_window` at `:221` — the
  first `archive_branch_restored` bounds certification; `admits` prefers
  `seq`, falls back to strict `decision <`, else excludes conservatively.

**Nothing structurally blocks a post-manipulation-1 root.** What blocks it
is §3.2.

### 3.2 The root-seeding trap — the single largest hazard in this design

`RootTrackState.from_record` (`accessibility.py:253-272`) seeds
`track_cells=()` and both signatures `""` when the root record lacks the
track keys. `decision_committed` events at the v325-class root carry no
track keys — `neural_planner.py:16215` sets
`legacy_track_reconstructed = metadata.get(
"human_prior_option_tracked_world_effect_cells") is None`, and
`object-removed-probe-2026-08-16.md:51-56` records exactly this for v325.

**Therefore: a second-manipulation probe rooted with
`--resume-state-run/--resume-state-decision` would re-seed the root track
empty, make manipulation 1 invisible to the predicate, and silently
collapse the two-manipulation hold back into the one-manipulation hold.**
The run would look clean and certify the wrong thing.

**Design ruling:** the C2 root must be taken from an archive or option
branch, which *does* carry the track block —
`--resume-state-archive-id`, `--resume-state-option-event-seq`, or
`--resume-state-checkpoint-event-seq` (`neural_run.py:1368`, `:1375`,
`:1383`; mutually exclusive, validated `:1448-1459`). A precondition must assert
`legacy_track_reconstructed == false` and a non-empty seeded root track in
the run's own `human_prior_root_object_state_seeded` event before any bit
is scored.

### 3.3 Roots, arms, budgets

**Roots.** Two things, kept distinct in the instrument:
`root_record` (a `decision_committed`-shaped mapping → `RootTrackState`)
and `root_state_signature` (a caller-supplied opaque provenance string,
`accessibility.py:466`, carried verbatim into
`AccessibilityDelta.source/target_state_signature`). Physical roots are
`decision_snapshot_stored.state_sha256` files under `states/`
(`run_logging.py:182-208`), digest-verified on restore
(`replay.py:177-179`), with run-level provenance in
`manifest.json → metadata.episodic_resume` (`neural_run.py:2263-2321`).

**Arms** have no representation in code. An arm is one full native run with
a distinct `--run-id`, byte-identical flags except one declared field
(`paired-accessibility-probe-2026-08-16.md:21-46`;
`wp8-commit-ladder-design-2026-08-18.md:872-889`). Ceilings are enforced by
an external watchdog — there are no ceiling flags.

**Budgets** for a chained probe must be declared in `ProbeBudget` and, per
§1.6, **computed on the graph, not on Manhattan distance**. At depth 12 the
existing family cannot reach either heart; the R1/R3 depth must be derived
from the measured route lengths (15 / 36 / 4) plus the graph detour at
column 12 rows 6–8.

### 3.4 The certification predicate under a two-manipulation hold

Stated for R4, to be fixed before execution:

> A branch is **certified configuration-held under the two-manipulation
> configuration C2** iff, measured against the C2 root record:
> `anonymous_object_track_cells == root.track_cells` **and**
> `human_prior_option_tracked_world_state_signature ==
> root.tracked_state_signature` **and**
> `anonymous_object_track_confirmed_world_effect_signature ==
> root.confirmed_effect_signature`, evaluated inside the
> `certification_window` (branches strictly before the run's first
> `archive_branch_restored`).

Three disclosures that must ship with it:

1. **Accumulated-track semantics (§4.29) are not fixed.**
   `anonymous_object_track_cells` records "changed at some point", not
   "still changed" — five of six committed cells at v324 d7 had physically
   relaxed. So C2's root track is a *superset-accumulated* set that will
   include transit cells and the `(14,5)` HUD counter. The predicate is
   therefore **conservative in the right direction** (it flags too much as
   departure, never too little), but the certified-cell count it yields is
   a lower bound. Say so in the record.
2. **Autonomous leak on the northern route.** The row-2 patroller (types
   60/61) registers only in button branches per §4.29, but every route to
   `(8,4)` crosses its corridor. The predicate must declare row 2 as an
   autonomous modulo-set or, better, the probe must report the patroller's
   registrations separately rather than modulo them away.
3. **Heart pickups register the heart cell.** Both scored targets are heart
   cells, so a collecting branch self-excludes under a naive predicate.
   Handle as v325 did — declare the exclusion explicitly and score
   *arrival*, with collection reported as a separate, uncertified fact.

### 3.5 How the record store represents a chained configuration — it cannot

Schema (this file is untouched by the concurrent edit noted in §7.6):
`accessibility_preference.py:112` `AccessibilityRecordProvenance`
(`run_id, preregistration_doc, configuration_signature, verification,
certification_predicate, certified_branches, total_branches, search_depth,
search_beam`) and `:171` `CertifiedAccessibilityRecord` (`provenance,
certified_cells, certified_open_frontiers, certified_milestone_cells,
preparation_outcome_category, confirmed_manipulation_count`). Loader:
`neural_planner.py:29343` `load_verified_accessibility_records`; refuses
non-`certified_hold` provenance (`:29408`), duplicate signatures
(`:29436`), and a duplicate `root_configuration` designation (`:29399`).
Store class at `:29317`, `root_record` property at `:29338`.

**The key is a single opaque `configuration_signature` string.** Not a
manipulation id, not a state sha, not a lineage. Resolution is string
equality against the in-run tracked world-state signature
(`_resolve_verified_accessibility_current_record` at `:19477`;
`_archive_verified_accessibility_bonus` at `:19522`), which is
`world_effect_cells_state_signature` (`object_tracks.py:254`) —
`sha256(...)[:16]` over masked appearances of the accumulated tracked
cells, returning `""` when the cell set is empty (`:273`).
`AccessibilityRecordProvenance.__post_init__` **rejects an empty
signature** (`:138-139`), which is why the v324 baseline record in
`wp8lite-accessibility-records.json` carries the sentinel key
`prepush-root-empty-track-unmatchable` with `root_configuration: true`.

Consequences for a chained record, all forced:

- C2 keys on a *different, non-empty* accumulated signature. It is
  representable as a key. It is **not** representable as "C1 plus M2".
- The existing store already smuggles lineage into free text: the
  `85fd9014d58deb42` record's `run_id` is two run ids joined with `+`, and
  its `certification_predicate` is a 60-word provenance narrative.
  Extending that practice is the zero-code option and should be named as
  such, not pretended to be a schema.
- The minimal honest schema change is **one nullable field,
  `parent_configuration_signature`**, plus a loader rule that a record
  naming a parent must find that parent in the same file. That is additive,
  refuses nothing that loads today, and makes `confirmed_manipulation_count`
  meaningful for the first time (it is `0` on the removal record and `2` on
  the v322 displacement record today — i.e. already not load-bearing).
- `certified_open_frontiers` should finally be populated. Had it been
  populated for the removal record, §1 would have been visible in
  2026-08-16: the boundary cells `(12,5)` (floor) and `(12,12)` (blocking)
  are exactly what distinguishes a budget from a wall.

### 3.6 Precondition, not an assumption

R4 should not run before **WP2's endpoint-relative track contract**
(roadmap §17 item 3) lands. Without it, the C2 hold predicate is defined
over an accumulated set whose staleness was the subject of a recorded
correction (§4.29), and a chained record would inherit that defect
compounded twice.

---

## 4. Q4 — Discriminator design

`(12,11)` retired under `wp8-commit-ladder-design-2026-08-18.md` §5.7
retirement condition 1 (E8 bit 4 passed). A replacement needs **fresh**
control-never-does-it evidence at the new target, and §4.47's rule applies:
**validity is a trend property, not a binary at one budget.**

### 4.1 `(8,4)` is a weak discriminator and should not be used

Corpus census (streaming scan of all 353 `*room3*` runs for
`human_prior_milestone_reward ∈ {20,25}` co-occurring with
`human_prior_target_player_slot [128,64]`):

- **`(8,4)` collected in 44 runs.** v10-era, v143, v144, v146, v147, v149,
  v151, v161, v175, v176, v177b, v207, v214, v236, v239, v240, v247, v249,
  v250, and more.
- Zero of those are in the v3xx Gate-4 family — but 11 v3xx runs
  (v300–v312) *occupied* the cell.

So "the control never does it" would be true only at this root, and the
corpus shows the collection is routine elsewhere. Under §4.47 that is
exactly the pattern that converts a capability claim into a speed claim.

Worse, there is a specific trap. The seeded memory in every run of this
family (`episodic_human_prior_memory_seeded`, v318 seq 13 and v344 seq 14)
carries `exhausted_milestone_goal_slots: 1`, `exhausted_milestone_transitions: 5`,
`exhausted_milestone_contexts: 5`. The v10-era chain that produced those
values learned the transition `[(128,64),(144,192)] -> [(144,192)]`
(`room3-milestone-credit-correction-2026-08-13.md:56-57`, `:119-121`) —
that is, **collect `(8,4)` when the remaining set is exactly
`{(8,4),(9,12)}`**, which is precisely the state the E8 lineage now
occupies (§1.1). The ordering filter is a soft
`policy_effect=milestone_priority_only` preference with fail-open
(`:134-143`), not a hard veto, but a treatment scored on collecting `(8,4)`
would be measuring itself against the agent's own learned ordering prior.
**Identifying which slot that one exhausted goal slot is must be an offline
precondition (P-ORD, §6.5) — I could not resolve it from telemetry, only
narrow it.**

### 4.2 `(9,12)` is the strong discriminator

Same scan, `[144,192]`:

- **Collected in exactly 2 of 353 runs** — `entity-v294-room3-alternate-
  order-extended-d18x18` and `entity-v303-room3-fresh-configuration-
  learning-d24x18` — both inside search branches, both from roots where
  `(8,4)` was already collected.
- **Never collected on a committed decision anywhere in the corpus.**
  Aggregating `human_prior_collected_heart_slots` over every
  `decision_committed`: `[[128,64]]` ×11, `[[192,176]]` ×7, `[[96,128]]`
  ×23, `[[128,128]]` ×23 — `[144,192]` never appears.
- **Never reached in any v3xx Gate-4-family run.**
- 36 graph steps from `(12,11)`, versus 15 for `(8,4)`.
- It is the heart the learned ordering says should come *before* `(8,4)`,
  so a treatment that collects it is working with the agent's prior rather
  than against it.

### 4.3 The §4.47 trend test, specified rather than gestured at

A binary "control did not collect in N decisions" is insufficient. The
control-arm pre-check (R2, §6.2) must report, and the discriminator is
declared **alive only if all four hold**:

1. **No collection.** The control does not collect `(144,192)` in-window.
2. **No corridor entry.** The control never reaches any cell of the row-12
   corridor `{(2,12)…(11,12)}` or column 2 `{(2,3)…(2,11)}`. Reaching the
   corridor at all makes subsequent collection ordinary novelty-driven
   wandering, exactly as `(12,10)` did for `(12,11)` in §4.48.
3. **No convergent trend.** Fit the control's per-decision minimum graph
   distance to `(9,12)` over the final third of the window; the slope must
   not be negative. §4.48's trace for `(12,11)` was
   `3,3,4,3,4,4,3,2,1,4,5,5,5,4,5,6` — oscillating then diverging, which is
   what a live discriminator looks like.
4. **Reported alongside a second metric.** Minimum distance achieved, and
   the decision at which it occurred. A control that gets to distance 1 and
   bounces (the §4.48 signature) is a *live* discriminator; a control that
   descends monotonically is a dying one.

Distances are **graph distances over the measured passable set**, not
Chebyshev or Manhattan — §1.6 shows column 12 is not vertically traversable
between rows 6–8, so metric distance systematically understates cost here.
Report Chebyshev too, for comparability with the §4.47/§4.48 traces.

### 4.4 What would have to happen for the discriminator to be dead

Pre-declared, so it cannot be argued after the fact:

- **DEAD-1.** Any control arm collects `(144,192)` in-window. Immediate;
  §4.47's rule.
- **DEAD-2.** Any control arm enters the row-12 corridor or column 2. The
  remaining gap would then be traverse length, not capability.
- **DEAD-3.** The control's minimum-distance trend over the final third of
  the window is significantly negative. Speed claim, not capability claim.
- **DEAD-4.** R1 (§6.1) shows `(9,12)` is reached under certified hold by
  an ordinary deeper search with no intervention at all. Then the gap is a
  **budget parameter**, and no Gate-4-shaped claim is available at this
  target — the honest write-up is "the envelope was depth-limited", and the
  program returns to Gate 2 on its own merits (§2.5) or to Gate 6 directly.

DEAD-4 is the likely outcome and the design must want it. It is the
cheapest result and the most informative.

### 4.5 Do not reuse `(8,4)` as a fallback if `(9,12)` dies

If `(9,12)` retires, the correct move is **not** to substitute `(8,4)` —
§4.1 shows it is weaker on every axis. It is to accept that Room 3 has run
out of discriminators and that Gate 4's outcome criterion is met and its
deliberateness criterion is unmeasurable at this root, exactly as §4.56
scope limit 1 already concedes. Gate 7 (a different training room) is then
the honest next venue.

---

## 5. Q5 — Honest scoping

### 5.1 What this program can claim if it passes

**R1 (reachability re-measurement).** A certified statement of how much of
Room 3 is reachable under configuration hold from the post-first-manipulation
configuration at a declared, larger budget — and therefore a correction to
the certified record's `certified_open_frontiers` and to the eight
documents listed in §1.7. This is a **Gate 3 instrument result**, on the
assisted track.

**R2 (fresh control evidence).** A validated or retired discriminator at
`(9,12)`, with a trend, not a binary.

**R3 (`(12,12)` discovery).** Either the project's first
*predicted-then-verified* accessibility delta, or a clean certified
negative that `(12,12)` is not manipulable by any action the curiosity
machinery generates within budget. Both are publishable results.

**R4 (Gate 2 proper).** That the agent can preserve a first verified object
transition while discovering and representing a second one, with distinct
source/target track evidence and a replayable complete configuration
(roadmap §12 Gate 2 success criteria) — **only** if R3 returns a
manipulation.

### 5.2 What it cannot claim, under any outcome

1. **Not deliberateness.** Attribution has failed in both experiments that
   tested it — E7 (`deposit_events_without_a_hypothesis: 14`, identical
   85,601 event counts) and E8 (v345
   `trajectory_identical_to_treatment: true`, collected at d17). Per
   roadmap §24 item 1, no Gate 4 claim may cite E8 as evidence of
   hypothesis-driven choice, and this design inherits that limitation
   verbatim. **If any arm here reads certified cells through the same
   objective path, the report must say on its face that hypothesis scoping
   is unattributed.**
2. **Not a sixth lever.** Roadmap §22 item 1 closed the bolt-on program:
   five levers, five named mechanisms (§4.43 redundancy, §4.45/§4.46 no
   opportunity, §4.50 supply starvation, §4.51 candidate absent, §4.53
   objective absent). Nothing in §6 adds a seam. R1 is a measurement; R2 is
   a control; R3 changes a *learning* configuration, not a planning one.
3. **Not a second manipulation, if R1 succeeds.** If `(8,4)` or `(9,12)` is
   reached by a deeper ordinary search, the result explicitly *is not* a
   Gate 2 result and must not be reported as one.
4. **Not strict-track.** Everything here is assisted lineage — the certified
   records derive from the player-anchored hold instrument
   (`accessibility_preference.py:66` `VERIFICATION_CERTIFIED_HOLD`, `:166`
   `.certified`).
   WP5's promoted-to-shadow masking convention (§4.42) remains the strict
   path.
5. **Not room completion.** Room 3 completion needs both hearts **and** the
   chest at `(4,6)`. `human_prior_chest_obtained` is `False` in every run I
   examined, including `entity-v308-room3-all-hearts-chest-d18`, which
   resumed from an all-hearts-collected state and still did not obtain it.
   Gate 6 has a third component nobody has measured.
6. **Not generalisation.** The passability map, the route lengths, and the
   census are facts about Room 3 at these roots. Gate 7 is where transfer
   gets tested, and roadmap §12 already forbids a transfer claim that
   depends on an absolute coordinate.

### 5.3 The §5 invariants, applied

Roadmap §5 lists twelve invariants (`roadmap.md:153-178`). **Five more were
declared in the amendments and were never merged into §5's numbered list**
— they exist only in amendment prose. I cite them by amendment location and
flag the merge gap; correcting it is outside this document's ownership.

| # | Invariant | Source | How this design honours it |
| --- | --- | --- | --- |
| A | An intervention that narrows exploration must prove it does not starve the supply later progress consumes | §20 item 1, `roadmap.md:1666-1668` | R1 and R2 add **no** preference and **no** steering. R1 changes only a search budget; supply is measured (archive column range, deposit count) and must not narrow. |
| B | A capability layer bolted onto the incumbent must state which incumbent events drive its state transitions | §21 item 3, `roadmap.md:1702-1705` | No new layer is proposed. R3 changes entity-learning horizons, whose transitions are driven by the incumbent's own probe events (`human_prior_adjacent_entity_probe_summary`, `_proactive_entity_probe_completed`). |
| C | **Before building a capability layer, identify the actuator that will execute its final step and verify it exists** | §22 item 4, `roadmap.md:1734-1737` | The actuator for `(8,4)` is the ordinary 16-frame directional primitive — demonstrated 15 times along the measured route, and demonstrated *collecting a heart* at this exact geometry (E8 d17, `down` @ 16 frames, `milestone_reward 25.0`). The actuator for `(9,12)` is v303's verified 24-action branch. **Both actuators are verified to exist before anything is designed on top of them.** |
| D | Before building where a failure "must" live, verify the code path was reached | §23 item 3, `roadmap.md:1761-1764` | This document *is* that pass, applied to the premise rather than to a code path: the failure was asserted to live at a wall, and the wall is not there. §1.6 is the `branches_examined: 0` equivalent — the search reached the ceiling, not a boundary. |
| E | **Every lever ships with a counterfactual instrument recording what the incumbent would have done** | §24 item 4, `roadmap.md:1805-1807` | R1 ships a matched control at the incumbent depth 12 so the depth delta is attributable. R3 ships a matched learning-disabled arm. Neither result may be credited without its counterfactual — R-B was nearly credited without one. |
| F | **Measured divergence is a precondition, not an assumption, before paying complexity for hypothesis scoping** | §24 item 5, `roadmap.md:1798-1804` | **No arm in §6 is hypothesis-scoped.** No relational authority, no published chain, no store-read objective. If a later phase wants one, it must first exhibit an instance where the standing rule and the hypothesis-scoped version diverge. |

And from §5 proper, the two that bite hardest here: **#7** (unknown
behaviour fails open to experimentation — `(12,12)`'s ten inert
observations are not evidence of inertness, they are evidence of not having
probed) and **#9** (no experiment without a falsifiable hypothesis, bounded
budget, and declared stopping condition — R1's DEAD-4 is the stopping
condition for the whole program).

---

## 6. The experiment program — preregistration-ready, NOT preregistered

Ordered so that the cheapest step can cancel the most expensive. Nothing
below is fixed until the roots are known.

### 6.1 R1 — the reachability re-measurement (runs first; can void everything after it)

*This is this design's E3-pre / P2: a cheap check that can change the
experiment.*

**Question.** From the post-first-manipulation configuration, at a declared
larger search budget and with no intervention of any kind, how many cells
are reachable under certified configuration hold — and are `(8,4)` and/or
`(9,12)` among them?

**Root.** The E8 lineage's post-d17 state, taken from an **archive or
option-branch snapshot** so the track block survives (§3.2), not from
`--resume-state-decision`. Candidate: a v344 post-d17 archive at or near
`(12,11)` with hold signature `85fd9014d58deb42`. Exact archive id fixed
when the root is chosen; `state_sha256`, `episodic_resume` block, and
`legacy_track_reconstructed == false` recorded before the run.

**Arms.** Two, sequential, byte-identical except one field:

| arm | `--human-prior-option-search-depth` | scored |
| --- | --- | --- |
| Control | 12 (the incumbent family value) | yes — the §5.3(E) counterfactual |
| Treatment | ≥ 16, derived from §1.4's graph route lengths, declared before running | yes |

Everything else byte-identical, including `beam_width 128`,
`human_prior_option_search_milestone_extension 0`, authority `off`,
`verified_accessibility_weight 0.0`, and the frozen behaviour checkpoint
`984b83c3…`.

**Bits (shape fixed here; thresholds fixed when the root is known).**

- **Bit A — instrument integrity.** The root seeds a **non-empty** track;
  `legacy_track_reconstructed == false`; the certification window is
  non-degenerate (at least one decision's branches precede the first
  `archive_branch_restored`).
- **Bit B — the certified envelope grows.** The treatment's certified
  configuration-held coverage is a strict superset of the control's, and
  the control reproduces the 24-cell record (or its post-d17 analogue) — an
  in-vivo re-validation of `85fd9014d58deb42`.
- **Bit C — the northern corridor certifies.** `(12,5) (12,4) (12,3)
  (12,2) (11,2)` appear in the treatment's **certified** coverage. These
  are already known reachable (§1.2); the new fact is whether they hold
  configuration.
- **Bit D — the scored bit.** Does ≥ 1 certified configuration-held branch
  reach `(8,4)`? Separately: `(9,12)`?
- **Bit E — safety.** Zero life losses in both arms. Non-trivial: the row-2
  patroller is on the route. A life loss is a *finding* about the corridor,
  not a defect.

**Reading, fixed in advance.** Bit D YES for `(8,4)` ⇒ **DEAD-4 fires**;
the second-manipulation framing for `(8,4)` is withdrawn, §1.7's eight
document sites are corrected, and Gate 6 proceeds as a search-budget
problem. Bit D NO at completed depth is **censored-negative**, never
"unreachable" — and it is only then that R3 becomes justified.

**Cost.** One paired probe of this class is ~25–30 min per arm on the M5
(roadmap §17 item 6). A deeper search costs more; declare the wall ceiling
and the external watchdog before running.

### 6.2 R2 — fresh control-never-does-it evidence at `(9,12)`

Runs only if R1's Bit D is NO for `(9,12)`. One long-horizon authority-`off`
run from the post-E8 state, reporting §4.3's four-part trend test. This is
the direct analogue of E3-pre/v332 and v333, and like them it must be
allowed to kill the discriminator (§4.4) before any treatment is built.

Additionally required, because §4.46's finding recurs: **report option-search
counts.** The Gate-4 family runs one search — the resume audit — in most
runs; v344 managed two. A reachability claim at depth 36 that depends on
searches the planner does not schedule is not a reachability claim.

### 6.3 R3 — the `(12,12)` manipulability discovery probe (conditional)

Runs only if R1 says `(9,12)` is not certifiably reachable and R2 says the
discriminator is alive.

**This is a learning run, and that is the whole point.** Every v322–v345 run
was frozen with empty entity horizons (§2.3), so the absence of evidence
about `(12,12)` is an artefact of the configuration. R3 re-enables
`--anonymous-entity-causal-horizons` and lets the existing curiosity
machinery probe.

Two arms: learning-enabled treatment vs. the frozen incumbent (§5.3(E)).
Scored on whether any confirmed transition is recorded at `(12,12)` — i.e.
`observed_manipulation_effect = True` with `evidence_accepted = True` and a
non-empty `causal_attribution`, which **has never happened for any Room 3
locus in 18,031 rows**.

**A clean negative is a real result and must be pre-accepted as one.** The
pushed-block geometry argues against success: row 13 is the boundary
directly south of `(12,12)`, so a push away from a player standing at
`(12,11)` has nowhere to go. Record that prediction now so the outcome can
contradict it.

Provenance discipline: a learning-enabled Room 3 run writes to the assisted
development partition; `configs/evaluation-partitions.json` and the
strict-lineage linter both apply, and the frozen-parameter audit must be
re-run before and after (`docs/protocol.md`, freeze audit).

### 6.4 R4 — Gate 2 proper (conditional on R3 returning a manipulation)

Only if R3 confirms a second manipulable transition. Then, and only then:

- Certify C2 from a track-carrying archive root under §3.4's predicate.
- Emit **two** `AccessibilityDelta` records (C0→C1 and C1→C2), since the
  instrument is binary (§3.1), with the chain carried either in the free-text
  provenance fields as the store already does, or via the one additive
  `parent_configuration_signature` field proposed in §3.5.
- Populate `certified_open_frontiers` on all three records. Had it been
  populated in 2026-08-16, this document would have been unnecessary.
- Score roadmap §12's Gate 2 criteria verbatim: two transitions with
  distinct source/target track evidence; the first transition survives
  descendants and resume; the complete configuration is replayable.

### 6.5 Preconditions to discharge offline, before any emulator time

| # | Precondition | How checked | If it fails |
| --- | --- | --- | --- |
| P-ROOT | A post-d17 v344/v345 archive exists carrying the track block and hold signature `85fd9014d58deb42` | Offline scan of `option_archive_snapshot_stored` / `archive_branch_added` for track fields and signature | R1 roots at the v324 d7 snapshot instead and the northern claim is measured from `(7,6)`, at greater depth cost |
| P-SEED | That root seeds a **non-empty** `RootTrackState` | Replay `seed_human_prior_root_object_state` offline; assert `legacy_track_reconstructed == false` | §3.2 fires; R1 is re-rooted, not re-interpreted |
| P-DEPTH | The declared treatment depth covers the graph route, including the column-12 rows 6–8 detour | Offline BFS over the R1 root's passable set, reusing §1.3's method on that root's frames | Depth re-derived and re-declared **before** the run, never after |
| P-ORD | Identify which single slot `exhausted_milestone_goal_slots: 1` refers to, in the `{(8,4),(9,12)}` heart context | Offline reconstruction of the seeded memory from v318 d1, as `wp8-commit-ladder-design-2026-08-18.md` §5.1 P2 did for milestone outcomes | If it is `(128,64)`, `(8,4)` is formally disqualified as a discriminator (§4.1) and the finding is recorded either way |
| P-LINT | Strict-lineage linter clean; assisted/strict boundary intact for a learning-enabled R3 | `python -m lolo_agent.strict_lineage …` | Fix before running |

### 6.6 VOID conditions (a VOID is not evidence)

Inherited from `wp8-commit-ladder-design-2026-08-18.md` §5.5 / §8.9, with
two additions specific to this design:

1. **Config inequality** — the scored pair's `planning_config` differ in any
   field except the one declared field.
2. **Records inequality** — both arms load identical records, identical
   signatures, `verified_accessibility_weight 0.0`.
3. **Root defect** — either manifest's `episodic_resume` block does not
   record the declared source run, decision, and `state_source_events_sha256`.
4. **Budget defect** — either arm exceeds the declared wall ceiling and is
   killed before `run_finished`.
5. **Control-invariance defect** — the control does not reproduce its
   declared predecessor state-for-state. **A crashed arm is VOID, not FAIL.**
6. **NEW — seeding defect (§3.2).** Either arm reports
   `legacy_track_reconstructed == true` or an empty root track at a root
   declared post-manipulation. This VOIDs rather than FAILs because the run
   would have certified a different configuration than the one named.
7. **NEW — window degeneracy.** The first `archive_branch_restored` precedes
   the first branch, leaving an empty certification window
   (`accessibility.py:221`). Report and re-root; do not certify.

Budget-exhausted non-reach is **censored**, never "unreachable"
(`AccessibilityDelta.non_reach_censored` is structurally `True`,
`accessibility.py:527-546`).

### 6.7 Health-check rule (§4.52 and its E8 addendum), fixed before running

- The process is `python -m lolo_agent.neural_run` — **underscores, module
  form**. A hyphenated `pgrep` can never match and its silence is not death.
- Liveness is judged by **`kill -0 $PID` against the launcher-recorded PID**,
  which shares no author with the run's telemetry — not by a `pgrep`
  pattern that can match the monitor itself (the E8 mirror-image failure).
- Progress is judged from the run's own telemetry: monotone `events.jsonl`
  growth and arrival at expected seq milestones. **Zero committed decisions
  at 6k events is on-profile, not a death** — the first
  `decision_committed` lands near seq 75,742 in this family.
- Two readings that share an author are one signal, not two.

### 6.8 Scoring

A single deterministic scorer walks each arm's `events.jsonl` once, applies
the fixed bits verbatim, and writes `experiments/lolo1-wp5/r1-gate2-report.json`
with a canonical-JSON `digest_sha256`. Run end-to-end **twice**; both reports
byte-identical. **Validated against v325/v326 first**: it must reproduce the
24-cell certified envelope, 1,530 certified of 9,691 branches, and the
Jaccard-1.0 repetition, before it is trusted on R1.

Additionally, and specific to this design: the scorer must reproduce the
three §1 measurements independently — the endpoint-cell extraction, the
pixel-modal passability map, and the BFS route lengths — so that the claim
"the envelope was depth-limited" is re-derived by the scorer rather than
inherited from this document.

---

## 7. Risks

**7.1 The premise correction could itself be wrong.** §1's passability map is
derived from modal tile colour, and colour is not physics. Two guards: the
353-run occupancy census is independent of colour, and v303's 24-action
verified branch is independent of both. All three agree. The residual risk
is that the *post-removal, three-hearts-collected* configuration differs
from every configuration measured — which is exactly what R1 tests in vivo,
before anything is built.

**7.2 The row-2 patroller could make the northern route unsurvivable.** The
203 northern endpoints in v344 d19 had zero life losses, but none went past
`(11,2)`, and the patroller's registrations cluster at `(8,2)`–`(12,2)`. A
life loss on the route is a finding (Bit E), not a defect — but it would
convert R1 from a budget question into a hazard question, and the
`human_prior_life_loss_confirmed` path must be left un-suppressed.

**7.3 Deeper search is not free, and depth is a lever with a bad history.**
§4.50 established that narrowing exploration starves supply; the converse —
widening it — has not been tested in this family and could change archive
geography. R1's control arm at depth 12 exists precisely to attribute any
such change, and Bit B requires the treatment's coverage to be a *superset*,
not merely different.

**7.4 `(12,12)` may simply be scenery.** §2.5 states the counterweight
honestly. R3 should be scoped so a clean negative closes the question rather
than motivating a fourth attempt at the same tile.

**7.5 The `neural_planner.py` monolith.** 29,142 lines and growing. This
design touches **no** planner code — R1 changes a CLI budget, R3 changes
learning horizons. That is deliberate: after five levers and one closed
bolt-on program, the cheapest useful next result should not require editing
`decide()`.

**7.6 Line anchors rot, and they rotted while this document was being
written.** Between the first and last read of this session, another writer
modified `lolo_agent/neural_planner.py` (net −230 lines),
`lolo_agent/neural_run.py`, `tests/test_ensemble_planner.py`, and appended
~245 lines to `docs/wp8-commit-ladder-design-2026-08-18.md` — consistent
with roadmap §24 item 3's standing instruction to remove R-B and with the
E8 results being written up. **Every `neural_planner.py` and
`neural_run.py` anchor here was re-verified against the tree after that
change and corrected where it had drifted** (`load_verified_accessibility_
records` moved 29469 → 29343; `_relational_navigation_cell` 20272 → 20255).
`accessibility.py`, `accessibility_preference.py`, `object_tracks.py`,
`entity_behavior.py`, `goal_prior.py`, `run_logging.py`, and `replay.py`
were untouched. Anchors must still be re-verified, not copied — the
predecessor doc's decayed ~700 lines in one working day (§4.49), and this
one decayed 126 lines in under an hour.

---

## 8. What this document does not claim

- **No code exists.** R1–R4 are specifications. No flag, no gate, no
  scorer, no test has been written.
- **No bit is scored, and nothing here is preregistered.** Bits are shapes
  with fixed *readings*; their thresholds and the roots they attach to are
  deliberately unfixed. Preregistration happens when P-ROOT through P-LINT
  are discharged.
- **No emulator time was spent.** Every measurement in §1 and §4 is a
  read-only analysis of stored artifacts: `events.jsonl`,
  `entity_behaviors.csv`, `manifest.json`, and stored PNG frames.
- **The §1 tile map and the connectivity derived from it are my
  derivation**, grounded in stored run artifacts, not a pre-existing repo
  fact and not verified against a live emulator.
- **The corpus census aggregates across configurations.** The 353-run
  occupancy union spans many roots and many heart states. It establishes
  that cells have been occupied; it does not by itself establish they are
  occupiable under the C1 hold. R1 is what would establish that.
- **P-ORD is unresolved.** Whether the one exhausted milestone goal slot is
  `(128,64)` is not determinable from the telemetry I read and must be
  settled by offline memory reconstruction before `(8,4)` is used or
  formally disqualified.
- **No claim about Gate 4 deliberateness is made or implied**, and none may
  be built on R1–R4 without an attribution instrument that has not yet
  succeeded twice.
- **No claim about Room 3 completion.** The chest at `(4,6)` has never been
  obtained in any run examined, including one that resumed from an
  all-hearts state.
