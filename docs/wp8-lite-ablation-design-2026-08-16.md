# WP8-lite verified-accessibility preference — module and preregistered restore-selection ablation (2026-08-16)

Status: module + fixtures landed; ablation PREREGISTERED before execution;
seam patch DESCRIBED, deliberately NOT applied (`lolo_agent/neural_planner.py`
is owned by another lane — see §4.0)
Authority: roadmap §17 item 1 ("WP8-lite precedes WP8"), direction-review
Amendment A restore-selection ablation (`docs/direction-review-2026-08-16.md`
§3.A), WP8 scoring rule (`docs/roadmap.md` §7 WP8)
Evidence base: certified paired-probe series v322–v326
(`docs/paired-accessibility-probe-2026-08-16.md`,
`docs/object-removed-probe-2026-08-16.md`, learnings §4.26–§4.30)

## 1. What this is

The v322–v326 series demonstrated the roadmap §3 thesis natively: a
confirmed, persistent manipulation can be certified accessibility-neutral
(the eastward push, §4.28) while a removal-class manipulation is certified
accessibility-improving (7 → 24 certified cells including a
milestone-bearing cell, §4.30) — and the planner preserved the neutral
configuration for four run-generations while stumbling into the valuable one
non-deliberately. The bottleneck is valuation, not discovery.

WP8-lite answers with the smallest sanctioned mechanism: a
**verified-accessibility preference term** in the two existing
archive/restore-selection seams, computed by a pure module from certified
accessibility records only, promoted **only** through the preregistered
matched-budget paired ablation in §3. A mixed result is a FAIL (spatial-v10
lesson). `relational_planner.py` extraction follows the ablation's outcome
as the declared fallback (Amendment E).

## 2. The module (landed with this doc)

`lolo_agent/accessibility_preference.py`, stdlib-only, pure (no emulator,
no files, no planner state), tested by
`tests/test_accessibility_preference.py`.

Contract:

- Input: two `CertifiedAccessibilityRecord`s — the candidate configuration's
  and the current configuration's. A record carries cells reached under
  **certified configuration-hold** (the v322–v326 predicate: anonymous
  object track cells and tracked state signature match the probe root
  throughout the branch), certified-open frontier edges, certified
  milestone-bearing cells, and explicit provenance (run id,
  preregistration doc, configuration signature, verification kind,
  certification predicate, certified/total branch counts, search budget).
- Output: `AccessibilityPreferenceComponents` — a separately-loggable
  additive bonus with **every component exposed** (`log_fields()` emits the
  full flat decomposition under the `verified_accessibility_` prefix).
- WP8 scoring rule enforced structurally: a record whose provenance is not
  `certified_hold` (e.g. `predicted`) scores exactly zero on either side of
  the comparison, with the refusal exposed. Unverified predicted
  accessibility can never score as observed.
- Hardened success metric (Amendment A): the scored components are
  previously-unreachable **cells**, previously-unreachable **frontiers**
  (certified-open edges whose target is certified-reachable in *neither*
  configuration), and previously-unreachable **milestone-bearing cells**.
  Raw new-affordance counts are deliberately unscored: frontier edges into
  already-reachable cells and confirmed-manipulation counts are exposed in
  the components (`churn_excluded_frontiers`,
  `confirmed_manipulation_count`) and contribute exactly zero — a
  configuration that mints affordances by moving objects around cannot
  outscore a configuration with certified new reachable cells (regression
  fixture: `ChurnGamingTests`).
- Censoring discipline (learnings §2): certified-in-current cells absent
  from the candidate record are logged as `censored_current_only_cells`
  and never penalized; no component is ever negative.
- Preregistered module weights, fixed now for the ablation:
  `new_cell_weight=1.0`, `new_frontier_weight=1.0`,
  `new_milestone_weight=8.0` (the `AccessibilityPreferenceConfig`
  defaults). The seam multiplies the module total by a single planner
  config weight (§4.1) that is the only difference between ablation arms.
- Strict-lineage: `python -m lolo_agent.strict_lineage
  lolo_agent/accessibility_preference.py` reports `assisted: false`. The
  *records* that will feed it from v322–v326 are assisted-lineage
  (player-anchored hold instrument); the ablation is therefore an
  assisted-track experiment and claims nothing strict.

## 3. Preregistered matched-budget paired ablation

### 3.1 Question

Does adding the verified-accessibility preference term to the existing
archive/restore ranking make the planner *deliberately* restore the
certified accessibility-improving configuration and convert it into
milestone progress faster than the current frontier score, at matched
budget?

### 3.2 Arms

Two arms, one run each, byte-identical in every input and flag except one
planner-config scalar:

- **Control:** `verified_accessibility_weight = 0.0` — by construction the
  term is never computed and ranking is bit-identical to today's frontier
  score (§4.6 invariance argument).
- **Treatment:** `verified_accessibility_weight = 1.0`.

Shared setup (both arms):

- Code: one build containing the applied §4 patch; digests of host, core,
  ROM, checkpoints, and resumed `events.jsonl` recorded pre-launch, as in
  v322–v326.
- Root: the v318-lineage memory resume used by every probe arm, physical
  state = the certified pre-manipulation configuration (v324-class
  checkpoint), with the option archive seeded so that restore selection
  faces a real choice: at least one archived branch carrying the
  removal-class configuration signature (from the digest-verified v325/v326
  archives) and the ordinary neutral-configuration branches. The exact
  resume run/decision/state digests are fixed in a short preregistration
  addendum appended to this doc before launch (recon step; no scored
  quantity may be chosen after seeing either arm's output).
- Certified record store, loaded identically in **both** arms via the
  provenance-checked import path: the removal-class record (24-cell
  certified envelope, milestone-bearing cell of the `(12,11)` class,
  outcome category `removal`, provenance v325/v326) and the
  neutral/pre-manipulation records (7-cell envelopes, provenance
  v322/v324). Loading records in the control arm too keeps the arms
  code-path-identical; the weight alone gates scoring.
- Planning flags: depth 12, beam 128, v320 reserve profile, probes enabled,
  8 decisions — byte-identical to the v322–v326 series.

### 3.3 Matched budget

- **Fixed 10,000-branch budget per arm:** every scored bit is evaluated
  over each arm's first 10,000 `human_prior_option_branch_verified` events
  (the run itself proceeds to its decision count; the analysis window is
  hard-truncated at branch 10,000 in both arms). 10k sits inside the
  demonstrated per-run envelope (9,691–12,267 branches) so neither arm is
  starved, and the window is identical by construction.
- Wall-clock ceiling 10,800 s per arm, external watchdog (no ceiling flags
  exist in `lolo-neural-run`); event expectation ≤ 200k per arm, overrun
  reported.
- One native run at a time; **no depth/beam escalation; no rerun on an
  identical negative result.** Expected cost ~25–30 min/arm on the M5
  (§13 economics).

### 3.4 Preregistered bits (ALL must pass; ANY mixed outcome = FAIL)

1. **Deliberate selection.** Within the window, the treatment arm emits an
   `archive_branch_restored` event whose restored branch carries the
   removal-class configuration signature AND whose logged
   `verified_accessibility_total_bonus` is positive with the component
   decomposition showing certified new cells or a certified new
   milestone-bearing cell (i.e., the preference term ranked it, and for the
   hardened reason — not churn).
2. **Matched-budget consequence advantage.** Within the window, the
   treatment arm's committed trajectory reaches at least one
   previously-unreachable cell (outside the certified 7-cell baseline
   envelope) or collects the milestone at the previously-unreachable
   milestone-bearing cell, at a strictly earlier decision index than the
   control arm does (control never doing so within the window also
   satisfies this bit). Metric is cells/milestones only — never
   affordance counts.
3. **No safety regression.** The treatment arm records no more life-loss
   confirmations than the control arm within the window.

Outcome rules:

- **PASS (all three bits):** the preference term is promoted from
  engineering to the Gate 4 wiring path; next step per roadmap §17
  ("preparation → milestone within budget") on the monolith seams.
- **FAIL (any bit fails or is mixed):** the term stays engineering-only,
  the result is recorded in learnings, and the declared fallback is
  evaluated: `relational_planner.py` extraction (Amendment E) rather than
  weight tuning. **No post-hoc weight search on this dataset.**
- **VOID (disclosed defect, not evidence):** if the treatment arm never
  faces the choice — no archived branch with the removal-class signature
  exists within the window in either arm — the seeding is defective; fix
  the seeding, disclose, and rerun both arms once. A void is not a FAIL
  and not a PASS.
- Budget-exhausted non-reach is censored, never "unreachable"
  (learnings §2, §4.14).

### 3.5 Logging requirements

Every restore-selection event in both arms must carry the full
`verified_accessibility_*` component decomposition (§4.4), so the paired
analysis can attribute every ranking difference to named components. The
control arm logs the term as unscored/absent (weight 0.0), which is itself
the negative control on logging overhead.

### 3.6 Provenance and claims hygiene

All certified records feeding this ablation derive from assisted-track
probes (v322–v326); the ablation outcome is an assisted-track engineering
result about the planner's valuation seams. It contributes nothing to
strict-track claims (WP5 gate unaffected). The module itself is
strict-clean by the linter; only its inputs are assisted here.

## 4. Exact seam integration plan (ready-to-apply patch description)

### 4.0 Ownership and drift

**This patch is NOT applied by this lane.** `lolo_agent/neural_planner.py`
is owned by another session; this section is written to be applied by the
owner as-is. All quoted code and line numbers were read from the working
tree on 2026-08-16 (post-WP1 extraction). The coordinating references cited
`_archive_frontier_score` at ~:19507 and the reserve block at
~:18285–18775; in today's tree they sit at **:19093** and
**:18225–:18359** respectively — re-verify at apply time with the grep
anchors given per hunk, not the line numbers.

### 4.1 Config field (anchor: `class NeuralPlanningConfig`, ~:56)

Add one field to the frozen `NeuralPlanningConfig` dataclass, alongside the
existing archive weights:

```python
    verified_accessibility_weight: float = 0.0
```

and a validation clause in the existing `__post_init__`-style check block
(anchor: `"human-prior option search milestone reserve must be "`, ~:746):

```python
        if not math.isfinite(self.verified_accessibility_weight) or (
            self.verified_accessibility_weight < 0.0
        ):
            raise ValueError(
                "verified accessibility weight must be finite and "
                "non-negative"
            )
```

Default `0.0` keeps every existing run and test byte-identical (§4.6).

### 4.2 Record store and helper (anchor: `_archive_causal_spatial_bonus`, ~:19064)

Module import (top of file, with the other `lolo_agent` imports):

```python
from .accessibility_preference import (
    AccessibilityPreferenceComponents,
    CertifiedAccessibilityRecord,
    verified_accessibility_preference,
)
```

Planner state: a mapping from tracked world-state signature to certified
record, empty by default, populated only by the provenance-checked import
path (never inferred in-run):

```python
        self.verified_accessibility_records: Dict[
            str, CertifiedAccessibilityRecord
        ] = {}
```

New helper method, placed directly above `_archive_frontier_score`:

```python
    def _archive_verified_accessibility_bonus(
        self, branch: _ArchivedBranch
    ) -> Tuple[float, Optional[AccessibilityPreferenceComponents]]:
        """Verified-accessibility preference term (WP8-lite).

        Certified records only; a missing record on either side scores
        0.0 (unverified accessibility never scores as observed —
        docs/wp8-lite-ablation-design-2026-08-16.md).
        """

        if self.config.verified_accessibility_weight <= 0.0:
            return 0.0, None
        candidate = self.verified_accessibility_records.get(
            branch.tracked_world_state_signature
        )
        current = self.verified_accessibility_records.get(
            self.current_human_prior_root_object_state
            .tracked_world_state_signature
        )
        if candidate is None or current is None:
            return 0.0, None
        components = verified_accessibility_preference(candidate, current)
        return (
            self.config.verified_accessibility_weight
            * components.total_bonus,
            components,
        )
```

Key facts verified in today's tree: `_ArchivedBranch` carries
`tracked_world_state_signature: str = ""` (~:264), and the planner's
current configuration is `self.current_human_prior_root_object_state`
(seeded ~:1172, consumed as the root object state ~:9205), whose
`.tracked_world_state_signature` is exactly the value the certification
predicate compares against. An empty-signature lookup misses the mapping
and yields 0.0 — the correct refusal for an uncertified current
configuration.

### 4.3 `_archive_frontier_score` (anchor: `def _archive_frontier_score`, ~:19093)

Current code — head:

```python
    def _archive_frontier_score(self, branch: _ArchivedBranch) -> float:
        own_value = self._frontier_estimate(
            branch.frontier_signature
            or self._fallback_frontier_signature(branch.frame)
        )
```

Current code — tail (both return branches, ~:19170–:19192):

```python
        if choice_is_known:
            return (
                choice_value
                + option_bonus
                + causal_spatial_bonus
                + causal_cell_coverage_bonus
                + behavioral_edge_coverage_bonus
                + affordance_bonus
                + causal_event_bonus
                + goal_navigation_bonus
                + goal_progress_bonus
            )
        return (
            max(own_value, self.config.frontier_origin_weight * origin_value)
            + option_bonus
            + causal_spatial_bonus
            + causal_cell_coverage_bonus
            + behavioral_edge_coverage_bonus
            + affordance_bonus
            + causal_event_bonus
            + goal_navigation_bonus
            + goal_progress_bonus
        )
```

Replacement: insert one term computation after the existing
`causal_event_bonus` assignment (~:19125–:19129):

```python
        verified_accessibility_bonus, _verified_accessibility_components = (
            self._archive_verified_accessibility_bonus(branch)
        )
```

and append `+ verified_accessibility_bonus` as the final addend of **both**
return expressions (after `+ goal_progress_bonus` in each). No other line
changes. Because every `restore_key` lambda (~:26113, :26138, :26147,
:26156, :26164, :26183, :26191, :26197, :27146) and the archive-append
logging sites (~:23342, :23775) call `_archive_frontier_score`, the term
flows into every ranking variant and into `archive_frontier_value`
telemetry through this single function — the reason this seam was chosen.

### 4.4 Restore telemetry (anchors: `selected_frontier_value =`, ~:26512; `persistent_frontier_value=selected_frontier_value,`, ~:26738)

Current code:

```python
        selected_frontier_value = self._archive_frontier_score(branch)
```

Replacement (directly after that line):

```python
        (
            selected_verified_accessibility_bonus,
            selected_verified_accessibility_components,
        ) = self._archive_verified_accessibility_bonus(branch)
```

Current code in the `archive_branch_restored` emission (~:26717–:26738):

```python
            score=branch.score,
            persistent_frontier_value=selected_frontier_value,
```

Replacement — add immediately after the `persistent_frontier_value` line:

```python
            verified_accessibility_bonus=(
                selected_verified_accessibility_bonus
            ),
            **(
                selected_verified_accessibility_components.log_fields()
                if selected_verified_accessibility_components is not None
                else {
                    "verified_accessibility_scored": False,
                    "verified_accessibility_refusal_reason": (
                        "record_missing_or_disabled"
                    ),
                    "verified_accessibility_total_bonus": 0.0,
                }
            ),
```

This satisfies the WP8 rule at the seam: the weighted term and its full
component decomposition are logged on every restore, and an unscored term
is logged as unscored rather than silently zero. The second
`persistent_frontier_value=selected_frontier_value` emission (~:27045)
gets the same two additions if the owner confirms it is a restore-class
event (it shares `selected_frontier_value`; verify its event name at apply
time).

### 4.5 World-state reserve (anchors: `def _human_prior_world_state_reserve_candidates`, ~:18225; `def topology_rank`, ~:18293; call site `self._human_prior_world_state_reserve_candidates(`, ~:10943)

Current classmethod head:

```python
    @classmethod
    def _human_prior_world_state_reserve_candidates(
        cls,
        nodes: Sequence[_HumanPriorOptionNode],
    ) -> Tuple[_HumanPriorOptionNode, ...]:
```

Replacement head (the method stays a pure classmethod; the planner passes a
rank callable so instance state never leaks in):

```python
    @classmethod
    def _human_prior_world_state_reserve_candidates(
        cls,
        nodes: Sequence[_HumanPriorOptionNode],
        verified_accessibility_rank: Optional[
            Callable[[str], float]
        ] = None,
    ) -> Tuple[_HumanPriorOptionNode, ...]:
```

Current `topology_rank` (inside the same method):

```python
        def topology_rank(node: _HumanPriorOptionNode) -> tuple:
            return (
                node.world_state_reachability_axes,
                node.world_state_reachability_count,
                node.world_state_reachability_span,
                cls._human_prior_world_state_reserve_key(node),
            )
```

Replacement:

```python
        def topology_rank(node: _HumanPriorOptionNode) -> tuple:
            verified_bonus = (
                verified_accessibility_rank(
                    node.tracked_world_state_signature
                )
                if verified_accessibility_rank is not None
                else 0.0
            )
            return (
                verified_bonus > 0.0,
                verified_bonus,
                node.world_state_reachability_axes,
                node.world_state_reachability_count,
                node.world_state_reachability_span,
                cls._human_prior_world_state_reserve_key(node),
            )
```

Current call site (~:10942–:10946):

```python
                world_state_candidates = list(
                    self._human_prior_world_state_reserve_candidates(
                        observed_candidates
                    )
                )
```

Replacement:

```python
                world_state_candidates = list(
                    self._human_prior_world_state_reserve_candidates(
                        observed_candidates,
                        verified_accessibility_rank=(
                            self._verified_accessibility_reserve_rank
                        ),
                    )
                )
```

with one new small instance method beside the §4.2 helper:

```python
    def _verified_accessibility_reserve_rank(self, signature: str) -> float:
        if (
            self.config.verified_accessibility_weight <= 0.0
            or not signature
        ):
            return 0.0
        candidate = self.verified_accessibility_records.get(signature)
        current = self.verified_accessibility_records.get(
            self.current_human_prior_root_object_state
            .tracked_world_state_signature
        )
        if candidate is None or current is None:
            return 0.0
        return (
            self.config.verified_accessibility_weight
            * verified_accessibility_preference(
                candidate, current
            ).total_bonus
        )
```

Node fact verified in today's tree: `_HumanPriorOptionNode` carries
`tracked_world_state_signature` (~:305, consumed at ~:18245). The
representative-selection key (`_human_prior_world_state_reserve_key`,
~:17871) is intentionally untouched — the term reorders which world-state
configurations lead the reserve, not which endpoint represents a
configuration.

### 4.6 Behavior-invariance argument (control arm and existing tests)

With `verified_accessibility_weight = 0.0` (the default): §4.2 and §4.5
helpers return before any lookup, `_archive_frontier_score` adds a constant
`0.0`, and `topology_rank` prepends the constant elements `(False, 0.0)` to
every tuple, which cannot change any comparison ordering. The restore event
gains only constant unscored fields. The control arm is therefore the
current frontier score exactly, and the full existing suite must pass on
the patched build with defaults before either arm launches.

### 4.7 Record import path (deliberately out of this patch)

Populating `verified_accessibility_records` from the v322–v326 probe
outputs belongs to the provenance-checked importer lane
(`experience_import.py` ownership); until it lands, the harness for the
ablation may load the records via a run-setup hook that constructs
`CertifiedAccessibilityRecord`s with explicit v325/v326 provenance. No
record may ever be synthesized in-run, and `verification` must be
`certified_hold` for every imported record — the module refuses anything
else at score time.

## 5. Falsifiable promotion rule (restated)

The preference term earns planning authority only by passing §3.4 in full.
Any mixed result is a FAIL. On FAIL, the recorded next step is the
Amendment E fallback (`relational_planner.py` extraction), not weight
tuning, not budget escalation, not a rerun on the same dataset.
