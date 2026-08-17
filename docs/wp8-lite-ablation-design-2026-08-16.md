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

## 6. Preregistration addendum — resume root, commands, record store, window (appended 2026-08-17, BEFORE either arm runs)

Fixed now per §3.2 ("The exact resume run/decision/state digests are fixed
in a short preregistration addendum appended to this doc before launch").
The §4 seam patch is applied in the working tree as of commit `fa62287`
("Wire verified accessibility preference into restore selection"); no
scored quantity below was chosen after seeing either arm's output —
neither arm has run.

### 6.1 Exact resume root (both arms, identical)

- Memory: v318 decision-1 memory —
  `--resume-run experiments/lolo1-entity-v10/evaluations/entity-v318-room3-known-push-connected-mask-d2
  --resume-decision 1 --resume-option-search` (decoupled memory/state, as
  in every probe arm v322–v326).
- Physical state: the v318 **seq-2026 pre-push checkpoint** —
  `--resume-state-run <same v318 run dir>
  --resume-state-checkpoint-event-seq 2026`. Verified in v318's
  `events.jsonl`: seq 2026 is `goal_milestone_checkpoint_snapshot_stored`,
  decision 1, `state_id state-00000002`, state sha256
  `33addc6c7c6828bf13d35ed0666ce7712647a8b614a12e343e96ff87ddcbfb92` —
  byte-identical to the pre-push rollback checkpoint v323/v324 resumed
  from (their manifests record `state_source_checkpoint_event_seq: 2026`,
  source events sha `0bbe1d1571d2d9d02b03e51816acc07a7945ba97256ec6e710ff88c7179b6f83`;
  re-verified against the file on 2026-08-17).
- Why this root: it is the lineage where removal-class discovery and
  restore-choice stagnation demonstrably occur. In both pre-push-rooted
  searches from this exact root (v323 and v324, deterministic replays of
  one another), the decision-1 search discovered the removal and archived
  it — four `human_prior_option_archive_added` events carrying tracked
  world-state signature `85fd9014d58deb42` at seqs 62943/62948/62983/63003
  in each run — and goal-exhaustion stagnation then forced restore
  selection (`archive_branch_restored` at decisions 2/5/8 in v324). The
  §3.2 "seeded archive" requirement is therefore satisfied by the
  deterministic decision-1 search itself; no external archive injection is
  used. If the removal-class archived branch nevertheless fails to appear
  within the window in either arm, the §3.4 VOID rule applies as written.
- Input digests (re-verified 2026-08-17, all equal to the v322–v326
  manifests): host `c03694c5dd2245bc7d7e0702b1f57b2ef51adc5de4d7c5abb0261408b3e891f3`,
  core `a3450a09262109534a46098abbb00d2b016168da4b4351027b2eae5a40024886`,
  ROM `914c676959612fc6738a297b6b799dff848e43de4e9bd3c9f3c6783efd059e01`,
  neural checkpoint `bb7a7a37aaba2c4c37efe8f521e69fb428cd962c8c2831c20782305284f678b9`,
  entity-behavior checkpoint `984b83c340489b333799972082f94fe75399ba656a34fc8c9dc942f125c7c6aa`.

### 6.2 Full command lines (both arms)

One command, two arms; the arms differ ONLY in the
`--human-prior-accessibility-preference-weight` value (0.0 control, 1.0
treatment) and the `--run-id`. `--human-prior-accessibility-records` is
passed in BOTH arms (§3.2: records loaded identically; the weight alone
gates scoring). All other flags are the v323/v324 profile (reconstructed
from v324's manifest `planning_config`, which records every non-default
field; the 1-decision smoke below confirmed the reconstruction reproduces
that config and lineage):

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
  --human-prior-accessibility-preference-weight <0.0 | 1.0>
```

- Control arm: `--run-id entity-v327-room3-wp8lite-ablation-control-w0-d12`
  with weight `0.0`.
- Treatment arm: `--run-id entity-v328-room3-wp8lite-ablation-treatment-w1-d12`
  with weight `1.0`.
- Pre-launch check (mandatory, automatable): each arm's manifest
  `planning_config` must equal v324's manifest `planning_config` in every
  field except `verified_accessibility_weight`, and the
  `verified_accessibility_records_loaded` event must report
  `record_count: 3` with the §6.4 content signatures. Abort the launch on
  any mismatch.
- Budgets (restating §3.3): decisions 8; wall-clock ceiling **10,800 s per
  arm** via external watchdog; event expectation ≤ 200k per arm, overrun
  reported; one native run at a time; no depth/beam escalation; no rerun
  on an identical negative result.

### 6.3 Scoring window and bits (restated verbatim; fixed)

Scoring window: every scored bit is evaluated over each arm's **first
10,000 `human_prior_option_branch_verified` events** (the run proceeds to
its decision count; the analysis window is hard-truncated at branch 10,000
in both arms). The demonstrated per-run envelope from this root is
9,691–12,267 branches.

The three preregistered bits, verbatim from §3.4 (ALL must pass; ANY mixed
outcome = FAIL):

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

VOID rule, verbatim from §3.4: **VOID (disclosed defect, not evidence):**
if the treatment arm never faces the choice — no archived branch with the
removal-class signature exists within the window in either arm — the
seeding is defective; fix the seeding, disclose, and rerun both arms once.
A void is not a FAIL and not a PASS. Budget-exhausted non-reach is
censored, never "unreachable" (learnings §2, §4.14).

### 6.4 Certified record store and signature mapping rule

Records file: `experiments/lolo1-wp5/wp8lite-accessibility-records.json`
(gitignored artifact), sha256
`cb8449031d7ae5a2eeac5a4aad5652c6320371f1ffec8fd7978623faf3fd9aa9`,
loaded in both arms via `load_verified_accessibility_records` (the §4.7
provenance-checked path; `certified_hold` only). Three records; loader
content signatures `85fd9014d58deb42 → 15604cb5…`,
`596a1c8a3c0fc8be → 37ea410d…`,
`prepush-root-empty-track-unmatchable → 47975c94…`.

**Mapping rule (fixed now).** `_archive_verified_accessibility_bonus`
looks records up by the branch's accumulated tracked world-state
signature, so each record is keyed by a signature extracted from real
branch events, never by an invented digest:

1. **Removed-configuration record** (24 certified cells incl. milestone
   cell `(12,11)`; provenance v325+v326, 1,530 certified held branches of
   9,691 in the pre-restore window `seq < 15054`, 24-cell envelope
   re-derived from raw v325 telemetry on 2026-08-17 and identical to the
   preregistered doc's list) — keyed `85fd9014d58deb42`, the signature
   removal-class archived branches actually carry in a pre-push-rooted
   search: in each of v323 and v324 it is carried by 23
   `human_prior_option_branch_verified` events and 4
   `human_prior_option_archive_added` events, with accumulated track set
   `[[2,6],[3,7],[7,6],[11,6],[12,6],[14,5]]` — the same set the v324
   committed removal trajectory carried at d2–d7 (§4.29 decomposition).
   **Variant threshold rule:** accumulated-set variants of the removal
   class appearing in ≥10 branches would each be included mapping to this
   same record; in the evidence the sub-threshold variants are
   `7f20180008c6ecea` (3 branches, archived once, same set),
   `3cd210810b0a4038` (2 branches, archived once, same set), and superset
   `ab54160b953311bd` (2 branches, archived once, set adds
   `(2,5),(4,7),(8,6)`), all excluded by the rule; `85fd9014d58deb42` is
   the only qualifying key. Shot-in-place classes
   (`e0cb9a5836911f22`/`5ee64de9bf2c8579`, set `[[7,6],[14,5]]`) are not
   removal-class (no expulsion transit evidence) and are unmapped.
2. **Pushed-configuration record** (certified-neutral 7-cell envelope;
   provenance v322, 4,061/4,061 endpoints held by world-hash uniformity)
   — keyed `596a1c8a3c0fc8be`, the tracked world-state signature the v322
   root object state actually carried
   (`human_prior_root_object_state_seeded` seq 14, WP1 legacy
   reconstruction, cells `[[8,6]]`). Disclosed: no v322 *branch* event
   repeats this signature (the accumulated-track signature re-hashes
   appearances per endpoint frame and fragments into per-frame variants),
   and in a pre-push-rooted search pushed-class branches carry
   `[[7,6],[8,6],[14,5]]`-set signatures that fragment into 42 variants
   (max 25 branches, none archived, none qualifying under the ≥10 rule for
   any record). This record is therefore **expected to be inert in
   ranking** in both arms; it is loaded to keep the store honest per §3.2
   (the neutral configuration present with its certified envelope, so the
   treatment never scores against an artificially impoverished store).
3. **Pre-push record** (7-cell envelope; provenance v324, 1,756/12,232
   certified held) — carried under the deliberately non-matching sentinel
   key `prepush-root-empty-track-unmatchable`. See §6.5.

Module sanity checks on the loaded store (run 2026-08-17): removed vs
pre-push scores +25.0 (17 new cells + 1 milestone × 8.0), pushed vs
pre-push scores 0.0 (honest neutral), removed vs removed scores 0.0.

### 6.5 Disclosed staging gap: the empty current-side signature

The pre-push (root/current) configuration's actually-carried in-run
signature is the **empty string**: `world_effect_cells_state_signature`
returns `""` for an empty track, v323/v324/v325/v326 all seeded their
roots with `tracked_world_state_signature = None/""`, and v324's 1,756
certified held branches all log the signature as null.
`AccessibilityRecordProvenance` structurally refuses an empty
`configuration_signature`, so no record can be keyed by the value the
pre-push root actually carries, and the applied §4.2 helper's current-side
lookup (`self.current_human_prior_root_object_state
.tracked_world_state_signature`) will miss at the pre-push root. The §4.2
note reads this as "the correct refusal for an uncertified current
configuration", but at this root the current configuration IS certified
(the v324 record) — it merely carries the one signature the store cannot
represent.

Consequence, stated before launch: while the current root object state
carries the empty signature, `_archive_verified_accessibility_bonus`
returns 0.0 for every candidate — including removal-class archived
branches — in the treatment arm, and the term can begin scoring only if
the current root object state acquires a store-mapped signature mid-run
(e.g. a restore installing the removal-class configuration, after which
candidate == current and the bonus is still 0.0). Under that reading, bit
1 cannot fire for the mapping reason rather than a valuation reason. If
the treatment arm's window ends with every
`verified_accessibility_refusal_reason`/unscored restore attributable to
the unresolvable current-side record — verifiable from the §3.5 logging —
the outcome is declared a **VOID-class staging defect** (disclosed here,
before execution): fix the current-side mapping (owner seam lane; e.g.
resolve the current record through a designated root-record key rather
than the raw signature, or seed the root state with a certified
signature), disclose, and rerun both arms once, per the §3.4 VOID
sentence. It is not a FAIL of the preference term and not evidence about
valuation. No other reinterpretation of the bits is licensed.

### 6.6 One-decision control-arm smoke (staging verification; not an arm)

One 1-decision smoke of the control command (weight 0.0, records loaded,
`--log-root` outside the repo, run-id
`wp8lite-smoke-control-records-d1`) is the only execution performed at
staging time. Verified: `verified_accessibility_records_loaded` fires at
seq 3 with `record_count: 3`, the three §6.4 content signatures, and
`verified_accessibility_weight: 0.0`; results of the completed smoke
(resume lineage, frozen audit, config equality against v324's manifest)
are recorded below.

- Smoke result: RECORDED AFTER COMPLETION — see §6.7.

### 6.7 Smoke results (appended 2026-08-17 after the smoke completed)

Run `wp8lite-smoke-control-records-d1` (control command of §6.2, decisions
1, log-root outside the repo): **complete, exit 0**, 75,813 events,
`run_finished` emitted, manifest status `complete`.

- **Records loaded:** `verified_accessibility_records_loaded` at seq 3 —
  `record_count: 3`, `verified_accessibility_weight: 0.0`, content
  signatures exactly the §6.4 values
  (`85fd9014d58deb42 → 15604cb504868b33…`,
  `596a1c8a3c0fc8be → 37ea410d76472a12…`,
  `prepush-root-empty-track-unmatchable → 47975c94dea2b0fe…`).
- **Lineage:** manifest `episodic_resume` block equals v324's in every
  field (source run/decision/seqs, both events sha `0bbe1d15…`,
  decoupled memory/state, option-archive import skipped) except the new
  informational `state_source_reward_track: "assisted"` field added to
  the manifest schema after v324. `planning_config` equals v324's in all
  117 shared fields; the only difference is the new
  `verified_accessibility_weight: 0.0`.
- **Root seeding:** `human_prior_root_object_state_seeded` reports
  `tracked_world_effect_cells: []`,
  `tracked_world_state_signature: null`,
  `legacy_track_reconstructed: true` — confirming the §6.5 disclosed gap
  under the patched build.
- **Deterministic reproduction of the seeded choice:** the decision-1
  search verified 12,232 branches (v323/v324 scale exactly) and emitted
  the same 13 `human_prior_option_archive_added` events with the same
  signatures and accumulated track sets as v323/v324, including **four
  removal-class archives carrying `85fd9014d58deb42`** (seqs
  62944/62949/62984/63004; +1 vs v323/v324's seqs, accounted for by the
  extra records-loaded event). The §3.2 seeding premise and the §4.6
  control-arm invariance argument are both corroborated on the patched
  build with records loaded.
- **Safety/lineage audits:** zero `human_prior_life_loss_confirmed`
  branches; decision 1 committed (seq 75,742); `frozen_parameter_audit`
  **pass** (parameter sha `0622f3c8…` unchanged before/after, matching
  the v324 checkpoint parameter sha) and
  `anonymous_entity_behavior_parameter_audit` **pass** (sha `1f5b5a13…`
  unchanged) — the frozen build mutated nothing.
- No `archive_branch_restored` occurred within one decision, as expected
  from v324 (first restore at decision 2); the restore-time
  `verified_accessibility_*` logging is therefore exercised by the arms,
  not the smoke, and remains covered by the §6.2 pre-launch check plus
  §3.5.

Staging is complete. Neither ablation arm has been run.

### 6.8 Disclosure — root/current baseline designation fixes the §6.5 current-side mapping defect (appended 2026-08-17, BEFORE either arm runs)

The §6.5 disclosed staging gap is a VOID-class defect by its own terms:
at the preregistered root the planner's current tracked world-state
signature is the empty string, `AccessibilityRecordProvenance` refuses
empty keys, so the current-side lookup structurally missed and the
treatment bonus was 0.0 for every candidate. Fixed now, before either arm
has run, via §6.5's own remedy clause ("resolve the current record
through a designated root-record key rather than the raw signature").
Neither arm has run; no scored quantity was chosen after seeing arm
output.

**Mechanism.** The records file may designate exactly ONE record as the
root/current baseline via a `"root_configuration": true` entry field.
`load_verified_accessibility_records` returns a
`VerifiedAccessibilityRecordStore` (a dict subclass): the lookup dict
still refuses empty `configuration_signature` keys, the designated record
still carries its real, deliberately unmatchable sentinel signature
(`prepush-root-empty-track-unmatchable`), and the designation is stored
separately on the store (`root_configuration_signature`). More than one
designation, or a non-boolean flag, is refused at load. The designation
is store metadata, never record content: all three §6.4 record content
signatures are unchanged (re-verified after the edit —
`85fd9014d58deb42 → 15604cb504868b33…`,
`596a1c8a3c0fc8be → 37ea410d76472a12…`,
`prepush-root-empty-track-unmatchable → 47975c94dea2b0fe…`), so the §6.2
pre-launch check values stand as written.

**Current-side resolution rule (both seams:
`_archive_verified_accessibility_bonus` and
`_verified_accessibility_reserve_rank`, via the shared
`_resolve_verified_accessibility_current_record`).**

1. If the current root object state's tracked world-state signature is
   non-empty and mapped in the store → that record (`mapped`).
2. Else, if the signature is the empty string — the one value the store
   structurally cannot represent, i.e. exactly the preregistered root of
   §6.1 — and a baseline record is designated → the baseline record
   (`baseline`).
3. Else → no record, refusal exactly as before (`missing`).

A non-empty signature without a record is a genuinely unknown
configuration and never falls back to the baseline: the empty-only gate
in step 2 is deliberate, so refusal semantics for uncertified
configurations (including sub-threshold removal-class variants such as
`7f20180008c6ecea`, §6.4) are fully preserved. Candidate-side resolution
is unchanged. Weight 0.0 continues to consult nothing (the exploding-store
invariance test still passes; control-arm invariance argument of §4.6
unchanged).

**Telemetry.** Restore-selection events (`archive_branch_restored` and
the committed-decision restore telemetry) now carry
`verified_accessibility_current_source: mapped|baseline|missing`
(`disabled` at weight 0.0), captured at selection time before the restore
rebinds the current root object state, so a §6.5-style outcome audit can
attribute every scored/unscored restore to its resolution path.
`verified_accessibility_records_loaded` now additionally reports
`root_configuration_signature` (the §6.2 pre-launch check fields —
`record_count: 3`, the §6.4 content signatures — are unchanged; this is
an additive field, like the `state_source_reward_track` manifest field
noted in §6.7).

**Records file.** `experiments/lolo1-wp5/wp8lite-accessibility-records.json`
now carries `"root_configuration": true` on the pre-push record only. New
file sha256
`cf01a67aca2b6e8feeab38c0c85520dec2470cba2a5f2257cd817912c204d1fe`
(supersedes the §6.4 value `cb8449…`; record content and keys are
otherwise byte-identical). Module sanity values re-verified on the loaded
store through the designated baseline: removed vs pre-push baseline
+25.0 (17 new cells + 1 milestone × 8.0), pushed vs baseline 0.0,
removed vs removed 0.0.

**Tests.** Appended to `tests/test_accessibility_preference.py`
(`PlannerSeamRootBaselineTests`, `RootConfigurationLoaderTests`):
baseline resolution fires at an empty-signature root and flips restore
selection to the certified removal-class branch; removal-candidate vs
baseline scores exactly the +25.0 sanity value; candidate == baseline
scores 0.0; missing baseline preserves refusal (and logs
`current_source: missing`); a representable-but-unmapped current
signature still refuses; a mapped current signature wins over the
baseline; weight 0.0 never consults a designated store; duplicate or
non-boolean `root_configuration` is refused at load. Full suite: 908
tests, OK, 4 skipped (895 + 13 appended).

**Consequence for the preregistered bits.** With the root identified, the
§6.5 consequence paragraph is discharged: at the §6.1 root the treatment
arm's current side resolves to the certified v324 pre-push record via the
baseline designation, so bit 1 can fire (or fail) for valuation reasons,
which is what the ablation measures. The §3.4 bits, window, budgets, and
VOID rule are unchanged. Per §6.5/§3.4, both arms run (once) only after
this disclosed fix — which is the state we are in: neither arm has run.

**Smoke rerun (§6.6 command, this build).** Run
`wp8lite-smoke-control-records-d1-rootfix` (control command of §6.2,
decisions 1, weight 0.0, log-root outside the repo): **complete, exit
0**, and byte-parity with §6.7 on every recorded reference value —
75,813 events, `run_finished` emitted, manifest status `complete`;
decision-1 search verified **12,232 branches**; the same 13
`human_prior_option_archive_added` events including the **four
removal-class archives carrying `85fd9014d58deb42` at seqs
62944/62949/62984/63004** (§6.7-identical); decision 1 committed at seq
75,742 (§6.7-identical); zero `human_prior_life_loss_confirmed`;
`frozen_parameter_audit` **pass** and
`anonymous_entity_behavior_parameter_audit` **pass**. The determinism of
the seeded choice is therefore unaffected by the fix.
`verified_accessibility_records_loaded` at seq 3 now reports
`record_count: 3`, the three unchanged §6.4 content signatures,
`verified_accessibility_weight: 0.0`, **and
`root_configuration_signature: "prepush-root-empty-track-unmatchable"`**
— the baseline designation is visible at load time, satisfying the §6.5
audit requirement. `human_prior_root_object_state_seeded` still reports
`tracked_world_state_signature: null` with
`legacy_track_reconstructed: true`: the root really does carry the empty
signature, and it is now covered by the designated baseline whenever the
weight is positive (the smoke's weight is 0.0, so the term consulted
nothing, per the §4.6 invariance argument). Manifest `planning_config`
equals v324's in all 117 shared fields with the single addition
`verified_accessibility_weight: 0.0`, and the `episodic_resume` block
carries the §6.1 source seqs and events sha `0bbe1d15…` — the §6.2
pre-launch check reproduces on the fixed build. As in §6.7, no
`archive_branch_restored` occurs within one decision; the
`verified_accessibility_current_source` restore-time field is exercised
by the arms and by the appended seam tests.

Staging of the §6.5 fix is complete. Neither ablation arm has been run.
