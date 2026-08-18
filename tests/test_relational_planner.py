import ast
import json
import unittest
from dataclasses import fields as dataclass_fields
from pathlib import Path

from lolo_agent.accessibility_preference import (
    AccessibilityRecordProvenance,
    CertifiedAccessibilityRecord,
    OUTCOME_NONE,
    OUTCOME_REMOVAL,
    REFUSAL_CANDIDATE_NOT_CERTIFIED,
    VERIFICATION_CERTIFIED_HOLD,
    VERIFICATION_PREDICTED,
    verified_accessibility_preference,
)
from lolo_agent.relational_planner import (
    ACHIEVED_CERTIFIED_CELL_REACHED,
    ACHIEVED_CONFIGURATION_MAPS,
    ACHIEVED_HELD_ACROSS_TRANSITION,
    ADVANCE_BUDGET_EXHAUSTED,
    ADVANCE_CONTINUE,
    ADVANCE_HOLD_VIOLATED,
    ADVANCE_HYPOTHESIS_ACHIEVED,
    ADVANCE_REPLAN,
    CURRENT_SOURCE_BASELINE,
    CURRENT_SOURCE_MAPPED,
    CURRENT_SOURCE_MISSING,
    HypothesisKind,
    HypothesisPlan,
    InitiationCondition,
    REALIZATION_REACH_CELLS_UNDER_HOLD,
    REALIZATION_REPRODUCE_TRANSITION,
    REALIZATION_RESTORE_ARCHIVE,
    REFUSAL_CURRENT_RECORD_MISSING,
    RELATION_DIFFERS_FROM_RECORD,
    RELATION_MAPS_TO_RECORD,
    RealizedOption,
    RelationalPlannerConfig,
    RelationalStateView,
    ArchiveCandidateView,
    TERMINATED_BUDGET_EXHAUSTED,
    TERMINATED_CONTRADICTED,
    TERMINATED_HOLD_VIOLATED,
    TERMINATED_REPLANNED,
    TerminationCondition,
    TransitionRuleView,
    VerifiedTransitionSummary,
    advance,
    configuration_maps,
    hypothesis_log_fields,
    initiation_satisfied,
    navigation_preference,
    objective_hold_signature,
    objective_target_cells,
    option_initiation_satisfied,
    option_key,
    propose,
    target_cell_distance,
    record_attempt,
    record_success,
    resolve_current_record,
    score_hypothesis_candidate,
)
from lolo_agent.strict_lineage import lint_strict_lineage


# Anonymous synthetic fixtures: a small baseline envelope and a candidate
# configuration whose certified coverage adds an eastern band with one
# milestone-bearing cell. Cells are fixture data, not module constants.
BASELINE_CELLS = tuple((1, y) for y in range(1, 6))
EXTRA_CELLS = tuple((x, 3) for x in range(2, 6))
MILESTONE_CELL = (5, 3)
CANDIDATE_CELLS = BASELINE_CELLS + EXTRA_CELLS

CURRENT_SIGNATURE = "configuration-current"
CANDIDATE_SIGNATURE = "configuration-candidate"
BASELINE_SENTINEL = "configuration-root-baseline"


def provenance(
    configuration_signature: str,
    verification: str = VERIFICATION_CERTIFIED_HOLD,
) -> AccessibilityRecordProvenance:
    return AccessibilityRecordProvenance(
        run_id="fixture-run",
        preregistration_doc="docs/fixture-preregistration.md",
        configuration_signature=configuration_signature,
        verification=verification,
        certification_predicate=(
            "anonymous_object_track_cells == []"
            if verification == VERIFICATION_CERTIFIED_HOLD
            else ""
        ),
        certified_branches=(
            7 if verification == VERIFICATION_CERTIFIED_HOLD else 0
        ),
        total_branches=100,
        search_depth=6,
        search_beam=16,
    )


def record(
    configuration_signature: str,
    cells=BASELINE_CELLS,
    milestone_cells=(),
    verification: str = VERIFICATION_CERTIFIED_HOLD,
    outcome_category: str = OUTCOME_NONE,
) -> CertifiedAccessibilityRecord:
    return CertifiedAccessibilityRecord(
        provenance=provenance(configuration_signature, verification),
        certified_cells=tuple(cells),
        certified_milestone_cells=tuple(milestone_cells),
        preparation_outcome_category=outcome_category,
    )


class _RecordStore(dict):
    """Duck-typed twin of the monolith's record store (section 6.8)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.root_configuration_signature = None

    @property
    def root_record(self):
        if self.root_configuration_signature is None:
            return None
        return self.get(self.root_configuration_signature)


def store(records, root_signature=None) -> _RecordStore:
    built = _RecordStore(records)
    built.root_configuration_signature = root_signature
    return built


def state_view(
    configuration_signature: str = CURRENT_SIGNATURE,
    remaining=(MILESTONE_CELL,),
    player_cell=None,
    decision_index: int = 1,
) -> RelationalStateView:
    return RelationalStateView(
        configuration_signature=configuration_signature,
        track_set_signature="track-set-signature",
        player_cell=player_cell,
        remaining_milestone_cells=tuple(remaining),
        decision_index=decision_index,
    )


def default_records() -> _RecordStore:
    return store(
        {
            CURRENT_SIGNATURE: record(CURRENT_SIGNATURE),
            CANDIDATE_SIGNATURE: record(
                CANDIDATE_SIGNATURE,
                cells=CANDIDATE_CELLS,
                milestone_cells=(MILESTONE_CELL,),
                outcome_category=OUTCOME_REMOVAL,
            ),
        }
    )


def archive_candidate(
    signature: str = CANDIDATE_SIGNATURE,
    state_id: str = "state-A",
    baseline_score: float = 10.0,
) -> ArchiveCandidateView:
    return ArchiveCandidateView(
        state_id=state_id,
        configuration_signature=signature,
        baseline_score=baseline_score,
        verified_option=True,
    )


def summary(
    configuration_signature: str,
    remaining=(MILESTONE_CELL,),
    kind: str = "committed_decision",
    restored_state_id=None,
    player_cell=None,
    decision_index: int = 2,
) -> VerifiedTransitionSummary:
    return VerifiedTransitionSummary(
        kind=kind,
        decision_index=decision_index,
        configuration_signature=configuration_signature,
        track_set_signature="track-set-signature",
        player_cell=player_cell,
        remaining_milestone_cells=tuple(remaining),
        restored_state_id=restored_state_id,
    )


def propose_default(
    records=None,
    state=None,
    archive=(archive_candidate(),),
    rules=(),
    realized_options=(),
    config=RelationalPlannerConfig(),
) -> HypothesisPlan:
    return propose(
        state if state is not None else state_view(),
        records if records is not None else default_records(),
        archive,
        rules,
        realized_options,
        config,
    )


def by_kind(plan: HypothesisPlan, kind: HypothesisKind):
    return next(h for h in plan.hypotheses if h.kind is kind)


class ConfigValidationTests(unittest.TestCase):
    def test_rejects_non_positive_queue_and_budgets(self) -> None:
        with self.assertRaises(ValueError):
            RelationalPlannerConfig(max_queue=0)
        with self.assertRaises(ValueError):
            RelationalPlannerConfig(decision_budget=0)
        with self.assertRaises(ValueError):
            RelationalPlannerConfig(establish_branch_budget=-1)
        with self.assertRaises(ValueError):
            RelationalPlannerConfig(search_cost_per_branch=-0.5)

    def test_condition_validation(self) -> None:
        with self.assertRaises(ValueError):
            InitiationCondition(
                configuration_relation="somewhere_nice",
                required_record_signature="sig",
            )
        with self.assertRaises(ValueError):
            InitiationCondition(
                configuration_relation=RELATION_MAPS_TO_RECORD,
                required_record_signature="",
            )
        with self.assertRaises(ValueError):
            TerminationCondition(
                achieved_when="eventually",
                violated_when="",
                decision_budget=1,
            )
        with self.assertRaises(ValueError):
            TerminationCondition(
                achieved_when=ACHIEVED_CONFIGURATION_MAPS,
                violated_when="",
                decision_budget=0,
            )


class CurrentRecordResolutionTests(unittest.TestCase):
    """The section 6.8 baseline-designation rule, mirrored exactly."""

    def test_mapped_signature_resolves_directly(self) -> None:
        records = default_records()
        record_found, source = resolve_current_record(
            state_view(CURRENT_SIGNATURE), records
        )
        self.assertIs(record_found, records[CURRENT_SIGNATURE])
        self.assertEqual(source, CURRENT_SOURCE_MAPPED)

    def test_empty_signature_resolves_only_to_designated_baseline(
        self,
    ) -> None:
        records = default_records()
        self.assertEqual(
            resolve_current_record(state_view(""), records),
            (None, CURRENT_SOURCE_MISSING),
        )
        designated = store(
            {
                BASELINE_SENTINEL: record(BASELINE_SENTINEL),
                CANDIDATE_SIGNATURE: record(
                    CANDIDATE_SIGNATURE, cells=CANDIDATE_CELLS
                ),
            },
            root_signature=BASELINE_SENTINEL,
        )
        resolved, source = resolve_current_record(
            state_view(""), designated
        )
        self.assertIs(resolved, designated[BASELINE_SENTINEL])
        self.assertEqual(source, CURRENT_SOURCE_BASELINE)

    def test_unknown_nonempty_signature_never_falls_back(self) -> None:
        designated = store(
            {BASELINE_SENTINEL: record(BASELINE_SENTINEL)},
            root_signature=BASELINE_SENTINEL,
        )
        self.assertEqual(
            resolve_current_record(
                state_view("configuration-unknown"), designated
            ),
            (None, CURRENT_SOURCE_MISSING),
        )

    def test_baseline_equivalence_in_configuration_maps(self) -> None:
        plan = propose(
            state_view(""),
            store(
                {
                    BASELINE_SENTINEL: record(BASELINE_SENTINEL),
                    CANDIDATE_SIGNATURE: record(
                        CANDIDATE_SIGNATURE,
                        cells=CANDIDATE_CELLS,
                        milestone_cells=(MILESTONE_CELL,),
                    ),
                },
                root_signature=BASELINE_SENTINEL,
            ),
            (archive_candidate(),),
            (),
            (),
            RelationalPlannerConfig(),
        )
        establish = by_kind(plan, HypothesisKind.ESTABLISH_CONFIGURATION)
        self.assertFalse(configuration_maps("", establish))
        baseline_targets = [
            h
            for h in plan.hypotheses
            if h.target_is_designated_baseline
        ]
        for hypothesis in baseline_targets:
            self.assertTrue(configuration_maps("", hypothesis))


class HypothesisGenerationTests(unittest.TestCase):
    """Design test 1: generation from certified records and track state."""

    def test_removal_record_with_archive_candidate_yields_chain(
        self,
    ) -> None:
        plan = propose_default()
        kinds = [h.kind for h in plan.hypotheses]
        self.assertEqual(
            kinds,
            [
                HypothesisKind.ESTABLISH_CONFIGURATION,
                HypothesisKind.HOLD_CONFIGURATION,
                HypothesisKind.EXPLOIT_CONFIGURATION,
            ],
        )
        establish, hold, exploit = plan.hypotheses
        self.assertEqual(
            establish.realization.kind, REALIZATION_RESTORE_ARCHIVE
        )
        self.assertEqual(
            establish.realization.payload["configuration_signature"],
            CANDIDATE_SIGNATURE,
        )
        self.assertEqual(
            establish.realization.payload["state_id"], "state-A"
        )
        self.assertIsNone(establish.chain_parent_id)
        self.assertEqual(hold.chain_parent_id, establish.hypothesis_id)
        self.assertEqual(exploit.chain_parent_id, hold.hypothesis_id)
        self.assertEqual(plan.active_id, establish.hypothesis_id)
        self.assertEqual(
            exploit.realization.kind, REALIZATION_REACH_CELLS_UNDER_HOLD
        )
        self.assertEqual(
            tuple(exploit.realization.payload["target_cells"]),
            (MILESTONE_CELL,),
        )
        self.assertTrue(
            exploit.initiation.requires_uncollected_certified_milestone
        )

    def test_no_certified_record_fails_open_to_nothing(self) -> None:
        empty = propose_default(records=store({}))
        self.assertEqual(empty.hypotheses, ())
        self.assertIsNone(empty.active_id)
        predicted_only = propose_default(
            records=store(
                {
                    CANDIDATE_SIGNATURE: record(
                        CANDIDATE_SIGNATURE,
                        cells=CANDIDATE_CELLS,
                        verification=VERIFICATION_PREDICTED,
                    )
                }
            )
        )
        self.assertEqual(predicted_only.hypotheses, ())

    def test_no_realization_path_drops_the_establish_chain(self) -> None:
        plan = propose_default(archive=(), rules=())
        self.assertEqual(
            [h.kind for h in plan.hypotheses],
            [],
        )

    def test_rule_fallback_uses_reproduce_transition(self) -> None:
        matching_rule = TransitionRuleView(
            interaction_signature="interaction-a",
            transition_kind="removal",
            posterior=0.75,
            samples=4,
            inert_probability=0.1,
        )
        wrong_kind = TransitionRuleView(
            interaction_signature="interaction-b",
            transition_kind="displacement",
            posterior=0.9,
            samples=9,
            inert_probability=0.0,
        )
        plan = propose_default(
            archive=(), rules=(wrong_kind, matching_rule)
        )
        establish = by_kind(plan, HypothesisKind.ESTABLISH_CONFIGURATION)
        self.assertEqual(
            establish.realization.kind, REALIZATION_REPRODUCE_TRANSITION
        )
        self.assertEqual(
            establish.realization.payload["interaction_signature"],
            "interaction-a",
        )
        self.assertEqual(
            establish.realization.payload["expected_transition_kind"],
            "removal",
        )

    def test_current_configuration_proposes_hold_exploit_only(
        self,
    ) -> None:
        plan = propose_default(
            state=state_view(CANDIDATE_SIGNATURE),
        )
        kinds = [h.kind for h in plan.hypotheses]
        self.assertEqual(
            kinds,
            [
                HypothesisKind.HOLD_CONFIGURATION,
                HypothesisKind.EXPLOIT_CONFIGURATION,
            ],
        )
        hold, exploit = plan.hypotheses
        self.assertIsNone(hold.chain_parent_id)
        self.assertEqual(exploit.chain_parent_id, hold.hypothesis_id)
        self.assertEqual(plan.active_id, hold.hypothesis_id)

    def test_spent_milestones_fail_open_on_current_configuration(
        self,
    ) -> None:
        plan = propose_default(
            state=state_view(CANDIDATE_SIGNATURE, remaining=()),
        )
        self.assertEqual(plan.hypotheses, ())


class DeterminismTests(unittest.TestCase):
    """Design test 2: bounded queue, deterministic tie-breaking."""

    def test_identical_inputs_produce_byte_identical_plans(self) -> None:
        first = propose_default()
        second = propose_default()
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(
                [hypothesis_log_fields(h) for h in first.hypotheses],
                sort_keys=True,
            ),
            json.dumps(
                [hypothesis_log_fields(h) for h in second.hypotheses],
                sort_keys=True,
            ),
        )

    def test_queue_never_exceeds_max_queue(self) -> None:
        records = default_records()
        for index in range(4):
            signature = f"configuration-extra-{index}"
            records[signature] = record(
                signature,
                cells=CANDIDATE_CELLS + ((7 + index, 3),),
                milestone_cells=(MILESTONE_CELL,),
            )
        archive = tuple(
            archive_candidate(signature=s, state_id=f"state-{s}")
            for s in sorted(records)
        )
        config = RelationalPlannerConfig(max_queue=2)
        plan = propose_default(
            records=records, archive=archive, config=config
        )
        self.assertEqual(len(plan.hypotheses), 2)

    def test_chain_order_ranks_by_score_then_signature(self) -> None:
        weaker = "configuration-weaker"
        records = default_records()
        records[weaker] = record(
            weaker,
            cells=BASELINE_CELLS + ((2, 1),),
        )
        archive = (
            archive_candidate(),
            archive_candidate(signature=weaker, state_id="state-B"),
        )
        plan = propose_default(records=records, archive=archive)
        establishes = [
            h
            for h in plan.hypotheses
            if h.kind is HypothesisKind.ESTABLISH_CONFIGURATION
        ]
        self.assertEqual(
            establishes[0].target_configuration_signature,
            CANDIDATE_SIGNATURE,
        )


class ProvenanceRefusalTests(unittest.TestCase):
    """Design test 3: the predicted-provenance refusal, inherited."""

    def test_predicted_candidate_scores_exactly_zero_with_refusal(
        self,
    ) -> None:
        predicted = record(
            CANDIDATE_SIGNATURE,
            cells=CANDIDATE_CELLS,
            milestone_cells=(),
            verification=VERIFICATION_PREDICTED,
        )
        score = score_hypothesis_candidate(
            HypothesisKind.ESTABLISH_CONFIGURATION,
            predicted,
            record(CURRENT_SIGNATURE),
            state_view(),
            RelationalPlannerConfig(search_cost_per_branch=0.0),
            realization_kind=REALIZATION_RESTORE_ARCHIVE,
            branch_budget=0,
        )
        self.assertEqual(score.total, 0.0)
        self.assertEqual(score.verified_milestone_evidence, 0.0)
        self.assertEqual(score.expected_accessibility_improvement, 0.0)
        self.assertFalse(score.accessibility_scored)
        self.assertEqual(
            score.accessibility_refusal_reason,
            REFUSAL_CANDIDATE_NOT_CERTIFIED,
        )
        fields = score.log_fields()
        self.assertFalse(
            fields["relational_hypothesis_accessibility_scored"]
        )
        self.assertEqual(
            fields["relational_hypothesis_accessibility_refusal_reason"],
            REFUSAL_CANDIDATE_NOT_CERTIFIED,
        )

    def test_predicted_milestone_cells_never_score_as_evidence(
        self,
    ) -> None:
        # A predicted record carrying milestone cells must not convert
        # them into verified milestone evidence.
        predicted = CertifiedAccessibilityRecord(
            provenance=provenance(
                CANDIDATE_SIGNATURE, VERIFICATION_PREDICTED
            ),
            certified_cells=CANDIDATE_CELLS,
            certified_milestone_cells=(MILESTONE_CELL,),
        )
        score = score_hypothesis_candidate(
            HypothesisKind.ESTABLISH_CONFIGURATION,
            predicted,
            record(CURRENT_SIGNATURE),
            state_view(),
            RelationalPlannerConfig(search_cost_per_branch=0.0),
            realization_kind=REALIZATION_RESTORE_ARCHIVE,
            branch_budget=0,
        )
        self.assertEqual(score.verified_milestone_evidence, 0.0)
        self.assertEqual(score.total, 0.0)

    def test_missing_current_record_refuses_the_accessibility_term(
        self,
    ) -> None:
        score = score_hypothesis_candidate(
            HypothesisKind.ESTABLISH_CONFIGURATION,
            default_records()[CANDIDATE_SIGNATURE],
            None,
            state_view("configuration-unknown"),
            RelationalPlannerConfig(search_cost_per_branch=0.0),
            realization_kind=REALIZATION_RESTORE_ARCHIVE,
            branch_budget=0,
        )
        self.assertEqual(score.expected_accessibility_improvement, 0.0)
        self.assertFalse(score.accessibility_scored)
        self.assertEqual(
            score.accessibility_refusal_reason,
            REFUSAL_CURRENT_RECORD_MISSING,
        )
        # Certified milestone evidence remains scoreable: it is a record
        # property, not a comparison against the unknown configuration.
        self.assertEqual(score.verified_milestone_evidence, 1.0)


class InertDownRankingTests(unittest.TestCase):
    """Design test 4: known-inert strictly lowers the score, logged."""

    def test_inert_probability_is_subtractive_and_logged(self) -> None:
        def scored(inert: float):
            return score_hypothesis_candidate(
                HypothesisKind.ESTABLISH_CONFIGURATION,
                default_records()[CANDIDATE_SIGNATURE],
                default_records()[CURRENT_SIGNATURE],
                state_view(),
                RelationalPlannerConfig(),
                realization_kind=REALIZATION_REPRODUCE_TRANSITION,
                branch_budget=8,
                rule=TransitionRuleView(
                    interaction_signature="interaction-a",
                    transition_kind="removal",
                    posterior=0.5,
                    samples=4,
                    inert_probability=inert,
                    causal_hazard_probability=0.0,
                ),
            )

        clean = scored(0.0)
        inert = scored(0.4)
        self.assertLess(inert.total, clean.total)
        self.assertEqual(inert.predicted_inert_probability, 0.4)
        self.assertEqual(
            inert.log_fields()[
                "relational_hypothesis_predicted_inert_probability"
            ],
            0.4,
        )
        self.assertAlmostEqual(clean.total - inert.total, 0.4)


class ExactOutcomeOverrideTests(unittest.TestCase):
    """Design test 5: a contradicting verified transition forces replan."""

    def test_contradicted_restore_replans_never_continues(self) -> None:
        plan = propose_default()
        establish_id = plan.active_id
        result = advance(
            plan,
            summary(
                "configuration-elsewhere",
                kind="archive_restore",
                restored_state_id="state-Z",
            ),
        )
        self.assertEqual(result.outcome, ADVANCE_REPLAN)
        self.assertNotEqual(result.outcome, ADVANCE_CONTINUE)
        self.assertIn(
            (establish_id, TERMINATED_CONTRADICTED), result.terminated
        )
        self.assertIsNone(result.plan.active_id)
        # The dependent chain is terminated with it, reason-coded.
        reasons = dict(result.terminated)
        self.assertEqual(
            sorted(reasons.values()),
            sorted(
                [
                    TERMINATED_CONTRADICTED,
                    TERMINATED_REPLANNED,
                    TERMINATED_REPLANNED,
                ]
            ),
        )

    def test_advance_without_active_hypothesis_replans(self) -> None:
        plan = HypothesisPlan(hypotheses=(), active_id=None)
        result = advance(plan, summary(CURRENT_SIGNATURE))
        self.assertEqual(result.outcome, ADVANCE_REPLAN)


class ChainMechanicsTests(unittest.TestCase):
    """Design test 6: establish -> hold -> exploit chain semantics."""

    def test_full_chain_walk_to_milestone_achievement(self) -> None:
        plan = propose_default()
        establish, hold, exploit = plan.hypotheses

        realized = advance(
            plan,
            summary(
                CANDIDATE_SIGNATURE,
                kind="archive_restore",
                restored_state_id="state-A",
            ),
        )
        self.assertEqual(realized.outcome, ADVANCE_HYPOTHESIS_ACHIEVED)
        self.assertEqual(realized.achieved_id, establish.hypothesis_id)
        self.assertEqual(realized.realized_id, establish.hypothesis_id)
        self.assertEqual(realized.activated_id, hold.hypothesis_id)
        self.assertEqual(
            realized.plan.active_id, hold.hypothesis_id
        )
        self.assertIn(
            establish.hypothesis_id, realized.plan.achieved_ids
        )

        held = advance(realized.plan, summary(CANDIDATE_SIGNATURE))
        self.assertEqual(held.outcome, ADVANCE_HYPOTHESIS_ACHIEVED)
        self.assertEqual(held.achieved_id, hold.hypothesis_id)
        self.assertIsNone(held.realized_id)
        self.assertEqual(held.activated_id, exploit.hypothesis_id)

        collected = advance(
            held.plan,
            summary(CANDIDATE_SIGNATURE, remaining=()),
        )
        self.assertEqual(collected.outcome, ADVANCE_HYPOTHESIS_ACHIEVED)
        self.assertEqual(collected.achieved_id, exploit.hypothesis_id)
        self.assertEqual(collected.realized_id, exploit.hypothesis_id)
        self.assertEqual(collected.collected_cells, (MILESTONE_CELL,))
        self.assertIsNone(collected.plan.active_id)

    def test_exploit_achieves_when_a_target_cell_is_reached(self) -> None:
        plan = propose_default()
        realized = advance(
            plan,
            summary(
                CANDIDATE_SIGNATURE,
                kind="archive_restore",
                restored_state_id="state-A",
            ),
        )
        held = advance(realized.plan, summary(CANDIDATE_SIGNATURE))
        reached = advance(
            held.plan,
            summary(CANDIDATE_SIGNATURE, player_cell=MILESTONE_CELL),
        )
        self.assertEqual(reached.outcome, ADVANCE_HYPOTHESIS_ACHIEVED)

    def test_exploit_refuses_to_initiate_without_verified_parent(
        self,
    ) -> None:
        plan = propose_default()
        exploit = by_kind(plan, HypothesisKind.EXPLOIT_CONFIGURATION)
        self.assertFalse(
            initiation_satisfied(
                exploit,
                CANDIDATE_SIGNATURE,
                (MILESTONE_CELL,),
                achieved_ids=(),
            )
        )
        hold = by_kind(plan, HypothesisKind.HOLD_CONFIGURATION)
        self.assertTrue(
            initiation_satisfied(
                exploit,
                CANDIDATE_SIGNATURE,
                (MILESTONE_CELL,),
                achieved_ids=(hold.hypothesis_id,),
            )
        )

    def test_hold_violation_aborts_the_chain_with_reasons(self) -> None:
        plan = propose_default()
        hold = by_kind(plan, HypothesisKind.HOLD_CONFIGURATION)
        exploit = by_kind(plan, HypothesisKind.EXPLOIT_CONFIGURATION)
        realized = advance(
            plan,
            summary(
                CANDIDATE_SIGNATURE,
                kind="archive_restore",
                restored_state_id="state-A",
            ),
        )
        violated = advance(
            realized.plan, summary("configuration-elsewhere")
        )
        self.assertEqual(violated.outcome, ADVANCE_HOLD_VIOLATED)
        self.assertIn(
            (hold.hypothesis_id, TERMINATED_HOLD_VIOLATED),
            violated.terminated,
        )
        self.assertIn(
            (exploit.hypothesis_id, TERMINATED_REPLANNED),
            violated.terminated,
        )
        self.assertIsNone(violated.plan.active_id)

    def test_exploit_hold_violation_terminates_the_exploit(self) -> None:
        plan = propose_default()
        exploit = by_kind(plan, HypothesisKind.EXPLOIT_CONFIGURATION)
        realized = advance(
            plan,
            summary(
                CANDIDATE_SIGNATURE,
                kind="archive_restore",
                restored_state_id="state-A",
            ),
        )
        held = advance(realized.plan, summary(CANDIDATE_SIGNATURE))
        violated = advance(
            held.plan, summary("configuration-elsewhere")
        )
        self.assertEqual(violated.outcome, ADVANCE_HOLD_VIOLATED)
        self.assertIn(
            (exploit.hypothesis_id, TERMINATED_HOLD_VIOLATED),
            violated.terminated,
        )

    def test_budget_exhaustion_is_reported_and_terminates(self) -> None:
        config = RelationalPlannerConfig(decision_budget=2)
        plan = propose_default(config=config)
        establish_id = plan.active_id
        first = advance(plan, summary(CURRENT_SIGNATURE))
        self.assertEqual(first.outcome, ADVANCE_CONTINUE)
        second = advance(first.plan, summary(CURRENT_SIGNATURE))
        self.assertEqual(second.outcome, ADVANCE_BUDGET_EXHAUSTED)
        self.assertIn(
            (establish_id, TERMINATED_BUDGET_EXHAUSTED),
            second.terminated,
        )
        self.assertIsNone(second.plan.active_id)

    def test_hold_without_successor_achieves_after_budget(self) -> None:
        records = store(
            {
                CURRENT_SIGNATURE: record(CURRENT_SIGNATURE),
                CANDIDATE_SIGNATURE: record(
                    CANDIDATE_SIGNATURE, cells=CANDIDATE_CELLS
                ),
            }
        )
        config = RelationalPlannerConfig(decision_budget=2)
        plan = propose(
            state_view(),
            records,
            (archive_candidate(),),
            (),
            (),
            config,
        )
        kinds = [h.kind for h in plan.hypotheses]
        self.assertEqual(
            kinds,
            [
                HypothesisKind.ESTABLISH_CONFIGURATION,
                HypothesisKind.HOLD_CONFIGURATION,
                HypothesisKind.EXPLOIT_CONFIGURATION,
            ],
        )
        # Exploit here targets newly-reachable cells, not milestones.
        exploit = by_kind(plan, HypothesisKind.EXPLOIT_CONFIGURATION)
        self.assertFalse(
            exploit.initiation.requires_uncollected_certified_milestone
        )
        self.assertEqual(
            tuple(exploit.realization.payload["target_cells"]),
            tuple(sorted(EXTRA_CELLS)),
        )


class ScoreDecompositionTests(unittest.TestCase):
    """Design test 8: every component logged; totals equal sums."""

    def test_log_fields_expose_every_component_and_serialize(self) -> None:
        plan = propose_default(
            realized_options=(
                RealizedOption(
                    kind=HypothesisKind.ESTABLISH_CONFIGURATION.value,
                    target_configuration_signature=CANDIDATE_SIGNATURE,
                    record_content_signature="record-content",
                    initiation=InitiationCondition(
                        configuration_relation=RELATION_DIFFERS_FROM_RECORD,
                        required_record_signature="record-content",
                    ),
                    termination=TerminationCondition(
                        achieved_when=ACHIEVED_CONFIGURATION_MAPS,
                        violated_when="",
                        decision_budget=4,
                    ),
                    transfer_evidence_count=2,
                    attempt_count=5,
                ),
            )
        )
        establish = by_kind(plan, HypothesisKind.ESTABLISH_CONFIGURATION)
        score = establish.score
        fields = hypothesis_log_fields(establish)
        for key in (
            "relational_hypothesis_verified_milestone_evidence",
            "relational_hypothesis_expected_accessibility_improvement",
            "relational_hypothesis_information_gain",
            "relational_hypothesis_option_transfer_evidence",
            "relational_hypothesis_reversibility_confidence",
            "relational_hypothesis_causal_terminal_risk",
            "relational_hypothesis_predicted_inert_probability",
            "relational_hypothesis_search_cost",
            "relational_hypothesis_repeated_experiment_count",
            "relational_hypothesis_accessibility_scored",
            "relational_hypothesis_accessibility_refusal_reason",
            "relational_hypothesis_total_score",
        ):
            self.assertIn(key, fields)
        self.assertEqual(
            fields["relational_hypothesis_total_score"], score.total
        )
        self.assertEqual(
            score.total,
            score.verified_milestone_evidence
            + score.expected_accessibility_improvement
            + score.information_gain
            + score.option_transfer_evidence
            + score.reversibility_confidence
            - score.causal_terminal_risk
            - score.predicted_inert_probability
            - score.search_cost
            - score.repeated_experiment_count,
        )
        self.assertEqual(score.option_transfer_evidence, 2.0)
        self.assertEqual(score.repeated_experiment_count, 3.0)
        self.assertGreater(score.expected_accessibility_improvement, 0.0)
        self.assertEqual(score.verified_milestone_evidence, 1.0)
        json.dumps(fields)

    def test_accessibility_term_matches_the_pure_preference(self) -> None:
        records = default_records()
        establish = by_kind(
            propose_default(records=records),
            HypothesisKind.ESTABLISH_CONFIGURATION,
        )
        expected = verified_accessibility_preference(
            records[CANDIDATE_SIGNATURE], records[CURRENT_SIGNATURE]
        )
        self.assertEqual(
            establish.score.expected_accessibility_improvement,
            expected.total_bonus,
        )


class OptionStorageTests(unittest.TestCase):
    """Design test 7: relational, coordinate-free, macro-free options."""

    def _option(self) -> RealizedOption:
        plan = propose_default()
        exploit = by_kind(plan, HypothesisKind.EXPLOIT_CONFIGURATION)
        attempted = record_attempt(None, exploit)
        return record_success(attempted, exploit)

    def test_round_trip_serialization(self) -> None:
        option = self._option()
        payload = option.to_payload()
        restored = RealizedOption.from_payload(
            json.loads(json.dumps(payload))
        )
        self.assertEqual(option, restored)
        self.assertEqual(option.attempt_count, 1)
        self.assertEqual(option.transfer_evidence_count, 1)

    def test_persisted_payload_carries_no_coordinates_or_controls(
        self,
    ) -> None:
        payload = self._option().to_payload()
        text = json.dumps(payload)
        self.assertNotIn("target_cells", text)
        self.assertNotIn("path", text)
        self.assertNotIn("durations", text)
        self.assertNotIn("actions", text)
        # No bare coordinate pair appears anywhere in the payload.
        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, (list, tuple)):
                self.assertFalse(
                    len(value) == 2
                    and all(isinstance(v, int) for v in value),
                    f"coordinate-shaped value {value!r} in payload",
                )
                for item in value:
                    walk(item)

        walk(payload)

    def test_from_payload_refuses_smuggled_fields(self) -> None:
        payload = self._option().to_payload()
        payload["controls"] = [["RIGHT", 8]]
        with self.assertRaises(ValueError):
            RealizedOption.from_payload(payload)

    def test_translated_layout_still_matches_initiation(self) -> None:
        option = self._option()
        offset = (3, 2)
        translated_cells = tuple(
            (x + offset[0], y + offset[1]) for x, y in CANDIDATE_CELLS
        )
        translated_milestone = (
            MILESTONE_CELL[0] + offset[0],
            MILESTONE_CELL[1] + offset[1],
        )
        translated_records = store(
            {
                "configuration-translated": record(
                    "configuration-translated",
                    cells=translated_cells,
                    milestone_cells=(translated_milestone,),
                    outcome_category=OUTCOME_REMOVAL,
                )
            }
        )
        translated_state = state_view(
            "configuration-translated",
            remaining=(translated_milestone,),
        )
        self.assertTrue(
            option_initiation_satisfied(
                option, translated_state, translated_records
            )
        )
        # Once every certified milestone at the resolved record is
        # collected, the milestone-gated option no longer initiates.
        self.assertFalse(
            option_initiation_satisfied(
                option,
                state_view(
                    "configuration-translated", remaining=()
                ),
                translated_records,
            )
        )
        # No certified record: never initiates.
        self.assertFalse(
            option_initiation_satisfied(
                option, translated_state, store({})
            )
        )

    def test_option_key_is_stable(self) -> None:
        plan = propose_default()
        exploit = by_kind(plan, HypothesisKind.EXPLOIT_CONFIGURATION)
        self.assertEqual(
            option_key(exploit),
            f"exploit_configuration:{CANDIDATE_SIGNATURE}",
        )


class HypothesisIdentityTests(unittest.TestCase):
    def test_hypothesis_id_is_a_content_digest(self) -> None:
        first = propose_default()
        second = propose_default()
        for a, b in zip(first.hypotheses, second.hypotheses):
            self.assertEqual(a.hypothesis_id, b.hypothesis_id)
            self.assertEqual(len(a.hypothesis_id), 64)
            int(a.hypothesis_id, 16)
        # Content sensitivity: a different target configuration changes
        # the digest.
        other_records = store(
            {
                CURRENT_SIGNATURE: record(CURRENT_SIGNATURE),
                "configuration-other": record(
                    "configuration-other",
                    cells=CANDIDATE_CELLS,
                    milestone_cells=(MILESTONE_CELL,),
                ),
            }
        )
        other = propose_default(
            records=other_records,
            archive=(
                archive_candidate(signature="configuration-other"),
            ),
        )
        self.assertNotEqual(
            first.hypotheses[0].hypothesis_id,
            other.hypotheses[0].hypothesis_id,
        )


# ---------------------------------------------------------------------------
# WP8 navigation target — pure seam S3
# (docs/wp8-search-scheduling-design-2026-08-17.md section 4.2 mechanism
# (b), section 5.2 items 1-4). Everything here is pure over an already
# proposed hypothesis: no planner state, no configuration weight.
# ---------------------------------------------------------------------------


class NavigationObjectivePublicationTests(unittest.TestCase):
    """Section 5.2 item 1: only exploit objectives publish cells."""

    def test_only_the_exploit_objective_publishes_certified_cells(
        self,
    ) -> None:
        plan = propose_default()
        establish = by_kind(plan, HypothesisKind.ESTABLISH_CONFIGURATION)
        hold = by_kind(plan, HypothesisKind.HOLD_CONFIGURATION)
        exploit = by_kind(plan, HypothesisKind.EXPLOIT_CONFIGURATION)

        # Regression twin of _exploit_target_cells: the exploit's cells
        # are exactly the uncollected certified milestone cells.
        self.assertEqual(
            objective_target_cells(exploit), (MILESTONE_CELL,)
        )
        self.assertEqual(objective_target_cells(establish), ())
        self.assertEqual(objective_target_cells(hold), ())
        self.assertEqual(
            objective_hold_signature(exploit), CANDIDATE_SIGNATURE
        )
        self.assertEqual(
            objective_hold_signature(hold), CANDIDATE_SIGNATURE
        )
        # An establish objective names no hold configuration at all.
        self.assertEqual(objective_hold_signature(establish), "")

    def test_newly_reachable_cells_publish_when_no_milestone_remains(
        self,
    ) -> None:
        plan = propose_default(
            state=state_view(remaining=()),
        )
        exploit = by_kind(plan, HypothesisKind.EXPLOIT_CONFIGURATION)
        self.assertEqual(
            objective_target_cells(exploit), tuple(sorted(EXTRA_CELLS))
        )

    def test_publication_is_canonical_and_deterministic(self) -> None:
        plan = propose_default()
        exploit = by_kind(plan, HypothesisKind.EXPLOIT_CONFIGURATION)
        first = objective_target_cells(exploit)
        second = objective_target_cells(exploit)
        self.assertEqual(first, second)
        self.assertEqual(list(first), sorted(set(first)))


class NavigationPreferenceTests(unittest.TestCase):
    """Section 5.2 items 2-4: deterministic, fail-open, hold-gated."""

    def exploit(self):
        return by_kind(
            propose_default(), HypothesisKind.EXPLOIT_CONFIGURATION
        )

    def test_distance_is_grid_based_and_deterministic(self) -> None:
        exploit = self.exploit()
        target = MILESTONE_CELL
        near = (target[0] - 1, target[1])
        far = (target[0] - 4, target[1] - 2)
        self.assertEqual(target_cell_distance((target,), target), 0)
        self.assertEqual(target_cell_distance((target,), near), 1)
        self.assertEqual(target_cell_distance((target,), far), 6)
        # Identical inputs produce byte-identical ordering keys.
        keys = [navigation_preference(exploit, near) for _ in range(5)]
        self.assertEqual(len(set(keys)), 1)
        self.assertGreater(
            navigation_preference(exploit, near),
            navigation_preference(exploit, far),
        )

    def test_nearest_of_several_targets_decides(self) -> None:
        exploit = by_kind(
            propose_default(state=state_view(remaining=())),
            HypothesisKind.EXPLOIT_CONFIGURATION,
        )
        cells = objective_target_cells(exploit)
        self.assertGreater(len(cells), 1)
        probe = (cells[0][0], cells[0][1] + 1)
        self.assertEqual(target_cell_distance(cells, probe), 1)

    def test_fail_open_without_targets_or_without_a_cell(self) -> None:
        plan = propose_default()
        establish = by_kind(plan, HypothesisKind.ESTABLISH_CONFIGURATION)
        hold = by_kind(plan, HypothesisKind.HOLD_CONFIGURATION)
        exploit = by_kind(plan, HypothesisKind.EXPLOIT_CONFIGURATION)
        neutral = navigation_preference(exploit, None)
        # A non-reach realization and an empty target set are BOTH
        # indistinguishable from "no opinion": the consumer's incumbent
        # ordering decides.
        self.assertEqual(navigation_preference(establish, MILESTONE_CELL), neutral)
        self.assertEqual(navigation_preference(hold, MILESTONE_CELL), neutral)
        self.assertIsNone(target_cell_distance((), MILESTONE_CELL))
        self.assertIsNone(target_cell_distance((MILESTONE_CELL,), None))

    def test_hold_gating_outranks_distance(self) -> None:
        exploit = self.exploit()
        target = MILESTONE_CELL
        adjacent_but_departed = navigation_preference(
            exploit, (target[0] - 1, target[1]), "some-other-configuration"
        )
        far_but_held = navigation_preference(
            exploit,
            (target[0] - 4, target[1] - 2),
            CANDIDATE_SIGNATURE,
        )
        # A candidate outside the held configuration sorts strictly below
        # every candidate inside it, however much distance it closes
        # (learnings section 4.7's coupling requirement).
        self.assertLess(adjacent_but_departed, far_but_held)
        # None means "the caller enforces the hold predicate itself".
        self.assertEqual(
            navigation_preference(exploit, target, None)[0], 1
        )

    def test_preference_is_a_bounded_tie_break_not_a_reward(self) -> None:
        exploit = self.exploit()
        target = MILESTONE_CELL
        key = navigation_preference(exploit, target, CANDIDATE_SIGNATURE)
        # Section 4.7 guard: the key is an integer tuple ordered inside an
        # already-filtered candidate set. It carries no float weight, no
        # configuration knob, and cannot be summed into any score.
        self.assertEqual(len(key), 3)
        for component in key:
            self.assertIsInstance(component, int)
            self.assertNotIsInstance(component, bool)
        self.assertEqual(key, (1, 1, 0))


class ModuleBoundaryTests(unittest.TestCase):
    """Strict-lineage and no-room-literal checks (WP8 requirement)."""

    MODULE_PATH = (
        Path(__file__).resolve().parent.parent
        / "lolo_agent"
        / "relational_planner.py"
    )

    def test_strict_lineage_lint_is_clean(self) -> None:
        report = lint_strict_lineage([self.MODULE_PATH]).to_dict()
        self.assertFalse(report["assisted"])
        module_report = report["modules"][0]
        self.assertFalse(module_report["assisted"])
        self.assertEqual(module_report["findings"], [])

    def test_module_contains_no_coordinate_constants(self) -> None:
        tree = ast.parse(self.MODULE_PATH.read_text())
        offenders = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and all(
                isinstance(element, ast.Constant)
                and isinstance(element.value, int)
                for element in node.elts
            )
        ]
        self.assertEqual(
            offenders,
            [],
            "room-scoped coordinate constants are forbidden in the "
            "relational planner module",
        )

    def test_condition_dataclasses_carry_no_cell_fields(self) -> None:
        for dataclass_type in (InitiationCondition, TerminationCondition):
            for field in dataclass_fields(dataclass_type):
                self.assertNotIn("cell", field.name)
                self.assertNotIn("slot", field.name)
                self.assertNotIn("coordinate", field.name)


class NavigationSeamAblationContractTests(unittest.TestCase):
    """E5 (learnings section 4.50): the S2-only ablation is planner-side.

    E3 failed because seam S1 steered the commit ladder and narrowed the
    archive geography that later restores consume. E5 disables S1 and
    keeps S2. The seam selector lives entirely in the consuming planner:
    this module publishes the same key either way, so ``restore_only``
    cannot change WHAT is published, only WHO reads it.
    """

    def exploit(self):
        return by_kind(
            propose_default(), HypothesisKind.EXPLOIT_CONFIGURATION
        )

    def test_module_exposes_no_seam_selector(self) -> None:
        import lolo_agent.relational_planner as module

        source = Path(module.__file__).read_text()
        for token in (
            "navigation_seams",
            "restore_only",
            "commit_only",
        ):
            self.assertNotIn(token, source, token)

    def test_the_closing_key_is_what_S2_alone_needs(self) -> None:
        # The section 4.48 instant, as a pure-module contract: among
        # candidates INSIDE the held configuration the certified-adjacent
        # one must outrank the distant one, and any candidate outside the
        # held configuration must sort strictly below both — regardless
        # of how near it stands. This is the entire content of the E5
        # treatment; nothing about the commit ladder is involved.
        exploit = self.exploit()
        hold = objective_hold_signature(exploit)
        self.assertTrue(hold)
        target = objective_target_cells(exploit)[0]
        adjacent = (target[0] - 1, target[1])
        distant = (target[0] - 5, target[1] - 2)
        held_adjacent = navigation_preference(
            exploit, adjacent, configuration_signature=hold
        )
        held_distant = navigation_preference(
            exploit, distant, configuration_signature=hold
        )
        departed_on_target = navigation_preference(
            exploit, target, configuration_signature="some-other-sig"
        )
        self.assertGreater(held_adjacent, held_distant)
        self.assertLess(departed_on_target, held_distant)
        self.assertEqual(departed_on_target[0], 0)

    def test_the_key_is_stable_across_repeated_reads(self) -> None:
        # A restore-only intervention fires at most once per stagnation
        # instant, so a key that drifted between reads would be
        # unfalsifiable. Pin determinism at the module boundary.
        exploit = self.exploit()
        hold = objective_hold_signature(exploit)
        target = objective_target_cells(exploit)[0]
        keys = {
            navigation_preference(
                exploit, target, configuration_signature=hold
            )
            for _ in range(8)
        }
        self.assertEqual(len(keys), 1)


if __name__ == "__main__":
    unittest.main()
