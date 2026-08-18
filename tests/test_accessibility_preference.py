import json
import tempfile
import unittest

try:  # optional ML extra (imported transitively by the modules below)
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise unittest.SkipTest(
        "requires the optional 'ml' extra: pip install -e '.[ml]'"
    ) from exc
from pathlib import Path

from lolo_agent.accessibility_preference import (
    AccessibilityPreferenceConfig,
    AccessibilityRecordProvenance,
    CertifiedAccessibilityRecord,
    OUTCOME_DISPLACEMENT,
    OUTCOME_NONE,
    OUTCOME_REMOVAL,
    REFUSAL_CANDIDATE_NOT_CERTIFIED,
    REFUSAL_CURRENT_NOT_CERTIFIED,
    VERIFICATION_CERTIFIED_HOLD,
    VERIFICATION_PREDICTED,
    verified_accessibility_preference,
)
from lolo_agent.ensemble_world_model import EnsembleVisualDynamicsModel
from lolo_agent.environment import Action
from lolo_agent.goal_prior import HeartGoalAnalysis
from lolo_agent.neural_planner import (
    NeuralPlan,
    NeuralPlanningConfig,
    VerifiedAccessibilityRecordStore,
    VerifiedNeuralAgent,
    _ArchivedBranch,
    _HumanPriorOptionNode,
    load_verified_accessibility_records,
)
from lolo_agent.object_tracks import HumanPriorRootObjectState, ObjectTrackSet
from lolo_agent.pixels import Frame


# Synthetic fixtures shaped like the certified v322-v326 series: a 7-cell
# baseline envelope, and a removal-class candidate whose certified coverage
# adds the band and the eastern region including one milestone-bearing cell.
BASELINE_CELLS = (
    (6, 6),
    (6, 7),
    (6, 8),
    (6, 9),
    (6, 10),
    (7, 10),
    (8, 10),
)
REMOVED_CELLS = BASELINE_CELLS + (
    (7, 6),
    (8, 6),
    (8, 7),
    (8, 8),
    (9, 8),
    (10, 6),
    (10, 7),
    (10, 8),
    (11, 6),
    (11, 7),
    (11, 8),
    (12, 6),
    (12, 7),
    (12, 8),
    (12, 9),
    (12, 10),
    (12, 11),
)
MILESTONE_CELL = (12, 11)


def provenance(
    run_id: str = "fixture-run",
    verification: str = VERIFICATION_CERTIFIED_HOLD,
    configuration_signature: str = "config-a",
    certified_branches: int = 135,
    total_branches: int = 9691,
) -> AccessibilityRecordProvenance:
    return AccessibilityRecordProvenance(
        run_id=run_id,
        preregistration_doc="docs/fixture-preregistration.md",
        configuration_signature=configuration_signature,
        verification=verification,
        certification_predicate=(
            "anonymous_object_track_cells == []"
            if verification == VERIFICATION_CERTIFIED_HOLD
            else ""
        ),
        certified_branches=(
            certified_branches
            if verification == VERIFICATION_CERTIFIED_HOLD
            else 0
        ),
        total_branches=total_branches,
        search_depth=12,
        search_beam=128,
    )


def record(
    cells=BASELINE_CELLS,
    frontiers=(),
    milestone_cells=(),
    verification: str = VERIFICATION_CERTIFIED_HOLD,
    configuration_signature: str = "config-a",
    outcome_category: str = OUTCOME_NONE,
    manipulations: int = 0,
) -> CertifiedAccessibilityRecord:
    return CertifiedAccessibilityRecord(
        provenance=provenance(
            verification=verification,
            configuration_signature=configuration_signature,
        ),
        certified_cells=tuple(cells),
        certified_open_frontiers=tuple(frontiers),
        certified_milestone_cells=tuple(milestone_cells),
        preparation_outcome_category=outcome_category,
        confirmed_manipulation_count=manipulations,
    )


class ProvenanceValidationTests(unittest.TestCase):
    def test_rejects_empty_run_id(self) -> None:
        with self.assertRaises(ValueError):
            provenance(run_id="")

    def test_rejects_unknown_verification(self) -> None:
        with self.assertRaises(ValueError):
            AccessibilityRecordProvenance(
                run_id="run",
                preregistration_doc="doc",
                configuration_signature="sig",
                verification="observed",
                certification_predicate="p",
                certified_branches=1,
                total_branches=1,
                search_depth=12,
                search_beam=128,
            )

    def test_certified_hold_requires_predicate_and_branches(self) -> None:
        with self.assertRaises(ValueError):
            AccessibilityRecordProvenance(
                run_id="run",
                preregistration_doc="doc",
                configuration_signature="sig",
                verification=VERIFICATION_CERTIFIED_HOLD,
                certification_predicate="",
                certified_branches=1,
                total_branches=1,
                search_depth=12,
                search_beam=128,
            )
        with self.assertRaises(ValueError):
            AccessibilityRecordProvenance(
                run_id="run",
                preregistration_doc="doc",
                configuration_signature="sig",
                verification=VERIFICATION_CERTIFIED_HOLD,
                certification_predicate="p",
                certified_branches=0,
                total_branches=1,
                search_depth=12,
                search_beam=128,
            )

    def test_rejects_certified_above_total(self) -> None:
        with self.assertRaises(ValueError):
            AccessibilityRecordProvenance(
                run_id="run",
                preregistration_doc="doc",
                configuration_signature="sig",
                verification=VERIFICATION_CERTIFIED_HOLD,
                certification_predicate="p",
                certified_branches=10,
                total_branches=9,
                search_depth=12,
                search_beam=128,
            )


class RecordValidationTests(unittest.TestCase):
    def test_canonicalizes_cells_and_is_order_insensitive(self) -> None:
        shuffled = record(cells=((8, 10), (6, 6), (6, 7), (6, 6)))
        ordered = record(cells=((6, 6), (6, 7), (8, 10)))
        self.assertEqual(shuffled.certified_cells, ordered.certified_cells)
        self.assertEqual(
            shuffled.content_signature(), ordered.content_signature()
        )

    def test_signature_is_deterministic_and_content_sensitive(self) -> None:
        self.assertEqual(
            record().content_signature(), record().content_signature()
        )
        self.assertNotEqual(
            record().content_signature(),
            record(cells=REMOVED_CELLS).content_signature(),
        )

    def test_milestone_cells_must_be_certified(self) -> None:
        with self.assertRaises(ValueError):
            record(milestone_cells=(MILESTONE_CELL,))

    def test_frontier_source_must_be_certified(self) -> None:
        with self.assertRaises(ValueError):
            record(frontiers=(((0, 0), (0, 1)),))

    def test_frontier_must_join_distinct_cells(self) -> None:
        with self.assertRaises(ValueError):
            record(frontiers=(((6, 6), (6, 6)),))

    def test_rejects_unsanctioned_outcome_category(self) -> None:
        with self.assertRaises(ValueError):
            record(outcome_category="teleportation")

    def test_rejects_negative_manipulation_count(self) -> None:
        with self.assertRaises(ValueError):
            record(manipulations=-1)


class PreferenceScoringTests(unittest.TestCase):
    def test_identical_configurations_score_zero(self) -> None:
        components = verified_accessibility_preference(record(), record())
        self.assertTrue(components.scored)
        self.assertEqual(components.total_bonus, 0.0)
        self.assertEqual(components.newly_reachable_cells, ())
        self.assertEqual(components.newly_open_frontiers, ())
        self.assertEqual(components.newly_reachable_milestone_cells, ())

    def test_certified_new_cells_score_and_are_exposed(self) -> None:
        candidate = record(
            cells=REMOVED_CELLS,
            milestone_cells=(MILESTONE_CELL,),
            configuration_signature="config-removed",
            outcome_category=OUTCOME_REMOVAL,
        )
        components = verified_accessibility_preference(candidate, record())
        self.assertTrue(components.scored)
        self.assertEqual(len(components.newly_reachable_cells), 17)
        self.assertEqual(components.new_cell_bonus, 17.0)
        self.assertEqual(
            components.newly_reachable_milestone_cells, (MILESTONE_CELL,)
        )
        self.assertEqual(components.new_milestone_bonus, 8.0)
        self.assertEqual(components.total_bonus, 25.0)
        self.assertEqual(
            components.total_bonus,
            components.new_cell_bonus
            + components.new_frontier_bonus
            + components.new_milestone_bonus,
        )

    def test_fewer_certified_cells_is_censored_never_negative(self) -> None:
        candidate = record(
            cells=BASELINE_CELLS[:3],
            configuration_signature="config-partial",
        )
        components = verified_accessibility_preference(candidate, record())
        self.assertTrue(components.scored)
        self.assertEqual(components.total_bonus, 0.0)
        self.assertEqual(
            components.censored_current_only_cells,
            tuple(sorted(set(BASELINE_CELLS) - set(BASELINE_CELLS[:3]))),
        )
        self.assertGreaterEqual(components.new_cell_bonus, 0.0)
        self.assertGreaterEqual(components.new_frontier_bonus, 0.0)
        self.assertGreaterEqual(components.new_milestone_bonus, 0.0)

    def test_excluded_footprint_cells_carry_no_credit(self) -> None:
        candidate = record(
            cells=BASELINE_CELLS + ((7, 6), (8, 6)),
            configuration_signature="config-footprint",
        )
        components = verified_accessibility_preference(
            candidate, record(), excluded_cells={(7, 6), (8, 6)}
        )
        self.assertEqual(components.total_bonus, 0.0)
        self.assertEqual(components.excluded_cells, ((7, 6), (8, 6)))
        self.assertEqual(components.newly_reachable_cells, ())

    def test_known_milestone_cells_extend_milestone_credit(self) -> None:
        candidate = record(
            cells=REMOVED_CELLS,
            configuration_signature="config-removed",
        )
        components = verified_accessibility_preference(
            candidate, record(), known_milestone_cells={MILESTONE_CELL}
        )
        self.assertEqual(
            components.newly_reachable_milestone_cells, (MILESTONE_CELL,)
        )
        self.assertEqual(components.new_milestone_bonus, 8.0)

    def test_already_reachable_milestone_cell_earns_no_credit(self) -> None:
        candidate = record(
            cells=BASELINE_CELLS,
            milestone_cells=((8, 10),),
            configuration_signature="config-same",
        )
        components = verified_accessibility_preference(
            candidate, record(), known_milestone_cells={(8, 10)}
        )
        self.assertEqual(components.newly_reachable_milestone_cells, ())
        self.assertEqual(components.new_milestone_bonus, 0.0)

    def test_new_frontier_into_unreached_space_scores(self) -> None:
        candidate = record(
            frontiers=(((8, 10), (9, 10)),),
            configuration_signature="config-frontier",
        )
        components = verified_accessibility_preference(candidate, record())
        self.assertEqual(
            components.newly_open_frontiers, (((8, 10), (9, 10)),)
        )
        self.assertEqual(components.new_frontier_bonus, 1.0)
        self.assertEqual(components.total_bonus, 1.0)

    def test_frontier_already_known_to_current_earns_nothing(self) -> None:
        edge = ((8, 10), (9, 10))
        candidate = record(
            frontiers=(edge,), configuration_signature="config-frontier"
        )
        current = record(frontiers=(edge,))
        components = verified_accessibility_preference(candidate, current)
        self.assertEqual(components.newly_open_frontiers, ())
        self.assertEqual(components.total_bonus, 0.0)

    def test_frontier_into_reachable_cell_is_churn_excluded(self) -> None:
        candidate = record(
            frontiers=(((6, 6), (6, 7)),),
            configuration_signature="config-churn-edge",
        )
        components = verified_accessibility_preference(candidate, record())
        self.assertEqual(components.newly_open_frontiers, ())
        self.assertEqual(
            components.churn_excluded_frontiers, (((6, 6), (6, 7)),)
        )
        self.assertEqual(components.total_bonus, 0.0)

    def test_frontier_target_reached_by_candidate_counts_as_cell(self) -> None:
        candidate = record(
            cells=BASELINE_CELLS + ((9, 10),),
            frontiers=(((8, 10), (9, 10)),),
            configuration_signature="config-cell-and-edge",
        )
        components = verified_accessibility_preference(candidate, record())
        self.assertEqual(components.newly_reachable_cells, ((9, 10),))
        self.assertEqual(components.newly_open_frontiers, ())
        self.assertEqual(
            components.churn_excluded_frontiers, (((8, 10), (9, 10)),)
        )
        self.assertEqual(components.total_bonus, 1.0)

    def test_custom_weights_apply(self) -> None:
        candidate = record(
            cells=REMOVED_CELLS,
            milestone_cells=(MILESTONE_CELL,),
            configuration_signature="config-removed",
        )
        config = AccessibilityPreferenceConfig(
            new_cell_weight=0.5,
            new_frontier_weight=0.0,
            new_milestone_weight=2.0,
        )
        components = verified_accessibility_preference(
            candidate, record(), config=config
        )
        self.assertEqual(components.new_cell_bonus, 8.5)
        self.assertEqual(components.new_milestone_bonus, 2.0)
        self.assertEqual(components.total_bonus, 10.5)

    def test_rejects_negative_weights(self) -> None:
        with self.assertRaises(ValueError):
            AccessibilityPreferenceConfig(new_cell_weight=-1.0)


class UnverifiedRefusalTests(unittest.TestCase):
    def test_predicted_candidate_never_scores(self) -> None:
        candidate = record(
            cells=REMOVED_CELLS,
            verification=VERIFICATION_PREDICTED,
            configuration_signature="config-predicted",
        )
        components = verified_accessibility_preference(candidate, record())
        self.assertFalse(components.scored)
        self.assertEqual(
            components.refusal_reason, REFUSAL_CANDIDATE_NOT_CERTIFIED
        )
        self.assertEqual(components.total_bonus, 0.0)
        self.assertEqual(components.new_cell_bonus, 0.0)
        self.assertEqual(components.new_frontier_bonus, 0.0)
        self.assertEqual(components.new_milestone_bonus, 0.0)
        self.assertEqual(components.newly_reachable_cells, ())

    def test_predicted_baseline_never_scores(self) -> None:
        current = record(
            verification=VERIFICATION_PREDICTED,
            configuration_signature="config-predicted",
        )
        candidate = record(
            cells=REMOVED_CELLS, configuration_signature="config-removed"
        )
        components = verified_accessibility_preference(candidate, current)
        self.assertFalse(components.scored)
        self.assertEqual(
            components.refusal_reason, REFUSAL_CURRENT_NOT_CERTIFIED
        )
        self.assertEqual(components.total_bonus, 0.0)


class ChurnGamingTests(unittest.TestCase):
    """A config that mints affordances by moving objects around must never
    outscore a config with certified new reachable cells."""

    def test_affordance_churn_scores_zero(self) -> None:
        churn = record(
            cells=BASELINE_CELLS,
            frontiers=(
                ((6, 7), (5, 7)),
                ((6, 8), (5, 8)),
                ((6, 6), (6, 7)),
                ((6, 9), (6, 10)),
            ),
            configuration_signature="config-churn",
            outcome_category=OUTCOME_DISPLACEMENT,
            manipulations=9,
        )
        current = record(
            frontiers=(((6, 7), (5, 7)), ((6, 8), (5, 8))),
        )
        components = verified_accessibility_preference(churn, current)
        self.assertTrue(components.scored)
        # The two frontier edges into already-reachable cells are exposed
        # as churn and score nothing; the manipulation count is exposed
        # and scores nothing.
        self.assertEqual(components.total_bonus, 0.0)
        self.assertEqual(
            components.churn_excluded_frontiers,
            (((6, 6), (6, 7)), ((6, 9), (6, 10))),
        )
        self.assertEqual(
            components.candidate_confirmed_manipulation_count, 9
        )

    def test_churn_never_outscores_certified_new_cells(self) -> None:
        current = record()
        churn = record(
            cells=BASELINE_CELLS,
            frontiers=(
                ((6, 6), (6, 7)),
                ((6, 7), (6, 8)),
                ((6, 8), (6, 9)),
                ((6, 9), (6, 10)),
                ((6, 10), (7, 10)),
                ((7, 10), (8, 10)),
                ((8, 10), (6, 10)),
            ),
            configuration_signature="config-churn",
            outcome_category=OUTCOME_DISPLACEMENT,
            manipulations=25,
        )
        removal = record(
            cells=BASELINE_CELLS + ((8, 7), (8, 8), (9, 8)),
            configuration_signature="config-removed",
            outcome_category=OUTCOME_REMOVAL,
            manipulations=1,
        )
        churn_components = verified_accessibility_preference(churn, current)
        removal_components = verified_accessibility_preference(
            removal, current
        )
        self.assertEqual(churn_components.total_bonus, 0.0)
        self.assertEqual(removal_components.total_bonus, 3.0)
        self.assertGreater(
            removal_components.total_bonus, churn_components.total_bonus
        )


class LogFieldsTests(unittest.TestCase):
    def test_log_fields_expose_every_component_and_serialize(self) -> None:
        candidate = record(
            cells=REMOVED_CELLS,
            milestone_cells=(MILESTONE_CELL,),
            configuration_signature="config-removed",
            outcome_category=OUTCOME_REMOVAL,
            manipulations=1,
        )
        components = verified_accessibility_preference(candidate, record())
        fields = components.log_fields()
        for key in (
            "verified_accessibility_candidate_signature",
            "verified_accessibility_current_signature",
            "verified_accessibility_scored",
            "verified_accessibility_refusal_reason",
            "verified_accessibility_new_cells",
            "verified_accessibility_new_cell_count",
            "verified_accessibility_new_cell_bonus",
            "verified_accessibility_new_frontiers",
            "verified_accessibility_new_frontier_count",
            "verified_accessibility_new_frontier_bonus",
            "verified_accessibility_new_milestone_cells",
            "verified_accessibility_new_milestone_count",
            "verified_accessibility_new_milestone_bonus",
            "verified_accessibility_censored_current_only_cells",
            "verified_accessibility_excluded_cells",
            "verified_accessibility_churn_excluded_frontiers",
            "verified_accessibility_confirmed_manipulation_count",
            "verified_accessibility_outcome_category",
            "verified_accessibility_total_bonus",
        ):
            self.assertIn(key, fields)
        self.assertEqual(fields["verified_accessibility_new_cell_count"], 17)
        self.assertEqual(
            fields["verified_accessibility_total_bonus"],
            components.total_bonus,
        )
        json.dumps(fields)


# ---------------------------------------------------------------------------
# Planner-seam integration (WP8-lite seam patch,
# docs/wp8-lite-ablation-design-2026-08-16.md section 4): the preference term
# wired into the archive/restore-selection seams behind
# NeuralPlanningConfig.verified_accessibility_weight (default 0.0).
# ---------------------------------------------------------------------------


class _RecordingLogger:
    def __init__(self) -> None:
        self.events = []

    def log(self, event_type: str, **fields) -> None:
        self.events.append({"event": event_type, **fields})


class _SeamEnv:
    """Deterministic position-coded frames with handle-tracked save states."""

    def __init__(self) -> None:
        self.position = 0
        self.serial = 0
        self.active_states = set()

    def reset(self) -> Frame:
        self.position = 0
        self.active_states = set()
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if action == Action.RIGHT:
            self.position = min(63, self.position + frames)
        return self._frame()

    def save_state(self):
        self.serial += 1
        state = (self.serial, self.position)
        self.active_states.add(state)
        return state

    def load_state(self, state) -> Frame:
        if state not in self.active_states:
            raise RuntimeError("unknown save-state handle")
        self.position = state[1]
        return self._frame()

    def release_state(self, state) -> None:
        self.active_states.remove(state)

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[self.position] = 255
        return Frame(8, 8, 1, bytes(pixels))


class _ExplodingRecordStore(dict):
    """Fails the test if the seam consults the store at weight 0.0."""

    def get(self, key, default=None):
        raise AssertionError(
            "verified-accessibility record store consulted at weight 0.0"
        )


CURRENT_SIGNATURE = "current-sig"
NEUTRAL_SIGNATURE = "neutral-sig"
REMOVAL_SIGNATURE = "removal-sig"


def _seam_records(removal_verification: str = VERIFICATION_CERTIFIED_HOLD):
    return {
        CURRENT_SIGNATURE: record(
            configuration_signature=CURRENT_SIGNATURE
        ),
        NEUTRAL_SIGNATURE: record(
            configuration_signature=NEUTRAL_SIGNATURE
        ),
        REMOVAL_SIGNATURE: record(
            cells=REMOVED_CELLS,
            milestone_cells=(MILESTONE_CELL,),
            configuration_signature=REMOVAL_SIGNATURE,
            outcome_category=OUTCOME_REMOVAL,
            verification=removal_verification,
        ),
    }


def _seam_agent(weight: float, records):
    """Agent plus a two-branch archive facing the ablation's real choice.

    The archived branches are byte-identical except for their tracked
    world-state signature and their stored score: the certified-neutral
    branch carries the higher plain score, so weight 0.0 must restore it
    and only a scored preference term can flip selection to the certified
    removal-class branch.
    """

    env = _SeamEnv()
    logger = _RecordingLogger()
    agent = VerifiedNeuralAgent(
        env,
        EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        ),
        "cpu",
        NeuralPlanningConfig(
            actions=(Action.LEFT, Action.RIGHT),
            planning_depth=1,
            verified_accessibility_weight=weight,
        ),
        event_logger=logger,
    )
    agent.reset()
    agent.current_human_prior_root_object_state = HumanPriorRootObjectState(
        tracked_world_state_signature=CURRENT_SIGNATURE
    )
    if records is not None:
        agent.verified_accessibility_records = records
    env.position = 1
    frame = env._frame()

    def branch(signature: str, score: float) -> _ArchivedBranch:
        return _ArchivedBranch(
            state=env.save_state(),
            frame=frame,
            plan=NeuralPlan((Action.RIGHT,), (1,), score, 0.0),
            score=score,
            scene="archived-elsewhere",
            created=0,
            origin_signature="origin",
            tracked_world_state_signature=signature,
        )

    neutral = branch(NEUTRAL_SIGNATURE, 5.0)
    removal = branch(REMOVAL_SIGNATURE, 4.0)
    agent.archive = [neutral, removal]
    agent.visual_stagnation_streak = 99
    agent.autonomous_grace_remaining = 0
    return env, logger, agent, neutral, removal


def _restored_event(logger: _RecordingLogger):
    return next(
        event
        for event in logger.events
        if event["event"] == "archive_branch_restored"
    )


def _committed_event(logger: _RecordingLogger):
    return next(
        event
        for event in logger.events
        if event["event"] == "decision_committed"
    )


def _reserve_node(
    signature: str,
    effect_cells,
    player_slot,
) -> _HumanPriorOptionNode:
    analysis = HeartGoalAnalysis(
        reliable=True,
        known_slots=((7, 0),),
        source_present=((7, 0),),
        target_present=((7, 0),),
        collected=(),
        target_similarities=(),
        heart_reward=0.0,
        all_hearts_reward=0.0,
        chest_reward=0.0,
        navigation_reward=0.0,
        life_loss_penalty=0.0,
        total_reward=0.0,
        global_visual_change=0.0,
        target_intensity=1.0,
        source_player_slot=(0, 0),
        target_player_slot=player_slot,
        source_heart_distance=None,
        target_heart_distance=None,
        source_chest_slot=None,
        target_chest_slot=None,
        source_chest_distance=None,
        target_chest_distance=None,
        chest_completed=False,
        source_life_signature="life",
        target_life_signature="life",
        life_counter_changed=False,
        dark_transition_started=False,
        life_loss_confirmed=False,
    )
    return _HumanPriorOptionNode(
        state=None,
        frame=Frame(8, 8, 1, bytes(64)),
        path=(Action.RIGHT,),
        durations=(1,),
        analysis=analysis,
        source_signature="source",
        target_signature=f"target-{signature}",
        score=1.0,
        depth=1,
        target_state_visits=0,
        target_position_visits=0,
        tracked_world_effect_cells=tuple(effect_cells),
        tracked_world_state_signature=signature,
    )


class PlannerSeamWeightZeroInvarianceTests(unittest.TestCase):
    """Control-arm invariance: weight 0.0 is byte-identical ranking.

    The primary net is the full existing suite passing unchanged on the
    patched build (design section 4.6); these tests additionally prove the
    seam-local claims directly.
    """

    def test_default_config_weight_is_zero(self) -> None:
        self.assertEqual(
            NeuralPlanningConfig().verified_accessibility_weight, 0.0
        )

    def test_zero_weight_never_consults_the_record_store(self) -> None:
        _env, _logger, agent, neutral, removal = _seam_agent(
            0.0, _ExplodingRecordStore()
        )
        for branch in (neutral, removal):
            self.assertEqual(
                agent._archive_verified_accessibility_bonus(branch),
                (0.0, None),
            )
        self.assertEqual(
            agent._verified_accessibility_reserve_rank(REMOVAL_SIGNATURE),
            0.0,
        )

    def test_zero_weight_frontier_scores_are_identical_with_records(
        self,
    ) -> None:
        _env, _logger, loaded, ln, lr = _seam_agent(0.0, _seam_records())
        _env2, _logger2, bare, bn, br = _seam_agent(0.0, None)
        self.assertEqual(
            loaded._archive_frontier_score(ln),
            bare._archive_frontier_score(bn),
        )
        self.assertEqual(
            loaded._archive_frontier_score(lr),
            bare._archive_frontier_score(br),
        )

    def test_zero_weight_restore_selection_and_telemetry_identical(
        self,
    ) -> None:
        env_a, logger_a, loaded, *_rest = _seam_agent(0.0, _seam_records())
        env_b, logger_b, bare, *_rest2 = _seam_agent(0.0, None)

        decision_a = loaded._restore_if_stagnant()
        decision_b = bare._restore_if_stagnant()

        self.assertIsNotNone(decision_a)
        self.assertIsNotNone(decision_b)
        assert decision_a is not None and decision_b is not None
        self.assertEqual(decision_a.action, decision_b.action)
        self.assertEqual(
            decision_a.action_frames, decision_b.action_frames
        )
        self.assertEqual(decision_a.score, decision_b.score)
        self.assertEqual(env_a.position, env_b.position)
        # The higher-scored certified-neutral branch wins the tiebreak in
        # both agents: the loaded records changed nothing at weight 0.0.
        self.assertEqual(
            [b.tracked_world_state_signature for b in loaded.archive],
            [REMOVAL_SIGNATURE],
        )
        self.assertEqual(
            [b.tracked_world_state_signature for b in bare.archive],
            [REMOVAL_SIGNATURE],
        )
        restored_a = _restored_event(logger_a)
        restored_b = _restored_event(logger_b)
        self.assertEqual(restored_a, restored_b)
        self.assertEqual(
            _committed_event(logger_a), _committed_event(logger_b)
        )
        self.assertEqual(
            [event["event"] for event in logger_a.events],
            [event["event"] for event in logger_b.events],
        )
        # The unscored term is logged as unscored, never silently absent.
        self.assertFalse(restored_a["verified_accessibility_scored"])
        self.assertEqual(
            restored_a["verified_accessibility_refusal_reason"],
            "record_missing_or_disabled",
        )
        self.assertEqual(restored_a["verified_accessibility_bonus"], 0.0)
        self.assertEqual(
            restored_a["verified_accessibility_total_bonus"], 0.0
        )

    def test_zero_weight_reserve_ranking_unchanged(self) -> None:
        _env, _logger, agent, *_rest = _seam_agent(0.0, _seam_records())
        nodes = [
            _reserve_node(REMOVAL_SIGNATURE, ((2, 2),), (1, 0)),
            _reserve_node(NEUTRAL_SIGNATURE, ((3, 3),), (2, 0)),
        ]
        plain = VerifiedNeuralAgent._human_prior_world_state_reserve_candidates(
            nodes
        )
        ranked = VerifiedNeuralAgent._human_prior_world_state_reserve_candidates(
            nodes,
            verified_accessibility_rank=(
                agent._verified_accessibility_reserve_rank
            ),
        )
        self.assertEqual(
            [node.tracked_world_state_signature for node in plain],
            [node.tracked_world_state_signature for node in ranked],
        )


class PlannerSeamTreatmentTests(unittest.TestCase):
    """Treatment arm: a scored term selects deliberately and logs fully."""

    def test_restore_prefers_certified_removal_configuration(self) -> None:
        env, logger, agent, neutral, removal = _seam_agent(
            1.0, _seam_records()
        )
        expected_components = verified_accessibility_preference(
            agent.verified_accessibility_records[REMOVAL_SIGNATURE],
            agent.verified_accessibility_records[CURRENT_SIGNATURE],
        )
        self.assertGreater(expected_components.total_bonus, 0.0)
        # Captured before the restore commits: restoring rebinds the
        # current root object state, which changes later scores.
        expected_frontier_value = agent._archive_frontier_score(removal)

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertTrue(decision.restored_archive)
        # The certified removal-class branch was selected despite its lower
        # plain score: the preference term ranked it, for the hardened
        # reason (certified new cells and a certified milestone cell).
        self.assertEqual(
            [b.tracked_world_state_signature for b in agent.archive],
            [NEUTRAL_SIGNATURE],
        )
        restored = _restored_event(logger)
        self.assertTrue(restored["verified_accessibility_scored"])
        self.assertIsNone(
            restored["verified_accessibility_refusal_reason"]
        )
        self.assertEqual(
            restored["verified_accessibility_total_bonus"],
            expected_components.total_bonus,
        )
        self.assertEqual(
            restored["verified_accessibility_bonus"],
            1.0 * expected_components.total_bonus,
        )
        self.assertEqual(
            restored["persistent_frontier_value"],
            expected_frontier_value,
        )
        # Every component of the module's decomposition is logged.
        for key, value in expected_components.log_fields().items():
            self.assertIn(key, restored)
            self.assertEqual(restored[key], value)
        committed = _committed_event(logger)
        self.assertEqual(
            committed["verified_accessibility_total_bonus"],
            expected_components.total_bonus,
        )
        self.assertEqual(
            committed["verified_accessibility_bonus"],
            expected_components.total_bonus,
        )

    def test_seam_bonus_scales_with_the_config_weight(self) -> None:
        _env, logger, agent, _neutral, removal = _seam_agent(
            2.0, _seam_records()
        )
        expected_components = verified_accessibility_preference(
            agent.verified_accessibility_records[REMOVAL_SIGNATURE],
            agent.verified_accessibility_records[CURRENT_SIGNATURE],
        )

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        restored = _restored_event(logger)
        self.assertEqual(
            restored["verified_accessibility_bonus"],
            2.0 * expected_components.total_bonus,
        )
        self.assertEqual(
            restored["verified_accessibility_total_bonus"],
            expected_components.total_bonus,
        )

    def test_missing_current_record_scores_zero_and_is_logged_unscored(
        self,
    ) -> None:
        records = _seam_records()
        del records[CURRENT_SIGNATURE]
        _env, logger, agent, _neutral, _removal = _seam_agent(1.0, records)

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        # Without a certified record for the current configuration the term
        # refuses to score and the plain tiebreak restores the
        # higher-scored neutral branch.
        self.assertEqual(
            [b.tracked_world_state_signature for b in agent.archive],
            [REMOVAL_SIGNATURE],
        )
        restored = _restored_event(logger)
        self.assertFalse(restored["verified_accessibility_scored"])
        self.assertEqual(
            restored["verified_accessibility_refusal_reason"],
            "record_missing_or_disabled",
        )

    def test_predicted_candidate_record_never_scores_at_the_seam(
        self,
    ) -> None:
        _env, logger, agent, _neutral, removal = _seam_agent(
            1.0, _seam_records(removal_verification=VERIFICATION_PREDICTED)
        )
        # At the seam the predicted removal-class record scores exactly
        # zero and exposes the refusal (WP8 scoring rule).
        bonus, components = agent._archive_verified_accessibility_bonus(
            removal
        )
        self.assertEqual(bonus, 0.0)
        assert components is not None
        self.assertFalse(components.scored)
        self.assertEqual(
            components.refusal_reason, REFUSAL_CANDIDATE_NOT_CERTIFIED
        )
        self.assertEqual(components.total_bonus, 0.0)

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        # With the removal record refused, the neutral branch keeps
        # winning on plain score: unverified predicted accessibility can
        # never flip restore selection.
        self.assertEqual(
            [b.tracked_world_state_signature for b in agent.archive],
            [REMOVAL_SIGNATURE],
        )
        restored = _restored_event(logger)
        # The restored neutral branch scored validly at zero (identical
        # certified envelopes), and its decomposition says so.
        self.assertTrue(restored["verified_accessibility_scored"])
        self.assertEqual(
            restored["verified_accessibility_total_bonus"], 0.0
        )
        self.assertEqual(restored["verified_accessibility_bonus"], 0.0)

    def test_reserve_ranking_prefers_certified_configuration(self) -> None:
        _env, _logger, agent, *_rest = _seam_agent(1.0, _seam_records())
        # The neutral configuration carries strictly better reachability
        # topology (observed from three player positions spanning both
        # axes); only the certified preference term can put the
        # removal-class configuration first.
        removal_node = _reserve_node(REMOVAL_SIGNATURE, ((2, 2),), (1, 0))
        neutral_nodes = [
            _reserve_node(NEUTRAL_SIGNATURE, ((3, 3),), player_slot)
            for player_slot in ((2, 0), (3, 0), (2, 1))
        ]
        nodes = [removal_node, *neutral_nodes]
        plain = VerifiedNeuralAgent._human_prior_world_state_reserve_candidates(
            nodes
        )
        self.assertEqual(
            plain[0].tracked_world_state_signature, NEUTRAL_SIGNATURE
        )
        ranked = VerifiedNeuralAgent._human_prior_world_state_reserve_candidates(
            nodes,
            verified_accessibility_rank=(
                agent._verified_accessibility_reserve_rank
            ),
        )
        self.assertEqual(
            ranked[0].tracked_world_state_signature, REMOVAL_SIGNATURE
        )


class RecordLoaderTests(unittest.TestCase):
    """Minimal provenance-checked JSON import path (design section 4.7)."""

    @staticmethod
    def _entry(
        signature: str = REMOVAL_SIGNATURE,
        verification: str = VERIFICATION_CERTIFIED_HOLD,
    ):
        return {
            "provenance": {
                "run_id": "entity-v325-room3-object-removed-probe-d12",
                "preregistration_doc": (
                    "docs/object-removed-probe-2026-08-16.md"
                ),
                "configuration_signature": signature,
                "verification": verification,
                "certification_predicate": (
                    "anonymous_object_track_cells == []"
                    if verification == VERIFICATION_CERTIFIED_HOLD
                    else ""
                ),
                "certified_branches": (
                    135
                    if verification == VERIFICATION_CERTIFIED_HOLD
                    else 0
                ),
                "total_branches": 9691,
                "search_depth": 12,
                "search_beam": 128,
            },
            "certified_cells": [list(cell) for cell in REMOVED_CELLS],
            "certified_open_frontiers": [
                [[12, 11], [13, 11]],
            ],
            "certified_milestone_cells": [list(MILESTONE_CELL)],
            "preparation_outcome_category": OUTCOME_REMOVAL,
            "confirmed_manipulation_count": 2,
        }

    def _load(self, payload):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "records.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_verified_accessibility_records(str(path))

    def test_round_trip_keyed_by_configuration_signature(self) -> None:
        records = self._load(
            [
                self._entry(),
                self._entry(signature=NEUTRAL_SIGNATURE),
            ]
        )
        self.assertEqual(
            sorted(records), [NEUTRAL_SIGNATURE, REMOVAL_SIGNATURE]
        )
        loaded = records[REMOVAL_SIGNATURE]
        self.assertEqual(loaded.certified_cells, tuple(sorted(REMOVED_CELLS)))
        self.assertEqual(
            loaded.certified_open_frontiers, (((12, 11), (13, 11)),)
        )
        self.assertEqual(loaded.certified_milestone_cells, (MILESTONE_CELL,))
        self.assertEqual(
            loaded.preparation_outcome_category, OUTCOME_REMOVAL
        )
        self.assertEqual(loaded.confirmed_manipulation_count, 2)
        self.assertTrue(loaded.provenance.certified)
        # The loaded record is scoreable against a certified baseline.
        components = verified_accessibility_preference(
            loaded, record(configuration_signature=CURRENT_SIGNATURE)
        )
        self.assertTrue(components.scored)
        self.assertGreater(components.total_bonus, 0.0)

    def test_refuses_predicted_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "certified_hold"):
            self._load([self._entry(verification=VERIFICATION_PREDICTED)])

    def test_refuses_duplicate_configuration_signatures(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._load([self._entry(), self._entry()])

    def test_refuses_non_list_payloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON list"):
            self._load({"records": []})

    def test_agent_rejects_negative_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "verified accessibility"):
            VerifiedNeuralAgent(
                _SeamEnv(),
                EnsembleVisualDynamicsModel(
                    latent_size=32, action_size=8, ensemble_size=2
                ),
                "cpu",
                NeuralPlanningConfig(verified_accessibility_weight=-0.5),
            )

    def test_agent_rejects_non_finite_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "verified accessibility"):
            VerifiedNeuralAgent(
                _SeamEnv(),
                EnsembleVisualDynamicsModel(
                    latent_size=32, action_size=8, ensemble_size=2
                ),
                "cpu",
                NeuralPlanningConfig(
                    verified_accessibility_weight=float("nan")
                ),
            )


# ---------------------------------------------------------------------------
# Root/current baseline designation (design sections 6.5/6.8): the pre-push
# ablation root's tracked world-state signature is the empty string — the one
# value the store structurally cannot key — so the records file designates
# exactly one record as the root/current baseline and the seam's current-side
# resolution falls back to it only at an empty-signature root. Refusal
# semantics for genuinely unknown (non-empty, unmapped) configurations are
# preserved, and weight 0.0 still consults nothing.
# ---------------------------------------------------------------------------


BASELINE_SENTINEL_SIGNATURE = "prepush-root-sentinel"


class _ExplodingDesignatedStore(VerifiedAccessibilityRecordStore):
    """Fails the test if the seam consults a designated store at 0.0."""

    def get(self, key, default=None):
        raise AssertionError(
            "designated record store consulted at weight 0.0"
        )


def _root_designated_records():
    """The shipped store's shape: baseline (designated, sentinel-keyed),
    certified-neutral, and removal-class records."""

    store = VerifiedAccessibilityRecordStore()
    store[BASELINE_SENTINEL_SIGNATURE] = record(
        configuration_signature=BASELINE_SENTINEL_SIGNATURE
    )
    store[NEUTRAL_SIGNATURE] = record(
        configuration_signature=NEUTRAL_SIGNATURE
    )
    store[REMOVAL_SIGNATURE] = record(
        cells=REMOVED_CELLS,
        milestone_cells=(MILESTONE_CELL,),
        configuration_signature=REMOVAL_SIGNATURE,
        outcome_category=OUTCOME_REMOVAL,
    )
    store.root_configuration_signature = BASELINE_SENTINEL_SIGNATURE
    return store


def _empty_root_seam_agent(weight: float, records):
    """Seam agent whose current root state is the ablation root: the
    tracked world-state signature seeds empty by construction (design
    section 6.5)."""

    env, logger, agent, neutral, removal = _seam_agent(weight, records)
    agent.current_human_prior_root_object_state = (
        ObjectTrackSet.empty().to_root_object_state()
    )
    return env, logger, agent, neutral, removal


class PlannerSeamRootBaselineTests(unittest.TestCase):
    """Section 6.5 fix: the designated baseline resolves the current side
    at the empty-signature ablation root; refusal survives elsewhere."""

    def test_root_track_state_seeds_the_empty_signature(self) -> None:
        # The defect precondition, pinned: the empty root track state
        # carries the empty signature, which the store can never key.
        _env, _logger, agent, *_rest = _empty_root_seam_agent(
            1.0, _root_designated_records()
        )
        self.assertEqual(
            agent.current_human_prior_root_object_state
            .tracked_world_state_signature,
            "",
        )
        with self.assertRaises(ValueError):
            provenance(configuration_signature="")

    def test_baseline_resolution_fires_at_empty_signature_root(self) -> None:
        _env, logger, agent, _neutral, removal = _empty_root_seam_agent(
            1.0, _root_designated_records()
        )
        bonus, components = agent._archive_verified_accessibility_bonus(
            removal
        )
        assert components is not None
        self.assertTrue(components.scored)
        self.assertIsNone(components.refusal_reason)
        self.assertGreater(bonus, 0.0)

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        # The certified removal-class branch was selected despite its
        # lower plain score: the baseline resolution let the preference
        # term rank it at the ablation root.
        self.assertEqual(
            [b.tracked_world_state_signature for b in agent.archive],
            [NEUTRAL_SIGNATURE],
        )
        restored = _restored_event(logger)
        self.assertEqual(
            restored["verified_accessibility_current_source"], "baseline"
        )
        self.assertTrue(restored["verified_accessibility_scored"])
        self.assertGreater(
            restored["verified_accessibility_total_bonus"], 0.0
        )
        committed = _committed_event(logger)
        self.assertEqual(
            committed["verified_accessibility_current_source"], "baseline"
        )

    def test_removal_candidate_vs_baseline_scores_the_sanity_value(
        self,
    ) -> None:
        # The section 6.4 sanity value: 17 certified new cells plus the
        # certified milestone-bearing cell (12, 11) at weight 8.0 = +25.0.
        _env, _logger, agent, _neutral, removal = _empty_root_seam_agent(
            1.0, _root_designated_records()
        )
        bonus, components = agent._archive_verified_accessibility_bonus(
            removal
        )
        assert components is not None
        self.assertTrue(components.scored)
        self.assertEqual(len(components.newly_reachable_cells), 17)
        self.assertEqual(
            components.newly_reachable_milestone_cells, (MILESTONE_CELL,)
        )
        self.assertEqual(components.total_bonus, 25.0)
        self.assertEqual(bonus, 25.0)

    def test_candidate_equal_to_baseline_scores_zero(self) -> None:
        env, _logger, agent, *_rest = _empty_root_seam_agent(
            1.0, _root_designated_records()
        )
        baseline_branch = _ArchivedBranch(
            state=env.save_state(),
            frame=env._frame(),
            plan=NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
            score=1.0,
            scene="archived-elsewhere",
            created=0,
            origin_signature="origin",
            tracked_world_state_signature=BASELINE_SENTINEL_SIGNATURE,
        )
        bonus, components = agent._archive_verified_accessibility_bonus(
            baseline_branch
        )
        assert components is not None
        self.assertTrue(components.scored)
        self.assertEqual(components.total_bonus, 0.0)
        self.assertEqual(bonus, 0.0)

    def test_missing_baseline_preserves_refusal(self) -> None:
        records = _root_designated_records()
        records.root_configuration_signature = None
        _env, logger, agent, _neutral, removal = _empty_root_seam_agent(
            1.0, records
        )
        self.assertEqual(
            agent._archive_verified_accessibility_bonus(removal),
            (0.0, None),
        )

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        # Without a designated baseline the term refuses at the empty
        # root exactly as before and the plain tiebreak restores the
        # higher-scored neutral branch.
        self.assertEqual(
            [b.tracked_world_state_signature for b in agent.archive],
            [REMOVAL_SIGNATURE],
        )
        restored = _restored_event(logger)
        self.assertFalse(restored["verified_accessibility_scored"])
        self.assertEqual(
            restored["verified_accessibility_refusal_reason"],
            "record_missing_or_disabled",
        )
        self.assertEqual(
            restored["verified_accessibility_current_source"], "missing"
        )

    def test_unknown_nonempty_current_signature_still_refuses(self) -> None:
        # A representable-but-unmapped current configuration is genuinely
        # unknown: the baseline never stands in for it (section 6.8).
        _env, _logger, agent, _neutral, removal = _seam_agent(
            1.0, _root_designated_records()
        )
        agent.current_human_prior_root_object_state = (
            HumanPriorRootObjectState(
                tracked_world_state_signature="unknown-sig"
            )
        )
        self.assertEqual(
            agent._archive_verified_accessibility_bonus(removal),
            (0.0, None),
        )
        self.assertEqual(
            agent._verified_accessibility_current_source(), "missing"
        )

    def test_mapped_current_signature_wins_over_the_baseline(self) -> None:
        # Candidate-side and mapped current-side resolution are unchanged:
        # with a live mapped signature the baseline is never consulted.
        _env, _logger, agent, _neutral, removal = _seam_agent(
            1.0, _root_designated_records()
        )
        agent.current_human_prior_root_object_state = (
            HumanPriorRootObjectState(
                tracked_world_state_signature=REMOVAL_SIGNATURE
            )
        )
        bonus, components = agent._archive_verified_accessibility_bonus(
            removal
        )
        assert components is not None
        self.assertTrue(components.scored)
        self.assertEqual(components.total_bonus, 0.0)
        self.assertEqual(bonus, 0.0)
        self.assertEqual(
            agent._verified_accessibility_current_source(), "mapped"
        )

    def test_zero_weight_never_consults_a_designated_store(self) -> None:
        store = _ExplodingDesignatedStore()
        store.root_configuration_signature = BASELINE_SENTINEL_SIGNATURE
        _env, _logger, agent, neutral, removal = _empty_root_seam_agent(
            0.0, store
        )
        for branch in (neutral, removal):
            self.assertEqual(
                agent._archive_verified_accessibility_bonus(branch),
                (0.0, None),
            )
        self.assertEqual(
            agent._verified_accessibility_reserve_rank(REMOVAL_SIGNATURE),
            0.0,
        )
        self.assertEqual(
            agent._verified_accessibility_current_source(), "disabled"
        )

    def test_reserve_ranking_uses_the_baseline_at_the_empty_root(
        self,
    ) -> None:
        _env, _logger, agent, *_rest = _empty_root_seam_agent(
            1.0, _root_designated_records()
        )
        self.assertEqual(
            agent._verified_accessibility_reserve_rank(REMOVAL_SIGNATURE),
            25.0,
        )
        self.assertEqual(
            agent._verified_accessibility_reserve_rank(NEUTRAL_SIGNATURE),
            0.0,
        )


class RootConfigurationLoaderTests(unittest.TestCase):
    """Loader-side designation: at most one root/current baseline."""

    @staticmethod
    def _entry(signature: str, root=None):
        entry = RecordLoaderTests._entry(signature=signature)
        if root is not None:
            entry["root_configuration"] = root
        return entry

    def _load(self, payload):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "records.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_verified_accessibility_records(str(path))

    def test_designated_baseline_is_stored_separately(self) -> None:
        records = self._load(
            [
                self._entry(REMOVAL_SIGNATURE),
                self._entry(BASELINE_SENTINEL_SIGNATURE, root=True),
            ]
        )
        self.assertIsInstance(records, VerifiedAccessibilityRecordStore)
        self.assertEqual(
            records.root_configuration_signature,
            BASELINE_SENTINEL_SIGNATURE,
        )
        self.assertIs(
            records.root_record, records[BASELINE_SENTINEL_SIGNATURE]
        )
        # The designation adds no lookup key and is not record content:
        # the designated record's content signature is unchanged.
        self.assertEqual(
            sorted(records),
            sorted([REMOVAL_SIGNATURE, BASELINE_SENTINEL_SIGNATURE]),
        )
        undesignated = self._load(
            [self._entry(BASELINE_SENTINEL_SIGNATURE)]
        )
        self.assertEqual(
            records[BASELINE_SENTINEL_SIGNATURE].content_signature(),
            undesignated[BASELINE_SENTINEL_SIGNATURE].content_signature(),
        )

    def test_undesignated_store_has_no_baseline(self) -> None:
        records = self._load(
            [self._entry(REMOVAL_SIGNATURE, root=False)]
        )
        self.assertIsNone(records.root_configuration_signature)
        self.assertIsNone(records.root_record)

    def test_duplicate_root_configuration_refused_at_load(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "duplicate root_configuration"
        ):
            self._load(
                [
                    self._entry(REMOVAL_SIGNATURE, root=True),
                    self._entry(BASELINE_SENTINEL_SIGNATURE, root=True),
                ]
            )

    def test_non_boolean_root_configuration_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "root_configuration"):
            self._load([self._entry(REMOVAL_SIGNATURE, root="yes")])


if __name__ == "__main__":
    unittest.main()
