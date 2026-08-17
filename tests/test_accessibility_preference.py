import json
import unittest

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


if __name__ == "__main__":
    unittest.main()
