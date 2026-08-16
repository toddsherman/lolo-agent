import math
import unittest

from lolo_agent.milestone_discovery import (
    EventProvenance,
    MatchedEndpointPair,
    MilestoneScoreConfig,
    VALENCE_BASIS_MIXED,
    VALENCE_BASIS_NOVEL_AND_PERSISTENT,
    VALENCE_BASIS_RETURN_CENSORED,
    VALENCE_BASIS_REVERSION_TO_SEEN,
    VALENCE_NEGATIVE,
    VALENCE_POSITIVE,
    VALENCE_UNRESOLVED,
    content_signature,
    discover_milestones,
    extract_event,
    extract_events,
    pool_values,
    score_events,
    seen_pool_from_pairs,
)


def provenance(decision: int = 1, action: str = "action-0") -> EventProvenance:
    return EventProvenance(
        run_id="fixture-run",
        decision=decision,
        branch_id=f"branch-{decision:04d}",
        action=action,
        duration=16,
    )


def pair(
    root,
    factual,
    control=None,
    successors=(),
    decision: int = 1,
    action: str = "action-0",
) -> MatchedEndpointPair:
    return MatchedEndpointPair(
        provenance=provenance(decision=decision, action=action),
        root=tuple(root),
        factual=tuple(factual),
        control=None if control is None else tuple(control),
        successors=tuple(tuple(successor) for successor in successors),
    )


class PoolValuesTests(unittest.TestCase):
    def test_pooling_is_deterministic_and_reduces_shape(self) -> None:
        values = tuple(range(16))
        first = pool_values(values, 4, 4, 2, 2)
        second = pool_values(values, 4, 4, 2, 2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_pooling_averages_cells(self) -> None:
        values = (0, 0, 8, 8)
        self.assertEqual(pool_values(values, 2, 2, 1, 1), (4,))

    def test_pooling_quantization(self) -> None:
        values = (16, 16, 16, 16)
        self.assertEqual(pool_values(values, 2, 2, 1, 1, quantization=8), (2,))

    def test_pooling_rejects_bad_shapes(self) -> None:
        with self.assertRaises(ValueError):
            pool_values((1, 2, 3), 2, 2, 1, 1)
        with self.assertRaises(ValueError):
            pool_values((1, 2, 3, 4), 2, 2, 4, 1)
        with self.assertRaises(ValueError):
            pool_values((1, 2, 3, 4), 2, 2, 1, 1, quantization=0)


class ContentSignatureTests(unittest.TestCase):
    def test_signature_is_deterministic(self) -> None:
        self.assertEqual(content_signature((1, 2, 3)), content_signature((1, 2, 3)))

    def test_signature_distinguishes_values_and_lengths(self) -> None:
        self.assertNotEqual(content_signature((1, 2)), content_signature((2, 1)))
        self.assertNotEqual(
            content_signature((12, 3)), content_signature((1, 23))
        )


class MatchedEndpointPairTests(unittest.TestCase):
    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            pair((1, 2), (1, 2, 3))
        with self.assertRaises(ValueError):
            pair((1, 2), (1, 2), control=(1,))
        with self.assertRaises(ValueError):
            pair((1, 2), (1, 2), successors=((1,),))

    def test_rejects_empty_root(self) -> None:
        with self.assertRaises(ValueError):
            pair((), ())

    def test_provenance_validation(self) -> None:
        with self.assertRaises(ValueError):
            EventProvenance("", 1, "branch", "action-0", 16)
        with self.assertRaises(ValueError):
            EventProvenance("run", 1, "branch", "action-0", 0)
        with self.assertRaises(ValueError):
            EventProvenance("run", 1, "branch", "action-0", 16, source="other")


class ExtractEventTests(unittest.TestCase):
    def test_no_difference_yields_no_event(self) -> None:
        self.assertIsNone(extract_event(pair((1, 2), (1, 2), control=(1, 2))))

    def test_action_dependent_change(self) -> None:
        event = extract_event(pair((0, 0), (0, 5), control=(0, 0)))
        assert event is not None
        self.assertEqual(event.changed_cells, ((1, 0, 5),))
        self.assertIs(event.action_dependent, True)

    def test_autonomous_change_is_not_action_dependent(self) -> None:
        event = extract_event(pair((0, 0), (0, 5), control=(0, 5)))
        assert event is not None
        self.assertIs(event.action_dependent, False)

    def test_missing_control_is_dependence_censored(self) -> None:
        event = extract_event(pair((0, 0), (0, 5)))
        assert event is not None
        self.assertIsNone(event.action_dependent)

    def test_partial_control_reproduction_is_dependence_censored(self) -> None:
        event = extract_event(pair((0, 0, 0), (0, 5, 7), control=(0, 5, 0)))
        assert event is not None
        self.assertIsNone(event.action_dependent)

    def test_reversion_detected_in_successors(self) -> None:
        event = extract_event(
            pair((0, 0), (0, 5), control=(0, 0), successors=((0, 5), (0, 0)))
        )
        assert event is not None
        self.assertIs(event.reverted, True)

    def test_persistence_detected_in_successors(self) -> None:
        event = extract_event(
            pair((0, 0), (0, 5), control=(0, 0), successors=((0, 5), (1, 5)))
        )
        assert event is not None
        self.assertIs(event.reverted, False)

    def test_no_successors_is_return_censored(self) -> None:
        event = extract_event(pair((0, 0), (0, 5), control=(0, 0)))
        assert event is not None
        self.assertIsNone(event.reverted)

    def test_signature_depends_on_cells_and_values(self) -> None:
        first = extract_event(pair((0, 0), (0, 5), control=(0, 0)))
        same = extract_event(pair((0, 0), (0, 5), control=(0, 0), decision=2))
        different_cell = extract_event(pair((0, 0), (5, 0), control=(0, 0)))
        different_value = extract_event(pair((0, 0), (0, 6), control=(0, 0)))
        assert first and same and different_cell and different_value
        self.assertEqual(first.signature, same.signature)
        self.assertNotEqual(first.signature, different_cell.signature)
        self.assertNotEqual(first.signature, different_value.signature)

    def test_extract_events_drops_pairs_without_difference(self) -> None:
        events = extract_events(
            [
                pair((1, 1), (1, 1), control=(1, 1)),
                pair((1, 1), (1, 9), control=(1, 1)),
            ]
        )
        self.assertEqual(len(events), 1)


class ScoreEventsTests(unittest.TestCase):
    def test_ubiquitous_signature_scores_zero_by_rarity(self) -> None:
        pairs = [
            pair((0, 0), (0, 5), control=(0, 0), successors=((1, 5),), decision=d)
            for d in range(1, 4)
        ]
        scores = score_events(extract_events(pairs))
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0].log_rarity, 0.0)
        self.assertEqual(scores[0].score, 0.0)

    def test_action_independent_signature_scores_zero(self) -> None:
        pairs = [
            pair((0, 0), (0, 5), control=(0, 5), successors=((1, 5),)),
            pair((9, 9), (8, 9), control=(8, 9), successors=((8, 8),), decision=2),
        ]
        scores = score_events(extract_events(pairs))
        for score in scores:
            self.assertEqual(score.action_dependence_rate, 0.0)
            self.assertEqual(score.score, 0.0)

    def test_fully_dependence_censored_signature_scores_zero(self) -> None:
        pairs = [
            pair((0, 0), (0, 5), successors=((1, 5),)),
            pair((9, 9), (9, 8), control=(9, 9), successors=((0, 8),), decision=2),
        ]
        scores = score_events(extract_events(pairs))
        censored = next(s for s in scores if s.dependence_evaluable == 0)
        self.assertEqual(censored.dependence_censored, 1)
        self.assertEqual(censored.action_dependence_rate, 0.0)
        self.assertEqual(censored.score, 0.0)

    def test_fully_return_censored_signature_scores_zero(self) -> None:
        pairs = [
            pair((0, 0), (0, 5), control=(0, 0)),
            pair((9, 9), (9, 8), control=(9, 9), successors=((0, 8),), decision=2),
        ]
        scores = score_events(extract_events(pairs))
        censored = next(s for s in scores if s.return_evaluable == 0)
        self.assertEqual(censored.return_censored, 1)
        self.assertEqual(censored.censored_non_return_factor, 0.0)
        self.assertEqual(censored.score, 0.0)
        self.assertEqual(censored.valence, VALENCE_UNRESOLVED)
        self.assertEqual(censored.valence_basis, VALENCE_BASIS_RETURN_CENSORED)

    def test_seen_successors_yield_zero_novelty_margin(self) -> None:
        successor = (1, 5)
        pairs = [
            pair((0, 0), (0, 5), control=(0, 0), successors=(successor,)),
            pair((9, 9), (9, 8), control=(9, 9), successors=((0, 8),), decision=2),
        ]
        seen = frozenset({content_signature(successor)})
        scores = score_events(extract_events(pairs), seen_signatures=seen)
        target = next(s for s in scores if s.occurrences == 1 and s.persistence_rate)
        if content_signature((0, 8)) in seen:
            self.fail("fixture leaked the wrong successor into the pool")
        seen_scores = [
            s
            for s in scores
            if s.provenance[0].decision == 1
        ]
        self.assertEqual(len(seen_scores), 1)
        self.assertEqual(seen_scores[0].successor_novelty_margin, 0.0)
        self.assertEqual(seen_scores[0].score, 0.0)
        self.assertGreater(target.occurrences, 0)

    def test_rare_dependent_persistent_novel_signature_scores_positive(self) -> None:
        target = pair(
            (0, 0, 0),
            (0, 0, 7),
            control=(0, 0, 0),
            successors=((0, 0, 7), (1, 0, 7)),
        )
        background = [
            pair(
                (2, 2, 2),
                (2, 3, 2),
                control=(2, 2, 2),
                successors=((2, 2, 2),),
                decision=d,
            )
            for d in range(2, 5)
        ]
        pairs = [target] + background
        report = discover_milestones(pairs, seen_pool_from_pairs(pairs))
        self.assertEqual(report.total_pairs, 4)
        self.assertEqual(report.pairs_without_event, 0)
        best = report.scores[0]
        self.assertEqual(best.occurrences, 1)
        self.assertAlmostEqual(best.log_rarity, math.log(4.0))
        self.assertEqual(best.action_dependence_rate, 1.0)
        self.assertEqual(best.censored_non_return_factor, 1.0)
        self.assertEqual(best.successor_novelty_margin, 1.0)
        self.assertAlmostEqual(best.score, math.log(4.0))
        self.assertEqual(best.valence, VALENCE_POSITIVE)
        self.assertEqual(best.valence_basis, VALENCE_BASIS_NOVEL_AND_PERSISTENT)

    def test_reversion_to_seen_classifies_negative(self) -> None:
        pairs = [
            pair(
                (0, 0),
                (0, 5),
                control=(0, 0),
                successors=((0, 5), (0, 0)),
            ),
            pair((9, 9), (9, 8), control=(9, 9), successors=((1, 8),), decision=2),
        ]
        report = discover_milestones(pairs, seen_pool_from_pairs(pairs))
        negative = next(s for s in report.scores if s.reversion_to_seen_rate > 0)
        self.assertEqual(negative.valence, VALENCE_NEGATIVE)
        self.assertEqual(negative.valence_basis, VALENCE_BASIS_REVERSION_TO_SEEN)
        self.assertEqual(negative.censored_non_return_factor, 0.0)
        self.assertEqual(negative.score, 0.0)

    def test_immediate_collapse_onto_seen_pool_is_negative(self) -> None:
        seen_state = (3, 3)
        pairs = [
            pair(
                (0, 0),
                (0, 5),
                control=(0, 0),
                successors=(seen_state, (0, 5)),
            ),
            pair((9, 9), (9, 8), control=(9, 9), successors=((1, 8),), decision=2),
        ]
        seen = frozenset({content_signature(seen_state)})
        scores = score_events(extract_events(pairs), seen_signatures=seen)
        collapsed = next(s for s in scores if s.provenance[0].decision == 1)
        self.assertEqual(collapsed.valence, VALENCE_NEGATIVE)
        self.assertEqual(collapsed.valence_basis, VALENCE_BASIS_REVERSION_TO_SEEN)

    def test_persistent_but_familiar_outcome_is_unresolved(self) -> None:
        familiar = ((1, 5), (2, 5), (3, 5), (4, 5))
        successors = ((0, 5),) + familiar
        pairs = [
            pair((0, 0), (0, 5), control=(0, 0), successors=successors),
            pair((9, 9), (9, 8), control=(9, 9), successors=((1, 8),), decision=2),
        ]
        seen = frozenset(content_signature(values) for values in familiar)
        scores = score_events(extract_events(pairs), seen_signatures=seen)
        target = next(s for s in scores if s.provenance[0].decision == 1)
        self.assertEqual(target.valence, VALENCE_UNRESOLVED)
        self.assertEqual(target.valence_basis, VALENCE_BASIS_MIXED)

    def test_scores_are_deterministic_under_input_order(self) -> None:
        pairs = [
            pair((0, 0), (0, 5), control=(0, 0), successors=((1, 5),)),
            pair((9, 9), (9, 8), control=(9, 9), successors=((0, 8),), decision=2),
            pair((4, 4), (4, 6), control=(4, 4), successors=((4, 6),), decision=3),
        ]
        forward = score_events(extract_events(pairs))
        backward = score_events(extract_events(list(reversed(pairs))))
        self.assertEqual(
            [(s.signature, s.score) for s in forward],
            [(s.signature, s.score) for s in backward],
        )

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            MilestoneScoreConfig(novelty_baseline=-0.1)
        with self.assertRaises(ValueError):
            MilestoneScoreConfig(negative_reversion_threshold=0.0)
        with self.assertRaises(ValueError):
            MilestoneScoreConfig(positive_novelty_threshold=1.5)

    def test_report_counts_pairs_without_events(self) -> None:
        pairs = [
            pair((1, 1), (1, 1), control=(1, 1)),
            pair((1, 1), (1, 9), control=(1, 1), successors=((1, 9),), decision=2),
        ]
        report = discover_milestones(pairs)
        self.assertEqual(report.total_pairs, 2)
        self.assertEqual(report.pairs_without_event, 1)
        self.assertEqual(report.total_events, 1)

    def test_seen_pool_excludes_factual_outcomes(self) -> None:
        pairs = [
            pair((0, 0), (0, 5), control=(0, 1), successors=((1, 5),)),
        ]
        pool = seen_pool_from_pairs(pairs)
        self.assertIn(content_signature((0, 0)), pool)
        self.assertIn(content_signature((0, 1)), pool)
        self.assertNotIn(content_signature((0, 5)), pool)
        self.assertNotIn(content_signature((1, 5)), pool)


if __name__ == "__main__":
    unittest.main()
