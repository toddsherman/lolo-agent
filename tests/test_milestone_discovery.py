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


# ---------------------------------------------------------------------------
# WP9a scoring-run additions (append-only): telemetry-reduction runner tests.
# Synthetic event dicts only; no telemetry is read. Preregistration:
# docs/milestone-scoring-2026-08-16.md section 1.
# ---------------------------------------------------------------------------

from lolo_agent.milestone_discovery import MatchedEndpointPair
from lolo_agent.milestone_discovery_run import (
    KIND_COLLECTION,
    KIND_LIFE_LOSS,
    KIND_SCENE_TRANSITION,
    REASON_MAPPED,
    REASON_RESTORE_COMMIT,
    assemble_run_pairs,
    canonical_json,
    content_digest,
    decode_visual_signature,
    evaluate_gates,
    reduce_run_events,
)


def hexsig(values) -> str:
    return bytes(values).hex()


ROOT1 = (1, 1, 1, 1)
NOOP1 = (1, 1, 1, 2)   # animation drift under neutral control
UP1 = (1, 9, 1, 2)     # action effect plus the same drift cell
ROOT2 = UP1
RIGHT2 = (1, 9, 7, 2)
NOOP2 = ROOT2          # neutral reproduces nothing at decision 2
ROOT4 = (5, 5, 5, 5)
OPT4 = (5, 5, 9, 5)
OPTN4 = (5, 5, 5, 5)
ROOT5 = (2, 2, 2, 2)
LEFT5 = (2, 8, 2, 2)


def synthetic_run_events():
    seq = [0]

    def event(name, **fields):
        seq[0] += 1
        row = {"event": name, "seq": seq[0], "attempt": 1}
        row.update(fields)
        return row

    return [
        event("env_reset", frame="f-reset", visual_signature=hexsig((0, 0, 0, 0))),
        event(
            "env_step",
            phase="bootstrap",
            frame="f-boot",
            visual_signature=hexsig((0, 0, 0, 1)),
        ),
        event(
            "decision_started",
            decision=1,
            frame="f-root1",
            visual_signature=hexsig(ROOT1),
        ),
        event("state_saved", state_id="state-1", frame="f-root1"),
        event(
            "branch_verified",
            decision=1,
            action="up",
            action_frames=16,
            branch_id="d1-b1",
            frame="f-up",
            visual_signature=hexsig(UP1),
        ),
        event(
            "branch_verified",
            decision=1,
            action="noop",
            action_frames=16,
            branch_id="d1-b2",
            frame="f-noop1",
            visual_signature=hexsig(NOOP1),
        ),
        event(
            "decision_committed",
            decision=1,
            action="up",
            action_frames=16,
            frame="f-up",
            visual_signature=hexsig(UP1),
            restored_archive=False,
            human_prior_collected_hearts=0,
            scene_signature="scene-1",
        ),
        event(
            "decision_started",
            decision=2,
            frame="f-up",
            visual_signature=hexsig(ROOT2),
        ),
        event(
            "branch_verified",
            decision=2,
            action="right",
            action_frames=16,
            branch_id="d2-b1",
            frame="f-right",
            visual_signature=hexsig(RIGHT2),
        ),
        event(
            "matched_neutral_verified",
            decision=2,
            action="noop",
            action_frames=16,
            frame="f-noop2",
            visual_signature=hexsig(NOOP2),
        ),
        event(
            "decision_committed",
            decision=2,
            action="right",
            action_frames=16,
            frame="f-right",
            visual_signature=hexsig(RIGHT2),
            restored_archive=False,
            human_prior_collected_hearts=1,
            scene_signature="scene-2",
        ),
        event(
            "decision_committed",
            decision=3,
            action="up",
            action_frames=16,
            frame="f-restored",
            visual_signature=hexsig(ROOT1),
            restored_archive=True,
            scene_signature="scene-2",
        ),
        event(
            "decision_started",
            decision=4,
            frame="f-root4",
            visual_signature=hexsig(ROOT4),
        ),
        event("state_saved", state_id="state-9", frame="f-root4"),
        event(
            "human_prior_option_branch_verified",
            decision=4,
            source_state_id="state-9",
            path=["down", "a"],
            durations=[16, 1],
            frame="f-opt",
            visual_signature=hexsig(OPT4),
        ),
        event(
            "human_prior_option_neutral_verified",
            decision=4,
            source_state_id="state-9",
            elapsed_frames=17,
            frame="f-optn",
            visual_signature=hexsig(OPTN4),
        ),
        event(
            "decision_started",
            decision=5,
            frame="f-root5",
            visual_signature=hexsig(ROOT5),
        ),
        event(
            "decision_committed",
            decision=5,
            action="left",
            action_frames=16,
            frame="f-left",
            visual_signature=hexsig(LEFT5),
            restored_archive=False,
            human_prior_life_loss_confirmed=True,
            scene_signature="scene-2",
        ),
    ]


class DecodeVisualSignatureTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        self.assertEqual(decode_visual_signature(hexsig(ROOT1)), ROOT1)

    def test_rejects_invalid(self) -> None:
        self.assertIsNone(decode_visual_signature(None))
        self.assertIsNone(decode_visual_signature(""))
        self.assertIsNone(decode_visual_signature("abc"))
        self.assertIsNone(decode_visual_signature("zz"))


class ReduceRunEventsTests(unittest.TestCase):
    def test_reduction_collects_expected_rows(self) -> None:
        reduction = reduce_run_events(synthetic_run_events(), "fixture-run")
        self.assertEqual(len(reduction.committed), 4)
        self.assertEqual(len(reduction.strict_branches), 2)
        self.assertIn((1, 1, 16), reduction.strict_noop_endpoints)
        self.assertIn((1, 2, 16), reduction.strict_noop_endpoints)
        self.assertEqual(len(reduction.option_branches), 1)
        self.assertIn(("state-9", 17), reduction.option_full_neutrals)
        self.assertEqual(reduction.state_frame["state-9"], "f-root4")
        # Pool: reset, bootstrap, and the first decision root only.
        self.assertEqual(
            reduction.pool_frames, ("f-reset", "f-boot", "f-root1")
        )
        self.assertEqual(reduction.skipped_rows, 0)

    def test_malformed_rows_are_counted_not_fatal(self) -> None:
        events = synthetic_run_events()
        events.append({"event": "state_saved", "seq": 999})
        events.append(
            {
                "event": "branch_verified",
                "seq": 1000,
                "decision": 9,
                "action": "up",
            }
        )
        reduction = reduce_run_events(events, "fixture-run")
        self.assertEqual(reduction.skipped_rows, 2)


class AssembleRunPairsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reduction = reduce_run_events(
            synthetic_run_events(), "fixture-run"
        )
        self.result = assemble_run_pairs(self.reduction)

    def pair_by_branch(self, branch_id: str) -> MatchedEndpointPair:
        for candidate in self.result.pairs:
            if candidate.provenance.branch_id == branch_id:
                return candidate
        raise AssertionError(f"no pair {branch_id}")

    def test_pair_counters(self) -> None:
        counters = self.result.counters
        self.assertEqual(counters["strict_branch_pairs"], 2)
        self.assertEqual(counters["option_branch_pairs"], 1)
        self.assertEqual(counters["committed_only_pairs"], 1)
        self.assertEqual(counters["restored_commits_excluded"], 1)
        self.assertEqual(counters["controls_resolved"], 3)
        self.assertEqual(counters["controls_unresolved"], 1)

    def test_strict_pair_matches_control_and_successors(self) -> None:
        pair = self.pair_by_branch("d1-b1")
        self.assertEqual(pair.root, ROOT1)
        self.assertEqual(pair.factual, UP1)
        self.assertEqual(pair.control, NOOP1)
        # Successors: decision 2's committed array, truncated by the
        # restored commit at decision 3.
        self.assertEqual(pair.successors, (RIGHT2,))
        self.assertEqual(pair.provenance.source, "telemetry")

    def test_option_pair_uses_search_root_and_full_neutral(self) -> None:
        pair = self.pair_by_branch("option-seq-15")
        self.assertEqual(pair.root, ROOT4)
        self.assertEqual(pair.factual, OPT4)
        self.assertEqual(pair.control, OPTN4)
        self.assertEqual(pair.provenance.action, "down,a")
        self.assertEqual(pair.provenance.duration, 17)
        self.assertEqual(pair.successors, ())

    def test_committed_transition_not_duplicated(self) -> None:
        branch_ids = [
            pair.provenance.branch_id for pair in self.result.pairs
        ]
        self.assertNotIn("committed-1-1", branch_ids)
        self.assertNotIn("committed-1-2", branch_ids)
        self.assertIn("committed-1-5", branch_ids)

    def test_committed_only_pair_is_dependence_censored(self) -> None:
        pair = self.pair_by_branch("committed-1-5")
        self.assertIsNone(pair.control)
        self.assertEqual(pair.root, ROOT5)
        self.assertEqual(pair.factual, LEFT5)

    def test_drift_signatures_come_from_control_arms(self) -> None:
        # ROOT1 -> NOOP1 drifts; ROOT2 -> NOOP2 and ROOT4 -> OPTN4 do not.
        self.assertEqual(len(self.result.drift_signatures), 1)

    def test_pool_signatures_are_pre_intervention_only(self) -> None:
        self.assertEqual(len(self.result.pool_signatures), 3)
        self.assertNotIn(
            content_signature(RIGHT2), self.result.pool_signatures
        )

    def test_instances_extracted_with_signatures(self) -> None:
        by_kind = {}
        for instance in self.result.instances:
            by_kind.setdefault(instance.kind, []).append(instance)
        self.assertEqual(len(by_kind[KIND_COLLECTION]), 1)
        collection = by_kind[KIND_COLLECTION][0]
        self.assertEqual(collection.decision, 2)
        self.assertEqual(collection.reason, REASON_MAPPED)
        expected = extract_event(
            MatchedEndpointPair(
                provenance=EventProvenance(
                    "x", 2, "b", "right", 16, source="telemetry"
                ),
                root=ROOT2,
                factual=RIGHT2,
            )
        )
        assert expected is not None
        self.assertEqual(collection.event_signature, expected.signature)
        self.assertEqual(len(by_kind[KIND_LIFE_LOSS]), 1)
        self.assertEqual(by_kind[KIND_LIFE_LOSS][0].decision, 5)
        # Scene transitions: scene-1 -> scene-2 at decision 2, and the
        # restored commit at decision 3 keeps scene-2 (no new instance).
        self.assertEqual(len(by_kind[KIND_SCENE_TRANSITION]), 1)
        self.assertEqual(by_kind[KIND_SCENE_TRANSITION][0].decision, 2)

    def test_restored_commit_instances_report_restore_reason(self) -> None:
        events = synthetic_run_events()
        for row in events:
            if (
                row["event"] == "decision_committed"
                and row["decision"] == 3
            ):
                row["human_prior_life_loss_confirmed"] = True
        result = assemble_run_pairs(
            reduce_run_events(events, "fixture-run")
        )
        losses = [
            instance
            for instance in result.instances
            if instance.kind == KIND_LIFE_LOSS and instance.decision == 3
        ]
        self.assertEqual(len(losses), 1)
        self.assertEqual(losses[0].reason, REASON_RESTORE_COMMIT)
        self.assertIsNone(losses[0].event_signature)


class RunnerScoringIntegrationTests(unittest.TestCase):
    def test_synthetic_run_scores_deterministically(self) -> None:
        result = assemble_run_pairs(
            reduce_run_events(synthetic_run_events(), "fixture-run")
        )
        first = score_events(
            extract_events(result.pairs), result.pool_signatures
        )
        second = score_events(
            extract_events(tuple(reversed(result.pairs))),
            result.pool_signatures,
        )
        self.assertEqual(
            [(s.signature, s.score, s.valence) for s in first],
            [(s.signature, s.score, s.valence) for s in second],
        )


class GateEvaluationTests(unittest.TestCase):
    def section(self, instances):
        return {"instances": instances}

    def test_gates_pool_instances_across_corpora(self) -> None:
        collection_pass = {
            "kind": KIND_COLLECTION,
            "valence": VALENCE_POSITIVE,
            "score": 1.5,
        }
        collection_fail = {"kind": KIND_COLLECTION, "reason": "no_pair"}
        loss_pass = {"kind": KIND_LIFE_LOSS, "valence": VALENCE_NEGATIVE}
        loss_fail = {"kind": KIND_LIFE_LOSS, "valence": VALENCE_UNRESOLVED}
        gates = evaluate_gates(
            self.section([collection_pass] * 3 + [collection_fail]),
            self.section([collection_pass] + [loss_pass] * 10 + [loss_fail]),
        )
        self.assertEqual(gates["collection_instances"], 5)
        self.assertEqual(gates["collection_positive_nonzero"], 4)
        self.assertAlmostEqual(gates["positive_recall"], 0.8)
        self.assertTrue(gates["positive_recall_gate_passed"])
        self.assertEqual(gates["life_loss_instances"], 11)
        self.assertEqual(gates["life_loss_negative"], 10)
        self.assertTrue(gates["negative_recall_gate_passed"])

    def test_positive_gate_requires_nonzero_score(self) -> None:
        zero_score = {
            "kind": KIND_COLLECTION,
            "valence": VALENCE_POSITIVE,
            "score": 0.0,
        }
        gates = evaluate_gates(
            self.section([zero_score]), self.section([])
        )
        self.assertEqual(gates["collection_positive_nonzero"], 0)
        self.assertFalse(gates["positive_recall_gate_passed"])


class ReportDeterminismTests(unittest.TestCase):
    def test_canonical_json_is_key_order_independent(self) -> None:
        first = {"b": 1, "a": [1, 2], "c": {"y": 0, "x": 1}}
        second = {"c": {"x": 1, "y": 0}, "a": [1, 2], "b": 1}
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(content_digest(first), content_digest(second))
