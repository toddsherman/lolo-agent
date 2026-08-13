from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lolo_agent.entity_behavior import AnonymousEntityBehaviorModel
from lolo_agent.environment import Action


class AnonymousEntityBehaviorModelTests(unittest.TestCase):
    def test_recurring_appearance_reuses_behavior_across_contexts(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=2)
        appearance = (1, 2, 3, 4)
        first = model.observe(
            appearance,
            Action.NOOP,
            4,
            "moves-toward-controlled-entity",
            context_signature="room-a",
            autonomous=True,
        )
        second = model.observe(
            (1, 2, 3, 5),
            Action.NOOP,
            4,
            "moves-toward-controlled-entity",
            context_signature="room-b",
            autonomous=True,
        )

        prediction = model.predict(
            (1, 2, 3, 4),
            Action.NOOP,
            4,
            context_signature="unseen-room",
            autonomous=True,
        )

        self.assertEqual(first.type_id, second.type_id)
        self.assertTrue(prediction.known)
        self.assertEqual(
            prediction.outcome_signature,
            "moves-toward-controlled-entity",
        )
        self.assertEqual(prediction.samples, 2)
        self.assertEqual(prediction.contexts_observed, 2)
        self.assertFalse(prediction.context_matched)

    def test_context_rule_overrides_conflicting_unconditional_rule(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=2)
        appearance = (8, 8, 2, 2)
        for _ in range(2):
            model.observe(
                appearance,
                Action.NOOP,
                8,
                "stationary",
                context_signature="condition-a",
                autonomous=True,
            )
        for _ in range(3):
            model.observe(
                appearance,
                Action.NOOP,
                8,
                "mobile",
                context_signature="condition-b",
                autonomous=True,
            )

        contextual = model.predict(
            appearance,
            Action.NOOP,
            8,
            context_signature="condition-a",
            autonomous=True,
        )
        unseen = model.predict(
            appearance,
            Action.NOOP,
            8,
            context_signature="condition-c",
            autonomous=True,
        )

        self.assertTrue(contextual.context_matched)
        self.assertEqual(contextual.outcome_signature, "stationary")
        self.assertEqual(contextual.outcome_probability, 1.0)
        self.assertFalse(unseen.context_matched)
        self.assertEqual(unseen.outcome_signature, "mobile")
        self.assertEqual(unseen.samples, 5)
        self.assertGreater(unseen.entropy, 0.0)

    def test_contradictory_evidence_reduces_confidence(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=1)
        appearance = (3, 1, 4, 1)
        model.observe(appearance, Action.A, 4, "state-one")
        certain = model.predict(appearance, Action.A, 4)
        model.observe(appearance, Action.A, 4, "state-two")
        conflicted = model.predict(appearance, Action.A, 4)

        self.assertGreater(certain.confidence, conflicted.confidence)
        self.assertEqual(conflicted.outcome_probability, 0.5)
        self.assertGreater(conflicted.entropy, certain.entropy)

    def test_unknown_appearance_is_not_created_by_prediction(self) -> None:
        model = AnonymousEntityBehaviorModel()
        prediction = model.predict((9, 9, 9), Action.LEFT, 4)

        self.assertIsNone(prediction.type_id)
        self.assertFalse(prediction.known)
        self.assertEqual(model.type_count, 0)

    def test_outcome_signature_ignores_minor_appearance_animation(self) -> None:
        first = AnonymousEntityBehaviorModel.effect_signature(
            (10, 10, 10),
            (11, 10, 10),
            (10, 10, 10),
            relative_effect_cells=((0, 0),),
        )
        second = AnonymousEntityBehaviorModel.effect_signature(
            (12, 10, 10),
            (13, 10, 10),
            (12, 10, 10),
            relative_effect_cells=((0, 0),),
        )

        self.assertEqual(first, second)

    def test_hazard_probability_is_empirical_not_a_type_label(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=2)
        appearance = (5, 5, 5)
        model.observe(
            appearance,
            Action.RIGHT,
            4,
            "contact-reset",
            hazardous=True,
        )
        model.observe(
            appearance,
            Action.RIGHT,
            4,
            "contact-reset",
            hazardous=True,
        )
        model.observe(
            appearance,
            Action.RIGHT,
            4,
            "blocked",
            hazardous=False,
        )

        prediction = model.predict(appearance, Action.RIGHT, 4)

        self.assertAlmostEqual(prediction.hazardous_probability, 2 / 3)
        self.assertNotIn("hazard", model.to_dict()["types"][0])

    def test_checkpoint_round_trip_preserves_predictions_and_digest(self) -> None:
        model = AnonymousEntityBehaviorModel(
            appearance_match_threshold=0.12,
            minimum_prediction_samples=1,
        )
        source = (1, 7, 2, 8)
        outcome = model.effect_signature(
            source,
            (2, 7, 2, 8),
            source,
            relative_effect_cells=((0, 0), (1, 0)),
            player_displacement=(1, 0),
        )
        model.observe(
            source,
            Action.RIGHT,
            16,
            outcome,
            context_signature="context",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anonymous-behavior.json"
            model.save(path)
            restored = AnonymousEntityBehaviorModel.load(path)

        self.assertEqual(restored.digest, model.digest)
        self.assertEqual(
            restored.predict(
                source,
                Action.RIGHT,
                16,
                context_signature="context",
            ),
            model.predict(
                source,
                Action.RIGHT,
                16,
                context_signature="context",
            ),
        )

    def test_exact_replayed_evidence_does_not_inflate_confidence(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=2)
        first = model.observe(
            (2, 3, 5),
            Action.A,
            4,
            "changed",
            evidence_id="same-save-state-branch",
        )
        digest = model.digest
        replay = model.observe(
            (2, 3, 5),
            Action.A,
            4,
            "changed",
            evidence_id="same-save-state-branch",
        )

        self.assertTrue(first.accepted)
        self.assertFalse(replay.accepted)
        self.assertEqual(replay.prediction_after.samples, 1)
        self.assertEqual(model.observation_count, 1)
        self.assertEqual(model.digest, digest)

    def test_surprise_uses_distribution_before_current_observation(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=1)
        appearance = (2, 7, 1, 8)
        model.observe(appearance, Action.LEFT, 4, "blocked")

        contradictory = model.observe(
            appearance,
            Action.LEFT,
            4,
            "moved",
        )

        self.assertGreater(contradictory.surprise, 20.0)
        self.assertEqual(
            contradictory.prediction_after.outcome_probability,
            0.5,
        )

    def test_observations_reject_inconsistent_feature_dimensions(self) -> None:
        model = AnonymousEntityBehaviorModel()
        model.observe((1, 2, 3), Action.NOOP, 4, "stable")

        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            model.observe((1, 2), Action.NOOP, 4, "stable")


if __name__ == "__main__":
    unittest.main()
