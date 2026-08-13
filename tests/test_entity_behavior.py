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

    def test_controlled_relation_context_is_translation_invariant(self) -> None:
        first = AnonymousEntityBehaviorModel.relational_context_signature(
            (5, 2),
            ((5, 4),),
        )
        translated = AnonymousEntityBehaviorModel.relational_context_signature(
            (11, 7),
            ((11, 9),),
        )
        farther = AnonymousEntityBehaviorModel.relational_context_signature(
            (5, 2),
            ((2, 3),),
        )

        self.assertEqual(first, translated)
        self.assertNotEqual(first, farther)
        self.assertIn("distance=2", first)
        self.assertIn("distance=3-4", farther)

    def test_controlled_relation_context_separates_near_and_far_behavior(
        self,
    ) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=2)
        appearance = (9, 2, 6, 5)
        near = model.relational_context_signature((5, 2), ((5, 4),))
        far = model.relational_context_signature((5, 2), ((2, 3),))
        for _ in range(2):
            model.observe(
                appearance,
                Action.NOOP,
                16,
                "transformed",
                context_signature=near,
                autonomous=True,
            )
            model.observe(
                appearance,
                Action.NOOP,
                16,
                "stationary",
                context_signature=far,
                autonomous=True,
            )

        self.assertEqual(
            model.predict(
                appearance,
                Action.NOOP,
                16,
                context_signature=near,
                autonomous=True,
            ).outcome_signature,
            "transformed",
        )
        self.assertEqual(
            model.predict(
                appearance,
                Action.NOOP,
                16,
                context_signature=far,
                autonomous=True,
            ).outcome_signature,
            "stationary",
        )

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
        self.assertEqual(prediction.causal_hazardous_probability, 0.0)
        self.assertEqual(prediction.causal_hazard_samples, 0)
        self.assertFalse(prediction.causal_hazard_known)
        self.assertNotIn("hazard", model.to_dict()["types"][0])

    def test_causal_hazard_provenance_excludes_correlated_terminal_rows(
        self,
    ) -> None:
        model = AnonymousEntityBehaviorModel(
            minimum_prediction_samples=2
        )
        appearance = (5, 2, 8, 1)
        context = "controlled-relative:v1:distance=1"
        for index in range(2):
            model.observe(
                appearance,
                Action.NOOP,
                224,
                "correlated-reset",
                context_signature=context,
                hazardous=True,
                autonomous=True,
                evidence_id=f"passive-{index}",
            )
        correlated = model.predict(
            appearance,
            Action.NOOP,
            224,
            context_signature=context,
            autonomous=True,
        )
        self.assertEqual(correlated.hazardous_probability, 1.0)
        self.assertFalse(correlated.causal_hazard_known)

        for index in range(2):
            evidence_id = f"causal-{index}"
            model.observe(
                appearance,
                Action.NOOP,
                224,
                "localized-reset",
                context_signature=context,
                hazardous=True,
                autonomous=True,
                evidence_id=evidence_id,
            )
            self.assertTrue(
                model.backfill_causal_hazard_evidence(
                    0,
                    Action.NOOP,
                    224,
                    context,
                    True,
                    evidence_id,
                    autonomous=True,
                )
            )
        attributed = model.predict(
            appearance,
            Action.NOOP,
            224,
            context_signature=context,
            autonomous=True,
        )
        self.assertTrue(attributed.causal_hazard_known)
        self.assertEqual(attributed.causal_hazard_samples, 2)
        self.assertEqual(
            attributed.causal_hazardous_probability, 1.0
        )
        self.assertFalse(
            model.backfill_causal_hazard_evidence(
                0,
                Action.NOOP,
                224,
                context,
                True,
                "causal-0",
                autonomous=True,
            )
        )

    def test_schema_four_checkpoint_has_no_inferred_causal_provenance(
        self,
    ) -> None:
        model = AnonymousEntityBehaviorModel(
            minimum_prediction_samples=1
        )
        appearance = (2, 3, 5)
        model.observe(
            appearance,
            Action.NOOP,
            224,
            "terminal",
            hazardous=True,
            autonomous=True,
            evidence_id="legacy-terminal",
            causal_hazard_evidence=True,
        )
        payload = model.to_dict()
        payload["schema_version"] = 4
        payload.pop("causal_hazard_evidence_ids")
        for rule in payload["rules"]:
            rule.pop("causal_hazardous")
            rule.pop("causal_hazard_samples")

        restored = AnonymousEntityBehaviorModel.from_dict(payload)
        prediction = restored.predict(
            appearance, Action.NOOP, 224, autonomous=True
        )

        self.assertEqual(prediction.hazardous_probability, 1.0)
        self.assertFalse(prediction.causal_hazard_known)

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

    def test_schema_three_checkpoint_loads_with_unconditional_fallback(
        self,
    ) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=1)
        appearance = (1, 7, 2, 8)
        model.observe(
            appearance,
            Action.NOOP,
            16,
            "stationary",
            context_signature="legacy-scene-hash",
            autonomous=True,
        )
        payload = model.to_dict()
        payload["schema_version"] = 3

        restored = AnonymousEntityBehaviorModel.from_dict(payload)
        prediction = restored.predict(
            appearance,
            Action.NOOP,
            16,
            context_signature="controlled-relative:v1:distance=2",
            autonomous=True,
        )

        self.assertTrue(prediction.known)
        self.assertFalse(prediction.context_matched)
        self.assertEqual(prediction.outcome_signature, "stationary")

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
