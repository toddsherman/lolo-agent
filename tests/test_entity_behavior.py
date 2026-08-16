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

    def test_factored_context_conditions_behavior_on_global_visual_phase(
        self,
    ) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=2)
        appearance = (8, 3, 5, 2)
        relation = model.relational_context_signature((4, 4), ((4, 5),))
        neighborhood = "local-layout"
        quiet_phase = model.phase_signature(((1, 1), (2, 2), (2, 2)))
        active_phase = model.phase_signature(((1, 1), (9, 9), (9, 9)))
        quiet = model.factored_context_signature(
            relation, neighborhood, quiet_phase
        )
        active = model.factored_context_signature(
            relation, neighborhood, active_phase
        )
        for index in range(2):
            model.observe(
                appearance,
                Action.NOOP,
                16,
                "stationary",
                context_signature=quiet,
                autonomous=True,
                evidence_id=f"quiet-{index}",
            )
            model.observe(
                appearance,
                Action.NOOP,
                16,
                "moves",
                context_signature=active,
                autonomous=True,
                evidence_id=f"active-{index}",
            )

        quiet_prediction = model.predict(
            appearance,
            Action.NOOP,
            16,
            context_signature=quiet,
            autonomous=True,
        )
        active_prediction = model.predict(
            appearance,
            Action.NOOP,
            16,
            context_signature=active,
            autonomous=True,
        )

        self.assertTrue(quiet_prediction.context_matched)
        self.assertTrue(active_prediction.context_matched)
        self.assertEqual(quiet_prediction.outcome_signature, "stationary")
        self.assertEqual(active_prediction.outcome_signature, "moves")

    def test_factored_context_generalizes_across_local_layouts(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=2)
        appearance = (4, 4, 7, 7)
        relation = model.relational_context_signature((3, 3), ((3, 4),))
        phase = model.phase_signature(((1, 1), (2, 2)))
        first = model.factored_context_signature(relation, "layout-a", phase)
        second = model.factored_context_signature(relation, "layout-b", phase)
        unseen = model.factored_context_signature(relation, "layout-c", phase)
        model.observe(
            appearance,
            Action.DOWN,
            16,
            "moves-one-cell",
            context_signature=first,
        )
        model.observe(
            appearance,
            Action.DOWN,
            16,
            "moves-one-cell",
            context_signature=second,
        )

        prediction = model.predict(
            appearance,
            Action.DOWN,
            16,
            context_signature=unseen,
        )

        self.assertTrue(prediction.known)
        self.assertTrue(prediction.context_matched)
        self.assertEqual(prediction.outcome_signature, "moves-one-cell")
        self.assertEqual(
            model.exact_context_samples(
                appearance,
                Action.DOWN,
                16,
                unseen,
            ),
            0,
        )
        self.assertEqual(
            model.exact_context_samples(
                appearance,
                Action.DOWN,
                16,
                first,
            ),
            1,
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

    def test_pixel_outcome_descriptor_exposes_learned_inert_semantics(
        self,
    ) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=2)
        appearance = (10, 10, 10)
        descriptor = model.effect_descriptor(
            appearance,
            appearance,
            appearance,
            player_displacement=(0, 0),
        )
        for index in range(2):
            model.observe(
                appearance,
                Action.LEFT,
                16,
                descriptor.signature,
                evidence_id=f"inert-{index}",
                outcome_descriptor=descriptor,
            )

        prediction = model.predict(
            appearance, Action.LEFT, 16
        )

        self.assertTrue(descriptor.intervention_inert)
        self.assertFalse(descriptor.measured_effect)
        self.assertTrue(prediction.known)
        self.assertEqual(prediction.outcome_descriptor, descriptor)
        self.assertEqual(prediction.semantic_samples, 2)
        self.assertEqual(prediction.semantic_coverage, 1.0)
        self.assertEqual(prediction.inert_probability, 1.0)
        self.assertGreater(prediction.inert_confidence, 0.6)
        self.assertEqual(prediction.measured_effect_probability, 0.0)

    def test_descriptor_distinguishes_movement_and_local_change(self) -> None:
        appearance = (10, 10, 10)
        moved = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            appearance,
            appearance,
            player_displacement=(1, 0),
        )
        changed = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            (90, 90, 90),
            appearance,
            relative_effect_cells=((0, 0),),
        )

        self.assertTrue(moved.controlled_movement)
        self.assertTrue(moved.measured_effect)
        self.assertFalse(moved.intervention_inert)
        self.assertTrue(changed.local_visual_change)
        self.assertTrue(changed.measured_effect)
        self.assertFalse(changed.intervention_inert)

    def test_descriptor_represents_push_transform_and_global_phase_change(
        self,
    ) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=1)
        appearance = (1, 1, 1, 1)
        pushed = model.effect_descriptor(
            appearance,
            appearance,
            appearance,
            entity_displacement=(0, 1),
        )
        transformed = model.effect_descriptor(
            appearance,
            (15, 15, 15, 15),
            appearance,
        )
        phase_changed = model.effect_descriptor(
            appearance,
            appearance,
            appearance,
            global_phase_change=True,
        )
        for action, descriptor in (
            (Action.DOWN, pushed),
            (Action.A, transformed),
            (Action.B, phase_changed),
        ):
            model.observe(
                appearance,
                action,
                4,
                descriptor.signature,
                outcome_descriptor=descriptor,
            )

        push_prediction = model.predict(appearance, Action.DOWN, 4)
        transform_prediction = model.predict(appearance, Action.A, 4)
        phase_prediction = model.predict(appearance, Action.B, 4)

        self.assertTrue(pushed.controlled_entity_displacement)
        self.assertTrue(transformed.controlled_appearance_transition)
        self.assertTrue(phase_changed.global_phase_change)
        self.assertEqual(push_prediction.entity_displacement_probability, 1.0)
        self.assertEqual(
            transform_prediction.appearance_transition_probability, 1.0
        )
        self.assertEqual(
            phase_prediction.global_phase_change_probability, 1.0
        )
        self.assertEqual(push_prediction.manipulation_probability, 1.0)
        self.assertEqual(transform_prediction.manipulation_probability, 1.0)
        self.assertEqual(phase_prediction.manipulation_probability, 1.0)

    def test_predictive_family_pools_behavior_across_animation_variants(
        self,
    ) -> None:
        model = AnonymousEntityBehaviorModel(
            appearance_match_threshold=0.01,
            minimum_prediction_samples=2,
        )
        first_appearance = (1, 1, 1, 1)
        second_appearance = (15, 15, 15, 15)
        descriptor = model.effect_descriptor(
            first_appearance,
            first_appearance,
            first_appearance,
            entity_displacement=(1, 0),
        )
        model.observe(
            first_appearance,
            Action.RIGHT,
            16,
            descriptor.signature,
            outcome_descriptor=descriptor,
        )
        model.observe(
            second_appearance,
            Action.RIGHT,
            16,
            descriptor.signature,
            outcome_descriptor=descriptor,
        )

        prediction = model.predict(
            first_appearance, Action.RIGHT, 16
        )

        self.assertEqual(model.type_count, 2)
        self.assertTrue(prediction.known)
        self.assertTrue(prediction.predictive_family_pooled)
        self.assertEqual(prediction.predictive_family_size, 2)
        self.assertEqual(prediction.samples, 2)
        self.assertEqual(prediction.entity_displacement_probability, 1.0)

        inert = model.effect_descriptor(
            second_appearance,
            second_appearance,
            second_appearance,
        )
        for _ in range(2):
            model.observe(
                second_appearance,
                Action.RIGHT,
                16,
                inert.signature,
                outcome_descriptor=inert,
            )

        split_prediction = model.predict(
            first_appearance, Action.RIGHT, 16
        )
        self.assertEqual(split_prediction.predictive_family_size, 1)
        self.assertFalse(split_prediction.predictive_family_pooled)

    def test_descriptor_preserves_limited_visual_resource_transition(
        self,
    ) -> None:
        appearance = (3, 1, 4, 1)
        first_use = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            (15, 9, 15, 9),
            appearance,
            # A distant changed cell can be a visual counter without the
            # model being told its meaning.
            relative_effect_cells=((12, -3),),
            global_phase_change=True,
        )
        local_only = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            (15, 9, 15, 9),
            appearance,
            relative_effect_cells=((0, 0),),
        )

        self.assertTrue(first_use.controlled_appearance_transition)
        self.assertTrue(first_use.global_phase_change)
        self.assertTrue(first_use.manipulation_effect)
        self.assertNotEqual(first_use.signature, local_only.signature)

    def test_schema_seven_phase_evidence_is_safely_remapped(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=1)
        appearance = (3, 1, 4, 1)
        legacy = model.effect_descriptor(
            appearance,
            appearance,
            appearance,
            global_phase_change=True,
        )
        model.observe(
            appearance,
            Action.A,
            4,
            legacy.signature,
            outcome_descriptor=legacy,
            evidence_id="legacy-single-cell-phase",
        )
        payload = model.to_dict()
        payload["schema_version"] = 7

        restored = AnonymousEntityBehaviorModel.from_dict(payload)
        prediction = restored.predict(appearance, Action.A, 4)

        self.assertEqual(restored.to_dict()["schema_version"], 9)
        self.assertEqual(prediction.samples, 1)
        self.assertIsNotNone(prediction.outcome_descriptor)
        self.assertFalse(prediction.outcome_descriptor.global_phase_change)
        self.assertEqual(prediction.global_phase_change_probability, 0.0)
        self.assertTrue(prediction.outcome_descriptor.intervention_inert)
        self.assertNotEqual(prediction.outcome_signature, legacy.signature)

    def test_removal_requires_persistence_horizon_verification(self) -> None:
        appearance = (10, 10, 10)
        removed_patch = (90, 90, 90)

        with self.assertRaisesRegex(ValueError, "persistence-horizon"):
            AnonymousEntityBehaviorModel.effect_descriptor(
                appearance,
                removed_patch,
                appearance,
                entity_removed=True,
            )

        verified = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            removed_patch,
            appearance,
            entity_removed=True,
            removal_persistence_verified=True,
        )
        self.assertTrue(verified.controlled_entity_removal)
        self.assertFalse(verified.controlled_appearance_transition)
        self.assertTrue(verified.manipulation_effect)
        self.assertEqual(verified.transition_kind, "removal")

    def test_removal_descriptor_rejects_contradictory_measurements(
        self,
    ) -> None:
        from lolo_agent.entity_behavior import BehaviorOutcomeDescriptor

        with self.assertRaisesRegex(ValueError, "transit evidence"):
            BehaviorOutcomeDescriptor(
                factual_source_relation="different",
                control_source_relation="same",
                factual_control_relation="different",
                removal_transit_cells=((1, 0),),
            )
        with self.assertRaisesRegex(ValueError, "keep a displacement"):
            BehaviorOutcomeDescriptor(
                factual_source_relation="different",
                control_source_relation="same",
                factual_control_relation="different",
                entity_removed=True,
                entity_displacement=(1, 0),
            )
        with self.assertRaisesRegex(ValueError, "still match its source"):
            BehaviorOutcomeDescriptor(
                factual_source_relation="same",
                control_source_relation="same",
                factual_control_relation="same",
                entity_removed=True,
            )
        with self.assertRaisesRegex(ValueError, "anonymous fingerprint"):
            BehaviorOutcomeDescriptor(
                factual_source_relation="different",
                control_source_relation="same",
                factual_control_relation="different",
                target_appearance="a-supplied-name",
            )

    def test_autonomous_removal_is_denied_controller_credit(self) -> None:
        appearance = (10, 10, 10)
        removed_patch = (90, 90, 90)
        reproduced = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            removed_patch,
            removed_patch,
            entity_removed=True,
            removal_persistence_verified=True,
        )

        self.assertTrue(reproduced.entity_removed)
        self.assertFalse(reproduced.controlled_entity_removal)
        self.assertFalse(reproduced.manipulation_effect)
        self.assertEqual(reproduced.transition_kind, "other")
        self.assertIsNone(reproduced.transition.source_cell)

    def test_descriptor_represents_removal_chain_families(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=1)
        appearance = (1, 1, 1, 1)
        transformed_patch = (60, 60, 60, 60)
        background = (200, 200, 200, 200)

        transformed = model.effect_descriptor(
            appearance,
            transformed_patch,
            appearance,
            transformation_target_feature=transformed_patch,
        )
        displaced = model.effect_descriptor(
            appearance,
            appearance,
            appearance,
            entity_displacement=(1, 0),
        )
        expelled = model.effect_descriptor(
            appearance,
            background,
            appearance,
            entity_removed=True,
            removal_persistence_verified=True,
            removal_transit_cells=((1, 0), (2, 0)),
        )
        for action, descriptor in (
            (Action.A, transformed),
            (Action.RIGHT, displaced),
            (Action.B, expelled),
        ):
            model.observe(
                appearance,
                action,
                16,
                descriptor.signature,
                outcome_descriptor=descriptor,
            )

        self.assertEqual(transformed.transition_kind, "transformation")
        self.assertEqual(displaced.transition_kind, "displacement")
        self.assertEqual(expelled.transition_kind, "expulsion")
        self.assertTrue(expelled.controlled_entity_expulsion)
        self.assertEqual(
            len({transformed.signature, displaced.signature, expelled.signature}),
            3,
        )

        transform_prediction = model.predict(appearance, Action.A, 16)
        displace_prediction = model.predict(appearance, Action.RIGHT, 16)
        expel_prediction = model.predict(appearance, Action.B, 16)

        self.assertEqual(
            transform_prediction.appearance_transition_probability, 1.0
        )
        self.assertEqual(transform_prediction.entity_removal_probability, 0.0)
        self.assertEqual(
            displace_prediction.entity_displacement_probability, 1.0
        )
        self.assertEqual(expel_prediction.entity_removal_probability, 1.0)
        self.assertEqual(expel_prediction.entity_expulsion_probability, 1.0)
        self.assertEqual(expel_prediction.appearance_transition_probability, 0.0)
        self.assertEqual(expel_prediction.manipulation_probability, 1.0)

    def test_transition_exposes_explicit_displacement_vectors(self) -> None:
        appearance = (1, 1, 1, 1)
        displaced = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            appearance,
            appearance,
            entity_displacement=(0, -1),
        )
        blocked = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            appearance,
            appearance,
            player_displacement=(0, 0),
        )
        removed = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            (90, 90, 90, 90),
            appearance,
            entity_removed=True,
            removal_persistence_verified=True,
            removal_transit_cells=((2, 0), (1, 0)),
        )

        displacement = displaced.transition
        self.assertEqual(displacement.kind, "displacement")
        self.assertEqual(displacement.source_cell, (0, 0))
        self.assertEqual(displacement.destination_cell, (0, -1))
        self.assertEqual(displacement.displacement, (0, -1))
        self.assertEqual(displacement.outcome_signature, displaced.signature)

        no_effect = blocked.transition
        self.assertEqual(no_effect.kind, "no_effect")
        self.assertEqual(no_effect.destination_cell, (0, 0))
        self.assertEqual(no_effect.displacement, (0, 0))

        expulsion = removed.transition
        self.assertEqual(expulsion.kind, "expulsion")
        self.assertEqual(expulsion.source_cell, (0, 0))
        self.assertIsNone(expulsion.destination_cell)
        self.assertIsNone(expulsion.displacement)
        self.assertEqual(expulsion.transit_cells, ((1, 0), (2, 0)))

    def test_transformation_target_ignores_animation_variants(self) -> None:
        appearance = (100, 100, 100, 100)
        animated = (104, 100, 100, 100)
        with_target = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            animated,
            appearance,
            transformation_target_feature=animated,
        )
        without_target = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            animated,
            appearance,
        )

        self.assertEqual(with_target.target_appearance, "")
        self.assertEqual(with_target.signature, without_target.signature)

        transformed_patch = (60, 60, 60, 60)
        first = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            transformed_patch,
            appearance,
            transformation_target_feature=transformed_patch,
        )
        # A small animation delta inside the coarse quantization bin keeps
        # the same target fingerprint and outcome signature.
        second = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            (61, 60, 60, 60),
            appearance,
            transformation_target_feature=(61, 60, 60, 60),
        )
        other_target = AnonymousEntityBehaviorModel.effect_descriptor(
            appearance,
            (240, 240, 240, 240),
            appearance,
            transformation_target_feature=(240, 240, 240, 240),
        )

        self.assertNotEqual(first.target_appearance, "")
        self.assertEqual(first.signature, second.signature)
        self.assertNotEqual(first.signature, other_target.signature)

    def test_passive_stationarity_is_not_an_intervention_effect(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=1)
        appearance = (10, 10, 10)
        stationary = model.effect_descriptor(
            appearance,
            appearance,
            appearance,
            relative_effect_cells=((0, 0),),
        )
        model.observe(
            appearance,
            Action.NOOP,
            16,
            stationary.signature,
            autonomous=True,
            outcome_descriptor=stationary,
        )

        prediction = model.predict(
            appearance, Action.NOOP, 16, autonomous=True
        )

        self.assertFalse(stationary.autonomous_visual_change)
        self.assertEqual(prediction.inert_probability, 0.0)
        self.assertEqual(prediction.local_visual_change_probability, 0.0)
        self.assertEqual(prediction.measured_effect_probability, 0.0)

    def test_schema_five_loads_with_unknown_outcome_semantics(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=1)
        appearance = (2, 3, 5)
        descriptor = model.effect_descriptor(
            appearance, appearance, appearance
        )
        model.observe(
            appearance,
            Action.A,
            4,
            descriptor.signature,
            outcome_descriptor=descriptor,
        )
        payload = model.to_dict()
        payload["schema_version"] = 5
        payload.pop("outcome_descriptors")

        restored = AnonymousEntityBehaviorModel.from_dict(payload)
        prediction = restored.predict(appearance, Action.A, 4)

        self.assertTrue(prediction.known)
        self.assertIsNone(prediction.outcome_descriptor)
        self.assertEqual(prediction.semantic_samples, 0)
        self.assertEqual(prediction.inert_probability, 0.0)

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
        descriptor = model.effect_descriptor(
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
            outcome_descriptor=descriptor,
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

    def test_schema_eight_checkpoint_loads_with_preserved_signatures(
        self,
    ) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=1)
        appearance = (1, 1, 1, 1)
        displaced = model.effect_descriptor(
            appearance,
            appearance,
            appearance,
            entity_displacement=(1, 0),
        )
        inert = model.effect_descriptor(
            appearance,
            appearance,
            appearance,
            player_displacement=(0, 0),
        )
        # A schema-8 writer never emitted the explicit transition fields, so
        # descriptors built without them must serialize identically.
        for key in ("target_appearance", "entity_removed", "removal_transit_cells"):
            self.assertNotIn(key, displaced.to_dict())
            self.assertNotIn(key, inert.to_dict())
        model.observe(
            appearance,
            Action.RIGHT,
            16,
            displaced.signature,
            outcome_descriptor=displaced,
            evidence_id="legacy-displacement",
        )
        model.observe(
            appearance,
            Action.LEFT,
            16,
            inert.signature,
            outcome_descriptor=inert,
            evidence_id="legacy-blocked",
        )
        payload = model.to_dict()
        payload["schema_version"] = 8

        restored = AnonymousEntityBehaviorModel.from_dict(payload)

        self.assertEqual(restored.to_dict()["schema_version"], 9)
        self.assertEqual(restored.outcome_descriptor_count, 2)
        prediction = restored.predict(appearance, Action.RIGHT, 16)
        # No remap: schema-8 signatures and evidence provenance survive.
        self.assertEqual(prediction.outcome_signature, displaced.signature)
        self.assertEqual(prediction.entity_displacement_probability, 1.0)
        self.assertFalse(
            restored.observe(
                appearance,
                Action.RIGHT,
                16,
                displaced.signature,
                outcome_descriptor=displaced,
                evidence_id="legacy-displacement",
            ).accepted
        )
        # A legacy descriptor that already implies a displacement surfaces
        # it as an explicit transition after migration.
        explicit = restored.transition_for(displaced.signature)
        self.assertIsNotNone(explicit)
        self.assertEqual(explicit.kind, "displacement")
        self.assertEqual(explicit.source_cell, (0, 0))
        self.assertEqual(explicit.destination_cell, (1, 0))
        self.assertEqual(explicit.displacement, (1, 0))
        kinds = {
            transition.outcome_signature: transition.kind
            for transition in restored.transitions()
        }
        self.assertEqual(kinds[inert.signature], "no_effect")

    def test_removal_round_trip_preserves_digest_and_provenance(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=1)
        appearance = (1, 1, 1, 1)
        expelled = model.effect_descriptor(
            appearance,
            (200, 200, 200, 200),
            appearance,
            entity_removed=True,
            removal_persistence_verified=True,
            removal_transit_cells=((1, 0), (2, 0)),
        )
        model.observe(
            appearance,
            Action.B,
            16,
            expelled.signature,
            outcome_descriptor=expelled,
            evidence_id="expulsion-branch",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anonymous-behavior.json"
            model.save(path)
            restored = AnonymousEntityBehaviorModel.load(path)

        self.assertEqual(restored.digest, model.digest)
        self.assertEqual(
            restored.predict(appearance, Action.B, 16),
            model.predict(appearance, Action.B, 16),
        )
        restored_descriptor = restored.predict(
            appearance, Action.B, 16
        ).outcome_descriptor
        self.assertEqual(restored_descriptor, expelled)
        self.assertEqual(restored_descriptor.transition_kind, "expulsion")
        replay = restored.observe(
            appearance,
            Action.B,
            16,
            expelled.signature,
            outcome_descriptor=expelled,
            evidence_id="expulsion-branch",
        )
        self.assertFalse(replay.accepted)
        self.assertEqual(restored.observation_count, 1)

    def test_frozen_style_queries_do_not_mutate_checkpoint(self) -> None:
        model = AnonymousEntityBehaviorModel(minimum_prediction_samples=1)
        appearance = (1, 1, 1, 1)
        expelled = model.effect_descriptor(
            appearance,
            (200, 200, 200, 200),
            appearance,
            entity_removed=True,
            removal_persistence_verified=True,
            removal_transit_cells=((1, 0),),
        )
        model.observe(
            appearance,
            Action.B,
            16,
            expelled.signature,
            outcome_descriptor=expelled,
        )
        restored = AnonymousEntityBehaviorModel.from_dict(model.to_dict())
        digest = restored.digest

        restored.predict((9, 9, 9, 9), Action.B, 16)
        restored.predict(appearance, Action.B, 16)
        restored.transitions()
        restored.transition_for(expelled.signature)

        self.assertEqual(restored.digest, digest)
        self.assertEqual(restored.type_count, 1)
        self.assertEqual(restored.outcome_descriptor_count, 1)

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
