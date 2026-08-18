import tempfile
import unittest

try:  # optional ML extra
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise unittest.SkipTest(
        "requires the optional 'ml' extra: pip install -e '.[ml]'"
    ) from exc
from pathlib import Path


from lolo_agent.ensemble_world_model import VisualSequence, split_sequence_runs
from lolo_agent.environment import Action
from lolo_agent.neural_world_model import ACTION_TO_INDEX, frame_tensor
from lolo_agent.pixels import Frame
from lolo_agent.spatial_returnability import SpatialReturnabilityModel
from lolo_agent.spatial_shadow import SpatialShadowEvaluator
from lolo_agent.spatial_world_model import (
    SpatialTokenDynamicsModel,
    causal_dataset_statistics,
    load_spatial_checkpoint,
    save_spatial_checkpoint,
    spatial_effect_target,
    spatial_sequence_loss,
    train_spatial_model,
    validate_spatial_model,
)


class SpatialWorldModelTests(unittest.TestCase):
    def test_run_split_never_leaks_source_provenance(self) -> None:
        frame = self.make_frame(0, 0, (255, 255, 255))
        sequences = [
            VisualSequence(
                group,
                (frame, frame),
                (Action.NOOP,),
                (1,),
                f"run-{group % 4}",
            )
            for group in range(20)
        ]
        training, validation = split_sequence_runs(sequences, validation_modulus=3)
        training_runs = {item.source_run_id for item in training}
        validation_runs = {item.source_run_id for item in validation}
        self.assertTrue(training_runs)
        self.assertTrue(validation_runs)
        self.assertFalse(training_runs & validation_runs)

    def test_run_split_balances_uneven_source_run_sizes(self) -> None:
        frame = self.make_frame(0, 0, (255, 255, 255))
        sequences = []
        for run_id, count in (("large", 60), ("medium", 25), ("small", 10), ("tiny", 5)):
            sequences.extend(
                VisualSequence(
                    group,
                    (frame, frame),
                    (Action.NOOP,),
                    (1,),
                    run_id,
                )
                for group in range(count)
            )
        training, validation = split_sequence_runs(
            sequences, validation_modulus=5
        )
        self.assertGreaterEqual(len(validation), 15)
        self.assertLessEqual(len(validation), 30)
        self.assertFalse(
            {item.source_run_id for item in training}
            & {item.source_run_id for item in validation}
        )

    def make_frame(self, x: int, y: int, color: tuple[int, int, int]) -> Frame:
        width = 32
        height = 32
        pixels = bytearray(width * height * 3)
        for row in range(y, min(y + 6, height)):
            for column in range(x, min(x + 6, width)):
                offset = (row * width + column) * 3
                pixels[offset : offset + 3] = bytes(color)
        return Frame(width, height, 3, bytes(pixels))

    def make_sequences(self) -> list[VisualSequence]:
        sequences = []
        for group, y in enumerate((2, 8, 14, 20), 1):
            source = self.make_frame(4, y, (255, 80, 40))
            target = self.make_frame(10, y, (255, 80, 40))
            sequences.append(
                VisualSequence(group, (source, target), (Action.RIGHT,), (4,))
            )
        return sequences

    def test_spatial_effect_target_is_unlabeled_and_localized(self) -> None:
        source = frame_tensor(self.make_frame(2, 2, (255, 0, 0))).unsqueeze(0)
        target = frame_tensor(self.make_frame(20, 20, (255, 0, 0))).unsqueeze(0)
        effect = spatial_effect_target(source, target, grid_size=8, effect_scale=0.05)
        self.assertEqual(tuple(effect.shape), (1, 8, 8))
        self.assertGreater(float(effect.max()), 0.5)
        self.assertGreater(int((effect > 0.2).sum()), 1)
        self.assertLess(int((effect > 0.2).sum()), 32)

    def test_causal_statistics_require_shared_pixel_roots(self) -> None:
        source = self.make_frame(4, 4, (255, 255, 255))
        target = self.make_frame(10, 4, (255, 255, 255))
        sequences = [
            VisualSequence(1, (source, source), (Action.NOOP,), (4,)),
            VisualSequence(1, (source, target), (Action.RIGHT,), (4,)),
            VisualSequence(2, (source, target), (Action.RIGHT,), (4,)),
        ]
        statistics = causal_dataset_statistics(sequences)
        self.assertEqual(statistics["causal_roots"], 2)
        self.assertEqual(statistics["counterfactual_roots"], 1)
        self.assertEqual(statistics["noop_control_roots"], 1)

    def test_rollout_preserves_spatial_tokens_and_predicts_effect_map(self) -> None:
        model = SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=4,
            duration_size=4,
        )
        source = frame_tensor(self.make_frame(4, 4, (255, 255, 255))).unsqueeze(0)
        actions = torch.tensor(
            [[ACTION_TO_INDEX[Action.RIGHT], ACTION_TO_INDEX[Action.NOOP]]]
        )
        durations = torch.tensor([[4, 2]])
        pixels, tokens, uncertainty, effects = model.rollout(
            source, actions, durations
        )
        self.assertEqual(tuple(pixels.shape), (1, 2, 3, 128, 128))
        self.assertEqual(tuple(tokens.shape), (1, 2, 8, 4, 4))
        self.assertEqual(tuple(uncertainty.shape), (1, 2))
        self.assertEqual(tuple(effects.shape), (1, 2, 4, 4))

    def test_flow_residual_renderer_starts_as_exact_persistence(self) -> None:
        model = SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=4,
            duration_size=4,
            renderer_kind="flow_residual",
        )
        source = frame_tensor(self.make_frame(4, 4, (255, 255, 255))).unsqueeze(0)
        tokens = model.encode(source)
        effect_logits = torch.ones((1, 4, 4)) * 10.0
        rendered = model.render_successor(source, tokens, tokens, effect_logits)
        self.assertLess(float((rendered - source).abs().max().detach()), 1e-5)

    def test_changed_patch_renderer_starts_as_exact_persistence(self) -> None:
        model = SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=4,
            duration_size=4,
            renderer_kind="changed_patch",
            residual_scale=1.0,
        )
        source = frame_tensor(self.make_frame(4, 4, (255, 255, 255))).unsqueeze(0)
        tokens = model.encode(source)
        effect_logits = torch.ones((1, 4, 4)) * 10.0
        rendered = model.render_successor(source, tokens, tokens, effect_logits)
        self.assertLess(float((rendered - source).abs().max().detach()), 1e-5)

    def test_anchored_rollout_predicts_each_endpoint_from_the_initial_frame(self) -> None:
        model = SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=4,
            duration_size=4,
            renderer_kind="changed_patch",
            renderer_rollout="anchored",
            residual_scale=1.0,
        )
        source = frame_tensor(self.make_frame(4, 4, (255, 255, 255))).unsqueeze(0)
        actions = torch.tensor(
            [[ACTION_TO_INDEX[Action.RIGHT], ACTION_TO_INDEX[Action.RIGHT]]]
        )
        durations = torch.tensor([[4, 4]])
        pixels, _tokens, _uncertainty, _effects = model.rollout(
            source, actions, durations
        )
        self.assertEqual(tuple(pixels.shape), (1, 2, 3, 128, 128))
        self.assertTrue(torch.isfinite(pixels).all())

    def test_shadow_evaluator_reports_plans_and_real_transition_error(self) -> None:
        model = SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=4,
            duration_size=4,
        )
        returnability = SpatialReturnabilityModel(
            token_size=8, hidden_size=8, ensemble_size=2
        )
        evaluator = SpatialShadowEvaluator(
            model, "cpu", returnability_model=returnability
        )
        observed_evaluator = SpatialShadowEvaluator(
            model,
            "cpu",
            returnability_model=returnability,
            returnability_observed_endpoints=True,
        )
        source = self.make_frame(4, 4, (255, 255, 255))
        target = self.make_frame(10, 4, (255, 255, 255))
        plans = evaluator.score_plans(
            source,
            [
                ((Action.RIGHT,), (4,)),
                ((Action.NOOP,), (4,)),
            ],
        )
        transition = evaluator.evaluate_transition(
            source, Action.RIGHT, 4, target
        )
        observed_transition = observed_evaluator.evaluate_transition(
            source, Action.RIGHT, 4, target
        )
        self.assertEqual(len(plans), 2)
        self.assertIn("spatial_shadow_score", plans[0])
        self.assertIn("spatial_shadow_usefulness_score", plans[0])
        self.assertIn("spatial_shadow_raw_activity_score", plans[0])
        self.assertIn("spatial_shadow_predicted_causal_change", plans[0])
        self.assertIn("spatial_shadow_predicted_causal_effect", plans[0])
        self.assertIn("spatial_shadow_predicted_change", plans[0])
        self.assertIn("spatial_shadow_predicted_returnability", plans[0])
        self.assertIn("spatial_shadow_returnability_uncertainty", plans[0])
        self.assertIn("spatial_shadow_effect_f1", transition)
        self.assertIn("spatial_shadow_beats_persistence", transition)
        self.assertIn("spatial_shadow_predicted_pixel_change", transition)
        self.assertIn("spatial_shadow_usefulness_score", transition)
        self.assertIn("spatial_shadow_predicted_returnability", transition)
        self.assertIn(
            "spatial_shadow_predicted_returnability", observed_transition
        )
        self.assertAlmostEqual(plans[1]["spatial_shadow_score"], 0.0, places=6)
        self.assertGreaterEqual(transition["spatial_shadow_effect_f1"], 0.0)
        with self.assertRaisesRegex(ValueError, "relation model"):
            SpatialShadowEvaluator(
                model, "cpu", returnability_observed_endpoints=True
            )

    def test_legacy_checkpoint_uses_the_original_blend_renderer(self) -> None:
        model = SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=4,
            duration_size=4,
            renderer_kind="blend",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-spatial.pt"
            save_spatial_checkpoint(model, path, planning_horizon=2)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint["version"] = 5
            checkpoint.pop("renderer_kind")
            checkpoint.pop("max_flow_pixels")
            checkpoint.pop("residual_scale")
            torch.save(checkpoint, path)
            loaded, _horizon = load_spatial_checkpoint(path, frozen=True)
        self.assertEqual(loaded.renderer_kind, "blend")

    def test_training_validation_and_checkpoint_round_trip(self) -> None:
        torch.manual_seed(3)
        sequences = self.make_sequences()
        model = SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=4,
            duration_size=4,
        )
        before = model.checkpoint_digest
        history = train_spatial_model(
            model, sequences[:3], "cpu", epochs=1, batch_size=3, seed=3
        )
        after = model.checkpoint_digest
        report = validate_spatial_model(model, sequences[3:], "cpu", batch_size=1)
        self.assertTrue(history)
        self.assertNotEqual(before, after)
        self.assertEqual(len(report.horizon_pixel_l1), 1)
        self.assertEqual(len(report.horizon_persistence_pixel_l1), 1)
        self.assertEqual(len(report.horizon_predicted_pixel_change), 1)
        self.assertEqual(len(report.horizon_actual_pixel_change), 1)
        self.assertEqual(len(report.horizon_effect_weighted_pixel_l1), 1)
        self.assertEqual(len(report.horizon_effect_weighted_persistence_l1), 1)
        self.assertEqual(len(report.horizon_effect_l1), 1)
        self.assertEqual(len(report.horizon_zero_effect_l1), 1)
        self.assertEqual(len(report.horizon_balanced_effect_l1), 1)
        self.assertEqual(len(report.horizon_zero_balanced_effect_l1), 1)
        self.assertEqual(len(report.horizon_effect_f1), 1)
        self.assertEqual(len(report.horizon_effect_prevalence), 1)
        self.assertEqual(len(report.horizon_uncertainty_effect_error_correlation), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial.pt"
            digest = save_spatial_checkpoint(model, path, planning_horizon=3)
            loaded, horizon = load_spatial_checkpoint(path, frozen=True)
        self.assertEqual(digest, loaded.checkpoint_digest)
        self.assertEqual(horizon, 3)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))

    def test_spatial_loss_weights_are_explicit_and_validated(self) -> None:
        model = SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=4,
            duration_size=4,
        )
        source = frame_tensor(self.make_frame(4, 4, (255, 255, 255)))
        target = frame_tensor(self.make_frame(10, 4, (255, 255, 255)))
        frames = torch.stack((source, target)).unsqueeze(0)
        actions = torch.tensor([[ACTION_TO_INDEX[Action.RIGHT]]])
        durations = torch.tensor([[4]])
        pixel_only, metrics = spatial_sequence_loss(
            model,
            frames,
            actions,
            durations,
            reconstruction_weight=0.0,
            pixel_weight=5.0,
            token_weight=0.0,
            effect_weight=0.0,
        )
        self.assertAlmostEqual(
            float(pixel_only.detach()),
            5.0 * metrics.pixel_prediction,
            places=6,
        )
        changed_only, changed_metrics = spatial_sequence_loss(
            model,
            frames,
            actions,
            durations,
            reconstruction_weight=0.0,
            pixel_weight=0.0,
            changed_region_weight=1.0,
            token_weight=0.0,
            effect_weight=0.0,
        )
        self.assertGreater(changed_metrics.changed_region_prediction, 0.0)
        self.assertAlmostEqual(
            float(changed_only.detach()),
            changed_metrics.changed_region_prediction,
            places=6,
        )
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            spatial_sequence_loss(
                model,
                frames,
                actions,
                durations,
                reconstruction_weight=0.0,
                pixel_weight=0.0,
                token_weight=0.0,
                effect_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
