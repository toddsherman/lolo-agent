import tempfile
import unittest
from pathlib import Path

import torch

from lolo_agent.ensemble_world_model import VisualSequence
from lolo_agent.environment import Action
from lolo_agent.neural_world_model import ACTION_TO_INDEX, frame_tensor
from lolo_agent.pixels import Frame
from lolo_agent.spatial_world_model import (
    SpatialTokenDynamicsModel,
    causal_dataset_statistics,
    load_spatial_checkpoint,
    save_spatial_checkpoint,
    spatial_effect_target,
    train_spatial_model,
    validate_spatial_model,
)


class SpatialWorldModelTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
