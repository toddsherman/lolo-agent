import unittest

try:  # optional ML extra
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise unittest.SkipTest(
        "requires the optional 'ml' extra: pip install -e '.[ml]'"
    ) from exc


from lolo_agent.environment import Action
from lolo_agent.neural_world_model import (
    VisualDynamicsModel,
    VisualTransition,
    frame_tensor,
    train_world_model,
)
from lolo_agent.pixels import Frame


class NeuralWorldModelTests(unittest.TestCase):
    def make_frame(self, offset: int) -> Frame:
        pixels = bytes((index + offset) % 256 for index in range(64 * 64 * 3))
        return Frame(64, 64, 3, pixels)

    def test_frame_preprocessing_and_model_shapes(self) -> None:
        frame = self.make_frame(0)
        tensor = frame_tensor(frame)
        self.assertEqual(tuple(tensor.shape), (3, 128, 128))
        model = VisualDynamicsModel(latent_size=32, action_size=8)
        source = tensor.unsqueeze(0)
        actions = torch.tensor([0], dtype=torch.long)
        reconstructed, predicted, latent = model(source, actions)
        self.assertEqual(tuple(reconstructed.shape), (1, 3, 128, 128))
        self.assertEqual(tuple(predicted.shape), (1, 3, 128, 128))
        self.assertEqual(tuple(latent.shape), (1, 32))

    def test_training_changes_parameters_and_freeze_preserves_digest(self) -> None:
        transitions = [
            VisualTransition(self.make_frame(index), Action.RIGHT, self.make_frame(index + 1))
            for index in range(2)
        ]
        model = VisualDynamicsModel(latent_size=32, action_size=8)
        before = model.checkpoint_digest
        history = train_world_model(
            model, transitions, device="cpu", epochs=1, batch_size=2, seed=1
        )
        after = model.checkpoint_digest
        self.assertTrue(history)
        self.assertNotEqual(before, after)
        model.freeze()
        self.assertEqual(after, model.checkpoint_digest)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

