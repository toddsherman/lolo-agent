import tempfile
import unittest
from pathlib import Path

import torch

from lolo_agent.environment import Action
from lolo_agent.pixels import Frame
from lolo_agent.sequence_store import StoredTransition
from lolo_agent.spatial_returnability import (
    ReturnabilityExample,
    SpatialReturnabilityModel,
    balanced_returnability_sample,
    build_returnability_specs,
    load_returnability_checkpoint,
    save_returnability_checkpoint,
    split_returnability_runs,
    train_returnability_model,
    validate_returnability_model,
)
from lolo_agent.spatial_world_model import SpatialTokenDynamicsModel


class SpatialReturnabilityTests(unittest.TestCase):
    def test_graph_labels_observed_returns_and_censors_unknown_edges(self) -> None:
        transitions = [
            StoredTransition("a", "b", Action.RIGHT, 4, "run-a"),
            StoredTransition("b", "a", Action.LEFT, 4, "run-a"),
            StoredTransition("c", "d", Action.RIGHT, 4, "run-a"),
        ]
        for action, target in zip(
            (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.A),
            ("u", "v", "w", "x", "y"),
        ):
            transitions.append(StoredTransition("d", target, action, 4, "run-a"))
        specs, statistics = build_returnability_specs(
            transitions, maximum_return_steps=1, minimum_endpoint_actions=5
        )
        labels = {
            (item.source_digest, item.target_digest): item.label for item in specs
        }
        self.assertEqual(labels[("a", "b")], 1)
        self.assertEqual(labels[("b", "a")], 1)
        self.assertEqual(labels[("c", "d")], 0)
        self.assertNotIn(("d", "u"), labels)
        self.assertGreater(statistics["censored_edges"], 0)

    def test_run_split_and_sampling_preserve_balanced_labels(self) -> None:
        transitions = []
        for run_index in range(4):
            run_id = f"run-{run_index}"
            transitions.extend(
                [
                    StoredTransition("a", "b", Action.RIGHT, 4, run_id),
                    StoredTransition("b", "a", Action.LEFT, 4, run_id),
                    StoredTransition("c", "d", Action.RIGHT, 4, run_id),
                ]
            )
            for action, target in zip(
                (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.A),
                ("u", "v", "w", "x", "y"),
            ):
                transitions.append(
                    StoredTransition("d", target, action, 4, run_id)
                )
        specs, _ = build_returnability_specs(transitions, 1, 5)
        training, validation = split_returnability_runs(specs, 2)
        self.assertFalse(
            {item.source_run_id for item in training}
            & {item.source_run_id for item in validation}
        )
        sample = balanced_returnability_sample(training, 4, seed=3)
        self.assertEqual(sum(item.label for item in sample), len(sample) // 2)

    def test_relation_head_trains_validates_and_round_trips(self) -> None:
        torch.manual_seed(2)
        spatial = SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=4,
            duration_size=2,
            renderer_kind="changed_patch",
        )
        relation = SpatialReturnabilityModel(8, hidden_size=8, ensemble_size=2)

        def frame(value: int) -> Frame:
            return Frame(32, 32, 3, bytes([value]) * (32 * 32 * 3))

        examples = [
            ReturnabilityExample(
                frame(20 + index * 5),
                f"target-{index}",
                Action.RIGHT if index % 2 else Action.LEFT,
                4,
                "run-a",
                index % 2,
            )
            for index in range(8)
        ]
        history = train_returnability_model(
            relation,
            spatial,
            examples,
            "cpu",
            epochs=1,
            batch_size=4,
            seed=4,
        )
        report = validate_returnability_model(
            relation, spatial, examples, "cpu", batch_size=4
        )
        self.assertEqual(len(history), 2)
        self.assertEqual(report.examples, 8)
        self.assertGreaterEqual(report.roc_auc, 0.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "returnability.pt"
            digest = save_returnability_checkpoint(
                relation, checkpoint, spatial.checkpoint_digest, 3, 5
            )
            restored, configuration = load_returnability_checkpoint(
                checkpoint, spatial.checkpoint_digest
            )
            self.assertEqual(restored.checkpoint_digest, digest)
            self.assertEqual(configuration["maximum_return_steps"], 3)
            with self.assertRaisesRegex(ValueError, "another spatial model"):
                load_returnability_checkpoint(checkpoint, "wrong-digest")

    def test_explicit_probe_training_uses_observed_endpoint_pixels(self) -> None:
        torch.manual_seed(7)
        spatial = SpatialTokenDynamicsModel(
            token_size=8,
            action_size=4,
            ensemble_size=2,
            grid_size=4,
            duration_size=2,
            renderer_kind="changed_patch",
        )
        relation = SpatialReturnabilityModel(8, hidden_size=8, ensemble_size=2)

        def frame(value: int) -> Frame:
            return Frame(32, 32, 3, bytes([value]) * (32 * 32 * 3))

        examples = []
        for index in range(4):
            source = frame(20 + index * 10)
            target = frame(25 + index * 10)
            examples.append(
                ReturnabilityExample(
                    source,
                    target.digest,
                    Action.RIGHT,
                    4,
                    "probe-run",
                    index % 2,
                    target,
                )
            )
        history = train_returnability_model(
            relation,
            spatial,
            examples,
            "cpu",
            epochs=1,
            batch_size=2,
            seed=8,
            use_observed_targets=True,
        )
        report = validate_returnability_model(
            relation,
            spatial,
            examples,
            "cpu",
            batch_size=2,
            use_observed_targets=True,
        )

        self.assertEqual(len(history), 2)
        self.assertEqual(report.examples, 4)
        missing_target = [
            ReturnabilityExample(
                frame(1), "missing", Action.LEFT, 4, "probe-run", 0
            )
        ]
        with self.assertRaisesRegex(ValueError, "target pixel frames"):
            validate_returnability_model(
                relation,
                spatial,
                missing_target,
                "cpu",
                use_observed_targets=True,
            )


if __name__ == "__main__":
    unittest.main()
