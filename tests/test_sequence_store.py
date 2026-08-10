import tempfile
import unittest
from pathlib import Path

from lolo_agent.ensemble_world_model import VisualSequence
from lolo_agent.environment import Action
from lolo_agent.pixels import Frame
from lolo_agent.sequence_store import SequenceStore


class SequenceStoreTests(unittest.TestCase):
    def test_segment_round_trip_and_frame_deduplication(self) -> None:
        first = Frame(2, 2, 1, b"\x01\x02\x03\x04")
        second = Frame(2, 2, 1, b"\x05\x06\x07\x08")
        sequences = [
            VisualSequence(0, (first, second), (Action.RIGHT,), (8,)),
            VisualSequence(0, (first, second), (Action.A,), (2,)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceStore(Path(directory))
            store.append_segment("cycle-000001", sequences)
            restored = store.load()
            statistics = store.statistics()
        self.assertEqual(restored, sequences)
        self.assertEqual(statistics["segments"], 1)
        self.assertEqual(statistics["sequences"], 2)
        self.assertEqual(statistics["unique_frames"], 2)

    def test_segment_creation_is_idempotence_safe(self) -> None:
        frame = Frame(1, 1, 1, b"\x00")
        sequence = VisualSequence(0, (frame, frame), (Action.NOOP,), (1,))
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceStore(Path(directory))
            store.append_segment("cycle-000001", [sequence])
            with self.assertRaises(FileExistsError):
                store.append_segment("cycle-000001", [sequence])

    def test_sample_decodes_only_a_deterministic_subset(self) -> None:
        frame = Frame(1, 1, 1, b"\x00")
        sequences = [
            VisualSequence(index, (frame, frame), (Action.NOOP,), (1,))
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceStore(Path(directory))
            store.append_segment("cycle-000001", sequences)
            first = store.load_sample(5, seed=11)
            second = store.load_sample(5, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertGreater(len({item.group for item in first}), 1)

    def test_reward_track_binding_prevents_dataset_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceStore(Path(directory))
            self.assertIsNone(store.reward_track)
            store.bind_reward_track("strict")
            store.bind_reward_track("strict")
            self.assertEqual(store.reward_track, "strict")
            with self.assertRaisesRegex(ValueError, "bound to the 'strict'"):
                store.bind_reward_track("assisted")

    def test_group_sample_keeps_counterfactual_branches_together(self) -> None:
        frame = Frame(1, 1, 1, b"\x00")
        sequences = []
        for group in range(10):
            sequences.extend(
                (
                    VisualSequence(group, (frame, frame), (Action.NOOP,), (1,)),
                    VisualSequence(group, (frame, frame), (Action.RIGHT,), (1,)),
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceStore(Path(directory))
            store.append_segment("cycle-000001", sequences)
            sampled = store.load_group_sample(3, seed=5)
        groups = {item.group for item in sampled}
        self.assertEqual(len(groups), 3)
        self.assertEqual(len(sampled), 6)
        for group in groups:
            self.assertEqual(sum(item.group == group for item in sampled), 2)


if __name__ == "__main__":
    unittest.main()
