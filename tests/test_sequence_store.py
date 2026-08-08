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


if __name__ == "__main__":
    unittest.main()
