import json
import tempfile
import unittest

try:  # optional ML extra (imported transitively by the modules below)
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise unittest.SkipTest(
        "requires the optional 'ml' extra: pip install -e '.[ml]'"
    ) from exc
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
            VisualSequence(0, (first, second), (Action.RIGHT,), (8,), "run-a"),
            VisualSequence(0, (first, second), (Action.A,), (2,), "run-a"),
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

    def test_transition_metadata_and_frame_subset_avoid_full_decode(self) -> None:
        first = Frame(2, 2, 1, b"\x01\x02\x03\x04")
        second = Frame(2, 2, 1, b"\x05\x06\x07\x08")
        third = Frame(2, 2, 1, b"\x09\x0a\x0b\x0c")
        sequence = VisualSequence(
            0,
            (first, second, third),
            (Action.RIGHT, Action.DOWN),
            (8, 4),
            "run-a",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceStore(Path(directory))
            store.append_segment("cycle-000001", [sequence])
            transitions = store.transition_metadata()
            frames = store.load_frame_subset((first.digest, third.digest))
        self.assertEqual(
            [(item.action, item.duration) for item in transitions],
            [(Action.RIGHT, 8), (Action.DOWN, 4)],
        )
        self.assertTrue(all(item.source_run_id == "run-a" for item in transitions))
        self.assertEqual(frames, {first.digest: first, third.digest: third})

    def test_legacy_segment_receives_conservative_run_provenance(self) -> None:
        frame = Frame(1, 1, 1, b"\x00")
        sequence = VisualSequence(
            0,
            (frame, frame),
            (Action.NOOP,),
            (1,),
            "new-run-id",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceStore(Path(directory))
            segment = store.append_segment("cycle-000001", [sequence])
            record = json.loads(segment.read_text(encoding="utf-8"))
            record["version"] = 1
            record.pop("source_run_id")
            segment.write_text(json.dumps(record) + "\n", encoding="utf-8")
            restored = store.load()
        self.assertEqual(
            restored[0].source_run_id,
            "legacy-segment:cycle-000001",
        )

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

    def test_group_sample_does_not_merge_restarted_counters_across_runs(self) -> None:
        frame = Frame(1, 1, 1, b"\x00")
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceStore(Path(directory))
            for segment, run_id in (("cycle-000001", "run-a"), ("cycle-000002", "run-b")):
                store.append_segment(
                    segment,
                    [
                        VisualSequence(
                            0,
                            (frame, frame),
                            (action,),
                            (1,),
                            run_id,
                        )
                        for action in (Action.NOOP, Action.RIGHT)
                    ],
                )
            sampled = store.load_group_sample(1, seed=5)
        self.assertEqual(len(sampled), 2)
        self.assertEqual(len({item.source_run_id for item in sampled}), 1)

    def test_group_sample_can_reserve_complete_multistep_groups(self) -> None:
        frame = Frame(1, 1, 1, b"\x00")
        sequences = []
        for group in range(5):
            sequences.append(
                VisualSequence(
                    group,
                    (frame, frame),
                    (Action.NOOP,),
                    (1,),
                    "run-a",
                )
            )
        for group in (3, 4):
            sequences.append(
                VisualSequence(
                    group,
                    (frame, frame, frame),
                    (Action.RIGHT, Action.NOOP),
                    (1, 1),
                    "run-a",
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            store = SequenceStore(Path(directory))
            store.append_segment("cycle-000001", sequences)
            sampled = store.load_group_sample(
                2,
                seed=5,
                minimum_multistep_groups=2,
            )
        self.assertEqual(
            {(item.source_run_id, item.group) for item in sampled},
            {("run-a", 3), ("run-a", 4)},
        )
        self.assertEqual(sum(len(item.actions) > 1 for item in sampled), 2)


if __name__ == "__main__":
    unittest.main()
