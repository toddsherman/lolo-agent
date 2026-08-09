import json
import tempfile
import unittest
from pathlib import Path

from lolo_agent.experience_import import (
    ExperienceSource,
    decode_logged_png,
    extract_experience,
)
from lolo_agent.pixels import Frame
from lolo_agent.run_logging import encode_png


class ExperienceImportTests(unittest.TestCase):
    def test_logged_png_round_trip(self) -> None:
        frame = Frame(3, 2, 3, bytes(range(18)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            path.write_bytes(encode_png(frame))
            self.assertEqual(decode_logged_png(path), frame)

    def test_extracts_verified_facts_without_scores_or_annotations(self) -> None:
        first = Frame(2, 2, 3, bytes(range(12)))
        second = Frame(2, 2, 3, bytes(range(1, 13)))
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            frames = run / "frames"
            frames.mkdir()
            for frame in (first, second):
                (frames / f"{frame.digest}.png").write_bytes(encode_png(frame))
            (run / "manifest.json").write_text(
                json.dumps({"run_id": "test-run"}), encoding="utf-8"
            )
            events = [
                {
                    "event": "level_annotation",
                    "seq": 1,
                    "label": "forbidden evaluator label",
                },
                {
                    "event": "env_step",
                    "seq": 2,
                    "phase": "agent",
                    "action": "right",
                    "action_frames": 8,
                    "source_frame": first.digest,
                    "target_frame": second.digest,
                },
                {
                    "event": "branch_verified",
                    "seq": 3,
                    "decision": 1,
                    "env_step_seq": 2,
                    "state_id": "state-1",
                    "combined_score": 999.0,
                },
            ]
            (run / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            sequences, metadata = extract_experience(ExperienceSource(run), 10)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].group, 10)
        self.assertEqual(sequences[0].durations, (8,))
        self.assertEqual(metadata["verified_transitions"], 1)
        self.assertNotIn("combined_score", metadata)


if __name__ == "__main__":
    unittest.main()
