import json
import tempfile
import unittest
from pathlib import Path

from lolo_agent.experience_import import (
    ExperienceSource,
    classify_reward_track,
    decode_logged_png,
    extract_experience,
)
from lolo_agent.neural_run import derive_reward_track
from lolo_agent.pixels import Frame
from lolo_agent.run_logging import encode_png


class ExperienceImportTests(unittest.TestCase):
    def test_reward_track_classification_keeps_assistance_explicit(self) -> None:
        self.assertEqual(classify_reward_track({}), "strict")
        self.assertEqual(
            classify_reward_track(
                {"metadata": {"reward_track": "strict_rule_free"}}
            ),
            "strict",
        )
        self.assertEqual(
            classify_reward_track({"metadata": {"reward_track": "human_prior_v2"}}),
            "assisted",
        )
        self.assertEqual(
            classify_reward_track(
                {
                    "metadata": {
                        "reward_track": "human_prior_resume_observational"
                    }
                }
            ),
            "assisted",
        )
        with self.assertRaisesRegex(ValueError, "unrecognized telemetry reward track"):
            classify_reward_track({"metadata": {"reward_track": "mystery"}})

    def test_strict_from_assisted_state_is_strict_importable(self) -> None:
        # Ratified 2026-08-16: strict-policy collection branched from an
        # assisted-era save state enters the strict store under a distinct,
        # disclosed track value.
        self.assertEqual(
            classify_reward_track(
                {"metadata": {"reward_track": "strict_from_assisted_state"}}
            ),
            "strict",
        )
        # Legacy manifests are not reclassified: the retired value keeps
        # its assisted classification.
        self.assertEqual(
            classify_reward_track(
                {
                    "metadata": {
                        "reward_track": "human_prior_resume_observational"
                    }
                }
            ),
            "assisted",
        )


class RewardTrackDerivationTests(unittest.TestCase):
    def test_strict_policy_with_strict_ancestry_is_rule_free(self) -> None:
        self.assertEqual(
            derive_reward_track(False, None, None), "strict_rule_free"
        )
        self.assertEqual(
            derive_reward_track(False, "strict", None), "strict_rule_free"
        )
        self.assertEqual(
            derive_reward_track(False, "strict", "strict"),
            "strict_rule_free",
        )

    def test_strict_policy_with_assisted_memory_ancestry_is_disclosed(
        self,
    ) -> None:
        self.assertEqual(
            derive_reward_track(False, "assisted", None),
            "strict_from_assisted_state",
        )
        self.assertEqual(
            derive_reward_track(False, "assisted", "strict"),
            "strict_from_assisted_state",
        )

    def test_assisted_state_source_ancestry_cannot_be_laundered(
        self,
    ) -> None:
        # The --resume-state-run track is consulted even when the memory
        # source is strict: the laundering loophole is closed.
        self.assertEqual(
            derive_reward_track(False, "strict", "assisted"),
            "strict_from_assisted_state",
        )
        self.assertEqual(
            derive_reward_track(False, None, "assisted"),
            "strict_from_assisted_state",
        )

    def test_assisted_policy_is_human_prior_regardless_of_ancestry(
        self,
    ) -> None:
        for memory_track in (None, "strict", "assisted"):
            for state_track in (None, "strict", "assisted"):
                self.assertEqual(
                    derive_reward_track(True, memory_track, state_track),
                    "human_prior_v2",
                )

    def test_derived_tracks_round_trip_through_classification(self) -> None:
        for memory_track, state_track, expected in (
            (None, None, "strict"),
            ("assisted", None, "strict"),
            ("strict", "assisted", "strict"),
        ):
            track = derive_reward_track(False, memory_track, state_track)
            self.assertEqual(
                classify_reward_track({"metadata": {"reward_track": track}}),
                expected,
            )
        self.assertEqual(
            classify_reward_track(
                {
                    "metadata": {
                        "reward_track": derive_reward_track(
                            True, "assisted", "assisted"
                        )
                    }
                }
            ),
            "assisted",
        )

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
        self.assertEqual(sequences[0].source_run_id, "test-run")
        self.assertEqual(metadata["verified_transitions"], 1)
        self.assertEqual(metadata["reward_track"], "strict")
        self.assertNotIn("combined_score", metadata)


if __name__ == "__main__":
    unittest.main()
