import json
import tempfile
import unittest
from pathlib import Path

from lolo_agent.neural_run import (
    StableSceneChangeDetector,
    load_episodic_decision_events,
)
from lolo_agent.pixels import Frame


class StableSceneChangeDetectorTests(unittest.TestCase):
    def frame(self, value: int) -> Frame:
        return Frame(6, 6, 3, bytes([value]) * 108)

    def test_requires_a_distinct_scene_to_remain_stable(self) -> None:
        initial = self.frame(96)
        detector = StableSceneChangeDetector(
            initial,
            stable_observations=2,
            warmup_decisions=1,
            minimum_difference=0.1,
        )
        self.assertIsNone(detector.observe(1, initial))
        self.assertIsNone(detector.observe(2, self.frame(0)))
        self.assertIsNone(detector.observe(3, self.frame(255)))
        result = detector.observe(4, self.frame(255))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["decision"], 4)
        self.assertEqual(result["stable_observations"], 2)
        self.assertTrue(result["dark_transition_observed"])

    def test_does_not_treat_a_stable_in_room_change_as_a_transition(self) -> None:
        initial = self.frame(96)
        detector = StableSceneChangeDetector(
            initial,
            stable_observations=2,
            warmup_decisions=1,
            minimum_difference=0.1,
        )
        self.assertIsNone(detector.observe(1, initial))
        self.assertIsNone(detector.observe(2, self.frame(255)))
        self.assertIsNone(detector.observe(3, self.frame(255)))

    def test_loads_recursive_episodic_decision_events_to_boundaries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            child = root / "child"
            parent.mkdir()
            child.mkdir()
            (parent / "manifest.json").write_text(
                json.dumps({"metadata": {}}), encoding="utf-8"
            )
            (parent / "events.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "event": "decision_committed",
                            "decision": decision,
                            "marker": f"parent-{decision}",
                        }
                    )
                    for decision in (1, 2)
                )
                + "\n",
                encoding="utf-8",
            )
            (child / "manifest.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "episodic_resume": {
                                "source_run": str(parent),
                                "source_decision": 1,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (child / "events.jsonl").write_text(
                "\n".join(
                    json.dumps(event)
                    for event in (
                        {
                            "event": "decision_committed",
                            "decision": 1,
                            "marker": "child-1",
                        },
                        {
                            "event": "pixel_novel_room_started",
                            "decision": 1,
                            "marker": "child-room-boundary",
                        },
                        {
                            "event": "goal_milestone_exhaustion_learned",
                            "decision": 1,
                            "marker": "child-hazard",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            events = load_episodic_decision_events(child, 1)

        self.assertEqual(
            [event["marker"] for event in events],
            [
                "parent-1",
                "child-1",
                "child-room-boundary",
                "child-hazard",
            ],
        )


if __name__ == "__main__":
    unittest.main()
