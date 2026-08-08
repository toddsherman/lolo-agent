import json
import tempfile
import unittest
from pathlib import Path

from lolo_agent.pixels import Frame
from lolo_agent.replay import ReplayCapture, committed_timeline, write_player


class ReplayTests(unittest.TestCase):
    def test_committed_timeline_excludes_rejected_branches(self) -> None:
        chosen = [{"frame": "chosen", "kind": "action_frame", "event_seq": 10}]
        rejected = [{"frame": "rejected", "kind": "action_frame", "event_seq": 11}]
        capture = ReplayCapture(
            full=[{"frame": "reset", "kind": "reset", "event_seq": 1, "attempt": 1}]
            + chosen
            + rejected,
            step_frames={10: chosen, 11: rejected},
            decision_frames={20: {"frame": "chosen", "kind": "decision", "event_seq": 20}},
            verified_events=2,
            checked_observations=3,
        )
        events = [
            {"event": "branch_verified", "state_id": "state-a", "env_step_seq": 10},
            {"event": "branch_verified", "state_id": "state-b", "env_step_seq": 11},
            {
                "event": "decision_committed",
                "seq": 20,
                "attempt": 1,
                "decision": 1,
                "committed_state_id": "state-a",
                "restored_archive": False,
                "path": ["right"],
                "score": 1.0,
            },
        ]
        timeline = committed_timeline(events, capture)
        self.assertEqual([item["frame"] for item in timeline], ["reset", "chosen"])
        self.assertTrue(timeline[1]["committed"])

    def test_player_is_self_contained_except_for_deduplicated_pngs(self) -> None:
        frame = Frame(2, 2, 3, bytes(range(12)))
        timeline = [{"frame": frame.digest, "kind": "decision", "event_seq": 1}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "player"
            manifest = write_player(
                output,
                "Replay test",
                timeline,
                {frame.digest: frame},
                120,
                "source-run",
                {"status": "pass"},
            )
            page = (output / "index.html").read_text(encoding="utf-8")
            stored_timeline = json.loads((output / "timeline.json").read_text())
            self.assertIn("240 fps", page)
            self.assertIn(frame.digest, page)
            self.assertEqual(stored_timeline, timeline)
            self.assertEqual(manifest["timeline_frames"], 1)
            self.assertEqual(
                (output / "frames" / f"{frame.digest}.png").read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )

    def test_committed_timeline_includes_evaluator_bootstrap(self) -> None:
        bootstrap = [
            {
                "frame": "power-on",
                "kind": "reset",
                "event_seq": 1,
                "attempt": 0,
                "phase": "bootstrap",
            },
            {
                "frame": "room",
                "kind": "action_frame",
                "event_seq": 2,
                "attempt": 0,
                "phase": "bootstrap",
            },
            {
                "frame": "room",
                "kind": "episode_start",
                "event_seq": 3,
                "attempt": 1,
                "phase": "agent",
            },
        ]
        capture = ReplayCapture(
            full=bootstrap,
            step_frames={},
            decision_frames={
                4: {"frame": "room", "kind": "decision", "event_seq": 4}
            },
            verified_events=0,
            checked_observations=3,
        )
        events = [
            {
                "event": "decision_committed",
                "seq": 4,
                "attempt": 1,
                "decision": 1,
                "committed_state_id": None,
                "restored_archive": False,
                "path": ["noop"],
                "score": 0.0,
            }
        ]
        timeline = committed_timeline(events, capture)
        self.assertEqual(
            [item["kind"] for item in timeline],
            ["reset", "action_frame", "episode_start"],
        )


if __name__ == "__main__":
    unittest.main()
