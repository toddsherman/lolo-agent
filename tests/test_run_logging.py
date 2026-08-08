import csv
import json
import tempfile
import unittest
from pathlib import Path

from lolo_agent.ensemble_world_model import EnsembleVisualDynamicsModel
from lolo_agent.environment import Action
from lolo_agent.log_summary import append_level_annotation, build_run_summary
from lolo_agent.mock_puzzle import MockPuzzleEnv
from lolo_agent.neural_planner import NeuralPlanningConfig, VerifiedNeuralAgent
from lolo_agent.run_logging import LoggedEnvironment, RunLogger, read_events


class RunLoggingTests(unittest.TestCase):
    def test_full_decision_trace_and_derived_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory), run_id="audit-test", fsync_interval=1)
            env = LoggedEnvironment(MockPuzzleEnv(), logger)
            model = EnsembleVisualDynamicsModel(latent_size=16, action_size=8, ensemble_size=2)
            agent = VerifiedNeuralAgent(
                env,
                model,
                "cpu",
                NeuralPlanningConfig(
                    actions=(Action.LEFT, Action.RIGHT),
                    planning_depth=1,
                    beam_width=2,
                    verify_actions=2,
                    action_frames=1,
                    scene_stagnation_visits=99,
                ),
                event_logger=logger,
            )
            agent.reset()
            decision = agent.decide()
            agent.clear_archive()
            logger.close()

            events = list(read_events(logger.run_dir))
            kinds = CounterLike(event["event"] for event in events)
            self.assertEqual(kinds["attempt_started"], 1)
            self.assertEqual(kinds["decision_committed"], 1)
            self.assertEqual(kinds["branch_verified"], 2)
            self.assertEqual(kinds["env_step"], 2)
            self.assertEqual(kinds["state_saved"], kinds["state_released"])
            self.assertEqual(decision.branches_examined, 2)
            events_by_seq = {event["seq"]: event for event in events}
            for branch in (event for event in events if event["event"] == "branch_verified"):
                self.assertEqual(events_by_seq[branch["env_step_seq"]]["event"], "env_step")
                self.assertEqual(events_by_seq[branch["state_save_seq"]]["event"], "state_saved")
                self.assertEqual(events_by_seq[branch["env_step_seq"]]["action"], branch["action"])

            serialized = (logger.run_dir / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"player"', serialized)
            self.assertNotIn('"crate"', serialized)
            self.assertNotIn("_token", serialized)
            self.assertIn("state-00000001", serialized)

            pngs = list((logger.run_dir / "frames").glob("*.png"))
            self.assertTrue(pngs)
            self.assertEqual(pngs[0].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

            append_level_annotation(logger.run_dir, "withheld-room-A", attempt=1)
            summary = build_run_summary(logger.run_dir)
            self.assertEqual(summary["committed_decisions"], 1)
            self.assertEqual(summary["verified_branches"], 2)
            self.assertEqual(summary["annotations"][0]["source"], "evaluator")
            self.assertTrue((logger.run_dir / "transitions.json").is_file())
            with (logger.run_dir / "decisions.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["level"], "withheld-room-A")

    def test_frames_are_content_addressed_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory), run_id="dedup-test")
            env = LoggedEnvironment(MockPuzzleEnv(), logger)
            first = env.reset()
            second = env.step(Action.NOOP)
            logger.close()
            self.assertEqual(first.digest, second.digest)
            self.assertEqual(len(list((logger.run_dir / "frames").glob("*.png"))), 1)
            manifest = json.loads((logger.run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["unique_frame_count"], 1)


class CounterLike(dict):
    def __init__(self, values):
        super().__init__()
        for value in values:
            self[value] = self.get(value, 0) + 1


if __name__ == "__main__":
    unittest.main()
