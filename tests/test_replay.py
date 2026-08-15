import json
import tempfile
import unittest
from pathlib import Path

from lolo_agent.environment import Action
from lolo_agent.pixels import Frame
from lolo_agent.replay import (
    ReplayCapture,
    _save_logged_state,
    committed_timeline,
    restore_logged_decision,
    restore_logged_goal_milestone_checkpoint,
    restore_logged_option_archive,
    restore_logged_option_branch,
    write_player,
)
from lolo_agent.run_logging import RunLogger


class ReplayTests(unittest.TestCase):
    def test_imported_archive_save_replays_without_moving_live_state(
        self,
    ) -> None:
        class ArchiveEnvironment:
            def __init__(self) -> None:
                self.position = 0
                self.states = set()

            def frame(self) -> Frame:
                pixels = bytearray(2)
                pixels[self.position] = 255
                return Frame(2, 1, 1, bytes(pixels))

            def save_state(self):
                state = [self.position]
                self.states.add(id(state))
                return state

            def load_state(self, state) -> Frame:
                self.position = state[0]
                return self.frame()

            def release_state(self, state) -> None:
                self.states.remove(id(state))

            def import_state(self, state: bytes, frame: Frame) -> Frame:
                self.position = state[0]
                imported = self.frame()
                self.assert_frame(imported, frame)
                return imported

            @staticmethod
            def assert_frame(actual: Frame, expected: Frame) -> None:
                if actual != expected:
                    raise RuntimeError("frame mismatch")

        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory), run_id="archive-replay")
            env = ArchiveEnvironment()
            live = env.frame()
            env.position = 1
            archived = env.frame()
            env.position = 0
            stored = logger.store_option_archive_snapshot(
                0, "state-1", b"\x01", archived
            )
            event = logger.log(
                "state_saved",
                state_id="state-1",
                imported_option_archive=True,
                option_archive_state_file=stored["state_file"],
                option_archive_state_sha256=stored["state_sha256"],
                **logger.frame_fields(archived),
            )
            logger.close()

            handle = _save_logged_state(
                env, logger.run_dir, event, live
            )

            self.assertEqual(env.frame(), live)
            self.assertEqual(env.load_state(handle), archived)
            env.release_state(handle)
            self.assertEqual(env.states, set())

    def test_decision_snapshot_restores_without_replaying_event_history(self) -> None:
        class SnapshotEnvironment:
            def __init__(self) -> None:
                self.imported = None

            def import_state(self, state: bytes, frame: Frame) -> Frame:
                self.imported = (state, frame)
                return frame

        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory), run_id="snapshot-run")
            frame = Frame(2, 2, 3, bytes(range(12)))
            logger.log("decision_committed", decision=7, **logger.frame_fields(frame))
            snapshot = b"opaque-emulator-state"
            stored = logger.store_decision_snapshot(7, snapshot, frame)
            logger.close()

            env = SnapshotEnvironment()
            restored = restore_logged_decision(env, logger.run_dir, 7)

            self.assertEqual(restored.frame, frame)
            self.assertEqual(restored.event_seq, stored["seq"])
            self.assertEqual(env.imported, (snapshot, frame))

    def test_option_archive_snapshot_restores_by_logged_state_id(self) -> None:
        class SnapshotEnvironment:
            def __init__(self) -> None:
                self.imported = None

            def import_state(self, state: bytes, frame: Frame) -> Frame:
                self.imported = (state, frame)
                return frame

        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory), run_id="option-run")
            frame = Frame(2, 2, 3, bytes(range(12)))
            state = b"opaque-option-state"
            stored = logger.store_option_archive_snapshot(
                4, "state-00000009", state, frame
            )
            logger.log(
                "human_prior_option_archive_added",
                decision=4,
                state_id="state-00000009",
                human_prior_world_target_context="world-two",
                **logger.frame_fields(frame),
            )
            logger.close()

            env = SnapshotEnvironment()
            restored = restore_logged_option_archive(
                env, logger.run_dir, "state-00000009"
            )

            self.assertEqual(restored.frame, frame)
            self.assertEqual(restored.event_seq, stored["seq"])
            self.assertEqual(restored.state_id, "state-00000009")
            self.assertEqual(
                restored.metadata["human_prior_world_target_context"],
                "world-two",
            )
            self.assertEqual(env.imported, (state, frame))

    def test_goal_milestone_checkpoint_restores_by_logged_event(self) -> None:
        class SnapshotEnvironment:
            def __init__(self) -> None:
                self.imported = None

            def import_state(self, state: bytes, frame: Frame) -> Frame:
                self.imported = (state, frame)
                return frame

        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory), run_id="checkpoint-run")
            frame = Frame(2, 2, 3, bytes(range(12)))
            state = b"pre-milestone-state"
            stored = logger.store_goal_milestone_checkpoint_snapshot(
                4,
                "state-checkpoint",
                state,
                frame,
                checkpoint_decision=3,
                choice=["visual-frontier", "right", 8],
                checkpoint_kind="goal_milestone",
                goal_heart_slots=[[16, 32], [48, 64]],
                goal_target_heart_slots=[[48, 64]],
                goal_target_heart_slots_known=True,
                human_prior_world_context_signature="anonymous-world",
                pose_action="right",
            )
            logger.close()

            env = SnapshotEnvironment()
            restored = restore_logged_goal_milestone_checkpoint(
                env,
                logger.run_dir,
                stored["seq"],
            )

            self.assertEqual(restored.frame, frame)
            self.assertEqual(restored.event_seq, stored["seq"])
            self.assertEqual(restored.decision, 3)
            self.assertEqual(restored.state_id, "state-checkpoint")
            self.assertEqual(
                restored.metadata["human_prior_world_context_signature"],
                "anonymous-world",
            )
            self.assertEqual(env.imported, (state, frame))

    def test_verified_option_branch_replays_from_logged_resume_root(self) -> None:
        class BranchEnvironment:
            def __init__(self, target: Frame) -> None:
                self.target = target
                self.imported = None
                self.steps = []

            def import_state(self, state: bytes, frame: Frame) -> Frame:
                self.imported = (state, frame)
                return frame

            def step(self, action: Action, frames: int) -> Frame:
                self.steps.append((action, frames))
                return self.target

        with tempfile.TemporaryDirectory() as directory:
            root_logger = RunLogger(Path(directory), run_id="root-run")
            root_frame = Frame(2, 2, 3, bytes(range(12)))
            root_state = b"root-option-state"
            root_logger.store_option_archive_snapshot(
                1,
                "state-root",
                root_state,
                root_frame,
            )
            root_logger.log(
                "human_prior_option_archive_added",
                decision=1,
                state_id="state-root",
                **root_logger.frame_fields(root_frame),
            )
            root_logger.close()

            branch_frame = Frame(2, 2, 3, bytes(reversed(range(12))))
            branch_logger = RunLogger(
                Path(directory),
                run_id="branch-run",
                metadata={
                    "episodic_resume": {
                        "state_source_run": str(root_logger.run_dir),
                        "state_source_archive_id": "state-root",
                    }
                },
            )
            branch_event = branch_logger.log(
                "human_prior_option_branch_verified",
                decision=1,
                path=["right"],
                durations=[4],
                **branch_logger.frame_fields(branch_frame),
            )
            branch_logger.close()

            env = BranchEnvironment(branch_frame)
            restored = restore_logged_option_branch(
                env,
                branch_logger.run_dir,
                branch_event["seq"],
            )

            self.assertEqual(restored.frame, branch_frame)
            self.assertEqual(restored.event_seq, branch_event["seq"])
            self.assertEqual(restored.metadata["path"], ["right"])
            self.assertEqual(env.imported, (root_state, root_frame))
            self.assertEqual(env.steps, [(Action.RIGHT, 4)])

    def test_verified_option_branch_replays_through_nested_branch_resume(
        self,
    ) -> None:
        class BranchEnvironment:
            def __init__(self, targets) -> None:
                self.targets = list(targets)
                self.imported = None
                self.steps = []

            def import_state(self, state: bytes, frame: Frame) -> Frame:
                self.imported = (state, frame)
                return frame

            def step(self, action: Action, frames: int) -> Frame:
                self.steps.append((action, frames))
                return self.targets.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            root_logger = RunLogger(Path(directory), run_id="nested-root")
            root_frame = Frame(2, 2, 3, bytes(range(12)))
            root_state = b"nested-root-state"
            root_logger.store_option_archive_snapshot(
                1,
                "state-root",
                root_state,
                root_frame,
            )
            root_logger.log(
                "human_prior_option_archive_added",
                decision=1,
                state_id="state-root",
                **root_logger.frame_fields(root_frame),
            )
            root_logger.close()

            first_frame = Frame(2, 2, 3, bytes(reversed(range(12))))
            first_logger = RunLogger(
                Path(directory),
                run_id="nested-first",
                metadata={
                    "episodic_resume": {
                        "state_source_run": str(root_logger.run_dir),
                        "state_source_archive_id": "state-root",
                    }
                },
            )
            first_event = first_logger.log(
                "human_prior_option_branch_verified",
                decision=1,
                path=["right"],
                durations=[4],
                **first_logger.frame_fields(first_frame),
            )
            first_logger.close()

            final_frame = Frame(2, 2, 3, bytes([7] * 12))
            final_logger = RunLogger(
                Path(directory),
                run_id="nested-final",
                metadata={
                    "episodic_resume": {
                        "state_source_run": str(first_logger.run_dir),
                        "state_source_option_event_seq": first_event["seq"],
                    }
                },
            )
            final_event = final_logger.log(
                "human_prior_option_branch_verified",
                decision=1,
                path=["down"],
                durations=[5],
                **final_logger.frame_fields(final_frame),
            )
            final_logger.close()

            env = BranchEnvironment([first_frame, final_frame])
            restored = restore_logged_option_branch(
                env,
                final_logger.run_dir,
                final_event["seq"],
            )

            self.assertEqual(restored.frame, final_frame)
            self.assertEqual(env.imported, (root_state, root_frame))
            self.assertEqual(
                env.steps,
                [(Action.RIGHT, 4), (Action.DOWN, 5)],
            )

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
