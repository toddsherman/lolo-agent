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

from lolo_agent.neural_run import (
    StableSceneChangeDetector,
    load_active_option_archives,
    load_active_goal_milestone_checkpoint,
    load_logged_decision_semantic_state,
    load_episodic_decision_events,
    iter_episodic_decision_events,
)
from lolo_agent.pixels import Frame
from lolo_agent.run_logging import RunLogger, sha256_file


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
                            "event": "archive_branch_restored",
                            "decision": 1,
                            "marker": "child-archive-restore",
                        },
                        {
                            "event": "branch_verified",
                            "decision": 1,
                            "marker": "child-branch",
                        },
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
                        {
                            "event": "human_prior_milestone_outcome_recorded",
                            "decision": 1,
                            "marker": "child-milestone-outcome",
                        },
                        {
                            "event": (
                                "human_prior_navigation_detour_expired"
                            ),
                            "decision": 1,
                            "marker": "child-detour-expired",
                        },
                        {
                            "event": "human_prior_option_branch_verified",
                            "decision": 1,
                            "marker": "child-option-branch",
                        },
                        {
                            "event": "human_prior_option_neutral_verified",
                            "decision": 1,
                            "marker": "child-option-neutral",
                        },
                        {
                            "event": (
                                "human_prior_option_local_neutral_verified"
                            ),
                            "decision": 1,
                            "marker": "child-option-local-neutral",
                        },
                        {
                            "event": "human_prior_option_archive_added",
                            "decision": 1,
                            "marker": "child-option-archive",
                        },
                        {
                            "event": "human_prior_option_search_started",
                            "decision": 1,
                            "marker": "child-option-search-started",
                        },
                        {
                            "event": "human_prior_option_search_completed",
                            "decision": 1,
                            "marker": "child-option-search-completed",
                        },
                        {
                            "event": "human_prior_ordering_progress_recorded",
                            "decision": 1,
                            "marker": "child-ordering-progress",
                        },
                        {
                            "event": "human_prior_ordering_hypothesis_disproved",
                            "decision": 1,
                            "marker": "child-ordering-disproof",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            events = load_episodic_decision_events(child, 1)
            streamed_events = list(
                iter_episodic_decision_events(child, 1)
            )

        self.assertEqual(
            [event["marker"] for event in events],
            [
                "parent-1",
                "child-archive-restore",
                "child-branch",
                "child-1",
                "child-room-boundary",
                "child-hazard",
                "child-milestone-outcome",
                "child-detour-expired",
                "child-option-branch",
                "child-option-neutral",
                "child-option-local-neutral",
                "child-option-archive",
                "child-option-search-started",
                "child-option-search-completed",
                "child-ordering-progress",
                "child-ordering-disproof",
            ],
        )
        self.assertEqual(streamed_events, events)

    def test_infers_exhausted_transition_from_resumed_milestone_search(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memory = root / "memory"
            state = root / "state"
            exhausted = root / "exhausted"
            for run in (memory, state, exhausted):
                run.mkdir()
            (memory / "manifest.json").write_text(
                json.dumps({"metadata": {}}), encoding="utf-8"
            )
            (memory / "events.jsonl").write_text(
                "", encoding="utf-8"
            )
            (state / "manifest.json").write_text(
                json.dumps({"metadata": {}}), encoding="utf-8"
            )
            (state / "events.jsonl").write_text(
                json.dumps(
                    {
                        "seq": 17,
                        "event": "human_prior_option_branch_verified",
                        "decision": 1,
                        "human_prior_source_hearts": [[16, 16], [32, 16]],
                        "human_prior_target_hearts": [[32, 16]],
                        "human_prior_chest_obtained": False,
                        "human_prior_target_chest_slot": [48, 16],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            state_digest = sha256_file(state / "events.jsonl")
            (exhausted / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "exhausted-run",
                        "metadata": {
                            "episodic_resume": {
                                "source_run": str(memory),
                                "source_decision": 1,
                                "state_source_run": str(state),
                                "state_source_run_id": "state-run",
                                "state_source_option_event_seq": 17,
                                "state_source_events_sha256": state_digest,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (exhausted / "events.jsonl").write_text(
                json.dumps(
                    {
                        "event": "human_prior_option_search_completed",
                        "decision": 1,
                        "positive_goal_eligible_endpoints": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            events = list(iter_episodic_decision_events(exhausted, 1))

        inferred = [
            event
            for event in events
            if event.get("resumed_milestone_exhaustion_inferred")
        ]
        self.assertEqual(len(inferred), 1)
        self.assertEqual(
            inferred[0]["exhausted_milestone_transition"],
            (((16, 16), (32, 16)), ((32, 16),), False),
        )
        self.assertEqual(inferred[0]["state_source_option_event_seq"], 17)
        self.assertEqual(inferred[0]["exhausted_goal_slot"], [48, 16])
        self.assertTrue(inferred[0]["ordering_hypothesis_reactivated"])
        self.assertTrue(inferred[0]["exhaustion_context_unscoped"])

    def test_loads_only_option_archives_active_at_decision_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = RunLogger(Path(temporary), run_id="archives")
            frame = self.frame(96)
            logger.store_option_archive_snapshot(
                1, "state-1", b"archive-state", frame
            )
            logger.log(
                "human_prior_option_archive_added",
                decision=1,
                state_id="state-1",
                path=["left"],
                durations=[4],
                frame=frame.digest,
            )
            logger.store_decision_snapshot(1, b"decision-1", frame)
            logger.log(
                "archive_branch_restored",
                decision=2,
                state_id="state-1",
            )
            logger.store_decision_snapshot(2, b"decision-2", frame)
            logger.close()

            active_at_one = load_active_option_archives(
                logger.run_dir, 1
            )
            active_at_two = load_active_option_archives(
                logger.run_dir, 2
            )

        self.assertEqual(len(active_at_one), 1)
        self.assertEqual(active_at_one[0].state, b"archive-state")
        self.assertEqual(active_at_one[0].frame, frame)
        self.assertEqual(active_at_one[0].source_state_id, "state-1")
        self.assertEqual(active_at_two, [])

    def test_loads_semantic_state_only_from_matching_decision_frame(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = RunLogger(Path(temporary), run_id="semantic-state")
            frame = self.frame(96)
            logger.log(
                "decision_committed",
                decision=1,
                action="down",
                human_prior_world_target_context="learned-context",
                **logger.frame_fields(frame),
            )
            logger.store_decision_snapshot(1, b"decision-1", frame)
            logger.close()

            semantic = load_logged_decision_semantic_state(
                logger.run_dir, 1
            )

        self.assertIsNotNone(semantic)
        assert semantic is not None
        self.assertEqual(
            semantic["human_prior_world_target_context"],
            "learned-context",
        )
        self.assertEqual(semantic["action"], "down")

    def test_loads_goal_milestone_checkpoint_active_at_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = RunLogger(Path(temporary), run_id="milestone")
            frame = self.frame(96)
            logger.store_goal_milestone_checkpoint_snapshot(
                1,
                "state-1",
                b"pre-heart-state",
                frame,
                checkpoint_decision=1,
                choice=["frontier", "down", 16],
                checkpoint_kind="goal_milestone",
                exploration_steps=3,
            )
            logger.store_decision_snapshot(1, b"decision-1", frame)
            logger.log(
                "goal_milestone_checkpoint_released",
                decision=2,
                state_id="state-1",
            )
            logger.store_decision_snapshot(2, b"decision-2", frame)
            logger.close()

            active_at_one = load_active_goal_milestone_checkpoint(
                logger.run_dir, 1
            )
            active_at_two = load_active_goal_milestone_checkpoint(
                logger.run_dir, 2
            )

        self.assertIsNotNone(active_at_one)
        assert active_at_one is not None
        self.assertEqual(active_at_one.state, b"pre-heart-state")
        self.assertEqual(active_at_one.frame, frame)
        self.assertEqual(active_at_one.metadata["exploration_steps"], 3)
        self.assertIsNone(active_at_two)

    def test_recovers_legacy_milestone_target_across_resume_ancestry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_frame = self.frame(96)
            target_frame = self.frame(128)
            parent = RunLogger(root, run_id="legacy-parent")
            parent.store_goal_milestone_checkpoint_snapshot(
                1,
                "state-1",
                b"pre-heart-state",
                source_frame,
                checkpoint_decision=1,
                choice=["frontier", "down", 16],
                checkpoint_kind="goal_milestone",
                goal_heart_slots=[[1, 2], [3, 4]],
                goal_target_heart_slots=[],
                exploration_steps=3,
            )
            parent.log(
                "decision_committed",
                decision=1,
                action="down",
                action_frames=16,
                source_behavioral_signature="frontier",
                parent_frame=source_frame.digest,
                human_prior_source_hearts=[[1, 2], [3, 4]],
                human_prior_target_hearts=[[3, 4]],
                **parent.frame_fields(target_frame),
            )
            parent.store_decision_snapshot(
                1, b"parent-decision-1", target_frame
            )
            parent.close()

            child = RunLogger(
                root,
                run_id="legacy-child",
                metadata={
                    "episodic_resume": {
                        "source_run": str(parent.run_dir),
                        "source_decision": 1,
                    }
                },
            )
            child.store_goal_milestone_checkpoint_snapshot(
                0,
                "state-1",
                b"pre-heart-state",
                source_frame,
                checkpoint_decision=1,
                choice=["frontier", "down", 16],
                checkpoint_kind="goal_milestone",
                goal_heart_slots=[[1, 2], [3, 4]],
                goal_target_heart_slots=[],
                goal_target_heart_slots_known=False,
                exploration_steps=7,
            )
            child.store_decision_snapshot(
                1, b"child-decision-1", target_frame
            )
            child.close()

            recovered = load_active_goal_milestone_checkpoint(
                child.run_dir, 1
            )

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(
            recovered.metadata["goal_target_heart_slots"], [[3, 4]]
        )
        self.assertTrue(
            recovered.metadata["goal_target_heart_slots_known"]
        )
        self.assertEqual(
            recovered.metadata["goal_target_heart_slots_source"],
            "legacy_decision_telemetry",
        )
        self.assertEqual(
            recovered.metadata["goal_target_heart_slots_source_decision"],
            1,
        )
        self.assertEqual(recovered.metadata["exploration_steps"], 7)


if __name__ == "__main__":
    unittest.main()
