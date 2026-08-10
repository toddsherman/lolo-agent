import csv
import json
import tempfile
import unittest
from pathlib import Path

from lolo_agent.bootstrap import (
    BootstrapFixture,
    BootstrapStep,
    apply_bootstrap_fixture,
)
from lolo_agent.ensemble_world_model import EnsembleVisualDynamicsModel
from lolo_agent.environment import Action
from lolo_agent.log_summary import append_level_annotation, build_run_summary
from lolo_agent.mock_puzzle import MockPuzzleEnv
from lolo_agent.neural_planner import NeuralPlanningConfig, VerifiedNeuralAgent
from lolo_agent.pixels import signature_key
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
                    visual_stagnation_visits=99,
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
            self.assertEqual(kinds["env_step"], 3)
            self.assertEqual(kinds["matched_neutral_verified"], 1)
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
            self.assertEqual(summary["committed_durations"], {"1": 1})
            self.assertEqual(summary["delayed_visual_returns"], 0)
            self.assertEqual(summary["delayed_return_recoveries"], 0)
            self.assertEqual(summary["persistent_frontier_updates"], 1)
            self.assertGreaterEqual(summary["visual_abstraction_clusters"], 1)
            self.assertGreaterEqual(summary["behavioral_abstraction_clusters"], 1)
            self.assertEqual(summary["behavioral_abstraction_deferrals"], 0)
            self.assertEqual(summary["behavior_probe_selections"], 1)
            self.assertEqual(summary["temporal_options_started"], 0)
            self.assertEqual(summary["temporal_option_samples"], 0)
            self.assertEqual(summary["delayed_temporal_option_samples"], 0)
            self.assertIn("action_effect_observations", summary)
            self.assertIn("action_effect_observations_by_action", summary)
            self.assertIn("action_effect_known_branches", summary)
            self.assertIn("learned_hazard_filter_events", summary)
            self.assertIn("human_prior_life_losses", summary)
            self.assertIn("human_prior_world_effect_confirmations", summary)
            self.assertIn("human_prior_world_effects_accepted", summary)
            self.assertIn("human_prior_world_effects_rejected", summary)
            self.assertIn(
                "human_prior_unique_world_effect_signatures", summary
            )
            self.assertIn(
                "human_prior_unique_committed_world_contexts", summary
            )
            self.assertIn(
                "human_prior_unique_committed_graph_states", summary
            )
            self.assertIn(
                "human_prior_unique_committed_player_positions", summary
            )
            self.assertIn(
                "human_prior_semantic_frontier_overrides", summary
            )
            self.assertIn("human_prior_best_first_filter_events", summary)
            self.assertIn(
                "human_prior_best_first_frontier_exhaustions", summary
            )
            self.assertIn("human_prior_graph_stagnation_events", summary)
            self.assertIn("life_hazard_checkpoints_created", summary)
            self.assertIn("life_hazard_checkpoint_restores", summary)
            self.assertIn("goal_milestone_checkpoints_created", summary)
            self.assertIn("goal_milestone_checkpoint_restores", summary)
            self.assertIn(
                "goal_milestone_descendant_invalidations", summary
            )
            self.assertIn(
                "goal_milestone_descendant_release_failures", summary
            )
            self.assertIn("learned_hazard_filtered_choices", summary)
            self.assertIn("archive_hazard_rejections", summary)
            self.assertIn("archive_branch_rejections", summary)
            self.assertIn("archive_rejections_by_reason", summary)
            self.assertIn("causal_outcome_archive_additions", summary)
            self.assertIn("causal_outcome_exhaustions", summary)
            self.assertIn("global_action_hazard_samples", summary)
            self.assertIn("matched_neutral_verifications", summary)
            self.assertIn("causal_spatial_observations", summary)
            self.assertIn("unique_causal_spatial_signatures", summary)
            self.assertIn("committed_causal_spatial_signatures", summary)
            self.assertIn("causal_cells_first_visited", summary)
            self.assertIn("causal_cell_coverage_bonus_total", summary)
            self.assertIn("causal_cell_coverage_mean", summary)
            self.assertIn("persistent_change_updates", summary)
            self.assertIn("persistent_change_activations", summary)
            self.assertIn("persistent_change_archive_filter_events", summary)
            self.assertIn("persistent_change_max_active_cells", summary)
            self.assertEqual(summary["spatial_shadow_evaluations"], 0)
            self.assertEqual(summary["spatial_shadow_beats_persistence"], 0)
            self.assertIn("spatial_shadow_mean_metrics", summary)
            self.assertTrue((logger.run_dir / "spatial_shadow.csv").is_file())
            self.assertIn("temporal_option_counterfactuals_armed", summary)
            self.assertIn("temporal_option_eligible_initiations", summary)
            self.assertEqual(
                summary["behavior_probe_selection_reasons"],
                {"coverage_rotation": 1},
            )
            self.assertEqual(summary["annotations"][0]["source"], "evaluator")
            self.assertTrue((logger.run_dir / "transitions.json").is_file())
            with (logger.run_dir / "decisions.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["level"], "withheld-room-A")
            self.assertNotEqual(rows[0]["persistent_frontier_value"], "")
            self.assertNotEqual(rows[0]["abstract_signature"], "")
            self.assertNotEqual(rows[0]["source_behavioral_signature"], "")
            self.assertNotEqual(rows[0]["target_frontier_signature"], "")
            self.assertNotEqual(rows[0]["temporal_option_value"], "")
            self.assertNotEqual(rows[0]["temporal_option_is_known"], "")
            self.assertNotEqual(rows[0]["temporal_option_value_source"], "")
            self.assertNotEqual(rows[0]["action_effect_value"], "")
            self.assertNotEqual(rows[0]["action_effect_is_known"], "")
            self.assertNotEqual(rows[0]["action_effect_samples"], "")
            self.assertNotEqual(rows[0]["action_effect_bonus"], "")
            self.assertNotEqual(rows[0]["causal_spatial_novelty"], "")
            self.assertNotEqual(rows[0]["causal_context_signature"], "")
            self.assertNotEqual(
                rows[0]["target_causal_context_signature"], ""
            )
            self.assertNotEqual(rows[0]["causal_event_detected"], "")
            self.assertNotEqual(rows[0]["causal_component_count"], "")
            self.assertIn("causal_event_basis", rows[0])
            self.assertIn("causal_event_novel_cells", rows[0])
            self.assertIn("causal_affordance_count", rows[0])
            self.assertIn("transition_spatial_signature", rows[0])
            self.assertNotEqual(rows[0]["causal_changed_pixels"], "")
            self.assertNotEqual(rows[0]["causal_change_centroid"], "")
            self.assertNotEqual(rows[0]["causal_spatial_bonus"], "")
            self.assertNotEqual(rows[0]["causal_cell_coverage"], "")
            self.assertNotEqual(rows[0]["causal_cell_unvisited"], "")
            self.assertNotEqual(rows[0]["causal_cell_count"], "")
            self.assertNotEqual(rows[0]["causal_cell_coverage_bonus"], "")
            self.assertNotEqual(rows[0]["persistent_change_enabled"], "")
            self.assertNotEqual(
                rows[0]["persistent_change_stability_decisions"], ""
            )
            self.assertNotEqual(
                rows[0]["persistent_change_minimum_value_drop"], ""
            )
            self.assertNotEqual(
                rows[0]["persistent_change_active_count"], ""
            )
            self.assertNotEqual(
                rows[0]["persistent_change_active_cells"], ""
            )
            self.assertNotEqual(rows[0]["duration_counts"], "")
            self.assertNotEqual(rows[0]["action_duration_counts"], "")
            self.assertNotEqual(rows[0]["active_temporal_option"], "")
            self.assertNotEqual(
                rows[0]["temporal_option_initiation_eligible"], ""
            )
            self.assertNotEqual(
                rows[0]["temporal_option_counterfactual_contrast"], ""
            )
            self.assertNotEqual(
                rows[0]["temporal_option_counterfactuals"], ""
            )
            self.assertNotEqual(
                rows[0]["temporal_option_delayed_counterfactual_armed"], ""
            )
            self.assertIn("human_prior_goal_phase", rows[0])
            self.assertIn("human_prior_remaining_hearts", rows[0])
            self.assertIn("human_prior_goal_reward", rows[0])
            self.assertIn("human_prior_life_loss_penalty", rows[0])
            self.assertIn("human_prior_life_loss_confirmed", rows[0])
            self.assertIn("human_prior_target_player_slot", rows[0])

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

    def test_bidirectional_probe_has_separate_phase_and_flat_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory), run_id="probe-audit")
            env = LoggedEnvironment(MockPuzzleEnv(), logger)
            model = EnsembleVisualDynamicsModel(
                latent_size=16, action_size=8, ensemble_size=2
            )
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
                    visual_stagnation_visits=99,
                    returnability_probe_depth=1,
                    returnability_probe_beam_width=1,
                    returnability_probe_pixel_l1_threshold=0.0,
                ),
                event_logger=logger,
            )
            agent.reset()
            agent.decide()
            agent.clear_archive()
            logger.close()

            events = list(read_events(logger.run_dir))
            probe_steps = [
                event
                for event in events
                if event["event"] == "env_step"
                and event.get("phase") == "returnability_probe"
            ]
            agent_steps = [
                event
                for event in events
                if event["event"] == "env_step"
                and event.get("phase") == "agent"
            ]
            self.assertEqual(len(probe_steps), 6)
            self.assertEqual(len(agent_steps), 3)
            summary = build_run_summary(logger.run_dir)
            self.assertEqual(summary["returnability_probe_branches"], 2)
            self.assertEqual(summary["returnability_probe_paths"], 4)
            self.assertGreaterEqual(
                summary["returnability_probe_branches_with_return"], 1
            )
            with (logger.run_dir / "returnability_probes.csv").open(
                encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["branch_id"] for row in rows))

    def test_evaluator_bootstrap_is_logged_but_excluded_from_agent_statistics(
        self,
    ) -> None:
        probe = MockPuzzleEnv()
        probe.reset()
        expected = probe.step(Action.RIGHT)
        fixture = BootstrapFixture(
            name="test-room",
            steps=(BootstrapStep(Action.RIGHT, 1),),
            expected_frame_sha256=expected.digest,
            expected_scene_signature=signature_key(
                expected.coarse_signature(columns=3, rows=3)
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(
                Path(directory),
                run_id="bootstrap-test",
                metadata={"bootstrap": {"fixture": fixture.name}},
            )
            env = LoggedEnvironment(MockPuzzleEnv(), logger)
            frame = apply_bootstrap_fixture(env, fixture)
            model = EnsembleVisualDynamicsModel(
                latent_size=16, action_size=8, ensemble_size=2
            )
            agent = VerifiedNeuralAgent(
                env,
                model,
                "cpu",
                NeuralPlanningConfig(actions=(Action.NOOP,), planning_depth=1),
                event_logger=logger,
            )
            agent.reset(initial_frame=frame)
            logger.close()

            events = list(read_events(logger.run_dir))
            bootstrap_steps = [
                event
                for event in events
                if event["event"] == "env_step"
                and event.get("phase") == "bootstrap"
            ]
            self.assertEqual(len(bootstrap_steps), 1)
            self.assertEqual(bootstrap_steps[0]["attempt"], 0)
            self.assertEqual(
                sum(event["event"] == "attempt_started" for event in events), 1
            )
            self.assertEqual(frame, expected)

            summary = build_run_summary(logger.run_dir)
            self.assertTrue(summary["bootstrap_completed"])
            self.assertEqual(summary["bootstrap_fixture"], "test-room")
            self.assertEqual(summary["bootstrap_actions"], {"right": 1})
            self.assertEqual(summary["bootstrap_frames"], 1)
            self.assertEqual(summary["investigated_actions"], {})


class CounterLike(dict):
    def __init__(self, values):
        super().__init__()
        for value in values:
            self[value] = self.get(value, 0) + 1


if __name__ == "__main__":
    unittest.main()
