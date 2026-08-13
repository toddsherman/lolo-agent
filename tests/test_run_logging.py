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
from lolo_agent.pixels import Frame, signature_key
from lolo_agent.run_logging import LoggedEnvironment, RunLogger, read_events


class PersistentStateEnv:
    def __init__(self) -> None:
        self.position = 0
        self.active_states = set()

    def _frame(self) -> Frame:
        pixels = bytearray(8)
        pixels[self.position] = 255
        return Frame(8, 1, 1, bytes(pixels))

    def reset(self) -> Frame:
        self.position = 0
        return self._frame()

    def observe(self) -> Frame:
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.RIGHT:
            self.position = 1
        return self._frame()

    def save_state(self):
        state = [self.position]
        self.active_states.add(id(state))
        return state

    def load_state(self, state) -> Frame:
        if id(state) not in self.active_states:
            raise RuntimeError("unknown state")
        self.position = state[0]
        return self._frame()

    def release_state(self, state) -> None:
        self.active_states.remove(id(state))

    def export_state(self) -> bytes:
        return bytes((self.position,))

    def import_state(self, state: bytes, frame: Frame) -> Frame:
        self.position = state[0]
        imported = self._frame()
        if imported.digest != frame.digest:
            raise RuntimeError("frame mismatch")
        return imported


class RunLoggingTests(unittest.TestCase):
    def test_anonymous_behavior_hazard_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory), run_id="behavior-summary")
            logger.log(
                "anonymous_entity_behavior_observed",
                anonymous_type_id=7,
                observed_outcome="terminal",
                behavior_known_before=True,
                hazard_probability_before=1.0,
                observed_hazard=True,
                outcome_matched_prediction=True,
                differential_terminal_visual_change=True,
                causal_attribution=True,
                evidence_eligible=False,
            )
            logger.log(
                "anonymous_entity_behavior_shadow_prediction",
                decision=1,
                action=Action.RIGHT,
                action_frames=1,
                horizon_frames=3,
                behavior_known=True,
                shadow_prediction_actionable=True,
                shadow_would_reject=True,
                hazard_probability=1.0,
                unconditional_behavior_known=True,
                unconditional_hazard_probability=0.5,
                shadow_hazard_threshold=0.9,
                shadow_policy_authority=False,
            )
            logger.log(
                "anonymous_entity_behavior_shadow_branch_evaluated",
                decision=1,
                action=Action.RIGHT,
                action_frames=1,
                shadow_would_reject=True,
                shadow_policy_authority=False,
                model_parameters_unchanged=True,
            )
            logger.log(
                "anonymous_entity_hazard_veto_evaluated",
                hazards_detected=1,
                hazards_filtered=1,
                fail_open=False,
            )
            logger.log("anonymous_entity_passive_horizon_verified")
            logger.log("anonymous_entity_causal_horizon_verified")
            logger.log(
                "anonymous_entity_causal_contrast_completed",
                decision=1,
                intervention_action=Action.RIGHT,
                intervention_frames=1,
                wait_frames=3,
                factual_hazard=True,
                hazard_contrast=True,
                newly_localized_candidates=1,
            )
            logger.close()

            summary = build_run_summary(logger.run_dir)
            shadow_csv_exists = (
                logger.run_dir / "entity_behavior_shadow.csv"
            ).is_file()
            shadow_branch_csv_exists = (
                logger.run_dir / "entity_behavior_shadow_branches.csv"
            ).is_file()

        self.assertEqual(
            summary["anonymous_entity_behavior_hazard_observations"], 1
        )
        self.assertEqual(
            summary[
                "anonymous_entity_behavior_known_hazard_predictions"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "anonymous_entity_behavior_hazard_classification_matches"
            ],
            1,
        )
        self.assertEqual(
            summary["anonymous_entity_behavior_terminal_observations"], 1
        )
        self.assertEqual(
            summary[
                "anonymous_entity_behavior_terminal_evidence_withheld"
            ],
            1,
        )
        self.assertEqual(
            summary["anonymous_entity_passive_horizon_branches"], 1
        )
        self.assertEqual(
            summary["anonymous_entity_causal_horizon_branches"], 1
        )
        self.assertEqual(summary["anonymous_entity_causal_contrasts"], 1)
        self.assertEqual(
            summary["anonymous_entity_causal_hazard_contrasts"], 1
        )
        self.assertEqual(
            summary["anonymous_entity_causal_candidates_localized"], 1
        )
        self.assertEqual(
            summary["anonymous_entity_causal_hazard_attributions"], 1
        )
        self.assertEqual(
            summary["anonymous_entity_shadow_branch_evaluations"], 1
        )
        self.assertEqual(
            summary["anonymous_entity_shadow_would_reject_branches"], 1
        )
        self.assertEqual(
            summary["anonymous_entity_shadow_policy_authority_branches"], 0
        )
        self.assertEqual(
            summary["anonymous_entity_shadow_parameter_audits_passed"], 1
        )
        self.assertEqual(
            summary["anonymous_entity_shadow_causal_outcomes_evaluable"], 1
        )
        self.assertEqual(
            summary["anonymous_entity_shadow_causal_true_positives"], 1
        )
        self.assertEqual(
            summary[
                "anonymous_entity_shadow_causal_classification_matches"
            ],
            1,
        )
        self.assertEqual(
            summary[
                "anonymous_entity_shadow_unconditional_causal_matches"
            ],
            0,
        )
        self.assertEqual(
            summary["anonymous_entity_shadow_persistence_causal_matches"],
            0,
        )
        self.assertEqual(
            summary["anonymous_entity_hazard_veto_evaluations"], 1
        )
        self.assertEqual(
            summary["anonymous_entity_hazard_veto_detections"], 1
        )
        self.assertEqual(
            summary["anonymous_entity_hazard_veto_filtered"], 1
        )
        self.assertEqual(
            summary["anonymous_entity_hazard_veto_fail_opens"], 0
        )
        self.assertTrue(shadow_csv_exists)
        self.assertTrue(shadow_branch_csv_exists)

    def test_persistent_option_archive_round_trip_preserves_live_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory), run_id="archive-round-trip")
            native = PersistentStateEnv()
            env = LoggedEnvironment(native, logger)
            root_frame = env.reset()
            root = env.save_state()
            archive_frame = env.step(Action.RIGHT, 1)
            archive = env.save_state()
            env.load_state(root)

            stored = env.persist_option_archive_state(
                archive, archive_frame, 1
            )

            self.assertIsNotNone(stored)
            self.assertEqual(env.observe(), root_frame)
            assert stored is not None
            payload = (
                logger.run_dir / str(stored["state_file"])
            ).read_bytes()
            imported = env.import_option_archive_state(
                payload,
                archive_frame,
                source_run_id="parent",
                source_state_id="state-2",
            )
            self.assertEqual(env.observe(), root_frame)
            self.assertEqual(env.load_state(imported), archive_frame)
            env.release_state(imported)
            env.release_state(archive)
            env.release_state(root)
            logger.close()

            events = list(read_events(logger.run_dir))
            counts = CounterLike(event["event"] for event in events)
            self.assertEqual(counts["option_archive_snapshot_stored"], 2)
            self.assertEqual(
                counts["episodic_option_archive_state_imported"], 1
            )
            self.assertEqual(counts["state_saved"], counts["state_released"])
            self.assertEqual(native.active_states, set())

    def test_persistent_milestone_checkpoint_round_trip_preserves_live_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory), run_id="milestone-round-trip")
            native = PersistentStateEnv()
            env = LoggedEnvironment(native, logger)
            root_frame = env.reset()
            checkpoint = env.save_state()
            env.step(Action.RIGHT, 1)
            live_frame = env.observe()

            stored = env.persist_goal_milestone_checkpoint_state(
                checkpoint,
                root_frame,
                1,
                choice=["frontier", "right", 1],
                checkpoint_kind="goal_milestone",
            )

            self.assertIsNotNone(stored)
            self.assertEqual(env.observe(), live_frame)
            assert stored is not None
            payload = (logger.run_dir / str(stored["state_file"])).read_bytes()
            imported = env.import_goal_milestone_checkpoint_state(
                payload,
                root_frame,
                source_run_id="parent",
                source_state_id="state-1",
                # Resume loading returns the complete immutable source event,
                # including logger-owned decision, state, and frame fields.
                metadata=stored,
            )
            self.assertEqual(env.observe(), live_frame)
            self.assertEqual(env.load_state(imported), root_frame)
            env.release_state(imported)
            env.release_state(checkpoint)
            logger.close()

            events = list(read_events(logger.run_dir))
            counts = CounterLike(event["event"] for event in events)
            self.assertEqual(
                counts["goal_milestone_checkpoint_snapshot_stored"], 2
            )
            self.assertEqual(
                counts[
                    "episodic_goal_milestone_checkpoint_state_imported"
                ],
                1,
            )
            snapshots = [
                event
                for event in events
                if event["event"]
                == "goal_milestone_checkpoint_snapshot_stored"
            ]
            self.assertEqual(snapshots[-1]["decision"], 0)
            self.assertEqual(
                snapshots[-1]["source_snapshot_decision"], 1
            )
            self.assertEqual(
                snapshots[-1]["choice"], ["frontier", "right", 1]
            )
            self.assertEqual(native.active_states, set())

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
            self.assertIn("human_prior_option_searches", summary)
            self.assertIn("human_prior_option_search_deferrals", summary)
            self.assertIn("human_prior_option_search_skips", summary)
            self.assertIn(
                "human_prior_option_search_budget_reopens", summary
            )
            self.assertIn("human_prior_option_cleanup_failures", summary)
            self.assertIn("human_prior_option_branches_verified", summary)
            self.assertIn(
                "human_prior_option_neutral_verifications", summary
            )
            self.assertIn(
                "human_prior_option_world_effect_observations", summary
            )
            self.assertIn(
                "human_prior_option_nonlocal_world_effect_observations",
                summary,
            )
            self.assertIn(
                "human_prior_unique_option_world_effect_signatures",
                summary,
            )
            self.assertIn(
                "human_prior_option_world_effect_stability_probes",
                summary,
            )
            self.assertIn(
                "human_prior_option_world_effect_stable", summary
            )
            self.assertIn(
                "human_prior_option_world_effect_local_candidates", summary
            )
            self.assertIn(
                "human_prior_option_world_effect_phase_audits", summary
            )
            self.assertIn(
                "human_prior_option_world_effect_phase_equivalent", summary
            )
            self.assertIn(
                "human_prior_option_world_effect_safe", summary
            )
            self.assertIn(
                "human_prior_option_world_effect_action_controls",
                summary,
            )
            self.assertIn(
                "human_prior_option_world_effect_action_controls_confirmed",
                summary,
            )
            self.assertIn(
                "human_prior_option_world_effect_local_controls", summary
            )
            self.assertIn(
                "human_prior_option_world_effect_local_controls_confirmed",
                summary,
            )
            self.assertIn(
                "human_prior_option_effect_frontier_evaluations", summary
            )
            self.assertIn(
                "human_prior_option_effect_frontier_eligible", summary
            )
            self.assertIn(
                "human_prior_option_effect_frontier_archives", summary
            )
            self.assertIn(
                "human_prior_option_entity_frontier_evaluations", summary
            )
            self.assertIn(
                "human_prior_option_entity_frontier_eligible", summary
            )
            self.assertIn(
                "human_prior_option_entity_frontier_archives", summary
            )
            self.assertIn(
                "human_prior_option_entity_curiosity_branches", summary
            )
            self.assertIn(
                "human_prior_option_entity_curiosity_beam_retained",
                summary,
            )
            self.assertIn(
                "human_prior_option_entity_curiosity_probes", summary
            )
            self.assertIn(
                "human_prior_option_entity_curiosity_known_probes",
                summary,
            )
            self.assertIn(
                "human_prior_option_entity_curiosity_transferable_probes",
                summary,
            )
            self.assertIn(
                "human_prior_option_entity_curiosity_cell_matches",
                summary,
            )
            self.assertIn(
                "human_prior_option_entity_curiosity_evidence_withheld",
                summary,
            )
            self.assertIn(
                "human_prior_option_entity_curiosity_evidence_accepted",
                summary,
            )
            self.assertIn(
                "anonymous_entity_behavior_observations", summary
            )
            self.assertIn(
                "anonymous_entity_behavior_evidence_accepted", summary
            )
            self.assertIn(
                "anonymous_entity_behavior_known_predictions", summary
            )
            self.assertIn(
                "anonymous_entity_behavior_prediction_matches", summary
            )
            self.assertIn(
                "anonymous_entity_behavior_hazard_observations", summary
            )
            self.assertIn(
                "anonymous_entity_behavior_known_hazard_predictions",
                summary,
            )
            self.assertIn(
                "anonymous_entity_behavior_hazard_classification_matches",
                summary,
            )
            self.assertIn(
                "anonymous_entity_behavior_terminal_observations", summary
            )
            self.assertIn(
                "anonymous_entity_behavior_terminal_evidence_withheld",
                summary,
            )
            self.assertIn(
                "anonymous_entity_passive_horizon_branches", summary
            )
            self.assertIn(
                "anonymous_entity_causal_horizon_branches", summary
            )
            self.assertIn("anonymous_entity_causal_contrasts", summary)
            self.assertIn(
                "anonymous_entity_causal_hazard_contrasts", summary
            )
            self.assertIn(
                "anonymous_entity_causal_candidates_localized", summary
            )
            self.assertIn(
                "anonymous_entity_causal_attributions", summary
            )
            self.assertIn(
                "anonymous_entity_behavior_types_observed", summary
            )
            self.assertTrue(
                (logger.run_dir / "entity_behaviors.csv").is_file()
            )
            self.assertIn("episodic_human_prior_memory_seeds", summary)
            self.assertIn(
                "episodic_human_prior_seeded_graph_states", summary
            )
            self.assertIn(
                "episodic_human_prior_seeded_player_positions", summary
            )
            self.assertIn(
                "episodic_human_prior_seeded_option_paths", summary
            )
            self.assertIn(
                "episodic_human_prior_seeded_temporal_options", summary
            )
            self.assertIn("option_archive_snapshots_stored", summary)
            self.assertIn(
                "episodic_option_archive_state_imports", summary
            )
            self.assertIn(
                "episodic_option_archive_seed_events", summary
            )
            self.assertIn("episodic_option_archives_seeded", summary)
            self.assertIn("episodic_option_archives_skipped", summary)
            self.assertIn("human_prior_option_archives_added", summary)
            self.assertIn("human_prior_options_committed", summary)
            self.assertIn("life_hazard_checkpoints_created", summary)
            self.assertIn("life_hazard_checkpoint_restores", summary)
            self.assertIn("goal_milestone_checkpoints_created", summary)
            self.assertIn("goal_milestone_checkpoint_restores", summary)
            self.assertIn("goal_milestone_exhaustions_learned", summary)
            self.assertIn("goal_milestone_exhaustion_restores", summary)
            self.assertIn(
                "goal_milestone_preparation_filter_evaluations", summary
            )
            self.assertIn(
                "goal_milestone_preparation_branches_filtered", summary
            )
            self.assertIn(
                "goal_milestone_preparation_precursors_filtered", summary
            )
            self.assertIn(
                "goal_milestone_preparation_filter_fail_opens", summary
            )
            self.assertIn(
                "goal_milestone_preparation_archives_preserved", summary
            )
            self.assertIn(
                "goal_milestone_preparation_archive_filter_events", summary
            )
            self.assertIn(
                "goal_milestone_preparation_archives_filtered", summary
            )
            self.assertIn(
                "goal_milestone_preparation_option_endpoints_rejected",
                summary,
            )
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
