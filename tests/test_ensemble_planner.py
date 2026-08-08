import unittest
import tempfile
from pathlib import Path

import torch

from lolo_agent.ensemble_world_model import (
    EnsembleVisualDynamicsModel,
    VisualSequence,
    train_ensemble_model,
    validate_ensemble_model,
    load_ensemble_checkpoint,
    save_ensemble_checkpoint,
)
from lolo_agent.environment import Action
from lolo_agent.mock_puzzle import MockPuzzleEnv
from lolo_agent.neural_planner import (
    NeuralPlan,
    NeuralPlanningConfig,
    VerifiedNeuralAgent,
    _ArchivedBranch,
    _BehaviorProbeSelection,
    _OptionCounterfactual,
    _TemporalOptionTrace,
)
from lolo_agent.pixels import Frame


class AutonomousAnimationEnv:
    def __init__(self) -> None:
        self.tick = 0

    def reset(self) -> Frame:
        self.tick = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        self.tick += frames
        return self._frame()

    def save_state(self) -> int:
        return self.tick

    def load_state(self, state: int) -> Frame:
        self.tick = state
        return self._frame()

    def _frame(self) -> Frame:
        return Frame(8, 8, 1, bytes([self.tick % 256]) * 64)


class AnimationPauseEnv:
    def __init__(self) -> None:
        self.tick = 0

    def reset(self) -> Frame:
        self.tick = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if self.tick >= 4 and action == Action.A:
            self.tick = 255
        elif self.tick < 4:
            self.tick = min(4, self.tick + frames)
        return self._frame()

    def save_state(self) -> int:
        return self.tick

    def load_state(self, state: int) -> Frame:
        self.tick = state
        return self._frame()

    def _frame(self) -> Frame:
        return Frame(
            8,
            8,
            1,
            bytes((index + self.tick) % 256 for index in range(64)),
        )


class DelayedCausalityEnv:
    def __init__(self) -> None:
        self.triggered = False
        self.tick = 0
        self.released = 0

    def reset(self) -> Frame:
        self.triggered = False
        self.tick = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if action == Action.START and self.tick == 0:
            self.triggered = True
        elif action == Action.NOOP:
            self.tick += frames
        return self._frame()

    def save_state(self) -> tuple[bool, int]:
        return self.triggered, self.tick

    def load_state(self, state: tuple[bool, int]) -> Frame:
        self.triggered, self.tick = state
        return self._frame()

    def release_state(self, state: tuple[bool, int]) -> None:
        self.released += 1

    def _frame(self) -> Frame:
        value = 255 if self.triggered and self.tick > 0 else 0
        return Frame(8, 8, 1, bytes([value]) * 64)


class ActionEffectEnv:
    def __init__(self) -> None:
        self.position = 0

    def reset(self) -> Frame:
        self.position = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        if action == Action.RIGHT:
            self.position = min(63, self.position + frames)
        elif action == Action.SELECT:
            self.position = 63
        return self._frame()

    def save_state(self) -> int:
        return self.position

    def load_state(self, state: int) -> Frame:
        self.position = state
        return self._frame()

    def _frame(self) -> Frame:
        pixels = bytearray(64)
        pixels[self.position] = 255
        return Frame(8, 8, 1, bytes(pixels))


class UniqueStateEnv(ActionEffectEnv):
    def __init__(self) -> None:
        super().__init__()
        self.serial = 0
        self.active_states = set()

    def reset(self) -> Frame:
        self.active_states = set()
        return super().reset()

    def save_state(self) -> tuple[int, int]:
        self.serial += 1
        state = (self.serial, self.position)
        self.active_states.add(state)
        return state

    def load_state(self, state: tuple[int, int]) -> Frame:
        if state not in self.active_states:
            raise RuntimeError("unknown save-state handle")
        self.position = state[1]
        return self._frame()

    def release_state(self, state: tuple[int, int]) -> None:
        if state not in self.active_states:
            raise RuntimeError("unknown save-state handle")
        self.active_states.remove(state)


class EnsemblePlannerTests(unittest.TestCase):
    def frame(self, offset: int) -> Frame:
        return Frame(32, 32, 3, bytes((index + offset) % 256 for index in range(32 * 32 * 3)))

    def test_multistep_training_and_validation(self) -> None:
        sequences = [
            VisualSequence(
                group=index,
                frames=(self.frame(index), self.frame(index + 1), self.frame(index + 2)),
                actions=(Action.RIGHT, Action.DOWN),
            )
            for index in range(2)
        ]
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        history = train_ensemble_model(model, sequences, "cpu", epochs=1, batch_size=2)
        report = validate_ensemble_model(model, sequences, "cpu", batch_size=2)
        self.assertTrue(history)
        self.assertEqual(len(report.horizon_pixel_l1), 2)
        self.assertEqual(len(report.horizon_uncertainty), 2)
        self.assertTrue(all(value >= 0 for value in report.horizon_uncertainty))

    def test_verified_planner_preserves_frozen_model(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                planning_depth=2,
                beam_width=4,
                verify_actions=2,
                action_frames=1,
            ),
        )
        before = model.checkpoint_digest
        agent.reset()
        decision = agent.decide()
        self.assertIn(decision.action, (Action.LEFT, Action.RIGHT))
        self.assertEqual(decision.branches_examined, 2)
        self.assertEqual(before, model.checkpoint_digest)

    def test_checkpoint_round_trip_is_frozen(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.pt"
            digest = save_ensemble_checkpoint(model, path, planning_horizon=3)
            loaded, horizon = load_ensemble_checkpoint(path, frozen=True)
        self.assertEqual(horizon, 3)
        self.assertEqual(digest, loaded.checkpoint_digest)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))

    def test_temporary_action_coverage_breaks_repetition(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = MockPuzzleEnv()
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
                action_coverage_weight=10.0,
                consecutive_repeat_weight=10.0,
            ),
        )
        agent.reset()
        actions = [decision.action for decision in agent.run(2)]
        self.assertEqual(set(actions), {Action.LEFT, Action.RIGHT})

    def test_duration_coverage_is_scoped_to_the_controller_action(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.UP),
                planning_depth=1,
                duration_coverage_weight=1.0,
            ),
        )
        agent.reset()
        agent.action_duration_counts[(Action.NOOP, 16)] = 100

        self.assertEqual(agent._action_penalty(Action.UP, 16), 0.0)
        self.assertEqual(agent._action_penalty(Action.NOOP, 16), 10.0)

    def test_matched_noop_branch_prioritizes_discovered_control(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.A, Action.RIGHT),
                planning_depth=1,
                beam_width=3,
                verify_actions=3,
                action_frames=4,
                actual_novelty_weight=0.0,
                scene_novelty_weight=0.0,
                prediction_error_weight=0.0,
                actual_change_weight=0.0,
                action_effect_weight=1.0,
                action_coverage_weight=0.0,
                duration_coverage_weight=0.0,
                consecutive_repeat_weight=0.0,
            ),
        )
        agent.reset()
        plans = [
            NeuralPlan((action,), (4,), 0.0, 0.0)
            for action in (Action.NOOP, Action.A, Action.RIGHT)
        ]
        agent.planner.plan = lambda _frame: plans

        decision = agent.decide()

        self.assertEqual(decision.action, Action.RIGHT)
        sources = {source for source, _action in agent.action_effect_samples}
        self.assertEqual(len(sources), 1)
        source = sources.pop()
        self.assertEqual(agent._action_effect_estimate(source, Action.A)[0], 0.0)
        self.assertEqual(agent._action_effect_estimate(source, Action.RIGHT)[0], 1.0)
        self.assertEqual(sum(agent.causal_spatial_visits.values()), 1)

    def test_causal_option_commits_a_neutral_observation_before_intervening(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=9, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            AutonomousAnimationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.RIGHT, Action.SELECT),
                planning_depth=1,
                beam_width=3,
                verify_actions=3,
                action_frames=4,
                actual_novelty_weight=0.0,
                scene_novelty_weight=0.0,
                prediction_error_weight=0.0,
                actual_change_weight=0.0,
                action_effect_weight=0.0,
                causal_spatial_novelty_weight=1.0,
                action_coverage_weight=0.0,
                duration_coverage_weight=0.0,
                consecutive_repeat_weight=0.0,
            ),
        )
        agent.reset()
        agent.pending_option_choice = ("source", Action.RIGHT, 4)
        agent.pending_option_decision = 0
        agent.pending_option_causal_evidence = True
        plans = [
            NeuralPlan((Action.NOOP,), (4,), 0.0, 0.0),
            NeuralPlan((Action.RIGHT,), (4,), 2.0, 0.0),
            NeuralPlan((Action.SELECT,), (4,), 1.0, 0.0),
        ]
        agent.planner.plan = lambda _frame: plans

        decision = agent.decide()

        self.assertEqual(decision.action, Action.NOOP)

    def test_causal_spatial_signature_localizes_matched_pixel_change(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                causal_spatial_columns=1,
                causal_spatial_rows=1,
            ),
        )
        neutral_pixels = bytearray(64)
        neutral_pixels[0] = 255
        factual_pixels = bytearray(64)
        factual_pixels[4] = 255

        signature, changed_pixels, centroid = agent._causal_spatial_effect(
            Frame(8, 8, 1, bytes(factual_pixels)),
            Frame(8, 8, 1, bytes(neutral_pixels)),
        )

        self.assertIsNotNone(signature)
        self.assertEqual(changed_pixels, 2)
        self.assertEqual(centroid, (2.0, 0.0))

    def test_learned_hazard_is_verified_but_not_committed_when_safe(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=9, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            ActionEffectEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.SELECT),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                action_frames=4,
                actual_novelty_weight=0.0,
                scene_novelty_weight=0.0,
                prediction_error_weight=0.0,
                actual_change_weight=0.0,
                action_effect_weight=10.0,
                action_coverage_weight=0.0,
                duration_coverage_weight=0.0,
                consecutive_repeat_weight=0.0,
            ),
        )
        agent.reset()
        agent._record_temporal_option_sample(
            ("prior-state", Action.SELECT, 1),
            -2.0,
            generalize_action_hazard=True,
        )
        plans = [
            NeuralPlan((action,), (4,), 0.0, 0.0)
            for action in (Action.NOOP, Action.SELECT)
        ]
        agent.planner.plan = lambda _frame: plans

        decision = agent.decide()

        self.assertEqual(decision.branches_examined, 2)
        self.assertEqual(decision.action, Action.NOOP)
        self.assertFalse(any(branch.plan.path[0] == Action.SELECT for branch in agent.archive))

    def test_same_decision_archive_pruning_releases_each_state_once(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=9, ensemble_size=2)
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.RIGHT, Action.SELECT),
                planning_depth=1,
                beam_width=3,
                verify_actions=3,
                action_frames=4,
                archive_capacity=1,
            ),
        )
        agent.reset()
        plans = [
            NeuralPlan((action,), (4,), 0.0, 0.0)
            for action in (Action.NOOP, Action.RIGHT, Action.SELECT)
        ]
        agent.planner.plan = lambda _frame: plans

        decision = agent.decide()

        self.assertEqual(decision.branches_examined, 3)
        self.assertLessEqual(len(agent.archive), 1)

    def test_stagnation_restores_an_alternative_branch(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                visual_stagnation_visits=1,
            ),
        )
        agent.reset()
        first = agent.decide()
        agent.visual_stagnation_streak = 1
        second = agent.decide()
        self.assertFalse(first.restored_archive)
        self.assertTrue(second.restored_archive)

    def test_duration_conditioned_planner_selects_a_press_length(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                action_durations=(1, 3),
                planning_depth=1,
                beam_width=4,
                verify_actions=4,
            ),
        )
        agent.reset()
        decision = agent.decide()
        self.assertIn(decision.action_frames, (1, 3))
        self.assertEqual(decision.planned_durations, (decision.action_frames,))
        self.assertEqual(decision.branches_examined, 4)

    def test_duration_checkpoint_round_trip(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=16,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duration-ensemble.pt"
            digest = save_ensemble_checkpoint(model, path, planning_horizon=2)
            loaded, horizon = load_ensemble_checkpoint(path, frozen=True)
        self.assertEqual(horizon, 2)
        self.assertEqual(digest, loaded.checkpoint_digest)
        self.assertTrue(loaded.duration_conditioned)
        self.assertEqual(loaded.max_action_frames, 16)

    def test_loaded_fixed_duration_checkpoint_rejects_a_different_duration(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            fixed_action_frames=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed-ensemble.pt"
            save_ensemble_checkpoint(model, path, planning_horizon=1)
            loaded, _ = load_ensemble_checkpoint(path, frozen=True)
        with self.assertRaises(ValueError):
            VerifiedNeuralAgent(
                MockPuzzleEnv(),
                loaded,
                "cpu",
                NeuralPlanningConfig(
                    actions=(Action.RIGHT,), planning_depth=1, action_frames=8
                ),
            )

    def test_uncontrollable_animation_selects_long_noop_without_archiving(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        agent = VerifiedNeuralAgent(
            AutonomousAnimationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.A, Action.NOOP),
                action_durations=(1, 4),
                planning_depth=1,
                beam_width=4,
                verify_actions=4,
            ),
        )
        agent.reset()
        decision = agent.decide()
        self.assertEqual(decision.action, Action.NOOP)
        self.assertEqual(decision.action_frames, 4)
        self.assertEqual(agent.archive, [])

    def test_autonomous_grace_ends_when_action_dependent_control_returns(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        agent = VerifiedNeuralAgent(
            AnimationPauseEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.A, Action.NOOP),
                action_durations=(1, 4),
                planning_depth=1,
                beam_width=4,
                verify_actions=4,
                autonomous_grace_decisions=2,
            ),
        )
        agent.reset()
        moving = agent.decide()
        controlled = agent.decide()
        self.assertEqual(moving.action, Action.NOOP)
        self.assertEqual(controlled.action, Action.A)
        self.assertEqual(agent.frame.pixels[0], 255)
        self.assertEqual(agent.autonomous_grace_remaining, 0)

    def test_archive_pruning_preserves_a_minority_scene(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                archive_capacity=3,
            ),
        )
        frame = Frame(2, 2, 1, b"\x00" * 4)
        plan = NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0)
        agent.archive = [
            _ArchivedBranch(index, frame, plan, float(index), "crowded", index)
            for index in range(4)
        ] + [_ArchivedBranch(99, frame, plan, 1.0, "minority", 0)]
        agent._prune_archive()
        self.assertEqual(len(agent.archive), 3)
        self.assertIn("minority", {branch.scene for branch in agent.archive})

    def test_archive_score_prefers_a_rare_causal_spatial_frontier(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        frame = agent.reset()
        plan = NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0)
        ordinary = _ArchivedBranch(1, frame, plan, 0.0, "scene", 1)
        spatial = _ArchivedBranch(
            2,
            frame,
            plan,
            0.0,
            "scene",
            1,
            causal_spatial_signature="new-grid-cell",
        )

        self.assertGreater(
            agent._archive_frontier_score(spatial),
            agent._archive_frontier_score(ordinary),
        )
        agent.causal_spatial_visits["new-grid-cell"] = 3
        self.assertEqual(agent._archive_causal_spatial_bonus(spatial), 1.0)

    def test_stagnation_can_restore_a_causal_frontier_in_the_same_scene(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = UniqueStateEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
            ),
        )
        current = agent.reset()
        state = env.save_state()
        alternative = Frame(8, 8, 1, bytes([0, 255]) + bytes(62))
        scene = agent._scene_signature(current)
        agent.archive.append(
            _ArchivedBranch(
                state=state,
                frame=alternative,
                plan=NeuralPlan((Action.RIGHT,), (1,), 1.0, 0.0),
                score=1.0,
                scene=scene,
                created=0,
                origin_signature=agent.current_frontier_signature,
                frontier_signature="causal-frontier",
                causal_spatial_signature="changed-cell",
            )
        )
        agent.visual_stagnation_streak = 1

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, alternative.digest)

    def test_verification_budget_covers_distinct_buttons_before_durations(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=8,
        )
        actions = (Action.LEFT, Action.RIGHT, Action.A)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=actions,
                action_durations=(1, 4),
                planning_depth=1,
                verify_actions=3,
            ),
        )
        agent.reset()
        source_scene = agent.current_scene
        decision = agent.decide()
        self.assertEqual(decision.branches_examined, 3)
        self.assertEqual(
            {action for scene, action in agent.scene_action_probes if scene == source_scene},
            set(actions),
        )

    def test_matched_control_probes_reserve_canonical_behavior_slots(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=16,
        )
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                action_durations=(1, 16),
                planning_depth=1,
                verify_actions=6,
            ),
        )
        ranked_actions = (
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.A,
            Action.B,
        )
        ranked = [
            NeuralPlan((action,), (1,), float(index), 0.0)
            for index, action in enumerate(ranked_actions)
        ]
        best = {
            (plan.path[0], plan.durations[0]): plan for plan in ranked
        }
        noop = NeuralPlan((Action.NOOP,), (16,), 0.0, 0.0)
        up = NeuralPlan((Action.UP,), (16,), 0.0, 0.0)
        right_long = NeuralPlan((Action.RIGHT,), (16,), 0.0, 0.0)
        start = NeuralPlan((Action.START,), (16,), 100.0, 0.0)
        best[(Action.NOOP, 16)] = noop
        best[(Action.UP, 16)] = up
        best[(Action.RIGHT, 16)] = right_long
        best[(Action.START, 16)] = start

        agent.reset()
        agent.last_action = Action.RIGHT
        agent.last_action_was_causal_spatial = True
        result = agent._add_control_probes(ranked, best)

        self.assertEqual(len(result), 6)
        self.assertIn(noop, result)
        self.assertIn(up, result)
        self.assertIn(right_long, result)
        self.assertNotIn(start, result)

    def test_causal_continuation_survives_the_neutral_observation(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32,
            action_size=8,
            ensemble_size=2,
            duration_conditioned=True,
            duration_size=4,
            max_action_frames=16,
        )
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                action_durations=(1, 16),
                planning_depth=1,
                verify_actions=6,
            ),
        )
        agent.reset()
        agent.active_temporal_option = _TemporalOptionTrace(
            choice=("source", Action.RIGHT, 16),
            initiation_decision=1,
            start_decision=2,
            entry_signature="source",
            entry_scene="scene",
            causal_evidence=True,
        )
        ranked = [
            NeuralPlan((Action.UP,), (1,), 0.0, 0.0),
            NeuralPlan((Action.DOWN,), (1,), 0.0, 0.0),
        ]
        right_long = NeuralPlan((Action.RIGHT,), (16,), 0.0, 0.0)
        best = {
            (Action.UP, 1): ranked[0],
            (Action.DOWN, 1): ranked[1],
            (Action.NOOP, 16): NeuralPlan((Action.NOOP,), (16,), 0.0, 0.0),
            (Action.RIGHT, 16): right_long,
        }

        result = agent._add_control_probes(ranked, best)

        self.assertIn(right_long, result)

    def test_active_behavior_probe_rotates_then_separates_hypotheses(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.UP, Action.DOWN),
                planning_depth=1,
                abstraction_latent_rmse_threshold=100.0,
                behavioral_abstraction_rmse_threshold=1e-9,
            ),
        )
        source = agent.reset()
        outcomes = {
            (Action.NOOP, 4): self.frame(10),
            (Action.UP, 4): self.frame(11),
            (Action.DOWN, 4): self.frame(12),
        }

        first = agent._behavior_probe_selection(source)
        self.assertEqual(first.reason, "coverage_rotation")
        self.assertEqual(first.selected_control, Action.UP)
        first_cluster = agent._behavioral_signature(
            source, outcomes, agent.current_frontier_signature, first
        )

        second = agent._behavior_probe_selection(source)
        self.assertEqual(second.selected_control, Action.DOWN)
        self.assertEqual(
            agent._behavioral_signature(
                source,
                outcomes,
                agent._new_provisional_signature(),
                second,
            ),
            first_cluster,
        )
        self.assertIn(
            (Action.DOWN, 4),
            agent.behavior_clusters[0].probe_centroids,
        )

        third = agent._behavior_probe_selection(source)
        self.assertEqual(third.selected_control, Action.UP)
        split_cluster = agent._behavioral_signature(
            source,
            {
                (Action.NOOP, 4): outcomes[(Action.NOOP, 4)],
                (Action.UP, 4): self.frame(200),
            },
            agent._new_provisional_signature(),
            third,
        )
        self.assertNotEqual(split_cluster, first_cluster)

        discriminating = agent._behavior_probe_selection(source)
        self.assertEqual(discriminating.reason, "hypothesis_separation")
        self.assertEqual(discriminating.selected_control, Action.UP)
        self.assertGreater(discriminating.hypothesis_separation, 0.0)

        ambiguous = _BehaviorProbeSelection(
            ((Action.NOOP, 4), (Action.RIGHT, 4)),
            discriminating.visual_cluster,
            "coverage_rotation",
            Action.RIGHT,
        )
        provisional = agent._new_provisional_signature()
        unresolved = agent._behavioral_signature(
            source,
            {
                (Action.NOOP, 4): outcomes[(Action.NOOP, 4)],
                (Action.RIGHT, 4): self.frame(13),
            },
            provisional,
            ambiguous,
        )
        self.assertEqual(unresolved, provisional)
        self.assertEqual(len(agent.behavior_clusters), 2)

    def test_delayed_visual_return_credits_the_loop_and_requests_recovery(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                planning_depth=1,
                delayed_return_min_length=3,
            ),
        )
        agent.reset()
        frames = [
            Frame(
                8,
                8,
                1,
                bytes((((x + y + offset) % 8) * 30) for y in range(8) for x in range(8)),
            )
            for offset in range(3)
        ]
        agent.visual_last_visit = {}
        transitions = (
            (1, "scene-a", Action.RIGHT, frames[0]),
            (2, "scene-b", Action.DOWN, frames[1]),
            (3, "scene-c", Action.LEFT, frames[2]),
            (4, "scene-d", Action.UP, frames[0]),
        )
        for decision, source_scene, action, target in transitions:
            agent.decision_index = decision
            agent._update_persistent_frontier(agent._signature(target), 1.0)
            agent._record_delayed_return(
                source_scene,
                action,
                4,
                target,
                agent._scene_signature(target),
            )

        self.assertTrue(agent.delayed_return_recovery)
        self.assertEqual(agent.delayed_return_loop_start, 1)
        self.assertEqual(sum(agent.delayed_return_costs.values()), 3)
        self.assertEqual(agent.delayed_return_costs[("scene-d", Action.UP, 4)], 1)
        self.assertLess(agent.frontier_values[agent._signature(frames[0])], 0.0)

    def test_persistent_frontier_accumulates_discounted_successor_novelty(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                frontier_credit_horizon=3,
                frontier_discount=1.0,
            ),
        )
        initial = agent.reset()
        initial_signature = agent.current_frontier_signature
        for decision, signature in enumerate(("one", "two", "three"), 1):
            agent.decision_index = decision
            agent._update_persistent_frontier(signature, 1.0)

        self.assertEqual(agent.frontier_values[initial_signature], 3.0)
        self.assertEqual(agent._frontier_estimate("one"), 2.0)
        self.assertEqual(agent._frontier_estimate("two"), 1.0)

    def test_repeated_identical_pixels_do_not_create_frontier_reward(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP,),
                planning_depth=1,
                verify_actions=1,
                action_frames=1,
            ),
        )
        initial = agent.reset()
        agent.run(3)

        self.assertEqual(
            agent._frontier_estimate(agent._abstract_signature(initial)), 0.0
        )
        self.assertTrue(
            all(trace.discounted_return == 0.0 for trace in agent.frontier_traces)
        )

    def test_frozen_encoder_clusters_nearby_frames_and_shares_choice_value(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                abstraction_latent_rmse_threshold=100.0,
            ),
        )
        before = model.checkpoint_digest
        agent.reset()
        first = self.frame(5)
        changed = bytearray(first.pixels)
        changed[0] = (changed[0] + 1) % 256
        nearby = Frame(first.width, first.height, first.channels, bytes(changed))
        different = Frame(32, 32, 3, b"\xff" * (32 * 32 * 3))
        self.assertEqual(
            agent._scene_signature(first), agent._scene_signature(nearby)
        )
        self.assertNotEqual(
            agent._scene_signature(first), agent._scene_signature(different)
        )

        first_cluster = agent._abstract_signature(first)
        nearby_cluster = agent._abstract_signature(nearby)
        different_cluster = agent._abstract_signature(different)
        self.assertEqual(first_cluster, nearby_cluster)
        self.assertNotEqual(first_cluster, different_cluster)
        choice = (first_cluster, Action.RIGHT, 4)
        agent.frontier_choice_values[choice] = 3.0
        agent.frontier_choice_samples[choice] = 1
        shared, known = agent._choice_frontier_estimate(
            nearby_cluster, Action.RIGHT, 4
        )
        self.assertTrue(known)
        self.assertEqual(shared, 3.0)
        self.assertEqual(before, model.checkpoint_digest)

    def test_behavioral_abstraction_shares_only_matching_observed_futures(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.LEFT, Action.RIGHT),
                planning_depth=1,
                abstraction_latent_rmse_threshold=100.0,
                behavioral_abstraction_rmse_threshold=1e-9,
            ),
        )
        before = model.checkpoint_digest
        source = agent.reset()
        matching_outcomes = {
            (Action.LEFT, 4): self.frame(6),
            (Action.RIGHT, 4): self.frame(7),
        }
        first = agent._behavioral_signature(
            source, matching_outcomes, agent.current_frontier_signature
        )
        choice = (first, Action.LEFT, 4)
        agent.frontier_choice_values[choice] = 3.0
        agent.frontier_choice_samples[choice] = 1

        second_provisional = agent._new_provisional_signature()
        value, known = agent._choice_frontier_estimate(
            second_provisional, Action.LEFT, 4
        )
        self.assertFalse(known)
        self.assertEqual(value, 0.0)
        agent.frontier_values[second_provisional] = 2.0
        agent.frontier_samples[second_provisional] = 1
        second = agent._behavioral_signature(
            source, matching_outcomes, second_provisional
        )
        self.assertEqual(first, second)
        self.assertEqual(agent.frontier_values[first], 2.0)
        self.assertNotIn(second_provisional, agent.frontier_values)
        value, known = agent._choice_frontier_estimate(second, Action.LEFT, 4)
        self.assertTrue(known)
        self.assertEqual(value, 3.0)

        different = agent._behavioral_signature(
            source,
            {
                (Action.LEFT, 4): self.frame(200),
                (Action.RIGHT, 4): self.frame(201),
            },
            agent._new_provisional_signature(),
        )
        self.assertNotEqual(first, different)
        value, known = agent._choice_frontier_estimate(different, Action.LEFT, 4)
        self.assertFalse(known)
        self.assertEqual(value, 0.0)
        self.assertEqual(before, model.checkpoint_digest)

    def test_frontier_choice_value_learns_a_delayed_return_outcome(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        agent.reset()
        agent.decision_index = 1
        agent._update_persistent_frontier(
            "target", 1.0, "source", Action.RIGHT, 4
        )
        provisional, known = agent._choice_frontier_estimate(
            "source", Action.RIGHT, 4
        )
        self.assertTrue(known)
        self.assertEqual(provisional, 1.0)

        agent._penalize_frontier_loop(1)

        learned, known = agent._choice_frontier_estimate(
            "source", Action.RIGHT, 4
        )
        self.assertTrue(known)
        self.assertEqual(learned, -agent.config.frontier_return_penalty)

    def test_known_bad_archive_choice_overrides_inherited_origin_value(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        frame = agent.reset()
        plan = NeuralPlan((Action.RIGHT,), (4,), 0.0, 0.0)
        branch = _ArchivedBranch(
            b"state", frame, plan, 0.0, "scene", 1, "valuable-origin"
        )
        agent.frontier_values["valuable-origin"] = 10.0
        self.assertGreater(agent._archive_frontier_score(branch), 0.0)
        choice = ("valuable-origin", Action.RIGHT, 4)
        agent.frontier_choice_values[choice] = -2.0
        agent.frontier_choice_samples[choice] = 1

        self.assertEqual(agent._archive_frontier_score(branch), -2.0)
        agent.temporal_option_values[choice] = 2.0
        agent.temporal_option_samples[choice] = 1
        self.assertEqual(
            agent._archive_frontier_score(branch),
            -2.0 + 2.0 * agent.config.temporal_option_score_weight,
        )

    def test_temporal_option_credits_an_initiating_action_through_passive_dynamics(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.START,), planning_depth=1),
        )
        before = model.checkpoint_digest
        agent.reset()
        choice = ("source", Action.START, 4)
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_causal_evidence = True

        agent.decision_index = 1
        agent._advance_temporal_option("animation-a", "scene-a", passive=True)
        agent.decision_index = 2
        agent._advance_temporal_option(
            "animation-b", "scene-b", passive=True, grace_continuation=True
        )
        self.assertIsNotNone(agent.active_temporal_option)
        self.assertEqual(agent.active_temporal_option.passive_decisions, 2)

        agent.decision_index = 3
        agent._advance_temporal_option("endpoint", "scene-c", passive=False)
        learned, known = agent._temporal_option_estimate(*choice)
        self.assertTrue(known)
        self.assertGreater(learned, 1.0)
        self.assertEqual(agent.temporal_option_samples[choice], 1)
        self.assertIsNone(agent.active_temporal_option)
        self.assertEqual(before, model.checkpoint_digest)

    def test_temporal_option_penalizes_a_return_to_its_source(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.A,), planning_depth=1),
        )
        agent.reset()
        choice = ("source", Action.A, 1)
        agent.behavior_visits["source"] = 1
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_causal_evidence = True
        for decision in range(1, 5):
            agent.decision_index = decision
            agent._advance_temporal_option(
                f"animation-{decision}",
                f"scene-{decision}",
                passive=True,
            )
        agent.decision_index = 5
        agent._advance_temporal_option("source", "scene", passive=False)

        learned, known = agent._temporal_option_estimate(*choice)
        self.assertTrue(known)
        self.assertLess(learned, 0.0)

    def test_single_neutral_observation_is_not_a_delayed_return_hazard(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.RIGHT,), planning_depth=1),
        )
        agent.reset()
        choice = ("source", Action.RIGHT, 16)
        agent.behavior_visits["source"] = 1
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_causal_evidence = True
        agent.decision_index = 1
        agent._advance_temporal_option("source", "room", passive=True)
        agent.decision_index = 2
        agent._advance_temporal_option("source", "room", passive=False)

        learned, known = agent._temporal_option_estimate(*choice)
        self.assertTrue(known)
        self.assertGreaterEqual(learned, 0.0)

    def test_temporal_option_penalizes_a_return_to_an_earlier_known_state(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.SELECT,), planning_depth=1),
        )
        agent.reset()
        choice = ("moved-state", Action.SELECT, 1)
        agent.behavior_visits["earlier-state"] = 1
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_causal_evidence = True
        for decision in range(1, 5):
            agent.decision_index = decision
            agent._advance_temporal_option(
                f"animation-{decision}",
                f"fade-{decision}",
                passive=True,
            )
        agent.decision_index = 5
        agent._advance_temporal_option("earlier-state", "room", passive=False)

        learned, known = agent._temporal_option_estimate(*choice)
        self.assertTrue(known)
        self.assertLess(learned, 0.0)

    def test_robust_direct_causal_return_generalizes_action_hazard(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=9, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.SELECT,), planning_depth=1),
        )
        agent.reset()
        choice = ("source", Action.SELECT, 1)
        agent.behavior_visits["known-endpoint"] = 1
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_causal_evidence = True
        for decision in range(1, 5):
            agent.decision_index = decision
            agent._advance_temporal_option(
                f"animation-{decision}",
                f"scene-{decision}",
                passive=True,
            )
        agent.decision_index = 5
        agent._advance_temporal_option(
            "known-endpoint", "endpoint-scene", passive=False
        )

        self.assertEqual(agent.temporal_option_action_samples[Action.SELECT], 1)
        self.assertLess(agent.temporal_option_action_values[Action.SELECT], 0.0)

    def test_temporal_option_action_prior_generalizes_unseen_state_and_duration(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.SELECT,), planning_depth=1),
        )
        agent.reset()
        exact_choice = ("moved-state", Action.SELECT, 1)
        agent._record_temporal_option_sample(
            exact_choice, -2.0, generalize_action_hazard=True
        )

        exact, exact_known = agent._temporal_option_estimate(*exact_choice)
        inherited, inherited_known = agent._temporal_option_estimate(
            "earlier-state", Action.SELECT, 8
        )

        self.assertTrue(exact_known)
        self.assertEqual(exact, -2.0)
        self.assertTrue(inherited_known)
        self.assertEqual(
            inherited,
            -2.0 * agent.config.temporal_option_action_prior_weight,
        )

        ordinary_choice = ("another-state", Action.UP, 16)
        agent._record_temporal_option_sample(ordinary_choice, -3.0)
        self.assertEqual(
            agent._temporal_option_estimate("unseen-state", Action.UP, 4),
            (0.0, False),
        )

    def test_temporal_option_requires_action_dependent_counterfactual_evidence(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.START, Action.NOOP), planning_depth=1),
        )
        identical = self.frame(10)
        changed = self.frame(200)
        start = NeuralPlan((Action.START,), (4,), 0.0, 0.0)
        noop = NeuralPlan((Action.NOOP,), (4,), 0.0, 0.0)
        candidate = (0.0, start, b"start", identical)
        uncaused = (0.0, noop, b"noop", identical)
        caused = (0.0, noop, b"noop", changed)

        eligible, contrast, count = agent._option_initiation_evidence(
            candidate, [candidate, uncaused]
        )
        self.assertFalse(eligible)
        self.assertEqual(contrast, 0.0)
        self.assertEqual(count, 1)
        self.assertIs(
            agent._delayed_option_counterfactual(
                candidate, [candidate, uncaused]
            ),
            uncaused,
        )

        eligible, contrast, count = agent._option_initiation_evidence(
            candidate, [candidate, caused]
        )
        self.assertTrue(eligible)
        self.assertGreater(contrast, agent.config.action_equivalence_threshold)
        self.assertEqual(count, 1)
        self.assertIsNone(
            agent._delayed_option_counterfactual(candidate, [candidate, caused])
        )

    def test_delayed_counterfactual_divergence_supplies_causal_option_evidence(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = DelayedCausalityEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.START, Action.A, Action.NOOP), planning_depth=1
            ),
        )
        agent.reset()
        root = env.save_state()
        immediate = env.step(Action.START)
        factual_state = env.save_state()
        env.load_state(root)
        counterfactual_frame = env.step(Action.A)
        counterfactual_state = env.save_state()
        self.assertEqual(immediate, counterfactual_frame)
        env.load_state(factual_state)
        factual_target = env.step(Action.NOOP)

        choice = ("source", Action.START, 1)
        agent.pending_option_choice = choice
        agent.pending_option_decision = 1
        agent.pending_option_counterfactual = _OptionCounterfactual(
            counterfactual_state,
            counterfactual_frame,
            Action.A,
            1,
        )
        agent.decision_index = 1
        agent._advance_temporal_option(
            "animation",
            "scene-a",
            passive=True,
            passive_action=Action.NOOP,
            passive_duration=1,
            factual_target=factual_target,
        )
        self.assertFalse(agent.active_temporal_option.causal_evidence)
        self.assertGreater(
            agent.active_temporal_option.counterfactual.maximum_contrast,
            agent.config.action_equivalence_threshold,
        )

        agent.frame = factual_target
        agent.decision_index = 2
        agent._advance_temporal_option("endpoint", "scene-b", passive=False)
        learned, known = agent._temporal_option_estimate(*choice)
        self.assertTrue(known)
        self.assertGreater(learned, 0.0)
        self.assertIsNone(agent.active_temporal_option)
        self.assertGreaterEqual(env.released, 2)

    def test_delayed_counterfactual_requires_endpoint_divergence(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = DelayedCausalityEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.START,), planning_depth=1),
        )
        endpoint = agent.reset()
        choice = ("source", Action.START, 1)
        agent.active_temporal_option = _TemporalOptionTrace(
            choice=choice,
            initiation_decision=1,
            start_decision=2,
            entry_signature="animation",
            entry_scene="scene",
            counterfactual=_OptionCounterfactual(
                env.save_state(), endpoint, Action.A, 1
            ),
            passive_decisions=3,
        )
        agent.frame = endpoint
        agent.decision_index = 4
        agent._advance_temporal_option("endpoint", "scene", passive=False)

        learned, known = agent._temporal_option_estimate(*choice)
        self.assertFalse(known)
        self.assertEqual(learned, 0.0)

    def test_new_delayed_intervention_supersedes_active_option_credit(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        agent = VerifiedNeuralAgent(
            MockPuzzleEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(actions=(Action.UP, Action.SELECT), planning_depth=1),
        )
        agent.reset()
        prior_choice = ("source", Action.UP, 16)
        agent.active_temporal_option = _TemporalOptionTrace(
            choice=prior_choice,
            initiation_decision=1,
            start_decision=2,
            entry_signature="animation",
            entry_scene="scene",
            causal_evidence=True,
        )

        superseded = agent._supersede_temporal_option_for_intervention(
            Action.SELECT, has_causal_candidate=True
        )

        self.assertTrue(superseded)
        self.assertIsNone(agent.active_temporal_option)
        self.assertEqual(agent._temporal_option_estimate(*prior_choice), (0.0, False))

        agent.active_temporal_option = _TemporalOptionTrace(
            choice=prior_choice,
            initiation_decision=1,
            start_decision=2,
            entry_signature="animation",
            entry_scene="scene",
        )
        self.assertFalse(
            agent._supersede_temporal_option_for_intervention(
                Action.NOOP, has_causal_candidate=True
            )
        )
        self.assertIsNotNone(agent.active_temporal_option)

    def test_archive_recovery_prefers_learned_persistent_frontier_value(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=1,
            ),
        )
        agent.reset()
        root = env.save_state()
        low_frame = env.step(Action.RIGHT)
        low_state = env.save_state()
        env.load_state(root)
        high_frame = env.step(Action.DOWN)
        high_state = env.save_state()
        low_plan = NeuralPlan((Action.RIGHT,), (1,), 100.0, 0.0)
        high_plan = NeuralPlan((Action.DOWN,), (1,), -100.0, 0.0)
        agent.archive = [
            _ArchivedBranch(
                low_state,
                low_frame,
                low_plan,
                100.0,
                "low-scene",
                1,
                "low-origin",
            ),
            _ArchivedBranch(
                high_state,
                high_frame,
                high_plan,
                -100.0,
                "high-scene",
                1,
                "high-origin",
            ),
        ]
        agent.frontier_values["low-origin"] = 1.0
        agent.frontier_values["high-origin"] = 10.0
        agent.visual_stagnation_streak = 1

        decision = agent._restore_if_stagnant()

        self.assertEqual(decision.frame.digest, high_frame.digest)

    def test_delayed_return_restores_a_distinct_branch_before_stagnation(self) -> None:
        model = EnsembleVisualDynamicsModel(latent_size=32, action_size=8, ensemble_size=2)
        env = MockPuzzleEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.RIGHT,),
                planning_depth=1,
                visual_stagnation_visits=99,
            ),
        )
        agent.reset()
        branch_frame = env.step(Action.RIGHT)
        branch_state = env.save_state()
        plan = NeuralPlan((Action.RIGHT,), (1,), 0.0, 0.0)
        agent.archive = [
            _ArchivedBranch(
                branch_state,
                branch_frame,
                plan,
                0.0,
                agent._scene_signature(branch_frame),
                2,
            )
        ]
        agent.delayed_return_recovery = True
        agent.delayed_return_loop_start = 1

        decision = agent._restore_if_stagnant()

        self.assertIsNotNone(decision)
        self.assertTrue(decision.restored_archive)
        self.assertFalse(agent.delayed_return_recovery)
