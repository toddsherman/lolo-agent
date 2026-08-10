import unittest

from lolo_agent.ensemble_world_model import EnsembleVisualDynamicsModel
from lolo_agent.environment import Action
from lolo_agent.goal_prior import (
    HEART_PROTOTYPE,
    OPEN_CHEST_EMPTY_PROTOTYPE,
    OPEN_CHEST_PROTOTYPE,
    PixelHeartGoalPrior,
)
from lolo_agent.neural_planner import (
    NeuralPlan,
    NeuralPlanningConfig,
    VerifiedNeuralAgent,
    _ArchivedBranch,
    _LifeHazardCheckpoint,
    _TemporalOptionTrace,
)
from lolo_agent.pixels import Frame


def room_frame(
    hearts=(),
    fill=(86, 29, 0),
    player=None,
    open_chest=None,
    life_glyph=None,
) -> Frame:
    width, height = 256, 240
    pixels = bytearray(fill * (width * height))
    for x, y in hearts:
        for row in range(16):
            for column in range(16):
                source = HEART_PROTOTYPE[row * 16 + column]
                offset = ((y + row) * width + x + column) * 3
                pixels[offset : offset + 3] = bytes(source)
    if open_chest is not None:
        x, y = open_chest
        for row in range(16):
            for column in range(16):
                source = OPEN_CHEST_PROTOTYPE[row * 16 + column]
                offset = ((y + row) * width + x + column) * 3
                pixels[offset : offset + 3] = bytes(source)
    if player is not None:
        x, y = player
        player_pixels = (
            [(21, 95, 217)] * 80
            + [(255, 255, 255)] * 60
            + [(0, 0, 0)] * 60
            + [(86, 29, 0)] * 56
        )
        for index, value in enumerate(player_pixels):
            row, column = divmod(index, 16)
            offset = ((y + row) * width + x + column) * 3
            pixels[offset : offset + 3] = bytes(value)
    if life_glyph is not None:
        for row, values in enumerate(life_glyph):
            for column, value in enumerate(values):
                colour = {
                    ".": (0, 0, 0),
                    "W": (255, 255, 255),
                    "M": (183, 30, 123),
                }[value]
                offset = ((48 + row) * width + 232 + column) * 3
                pixels[offset : offset + 3] = bytes(colour)
    return Frame(width, height, 3, bytes(pixels))


LIFE_FIVE = (
    "WWWWWWM.",
    "WWM.....",
    "WWWWWWM.",
    ".....WWM",
    ".....WWM",
    "WWM..WWM",
    ".WWWWWM.",
    "........",
)
LIFE_FOUR = (
    "WW...WWM",
    "WW...WWM",
    "WW...WWM",
    "WWWWWWWM",
    ".....WWM",
    ".....WWM",
    ".....WWM",
    "........",
)


class HeartRewardEnv:
    def __init__(self) -> None:
        self.heart_present = True

    def reset(self) -> Frame:
        self.heart_present = True
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.RIGHT:
            self.heart_present = False
        return self._frame()

    def save_state(self) -> bool:
        return self.heart_present

    def load_state(self, state: bool) -> Frame:
        self.heart_present = state
        return self._frame()

    def _frame(self) -> Frame:
        return room_frame(((48, 48),) if self.heart_present else ())


class HeartNavigationEnv:
    def __init__(self) -> None:
        self.player = (80, 192)

    def reset(self) -> Frame:
        self.player = (80, 192)
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.UP:
            self.player = (80, 176)
        elif action == Action.RIGHT:
            self.player = (96, 192)
        return self._frame()

    def save_state(self):
        return self.player

    def load_state(self, state) -> Frame:
        self.player = state
        return self._frame()

    def _frame(self) -> Frame:
        return room_frame(((80, 48),), player=self.player)


class ChestNavigationEnv:
    def __init__(self) -> None:
        self.player = (80, 112)

    def reset(self) -> Frame:
        self.player = (80, 112)
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        del frames
        if action == Action.LEFT:
            self.player = (64, 112)
        elif action == Action.RIGHT:
            self.player = (96, 112)
        return self._frame()

    def save_state(self):
        return self.player

    def load_state(self, state) -> Frame:
        self.player = state
        return self._frame()

    def _frame(self) -> Frame:
        return room_frame(
            player=self.player,
            open_chest=(32, 112),
            life_glyph=LIFE_FIVE,
        )


class PixelHeartGoalPriorTests(unittest.TestCase):
    def test_discovers_and_rewards_a_heart_disappearance(self) -> None:
        source = room_frame(((48, 48),))
        target = room_frame()
        prior = PixelHeartGoalPrior(heart_reward=25.0, all_hearts_reward=75.0)
        self.assertEqual(prior.observe_room(source), ((48, 48),))

        analysis = prior.analyze(source, target)

        self.assertTrue(analysis.reliable)
        self.assertEqual(analysis.collected, ((48, 48),))
        self.assertEqual(analysis.heart_reward, 25.0)
        self.assertEqual(analysis.all_hearts_reward, 75.0)
        self.assertEqual(analysis.total_reward, 100.0)

    def test_ignores_a_dark_transition(self) -> None:
        source = room_frame(((48, 48), (64, 48)))
        target = Frame(256, 240, 3, bytes(256 * 240 * 3))
        prior = PixelHeartGoalPrior()
        prior.observe_room(source)

        analysis = prior.analyze(source, target)

        self.assertFalse(analysis.reliable)
        self.assertEqual(analysis.collected, ())
        self.assertEqual(analysis.total_reward, 0.0)

    def test_non_heart_changes_do_not_create_progress(self) -> None:
        source = room_frame(((48, 48), (64, 48)))
        changed = bytearray(source.pixels)
        changed[(100 * source.width + 100) * 3] = 255
        target = Frame(source.width, source.height, source.channels, bytes(changed))
        prior = PixelHeartGoalPrior()
        prior.observe_room(source)

        analysis = prior.analyze(source, target)

        self.assertEqual(analysis.collected, ())
        self.assertEqual(analysis.remaining_hearts, 2)

    def test_navigation_reward_is_symmetric_pixel_progress(self) -> None:
        source = room_frame(((48, 48),), player=(80, 192))
        closer = room_frame(((48, 48),), player=(80, 176))
        farther = room_frame(((48, 48),), player=(96, 192))
        prior = PixelHeartGoalPrior(navigation_reward=2.0)
        prior.observe_room(source)

        closer_analysis = prior.analyze(source, closer)
        farther_analysis = prior.analyze(source, farther)

        self.assertEqual(closer_analysis.source_player_slot, (80, 192))
        self.assertEqual(closer_analysis.target_player_slot, (80, 176))
        self.assertEqual(closer_analysis.navigation_reward, 2.0)
        self.assertEqual(farther_analysis.navigation_reward, -2.0)
        self.assertEqual(closer_analysis.milestone_reward, 0.0)
        self.assertEqual(prior.distance_to_hearts(source), 11.0)
        self.assertEqual(prior.distance_to_hearts(closer), 10.0)

    def test_player_detector_rejects_a_magenta_chest_water_overlap(self) -> None:
        frame = room_frame(player=(48, 48))
        pixels = bytearray(frame.pixels)
        false_candidate = (
            [(21, 95, 217)] * 100
            + [(0, 0, 0)] * 100
            + [(255, 110, 204)] * 31
            + [(255, 255, 255)] * 25
        )
        for index, value in enumerate(false_candidate):
            row, column = divmod(index, 16)
            offset = ((96 + row) * frame.width + 32 + column) * 3
            pixels[offset : offset + 3] = bytes(value)
        overlapped = Frame(
            frame.width, frame.height, frame.channels, bytes(pixels)
        )

        prior = PixelHeartGoalPrior()

        self.assertEqual(prior.detect_player(overlapped), (48, 48))

    def test_temporal_player_tracking_rejects_a_distant_blue_teleport(self) -> None:
        source = room_frame(((80, 48),), player=(48, 48))
        target = room_frame(((80, 48),))
        pixels = bytearray(target.pixels)
        distant_candidate = (
            [(21, 95, 217)] * 100
            + [(0, 0, 0)] * 100
            + [(255, 255, 255)] * 56
        )
        for index, value in enumerate(distant_candidate):
            row, column = divmod(index, 16)
            offset = ((112 + row) * target.width + 160 + column) * 3
            pixels[offset : offset + 3] = bytes(value)
        target = Frame(
            target.width, target.height, target.channels, bytes(pixels)
        )
        prior = PixelHeartGoalPrior()
        prior.observe_room(source)

        self.assertEqual(prior.detect_player(target), (160, 112))

        analysis = prior.analyze(source, target)

        self.assertEqual(analysis.source_player_slot, (48, 48))
        self.assertIsNone(analysis.target_player_slot)
        self.assertEqual(analysis.navigation_reward, 0.0)

    def test_temporal_player_tracking_accepts_a_strong_adjacent_move(self) -> None:
        moved = room_frame(player=(80, 64))
        prior = PixelHeartGoalPrior()

        self.assertEqual(
            prior.detect_player(moved, reference=(64, 64)),
            (80, 64),
        )

    def test_open_chest_becomes_the_goal_after_the_last_heart(self) -> None:
        initial = room_frame(((48, 48),), life_glyph=LIFE_FIVE)
        opened = room_frame(
            player=(80, 112),
            open_chest=(32, 112),
            life_glyph=LIFE_FIVE,
        )
        closer = room_frame(
            player=(64, 112),
            open_chest=(32, 112),
            life_glyph=LIFE_FIVE,
        )
        prior = PixelHeartGoalPrior(navigation_reward=1.0)
        prior.observe_room(initial)
        collected = prior.analyze(initial, opened)
        prior.commit(collected, opened)

        analysis = prior.analyze(opened, closer)

        self.assertEqual(analysis.goal_phase, "open_chest")
        self.assertEqual(analysis.target_chest_slot, (32, 112))
        self.assertEqual(analysis.source_chest_distance, 3.0)
        self.assertEqual(analysis.target_chest_distance, 2.0)
        self.assertEqual(analysis.navigation_reward, 1.0)

    def test_open_chest_navigation_works_when_resuming_after_all_hearts(self) -> None:
        source = room_frame(
            player=(80, 112),
            open_chest=(32, 112),
            life_glyph=LIFE_FIVE,
        )
        target = room_frame(
            player=(64, 112),
            open_chest=(32, 112),
            life_glyph=LIFE_FIVE,
        )
        prior = PixelHeartGoalPrior(navigation_reward=1.0)
        prior.observe_room(source)

        analysis = prior.analyze(source, target)

        self.assertFalse(prior.initialized)
        self.assertEqual(analysis.target_chest_slot, (32, 112))
        self.assertEqual(analysis.navigation_reward, 1.0)

    def test_detects_the_empty_animation_frame_of_the_open_chest(self) -> None:
        pixels = bytearray((86, 29, 0) * (256 * 240))
        for row in range(16):
            for column in range(16):
                source = OPEN_CHEST_EMPTY_PROTOTYPE[row * 16 + column]
                offset = ((112 + row) * 256 + 32 + column) * 3
                pixels[offset : offset + 3] = bytes(source)
        frame = Frame(256, 240, 3, bytes(pixels))

        prior = PixelHeartGoalPrior()

        self.assertEqual(prior.detect_open_chest(frame), (32, 112))

    def test_chest_contact_before_a_transition_is_a_milestone(self) -> None:
        source = room_frame(
            player=(48, 112),
            open_chest=(32, 112),
            life_glyph=LIFE_FIVE,
        )
        target = Frame(256, 240, 3, bytes(256 * 240 * 3))
        prior = PixelHeartGoalPrior(chest_reward=100.0)
        prior.observe_room(source)

        analysis = prior.analyze(source, target)

        self.assertTrue(analysis.chest_completed)
        self.assertEqual(analysis.goal_phase, "chest_completed")
        self.assertEqual(analysis.chest_reward, 100.0)
        self.assertEqual(analysis.total_reward, 100.0)

    def test_life_change_requires_a_dark_transition_and_penalizes_loss(self) -> None:
        source = room_frame(((48, 48),), life_glyph=LIFE_FIVE)
        dark = Frame(256, 240, 3, bytes(256 * 240 * 3))
        reset = room_frame(((48, 48),), life_glyph=LIFE_FOUR)
        prior = PixelHeartGoalPrior(life_loss_penalty=100.0)
        prior.observe_room(source)

        unconfirmed = prior.analyze(source, reset)
        self.assertTrue(unconfirmed.life_counter_changed)
        self.assertFalse(unconfirmed.life_loss_confirmed)
        self.assertEqual(unconfirmed.life_loss_penalty, 0.0)

        transition = prior.analyze(source, dark)
        prior.commit(transition, dark)
        confirmed = prior.analyze(dark, reset)

        self.assertTrue(confirmed.life_loss_confirmed)
        self.assertEqual(confirmed.life_loss_penalty, -100.0)
        self.assertEqual(confirmed.total_reward, -100.0)

    def test_verified_planner_uses_open_chest_navigation_after_resume(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            ChestNavigationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.LEFT, Action.RIGHT),
                planning_depth=1,
                beam_width=3,
                verify_actions=3,
                action_frames=1,
                actual_novelty_weight=0.0,
                scene_novelty_weight=0.0,
                prediction_error_weight=0.0,
                actual_change_weight=0.0,
                action_effect_weight=0.0,
                causal_spatial_novelty_weight=0.0,
                action_coverage_weight=0.0,
                duration_coverage_weight=0.0,
                consecutive_repeat_weight=0.0,
                delayed_return_weight=0.0,
                human_prior_chest_reward=100.0,
                human_prior_navigation_reward=2.0,
            ),
        )

        agent.reset()
        decision = agent.decide()

        self.assertEqual(decision.action, Action.LEFT)
        self.assertGreaterEqual(decision.score, 2.0)

    def test_confirmed_life_loss_marks_the_pre_transition_choice_hazardous(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            HeartRewardEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(human_prior_life_loss_penalty=100.0),
        )
        agent.reset()
        source = room_frame(((48, 48),), life_glyph=LIFE_FIVE)
        dark = Frame(256, 240, 3, bytes(256 * 240 * 3))
        reset = room_frame(((48, 48),), life_glyph=LIFE_FOUR)
        prior = PixelHeartGoalPrior(life_loss_penalty=100.0)
        prior.observe_room(source)
        transition = prior.analyze(source, dark)
        agent._record_human_prior_outcome(
            transition, "danger-context", Action.RIGHT, 16, source, dark
        )
        prior.commit(transition, dark)
        confirmed = prior.analyze(dark, reset)
        agent._record_human_prior_outcome(
            confirmed, "reset-context", Action.NOOP, 16, dark, reset
        )

        choice = ("danger-context", Action.RIGHT, 16)
        self.assertEqual(agent.temporal_option_values[choice], -100.0)
        self.assertEqual(agent.temporal_option_samples[choice], 1)
        self.assertIsNone(agent.pending_life_hazard_choice)

    def test_death_during_room_calibration_preserves_and_attributes_loss(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        source = room_frame(
            open_chest=(32, 112),
            life_glyph=LIFE_FIVE,
        )
        dark = Frame(256, 240, 3, bytes(256 * 240 * 3))
        reset = room_frame(((48, 48),), life_glyph=LIFE_FOUR)
        agent = VerifiedNeuralAgent(
            HeartRewardEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(human_prior_life_loss_penalty=100.0),
        )
        agent.reset(initial_frame=source)
        assert agent.goal_prior is not None
        choice = ("danger-context", Action.RIGHT, 16)
        checkpoint = _LifeHazardCheckpoint(
            state=agent.env.save_state(),
            frame=source,
            choice=choice,
            decision=7,
            frontier_signature="danger-context",
            causal_context_signature="causal-context-root",
            scene=agent._scene_signature(source),
            pose_action=None,
            last_action=Action.NOOP,
            last_duration=16,
            action_streak=1,
            goal_heart_slots=(),
            goal_player_slot=None,
        )
        agent.active_temporal_option = _TemporalOptionTrace(
            choice=choice,
            initiation_decision=7,
            start_decision=8,
            entry_signature="animation",
            entry_scene="room",
            initiation_frame_digest=source.digest,
            recovery_checkpoint=checkpoint,
            causal_evidence=True,
        )

        transition = agent.goal_prior.analyze(source, dark)
        agent._record_human_prior_outcome(
            transition, "animation-context", Action.NOOP, 16, source, dark
        )
        agent.goal_prior.commit(transition, dark)
        confirmed = agent.goal_prior.analyze(dark, reset)
        committed = agent._commit_goal_prior(confirmed, reset)

        self.assertTrue(committed.life_loss_confirmed)
        self.assertEqual(committed.life_loss_penalty, -100.0)
        self.assertEqual(agent.goal_prior.known_slots, {(48, 48)})

        agent._record_human_prior_outcome(
            committed, "reset-context", Action.NOOP, 16, dark, reset
        )

        self.assertEqual(agent.temporal_option_values[choice], -100.0)
        self.assertEqual(agent.temporal_option_samples[choice], 1)
        self.assertNotIn(
            ("animation-context", Action.NOOP, 16),
            agent.temporal_option_values,
        )
        self.assertIs(agent.pending_life_recovery, checkpoint)

        recovered = agent._restore_after_life_loss()

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertTrue(recovered.restored_archive)
        self.assertEqual(recovered.frame.digest, source.digest)
        self.assertEqual(agent.current_frontier_signature, "danger-context")
        self.assertEqual(agent.goal_prior.current_slots(), ())
        self.assertIsNone(agent.pending_life_recovery)

    def test_verified_planner_uses_navigation_without_clipping_intrinsic(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            HeartNavigationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.UP, Action.RIGHT),
                planning_depth=1,
                beam_width=3,
                verify_actions=3,
                action_frames=1,
                actual_novelty_weight=0.0,
                scene_novelty_weight=0.0,
                prediction_error_weight=0.0,
                actual_change_weight=0.0,
                action_effect_weight=0.0,
                causal_spatial_novelty_weight=0.0,
                action_coverage_weight=0.0,
                duration_coverage_weight=0.0,
                consecutive_repeat_weight=0.0,
                delayed_return_weight=0.0,
                human_prior_heart_reward=25.0,
                human_prior_navigation_reward=2.0,
            ),
        )

        agent.reset()
        decision = agent.decide()

        self.assertEqual(decision.action, Action.UP)
        self.assertGreaterEqual(decision.score, 2.0)

    def test_archive_recovery_softly_prefers_a_closer_goal_checkpoint(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = HeartNavigationEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.UP,),
                planning_depth=1,
                human_prior_heart_reward=25.0,
                human_prior_navigation_reward=1.0,
            ),
        )
        agent.reset()
        current = room_frame(((80, 48),), player=(80, 112))
        closer = room_frame(((80, 48),), player=(80, 80))
        equal = room_frame(((80, 48),), player=(112, 80))
        agent.frame = current
        plan = NeuralPlan((Action.UP,), (1,), 1.0, 0.0)
        agent.archive = [
            _ArchivedBranch(
                state=(80, 80),
                frame=closer,
                plan=plan,
                score=0.0,
                scene=agent._scene_signature(closer),
                created=2,
                causal_event_outcome=True,
                goal_heart_slots=((80, 48),),
                goal_remaining_hearts=1,
                goal_total_hearts=1,
            ),
            _ArchivedBranch(
                state=(112, 80),
                frame=equal,
                plan=plan,
                score=0.0,
                scene=agent._scene_signature(equal),
                created=0,
                causal_event_outcome=True,
                goal_heart_slots=((80, 48),),
                goal_remaining_hearts=1,
                goal_total_hearts=1,
            ),
        ]
        agent.delayed_return_recovery = True

        self.assertGreater(
            agent._archive_frontier_score(agent.archive[0]),
            agent._archive_frontier_score(agent.archive[1]),
        )

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, closer.digest)

    def test_navigation_progress_temporarily_suppresses_delayed_recovery(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        env = HeartNavigationEnv()
        agent = VerifiedNeuralAgent(
            env,
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.UP,),
                planning_depth=1,
                human_prior_heart_reward=25.0,
                human_prior_navigation_reward=1.0,
                human_prior_navigation_recovery_grace=2,
            ),
        )
        agent.reset()
        current = room_frame(((80, 48),), player=(80, 80))
        farther = room_frame(((80, 48),), player=(80, 192))
        agent.frame = current
        plan = NeuralPlan((Action.UP,), (1,), 1.0, 0.0)
        agent.archive = [
            _ArchivedBranch(
                state=(80, 192),
                frame=farther,
                plan=plan,
                score=0.0,
                scene=agent._scene_signature(farther),
                created=0,
                causal_event_outcome=True,
                goal_heart_slots=((80, 48),),
                goal_remaining_hearts=1,
                goal_total_hearts=1,
            )
        ]
        agent.last_navigation_change_decision = 0
        agent.delayed_return_recovery = True

        suppressed = agent._restore_if_stagnant()

        self.assertIsNone(suppressed)
        self.assertEqual(agent.frame.digest, current.digest)
        self.assertFalse(agent.delayed_return_recovery)

        agent.decision_index = 2
        agent.delayed_return_recovery = True
        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(restored.frame.digest, farther.digest)

    def test_verified_planner_prioritizes_a_real_heart_event(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            HeartRewardEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                actions=(Action.NOOP, Action.RIGHT),
                planning_depth=1,
                beam_width=2,
                verify_actions=2,
                action_frames=1,
                actual_novelty_weight=0.0,
                scene_novelty_weight=0.0,
                prediction_error_weight=0.0,
                actual_change_weight=0.0,
                action_effect_weight=0.0,
                causal_spatial_novelty_weight=0.0,
                action_coverage_weight=0.0,
                duration_coverage_weight=0.0,
                consecutive_repeat_weight=0.0,
                delayed_return_weight=0.0,
                human_prior_heart_reward=25.0,
                human_prior_all_hearts_reward=75.0,
            ),
        )

        agent.reset()
        decision = agent.decide()

        self.assertEqual(decision.action, Action.RIGHT)
        self.assertGreaterEqual(decision.score, 90.0)

    def test_best_progress_survives_restoring_an_ancestor(self) -> None:
        source = room_frame(((48, 48), (64, 48)))
        one_collected = room_frame(((64, 48),))
        prior = PixelHeartGoalPrior()
        prior.observe_room(source)
        analysis = prior.analyze(source, one_collected)
        prior.commit(analysis, one_collected)
        self.assertEqual(prior.best_remaining_hearts, 1)

        prior.restore(((48, 48), (64, 48)), source)

        self.assertEqual(len(prior.current_slots()), 2)
        self.assertEqual(prior.best_remaining_hearts, 1)


if __name__ == "__main__":
    unittest.main()
