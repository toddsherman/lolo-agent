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
    def test_player_grid_snap_rounds_half_tile_forward(self) -> None:
        self.assertEqual(
            PixelHeartGoalPrior._snap_to_tile((104, 40)),
            (112, 48),
        )

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

    def test_temporal_player_tracking_recovers_from_overlapping_clipped_window(
        self,
    ) -> None:
        moved = room_frame(player=(72, 40))
        prior = PixelHeartGoalPrior()

        self.assertEqual(prior.detect_player(moved), (80, 48))
        self.assertEqual(
            prior.detect_player(moved, reference=(48, 48)),
            (80, 48),
        )

    def test_branch_tracking_uses_parent_reference_beyond_root_radius(self) -> None:
        source = room_frame(((160, 64),), player=(48, 64))
        target = room_frame(((160, 64),), player=(96, 64))
        prior = PixelHeartGoalPrior()
        prior.observe_room(source)

        root_anchored = prior.analyze(source, target)
        branch_anchored = prior.analyze(
            source,
            target,
            target_player_reference=(80, 64),
        )

        self.assertIsNone(root_anchored.target_player_slot)
        self.assertEqual(branch_anchored.source_player_slot, (48, 64))
        self.assertEqual(branch_anchored.target_player_slot, (96, 64))

    def test_player_pixel_mask_covers_palette_and_outline_halo(self) -> None:
        frame = room_frame(player=(80, 64))
        prior = PixelHeartGoalPrior()

        masked = prior.player_pixel_mask(frame, (80, 64))

        self.assertIn((80, 64), masked)
        self.assertIn((79, 64), masked)
        self.assertIn((80, 70), masked)
        self.assertNotIn((40, 40), masked)

    def test_player_pixel_mask_excludes_disconnected_white_object(self) -> None:
        source = room_frame(player=(80, 64))
        pixels = bytearray(source.pixels)
        for y in range(64, 72):
            for x in range(100, 108):
                offset = (y * source.width + x) * source.channels
                pixels[offset : offset + 3] = bytes((255, 255, 255))
        frame = Frame(
            source.width,
            source.height,
            source.channels,
            bytes(pixels),
        )

        masked = PixelHeartGoalPrior().player_pixel_mask(
            frame, (80, 64)
        )

        self.assertIn((80, 64), masked)
        self.assertNotIn((103, 68), masked)

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

    def test_player_contact_collects_chest_and_persists_completed_phase(self) -> None:
        source = room_frame(
            player=(48, 112),
            open_chest=(32, 112),
            life_glyph=LIFE_FIVE,
        )
        contact = room_frame(
            player=(32, 112),
            life_glyph=LIFE_FIVE,
        )
        departed = room_frame(
            player=(32, 128),
            life_glyph=LIFE_FIVE,
        )
        prior = PixelHeartGoalPrior(chest_reward=100.0)
        prior.known_slots = {(176, 48)}
        prior.current_present = set()
        prior.initialized = True
        prior.current_player_slot = (48, 112)

        collected = prior.analyze(source, contact)

        self.assertTrue(collected.chest_completed)
        self.assertTrue(collected.chest_obtained)
        self.assertEqual(collected.chest_reward, 100.0)
        prior.commit(collected, contact)

        after = prior.analyze(contact, departed)

        self.assertFalse(after.chest_completed)
        self.assertTrue(after.chest_obtained)
        self.assertEqual(after.chest_reward, 0.0)
        self.assertEqual(after.goal_phase, "chest_completed")

    def test_novel_room_reset_discovers_new_hearts_and_clears_chest(self) -> None:
        prior = PixelHeartGoalPrior()
        prior.known_slots = {(176, 48)}
        prior.current_present = set()
        prior.initialized = True
        prior.chest_obtained = True
        next_room = room_frame(
            hearts=((96, 128), (144, 192)),
            player=(128, 160),
            life_glyph=LIFE_FIVE,
        )

        discovered = prior.reset_room(next_room)

        self.assertEqual(discovered, ((96, 128), (144, 192)))
        self.assertEqual(prior.current_slots(), discovered)
        self.assertFalse(prior.chest_obtained)
        self.assertEqual(prior.current_player_slot, (128, 160))

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

    def test_life_loss_prefers_the_goal_milestone_checkpoint(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        milestone_frame = room_frame(
            ((48, 48),), life_glyph=LIFE_FIVE
        )
        source = room_frame(
            open_chest=(32, 112), life_glyph=LIFE_FIVE
        )
        dark = Frame(256, 240, 3, bytes(256 * 240 * 3))
        reset = room_frame(
            ((48, 48), (176, 48)), life_glyph=LIFE_FOUR
        )
        agent = VerifiedNeuralAgent(
            HeartRewardEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(human_prior_life_loss_penalty=100.0),
        )
        agent.reset(initial_frame=source)
        assert agent.goal_prior is not None
        milestone_choice = ("milestone-context", Action.LEFT, 16)
        milestone_state = object()
        milestone = _LifeHazardCheckpoint(
            state=milestone_state,
            frame=milestone_frame,
            choice=milestone_choice,
            decision=5,
            frontier_signature="milestone-context",
            causal_context_signature="causal-context-root",
            scene=agent._scene_signature(milestone_frame),
            pose_action=Action.LEFT,
            last_action=Action.NOOP,
            last_duration=16,
            action_streak=1,
            goal_heart_slots=((48, 48),),
            goal_player_slot=None,
            kind="goal_milestone",
        )
        causal_choice = ("danger-context", Action.RIGHT, 16)
        causal_state = object()
        causal = _LifeHazardCheckpoint(
            state=causal_state,
            frame=source,
            choice=causal_choice,
            decision=7,
            frontier_signature="danger-context",
            causal_context_signature="causal-context-root",
            scene=agent._scene_signature(source),
            pose_action=Action.RIGHT,
            last_action=Action.NOOP,
            last_duration=16,
            action_streak=1,
            goal_heart_slots=(),
            goal_player_slot=None,
        )
        agent.pending_goal_milestone_checkpoint = milestone
        agent.active_temporal_option = _TemporalOptionTrace(
            choice=causal_choice,
            initiation_decision=7,
            start_decision=8,
            entry_signature="animation",
            entry_scene="room",
            recovery_checkpoint=causal,
            causal_evidence=True,
        )

        transition = agent.goal_prior.analyze(source, dark)
        agent._record_human_prior_outcome(
            transition, "animation-context", Action.NOOP, 16, source, dark
        )
        agent.goal_prior.commit(transition, dark)
        confirmed = agent.goal_prior.analyze(dark, reset)
        committed = agent._commit_goal_prior(confirmed, reset)
        agent._record_human_prior_outcome(
            committed, "reset-context", Action.NOOP, 16, dark, reset
        )

        self.assertIs(agent.pending_life_recovery, milestone)
        self.assertEqual(
            agent.temporal_option_values[milestone_choice], -100.0
        )
        self.assertEqual(agent.temporal_option_values[causal_choice], -100.0)
        self.assertIs(
            agent.active_temporal_option.recovery_checkpoint, causal
        )
        plan = NeuralPlan((Action.UP,), (16,), 0.0, 0.0)
        retained_state = object()
        duplicate_descendant_state = object()
        stale_descendant_state = object()
        released_states = {id(stale_descendant_state)}

        def release_once(state) -> None:
            key = id(state)
            if key in released_states:
                raise RuntimeError("already released")
            released_states.add(key)

        agent.env.release_state = release_once
        agent.archive = [
            _ArchivedBranch(
                retained_state,
                milestone_frame,
                plan,
                0.0,
                agent._scene_signature(milestone_frame),
                5,
            ),
            _ArchivedBranch(
                duplicate_descendant_state,
                source,
                plan,
                0.0,
                agent._scene_signature(source),
                6,
            ),
            _ArchivedBranch(
                duplicate_descendant_state,
                source,
                plan,
                0.0,
                agent._scene_signature(source),
                7,
            ),
            _ArchivedBranch(
                stale_descendant_state,
                source,
                plan,
                0.0,
                agent._scene_signature(source),
                8,
            ),
        ]

        recovered = agent._restore_after_life_loss()

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.frame.digest, milestone_frame.digest)
        self.assertEqual(
            agent.current_frontier_signature, "milestone-context"
        )
        self.assertEqual(agent.goal_prior.current_slots(), ((48, 48),))
        self.assertIsNone(agent.active_temporal_option)
        self.assertEqual([branch.created for branch in agent.archive], [5])

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

    def test_navigation_reward_is_novelty_gated_per_player_endpoint(self) -> None:
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
                human_prior_phase_position_novelty=True,
            ),
        )
        source = agent.reset()
        target = room_frame(((80, 48),), player=(80, 176))
        analysis = agent.goal_prior.analyze(source, target)
        target_signature = agent._human_prior_graph_signatures(analysis)[1]

        first_score, _ = agent._human_prior_score(
            3.0, analysis, target_signature
        )
        agent._record_human_prior_player_position(
            target_signature, (80, 176)
        )
        repeated_score, _ = agent._human_prior_score(
            3.0, analysis, target_signature
        )

        self.assertEqual(first_score, 4.0)
        self.assertEqual(repeated_score, 3.0)
        self.assertEqual(
            agent._human_prior_effective_navigation_reward(
                analysis, target_signature
            ),
            0.0,
        )
        self.assertEqual(
            agent._human_prior_unexpanded_control_actions(target_signature),
            (Action.UP,),
        )
        agent._record_human_prior_graph_edge(
            target_signature, Action.UP, 1
        )
        self.assertEqual(
            agent._human_prior_unexpanded_control_actions(target_signature),
            (),
        )

    def test_navigation_regression_penalty_survives_revisited_endpoint(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            HeartNavigationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                human_prior_heart_reward=25.0,
                human_prior_navigation_reward=1.0,
            ),
        )
        agent.reset()
        source = room_frame(((80, 48),), player=(80, 176))
        target = room_frame(((80, 48),), player=(80, 192))
        assert agent.goal_prior is not None
        agent.goal_prior.current_player_slot = (80, 176)
        analysis = agent.goal_prior.analyze(source, target)
        target_signature = agent._human_prior_graph_signatures(analysis)[1]
        agent._record_human_prior_player_position(
            target_signature, (80, 192)
        )

        score, _ = agent._human_prior_score(
            3.0, analysis, target_signature
        )

        self.assertEqual(analysis.navigation_reward, -1.0)
        self.assertEqual(
            agent._human_prior_effective_navigation_reward(
                analysis, target_signature
            ),
            -1.0,
        )
        self.assertEqual(score, 2.0)

    def test_exhausted_goal_ordering_retargets_navigation_to_alternate(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            HeartNavigationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                human_prior_heart_reward=25.0,
                human_prior_navigation_reward=1.0,
            ),
        )
        agent.reset()
        hearts = ((48, 128), (144, 128))
        source = room_frame(hearts, player=(80, 128))
        target = room_frame(hearts, player=(96, 128))
        assert agent.goal_prior is not None
        agent.goal_prior.known_slots = set(hearts)
        agent.goal_prior.current_present = set(hearts)
        agent.goal_prior.current_player_slot = (80, 128)
        analysis = agent.goal_prior.analyze(source, target)
        agent.human_prior_exhausted_milestone_transitions.add(
            (hearts, ((144, 128),), False)
        )

        fields = agent._human_prior_ordering_navigation_fields(analysis)
        score, _ = agent._human_prior_score(0.0, analysis)

        self.assertEqual(analysis.navigation_reward, -1.0)
        self.assertTrue(fields["human_prior_navigation_retargeted"])
        self.assertEqual(
            fields["human_prior_navigation_failed_targets"],
            ((48, 128),),
        )
        self.assertEqual(
            fields["human_prior_navigation_active_targets"],
            ((144, 128),),
        )
        self.assertEqual(
            fields["human_prior_navigation_ordering_reward"], 1.0
        )
        self.assertEqual(score, 1.0)

    def test_ordering_retarget_stops_after_alternate_is_collected(self) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            HeartNavigationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(human_prior_navigation_reward=1.0),
        )
        agent.reset()
        hearts = ((48, 128), (144, 128))
        source = room_frame(hearts, player=(128, 128))
        target = room_frame(((48, 128),), player=(144, 128))
        assert agent.goal_prior is not None
        agent.goal_prior.known_slots = set(hearts)
        agent.goal_prior.current_present = set(hearts)
        agent.goal_prior.current_player_slot = (128, 128)
        analysis = agent.goal_prior.analyze(source, target)
        agent.human_prior_exhausted_milestone_transitions.add(
            (hearts, ((144, 128),), False)
        )

        fields = agent._human_prior_ordering_navigation_fields(analysis)

        self.assertEqual(analysis.collected, ((144, 128),))
        self.assertFalse(fields["human_prior_navigation_retargeted"])
        self.assertEqual(
            fields["human_prior_navigation_ordering_reward"],
            analysis.navigation_reward,
        )

    def test_exhausted_alternate_trial_disproves_ordering_interpretation(
        self,
    ) -> None:
        model = EnsembleVisualDynamicsModel(
            latent_size=32, action_size=8, ensemble_size=2
        )
        agent = VerifiedNeuralAgent(
            HeartNavigationEnv(),
            model,
            "cpu",
            NeuralPlanningConfig(
                human_prior_heart_reward=25.0,
                human_prior_navigation_reward=1.0,
                human_prior_option_search_depth=2,
            ),
        )
        agent.reset()
        hearts = ((48, 128), (144, 128))
        source = room_frame(hearts, player=(80, 128))
        moved = room_frame(hearts, player=(96, 128))
        collected = room_frame(((144, 128),), player=(48, 128))
        assert agent.goal_prior is not None
        agent.goal_prior.known_slots = set(hearts)
        agent.goal_prior.current_present = set(hearts)
        agent.goal_prior.current_player_slot = (80, 128)
        moved_analysis = agent.goal_prior.analyze(source, moved)
        collected_analysis = agent.goal_prior.analyze(source, collected)
        transition = (hearts, ((144, 128),), False)
        agent.human_prior_exhausted_milestone_transitions.add(transition)
        ordering_key = agent._human_prior_ordering_hypothesis_key(
            hearts, False
        )
        assert ordering_key is not None
        agent.human_prior_ordering_progress_hypotheses.add(ordering_key)
        agent.human_prior_milestone_outcomes.add(
            agent._human_prior_milestone_outcome_key(collected_analysis)
        )

        disproved = (
            agent._maybe_disprove_human_prior_ordering_hypothesis(
                moved_analysis,
                "source",
                "test_frontier_exhausted",
                "test-search-budget",
            )
        )
        fields = agent._human_prior_ordering_navigation_fields(
            moved_analysis
        )

        self.assertTrue(disproved)
        self.assertIn(
            ordering_key,
            agent.human_prior_disproved_ordering_hypotheses,
        )
        self.assertFalse(fields["human_prior_navigation_retargeted"])
        self.assertTrue(fields["human_prior_navigation_reconsidered"])
        self.assertEqual(
            fields["human_prior_navigation_reconsidered_targets"],
            ((48, 128),),
        )
        self.assertEqual(
            fields["human_prior_navigation_ordering_reward"],
            moved_analysis.navigation_reward,
        )
        self.assertFalse(
            agent._human_prior_milestone_transition_exhausted(
                collected_analysis
            )
        )
        self.assertFalse(
            agent._human_prior_milestone_outcome_known(
                collected_analysis
            )
        )

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

    def test_navigation_archive_restore_gets_local_expansion_grace(self) -> None:
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
                human_prior_best_first_archive=True,
                human_prior_graph_stagnation_visits=1,
            ),
        )
        agent.reset()
        target = room_frame(((80, 48),), player=(80, 64))
        plan = NeuralPlan((Action.UP,), (1,), 1.0, 0.0)
        agent.archive = [
            _ArchivedBranch(
                state=(80, 64),
                frame=target,
                plan=plan,
                score=1.0,
                scene=agent._scene_signature(target),
                created=0,
                origin_signature="source-frontier",
                frontier_signature="target-frontier",
                goal_heart_slots=((80, 48),),
                goal_remaining_hearts=1,
                goal_total_hearts=1,
                goal_player_slot=(80, 64),
                goal_source_signature="source-goal",
                goal_target_signature="target-goal",
            )
        ]
        agent.human_prior_graph_recovery_pending = True

        restored = agent._restore_if_stagnant()

        self.assertIsNotNone(restored)
        self.assertEqual(agent.goal_prior.current_player_slot, (80, 64))
        self.assertEqual(agent.last_navigation_change_decision, 1)
        self.assertTrue(
            agent._human_prior_navigation_recovery_grace_active()
        )

        alternate = room_frame(((80, 48),), player=(96, 64))
        agent.archive = [
            _ArchivedBranch(
                state=(96, 64),
                frame=alternate,
                plan=plan,
                score=1.0,
                scene=agent._scene_signature(alternate),
                created=1,
                origin_signature="source-frontier",
                frontier_signature="alternate-frontier",
                goal_heart_slots=((80, 48),),
                goal_remaining_hearts=1,
                goal_total_hearts=1,
                goal_player_slot=(96, 64),
                goal_source_signature="source-goal",
                goal_target_signature="alternate-goal",
            )
        ]
        agent.human_prior_graph_recovery_pending = True

        suppressed = agent._restore_if_stagnant()

        self.assertIsNone(suppressed)
        self.assertEqual(agent.goal_prior.current_player_slot, (80, 64))
        self.assertFalse(agent.human_prior_graph_recovery_pending)

        agent.last_navigation_change_decision = None
        agent._record_human_prior_player_position(
            "alternate-goal", (96, 64)
        )
        agent._record_human_prior_graph_edge_verification(
            "alternate-goal", Action.UP, 1
        )
        agent.human_prior_graph_recovery_pending = True

        fully_expanded_restore = agent._restore_if_stagnant()

        self.assertIsNone(fully_expanded_restore)
        self.assertEqual(agent.goal_prior.current_player_slot, (80, 64))
        self.assertIsNone(agent.last_navigation_change_decision)
        self.assertEqual(agent.archive, [])

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
                human_prior_life_loss_penalty=100.0,
            ),
        )

        agent.reset()
        decision = agent.decide()

        self.assertEqual(decision.action, Action.RIGHT)
        self.assertGreaterEqual(decision.score, 90.0)
        self.assertIsNotNone(agent.pending_goal_milestone_checkpoint)
        assert agent.pending_goal_milestone_checkpoint is not None
        self.assertEqual(
            agent.pending_goal_milestone_checkpoint.goal_heart_slots,
            ((48, 48),),
        )
        self.assertEqual(
            agent.pending_goal_milestone_checkpoint.kind,
            "goal_milestone",
        )

    def test_learned_delayed_hazard_blocks_a_positive_milestone(self) -> None:
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
                human_prior_life_loss_penalty=100.0,
            ),
        )
        agent.reset()
        choice = (agent.current_frontier_signature, Action.RIGHT, 1)
        agent._record_temporal_option_sample(choice, -100.0)

        decision = agent.decide()

        self.assertEqual(decision.action, Action.NOOP)
        self.assertIsNone(agent.pending_goal_milestone_checkpoint)

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

    def test_restored_descendant_updates_best_progress(self) -> None:
        source = room_frame(((48, 48), (64, 48)))
        one_collected = room_frame(((64, 48),))
        prior = PixelHeartGoalPrior()
        prior.observe_room(source)
        self.assertEqual(prior.best_remaining_hearts, 2)

        prior.restore(((64, 48),), one_collected)

        self.assertEqual(prior.current_slots(), ((64, 48),))
        self.assertEqual(prior.best_remaining_hearts, 1)


if __name__ == "__main__":
    unittest.main()
