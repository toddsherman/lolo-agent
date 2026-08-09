import unittest

from lolo_agent.ensemble_world_model import EnsembleVisualDynamicsModel
from lolo_agent.environment import Action
from lolo_agent.goal_prior import HEART_PROTOTYPE, PixelHeartGoalPrior
from lolo_agent.neural_planner import NeuralPlanningConfig, VerifiedNeuralAgent
from lolo_agent.pixels import Frame


def room_frame(hearts=(), fill=(86, 29, 0)) -> Frame:
    width, height = 256, 240
    pixels = bytearray(fill * (width * height))
    for x, y in hearts:
        for row in range(16):
            for column in range(16):
                source = HEART_PROTOTYPE[row * 16 + column]
                offset = ((y + row) * width + x + column) * 3
                pixels[offset : offset + 3] = bytes(source)
    return Frame(width, height, 3, bytes(pixels))


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
