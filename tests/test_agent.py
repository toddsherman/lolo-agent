import tempfile
import unittest
from pathlib import Path

from lolo_agent.agent import AgentConfig, BranchingAgent
from lolo_agent.checkpoint import load_model, save_model
from lolo_agent.environment import Action
from lolo_agent.mock_puzzle import MockPuzzleEnv
from lolo_agent.world_model import EmpiricalWorldModel, FrozenModelError


class AgentTests(unittest.TestCase):
    def test_save_state_branching_commits_exactly_one_action(self) -> None:
        env = MockPuzzleEnv()
        agent = BranchingAgent(
            env,
            config=AgentConfig(actions=(Action.LEFT, Action.RIGHT), planning_depth=2, beam_width=4),
        )
        agent.reset()
        decision = agent.decide()
        self.assertEqual(env.steps, 1)
        self.assertEqual(len(decision.planned_path), 2)
        self.assertEqual(decision.branches_examined, 6)

    def test_frozen_evaluation_does_not_change_checkpoint(self) -> None:
        env = MockPuzzleEnv()
        model = EmpiricalWorldModel()
        frame = env.reset()
        target = env.step(Action.RIGHT)
        model.observe(frame, Action.RIGHT, target)
        agent = BranchingAgent(
            env,
            model=model,
            config=AgentConfig(planning_depth=2, beam_width=5),
            training=False,
        )
        agent.reset()
        before = model.checkpoint_digest
        agent.run(5)
        after = model.checkpoint_digest
        self.assertEqual(before, after)

    def test_direct_update_fails_closed_when_frozen(self) -> None:
        env = MockPuzzleEnv()
        model = EmpiricalWorldModel()
        source = env.reset()
        target = env.step(Action.RIGHT)
        model.freeze()
        with self.assertRaises(FrozenModelError):
            model.observe(source, Action.RIGHT, target)

    def test_checkpoint_round_trip(self) -> None:
        env = MockPuzzleEnv()
        model = EmpiricalWorldModel()
        source = env.reset()
        target = env.step(Action.RIGHT)
        model.observe(source, Action.RIGHT, target)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            digest = save_model(model, path)
            restored = load_model(path)
        self.assertEqual(digest, restored.checkpoint_digest)

    def test_agent_runs_only_on_pixel_environment_contract(self) -> None:
        env = MockPuzzleEnv()
        agent = BranchingAgent(env, config=AgentConfig(planning_depth=1, beam_width=7))
        initial = agent.reset()
        decisions = agent.run(3)
        self.assertEqual(len(initial.pixels), initial.width * initial.height)
        self.assertEqual(len(decisions), 3)
        self.assertTrue(all(decision.frame.pixels for decision in decisions))

    def test_hidden_two_push_puzzle_is_solved_from_pixels(self) -> None:
        env = MockPuzzleEnv()
        agent = BranchingAgent(
            env,
            config=AgentConfig(
                actions=(Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT),
                planning_depth=2,
                beam_width=16,
            ),
        )
        agent.reset()
        agent.run(2)
        self.assertTrue(env.evaluator_solved())
        solved_frame = agent.frame
        agent.run(3)
        self.assertTrue(env.evaluator_solved())
        self.assertEqual(solved_frame, agent.frame)


if __name__ == "__main__":
    unittest.main()
