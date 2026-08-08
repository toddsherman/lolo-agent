import os
import unittest
from pathlib import Path

from lolo_agent.agent import AgentConfig, BranchingAgent
from lolo_agent.environment import Action
from lolo_agent.native_env import NativeHostError, NativeLibretroEnv


@unittest.skipUnless(
    os.environ.get("LOLO_ROM_PATH")
    and os.environ.get("LOLO_CORE_PATH")
    and os.environ.get("LOLO_NATIVE_HOST_PATH"),
    "set private ROM, core, and native host paths for integration test",
)
class NativeEnvironmentTests(unittest.TestCase):
    def make_env(self) -> NativeLibretroEnv:
        return NativeLibretroEnv(
            Path(os.environ["LOLO_NATIVE_HOST_PATH"]),
            Path(os.environ["LOLO_CORE_PATH"]),
            Path(os.environ["LOLO_ROM_PATH"]),
        )

    def test_opaque_state_replay_and_release(self) -> None:
        with self.make_env() as env:
            env.reset()
            for _ in range(300):
                env.step(Action.NOOP)
            root = env.save_state()
            env.step(Action.START, 2)
            for _ in range(60):
                first = env.step(Action.NOOP)
            env.load_state(root)
            env.step(Action.START, 2)
            for _ in range(60):
                second = env.step(Action.NOOP)
            self.assertEqual(first, second)
            env.release_state(root)
            with self.assertRaises(NativeHostError):
                env.load_state(root)

    def test_planner_releases_native_branch_states(self) -> None:
        with self.make_env() as env:
            agent = BranchingAgent(
                env,
                config=AgentConfig(
                    actions=(Action.LEFT, Action.RIGHT, Action.NOOP),
                    planning_depth=2,
                    beam_width=4,
                ),
            )
            agent.reset()
            decision = agent.decide()
            self.assertEqual(decision.branches_examined, 12)
