from __future__ import annotations

import argparse

from .agent import AgentConfig, BranchingAgent
from .mock_puzzle import MockPuzzleEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pixel-only branching baseline")
    parser.add_argument("--steps", type=int, default=40, help="committed controller decisions")
    parser.add_argument("--depth", type=int, default=2, help="planning horizon")
    parser.add_argument("--beam", type=int, default=12, help="beam width")
    parser.add_argument("--frozen", action="store_true", help="disable persistent learning")
    args = parser.parse_args()

    env = MockPuzzleEnv()
    config = AgentConfig(planning_depth=args.depth, beam_width=args.beam)
    agent = BranchingAgent(env, config=config, training=not args.frozen)
    agent.reset()
    for index in range(1, args.steps + 1):
        decision = agent.decide()
        path = ",".join(action.value for action in decision.planned_path)
        print(
            f"{index:03d} action={decision.action.value:<5} "
            f"score={decision.score:6.3f} plan={path:<12} "
            f"solved={env.evaluator_solved()}"
        )
        if env.evaluator_solved():
            break
    print(f"persistent_model_sha256={agent.model.checkpoint_digest}")


if __name__ == "__main__":
    main()
