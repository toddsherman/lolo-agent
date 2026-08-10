import unittest

from lolo_agent.bidirectional_probe import BidirectionalProbeCollector
from lolo_agent.environment import Action
from lolo_agent.pixels import Frame


class AnimatedReversibleEnv:
    def __init__(self) -> None:
        self.position = 0
        self.tick = 0
        self.phase = "agent"
        self.released = 0

    def reset(self) -> Frame:
        self.position = 0
        self.tick = 0
        return self._frame()

    def step(self, action: Action, frames: int = 1) -> Frame:
        self.tick += frames
        if action == Action.RIGHT:
            self.position = min(3, self.position + 1)
        elif action == Action.LEFT:
            self.position = max(0, self.position - 1)
        return self._frame()

    def save_state(self) -> tuple[int, int, object]:
        return self.position, self.tick, object()

    def load_state(self, state: tuple[int, int, object]) -> Frame:
        self.position, self.tick = state[:2]
        return self._frame()

    def release_state(self, _state: object) -> None:
        self.released += 1

    def _frame(self) -> Frame:
        pixels = bytearray(8)
        pixels[self.position] = 255
        pixels[4 + self.tick % 4] = 127
        return Frame(8, 1, 1, bytes(pixels))


class BidirectionalProbeTests(unittest.TestCase):
    def test_matched_noop_detects_return_despite_animation(self) -> None:
        env = AnimatedReversibleEnv()
        events = []
        source = env.reset()
        root = env.save_state()
        endpoint = env.step(Action.RIGHT, 1)
        endpoint_state = env.save_state()
        self.assertNotEqual(source.digest, env.step(Action.LEFT, 1).digest)
        env.load_state(endpoint_state)
        collector = BidirectionalProbeCollector(
            env,
            (Action.RIGHT, Action.LEFT, Action.NOOP),
            maximum_depth=1,
            beam_width=2,
            pixel_l1_threshold=0.0,
            emit=lambda event, **fields: events.append(
                {"event": event, **fields}
            ),
        )

        result = collector.collect(
            root_state=root,
            endpoint_state=endpoint_state,
            source_frame=source,
            endpoint_frame=endpoint,
            initial_action=Action.RIGHT,
            action_frames=1,
            decision=1,
            branch_id="decision-00000001-branch-01",
            candidate_rank=1,
        )

        self.assertTrue(result.return_observed)
        self.assertEqual(result.shortest_return_depth, 1)
        self.assertEqual(result.paths_evaluated, 3)
        returning = [
            event
            for event in events
            if event["event"] == "bidirectional_probe_step"
            and event["return_observed"]
        ]
        self.assertEqual([event["probe_action"] for event in returning], [Action.LEFT])
        self.assertEqual(returning[0]["matched_noop_l1"], 0.0)
        self.assertEqual(env.phase, "agent")
        self.assertEqual(env.load_state(endpoint_state), endpoint)
        self.assertEqual(env.released, 3)

    def test_no_return_is_scoped_to_the_configured_budget(self) -> None:
        env = AnimatedReversibleEnv()
        source = env.reset()
        root = env.save_state()
        endpoint = env.step(Action.RIGHT, 1)
        endpoint_state = env.save_state()
        events = []
        collector = BidirectionalProbeCollector(
            env,
            (Action.RIGHT, Action.NOOP),
            maximum_depth=2,
            beam_width=1,
            pixel_l1_threshold=0.0,
            emit=lambda event, **fields: events.append(
                {"event": event, **fields}
            ),
        )

        result = collector.collect(
            root_state=root,
            endpoint_state=endpoint_state,
            source_frame=source,
            endpoint_frame=endpoint,
            initial_action=Action.RIGHT,
            action_frames=1,
            decision=1,
            branch_id="decision-00000001-branch-01",
            candidate_rank=1,
        )

        self.assertFalse(result.return_observed)
        completed = next(
            event
            for event in events
            if event["event"] == "bidirectional_probe_completed"
        )
        self.assertTrue(completed["no_return_within_probe_budget"])
        self.assertEqual(result.paths_evaluated, 4)


if __name__ == "__main__":
    unittest.main()
