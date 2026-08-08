import os
import unittest
from pathlib import Path

from lolo_agent.environment import Action
from lolo_agent.libretro import LibretroEnv


class LibretroIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("LOLO_ROM_PATH") and os.environ.get("LOLO_CORE_PATH"),
        "set LOLO_ROM_PATH and LOLO_CORE_PATH for private integration test",
    )
    def test_pixels_and_save_states_are_deterministic(self) -> None:
        rom = Path(os.environ["LOLO_ROM_PATH"])
        core = Path(os.environ["LOLO_CORE_PATH"])
        with LibretroEnv(core, rom) as env:
            env.reset()
            for _ in range(120):
                frame = env.step(Action.NOOP)
            self.assertEqual((frame.width, frame.height, frame.channels), (256, 240, 3))
            self.assertGreater(len(set(frame.pixels)), 8)
            state = env.save_state()
            first_frame = env.step(Action.START, 2)
            first_state = env.save_state()
            env.load_state(state)
            second_frame = env.step(Action.START, 2)
            second_state = env.save_state()
            self.assertEqual(first_frame, second_frame)
            env.load_state(first_state)
            for _ in range(10):
                first_continuation = env.step(Action.NOOP)
            env.load_state(second_state)
            for _ in range(10):
                second_continuation = env.step(Action.NOOP)
            self.assertEqual(first_continuation, second_continuation)

            # Both continuation branches end at boot frame 132. Advance
            # to the skippable intro, then prove controller input changes the
            # pixel trajectory relative to an equally long NOOP branch.
            for _ in range(168):
                env.step(Action.NOOP)
            branch_root = env.save_state()
            env.step(Action.START, 2)
            for _ in range(60):
                start_frame = env.step(Action.NOOP)
            env.load_state(branch_root)
            for _ in range(62):
                noop_frame = env.step(Action.NOOP)
            self.assertNotEqual(start_frame, noop_frame)
