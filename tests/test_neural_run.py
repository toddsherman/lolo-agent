import unittest

from lolo_agent.neural_run import StableSceneChangeDetector
from lolo_agent.pixels import Frame


class StableSceneChangeDetectorTests(unittest.TestCase):
    def frame(self, value: int) -> Frame:
        return Frame(6, 6, 3, bytes([value]) * 108)

    def test_requires_a_distinct_scene_to_remain_stable(self) -> None:
        initial = self.frame(96)
        detector = StableSceneChangeDetector(
            initial,
            stable_observations=2,
            warmup_decisions=1,
            minimum_difference=0.1,
        )
        self.assertIsNone(detector.observe(1, initial))
        self.assertIsNone(detector.observe(2, self.frame(0)))
        self.assertIsNone(detector.observe(3, self.frame(255)))
        result = detector.observe(4, self.frame(255))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["decision"], 4)
        self.assertEqual(result["stable_observations"], 2)
        self.assertTrue(result["dark_transition_observed"])

    def test_does_not_treat_a_stable_in_room_change_as_a_transition(self) -> None:
        initial = self.frame(96)
        detector = StableSceneChangeDetector(
            initial,
            stable_observations=2,
            warmup_decisions=1,
            minimum_difference=0.1,
        )
        self.assertIsNone(detector.observe(1, initial))
        self.assertIsNone(detector.observe(2, self.frame(255)))
        self.assertIsNone(detector.observe(3, self.frame(255)))


if __name__ == "__main__":
    unittest.main()
