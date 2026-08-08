import unittest

from lolo_agent.pixels import Frame


class FrameTests(unittest.TestCase):
    def test_rejects_invalid_pixel_count(self) -> None:
        with self.assertRaises(ValueError):
            Frame(2, 2, 1, b"123")

    def test_signature_is_spatial_and_deterministic(self) -> None:
        first = Frame(2, 2, 1, bytes([0, 0, 255, 255]))
        second = Frame(2, 2, 1, bytes([255, 255, 0, 0]))
        self.assertEqual(first.coarse_signature(), first.coarse_signature())
        self.assertNotEqual(first.coarse_signature(), second.coarse_signature())

    def test_normalized_frame_difference(self) -> None:
        dark = Frame(2, 1, 1, bytes([0, 0]))
        light = Frame(2, 1, 1, bytes([255, 255]))
        self.assertEqual(dark.mean_absolute_difference(light), 1.0)


if __name__ == "__main__":
    unittest.main()
