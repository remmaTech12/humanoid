"""Unit tests for rod angle math, independent of Isaac Sim."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import rod_tilt_deg


class TestRodTilt(unittest.TestCase):
    def test_upright(self):
        self.assertEqual(rod_tilt_deg((0, 0, 1)), (0, 0, 0))

    def test_toward_world_x(self):
        rad = math.radians(5)
        tilt, x, y = rod_tilt_deg((math.sin(rad), 0, math.cos(rad)))
        self.assertAlmostEqual(tilt, 5)
        self.assertAlmostEqual(x, 5)
        self.assertAlmostEqual(y, 0)

    def test_toward_negative_y(self):
        rad = math.radians(8)
        tilt, x, y = rod_tilt_deg((0, -math.sin(rad), math.cos(rad)))
        self.assertAlmostEqual(tilt, 8)
        self.assertAlmostEqual(x, 0)
        self.assertAlmostEqual(y, -8)

    def test_invalid_zero(self):
        with self.assertRaises(ValueError):
            rod_tilt_deg((0, 0, 0))


if __name__ == "__main__":
    unittest.main()
