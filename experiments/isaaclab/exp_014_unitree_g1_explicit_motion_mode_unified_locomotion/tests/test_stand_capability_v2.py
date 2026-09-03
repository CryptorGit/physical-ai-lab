from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g1_explicit_motion_mode.stand_capability_v2 import (  # noqa: E402
    Exp014ResetToStandEvaluatorV2,
    Exp014StandHoldEvaluatorV2,
    legacy_whole_window_2s_average,
)


class StandCapabilityV2Tests(unittest.TestCase):
    def setUp(self):
        self.reset = Exp014ResetToStandEvaluatorV2()
        self.hold = Exp014StandHoldEvaluatorV2()

    def test_immediate_stable_both_pass(self):
        speed = np.full(200, 0.01)
        yaw = np.full(200, 0.01)
        reset = self.reset.evaluate(speed, yaw)
        self.assertTrue(reset.passed)
        self.assertTrue(self.hold.evaluate(speed, yaw, reset).passed)

    def test_settle_at_point_eight_seconds_passes(self):
        speed = np.r_[np.full(39, 0.2), np.full(161, 0.01)]
        yaw = speed.copy()
        result = self.reset.evaluate(speed, yaw)
        self.assertTrue(result.passed)
        self.assertEqual(result.acquisition_time_s, 0.8)

    def test_settle_at_one_point_one_seconds_fails(self):
        speed = np.r_[np.full(54, 0.2), np.full(146, 0.01)]
        result = self.reset.evaluate(speed, speed)
        self.assertFalse(result.passed)

    def test_reexit_after_half_second_fails(self):
        speed = np.r_[np.full(9, 0.2), np.full(25, 0.01), np.full(166, 0.2)]
        result = self.reset.evaluate(speed, speed)
        self.assertFalse(result.passed)
        self.assertGreaterEqual(result.re_exit_count, 1)

    def test_reset_pass_hold_fails(self):
        speed = np.r_[np.full(50, 0.01), np.full(150, 0.2)]
        reset = self.reset.evaluate(speed, speed)
        self.assertTrue(reset.passed)
        self.assertFalse(self.hold.evaluate(speed, speed, reset).passed)

    def test_legacy_only_fails(self):
        speed = np.r_[np.full(39, 0.25), np.full(161, 0.01)]
        yaw = speed.copy()
        reset = self.reset.evaluate(speed, yaw)
        hold = self.hold.evaluate(speed, yaw, reset)
        legacy = legacy_whole_window_2s_average(speed, yaw)
        self.assertTrue(reset.passed)
        self.assertTrue(hold.passed)
        self.assertFalse(legacy["passed"])


if __name__ == "__main__":
    unittest.main()
