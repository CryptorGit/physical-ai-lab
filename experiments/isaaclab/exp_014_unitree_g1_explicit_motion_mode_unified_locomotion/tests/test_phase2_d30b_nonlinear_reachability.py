from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
SCRIPT = HERE.parents[1] / "scripts/run_phase2_d30b_nonlinear_post_touchdown_reachability.py"
SPEC = importlib.util.spec_from_file_location("phase2_d30b", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class D30BMathTests(unittest.TestCase):
    def test_minimum_jerk_boundary_and_midpoint(self):
        self.assertEqual(MODULE.minimum_jerk(0.0), 0.0)
        self.assertEqual(MODULE.minimum_jerk(1.0), 1.0)
        self.assertAlmostEqual(MODULE.minimum_jerk(0.5), 0.5)

    def test_phase_delta_respects_fixed_bound_and_zero_before_td0(self):
        basis = type("B", (), {"components": np.eye(37)})()
        theta = np.ones(16)
        bound = np.full(4, 0.1)
        np.testing.assert_allclose(MODULE.phase_delta(theta, 99, [100, 120, 140, 160], basis, bound), 0.0)
        delta = MODULE.phase_delta(theta, 110, [100, 120, 140, 160], basis, bound)
        self.assertTrue(np.isfinite(delta).all())
        self.assertLessEqual(np.max(np.abs(delta)), 0.1)

    def test_strict_event_requires_two_off_three_on(self):
        events = []
        seen = set()
        history = [
            np.array([[False, False]]),
            np.array([[False, False]]),
            np.array([[True, False]]),
            np.array([[True, False]]),
            np.array([[True, False]]),
        ]
        MODULE.strict_events(history, 10, 0, events, seen)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["side"], "LEFT")


if __name__ == "__main__":
    unittest.main()
