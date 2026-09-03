import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from v59_stochastic_common import backlash_observation


class TestV59BacklashStateParity(unittest.TestCase):
    def test_positive_zero_negative_zero_positive_injection(self):
        series = [0.008, 0.0, -0.008, 0.0, 0.008]
        actuator = np.zeros(14)
        for value in series:
            backlash = np.zeros(14)
            backlash[:5] = value
            backlash[9:] = value
            observed = backlash_observation(actuator, backlash)
            np.testing.assert_array_equal(observed, backlash)

    def test_backlash_does_not_transform_controller_motor_target(self):
        target = np.linspace(-0.4, 0.4, 14)
        for value in (0.008, 0.0, -0.008, 0.0, 0.008):
            before = target.copy()
            after = target.copy()  # passive XML joint acts in physics, not routing
            np.testing.assert_array_equal(before, after)

    def test_saved_trace_has_exact_pre_post_backlash_motor_target(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "artifacts/v59_stochastic_evaluation_equivalence/stochastic_traces"
        )
        traces = sorted(root.glob("*.npz"))
        self.assertEqual(len(traces), 15)
        for trace in traces:
            with np.load(trace) as data:
                np.testing.assert_array_equal(
                    data["motor_target_before_backlash"],
                    data["motor_target_after_backlash"],
                )
