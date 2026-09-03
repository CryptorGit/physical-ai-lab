import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from v59_parity_common import COMMANDS, compose_motor_target, error_metrics


class TestV59ParityCommon(unittest.TestCase):
    def test_required_commands_are_exact(self):
        self.assertEqual(
            [(c["vx"], c["vy"], c["yaw_rate"]) for c in COMMANDS],
            [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0),
             (-0.1, 0.0, 0.0), (0.0, 0.0, 0.6),
             (-0.07, 0.0, -0.3)],
        )

    def test_reverse_teacher_route_is_strictly_below_minus_point_zero_two(self):
        common = dict(
            default=np.zeros(14), lower=np.full(14, -2.0),
            upper=np.full(14, 2.0), action_scale=0.25,
            previous_target=np.zeros(14), max_motor_velocity=np.full(14, 100.0),
            dt=0.02, backward_reference=np.arange(34) / 100,
            backward_actuator_indices=np.r_[0:5, 9:14],
            backward_joint_indices=np.r_[0:5, 11:16],
            backward_residual_scale=0.12, coupled_slope=0.0,
            coupled_intercept=2.0,
        )
        inactive = compose_motor_target(
            np.zeros(14), command=np.array([-0.02, 0, 0]), **common
        )
        active = compose_motor_target(
            np.zeros(14), command=np.array([-0.021, 0, 0]), **common
        )
        self.assertFalse(inactive.teacher_active)
        self.assertTrue(active.teacher_active)
        self.assertGreater(np.linalg.norm(active.motor_target), 0)

    def test_error_metrics_identity(self):
        metrics = error_metrics(np.ones(14), np.ones(14))
        self.assertEqual(metrics["max_abs_error"], 0.0)
        self.assertEqual(metrics["cosine_similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
