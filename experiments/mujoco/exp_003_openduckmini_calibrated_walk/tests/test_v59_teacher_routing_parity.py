from pathlib import Path
import unittest
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class TestV59TeacherRoutingParity(unittest.TestCase):
    def test_reverse_only_routes_nonzero_teacher(self):
        traces = ROOT / "artifacts/v59_evaluation_equivalence/golden_traces"
        for name in ("C2_backward", "C4_backward_right_max"):
            with np.load(traces / f"{name}.npz") as data:
                self.assertTrue((data["teacher_mode"] == 1).all())
                self.assertTrue((np.linalg.norm(data["teacher_action"], axis=1) > 0).all())
        for name in ("C0_stand", "C1_forward", "C3_yaw_left"):
            with np.load(traces / f"{name}.npz") as data:
                self.assertTrue((data["teacher_mode"] == 0).all())
