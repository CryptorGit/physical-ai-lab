import csv
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestV59MotorTargetParity(unittest.TestCase):
    def test_all_teacher_forced_motor_targets_pass(self):
        path = (
            ROOT
            / "artifacts/v59_evaluation_equivalence/comparison_tables/"
            "motor_target_parity.csv"
        )
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 500)
        self.assertLessEqual(
            max(float(row["max_abs_error"]) for row in rows), 1e-6
        )
