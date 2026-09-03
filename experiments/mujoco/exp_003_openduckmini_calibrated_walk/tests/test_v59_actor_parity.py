import csv
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestV59ActorParity(unittest.TestCase):
    def test_all_fixed_input_actor_rows_pass(self):
        path = (
            ROOT
            / "artifacts/v59_evaluation_equivalence/comparison_tables/actor_parity.csv"
        )
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 500)
        self.assertLessEqual(
            max(float(row["python_max_abs_error"]) for row in rows), 1e-6
        )
        self.assertLessEqual(
            max(float(row["onnx_max_abs_error"]) for row in rows), 1e-5
        )
