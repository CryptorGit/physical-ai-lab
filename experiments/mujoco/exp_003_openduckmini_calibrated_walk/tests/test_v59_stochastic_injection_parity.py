import csv
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestV59StochasticInjectionParity(unittest.TestCase):
    def test_all_fifteen_injection_cases_pass_controller_thresholds(self):
        path = (
            ROOT
            / "artifacts/v59_stochastic_evaluation_equivalence/"
            "sample_injection_results.csv"
        )
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(row["pass"] == "True" for row in rows))
        self.assertLessEqual(
            max(float(row["observation_max_abs_error"]) for row in rows), 1e-6
        )
        self.assertLessEqual(
            max(
                float(row["normalized_observation_max_abs_error"])
                for row in rows
            ),
            1e-6,
        )
        self.assertLessEqual(
            max(float(row["motor_target_max_abs_error"]) for row in rows), 1e-6
        )
