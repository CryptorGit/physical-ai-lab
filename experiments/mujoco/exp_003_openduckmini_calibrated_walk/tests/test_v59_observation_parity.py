from pathlib import Path
import unittest
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class TestV59ObservationParity(unittest.TestCase):
    def test_all_golden_observations_have_training_shape(self):
        traces = sorted(
            (ROOT / "artifacts/v59_evaluation_equivalence/golden_traces").glob(
                "*.npz"
            )
        )
        self.assertEqual(len(traces), 5)
        for trace in traces:
            with np.load(trace) as data:
                self.assertEqual(data["raw_observation"].shape, (100, 101))
                self.assertEqual(data["normalized_observation"].shape, (100, 101))
                self.assertTrue(np.isfinite(data["normalized_observation"]).all())
