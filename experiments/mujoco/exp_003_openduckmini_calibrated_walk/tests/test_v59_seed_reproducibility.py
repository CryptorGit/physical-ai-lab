import unittest
import numpy as np
import json
from pathlib import Path


class TestV59SeedReproducibility(unittest.TestCase):
    def test_numpy_backend_same_seed_is_exact(self):
        first = np.random.default_rng(59).uniform(size=(100, 14))
        second = np.random.default_rng(59).uniform(size=(100, 14))
        np.testing.assert_array_equal(first, second)

    def test_native_mjx_nonreproducibility_is_explicitly_recorded(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "artifacts/v59_stochastic_evaluation_equivalence/"
            "stochastic_parity_report.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(
            report["native_seed_reproducibility"][
                "closed_loop_mjx_bit_exact"
            ]
        )
