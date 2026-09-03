import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from v59_stochastic_common import inject_observation_noise, normalized_observation


class TestV59ObservationNoiseParity(unittest.TestCase):
    def test_noise_is_applied_before_normalization(self):
        raw = np.linspace(-1, 1, 101)
        noise = np.linspace(0.01, 0.02, 101)
        mean = np.linspace(-0.2, 0.2, 101)
        std = np.linspace(0.5, 1.5, 101)
        injected = inject_observation_noise(raw, noise)
        actual = normalized_observation(injected, mean, std)
        np.testing.assert_allclose(actual, (raw + noise - mean) / std)

    def test_saved_samples_reconstruct_noisy_observation(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "artifacts/v59_stochastic_evaluation_equivalence/stochastic_traces"
        )
        traces = sorted(root.glob("*.npz"))
        self.assertEqual(len(traces), 15)
        for trace in traces:
            with np.load(trace) as data:
                np.testing.assert_allclose(
                    data["raw_observation_before_noise"]
                    + data["noise_sample"],
                    data["raw_observation_after_noise"],
                    rtol=0,
                    atol=0,
                )
