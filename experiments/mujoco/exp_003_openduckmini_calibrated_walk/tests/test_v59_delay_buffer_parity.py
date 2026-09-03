import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from v59_stochastic_common import delay_buffer_step


class TestV59DelayBufferParity(unittest.TestCase):
    def test_explicit_action_series_for_delays_zero_one_two(self):
        actions = [np.full(14, value, dtype=np.float32) for value in range(5)]
        expected = {
            0: [0, 1, 2, 3, 4],
            1: [0, 0, 1, 2, 3],
            2: [0, 0, 0, 1, 2],
        }
        for delay in (0, 1, 2):
            history = np.zeros(42, dtype=np.float32)
            observed = []
            for action in actions:
                history, output = delay_buffer_step(history, action, delay)
                observed.append(int(output[0]))
            self.assertEqual(observed, expected[delay])
