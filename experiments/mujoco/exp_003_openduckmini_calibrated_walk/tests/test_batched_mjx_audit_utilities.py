from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_batched_mjx_reproducibility import (  # noqa: E402
    compare_data,
    sha_array,
)


class BatchedMjxAuditUtilityTest(unittest.TestCase):
    def test_sha_array_is_content_shape_and_dtype_sensitive(self) -> None:
        baseline = np.arange(8, dtype=np.float32).reshape(2, 4)
        self.assertEqual(sha_array(baseline), sha_array(baseline.copy()))
        self.assertNotEqual(sha_array(baseline), sha_array(baseline.astype(np.float64)))
        self.assertNotEqual(sha_array(baseline), sha_array(baseline.reshape(4, 2)))
        changed = baseline.copy()
        changed[1, 2] += 1.0
        self.assertNotEqual(sha_array(baseline), sha_array(changed))

    def test_compare_data_reports_first_environment_and_leaf_element(self) -> None:
        reference = {"state": {"crb": np.zeros((2, 3, 4), dtype=np.float32)}}
        comparison = {
            "state": {"crb": np.zeros((2, 3, 4), dtype=np.float32)}
        }
        comparison["state"]["crb"][1, 2, 3] = np.float32(2**-20)

        rows = compare_data(reference, comparison)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["field_path"], "state/crb")
        self.assertEqual(rows[0]["environment_index"], 1)
        self.assertEqual(rows[0]["element_index"], "[2, 3]")
        self.assertEqual(rows[0]["different_element_count"], 1)

    def test_compare_data_is_bitwise_not_tolerance_based(self) -> None:
        reference = {"value": np.array([[1.0]], dtype=np.float32)}
        comparison = {
            "value": np.nextafter(reference["value"], np.float32(2.0))
        }

        rows = compare_data(reference, comparison)

        self.assertTrue(rows)
        self.assertAlmostEqual(rows[0]["max_abs_error"], 2**-23)


if __name__ == "__main__":
    unittest.main()
