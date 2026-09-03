import importlib.util
from pathlib import Path
import unittest


MODULE = Path(__file__).parents[1] / "scripts" / "support_reward_v2r1.py"
SPEC = importlib.util.spec_from_file_location("support_reward_v2r1", MODULE)
reward = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reward)


class SupportRewardV2R1Tests(unittest.TestCase):
    def test_target_remains_at_peak_during_weight_decay(self):
        self.assertEqual(reward.corrected_target(0.60), 0.7)
        self.assertGreater(reward.corrected_weight_envelope(0.60), 0.0)

    def test_schedule_boundaries(self):
        self.assertEqual(reward.corrected_target(0.49), 0.7)
        self.assertEqual(reward.corrected_weight_envelope(0.49), 1.0)
        self.assertEqual(reward.corrected_target(0.50), 0.7)
        self.assertEqual(reward.corrected_weight_envelope(0.50), 1.0)
        self.assertEqual(reward.corrected_target(0.749), 0.7)
        self.assertGreater(reward.corrected_weight_envelope(0.749), 0.0)
        self.assertEqual(reward.corrected_weight_envelope(0.75), 0.0)

    def test_zero_support_is_masked_for_zero_and_peak_targets(self):
        for t_s in (0.0, 0.50):
            self.assertEqual(reward.corrected_load_reward(
                left_vertical_n=0,
                right_vertical_n=0,
                left_force_norm_n=0,
                right_force_norm_n=0,
                t_s=t_s,
                sigma_load=0.3,
            ), 0.0)

    def test_valid_symmetric_support_is_maximal_at_zero_target(self):
        actual = reward.corrected_load_reward(
            left_vertical_n=50,
            right_vertical_n=50,
            left_force_norm_n=50,
            right_force_norm_n=50,
            t_s=0.0,
            sigma_load=0.3,
        )
        self.assertLess(abs(actual - 1.0), 1.0e-12)

    def test_unsigned_target_is_mirror_invariant(self):
        left = reward.corrected_load_reward(
            left_vertical_n=85,
            right_vertical_n=15,
            left_force_norm_n=85,
            right_force_norm_n=15,
            t_s=0.50,
            sigma_load=0.3,
        )
        right = reward.corrected_load_reward(
            left_vertical_n=15,
            right_vertical_n=85,
            left_force_norm_n=15,
            right_force_norm_n=85,
            t_s=0.50,
            sigma_load=0.3,
        )
        self.assertLess(abs(left - right), 1.0e-12)
        self.assertGreater(left, 0.999999999)


if __name__ == "__main__":
    unittest.main()
