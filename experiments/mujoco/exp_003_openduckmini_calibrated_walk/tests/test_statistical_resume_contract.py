from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import jax.numpy as jnp


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from training.device_metrics import aggregate_rollout  # noqa: E402


class StatisticalResumeContractTest(unittest.TestCase):
    def test_protocol_changes_no_training_contract(self) -> None:
        contract = json.loads(
            (
                EXPERIMENT
                / "artifacts/statistical_resume_and_null_continuation/"
                "protocol_contract.json"
            ).read_text()
        )
        self.assertEqual(contract["objective"], "old_unbounded_dot")
        self.assertEqual(contract["statistical_resume"]["updates_per_trial"], 4)
        self.assertEqual(contract["statistical_resume"]["resume_boundary"], 2)
        self.assertEqual(
            contract["null_continuation"]["conditional_on"],
            "STATISTICAL_RESUME_PASS",
        )
        self.assertEqual(
            contract["null_continuation"]["interactions_per_run"], 250000
        )

    def test_joint_histogram_and_endpoint_counts_are_exact(self) -> None:
        commands = jnp.asarray(
            [
                [0.0, 0.0, 0.2, 0.0, 0.0, -0.2],
                [-0.1, 0.1, -0.3, 0.0, 0.0, 0.2],
            ]
        )
        sidecar = {
            "command": commands,
            "done": jnp.asarray([0, 1]),
            "fall": jnp.asarray([0, 1]),
            "truncation": jnp.asarray([0, 0]),
            "episode_start": jnp.asarray([1, 0]),
            "reward": jnp.asarray([2.0, 4.0]),
            "actual_velocity": commands[:, :3],
            "reward_terms": {},
        }
        result = aggregate_rollout(
            sidecar,
            official_commands=jnp.zeros((19, 3)),
            num_updates_per_batch=4,
            vx_edges=jnp.asarray([-0.12, -0.02, 0.02, 0.12]),
            vy_edges=jnp.asarray([-0.14, -0.02, 0.02, 0.14]),
            yaw_edges=jnp.asarray([-0.7, -0.1, 0.1, 0.7]),
            head_edges=jnp.asarray([-0.5, -0.1, 0.1, 0.5]),
        )
        self.assertEqual(int(result["rollout_sample_total"]), 2)
        self.assertEqual(
            int(jnp.sum(result["joint_command_head_histogram"])), 2
        )
        self.assertEqual(float(result["fall_rate"]), 0.5)
        self.assertEqual(float(result["termination_rate"]), 0.5)
        self.assertEqual(float(result["tracking_rmse"]), 0.0)


if __name__ == "__main__":
    unittest.main()
