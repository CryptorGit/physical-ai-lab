from pathlib import Path
import sys

import jax.numpy as jnp

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from training.device_metrics import OFF_GRID_COMMAND_ID, aggregate_rollout


def test_exact_and_off_grid_command_counts():
    official = jnp.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    commands = jnp.asarray(
        [[[0.0, 0.0, 0.0, 0, 0, 0, 0], [0.03, 0.02, 0.1, 0, 0, 0.2, 0]]]
    )
    sidecar = {
        "command": commands,
        "done": jnp.asarray([[0, 1]]),
        "fall": jnp.asarray([[0, 1]]),
        "truncation": jnp.asarray([[0, 0]]),
        "episode_start": jnp.asarray([[1, 0]]),
        "reward": jnp.asarray([[1.0, -1.0]]),
        "advantage": jnp.asarray([[0.5, -0.25]]),
        "reward_terms": {"alive": jnp.asarray([[1.0, 0.0]])},
    }
    metrics = aggregate_rollout(
        sidecar,
        official,
        num_updates_per_batch=4,
        vx_edges=jnp.asarray([-0.1, 0.1]),
        vy_edges=jnp.asarray([-0.1, 0.1]),
        yaw_edges=jnp.asarray([-0.1, 0.1]),
        head_edges=jnp.asarray([-0.1, 0.1]),
    )
    assert int(metrics["command_step_count"][0]) == 1
    assert int(metrics["command_step_count"][OFF_GRID_COMMAND_ID]) == 1
    assert int(metrics["minibatch_input_count"][0]) == 4
    assert int(metrics["command_fall_count"][OFF_GRID_COMMAND_ID]) == 1
    assert float(metrics["advantage_sum_by_command"][0]) == 0.5
    assert float(
        metrics["pre_update_policy_loss_sum_by_command"][OFF_GRID_COMMAND_ID]
    ) == 0.25
