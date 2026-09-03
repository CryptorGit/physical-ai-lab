"""Pure JAX telemetry reducers; these functions never draw random numbers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp


OFF_GRID_COMMAND_ID = 19
COMMAND_CLASS_COUNT = 20


def exact_command_ids(
    commands: jax.Array, official_commands: jax.Array, atol: float = 1e-7
) -> jax.Array:
    """Returns 0..18 for an exact official command and 19 for off-grid samples."""
    body = commands[..., :3]
    equal = jnp.all(jnp.abs(body[..., None, :] - official_commands) <= atol, axis=-1)
    candidate = jnp.argmax(equal, axis=-1)
    return jnp.where(jnp.any(equal, axis=-1), candidate, OFF_GRID_COMMAND_ID)


def fixed_bin(values: jax.Array, edges: jax.Array) -> jax.Array:
    return jnp.clip(jnp.digitize(values, edges), 0, edges.shape[0])


def joint_command_head_bins(
    commands: jax.Array,
    *,
    vx_edges: jax.Array,
    vy_edges: jax.Array,
    yaw_edges: jax.Array,
    head_edges: jax.Array,
) -> tuple[jax.Array, int]:
    """Flattens P(vx, vy, yaw, head-yaw) into a stable dense bin id."""
    vx = fixed_bin(commands[..., 0], vx_edges)
    vy = fixed_bin(commands[..., 1], vy_edges)
    yaw = fixed_bin(commands[..., 2], yaw_edges)
    head = fixed_bin(commands[..., 5], head_edges)
    sizes = (
        vx_edges.shape[0] + 1,
        vy_edges.shape[0] + 1,
        yaw_edges.shape[0] + 1,
        head_edges.shape[0] + 1,
    )
    flat = (((vx * sizes[1] + vy) * sizes[2] + yaw) * sizes[3] + head)
    return flat, int(sizes[0] * sizes[1] * sizes[2] * sizes[3])


def _counts(ids: jax.Array, weights: jax.Array, length: int) -> jax.Array:
    if weights.dtype == jnp.bool_:
        weights = weights.astype(jnp.int32)
    return jnp.bincount(ids.reshape(-1), weights=weights.reshape(-1), length=length)


def aggregate_rollout(
    sidecar: Mapping[str, jax.Array],
    official_commands: jax.Array,
    *,
    num_updates_per_batch: int,
    vx_edges: jax.Array,
    vy_edges: jax.Array,
    yaw_edges: jax.Array,
    head_edges: jax.Array,
) -> dict[str, jax.Array]:
    """Produces update-sized telemetry from scan outputs on device."""
    commands = sidecar["command"]
    ids = exact_command_ids(commands, official_commands)
    ones = jnp.ones(ids.shape, dtype=jnp.int32)
    done = sidecar["done"].astype(jnp.int32)
    fall = sidecar["fall"].astype(jnp.int32)
    truncation = sidecar["truncation"].astype(jnp.int32)
    valid_advantage = 1 - truncation
    episode_start = sidecar["episode_start"].astype(jnp.int32)
    joint_ids, joint_length = joint_command_head_bins(
        commands,
        vx_edges=vx_edges,
        vy_edges=vy_edges,
        yaw_edges=yaw_edges,
        head_edges=head_edges,
    )
    result = {
        "command_step_count": _counts(ids, ones, COMMAND_CLASS_COUNT),
        "command_episode_count": _counts(ids, episode_start, COMMAND_CLASS_COUNT),
        "command_survived_step_count": _counts(
            ids, 1 - done, COMMAND_CLASS_COUNT
        ),
        "command_termination_count": _counts(ids, done, COMMAND_CLASS_COUNT),
        "command_fall_count": _counts(ids, fall, COMMAND_CLASS_COUNT),
        "command_start_count": _counts(ids, episode_start, COMMAND_CLASS_COUNT),
        "rollout_sample_count": _counts(ids, ones, COMMAND_CLASS_COUNT),
        "valid_advantage_sample_count": _counts(
            ids, valid_advantage, COMMAND_CLASS_COUNT
        ),
        "minibatch_input_count": _counts(
            ids,
            valid_advantage * num_updates_per_batch,
            COMMAND_CLASS_COUNT,
        ),
        "termination_sample_count": _counts(ids, done, COMMAND_CLASS_COUNT),
        "joint_command_head_histogram": _counts(
            joint_ids, ones, joint_length
        ),
        "positive_yaw_sample_count": jnp.sum(commands[..., 2] > 0),
        "negative_yaw_sample_count": jnp.sum(commands[..., 2] < 0),
        "positive_yaw_survived_count": jnp.sum(
            (commands[..., 2] > 0) * (1 - done)
        ),
        "negative_yaw_survived_count": jnp.sum(
            (commands[..., 2] < 0) * (1 - done)
        ),
        "reward_sum_by_command": _counts(
            ids, sidecar["reward"], COMMAND_CLASS_COUNT
        ),
        "rollout_sample_total": jnp.asarray(ids.size, dtype=jnp.int32),
        "reward_sum": jnp.sum(sidecar["reward"]),
        "reward_mean": jnp.mean(sidecar["reward"]),
        "fall_count": jnp.sum(fall),
        "fall_rate": jnp.mean(fall.astype(jnp.float32)),
        "termination_count": jnp.sum(done),
        "termination_rate": jnp.mean(done.astype(jnp.float32)),
    }
    if "advantage" in sidecar:
        advantage = sidecar["advantage"]
        result["advantage_sum_by_command"] = _counts(
            ids, advantage, COMMAND_CLASS_COUNT
        )
        result["advantage_square_sum_by_command"] = _counts(
            ids, jnp.square(advantage), COMMAND_CLASS_COUNT
        )
        # At rollout collection the behaviour and target policies are identical,
        # so rho=1 and the per-sample surrogate contribution is -advantage.
        result["pre_update_policy_loss_sum_by_command"] = _counts(
            ids, -advantage, COMMAND_CLASS_COUNT
        )
        for sign_name, sign_mask in (
            ("positive_yaw", commands[..., 2] > 0),
            ("negative_yaw", commands[..., 2] < 0),
        ):
            sign_count = jnp.sum(sign_mask)
            sign_sum = jnp.sum(jnp.where(sign_mask, advantage, 0.0))
            sign_square_sum = jnp.sum(
                jnp.where(sign_mask, jnp.square(advantage), 0.0)
            )
            result[f"{sign_name}_valid_advantage_count"] = jnp.sum(
                sign_mask * valid_advantage
            )
            result[f"{sign_name}_advantage_mean"] = sign_sum / jnp.maximum(
                sign_count, 1
            )
            variance = sign_square_sum / jnp.maximum(sign_count, 1) - jnp.square(
                result[f"{sign_name}_advantage_mean"]
            )
            result[f"{sign_name}_advantage_std"] = jnp.sqrt(
                jnp.maximum(variance, 0.0)
            )
            result[f"{sign_name}_pre_update_policy_loss_sum"] = -sign_sum
    if "actual_velocity" in sidecar:
        velocity = sidecar["actual_velocity"]
        tracking_error = velocity - commands[..., :3]
        result["tracking_squared_error_sum"] = jnp.sum(
            jnp.square(tracking_error)
        )
        result["tracking_rmse"] = jnp.sqrt(jnp.mean(jnp.square(tracking_error)))
        result["actual_vx_mean"] = jnp.mean(velocity[..., 0])
        result["actual_vy_mean"] = jnp.mean(velocity[..., 1])
        result["actual_yaw_mean"] = jnp.mean(velocity[..., 2])
        for axis, name in enumerate(("vx", "vy", "yaw")):
            result[f"actual_{name}_sum_by_command"] = _counts(
                ids, velocity[..., axis], COMMAND_CLASS_COUNT
            )
    for name, value in sidecar.get("reward_terms", {}).items():
        result[f"reward_term/{name}/sum_by_command"] = _counts(
            ids, value, COMMAND_CLASS_COUNT
        )
        result[f"reward_term/{name}/active_by_command"] = _counts(
            ids, value != 0, COMMAND_CLASS_COUNT
        )
    return result


def tree_l2_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def tree_nonfinite_count(tree: Any) -> jax.Array:
    return sum(
        jnp.sum(~jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)
    )


def telemetry_transfer_bytes(metrics: Mapping[str, Any]) -> int:
    """Returns the exact host payload size after device aggregation."""
    import numpy as np

    return int(
        sum(np.asarray(x).nbytes for x in jax.tree_util.tree_leaves(metrics))
    )
