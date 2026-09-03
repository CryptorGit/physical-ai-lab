#!/usr/bin/env python3
"""GPU-only v59 corrected 15-second diagnostic evaluator.

This is an immutable-checkpoint research harness, not an acceptance evaluator.
It calls the historical training environment without modifying production or
legacy evaluation code.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pickle
from pathlib import Path
import platform
import subprocess
import sys
import time

import jax
import jax.numpy as jp
import jaxlib
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks as ppo_networks
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper
from mujoco_playground.config import locomotion_params

from playground.common import randomize
from playground.open_duck_mini_v2 import joystick

EXP = Path(__file__).resolve().parents[1]
TOOLS = EXP / "tools"
sys.path.insert(0, str(TOOLS))
from export_v59_stochastic_trace import training_environment_keys
from v59_mjx_diagnostic_common import canonical_tree_sha256, host_tree


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_block(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
            return


def set_commands(state, commands):
    info = dict(state.info)
    info["command"] = commands
    obs = dict(state.obs)
    obs["state"] = obs["state"].at[:, 6:13].set(commands)
    obs["privileged_state"] = obs["privileged_state"].at[:, 6:13].set(commands)
    return state.replace(info=info, obs=obs)


def reset_command_for_key(env, key):
    rng = key
    for _ in range(4):
        rng, _ = jax.random.split(rng)
    _, command_key = jax.random.split(rng)
    return env.sample_command(command_key)


def quat_to_rpy(quat):
    w, x, y, z = [quat[..., i] for i in range(4)]
    roll = jp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = jp.arcsin(jp.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def runtime_environment() -> dict:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    return {
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "mujoco_version": mujoco.__version__,
        "mjx_provenance": "mujoco.mjx from mujoco package",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "device_count": jax.device_count(),
        "process_count": jax.process_count(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "jit_enabled": not bool(jax.config.jax_disable_jit),
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "xla_flags": __import__("os").environ.get("XLA_FLAGS", ""),
        "nvidia_smi": smi,
        "float_precision": "float32",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("D", "S"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--master-seed", type=int, default=0)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required; got {jax.default_backend()}")
    if args.seconds > 15.0:
        raise ValueError("this diagnostic must not exceed 15 seconds")
    if args.seeds != 5:
        raise ValueError("the frozen diagnostic contract requires 5 seeds")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw_logs").mkdir(exist_ok=True)
    (output / "episode_snapshots").mkdir(exist_ok=True)
    command_manifest_path = output / "command_manifest.json"
    command_manifest = json.loads(command_manifest_path.read_text(encoding="utf-8"))
    definitions = command_manifest["commands"]
    count = len(definitions) * args.seeds
    steps = int(round(args.seconds / 0.02))
    if len(definitions) != 19 or steps != 750:
        raise ValueError("frozen evaluation must be 19 commands x 750 steps")

    config = joystick.default_config()
    if args.condition == "D":
        config.noise_config.level = 0.0
        config.noise_config.action_min_delay = 0
        config.noise_config.action_max_delay = 1
        config.push_config.enable = False
    env = joystick.Joystick(
        task="flat_terrain_backlash_calibrated", config=config
    )
    params = checkpoint.load(args.checkpoint)
    ppo_config = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )
    networks = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **dict(ppo_config.network_factory),
    )

    _, five_keys = training_environment_keys(args.master_seed, 4096)
    episode_keys = jp.stack(
        [five_keys[seed] for _ in definitions for seed in range(args.seeds)]
    )
    command_indices = np.repeat(np.arange(19), args.seeds)
    seed_indices = np.tile(np.arange(args.seeds), 19)

    if args.condition == "S":
        wrapper = BraxDomainRandomizationVmapWrapper(
            env, functools.partial(randomize.domain_randomize, rng=episode_keys)
        )
        reset_fn = jax.jit(wrapper.reset)
        step_batch = wrapper.step
        state = reset_fn(episode_keys)
        randomized_model = wrapper._mjx_model_v
        model_in_axes = wrapper._in_axes
        reset_keys = episode_keys
        head_commands = state.info["command"][:, 3:7]
    else:
        fixed_reset_key = five_keys[0]
        reset_keys = jp.repeat(fixed_reset_key[None, :], count, axis=0)
        reset_fn = jax.jit(jax.vmap(env.reset))
        step_batch = jax.vmap(env.step)
        state = reset_fn(reset_keys)
        randomized_model = env.mjx_model
        model_in_axes = jax.tree_util.tree_map(lambda _: None, env.mjx_model)
        per_seed_head = jp.stack(
            [reset_command_for_key(env, five_keys[i])[3:7] for i in range(5)]
        )
        head_commands = per_seed_head[jp.asarray(seed_indices)]

    body_commands = jp.asarray(
        [
            [definitions[index]["vx"], definitions[index]["vy"], definitions[index]["yaw_rate"]]
            for index in command_indices
        ],
        dtype=jp.float32,
    )
    commands = jp.concatenate([body_commands, head_commands], axis=1)
    state = set_commands(state, commands)
    policy_keys = jp.stack(
        [jax.random.fold_in(key, 0x59334233) for key in episode_keys]
    )

    snapshot = {
        "schema_version": 1,
        "condition": args.condition,
        "state": host_tree(state),
        "complete_mjx_data": host_tree(state.data),
        "randomized_model_or_reference_model": host_tree(randomized_model),
        "model_in_axes": host_tree(model_in_axes),
        "commands": np.asarray(commands),
        "master_seed": args.master_seed,
        "environment_keys": np.asarray(episode_keys),
        "reset_keys": np.asarray(reset_keys),
        "policy_keys": np.asarray(policy_keys),
        "controller_internal_state": host_tree(state.info),
        "delay_buffer": np.asarray(state.info["action_history"]),
        "rng_keys": np.asarray(state.info["rng"]),
    }
    snapshot_path = (
        output / "episode_snapshots" / f"condition_{args.condition.lower()}_batched.pkl"
    )
    with snapshot_path.open("wb") as stream:
        pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)

    resolved_config = env._config
    scales = {
        key: float(value)
        for key, value in resolved_config.reward_config.scales.items()
        if float(value) != 0.0
    }
    metric_keys = tuple(sorted(scales))
    weights = jp.asarray([abs(scales[key]) for key in metric_keys])
    actuator_qpos_addr = jp.asarray(
        np.asarray(env._mj_model.jnt_qposadr)[np.asarray(env.actuator_joint_ids)]
    )

    def rollout(initial_state, initial_policy_keys, params):
        def body(carry, _):
            current, keys = carry
            logits = networks.policy_network.apply(params[0], params[1], current.obs)
            loc, raw_scale = jp.split(logits, 2, axis=-1)
            split_keys = jax.vmap(jax.random.split)(keys)
            action_keys = split_keys[:, 0]
            next_keys = split_keys[:, 1]
            eps = jax.vmap(
                lambda key: jax.random.normal(key, (env.action_size,))
            )(action_keys)
            stochastic_action = jp.tanh(
                loc + (jax.nn.softplus(raw_scale) + 0.001) * eps
            )
            action = jp.tanh(loc) if args.condition == "D" else stochastic_action
            before = current
            stepped = step_batch(current, action)

            # Reproduce the exact delay draw from the pre-step environment RNG.
            split_environment = jax.vmap(
                lambda key: jax.random.split(key, 4)
            )(before.info["rng"])
            delay_keys = split_environment[:, 3]
            push_magnitude_keys = split_environment[:, 2]
            delays = jax.vmap(
                lambda key: jax.random.randint(
                    key,
                    (),
                    minval=env._config.noise_config.action_min_delay,
                    maxval=env._config.noise_config.action_max_delay,
                )
            )(delay_keys)
            histories = stepped.info["action_history"].reshape(
                (count, -1, env.action_size)
            )
            delayed = histories[jp.arange(count), delays]
            push_magnitudes = jax.vmap(
                lambda key: jax.random.uniform(
                    key,
                    minval=env._config.push_config.magnitude_range[0],
                    maxval=env._config.push_config.magnitude_range[1],
                )
            )(push_magnitude_keys)

            data = stepped.data
            local_velocity = jax.vmap(env.get_local_linvel)(data)
            gyro = jax.vmap(env.get_gyro)(data)
            gravity = jax.vmap(env.get_gravity)(data)
            quat = data.qpos[:, env._floating_base_qpos_addr + 3 : env._floating_base_qpos_addr + 7]
            roll, pitch, heading = quat_to_rpy(quat)
            contacts = stepped.obs["state"][:, 97:99] > 0.5
            feet_velocity = data.sensordata[:, env._foot_linvel_sensor_adr]
            slip = jp.linalg.norm(feet_velocity[:, :, :2], axis=-1)
            joint_qpos = data.qpos[:, actuator_qpos_addr]
            joint_qvel = jax.vmap(env.get_actuator_joints_qvel)(data.qvel)
            teacher_active = commands[:, 0] < -0.02
            teacher = env._default_actuator[None, :].repeat(count, axis=0)
            reference = stepped.info["current_reference_motion"]
            teacher = teacher.at[:, env._backward_actuator_indices].set(
                reference[:, env._backward_joint_indices]
            )
            teacher = jp.where(teacher_active[:, None], teacher, jp.zeros_like(teacher))
            bounded = jp.clip(delayed, -1.0, 1.0)
            positive_span = 0.9 * (
                env._actuator_uppers - env._default_actuator
            )
            negative_span = 0.9 * (
                env._default_actuator - env._actuator_lowers
            )
            directional_span = jp.where(
                bounded >= 0.0, positive_span, negative_span
            )
            base_span = jp.minimum(env._config.action_scale, directional_span)
            magnitude = jp.abs(bounded)
            direct_target = env._default_actuator + jp.sign(bounded) * (
                base_span * magnitude
                + (directional_span - base_span) * magnitude**5
            )
            turn_blend = jp.clip(jp.abs(commands[:, 2]) / 0.20, 0.0, 1.0)
            residual_scales = env._backward_residual_scale * jp.maximum(
                0.50, 1.0 - 0.50 * turn_blend
            )
            residual_scales = jp.repeat(
                residual_scales[:, None], env.action_size, axis=1
            )
            residual_scales = residual_scales.at[:, 5:9].set(
                (
                    env._backward_residual_scale
                    * (1.0 - 0.5 * turn_blend)
                )[:, None]
            )
            combined_pre_limit = jp.where(
                teacher_active[:, None],
                teacher + residual_scales * bounded,
                direct_target,
            )
            raw_metrics = jp.stack(
                [
                    stepped.metrics[
                        ("reward/" if scales[key] > 0 else "cost/") + key
                    ]
                    for key in metric_keys
                ],
                axis=-1,
            )
            contributions = raw_metrics * weights[None, :]
            head_violation = jax.vmap(env._head_frame_violation)(data)
            fall = (gravity[:, -1] < 0.65) | (
                data.qpos[:, env._floating_base_qpos_addr + 2] < 0.12
            )
            termination_reason = (
                fall.astype(jp.int32)
                + 2 * (head_violation > 0).astype(jp.int32)
                + 4 * (
                    jp.isnan(data.qpos).any(axis=1)
                    | jp.isnan(data.qvel).any(axis=1)
                ).astype(jp.int32)
            )
            target_limited = stepped.info["target_limit_violation"] > 0
            joint_limit = (joint_qpos <= env._actuator_lowers[None, :]) | (
                joint_qpos >= env._actuator_uppers[None, :]
            )
            logs = {
                "qpos": data.qpos,
                "qvel": data.qvel,
                "joint_qpos": joint_qpos,
                "joint_qvel": joint_qvel,
                "actual_velocity": local_velocity,
                "actual_yaw_rate": gyro[:, 2],
                "heading": heading,
                "roll": roll,
                "pitch": pitch,
                "base_height": data.qpos[:, env._floating_base_qpos_addr + 2],
                "vertical_velocity": local_velocity[:, 2],
                "teacher_active": teacher_active,
                "teacher_phase": stepped.info["imitation_phase"],
                "teacher_action": teacher,
                "actor_residual": action,
                "combined_action": combined_pre_limit,
                "delayed_action": delayed,
                "motor_target": stepped.info["motor_targets"],
                "foot_contacts": contacts,
                "foot_slip": slip,
                "actuator_force": data.actuator_force,
                "solver_niter": data._impl.solver_niter,
                "joint_limit_state": joint_limit,
                "action_clip_state": jp.abs(action) >= 1.0,
                "target_limit_state": target_limited,
                "termination": stepped.done > 0,
                "termination_reason_code": termination_reason,
                "fall": fall,
                "delay_index": delays,
                "push_direction_gate": stepped.info["push"],
                "push_velocity_increment": (
                    stepped.info["push"] * push_magnitudes[:, None]
                ),
                "reward_contribution": contributions,
            }
            # The evaluated command, including head components, remains fixed.
            stepped = set_commands(stepped, commands)
            return (stepped, next_keys), logs

        return jax.lax.scan(
            body, (initial_state, initial_policy_keys), None, length=steps
        )

    compiled = jax.jit(rollout)
    started = time.time()
    (final_state, final_policy_keys), logs = compiled(state, policy_keys, params)
    tree_block(logs)
    elapsed = time.time() - started
    logs_host = host_tree(logs)
    raw_path = output / "raw_logs" / f"condition_{args.condition.lower()}_raw.npz"
    np.savez_compressed(
        raw_path,
        **logs_host,
        commands=np.asarray(commands),
        command_indices=command_indices,
        seed_indices=seed_indices,
        environment_keys=np.asarray(episode_keys),
        reset_keys=np.asarray(reset_keys),
        initial_qpos=np.asarray(state.data.qpos),
        initial_qvel=np.asarray(state.data.qvel),
        initial_head_command=np.asarray(head_commands),
        initial_delay_buffer=np.asarray(state.info["action_history"]),
        initial_teacher_phase=np.asarray(state.info["imitation_phase"]),
        metric_keys=np.asarray(metric_keys),
        metric_scales=np.asarray([scales[key] for key in metric_keys]),
    )
    metadata = {
        "condition": args.condition,
        "diagnostic_only": True,
        "formal_acceptance_eligible": False,
        "enough_episodes": False,
        "requested_seconds": args.seconds,
        "control_steps": steps,
        "episodes": count,
        "commands": 19,
        "seeds_per_command": 5,
        "elapsed_wall_seconds_including_compile": elapsed,
        "runtime": runtime_environment(),
        "checkpoint": args.checkpoint,
        "scene": str(env._xml_path),
        "push": {
            "enabled": bool(resolved_config.push_config.enable),
            "implementation": "episode-time overwrite/addition of floating-base xy qvel; not external force",
            "interval_seconds": list(resolved_config.push_config.interval_range),
            "magnitude_range_m_per_s": list(resolved_config.push_config.magnitude_range),
        },
        "noise_level": float(resolved_config.noise_config.level),
        "action_delay_min_inclusive": int(resolved_config.noise_config.action_min_delay),
        "action_delay_max_exclusive": int(resolved_config.noise_config.action_max_delay),
        "head_locked": False,
        "head_command_source": "reset sample_command components 3:7; fixed for episode",
        "command_changes": 0,
        "snapshot": str(snapshot_path),
        "snapshot_tree_sha256": canonical_tree_sha256(snapshot),
        "raw_log": str(raw_path),
        "raw_log_sha256": sha256(raw_path),
        "metric_keys": metric_keys,
        "metric_scales": scales,
    }
    (output / f"condition_{args.condition.lower()}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
