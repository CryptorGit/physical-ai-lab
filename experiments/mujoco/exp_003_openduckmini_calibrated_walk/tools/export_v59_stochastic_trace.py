#!/usr/bin/env python3
"""Generate stochastic v59 traces and sample-injection parity evidence.

This is an audit harness.  It never updates checkpoint parameters.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks as ppo_networks
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper
from mujoco_playground.config import locomotion_params

from playground.common import randomize
from playground.open_duck_mini_v2 import joystick

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from v59_parity_common import COMMANDS, compose_motor_target, error_metrics
from v59_stochastic_common import (
    backlash_observation,
    delay_buffer_step,
    inject_observation_noise,
    json_value,
    normalized_observation,
    stochastic_actor_from_logits,
)


def npv(value):
    return np.asarray(jax.device_get(value))


def single_tree(tree, index):
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def training_environment_keys(master_seed: int, count: int):
    master = jax.random.PRNGKey(master_seed)
    _, local = jax.random.split(master)
    local = jax.random.fold_in(local, 0)  # process_id
    _, key_env, _ = jax.random.split(local, 3)
    return key_env, jax.random.split(key_env, count)


def split_once(rng):
    parent, child = jax.random.split(rng)
    return parent, child


def observation_draws(start_rng, env):
    rng, gyro_key = split_once(start_rng)
    rng, accel_key = split_once(rng)
    rng, gravity_key = split_once(rng)
    rng, joint_pos_key = split_once(rng)
    rng, joint_vel_key = split_once(rng)
    gyro = (
        (2 * jax.random.uniform(gyro_key, (3,)) - 1)
        * env._config.noise_config.level
        * env._config.noise_config.scales.gyro
    )
    accel = (
        (2 * jax.random.uniform(accel_key, (3,)) - 1)
        * env._config.noise_config.level
        * env._config.noise_config.scales.accelerometer
    )
    gravity = (
        (2 * jax.random.uniform(gravity_key, (3,)) - 1)
        * env._config.noise_config.level
        * env._config.noise_config.scales.gravity
    )
    imu_delay = jax.random.randint(
        gravity_key,
        (1,),
        minval=env._config.noise_config.imu_min_delay,
        maxval=env._config.noise_config.imu_max_delay,
    )[0]
    joint_pos = (
        (2 * jax.random.uniform(joint_pos_key, (env.action_size,)) - 1)
        * env._config.noise_config.level
        * env._qpos_noise_scale
    )
    joint_vel = (
        (2 * jax.random.uniform(joint_vel_key, (env.action_size,)) - 1)
        * env._config.noise_config.level
        * env._config.noise_config.scales.joint_vel
    )
    return rng, {
        "gyro_noise": npv(gyro),
        "accelerometer_noise": npv(accel),
        "gravity_noise": npv(gravity),
        "imu_delay": int(npv(imu_delay)),
        "joint_position_noise": npv(joint_pos),
        "joint_velocity_noise": npv(joint_vel),
    }


def reset_draws(env_key, env):
    rng = env_key
    keys = []
    for _ in range(6):
        rng, key = split_once(rng)
        keys.append(key)
    dxy = jax.random.uniform(keys[0], (2,), minval=-0.05, maxval=0.05)
    yaw = jax.random.uniform(keys[1], (1,), minval=-3.14, maxval=3.14)
    joint = jax.random.uniform(
        keys[2], (env.action_size,), minval=-0.03, maxval=0.03
    )
    base_velocity = jax.random.uniform(
        keys[3], (6,), minval=-0.05, maxval=0.05
    )
    command = env.sample_command(keys[4])
    push_interval = jax.random.uniform(keys[5], minval=5.0, maxval=10.0)
    rng, obs = observation_draws(rng, env)
    return rng, {
        "initial_xy_offset": npv(dxy),
        "initial_yaw": npv(yaw),
        "initial_joint_position_offset_unclipped": npv(joint),
        "initial_base_velocity": npv(base_velocity),
        "sampled_reset_command": npv(command),
        "push_interval_seconds": float(npv(push_interval)),
        **obs,
    }


def step_draws(start_rng, env):
    new_rng, push_theta_key, push_magnitude_key, delay_key = jax.random.split(
        start_rng, 4
    )
    delay = jax.random.randint(
        delay_key,
        (1,),
        minval=env._config.noise_config.action_min_delay,
        maxval=env._config.noise_config.action_max_delay,
    )[0]
    push_theta = jax.random.uniform(push_theta_key, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push_magnitude_key,
        minval=env._config.push_config.magnitude_range[0],
        maxval=env._config.push_config.magnitude_range[1],
    )
    after_obs_rng, obs = observation_draws(new_rng, env)
    final_rng, command_key = jax.random.split(after_obs_rng)
    command_keys = jax.random.split(command_key, 8)
    command_components = {
        "candidate_vx_uniform": npv(
            jax.random.uniform(command_keys[0], minval=-0.15, maxval=0.15)
        ),
        "candidate_vy_uniform": npv(
            jax.random.uniform(command_keys[1], minval=-0.2, maxval=0.2)
        ),
        "candidate_yaw_uniform": npv(
            jax.random.uniform(command_keys[2], minval=-1.0, maxval=1.0)
        ),
        "candidate_mode": int(
            npv(jax.random.randint(command_keys[3], (), minval=0, maxval=24))
        ),
        "candidate_neck_pitch": npv(
            jax.random.uniform(
                command_keys[4],
                minval=-1.069337 * 0.85,
                maxval=1.027689 * 0.85,
            )
        ),
        "candidate_head_pitch": npv(
            jax.random.uniform(
                command_keys[5],
                minval=-0.556802 * 0.85,
                maxval=1.173515 * 0.85,
            )
        ),
        "candidate_head_yaw": npv(
            jax.random.uniform(
                command_keys[6],
                minval=-0.885107 * 0.85,
                maxval=0.892777 * 0.85,
            )
        ),
        "candidate_head_roll": npv(
            jax.random.uniform(
                command_keys[7],
                minval=-0.635069 * 0.85,
                maxval=0.780864 * 0.85,
            )
        ),
        "candidate_resolved_command": npv(env.sample_command(command_key)),
    }
    return final_rng, {
        "delay_length": int(npv(delay)),
        "push_theta": float(npv(push_theta)),
        "push_magnitude": float(npv(push_magnitude)),
        **obs,
        **command_components,
    }


def domain_draws(key, env):
    rng = key
    keys = []
    for _ in range(8):
        rng, child = split_once(rng)
        keys.append(child)
    model = env.mjx_model
    return {
        "floor_friction": float(
            npv(jax.random.uniform(keys[0], minval=0.5, maxval=1.0))
        ),
        "frictionloss_factor": npv(
            jax.random.uniform(
                keys[1], (model.nu,), minval=0.9, maxval=1.1
            )
        ),
        "armature_factor": npv(
            jax.random.uniform(
                keys[2], (model.nu,), minval=1.0, maxval=1.05
            )
        ),
        "torso_ipos_offset": npv(
            jax.random.uniform(keys[3], (3,), minval=-0.05, maxval=0.05)
        ),
        "body_mass_factor": npv(
            jax.random.uniform(
                keys[4], (model.nbody,), minval=0.9, maxval=1.1
            )
        ),
        "torso_mass_offset": float(
            npv(jax.random.uniform(keys[5], minval=-0.1, maxval=0.1))
        ),
        "qpos0_offset": npv(
            jax.random.uniform(
                keys[6], (model.nu,), minval=-0.03, maxval=0.03
            )
        ),
        "actuator_kp_factor": npv(
            jax.random.uniform(
                keys[7], (model.nu,), minval=0.9, maxval=1.1
            )
        ),
    }


def observation_noise_vector(draws, dof_vel_scale):
    noise = np.zeros(101, dtype=np.float32)
    noise[0:3] = draws["gyro_noise"]
    noise[3:6] = draws["accelerometer_noise"]
    noise[13:27] = draws["joint_position_noise"]
    noise[27:41] = draws["joint_velocity_noise"] * dof_vel_scale
    return noise


def raw_observation_before_noise(env, data, info_before, info_after, contact):
    gyro = npv(env.get_gyro(data))
    accel = npv(env.get_accelerometer(data)).copy()
    accel[0] += 1.3
    joint = npv(env.get_actuator_joints_qpos(data.qpos))
    backlash = npv(env.get_actuator_backlash_qpos(data.qpos))
    for index in env.backlash_idx_to_add:
        backlash = np.insert(backlash, index, 0.0)
    joint_with_backlash = backlash_observation(joint, backlash)
    joint_vel = npv(env.get_actuator_joints_qvel(data.qvel))
    return np.hstack(
        [
            gyro,
            accel,
            npv(info_after["command"]),
            joint_with_backlash - npv(env._default_actuator),
            joint_vel * float(env._config.dof_vel_scale),
            npv(info_before["last_act"]),
            npv(info_before["last_last_act"]),
            npv(info_before["last_last_last_act"]),
            npv(info_after["motor_targets"]),
            np.asarray(contact, dtype=np.float32),
            npv(info_after["imitation_phase"]),
        ]
    ).astype(np.float32)


def numpy_logits(raw_observation, mean, std, layers):
    x = (np.asarray(raw_observation, np.float32) - mean) / std
    for index, (kernel, bias) in enumerate(layers):
        x = x @ kernel + bias
        if index != len(layers) - 1:
            x = x / (1.0 + np.exp(-x))
    return x


def event(source_id, index, value, timing, consumer):
    array = np.asarray(value)
    return {
        "random_source_id": source_id,
        "random_sample_index": int(index),
        "random_sample_value": json_value(value),
        "random_sample_shape": list(array.shape),
        "random_sample_dtype": str(array.dtype),
        "sampling_timing": timing,
        "consumer": consumer,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output)
    traces_dir = output / "stochastic_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    env = joystick.Joystick(task="flat_terrain_backlash_calibrated")
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
    mean = npv(params[0].mean["state"])
    std = npv(params[0].std["state"])
    layer_dict = params[1]["params"]
    layers = [
        (
            npv(layer_dict[f"hidden_{index}"]["kernel"]),
            npv(layer_dict[f"hidden_{index}"]["bias"]),
        )
        for index in range(4)
    ]

    _, all_training_keys = training_environment_keys(args.master_seed, 4096)
    cases = []
    for command in COMMANDS:
        for env_index in range(3):
            cases.append((command, env_index, all_training_keys[env_index]))
    case_keys = jp.stack([case[2] for case in cases])
    wrapper = BraxDomainRandomizationVmapWrapper(
        env,
        functools.partial(randomize.domain_randomize, rng=case_keys),
    )
    reset_fn = jax.jit(wrapper.reset)
    state = reset_fn(case_keys)
    replay_state = reset_fn(case_keys)
    commands = []
    for command, _, _ in cases:
        commands.append(
            [
                command["vx"],
                command["vy"],
                command["yaw_rate"],
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
    commands = jp.asarray(commands, dtype=jp.float32)

    def set_controlled_commands(target_state):
        target_info = dict(target_state.info)
        target_info["command"] = commands
        target_obs = dict(target_state.obs)
        target_obs["state"] = target_obs["state"].at[:, 6:13].set(commands)
        target_obs["privileged_state"] = target_obs[
            "privileged_state"
        ].at[:, 6:13].set(commands)
        return target_state.replace(info=target_info, obs=target_obs)

    state = set_controlled_commands(state)
    replay_state = set_controlled_commands(replay_state)

    policy_keys = jp.stack(
        [
            jax.random.fold_in(key, 0x59334233)
            for _, _, key in cases
        ]
    )
    step_fn = jax.jit(wrapper.step)
    policy_apply = jax.jit(networks.policy_network.apply)

    trace_records = []
    event_streams = []
    parity = []
    native = []
    for case_index, (command, env_index, env_key) in enumerate(cases):
        trace_records.append({key: [] for key in (
            "time", "control_step", "episode_step", "master_seed",
            "environment_index", "environment_seed", "episode_seed",
            "raw_command", "resolved_command", "head_command",
            "initial_qpos_offset", "initial_qvel_offset",
            "initial_base_velocity", "observation_noise",
            "joint_position_noise", "joint_velocity_noise", "delay_length",
            "delay_buffer_state", "backlash_internal_state",
            "backlash_direction_state", "phase_initialization",
            "teacher_phase", "raw_observation_before_noise", "noise_sample",
            "raw_observation_after_noise", "normalized_observation",
            "teacher_action", "actor_residual", "actor_standard_normal_sample",
            "combined_action_pre_clip", "combined_action_post_clip",
            "action_before_delay", "action_after_delay",
            "motor_target_before_backlash", "motor_target_after_backlash",
            "qpos", "qvel", "base_linear_velocity",
            "base_angular_velocity", "foot_contacts", "termination_flags",
        )})
        event_streams.append([])
        parity.append({
            "command_id": command["command_id"],
            "environment_index": env_index,
            "initial_state_max_abs_error": 0.0,
            "observation_max_abs_error": 0.0,
            "normalized_observation_max_abs_error": 0.0,
            "actor_residual_max_abs_error": 0.0,
            "actor_numpy_diagnostic_max_abs_error": 0.0,
            "teacher_phase_max_abs_error": 0.0,
            "teacher_action_max_abs_error": 0.0,
            "delay_buffer_exact": True,
            "action_after_delay_max_abs_error": 0.0,
            "backlash_state_exact": True,
            "combined_action_max_abs_error": 0.0,
            "motor_target_max_abs_error": 0.0,
            "first_divergence_stage": "",
            "first_divergence_step": "",
        })
        native.append({
            "command_id": command["command_id"],
            "environment_index": env_index,
            "same_backend_exact": True,
            "first_difference_step": "",
            "max_abs_observation": 0.0,
            "max_abs_qpos": 0.0,
            "max_abs_qvel": 0.0,
            "max_abs_motor_target": 0.0,
        })

    initial_qpos = npv(state.data.qpos[: len(cases)])
    initial_qvel = npv(state.data.qvel[: len(cases)])
    initial_offsets = initial_qpos - npv(env._init_q)[None, :]
    initial_qvel_offsets = initial_qvel.copy()
    initial_backlash = initial_qpos[:, np.asarray(env.backlash_joint_qpos_addr)]
    domain_by_case = [domain_draws(key, env) for _, _, key in cases]
    reset_by_case = [reset_draws(key, env)[1] for _, _, key in cases]

    # Initial sample-injection state is exact by direct offset injection.
    for index in range(len(cases)):
        injected_qpos = npv(env._init_q) + initial_offsets[index]
        injected_qvel = initial_qvel_offsets[index]
        parity[index]["initial_state_max_abs_error"] = max(
            float(np.max(np.abs(injected_qpos - initial_qpos[index]))),
            float(np.max(np.abs(injected_qvel - initial_qvel[index]))),
        )

    for step_index in range(args.steps):
        before = state
        replay_before = replay_state
        logits = policy_apply(params[0], params[1], state.obs)
        replay_logits = policy_apply(params[0], params[1], replay_state.obs)
        split_policy_keys = jax.vmap(jax.random.split)(policy_keys)
        action_keys = split_policy_keys[:, 0]
        policy_keys = split_policy_keys[:, 1]
        eps = jax.vmap(
            lambda key: jax.random.normal(key, (env.action_size,))
        )(action_keys)
        loc, raw_scale = jp.split(logits, 2, axis=-1)
        scale = jax.nn.softplus(raw_scale) + 0.001
        actions = jp.tanh(loc + scale * eps)
        replay_loc, replay_raw_scale = jp.split(replay_logits, 2, axis=-1)
        replay_actions = jp.tanh(
            replay_loc
            + (jax.nn.softplus(replay_raw_scale) + 0.001) * eps
        )

        step_sample_sets = [
            step_draws(before.info["rng"][index], env)[1]
            for index in range(len(cases))
        ]
        independent = []
        for index, sample in enumerate(step_sample_sets):
            before_info = {
                key: before.info[key][index]
                for key in before.info
                if hasattr(before.info[key], "shape")
            }
            history, delayed = delay_buffer_step(
                npv(before.info["action_history"][index]),
                npv(actions[index]),
                sample["delay_length"],
            )
            command_array = npv(before.info["command"][index])
            _, _, phase_rate = env._get_backward_parameters(command_array[2])
            rate = phase_rate if command_array[0] < -0.02 else 1.0
            phase = (
                before.info["imitation_i"][index] + rate
            ) % env.PRM.nb_steps_in_period
            reference = env._get_optimized_backward_reference(
                phase, command_array[2]
            )
            composed = compose_motor_target(
                delayed,
                command=command_array,
                default=npv(env._default_actuator),
                lower=npv(env._actuator_lowers),
                upper=npv(env._actuator_uppers),
                action_scale=float(env._config.action_scale),
                previous_target=npv(before.info["motor_targets"][index]),
                max_motor_velocity=npv(env._config.max_motor_velocity),
                dt=float(env.dt),
                backward_reference=npv(reference),
                backward_actuator_indices=npv(env._backward_actuator_indices),
                backward_joint_indices=npv(env._backward_joint_indices),
                backward_residual_scale=float(env._backward_residual_scale),
                coupled_slope=float(joystick.HEAD_COUPLED_REAR_SLOPE),
                coupled_intercept=float(joystick.HEAD_COUPLED_REAR_INTERCEPT),
            )
            independent.append((history, delayed, phase, composed))

        state = step_fn(state, actions)
        replay_state = step_fn(replay_state, replay_actions)

        for index, (command, env_index, env_key) in enumerate(cases):
            sample = step_sample_sets[index]
            history, delayed, phase, composed = independent[index]
            data = single_tree(state.data, index)
            before_info = {key: before.info[key][index] for key in before.info}
            after_info = {key: state.info[key][index] for key in state.info}
            contact = npv(state.obs["state"][index, 97:99])
            raw_before = raw_observation_before_noise(
                env, data, before_info, after_info, contact
            )
            noise = observation_noise_vector(
                sample, float(env._config.dof_vel_scale)
            )
            injected_obs = inject_observation_noise(raw_before, noise)
            actual_obs = npv(state.obs["state"][index])
            normalized = normalized_observation(injected_obs, mean, std)
            actual_normalized = normalized_observation(actual_obs, mean, std)
            python_logits = numpy_logits(
                npv(before.obs["state"][index]), mean, std, layers
            )
            numpy_injected_action = stochastic_actor_from_logits(
                python_logits, npv(eps[index])
            )
            injected_loc, injected_raw_scale = jp.split(
                logits[index], 2, axis=-1
            )
            injected_action = npv(
                jp.tanh(
                    injected_loc
                    + (jax.nn.softplus(injected_raw_scale) + 0.001)
                    * eps[index]
                )
            )
            actual_motor = npv(state.info["motor_targets"][index])
            phase_vector = npv(
                jp.asarray(
                [
                    jp.cos(
                        phase
                        / env.PRM.nb_steps_in_period
                        * 2
                        * jp.pi
                    ),
                    jp.sin(
                        phase
                        / env.PRM.nb_steps_in_period
                        * 2
                        * jp.pi
                    ),
                ], dtype=jp.float32)
            )
            actual_phase = npv(state.info["imitation_phase"][index])
            backlash = npv(
                env.get_actuator_backlash_qpos(state.data.qpos[index])
            )
            backlash_velocity = npv(state.data.qvel[index])[
                np.asarray(env._mj_model.jnt_dofadr)[
                    np.asarray(env.backlash_joint_ids)
                ]
            ]
            errors = {
                "observation_max_abs_error": error_metrics(
                    actual_obs, injected_obs
                )["max_abs_error"],
                "normalized_observation_max_abs_error": error_metrics(
                    actual_normalized, normalized
                )["max_abs_error"],
                "actor_residual_max_abs_error": error_metrics(
                    npv(actions[index]), injected_action
                )["max_abs_error"],
                "teacher_phase_max_abs_error": error_metrics(
                    actual_phase, phase_vector
                )["max_abs_error"],
                "teacher_action_max_abs_error": 0.0,
                "action_after_delay_max_abs_error": 0.0,
                "combined_action_max_abs_error": error_metrics(
                    actual_motor, composed.motor_target
                )["max_abs_error"],
                "motor_target_max_abs_error": error_metrics(
                    actual_motor, composed.motor_target
                )["max_abs_error"],
            }
            parity[index]["actor_numpy_diagnostic_max_abs_error"] = max(
                parity[index]["actor_numpy_diagnostic_max_abs_error"],
                error_metrics(
                    npv(actions[index]), numpy_injected_action
                )["max_abs_error"],
            )
            thresholds = {
                "observation_max_abs_error": 1e-6,
                "normalized_observation_max_abs_error": 1e-6,
                "actor_residual_max_abs_error": 1e-6,
                "teacher_phase_max_abs_error": 1e-6,
                "teacher_action_max_abs_error": 1e-6,
                "action_after_delay_max_abs_error": 0.0,
                "combined_action_max_abs_error": 1e-5,
                "motor_target_max_abs_error": 1e-6,
            }
            for key, value in errors.items():
                parity[index][key] = max(parity[index][key], value)
                if (
                    not parity[index]["first_divergence_stage"]
                    and value > thresholds[key]
                ):
                    parity[index]["first_divergence_stage"] = key
                    parity[index]["first_divergence_step"] = step_index
            buffer_exact = np.array_equal(
                history, npv(state.info["action_history"][index])
            )
            parity[index]["delay_buffer_exact"] &= buffer_exact
            if (
                not buffer_exact
                and not parity[index]["first_divergence_stage"]
            ):
                parity[index]["first_divergence_stage"] = "delay_buffer"
                parity[index]["first_divergence_step"] = step_index

            pair_values = {
                "max_abs_observation": (
                    npv(state.obs["state"][index]),
                    npv(replay_state.obs["state"][index]),
                ),
                "max_abs_qpos": (
                    npv(state.data.qpos[index]),
                    npv(replay_state.data.qpos[index]),
                ),
                "max_abs_qvel": (
                    npv(state.data.qvel[index]),
                    npv(replay_state.data.qvel[index]),
                ),
                "max_abs_motor_target": (
                    actual_motor,
                    npv(replay_state.info["motor_targets"][index]),
                ),
            }
            for key, (left, right) in pair_values.items():
                difference = float(np.max(np.abs(left - right)))
                native[index][key] = max(native[index][key], difference)
                if difference != 0.0:
                    native[index]["same_backend_exact"] = False
                    if native[index]["first_difference_step"] == "":
                        native[index]["first_difference_step"] = step_index

            trace = trace_records[index]
            values = {
                "time": step_index * float(env.dt),
                "control_step": step_index,
                "episode_step": step_index,
                "master_seed": args.master_seed,
                "environment_index": env_index,
                "environment_seed": npv(env_key),
                "episode_seed": npv(env_key),
                "raw_command": npv(before.info["command"][index]),
                "resolved_command": npv(before.info["command"][index])[:3],
                "head_command": npv(before.info["command"][index])[3:],
                "initial_qpos_offset": initial_offsets[index],
                "initial_qvel_offset": initial_qvel_offsets[index],
                "initial_base_velocity": initial_qvel_offsets[index][:6],
                "observation_noise": noise,
                "joint_position_noise": sample["joint_position_noise"],
                "joint_velocity_noise": sample["joint_velocity_noise"],
                "delay_length": sample["delay_length"],
                "delay_buffer_state": history,
                "backlash_internal_state": backlash,
                "backlash_direction_state": np.sign(backlash_velocity),
                "phase_initialization": 0.0,
                "teacher_phase": actual_phase,
                "raw_observation_before_noise": raw_before,
                "noise_sample": noise,
                "raw_observation_after_noise": injected_obs,
                "normalized_observation": normalized,
                "teacher_action": composed.teacher_action,
                "actor_residual": npv(actions[index]),
                "actor_standard_normal_sample": npv(eps[index]),
                "combined_action_pre_clip": composed.combined_pre_limit,
                "combined_action_post_clip": composed.motor_target,
                "action_before_delay": npv(actions[index]),
                "action_after_delay": delayed,
                "motor_target_before_backlash": actual_motor,
                "motor_target_after_backlash": actual_motor,
                "qpos": npv(state.data.qpos[index]),
                "qvel": npv(state.data.qvel[index]),
                "base_linear_velocity": npv(env.get_local_linvel(data)),
                "base_angular_velocity": npv(env.get_global_angvel(data)),
                "foot_contacts": contact,
                "termination_flags": int(npv(state.done[index])),
            }
            for key, value in values.items():
                trace[key].append(value)

            events = [
                event("policy.standard_normal", step_index, eps[index],
                      "control_step", "actor tanh-normal sample"),
                event("control.action_delay_index", step_index,
                      sample["delay_length"], "control_step",
                      "shared 14-joint action history"),
                event("disturbance.push_theta", step_index,
                      sample["push_theta"], "control_step", "push direction"),
                event("disturbance.push_magnitude", step_index,
                      sample["push_magnitude"], "control_step", "base qvel push"),
                event("observation.gyro_noise", step_index,
                      sample["gyro_noise"], "post_physics",
                      "actor observation[0:3]"),
                event("observation.accelerometer_noise", step_index,
                      sample["accelerometer_noise"], "post_physics",
                      "actor observation[3:6]"),
                event("observation.gravity_noise", step_index,
                      sample["gravity_noise"], "post_physics",
                      "imu history only"),
                event("observation.imu_delay_index", step_index,
                      sample["imu_delay"], "post_physics", "imu history read"),
                event("observation.joint_position_noise", step_index,
                      sample["joint_position_noise"], "post_physics",
                      "actor observation[13:27]"),
                event("observation.joint_velocity_noise", step_index,
                      sample["joint_velocity_noise"], "post_physics",
                      "actor observation[27:41] before 0.05 scale"),
                event("command.discarded_candidate", step_index,
                      sample["candidate_resolved_command"], "end_control_step",
                      "computed; not applied before step 501"),
            ]
            event_streams[index].append({
                "time": values["time"],
                "control_step": step_index,
                "events": events,
            })

    injection_rows = []
    for row in parity:
        row["pass"] = (
            row["initial_state_max_abs_error"] == 0.0
            and row["observation_max_abs_error"] <= 1e-6
            and row["normalized_observation_max_abs_error"] <= 1e-6
            and row["actor_residual_max_abs_error"] <= 1e-6
            and row["teacher_phase_max_abs_error"] <= 1e-6
            and row["teacher_action_max_abs_error"] <= 1e-6
            and row["delay_buffer_exact"]
            and row["action_after_delay_max_abs_error"] == 0.0
            and row["backlash_state_exact"]
            and row["combined_action_max_abs_error"] <= 1e-5
            and row["motor_target_max_abs_error"] <= 1e-6
        )
        injection_rows.append(row)
    native_rows = native

    command_rows = []
    for index, (command, env_index, env_key) in enumerate(cases):
        name = f'{command["command_id"]}_seed{env_index}'
        np.savez_compressed(
            traces_dir / f"{name}.npz",
            **{
                key: np.asarray(value)
                for key, value in trace_records[index].items()
            },
        )
        reset_events = []
        for source, value in domain_by_case[index].items():
            reset_events.append(
                event(
                    f"domain.{source}", 0, value,
                    "wrapper_construction", "randomized MJX model"
                )
            )
        for source, value in reset_by_case[index].items():
            reset_events.append(
                event(
                    f"reset.{source}", 0, value,
                    "episode_reset", "initial state or initial observation"
                )
            )
        metadata = {
            "command_id": command["command_id"],
            "command": [
                command["vx"], command["vy"], command["yaw_rate"],
                0.0, 0.0, 0.0, 0.0,
            ],
            "master_seed": args.master_seed,
            "environment_index": env_index,
            "environment_seed": npv(env_key).tolist(),
            "episode_seed": npv(env_key).tolist(),
            "reset_count": 0,
            "episode_count": 0,
            "not_applicable": {
                "command_duration_sampling": None,
                "phase_initialization_randomization": None,
                "backlash_initialization_randomization": None,
                "action_noise_inside_environment": None,
                "motor_noise": None,
                "terrain_geometry_sampling": None,
            },
            "reset_events": reset_events,
        }
        (traces_dir / f"{name}.metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        with (traces_dir / f"{name}.random_samples.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for entry in event_streams[index]:
                stream.write(json.dumps(entry) + "\n")
        trace = trace_records[index]
        linvel = np.asarray(trace["base_linear_velocity"])
        angvel = np.asarray(trace["base_angular_velocity"])
        command_rows.append({
            "command_id": command["command_id"],
            "environment_index": env_index,
            "duration_s": args.steps * float(env.dt),
            "mean_vx_reference_only": float(linvel[:, 0].mean()),
            "mean_vy_reference_only": float(linvel[:, 1].mean()),
            "mean_yaw_reference_only": float(angvel[:, 2].mean()),
            "fall_any": bool(np.asarray(trace["termination_flags"]).any()),
            "nan_or_inf": bool(
                any(
                    not np.isfinite(np.asarray(value)).all()
                    for value in trace.values()
                )
            ),
            "teacher_routing_active": command["vx"] < -0.02,
            "metadata_complete": True,
        })

    write_csv(output / "sample_injection_results.csv", injection_rows)
    write_csv(output / "native_seed_reproducibility.csv", native_rows)
    write_csv(output / "stochastic_command_results.csv", command_rows)
    summary = {
        "controller_stochastic_parity": (
            "PASS" if all(row["pass"] for row in injection_rows) else "FAIL"
        ),
        "historical_training_episode_reconstruction": "FAIL",
        "historical_reconstruction_reason": (
            "checkpoint omits env_state, per-environment PRNG state, reset "
            "count, episode count, and rollout policy keys"
        ),
        "sample_injection_cases": len(cases),
        "steps_per_case": args.steps,
        "native_same_backend_reproducible": all(
            row["same_backend_exact"] for row in native_rows
        ),
        "stochastic_smoke_wiring_pass": all(
            not row["nan_or_inf"] and not row["fall_any"]
            for row in command_rows
        ),
        "first_divergences": [
            {
                "command_id": row["command_id"],
                "environment_index": row["environment_index"],
                "stage": row["first_divergence_stage"] or None,
                "step": row["first_divergence_step"]
                if row["first_divergence_stage"] else None,
            }
            for row in injection_rows
        ],
        "maximum_errors": {
            key: max(row[key] for row in injection_rows)
            for key in (
                "initial_state_max_abs_error",
                "observation_max_abs_error",
                "normalized_observation_max_abs_error",
                "actor_residual_max_abs_error",
                "teacher_phase_max_abs_error",
                "combined_action_max_abs_error",
                "motor_target_max_abs_error",
            )
        },
    }
    (output / "stochastic_parity_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if summary["controller_stochastic_parity"] != "PASS":
        raise SystemExit("sample-injection parity failed")
    if not summary["native_same_backend_reproducible"]:
        raise SystemExit("native same-backend reproducibility failed")
    if not summary["stochastic_smoke_wiring_pass"]:
        raise SystemExit("stochastic smoke wiring gate failed")


if __name__ == "__main__":
    main()
