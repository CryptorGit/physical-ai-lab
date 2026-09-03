#!/usr/bin/env python3
"""Export deterministic v59 MJX traces and prove controller-path parity.

Run this with the historical WSL training virtualenv.  It performs no updates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np
import onnxruntime as ort
from brax.training.agents.ppo import checkpoint, networks as ppo_networks
from brax.training.acme import running_statistics
from mujoco_playground._src import mjx_env
from mujoco_playground.config import locomotion_params

from playground.open_duck_mini_v2 import joystick

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v59_parity_common import COMMANDS, compose_motor_target, error_metrics, numpy_actor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_numpy(value):
    return np.asarray(jax.device_get(value))


def make_env():
    config = joystick.default_config()
    config.noise_config.level = 0.0
    config.noise_config.action_min_delay = 0
    config.noise_config.action_max_delay = 1
    config.noise_config.imu_min_delay = 0
    config.noise_config.imu_max_delay = 1
    config.push_config.enable = False
    return joystick.Joystick(
        task="flat_terrain_backlash_calibrated", config=config
    )


def fixed_state(env, seed, command):
    state = env.reset(jax.random.PRNGKey(seed))
    data = mjx_env.init(
        env.mjx_model,
        qpos=env._init_q,
        qvel=jp.zeros(env.mjx_model.nv),
        ctrl=env._default_actuator,
    )
    info = dict(state.info)
    info["command"] = jp.asarray(command, dtype=jp.float32)
    info["step"] = 0
    info["last_act"] = jp.zeros(env.action_size)
    info["last_last_act"] = jp.zeros(env.action_size)
    info["last_last_last_act"] = jp.zeros(env.action_size)
    info["motor_targets"] = env._default_actuator
    info["action_history"] = jp.zeros(env.action_size)
    info["imitation_i"] = jp.zeros(())
    info["imitation_phase"] = jp.zeros(2)
    info["push_step"] = 0
    info["push"] = jp.zeros(2)
    info["last_contact"] = jp.zeros(2, dtype=bool)
    info["feet_air_time"] = jp.zeros(2)
    info["swing_peak"] = jp.zeros(2)
    ref = env.PRM.get_reference_motion(command[0], command[1], command[2], 0)
    info["current_reference_motion"] = jp.where(
        command[0] < -0.02,
        env._get_optimized_backward_reference(jp.zeros(()), command[2]),
        ref,
    )
    contact = jp.array(
        [
            joystick.geoms_colliding(data, geom, env._floor_geom_id)
            for geom in env._feet_geom_id
        ]
    )
    obs = env._get_obs(data, info, contact)
    return state.replace(data=data, obs=obs, info=info, done=jp.zeros(()))


def stage_values(env, state, action):
    cmd = to_numpy(state.info["command"])
    _, _, backward_rate = env._get_backward_parameters(cmd[2])
    rate = backward_rate if cmd[0] < -0.02 else 1.0
    phase = (state.info["imitation_i"] + rate) % env.PRM.nb_steps_in_period
    reference = env._get_optimized_backward_reference(phase, cmd[2])
    result = compose_motor_target(
        to_numpy(action),
        command=cmd,
        default=to_numpy(env._default_actuator),
        lower=to_numpy(env._actuator_lowers),
        upper=to_numpy(env._actuator_uppers),
        action_scale=float(env._config.action_scale),
        previous_target=to_numpy(state.info["motor_targets"]),
        max_motor_velocity=to_numpy(env._config.max_motor_velocity),
        dt=float(env.dt),
        backward_reference=to_numpy(reference),
        backward_actuator_indices=to_numpy(env._backward_actuator_indices),
        backward_joint_indices=to_numpy(env._backward_joint_indices),
        backward_residual_scale=float(env._backward_residual_scale),
        coupled_slope=float(joystick.HEAD_COUPLED_REAR_SLOPE),
        coupled_intercept=float(joystick.HEAD_COUPLED_REAR_INTERCEPT),
    )
    phase_vector = np.array(
        [
            np.cos(float(phase) / env.PRM.nb_steps_in_period * 2 * np.pi),
            np.sin(float(phase) / env.PRM.nb_steps_in_period * 2 * np.pi),
        ],
        dtype=np.float32,
    )
    return phase, phase_vector, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=59)
    args = parser.parse_args()
    out = Path(args.output)
    traces_dir = out / "golden_traces"
    tables_dir = out / "comparison_tables"
    smoke_dir = out / "smoke_results"
    for directory in (traces_dir, tables_dir, smoke_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = make_env()
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
    policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
    step_fn = jax.jit(env.step)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    mean = to_numpy(params[0].mean["state"])
    std = to_numpy(params[0].std["state"])
    layer_dict = params[1]["params"]
    layers = [
        (to_numpy(layer_dict[f"hidden_{i}"]["kernel"]),
         to_numpy(layer_dict[f"hidden_{i}"]["bias"]))
        for i in range(4)
    ]

    actor_rows = []
    motor_rows = []
    smoke_rows = []
    global_actor_max = {"python": 0.0, "onnx": 0.0}
    global_motor_max = 0.0
    first_actor_failure = None
    first_motor_failure = None
    for command_spec in COMMANDS:
        command = np.array(
            [
                command_spec["vx"],
                command_spec["vy"],
                command_spec["yaw_rate"],
                0.0, 0.0, 0.0, 0.0,
            ],
            dtype=np.float32,
        )
        state = fixed_state(env, args.seed, command)
        records = {}
        for name in (
            "time", "raw_command", "resolved_command", "head_command",
            "teacher_mode", "teacher_phase", "teacher_action",
            "raw_observation", "normalized_observation", "actor_residual_raw",
            "actor_residual_scaled", "combined_action_pre_clip",
            "combined_action_post_clip", "delayed_action", "motor_target",
            "qpos", "qvel", "base_linear_velocity", "base_angular_velocity",
            "foot_contacts", "termination_flags",
        ):
            records[name] = []
        no_motion_steps = 0
        fell = False
        for index in range(args.steps):
            raw_obs = to_numpy(state.obs["state"]).astype(np.float32)
            normalized = (raw_obs - mean) / std
            jax_action, _ = policy(state.obs, jax.random.PRNGKey(0))
            jax_action = to_numpy(jax_action).astype(np.float32)
            python_action = numpy_actor(raw_obs, mean, std, layers).astype(np.float32)
            onnx_action = session.run(
                None, {input_name: raw_obs[None, :]}
            )[0][0].astype(np.float32)
            python_error = error_metrics(jax_action, python_action)
            onnx_error = error_metrics(jax_action, onnx_action)
            global_actor_max["python"] = max(
                global_actor_max["python"], python_error["max_abs_error"]
            )
            global_actor_max["onnx"] = max(
                global_actor_max["onnx"], onnx_error["max_abs_error"]
            )
            if first_actor_failure is None and (
                python_error["max_abs_error"] > 1e-6
                or onnx_error["max_abs_error"] > 1e-5
            ):
                first_actor_failure = {
                    "command_id": command_spec["command_id"], "step": index,
                    "python_max_abs": python_error["max_abs_error"],
                    "onnx_max_abs": onnx_error["max_abs_error"],
                }
            phase, phase_vector, composition = stage_values(
                env, state, jax_action
            )
            next_state = step_fn(state, jp.asarray(jax_action))
            actual_motor = to_numpy(next_state.info["motor_targets"])
            motor_error = error_metrics(composition.motor_target, actual_motor)
            global_motor_max = max(global_motor_max, motor_error["max_abs_error"])
            if first_motor_failure is None and motor_error["max_abs_error"] > 1e-6:
                first_motor_failure = {
                    "command_id": command_spec["command_id"], "step": index,
                    "max_abs_error": motor_error["max_abs_error"],
                }
            contacts = to_numpy(
                jp.array([
                    joystick.geoms_colliding(
                        state.data, geom, env._floor_geom_id
                    )
                    for geom in env._feet_geom_id
                ])
            )
            linvel = to_numpy(env.get_local_linvel(state.data))
            angvel = to_numpy(env.get_global_angvel(state.data))
            no_motion_steps += int(np.linalg.norm(linvel[:2]) < 0.01)
            fell = fell or bool(to_numpy(next_state.done))
            values = {
                "time": index * float(env.dt),
                "raw_command": command,
                "resolved_command": command[:3],
                "head_command": command[3:],
                "teacher_mode": int(composition.teacher_active),
                "teacher_phase": phase_vector,
                "teacher_action": composition.teacher_action,
                "raw_observation": raw_obs,
                "normalized_observation": normalized,
                "actor_residual_raw": jax_action,
                "actor_residual_scaled": composition.residual_scaled,
                "combined_action_pre_clip": composition.combined_pre_limit,
                "combined_action_post_clip": composition.motor_target,
                "delayed_action": jax_action,
                "motor_target": actual_motor,
                "qpos": to_numpy(state.data.qpos),
                "qvel": to_numpy(state.data.qvel),
                "base_linear_velocity": linvel,
                "base_angular_velocity": angvel,
                "foot_contacts": contacts,
                "termination_flags": int(to_numpy(next_state.done)),
            }
            for key, value in values.items():
                records[key].append(value)
            actor_rows.append({
                "command_id": command_spec["command_id"], "step": index,
                "python_max_abs_error": python_error["max_abs_error"],
                "onnx_max_abs_error": onnx_error["max_abs_error"],
                "onnx_rmse": onnx_error["rmse"],
            })
            motor_rows.append({
                "command_id": command_spec["command_id"], "step": index,
                "max_abs_error": motor_error["max_abs_error"],
            })
            state = next_state
        np.savez_compressed(
            traces_dir / f'{command_spec["command_id"]}.npz',
            **{key: np.asarray(value) for key, value in records.items()},
        )
        smoke_rows.append({
            **command_spec,
            "seed": args.seed,
            "duration_s": args.steps * float(env.dt),
            "teacher_routing_active": bool(command_spec["vx"] < -0.02),
            "fall": fell,
            "no_motion_duration_s": no_motion_steps * float(env.dt),
            "nan_or_inf": bool(
                any(not np.isfinite(np.asarray(v)).all()
                    for v in records.values())
            ),
        })

    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    write_csv(tables_dir / "actor_parity.csv", actor_rows)
    write_csv(tables_dir / "motor_target_parity.csv", motor_rows)
    write_csv(smoke_dir / "training_equivalent_5x1x2s.csv", smoke_rows)

    actor_pass = (
        global_actor_max["python"] <= 1e-6
        and global_actor_max["onnx"] <= 1e-5
    )
    motor_pass = global_motor_max <= 1e-6
    report = {
        "gate": "PASS" if actor_pass and motor_pass else "FAIL",
        "checkpoint": {
            "path": args.checkpoint,
            "tree_sha256": "4e522903cfb3edf8dacfc2f5dc5b9510746711360748440c54097483f0ac38f1",
            "step": 33423360,
        },
        "onnx": {
            "path": args.onnx,
            "sha256": sha256_file(Path(args.onnx)),
            "exported_from_checkpoint": args.checkpoint,
        },
        "scene": str(env._config.xml_path)
            if hasattr(env._config, "xml_path") else
            "scene_flat_terrain_backlash_calibrated.xml",
        "deterministic_settings": {
            "noise": 0.0, "action_delay_samples": [0],
            "imu_delay_samples": [0], "push": False,
            "initial_velocity": 0.0, "command_switching": False,
        },
        "actor_parity": {
            "pass": actor_pass, "threshold_python": 1e-6,
            "threshold_onnx": 1e-5, "max_abs": global_actor_max,
            "first_failure": first_actor_failure,
        },
        "teacher_routing_parity": {
            "pass": motor_pass,
            "reverse_rule": "vx < -0.02",
            "reverse_commands": ["C2_backward", "C4_backward_right_max"],
        },
        "action_composition_parity": {
            "pass": motor_pass,
            "independent_numpy_vs_jax_environment_max_abs_error":
                global_motor_max,
        },
        "motor_target_parity": {
            "pass": motor_pass, "threshold": 1e-6,
            "max_abs_error": global_motor_max,
            "first_failure": first_motor_failure,
        },
        "smoke": smoke_rows,
        "legacy_first_divergence": {
            "stage": "scene selection",
            "training": "scene_flat_terrain_backlash_calibrated.xml",
            "legacy": "scene_flat_terrain.xml",
            "controller_divergence": "teacher routing at vx < -0.02",
        },
    }
    (out / "parity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (smoke_dir / "metadata.json").write_text(
        json.dumps({
            "kind": "diagnostic_smoke_not_formal_evaluation",
            "commands": list(COMMANDS), "seed": args.seed,
            "steps": args.steps, "control_dt": float(env.dt),
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not actor_pass:
        raise SystemExit("Actor parity failed; closed-loop conclusions forbidden")
    if not motor_pass:
        raise SystemExit("Motor-target parity failed")


if __name__ == "__main__":
    main()
