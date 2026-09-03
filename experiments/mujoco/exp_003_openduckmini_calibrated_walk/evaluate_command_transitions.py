"""Evaluate command changes without resetting the calibrated MuJoCo robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from evaluate_official_policy import (
    ACTION_SCALE,
    CONTROL_DT,
    DECIMATION,
    MAX_MOTOR_VELOCITY,
    OfficialPolicyEvaluator,
)


DEFAULT_SEQUENCE = (
    ("stand_0", (0.0, 0.0, 0.0), 3.0),
    ("forward", (0.1, 0.0, 0.0), 6.0),
    ("forward_to_reverse", (-0.1, 0.0, 0.0), 6.0),
    ("reverse_to_left_turn", (-0.07, 0.0, 0.3), 6.0),
    ("reverse_left_to_right_turn", (-0.07, 0.0, -0.3), 6.0),
    ("stand_1", (0.0, 0.0, 0.0), 3.0),
    ("left", (0.0, 0.1, 0.0), 6.0),
    ("left_to_right", (0.0, -0.1, 0.0), 6.0),
    ("stand_2", (0.0, 0.0, 0.0), 3.0),
    ("yaw_left", (0.0, 0.0, 0.6), 6.0),
    ("yaw_left_to_right", (0.0, 0.0, -0.6), 6.0),
    ("stand_3", (0.0, 0.0, 0.0), 3.0),
)


def parse_args() -> argparse.Namespace:
    experiment = Path(__file__).resolve().parent
    workspace = experiment.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=workspace
        / ".openduck_playground_source_review"
        / "playground"
        / "open_duck_mini_v2"
        / "xmls"
        / "scene_flat_terrain_backlash_calibrated.xml",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=experiment / "artifacts" / "calibrated_hybrid_policy_v22.onnx",
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=workspace
        / ".openduck_playground_source_review"
        / "playground"
        / "open_duck_mini_v2"
        / "data"
        / "polynomial_coefficients_calibrated.pkl",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--initial-joint-noise", type=float, default=0.03)
    parser.add_argument("--initial-base-speed", type=float, default=0.1)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument(
        "--backward-left-turn-gait",
        type=Path,
        default=workspace
        / ".openduck_runtime_source_review"
        / "optimized_backward_left_turn_gait.json",
    )
    parser.add_argument(
        "--backward-right-turn-gait",
        type=Path,
        default=workspace
        / ".openduck_runtime_source_review"
        / "optimized_backward_right_turn_gait.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=experiment / "artifacts" / "command_transition_acceptance.json",
    )
    return parser.parse_args()


def run_sequence(
    evaluator: OfficialPolicyEvaluator,
    seed: int,
    initial_joint_noise: float,
    initial_base_speed: float,
    warmup_seconds: float,
) -> dict:
    rng = np.random.default_rng(seed)
    model = evaluator.model
    data = mujoco.MjData(model)
    home = model.keyframe("home")
    data.qpos[:] = home.qpos
    joint_ranges = model.jnt_range[evaluator.actuator_joint_ids]
    noise = rng.uniform(
        -initial_joint_noise, initial_joint_noise, size=model.nu
    )
    data.qpos[evaluator.actuator_qpos_addr] = np.clip(
        data.qpos[evaluator.actuator_qpos_addr] + noise,
        joint_ranges[:, 0] + 0.005,
        joint_ranges[:, 1] - 0.005,
    )
    push_angle = rng.uniform(-np.pi, np.pi)
    push_magnitude = rng.uniform(0.0, initial_base_speed)
    data.qvel[:2] = push_magnitude * np.array(
        [np.cos(push_angle), np.sin(push_angle)]
    )
    data.ctrl[:] = data.qpos[evaluator.actuator_qpos_addr]
    mujoco.mj_forward(model, data)

    default = np.asarray(home.ctrl, dtype=np.float64).copy()
    targets = default.copy()
    previous_targets = default.copy()
    action_history = [np.zeros(model.nu, dtype=np.float32) for _ in range(3)]
    previous_position = data.xpos[evaluator.trunk_body_id].copy()
    phase_index = 0.0
    head_peak = 0.0
    limit_violations = 0
    fell = False
    segments = []

    for name, command_xyz, duration in DEFAULT_SEQUENCE:
        velocities = []
        yaw_rates = []
        uprights = []
        heights = []
        contacts = []
        steps = int(round(duration / CONTROL_DT))
        warmup_steps = int(round(warmup_seconds / CONTROL_DT))
        for control_step in range(steps):
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            backward_scales, backward_rate = evaluator.backward_parameters(
                command_xyz[2]
            )
            phase_delta = backward_rate if command_xyz[0] < -0.02 else 1.0
            phase_index = (phase_index + phase_delta) % evaluator.phase_steps
            phase = phase_index / evaluator.phase_steps * 2.0 * np.pi
            policy_command = np.asarray(command_xyz, dtype=np.float64).copy()
            if policy_command[2] > 0.0:
                policy_command[1] -= 0.06
            command = np.asarray(
                [*policy_command, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
            )
            observation = evaluator._observation(
                data,
                command,
                default,
                targets,
                action_history,
                phase,
            )
            action = evaluator.session.run(
                None, {"obs": observation[None, :]}
            )[0][0].astype(np.float32)
            action_history = [
                action.copy(),
                action_history[0].copy(),
                action_history[1].copy(),
            ]
            bounded = np.clip(action, -1.0, 1.0)
            bounded[5:9] = 0.0
            positive_span = 0.9 * (joint_ranges[:, 1] - default)
            negative_span = 0.9 * (default - joint_ranges[:, 0])
            directional_span = np.where(
                bounded >= 0.0, positive_span, negative_span
            )
            base_span = np.minimum(ACTION_SCALE, directional_span)
            magnitude = np.abs(bounded)
            target_magnitude = (
                base_span * magnitude
                + (directional_span - base_span) * magnitude**5
            )
            targets = default + np.sign(bounded) * target_magnitude
            if command_xyz[0] < -0.02:
                targets = evaluator._backward_feedforward(
                    phase_index,
                    default,
                    joint_ranges,
                    bounded,
                    gait_scales=backward_scales,
                )
            maximum_delta = MAX_MOTOR_VELOCITY * CONTROL_DT
            targets = np.clip(
                targets,
                previous_targets - maximum_delta,
                previous_targets + maximum_delta,
            )
            outside = (targets < model.actuator_ctrlrange[:, 0]) | (
                targets > model.actuator_ctrlrange[:, 1]
            )
            limit_violations += int(np.count_nonzero(outside))
            targets = np.clip(
                targets,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            targets[5:9] = 0.0
            head_peak = max(head_peak, float(np.max(np.abs(targets[5:9]))))
            previous_targets = targets.copy()
            data.ctrl[:] = targets

            position = data.xpos[evaluator.trunk_body_id].copy()
            rotation = data.xmat[evaluator.trunk_body_id].reshape(3, 3)
            local_velocity = (
                rotation.T @ ((position - previous_position) / CONTROL_DT)
            )
            previous_position = position
            upright = float(rotation[2, 2])
            if control_step >= warmup_steps:
                velocities.append(local_velocity)
                yaw_rates.append(
                    float(evaluator._sensor(data, "global_angvel")[2])
                )
                uprights.append(upright)
                heights.append(float(position[2]))
                contacts.append(evaluator._feet_contacts(data))
            if position[2] < 0.08 or upright < 0.25:
                fell = True
                break

        velocity = np.mean(np.asarray(velocities), axis=0)
        yaw_rate = float(np.mean(yaw_rates))
        contact_array = np.asarray(contacts)
        segments.append(
            {
                "name": name,
                "command": list(command_xyz),
                "completed": not fell,
                "mean_velocity_xyz": velocity.tolist(),
                "mean_yaw_rate": yaw_rate,
                "minimum_upright": float(np.min(uprights)),
                "minimum_height": float(np.min(heights)),
                "single_support_rate": float(
                    np.logical_xor(
                        contact_array[:, 0], contact_array[:, 1]
                    ).mean()
                ),
                "flight_rate": float(
                    (contact_array.sum(axis=1) == 0).mean()
                ),
            }
        )
        if fell:
            break

    return {
        "seed": seed,
        "fell": fell,
        "joint_target_limit_violations": limit_violations,
        "head_target_peak": head_peak,
        "segments": segments,
    }


def segment_passed(segment: dict) -> dict:
    command = np.asarray(segment["command"])
    velocity = np.asarray(segment["mean_velocity_xyz"])
    yaw_rate = float(segment["mean_yaw_rate"])
    moving_linear = np.max(np.abs(command[:2])) > 0.0
    moving_yaw = abs(command[2]) > 0.0
    if moving_linear:
        axis = int(np.argmax(np.abs(command[:2])))
        primary_error = abs(velocity[axis] - command[axis])
        orthogonal = abs(velocity[1 - axis])
    else:
        primary_error = 0.0
        orthogonal = float(np.linalg.norm(velocity[:2]))
    checks = {
        "completed": segment["completed"],
        "upright": segment["minimum_upright"] >= 0.85,
        "height": segment["minimum_height"] >= 0.12,
        "primary_velocity": (not moving_linear) or primary_error <= 0.06,
        "orthogonal_velocity": orthogonal <= (0.05 if moving_linear else 0.04),
        "yaw_rate": (not moving_yaw) or abs(yaw_rate - command[2]) <= 0.25,
        "stop_yaw": moving_yaw or abs(yaw_rate) <= 0.20,
        "single_support": (
            not (moving_linear or moving_yaw)
            or segment["single_support_rate"] >= 0.08
        ),
        "flight": segment["flight_rate"] <= 0.15,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "name": segment["name"],
        "passed": all(checks.values()),
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    evaluator = OfficialPolicyEvaluator(
        args.scene, args.policy, args.reference_data
    )
    evaluator.load_backward_turn_profile(1, args.backward_left_turn_gait)
    evaluator.load_backward_turn_profile(-1, args.backward_right_turn_gait)
    evaluator.backward_turn_minimum_yaw = 0.05
    evaluator.backward_turn_minimum_blend = 0.75
    evaluator.backward_turn_maximum_blend = 0.87
    episodes = [
        run_sequence(
            evaluator,
            seed,
            args.initial_joint_noise,
            args.initial_base_speed,
            args.warmup_seconds,
        )
        for seed in range(args.episodes)
    ]
    checks = [
        {
            "seed": episode["seed"],
            "passed": (
                not episode["fell"]
                and episode["joint_target_limit_violations"] == 0
                and episode["head_target_peak"] == 0.0
                and len(episode["segments"]) == len(DEFAULT_SEQUENCE)
                and all(
                    segment_passed(segment)["passed"]
                    for segment in episode["segments"]
                )
            ),
            "segments": [
                segment_passed(segment) for segment in episode["segments"]
            ],
        }
        for episode in episodes
    ]
    payload = {
        "policy": str(args.policy.resolve()),
        "sequence": [
            {"name": name, "command": list(command), "seconds": seconds}
            for name, command, seconds in DEFAULT_SEQUENCE
        ],
        "episodes": episodes,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
