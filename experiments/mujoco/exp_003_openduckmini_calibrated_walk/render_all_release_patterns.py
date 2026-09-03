"""Render every command pattern used to qualify calibrated hybrid policy v22."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_official_policy import (
    ACTION_SCALE,
    CONTROL_DT,
    DECIMATION,
    MAX_MOTOR_VELOCITY,
    OfficialPolicyEvaluator,
)


BASIC_PATTERNS = (
    ("basic_01_stop", "STOP", (0.0, 0.0, 0.0)),
    ("basic_02_forward", "FORWARD", (0.10, 0.0, 0.0)),
    ("basic_03_backward", "BACKWARD", (-0.10, 0.0, 0.0)),
    ("basic_04_left", "LEFT", (0.0, 0.10, 0.0)),
    ("basic_05_right", "RIGHT", (0.0, -0.10, 0.0)),
    ("basic_06_yaw_left", "YAW LEFT", (0.0, 0.0, 0.60)),
    ("basic_07_yaw_right", "YAW RIGHT", (0.0, 0.0, -0.60)),
)

COMPOUND_PATTERNS = (
    ("compound_01_forward_left", "FORWARD + LEFT", (0.07, 0.05, 0.0)),
    ("compound_02_forward_right", "FORWARD + RIGHT", (0.07, -0.05, 0.0)),
    (
        "compound_03_forward_yaw_left",
        "FORWARD + YAW LEFT",
        (0.07, 0.0, 0.30),
    ),
    (
        "compound_04_forward_yaw_right",
        "FORWARD + YAW RIGHT",
        (0.07, 0.0, -0.30),
    ),
    (
        "compound_05_forward_left_yaw_left",
        "FORWARD + LEFT + YAW LEFT",
        (0.07, 0.04, 0.30),
    ),
    (
        "compound_06_forward_right_yaw_right",
        "FORWARD + RIGHT + YAW RIGHT",
        (0.07, -0.04, -0.30),
    ),
    (
        "compound_07_backward_yaw_left_01",
        "BACKWARD + YAW LEFT 0.1",
        (-0.07, 0.0, 0.10),
    ),
    (
        "compound_08_backward_yaw_left_02",
        "BACKWARD + YAW LEFT 0.2",
        (-0.07, 0.0, 0.20),
    ),
    (
        "compound_09_backward_yaw_left_03",
        "BACKWARD + YAW LEFT 0.3",
        (-0.07, 0.0, 0.30),
    ),
    (
        "compound_10_backward_yaw_right_01",
        "BACKWARD + YAW RIGHT 0.1",
        (-0.07, 0.0, -0.10),
    ),
    (
        "compound_11_backward_yaw_right_02",
        "BACKWARD + YAW RIGHT 0.2",
        (-0.07, 0.0, -0.20),
    ),
    (
        "compound_12_backward_yaw_right_03",
        "BACKWARD + YAW RIGHT 0.3",
        (-0.07, 0.0, -0.30),
    ),
)


def parse_args() -> argparse.Namespace:
    experiment = Path(__file__).resolve().parent
    workspace = experiment.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--only",
        choices=("basic", "compound", "backward", "all"),
        default="all",
    )
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
        "--output-dir",
        type=Path,
        default=experiment / "artifacts" / "videos_v22",
    )
    parser.add_argument(
        "--combined-name",
        default="all_19_release_patterns_v22.mp4",
    )
    parser.add_argument("--backward-residual-scale", type=float, default=0.0)
    parser.add_argument(
        "--backward-turn-leg-residual-floor", type=float, default=0.10
    )
    parser.add_argument(
        "--backward-turn-maximum-blend", type=float, default=1.0
    )
    parser.add_argument("--learned-reverse-policy", action="store_true")
    return parser.parse_args()


def add_label(
    frame: np.ndarray,
    label: str,
    command: tuple[float, float, float],
    pattern_index: int,
    pattern_count: int,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 60), fill=(0, 0, 0, 175))
    font = ImageFont.load_default(size=18)
    small_font = ImageFont.load_default(size=14)
    draw.text(
        (14, 8),
        f"{pattern_index:02d}/{pattern_count:02d}  {label}",
        font=font,
        fill=(255, 255, 255, 255),
    )
    draw.text(
        (14, 35),
        f"command  vx={command[0]:+.2f}  vy={command[1]:+.2f}  yaw={command[2]:+.2f}",
        font=small_font,
        fill=(215, 230, 255, 255),
    )
    return np.asarray(image)


def render_pattern(
    evaluator: OfficialPolicyEvaluator,
    renderer: mujoco.Renderer,
    camera: mujoco.MjvCamera,
    writer,
    combined_writer,
    label: str,
    command_xyz: tuple[float, float, float],
    seconds: float,
    fps: int,
    pattern_index: int,
    pattern_count: int,
) -> tuple[int, bool]:
    model = evaluator.model
    data = mujoco.MjData(model)
    home = model.keyframe("home")
    data.qpos[:] = home.qpos
    data.ctrl[:] = data.qpos[evaluator.actuator_qpos_addr]
    mujoco.mj_forward(model, data)

    joint_ranges = model.jnt_range[evaluator.actuator_joint_ids]
    default = np.asarray(home.ctrl, dtype=np.float64).copy()
    targets = default.copy()
    previous_targets = default.copy()
    action_history = [np.zeros(model.nu, dtype=np.float32) for _ in range(3)]
    policy_command = list(command_xyz)
    if evaluator.calibrated_hardware and policy_command[2] > 0.0:
        policy_command[1] -= 0.06
    command = np.asarray([*policy_command, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    phase_index = 0.0

    control_hz = round(1.0 / CONTROL_DT)
    if control_hz % fps:
        raise ValueError(f"fps must divide the {control_hz} Hz controller rate")
    render_every = control_hz // fps
    target_control_steps = int(round(seconds / CONTROL_DT))
    frame_count = 0
    fell = False

    for sim_step in range(target_control_steps * DECIMATION):
        mujoco.mj_step(model, data)
        if (sim_step + 1) % DECIMATION:
            continue

        if command_xyz[0] < 0.0 and evaluator.calibrated_hardware:
            _, _, phase_delta = evaluator.backward_parameters(command_xyz[2])
        else:
            phase_delta = 1.0
        phase_index = (phase_index + phase_delta) % evaluator.phase_steps
        phase = phase_index / evaluator.phase_steps * 2.0 * np.pi
        observation = evaluator._observation(
            data, command, default, targets, action_history, phase
        )
        action = evaluator.session.run(None, {"obs": observation[None, :]})[0][0]
        action = np.asarray(action, dtype=np.float32)
        action_history = [
            action.copy(),
            action_history[0].copy(),
            action_history[1].copy(),
        ]

        if evaluator.calibrated_hardware:
            action_for_control = action.copy()
            bounded_action = np.clip(action_for_control, -1.0, 1.0)
            positive_span = 0.9 * (joint_ranges[:, 1] - default)
            negative_span = 0.9 * (default - joint_ranges[:, 0])
            directional_span = np.where(
                bounded_action >= 0.0, positive_span, negative_span
            )
            base_span = np.minimum(ACTION_SCALE, directional_span)
            magnitude = np.abs(bounded_action)
            target_magnitude = (
                base_span * magnitude
                + (directional_span - base_span) * magnitude**5
            )
            targets = default + np.sign(bounded_action) * target_magnitude

            if (
                evaluator.use_backward_feedforward
                and command_xyz[0] < -0.02
            ):
                backward_scales, backward_biases, _ = (
                    evaluator.backward_parameters(command_xyz[2])
                )
                targets = evaluator._backward_feedforward(
                    phase_index,
                    default,
                    joint_ranges,
                    bounded_action,
                    gait_scales=backward_scales,
                    gait_biases=backward_biases,
                    leg_residual_factor=max(
                        evaluator.backward_turn_leg_residual_floor,
                        1.0
                        - float(
                            np.clip(abs(command_xyz[2]) / 0.2, 0.0, 1.0)
                        ),
                    ),
                    head_residual_factor=(
                        1.0
                        - 0.5
                        * float(
                            np.clip(abs(command_xyz[2]) / 0.2, 0.0, 1.0)
                        )
                    ),
                )

            maximum_delta = MAX_MOTOR_VELOCITY * CONTROL_DT
            targets = np.clip(
                targets,
                previous_targets - maximum_delta,
                previous_targets + maximum_delta,
            )
            targets = np.clip(
                targets,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            targets[6] = min(
                targets[6],
                0.458 - 0.984 * targets[5],
            )
        else:
            # Original v2 runtime contract: direct affine action mapping.
            targets = default + action * ACTION_SCALE
        previous_targets = targets.copy()
        data.ctrl[:] = targets

        control_step = (sim_step + 1) // DECIMATION
        if control_step % render_every == 0:
            trunk = data.xpos[evaluator.trunk_body_id]
            camera.lookat[:] = (float(trunk[0]), float(trunk[1]), 0.13)
            renderer.update_scene(data, camera=camera)
            frame = add_label(
                renderer.render(),
                label,
                command_xyz,
                pattern_index,
                pattern_count,
            )
            writer.append_data(frame)
            combined_writer.append_data(frame)
            frame_count += 1

        trunk = data.xpos[evaluator.trunk_body_id]
        upright = data.xmat[evaluator.trunk_body_id].reshape(3, 3)[2, 2]
        if trunk[2] < 0.08 or upright < 0.25:
            fell = True
            break

    return frame_count, fell


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = OfficialPolicyEvaluator(args.scene, args.policy, args.reference_data)
    evaluator.use_backward_feedforward = not args.learned_reverse_policy
    evaluator.backward_residual_scale = args.backward_residual_scale
    if evaluator.calibrated_hardware:
        evaluator.load_backward_turn_profile(1, args.backward_left_turn_gait)
        evaluator.load_backward_turn_profile(-1, args.backward_right_turn_gait)
        evaluator.backward_turn_minimum_yaw = 0.1
        evaluator.backward_turn_minimum_blend = 0.0
        evaluator.backward_turn_maximum_blend = (
            args.backward_turn_maximum_blend
        )
        evaluator.backward_turn_leg_residual_floor = (
            args.backward_turn_leg_residual_floor
        )

    if args.only == "basic":
        patterns = BASIC_PATTERNS
    elif args.only == "compound":
        patterns = COMPOUND_PATTERNS
    elif args.only == "backward":
        patterns = [BASIC_PATTERNS[2], *COMPOUND_PATTERNS[6:]]
    else:
        patterns = BASIC_PATTERNS + COMPOUND_PATTERNS

    renderer = mujoco.Renderer(
        evaluator.model, height=args.height, width=args.width
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.15
    camera.azimuth = 135.0
    camera.elevation = -18.0

    combined_path = args.output_dir / args.combined_name
    combined_writer = imageio.get_writer(
        combined_path, fps=args.fps, codec="libx264", quality=8
    )
    try:
        for index, (slug, label, command) in enumerate(patterns, start=1):
            output_path = args.output_dir / f"{slug}.mp4"
            print(f"[{index:02d}/{len(patterns):02d}] {label}: {output_path}")
            with imageio.get_writer(
                output_path, fps=args.fps, codec="libx264", quality=8
            ) as writer:
                frame_count, fell = render_pattern(
                    evaluator,
                    renderer,
                    camera,
                    writer,
                    combined_writer,
                    label,
                    command,
                    args.seconds,
                    args.fps,
                    index,
                    len(patterns),
                )
            print(f"  frames={frame_count} fell={fell}")
    finally:
        combined_writer.close()
        renderer.close()

    print(f"Combined video: {combined_path.resolve()}")


if __name__ == "__main__":
    main()
