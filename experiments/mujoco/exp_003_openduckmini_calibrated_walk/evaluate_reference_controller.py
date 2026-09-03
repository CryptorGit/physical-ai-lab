"""Evaluate the calibrated polynomial reference as direct actuator targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


EXPERIMENT = Path(__file__).resolve().parent
WORKSPACE = EXPERIMENT.parents[2]
PLAYGROUND = WORKSPACE / ".openduck_playground_source_review"
sys.path.insert(0, str(PLAYGROUND))

from playground.common.poly_reference_motion_numpy import PolyReferenceMotion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=PLAYGROUND
        / "playground/open_duck_mini_v2/xmls/"
        "scene_flat_terrain_backlash_calibrated.xml",
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=PLAYGROUND
        / "playground/open_duck_mini_v2/data/"
        "polynomial_coefficients_calibrated.pkl",
    )
    parser.add_argument("--vx", type=float, default=-0.1)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--max-motor-velocity", type=float, default=5.24)
    parser.add_argument(
        "--gait-json",
        type=Path,
        help="Optional optimized gait parameters produced by optimize_reference_scales.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    home = model.keyframe("home")
    data.qpos[:] = home.qpos
    data.ctrl[:] = home.ctrl
    mujoco.mj_forward(model, data)

    reference = PolyReferenceMotion(str(args.reference_data.resolve()))
    optimized = None
    if args.gait_json:
        optimized = json.loads(args.gait_json.resolve().read_text(encoding="utf-8"))
        scales = np.asarray(
            optimized["parameters"]["joint_amplitude_scales"], dtype=np.float64
        )
        phase_rate = float(optimized["parameters"]["phase_rate"])
        frames = np.asarray(
            [
                reference.get_reference_motion(args.vx, args.vy, args.yaw, index)
                for index in range(reference.nb_steps_in_period)
            ],
            dtype=np.float64,
        )
        leg_frames = np.concatenate([frames[:, :5], frames[:, 11:16]], axis=1)
        means = leg_frames.mean(axis=0)
        deviations = leg_frames - means
        leg_actuators = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
        home_ctrl = np.asarray(home.ctrl, dtype=np.float64).copy()
        lower = home_ctrl + 0.9 * (
            model.actuator_ctrlrange[:, 0] - home_ctrl
        )
        upper = home_ctrl + 0.9 * (
            model.actuator_ctrlrange[:, 1] - home_ctrl
        )
    floor_body = int(model.geom_bodyid[model.geom("floor").id])
    left_body = model.body("foot_assembly").id
    right_body = model.body("foot_assembly_2").id
    trunk_body = model.body("trunk_assembly").id
    start = data.xpos[trunk_body].copy()
    previous_position = start.copy()

    contacts = []
    heights = []
    uprights = []
    positions = []
    local_velocities = []
    local_angular_velocities = []
    previous_targets = np.asarray(home.ctrl, dtype=np.float64).copy()
    max_delta = args.max_motor_velocity * 0.02
    control_steps = int(round(args.seconds / 0.02))
    fell = False
    phase = 0.0

    for control_step in range(control_steps):
        if optimized is None:
            frame = np.asarray(
                reference.get_reference_motion(
                    args.vx, args.vy, args.yaw, control_step
                ),
                dtype=np.float64,
            )
            targets = previous_targets.copy()
            targets[:5] = frame[:5]
            targets[5:9] = 0.0
            targets[9:] = frame[11:16]
        else:
            frame_index = int(np.floor(phase)) % len(frames)
            next_index = (frame_index + 1) % len(frames)
            fraction = phase - np.floor(phase)
            leg_target = means + scales * (
                (1.0 - fraction) * deviations[frame_index]
                + fraction * deviations[next_index]
            )
            targets = home_ctrl.copy()
            targets[leg_actuators] = leg_target
            targets[5:9] = 0.0
            targets = np.clip(targets, lower, upper)
            phase = (phase + phase_rate) % len(frames)
        targets = np.clip(
            targets,
            previous_targets - max_delta,
            previous_targets + max_delta,
        )
        targets = np.clip(
            targets,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        data.ctrl[:] = targets
        previous_targets = targets

        for _ in range(10):
            mujoco.mj_step(model, data)

        left = False
        right = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {
                int(model.geom_bodyid[contact.geom1]),
                int(model.geom_bodyid[contact.geom2]),
            }
            left |= pair == {floor_body, left_body}
            right |= pair == {floor_body, right_body}
        contacts.append([left, right])
        position = data.xpos[trunk_body].copy()
        positions.append(position)
        rotation = data.xmat[trunk_body].reshape(3, 3)
        local_velocities.append(
            rotation.T @ ((position - previous_position) / 0.02)
        )
        previous_position = position
        local_angular_velocities.append(rotation.T @ data.qvel[3:6])
        heights.append(float(data.xpos[trunk_body, 2]))
        upright = float(data.xmat[trunk_body].reshape(3, 3)[2, 2])
        uprights.append(upright)
        if upright < 0.65 or heights[-1] < 0.12:
            fell = True
            break

    elapsed = len(positions) * 0.02
    displacement = np.asarray(positions[-1]) - start
    contact_array = np.asarray(contacts, dtype=np.float64)
    local_velocity_array = np.asarray(local_velocities)
    local_angular_velocity_array = np.asarray(local_angular_velocities)
    steady_start = len(local_velocity_array) // 5
    payload = {
        "command": [args.vx, args.vy, args.yaw],
        "gait_json": str(args.gait_json.resolve()) if args.gait_json else None,
        "elapsed_seconds": elapsed,
        "fell": fell,
        "displacement_xyz": displacement.tolist(),
        "mean_velocity_xyz": (displacement / max(elapsed, 1e-9)).tolist(),
        "steady_mean_local_velocity_xyz": local_velocity_array[
            steady_start:
        ].mean(axis=0).tolist(),
        "steady_mean_local_angular_velocity_xyz": local_angular_velocity_array[
            steady_start:
        ].mean(axis=0).tolist(),
        "minimum_height": min(heights),
        "minimum_upright": min(uprights),
        "left_contact_rate": float(contact_array[:, 0].mean()),
        "right_contact_rate": float(contact_array[:, 1].mean()),
        "single_support_rate": float(
            np.logical_xor(contact_array[:, 0], contact_array[:, 1]).mean()
        ),
        "flight_rate": float((contact_array.sum(axis=1) == 0).mean()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
