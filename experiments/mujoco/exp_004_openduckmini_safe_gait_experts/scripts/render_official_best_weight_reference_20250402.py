#!/usr/bin/env python3
"""Render the published Open Duck Mini BEST_WALK_ONNX_2 policy faithfully.

This intentionally uses ``MjInfer`` from the unmodified public Open Duck
Playground checkout at commit 1842c8f46a67cb5d6b74e5aaf08c8702cde6e74f
(2025-04-02).  The controller below is the body of its ``MjInfer.run`` loop
with only the interactive viewer and wall-clock sleep replaced by a fixed
off-screen camera.  In particular, it does not apply the exp_003 evaluator's
calibration, action masking, target guard, posture adjustment, or command
profiles.

This is a simulation provenance tool only.  It never opens a serial device or
addresses real robot hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

# Must be set before importing MuJoCo.  EGL is the headless backend available
# in the WSL training environment; an explicit environment setting may override
# this for a different host.
os.environ.setdefault("MUJOCO_GL", "egl")

import mediapy as media
import mujoco
import numpy as np


# MediaPy delegates MP4 encoding to ffmpeg.  The training venv includes the
# maintained imageio-ffmpeg binary even when the WSL distribution has no system
# ``ffmpeg`` package installed.  Set the documented override only if the caller
# did not already select an encoder.
if "FFMPEG_BINARY" not in os.environ:
    import imageio_ffmpeg

    os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()
media.set_ffmpeg(os.environ["FFMPEG_BINARY"])


PUBLIC_PLAYGROUND_COMMIT = "1842c8f46a67cb5d6b74e5aaf08c8702cde6e74f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_command(text: str) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=np.float64)
    if values.shape != (7,):
        raise argparse.ArgumentTypeError(
            "command must contain seven comma-separated values: "
            "vx,vy,wz,neck_pitch,head_pitch,head_yaw,head_roll"
        )
    return values


def parse_vector(text: str) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=np.float64)
    if values.shape != (3,):
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return values


def import_official_infer(playground_root: Path):
    root = playground_root.resolve()
    package = root / "playground" / "open_duck_mini_v2" / "mujoco_infer.py"
    if not package.is_file():
        raise FileNotFoundError(
            f"{package} is missing; --playground-root must name the public "
            "Open Duck Playground checkout."
        )
    sys.path.insert(0, str(root))
    from playground.open_duck_mini_v2.mujoco_infer import (  # pylint: disable=import-outside-toplevel
        MjInfer,
        USE_MOTOR_SPEED_LIMITS,
    )

    return MjInfer, USE_MOTOR_SPEED_LIMITS


def update_control(infer, use_motor_speed_limits: bool) -> tuple[np.ndarray, np.ndarray]:
    """Exact non-viewer portion of public ``MjInfer.run`` at the control tick."""
    infer.imitation_i += 1.0 * infer.phase_frequency_factor
    infer.imitation_i = infer.imitation_i % infer.PRM.nb_steps_in_period
    infer.imitation_phase = np.array(
        [
            np.cos(infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi),
            np.sin(infer.imitation_i / infer.PRM.nb_steps_in_period * 2 * np.pi),
        ]
    )

    observation = infer.get_obs(infer.data, infer.commands)
    action = np.asarray(infer.policy.infer(observation), dtype=np.float64)

    infer.last_last_last_action = infer.last_last_action.copy()
    infer.last_last_action = infer.last_action.copy()
    infer.last_action = action.copy()
    infer.motor_targets = infer.default_actuator + action * infer.action_scale

    if use_motor_speed_limits:
        infer.motor_targets = np.clip(
            infer.motor_targets,
            infer.prev_motor_targets
            - infer.max_motor_velocity * (infer.sim_dt * infer.decimation),
            infer.prev_motor_targets
            + infer.max_motor_velocity * (infer.sim_dt * infer.decimation),
        )
        infer.prev_motor_targets = infer.motor_targets.copy()

    infer.data.ctrl = infer.motor_targets.copy()
    return observation, action


def render_frame(renderer: mujoco.Renderer, data: mujoco.MjData, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def root_metrics(infer) -> tuple[np.ndarray, float]:
    root_address = int(infer._floating_base_qpos_addr)  # Official base helper's address.
    root_qpos = infer.data.qpos[root_address : root_address + 7]
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, root_qpos[3:7])
    # The last component of R's third column is body-up projected onto world-up.
    return root_qpos[:3].copy(), float(rotation[8])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playground-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--video-name",
        default="official_best_walk_onnx_2_fixed_camera.mp4",
        help="MP4 filename inside --output-dir (no directory components)",
    )
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=25.0)
    # The unmodified public XML sets its offscreen framebuffer width to 640.
    # Keep the capture within that source-model limit rather than editing XML.
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--command",
        type=parse_command,
        default=parse_command("0.15,0,0,0,0,0,0"),
        help="vx,vy,wz,neck_pitch,head_pitch,head_yaw,head_roll",
    )
    parser.add_argument("--lookat", type=parse_vector, default=parse_vector("0,0,0.25"))
    parser.add_argument("--distance", type=float, default=1.15)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-18.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seconds <= 0 or args.fps <= 0:
        raise ValueError("--seconds and --fps must be positive")

    playground_root = args.playground_root.resolve()
    policy_path = args.policy.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(args.video_name)
    if video_name.name != args.video_name or video_name.suffix.lower() != ".mp4":
        raise ValueError("--video-name must be a simple .mp4 filename")

    renderer_script = Path(__file__).resolve()
    xml_path = playground_root / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
    reference_path = playground_root / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
    infer_path = playground_root / "playground/open_duck_mini_v2/mujoco_infer.py"
    base_path = playground_root / "playground/open_duck_mini_v2/mujoco_infer_base.py"
    for required_path in (policy_path, xml_path, reference_path, infer_path, base_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    MjInfer, use_motor_speed_limits = import_official_infer(playground_root)
    infer = MjInfer(
        str(xml_path), str(reference_path), str(policy_path), standing=False
    )
    infer.commands = args.command.tolist()

    contract = {
        "sim_dt": infer.sim_dt,
        "decimation": infer.decimation,
        "action_scale": infer.action_scale,
        "max_motor_velocity": infer.max_motor_velocity,
        "motor_speed_limits": bool(use_motor_speed_limits),
        "observation_size": 101,
        "action_size": 14,
    }
    expected_contract = {
        "sim_dt": 0.002,
        "decimation": 10,
        "action_scale": 0.25,
        "max_motor_velocity": 5.24,
        "motor_speed_limits": True,
        "observation_size": 101,
        "action_size": 14,
    }
    if contract != expected_contract:
        raise RuntimeError(f"unexpected official inference contract: {contract}")

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = args.lookat
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation

    frame_period = 1.0 / args.fps
    next_frame_time = infer.data.time
    frame_count = 0
    total_steps = math.ceil(args.seconds / infer.sim_dt)
    frames: list[np.ndarray] = []
    initial_position, initial_upright = root_metrics(infer)
    min_height = float(initial_position[2])
    min_upright = initial_upright
    first_observation = None
    first_action = None
    first_target = None

    # The public constructor performs one MuJoCo step before assigning its home
    # qpos.  We deliberately do not call mj_forward here, preserving its first
    # subsequent step and control-tick semantics.
    with mujoco.Renderer(infer.model, height=args.height, width=args.width) as renderer:
        for counter in range(1, total_steps + 1):
            mujoco.mj_step(infer.model, infer.data)
            if counter % infer.decimation == 0:
                observation, action = update_control(infer, use_motor_speed_limits)
                if first_observation is None:
                    first_observation = observation.copy()
                    first_action = action.copy()
                    first_target = infer.motor_targets.copy()

            position, upright = root_metrics(infer)
            min_height = min(min_height, float(position[2]))
            min_upright = min(min_upright, upright)
            elapsed = infer.data.time - 0.002  # Constructor's initial mj_step.
            while elapsed + 1e-12 >= next_frame_time:
                frames.append(render_frame(renderer, infer.data, camera))
                frame_count += 1
                next_frame_time += frame_period

    if not frames or first_observation is None or first_action is None or first_target is None:
        raise RuntimeError("simulation did not reach a control tick and frame capture")

    video_path = output_dir / video_name
    media.write_video(str(video_path), np.stack(frames), fps=args.fps, codec="h264")
    final_position, final_upright = root_metrics(infer)

    manifest = {
        "purpose": "faithful headless replay of public official simulation inference",
        "not_real_robot": True,
        "public_playground_commit": PUBLIC_PLAYGROUND_COMMIT,
        "paths": {
            "playground_root": str(playground_root),
            "policy": str(policy_path),
            "xml": str(xml_path),
            "reference_motion": str(reference_path),
            "official_infer_source": str(infer_path),
            "official_base_source": str(base_path),
            "renderer_script": str(renderer_script),
            "video": str(video_path),
        },
        "sha256": {
            "policy": sha256(policy_path),
            "xml": sha256(xml_path),
            "reference_motion": sha256(reference_path),
            "official_infer_source": sha256(infer_path),
            "official_base_source": sha256(base_path),
            "renderer_script": sha256(renderer_script),
        },
        "command": args.command.tolist(),
        "contract": contract,
        "camera": {
            "lookat": args.lookat.tolist(),
            "distance": args.distance,
            "azimuth": args.azimuth,
            "elevation": args.elevation,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
        },
        "rollout": {
            "seconds_requested": args.seconds,
            "physics_steps": total_steps,
            "frames": frame_count,
            "initial_root_position": initial_position.tolist(),
            "final_root_position": final_position.tolist(),
            "horizontal_displacement": (final_position[:2] - initial_position[:2]).tolist(),
            "initial_upright_cosine": initial_upright,
            "final_upright_cosine": final_upright,
            "minimum_root_height": min_height,
            "minimum_upright_cosine": min_upright,
        },
        "first_control_tick": {
            "observation_sha256": hashlib.sha256(first_observation.tobytes()).hexdigest(),
            "action": first_action.tolist(),
            "motor_target": first_target.tolist(),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"video={video_path}")
    print(f"manifest={manifest_path}")
    print(json.dumps(manifest["rollout"], indent=2))


if __name__ == "__main__":
    main()
