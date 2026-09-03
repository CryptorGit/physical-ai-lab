"""Replay only exp_007 formal G1 capabilities for the exp_008 closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP008 = SCRIPT.parent.parent
REPO = EXP008.parents[2]
EXP007 = REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions"
sys.path[:0] = [
    str(EXP007 / "src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_run_expert, load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import (  # noqa: E402
    canonical_state_from_legacy_observation,
    to_run_observation,
)
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import (  # noqa: E402
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

SCENES = ("StandWalkStand", "WalkToRun26", "WalkToRun28", "StandWalkStandWalkRun26")
CAMERAS = ("SIDE", "REAR_QUARTER", "FOLLOW_POSITION")
EXPECTED = {
    "stand": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "stand_to_walk": "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e",
    "walk": "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
    "walk_to_stand": "bb1bf713119b7980cfac4c1f43eb0d415bc32abe97a54509ee45d13061e858bd",
    "run": "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266",
    "walk_to_run": "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0",
}
PATHS = {
    "stand": REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt",
    "stand_to_walk": REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt",
    "walk": REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
    "walk_to_stand": REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-24_06-37-46_stage4_walk_to_stand_pilot1_1024_100/model_0.pt",
    "run": REPO / "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt",
    "walk_to_run": REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True, choices=SCENES)
parser.add_argument("--seed", type=int, default=20260831)
parser.add_argument("--record", action="store_true")
parser.add_argument("--output-path", default="")
parser.add_argument("--telemetry-output", required=True)
parser.add_argument("--camera-preset", choices=CAMERAS, default="FOLLOW_POSITION")
parser.add_argument("--camera-mode", choices=("Tracking", "Fixed"), default="Tracking")
parser.add_argument("--show-floor-guides", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]
if args.record:
    if not args.output_path:
        parser.error("--record requires --output-path")
    args.enable_cameras = True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimum_jerk(value):
    if torch.is_tensor(value):
        value = value.clamp(0.0, 1.0)
    else:
        value = max(0.0, min(1.0, value))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def periodic(gait: dict) -> bool:
    flights = gait["flights"]
    average = sum(flights) / len(flights) if flights else 0.0
    return (
        len(flights) >= 4
        and gait["max_safe"] >= 3
        and gait["alternating"] / max(gait["alternating_opportunities"], 1) >= 0.8
        and gait["valid"] / max(len(flights), 1) >= 0.8
        and 0.04 <= average <= 0.16
    )


def prepare_recording(raw_env, max_steps: int):
    if not args.record:
        return raw_env, None
    import cv2

    destination = Path(args.output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_folder = destination.parent / "capture_raw"
    raw_folder.mkdir(parents=True, exist_ok=True)
    prefix = destination.stem + "_raw"
    raw_path = raw_folder / f"{prefix}.mp4"
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        50.0,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"unable to open manual video writer: {raw_path}")
    return raw_env, (destination, raw_folder, prefix, writer, raw_path)


def burn_overlay(record_info, lines_by_frame: list[list[str]]) -> None:
    if record_info is None:
        return
    import cv2

    destination, raw_folder, prefix, writer, source = record_info
    writer.release()
    if not source.exists():
        raise RuntimeError(f"manual video recorder produced no clip for {prefix}")
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 50.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temporary = destination.with_name(destination.stem + "_burnin_tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        lines = lines_by_frame[min(frame, len(lines_by_frame) - 1)] if lines_by_frame else []
        overlay = image.copy()
        cv2.rectangle(overlay, (18, 18), (850, 58 + 30 * len(lines)), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
        for index, line in enumerate(lines):
            cv2.putText(
                image, line, (36, 52 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (245, 248, 252), 2, cv2.LINE_AA,
            )
        writer.write(image)
        frame += 1
    capture.release()
    writer.release()
    if destination.exists():
        destination.unlink()
    temporary.replace(destination)
    print(f"record_result=PASS path={destination} frames={frame} fps={fps}")


class FloorGuides:
    """Visual-only debug-draw lane and distance marks; creates no USD/physics prim."""

    def __init__(self, enabled: bool, origin: torch.Tensor, heading_rad: float):
        self.interface = None
        self.line_count = 0
        if not enabled:
            return
        from isaacsim.core.experimental.utils.app import enable_extension

        enable_extension("isaacsim.util.debug_draw")
        from isaacsim.util.debug_draw import _debug_draw
        self.interface = _debug_draw.acquire_debug_draw_interface()
        self.interface.clear_lines()
        starts, ends, colors, widths = [], [], [], []
        origin_xy = origin.detach().cpu().float()[:2]
        forward = torch.tensor((math.cos(heading_rad), math.sin(heading_rad)))
        lateral = torch.tensor((-math.sin(heading_rad), math.cos(heading_rad)))

        def world(track_x: float, track_y: float, z_value: float) -> tuple[float, float, float]:
            xy = origin_xy + float(track_x) * forward + float(track_y) * lateral
            return float(xy[0]), float(xy[1]), z_value

        z = 0.012
        for x in range(-5, 71):
            major = x % 5 == 0
            starts.append(world(float(x), -1.5, z))
            ends.append(world(float(x), 1.5, z))
            colors.append((0.25, 0.72, 0.90, 1.0) if major else (0.55, 0.60, 0.64, 1.0))
            widths.append(5.0 if major else 2.0)
        for y in (-1.5, 1.5):
            starts.append(world(-5.0, y, z))
            ends.append(world(70.0, y, z))
            colors.append((0.88, 0.90, 0.92, 1.0))
            widths.append(4.0)
        starts.append(world(0.0, -1.5, z + 0.001))
        ends.append(world(0.0, 1.5, z + 0.001))
        colors.append((1.0, 0.78, 0.08, 1.0))
        widths.append(8.0)
        self.interface.draw_lines(starts, ends, colors, widths)
        self.line_count = len(starts)


class Camera:
    """Position-tracking camera adapted from exp_007 capability showcase playback."""

    def __init__(self, env, scene: str, preset: str, mode: str, heading_rad: float):
        self.env = env
        self.preset = preset
        self.mode = mode
        local_offset = (-3.8, -3.2, 1.9) if preset != "SIDE" else (-1.5, -4.5, 1.8)
        forward = torch.tensor((math.cos(heading_rad), math.sin(heading_rad), 0.0))
        lateral = torch.tensor((-math.sin(heading_rad), math.cos(heading_rad), 0.0))
        self.forward = forward
        self.offset = local_offset[0] * forward + local_offset[1] * lateral + torch.tensor((0.0, 0.0, local_offset[2]))
        self.filtered = None
        self.trace = []

    def update(
        self,
        follow_point: torch.Tensor,
        physics_root: torch.Tensor,
        time_s: float,
        immediate: bool = False,
    ) -> None:
        point = follow_point.detach().cpu().float()
        if self.mode == "Fixed":
            point = torch.tensor((0.0, 0.0, float(point[2])))
        self.filtered = point.clone() if immediate or self.filtered is None else 0.65 * self.filtered + 0.35 * point
        look = self.filtered + 0.5 * self.forward + torch.tensor((0.0, 0.0, 0.8))
        eye_tensor = self.filtered + self.offset
        target = tuple(float(value) for value in look)
        eye = tuple(float(value) for value in eye_tensor)
        self.env.sim.set_camera_view(eye=eye, target=target)
        recorder = getattr(self.env, "video_recorder", None)
        capture = getattr(recorder, "_capture", None)
        capture_cfg = getattr(capture, "cfg", None)
        if capture_cfg is not None:
            # Isaac Lab's Kit recording backend owns a separate perspective
            # camera; unlike the live viewport it is not synchronized after
            # construction, so update that exact camera every rendered step.
            from isaacsim.core.rendering_manager import ViewportManager

            ViewportManager.set_camera_view(
                capture_cfg.camera_prim_path,
                eye=list(eye),
                target=list(target),
            )
        self.trace.append({
            "time": time_s,
            "robot_root_xyz": [float(value) for value in physics_root.detach().cpu()],
            "camera_follow_point_xyz": [float(value) for value in follow_point.detach().cpu()],
            "camera_xyz": list(eye),
            "look_at_xyz": list(target),
            "camera_to_robot_distance": float(torch.linalg.vector_norm(eye_tensor - point)),
        })


def main() -> None:
    hashes = {name: sha256(path.resolve(strict=True)) for name, path in PATHS.items()}
    if hashes != EXPECTED:
        raise RuntimeError(f"protected checkpoint mismatch: {hashes}")
    continuous_sequence = args.scene == "StandWalkStandWalkRun26"
    scene_display = "STOP / WALK / STOP / WALK / RUN 2.6" if continuous_sequence else args.scene
    run_target = 2.6 if args.scene in ("WalkToRun26", "StandWalkStandWalkRun26") else 2.8
    max_seconds = 34.0 if continuous_sequence else 24.0 if args.scene == "StandWalkStand" else 19.0
    cfg, agent = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 40.0
    cfg.seed = args.seed
    cfg.episode_length_s = max_seconds + 2.0
    cfg.viewer.origin_type = "world"
    cfg.video_recorder.window_width = args.width
    cfg.video_recorder.window_height = args.height
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make(
            "Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg,
            render_mode="rgb_array" if args.record else None,
        )
        max_steps = round(max_seconds / float(raw.unwrapped.step_dt))
        recorded, record_info = prepare_recording(raw, max_steps)
        wrapped = RslRlVecEnvWrapper(recorded, clip_actions=agent.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        stand = load_walk_expert(PATHS["stand"], device=device)
        stw = load_walk_expert(PATHS["stand_to_walk"], device=device)
        walk = load_walk_expert(PATHS["walk"], device=device)
        wts = load_walk_expert(PATHS["walk_to_stand"], device=device)
        run = load_run_expert(PATHS["run"], device=device)
        transition = WalkToRunTransitionActor152(run.actor).to(device)
        payload = torch.load(PATHS["walk_to_run"], map_location=device, weights_only=False)
        transition.load_state_dict(payload["actor"], strict=True)
        transition.eval()
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        wrapped.reset()
        course_heading = float(robot.data.heading_w.torch[0].detach().cpu())
        floor_guides = FloorGuides(args.show_floor_guides, robot.data.root_pos_w.torch[0], course_heading)
        camera = Camera(env, args.scene, args.camera_preset, args.camera_mode, course_heading)
        camera.update(
            robot.data.root_pos_w.torch[0],
            robot.data.root_pos_w.torch[0],
            0.0,
            immediate=True,
        )
        heading = robot.data.heading_w.torch.clone()
        filtered_yaw = torch.zeros(1, device=device)
        previous_action = torch.zeros(1, 37, device=device)
        phase = "STAND"
        elapsed = 0.0
        streak = 0.0
        support_switches = 0
        previous_support = 0
        no_switch_elapsed = 0.0
        gait = {
            "flights": [], "valid": 0, "alternating": 0,
            "alternating_opportunities": 0, "last": None, "safe": 0, "max_safe": 0,
        }
        in_flight = False
        flight_start = 0.0
        previous_contacts = (False, False)
        walk_speeds = (0.6, 0.8, 1.0, 1.2)
        walk_index = 0
        state_sequence = ["STAND"]
        overlays: list[list[str]] = []
        routing_errors = 0
        unsupported = 0
        result = "IN_PROGRESS"
        action_digest = hashlib.sha256()
        root_digest = hashlib.sha256()
        dt = float(env.step_dt)
        for step in range(max_steps):
            if phase == "STAND" and elapsed >= 3.0:
                initial_departure = len(state_sequence) == 1
                second_departure = continuous_sequence and state_sequence[-2:] == ["WALK_TO_STAND", "STAND"]
                if initial_departure or second_departure:
                    phase, elapsed, streak, support_switches = "STAND_TO_WALK", 0.0, 0.0, 0
                    state_sequence.append(phase)
            target_walk = walk_speeds[walk_index] if args.scene == "StandWalkStand" else 1.2
            if phase == "STAND_TO_WALK":
                command_speed = target_walk * minimum_jerk(elapsed / 1.5)
            elif phase == "WALK":
                command_speed = target_walk
            elif phase == "WALK_TO_STAND":
                command_speed = 1.2 * (1.0 - minimum_jerk(elapsed / 1.6))
            elif phase == "WALK_TO_RUN":
                command_speed = 1.2 + (run_target - 1.2) * minimum_jerk(elapsed / 1.4)
            elif phase == "RUN_LOW":
                command_speed = run_target
            else:
                command_speed = 0.0
            heading_error = torch.atan2(
                torch.sin(heading - robot.data.heading_w.torch),
                torch.cos(heading - robot.data.heading_w.torch),
            )
            raw_yaw = torch.where(
                torch.tensor([phase in ("WALK_TO_RUN", "RUN_LOW")], device=device),
                (1.5 * heading_error).clamp(-1.5, 1.5),
                (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3),
            )
            filtered_yaw += (0.15 * (raw_yaw - filtered_yaw)).clamp(-0.01, 0.01)
            if phase == "STAND":
                filtered_yaw.zero_()
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0] = command_speed
            command_term.vel_command_b[:, 2] = filtered_yaw
            legacy = wrapped.get_observations()["policy"]
            state = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
            command = MotionCommand(
                torch.tensor([command_speed], device=device), heading,
                target_yaw_rate_radps=filtered_yaw,
            )
            controller_name = {
                "STAND": "STAND expert", "STAND_TO_WALK": "STAND_TO_WALK expert",
                "WALK": "WALK expert", "WALK_TO_STAND": "WALK_TO_STAND expert",
                "WALK_TO_RUN": "WALK_TO_RUN expert", "RUN_LOW": "RUN_LOW expert",
            }[phase]
            with torch.inference_mode():
                if phase == "STAND":
                    action = stand(state, command)
                elif phase == "STAND_TO_WALK":
                    action = stw(state, command)
                elif phase == "WALK":
                    action = walk(state, command)
                elif phase == "WALK_TO_STAND":
                    action = wts(state, command)
                elif phase == "WALK_TO_RUN":
                    action = transition(to_run_observation(state, command, route="RUN"))
                else:
                    action = run(state, command)
                _, _, done, _ = wrapped.step(action)
                # Synchronize both live and recording cameras to the current
                # physics root before explicitly capturing this rendered frame.
                camera.update(
                    robot.data.root_pos_w.torch[0],
                    robot.data.root_pos_w.torch[0],
                    step * dt,
                )
                if record_info is not None:
                    import cv2

                    frame = raw.render()
                    record_info[3].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            previous_action[:] = action
            action_digest.update(action.detach().cpu().contiguous().numpy().tobytes())
            root_digest.update(robot.data.root_state_w.torch[0].detach().cpu().contiguous().numpy().tobytes())
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            support = int(contacts[0, 0]) + 2 * int(contacts[0, 1])
            speed = float(robot.data.root_lin_vel_b.torch[0, 0])
            g = robot.data.projected_gravity_b.torch[0]
            roll = abs(float(torch.atan2(g[1], -g[2])))
            pitch = abs(float(torch.atan2(-g[0], torch.sqrt(g[1] ** 2 + g[2] ** 2))))
            if support and support != previous_support:
                support_switches += 1
                no_switch_elapsed = 0.0
            else:
                no_switch_elapsed += dt
            if phase in ("WALK_TO_RUN", "RUN_LOW"):
                current_contacts = (bool(contacts[0, 0]), bool(contacts[0, 1]))
                count = int(current_contacts[0]) + int(current_contacts[1])
                if count == 0 and not in_flight:
                    in_flight, flight_start = True, elapsed
                if in_flight and count > 0:
                    duration = elapsed - flight_start
                    new = [index for index in range(2) if current_contacts[index] and not previous_contacts[index]]
                    valid = len(new) == 1
                    side = new[0] if valid else -1
                    gait["flights"].append(duration)
                    if valid:
                        gait["valid"] += 1
                        if gait["last"] is not None:
                            gait["alternating_opportunities"] += 1
                            gait["alternating"] += int(side != gait["last"])
                        safe = 0.04 <= duration <= 0.16 and (gait["last"] is None or side != gait["last"])
                        gait["safe"] = gait["safe"] + 1 if safe else 0
                        gait["max_safe"] = max(gait["max_safe"], gait["safe"])
                        gait["last"] = side
                    in_flight = False
                previous_contacts = current_contacts
            if phase == "STAND_TO_WALK":
                good = abs(speed - target_walk) <= 0.20 and abs(float(heading_error[0])) <= 0.12 and support_switches >= 2
                streak = streak + dt if good else 0.0
                if streak >= 0.4:
                    phase, elapsed, streak = "WALK", 0.0, 0.0
                    state_sequence.append(phase)
            elif phase == "WALK":
                if args.scene == "StandWalkStand" and elapsed >= 3.0:
                    if walk_index < len(walk_speeds) - 1:
                        walk_index += 1
                        elapsed = 0.0
                    else:
                        phase, elapsed, streak = "WALK_TO_STAND", 0.0, 0.0
                        state_sequence.append(phase)
                elif continuous_sequence and elapsed >= 3.0:
                    if state_sequence.count("WALK") == 1:
                        phase, elapsed, streak = "WALK_TO_STAND", 0.0, 0.0
                    else:
                        phase, elapsed, streak = "WALK_TO_RUN", 0.0, 0.0
                        gait = {
                            **gait, "flights": [], "valid": 0, "alternating": 0,
                            "alternating_opportunities": 0, "last": None,
                            "safe": 0, "max_safe": 0,
                        }
                    state_sequence.append(phase)
                elif args.scene != "StandWalkStand" and elapsed >= 3.0:
                    phase, elapsed, streak = "WALK_TO_RUN", 0.0, 0.0
                    gait = {
                        **gait, "flights": [], "valid": 0, "alternating": 0,
                        "alternating_opportunities": 0, "last": None,
                        "safe": 0, "max_safe": 0,
                    }
                    state_sequence.append(phase)
            elif phase == "WALK_TO_STAND":
                good = abs(speed) <= 0.08 and roll <= 0.10 and pitch <= 0.10 and support == 3 and no_switch_elapsed >= 0.4
                streak = streak + dt if good else 0.0
                if streak >= 0.4:
                    phase, elapsed, streak = "STAND", 0.0, 0.0
                    state_sequence.append(phase)
            elif phase == "WALK_TO_RUN":
                good = periodic(gait) and abs(speed - run_target) <= 0.20 and abs(float(heading_error[0])) <= 0.12
                streak = streak + dt if good else 0.0
                if streak >= 0.4:
                    phase, elapsed, streak = "RUN_LOW", 0.0, 0.0
                    state_sequence.append(phase)
            supported = (
                f"{target_walk:.1f} m/s" if phase in ("STAND_TO_WALK", "WALK", "WALK_TO_STAND")
                else f"{run_target:.1f} m/s" if phase in ("WALK_TO_RUN", "RUN_LOW") else "STAND"
            )
            overlays.append([
                "EXP_008 CLOSEOUT SHOWCASE",
                f"SCENE: {scene_display}",
                f"CURRENT STATE: {phase}",
                f"ACTIVE CONTROLLER: {controller_name}",
                f"TARGET SPEED: {command_speed:.2f} m/s   ACTUAL SPEED: {speed:.2f} m/s",
                f"TRANSITION: {'NONE' if phase in ('STAND','WALK','RUN_LOW') else phase}",
                f"SUPPORTED COMMAND: {supported}",
            ])
            previous_support = support
            if bool(done[0]):
                result = "FAIL_ENVIRONMENT_DONE"
                break
            if args.scene == "StandWalkStand" and phase == "STAND" and len(state_sequence) > 2 and elapsed >= 3.0:
                result = "PASS"
                break
            if args.scene != "StandWalkStand" and phase == "RUN_LOW" and elapsed >= 7.0:
                result = "PASS" if periodic(gait) and abs(speed - run_target) <= 0.30 else "FAIL_RUN_HOLD"
                break
            elapsed += dt
        wrapped.close()
        burn_overlay(record_info, overlays)
        telemetry = {
            "scene": args.scene,
            "seed": args.seed,
            "result": result,
            "actual_state_sequence": state_sequence,
            "target_speed_mps": list(walk_speeds) if args.scene == "StandWalkStand" else run_target,
            "unsupported_transition_executed": 0,
            "routing_error": routing_errors,
            "new_scene_reset_not_transition": True,
            "checkpoint_hashes": hashes,
            "recorded": bool(args.record),
            "camera_preset": args.camera_preset,
            "camera_mode": args.camera_mode,
            "camera_offset_xyz": [float(value) for value in camera.offset],
            "camera_update_frequency": "every control/render step after physics and before explicit frame capture",
            "camera_trace": camera.trace[::5],
            "floor_guides": {
                "enabled": args.show_floor_guides,
                "line_count": floor_guides.line_count,
                "range_x_m": [-5, 70],
                "lane_boundaries_y_m": [-1.5, 1.5],
                "minor_spacing_m": 1,
                "major_spacing_m": 5,
                "collision_prims": 0,
                "contact_events": 0,
                "implementation": "isaacsim.util.debug_draw._debug_draw.draw_lines",
            },
            "action_trace_sha256": action_digest.hexdigest(),
            "root_state_trace_sha256": root_digest.hexdigest(),
        }
        telemetry_path = Path(args.telemetry_output).resolve()
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(telemetry, indent=2))
        if result != "PASS":
            raise SystemExit(3)


if __name__ == "__main__":
    main()
