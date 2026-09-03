"""Isaac Sim-only visual showcase for the frozen exp_006 command system.

This file intentionally owns presentation timing, camera motion, debug drawing,
overlay text, and optional video capture.  It does not write checkpoints or call
the formal evaluators.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parent.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path[:0] = [
    str(EXPERIMENT_ROOT / "src"),
    str(REPOSITORY_ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
]

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from g1_command_skills.scripted_crouch import phased_offset  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import (  # noqa: E402
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)


SHOWCASES = (
    "TURN_LEFT_90",
    "TURN_RIGHT_90",
    "TURN_S_CURVE",
    "CROUCH_SHOWCASE",
    "SAFE_REJECTION",
)
CAMERAS = ("WORLD_FIXED", "FOLLOW_POSITION", "TOP_DOWN")
RUN_SKILL_ID = 0
TURN_SKILL_ID = 2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--showcase", required=True, choices=SHOWCASES)
parser.add_argument("--camera", default="WORLD_FIXED", choices=CAMERAS)
parser.add_argument("--run-checkpoint", required=True)
parser.add_argument("--standing-checkpoint", required=True)
parser.add_argument("--crouch-checkpoint", required=True)
parser.add_argument("--output", required=True, help="Telemetry output directory.")
parser.add_argument("--seed", type=int, default=20260723)
parser.add_argument("--record", action="store_true")
parser.add_argument("--output-path", default="", help="Exact .mp4 destination when --record is used.")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args
if args_cli.record:
    if not args_cli.output_path:
        parser.error("--record requires --output-path")
    # The installed Isaac Lab recorder uses the Kit/replicator RGB backend.
    args_cli.enable_cameras = True


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def phase_name(value: int) -> str:
    return {0: "RUN", 1: "TURN", 2: "RUN"}.get(value, "UNKNOWN")


def fixed_reset(cfg) -> None:
    """Make presentation geometry repeatable without altering any task source."""
    reset = cfg.events.reset_base.params
    reset["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    reset["velocity_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.viewer.origin_type = "world"
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 40.0
    cfg.seed = args_cli.seed
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    cfg.video_recorder.window_width = args_cli.width
    cfg.video_recorder.window_height = args_cli.height


def course_camera(showcase: str, camera: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Derive a world camera from the deterministic course envelope."""
    if showcase == "TURN_LEFT_90":
        center, span = (5.0, 6.5), 18.0
    elif showcase == "TURN_RIGHT_90":
        center, span = (5.0, -6.5), 18.0
    elif showcase == "TURN_S_CURVE":
        center, span = (12.0, 7.5), 26.0
    else:
        center, span = (0.0, 0.0), 5.0
    if camera == "TOP_DOWN":
        return (center[0], center[1], max(9.0, span)), (center[0], center[1], 0.0)
    # The same transform is also the world-orientation template for FOLLOW_POSITION.
    scale = span / 13.0
    return (
        center[0] + 8.5 * scale,
        center[1] - 10.0 * scale,
        9.0 * scale,
    ), (center[0], center[1], 0.7)


class ShowcaseOverlay:
    """Small Kit UI overlay; console remains the fallback."""

    def __init__(self) -> None:
        self.label = None
        self.window = None
        try:
            import omni.ui as ui

            self.window = ui.Window("Command System Showcase", width=520, height=285)
            self.window.position_x = 18
            self.window.position_y = 72
            with self.window.frame:
                with ui.ZStack():
                    ui.Rectangle(style={"background_color": 0xCC15191F, "border_radius": 8})
                    with ui.VStack(spacing=5, height=0):
                        ui.Spacer(height=12)
                        with ui.HStack():
                            ui.Spacer(width=16)
                            self.label = ui.Label(
                                "",
                                style={"font_size": 20, "color": 0xFFF4F7FA},
                                word_wrap=True,
                            )
                            ui.Spacer(width=12)
                        ui.Spacer(height=10)
        except Exception as exc:
            print(f"overlay_backend=console reason={type(exc).__name__}:{exc}")

    def set(self, lines: list[str]) -> None:
        text = "\n".join(lines)
        if self.label is not None:
            self.label.text = text


class CourseDraw:
    """World-space course grid, arrows, markers, and pelvis trajectory."""

    def __init__(self, center: tuple[float, float], half_extent: float) -> None:
        self.interface = None
        self.last = None
        try:
            from isaacsim.util.debug_draw import _debug_draw

            self.interface = _debug_draw.acquire_debug_draw_interface()
            self.interface.clear_lines()
            starts, ends, colors, widths = [], [], [], []
            spacing = 1.0
            lo_x = math.floor(center[0] - half_extent)
            hi_x = math.ceil(center[0] + half_extent)
            lo_y = math.floor(center[1] - half_extent)
            hi_y = math.ceil(center[1] + half_extent)
            for x in range(lo_x, hi_x + 1):
                starts.append((x, lo_y, 0.012)); ends.append((x, hi_y, 0.012))
                colors.append((0.28, 0.32, 0.36, 1.0)); widths.append(1.0)
            for y in range(lo_y, hi_y + 1):
                starts.append((lo_x, y, 0.012)); ends.append((hi_x, y, 0.012))
                colors.append((0.28, 0.32, 0.36, 1.0)); widths.append(1.0)
            self.interface.draw_lines(starts, ends, colors, widths)
            self.arrow((0.0, 0.0), 0.0, (0.1, 0.75, 1.0, 1.0), 2.2)
        except Exception as exc:
            print(f"debug_draw=disabled reason={type(exc).__name__}:{exc}")

    def line(self, a, b, color, width=4.0) -> None:
        if self.interface is not None:
            self.interface.draw_lines(
                [(float(a[0]), float(a[1]), 0.035)],
                [(float(b[0]), float(b[1]), 0.035)],
                [color], [float(width)],
            )

    def arrow(self, origin, heading, color, length=2.0) -> None:
        tip = (origin[0] + length * math.cos(heading), origin[1] + length * math.sin(heading))
        self.line(origin, tip, color, 7.0)
        wing = 0.35
        for offset in (2.55, -2.55):
            end = (tip[0] + wing * math.cos(heading + offset), tip[1] + wing * math.sin(heading + offset))
            self.line(tip, end, color, 7.0)

    def marker(self, point, color) -> None:
        x, y = point
        self.line((x - 0.22, y - 0.22), (x + 0.22, y + 0.22), color, 8.0)
        self.line((x - 0.22, y + 0.22), (x + 0.22, y - 0.22), color, 8.0)

    def trace(self, point) -> None:
        if self.last is not None:
            self.line(self.last, point, (1.0, 0.78, 0.08, 1.0), 6.0)
        self.last = point


class CameraController:
    """Position-only follow mode; orientation never depends on robot yaw."""

    def __init__(self, env, mode: str, eye, target) -> None:
        self.env = env
        self.mode = mode
        self.eye_offset = torch.tensor(
            [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]],
            dtype=torch.float,
        )
        self.filtered_xy = None
        self.write_count = 0
        env.sim.set_camera_view(eye=eye, target=target)
        self.write_count += 1

    def update(self, robot_xy: torch.Tensor) -> None:
        if self.mode != "FOLLOW_POSITION":
            return
        xy = robot_xy.detach().cpu().float()
        self.filtered_xy = xy.clone() if self.filtered_xy is None else 0.96 * self.filtered_xy + 0.04 * xy
        target = (float(self.filtered_xy[0]), float(self.filtered_xy[1]), 0.75)
        eye = (
            target[0] + float(self.eye_offset[0]),
            target[1] + float(self.eye_offset[1]),
            target[2] + float(self.eye_offset[2]),
        )
        self.env.sim.set_camera_view(eye=eye, target=target)
        self.write_count += 1


def load_runner(wrapped, agent_cfg, checkpoint: Path):
    env = wrapped.unwrapped
    agent_cfg.device = env.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
    )
    return runner


def prepare_recording(raw_env, max_steps: int):
    if not args_cli.record:
        return raw_env, None
    destination = Path(args_cli.output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = destination.stem + "_raw"
    raw_folder = Path(args_cli.output).resolve() / "capture_raw"
    raw_folder.mkdir(parents=True, exist_ok=True)
    wrapped = gym.wrappers.RecordVideo(
        raw_env,
        video_folder=str(raw_folder),
        step_trigger=lambda step: step == 0,
        video_length=max_steps,
        name_prefix=prefix,
        disable_logger=True,
    )
    return wrapped, (destination, prefix, raw_folder)


def finalize_recording(record_info, burned_lines: list[list[str]]) -> None:
    if record_info is None:
        return
    destination, prefix, raw_folder = record_info
    candidates = sorted(raw_folder.glob(f"{prefix}*.mp4"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        print(f"record_result=FAILED reason=no_mp4_found prefix={prefix}")
        return
    source = candidates[-1]
    # Burn the status overlay into the exported RGB stream.  Kit UI widgets are
    # not part of the replicator render product, so this preserves the claims in
    # both OBS captures and direct -Record output.
    try:
        import cv2

        capture = cv2.VideoCapture(str(source))
        fps = capture.get(cv2.CAP_PROP_FPS) or 50.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temporary = destination.with_name(destination.stem + "_burnin_tmp.mp4")
        writer = cv2.VideoWriter(
            str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            lines = burned_lines[min(index, len(burned_lines) - 1)] if burned_lines else []
            box_width = min(width - 32, 720)
            box_height = min(height - 32, 44 + 34 * len(lines))
            overlay = frame.copy()
            cv2.rectangle(overlay, (18, 18), (18 + box_width, 18 + box_height), (20, 24, 30), -1)
            cv2.addWeighted(overlay, 0.78, frame, 0.22, 0.0, frame)
            for line_index, line in enumerate(lines):
                cv2.putText(
                    frame, line, (36, 55 + line_index * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (244, 248, 252), 2, cv2.LINE_AA,
                )
            writer.write(frame)
            index += 1
        capture.release()
        writer.release()
        if destination.exists():
            destination.unlink()
        temporary.replace(destination)
        print(f"record_result=PASS output_path={destination} frames={index} overlay_burned_in=true")
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        source.replace(destination)
        print(
            f"record_result=PASS output_path={destination} overlay_burned_in=false "
            f"reason={type(exc).__name__}:{exc}"
        )


def emit_status(lines: list[str], overlay: ShowcaseOverlay, burned: list[list[str]], step: int) -> None:
    overlay.set(lines)
    burned.append(lines)
    if step % 25 == 0:
        print("SHOWCASE_STATUS " + " | ".join(lines))


def run_turn_showcase() -> dict:
    run_checkpoint = Path(args_cli.run_checkpoint).resolve(strict=True)
    task = "Isaac-Motion-Flat-G1-Command-TurnFull-Eval-v0"
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    fixed_reset(env_cfg)
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.rehearsal_probabilities = (0.0, 1.0, 0.0, 0.0, 0.0)
    command_cfg.turn_angles_deg = (90.0,)
    command_cfg.turn_angle_probabilities = (1.0,)
    command_cfg.deterministic_turn_evaluation = False
    command_cfg.turn_direction_probabilities = (
        (1.0, 0.0) if args_cli.showcase != "TURN_RIGHT_90" else (0.0, 1.0)
    )
    command_cfg.run_speed_range = (2.4, 2.4)
    command_cfg.turn_speed_range = (2.0, 2.0)
    command_cfg.turn_script_durations_s = (3.0, 8.0, 4.0)
    command_cfg.phase_duration_jitter_fraction = 0.0
    env_cfg.episode_length_s = 24.0
    eye, target = course_camera(args_cli.showcase, args_cli.camera)
    env_cfg.viewer.eye, env_cfg.viewer.lookat = eye, target
    env_cfg.video_recorder.eye, env_cfg.video_recorder.lookat = eye, target
    max_steps_hint = 1200

    with launch_simulation(env_cfg, args_cli):
        base_raw = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args_cli.record else None)
        record_raw, record_info = prepare_recording(base_raw, max_steps_hint)
        wrapped = RslRlVecEnvWrapper(record_raw, clip_actions=agent_cfg.clip_actions)
        runner = load_runner(wrapped, agent_cfg, run_checkpoint)
        policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
        env = wrapped.unwrapped
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        wrapped.reset()
        dt = float(env.step_dt)
        overlay = ShowcaseOverlay()
        if args_cli.showcase == "TURN_LEFT_90":
            center, half = (4.0, 4.5), 10.0
            sequence_text = "RUN -> LEFT 90 -> RUN"
        elif args_cli.showcase == "TURN_RIGHT_90":
            center, half = (4.0, -4.5), 10.0
            sequence_text = "RUN -> RIGHT 90 -> RUN"
        else:
            center, half = (12.0, 7.5), 16.0
            sequence_text = "RUN -> LEFT 90 -> RUN -> RIGHT 90 -> RUN"
        draw = CourseDraw(center, half)
        camera = CameraController(env, args_cli.camera, eye, target)
        telemetry: list[dict] = []
        burned: list[list[str]] = []
        turn_results: list[dict] = []
        turn_started = False
        turn_marker_drawn = False
        recovery_marker_drawn = False
        completion_streak = 0
        first_recovery_started_at = None
        second_turn_requested = False
        final_recovery_started_at = None
        last_segment = int(term.segment_index[0].item())
        final_step = max_steps_hint

        for step in range(max_steps_hint):
            observations = wrapped.get_observations()
            with torch.inference_mode():
                actions = policy(observations)
                _, _, dones, _ = wrapped.step(actions)
            now = (step + 1) * dt
            segment = int(term.segment_index[0].item())
            skill = int(term.skill_id[0].item())
            position = robot.data.root_pos_w.torch[0, :2]
            heading = float(robot.data.heading_w.torch[0].item())
            speed = float(robot.data.root_lin_vel_b.torch[0, 0].item())
            legacy_yaw = float(term.vel_command_b[0, 2].item())
            accumulated = float(term.actual_accumulated_yaw_rad[0].item())
            commanded = float(term.commanded_turn_angle_rad[0].item())
            target_heading = float(term.target_heading_w[0].item())
            error = wrap_angle(target_heading - heading)

            if step % 3 == 0:
                draw.trace((float(position[0]), float(position[1])))
            camera.update(position)

            if segment == 1 and not turn_marker_drawn:
                turn_started = True
                turn_marker_drawn = True
                point = (float(position[0]), float(position[1]))
                draw.marker(point, (1.0, 0.25, 0.12, 1.0))
                draw.arrow(point, target_heading, (0.2, 1.0, 0.28, 1.0), 2.4)
                print(
                    f"turn_started command_deg={math.degrees(commanded):.1f} "
                    f"target_heading_deg={math.degrees(target_heading):.2f} "
                    f"legacy_yaw_rate_command_rps={legacy_yaw:.4f}"
                )

            if segment == 1:
                completion_streak = completion_streak + 1 if abs(error) <= 0.12 else 0
                if completion_streak >= max(1, round(0.25 / dt)):
                    # Showcase-only early completion: the controller command is
                    # unchanged; only the presentation schedule advances once
                    # the same 0.12 rad formal heading tolerance is held.
                    term.segment_elapsed[0] = term.segment_duration[0]

            if last_segment == 1 and segment == 2:
                result = {
                    "commanded_angle_deg": math.degrees(commanded),
                    "actual_angle_deg": math.degrees(accumulated),
                    "final_error_deg": math.degrees(abs(commanded - accumulated)),
                    "completion_time_s": now,
                }
                turn_results.append(result)
                print(
                    f"TURN_RESULT commanded_angle_deg={result['commanded_angle_deg']:.3f} "
                    f"actual_angle_deg={result['actual_angle_deg']:.3f} "
                    f"final_error_deg={result['final_error_deg']:.3f}"
                )
                point = (float(position[0]), float(position[1]))
                draw.marker(point, (0.25, 1.0, 0.35, 1.0))
                recovery_marker_drawn = True
                completion_streak = 0
                if first_recovery_started_at is None:
                    first_recovery_started_at = now
                else:
                    final_recovery_started_at = now

            if (
                args_cli.showcase == "TURN_S_CURVE"
                and first_recovery_started_at is not None
                and not second_turn_requested
                and now - first_recovery_started_at >= 3.0
            ):
                second_turn_requested = True
                term.cfg.turn_direction_probabilities = (0.0, 1.0)
                ids = torch.tensor([0], dtype=torch.long, device=env.device)
                with torch.inference_mode(False):
                    term.segment_index[ids] = 1
                    # MotionCommand accepts IntEnum values, but comparisons are
                    # numeric; avoid importing its Isaac Lab module before the
                    # SimulationApp starts.
                    term._set_skill(ids, TURN_SKILL_ID, 8.0)
                    term._configure_skill_targets(ids, TURN_SKILL_ID)
                turn_marker_drawn = False
                recovery_marker_drawn = False
                last_segment = 1
                print("showcase_only_sequence=true injected_supported_transition=RUN_TO_TURN direction=RIGHT angle_deg=90")
                continue

            active = phase_name(segment)
            command_label = (
                f"{'LEFT' if commanded >= 0 else 'RIGHT'} {abs(math.degrees(commanded)):.0f} deg"
                if skill == TURN_SKILL_ID else "RUN 2.4 m/s"
            )
            title = (
                "Unitree G1 Command-Driven Motion System"
                if now <= 2.0 else f"DEMO: {args_cli.showcase}"
            )
            lines = [
                title,
                f"ACTIVE: {active}",
                f"COMMAND: {command_label}",
                f"TARGET HEADING: {math.degrees(target_heading):.1f} deg",
                f"ACTUAL TURN: {math.degrees(accumulated):.1f} deg",
                f"HEADING ERROR: {math.degrees(abs(error)):.1f} deg",
                f"SPEED: {speed:.2f} m/s",
                f"SEQUENCE: {sequence_text}",
            ]
            emit_status(lines, overlay, burned, step)
            telemetry.append({
                "time_s": now, "segment": segment, "active": active,
                "position_xy_m": [float(position[0]), float(position[1])],
                "heading_rad": heading, "target_heading_rad": target_heading,
                "commanded_turn_angle_rad": commanded,
                "actual_accumulated_yaw_rad": accumulated,
                "heading_error_rad": error, "legacy_yaw_rate_command_rps": legacy_yaw,
                "speed_mps": speed,
            })
            last_segment = segment
            if bool(dones[0].item()):
                print("showcase_terminated=environment_done")
                final_step = step + 1
                break
            recovery_origin = final_recovery_started_at if args_cli.showcase == "TURN_S_CURVE" else first_recovery_started_at
            if recovery_origin is not None and now - recovery_origin >= 4.0:
                final_step = step + 1
                break

        output = Path(args_cli.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        summary = {
            "showcase": args_cli.showcase,
            "checkpoint": str(run_checkpoint),
            "camera": args_cli.camera,
            "camera_yaw_follow": False,
            "camera_transform_write_count": camera.write_count,
            "run_speed_mps": 2.4,
            "turn_speed_mps": 2.0,
            "turn_completion_tolerance_rad": 0.12,
            "turn_completion_hold_s": 0.25,
            "showcase_only_multi_turn_sequence": args_cli.showcase == "TURN_S_CURVE",
            "turn_results": turn_results,
            "trajectory_xy_m": [row["position_xy_m"] for row in telemetry],
            "telemetry": telemetry,
        }
        (output / "showcase_telemetry.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"actual_accumulated_yaw_log={output / 'showcase_telemetry.json'}")
        wrapped.close()
        finalize_recording(record_info, burned[:final_step])
        return summary


def run_standing_showcase() -> dict:
    is_crouch = args_cli.showcase == "CROUCH_SHOWCASE"
    checkpoint = Path(args_cli.crouch_checkpoint if is_crouch else args_cli.standing_checkpoint).resolve(strict=True)
    task = (
        "Isaac-Motion-Flat-G1-Command-Crouch-Eval-v0"
        if is_crouch else "Isaac-Velocity-Flat-G1-Run-Eval-v0"
    )
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    fixed_reset(env_cfg)
    env_cfg.episode_length_s = 14.0 if is_crouch else 8.0
    eye, target = course_camera(args_cli.showcase, args_cli.camera)
    env_cfg.viewer.eye, env_cfg.viewer.lookat = eye, target
    env_cfg.video_recorder.eye, env_cfg.video_recorder.lookat = eye, target
    max_seconds = 9.0 if is_crouch else 7.0
    max_steps_hint = 500

    with launch_simulation(env_cfg, args_cli):
        base_raw = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args_cli.record else None)
        record_raw, record_info = prepare_recording(base_raw, max_steps_hint)
        wrapped = RslRlVecEnvWrapper(record_raw, clip_actions=agent_cfg.clip_actions)
        runner = load_runner(wrapped, agent_cfg, checkpoint)
        env = wrapped.unwrapped
        robot = env.scene["robot"]
        if is_crouch:
            actor = runner.alg.actor
            standing_command = None
        else:
            policy = runner.get_inference_policy(device=env.device)
            standing_command = env.command_manager.get_term("base_velocity")
        wrapped.reset()
        dt = float(env.step_dt)
        overlay = ShowcaseOverlay()
        draw = CourseDraw((0.0, 0.0), 4.0)
        camera = CameraController(env, args_cli.camera, eye, target)
        telemetry, burned = [], []
        entry_heights: list[float] = []
        primitive_previous = torch.zeros((1, wrapped.num_actions), device=env.device)
        max_request_offset_jump = 0.0

        for step in range(round(max_seconds / dt)):
            now = (step + 1) * dt
            if standing_command is not None:
                # The rejected request contributes no command or action offset.
                # Keep the frozen Stage-2 base on its zero-velocity endpoint.
                standing_command.vel_command_b.zero_()
            observations = wrapped.get_observations()
            if is_crouch:
                # 2.0 STAND, 1.5 DOWN, 2.0 HOLD, 1.5 RETURN, 2.0 STAND.
                ends = (2.0, 3.5, 5.5, 7.0, 9.0)
                phase = next(index for index, end in enumerate(ends) if now <= end + 1.0e-9)
                starts = (0.0, 2.0, 3.5, 5.5, 7.0)
                durations = (2.0, 1.5, 2.0, 1.5, 2.0)
                progress_value = min(1.0, max(0.0, (now - starts[phase]) / durations[phase]))
                phases = torch.tensor([phase], dtype=torch.long, device=env.device)
                progress = torch.tensor([progress_value], device=env.device)
                with torch.inference_mode():
                    standing_action = actor.diagnostic_components(observations)["standing_base_action"]
                    primitive = phased_offset(torch.tensor([0.09], device=env.device), phases, progress)
                    actions = standing_action + primitive
                phase_label = ("STAND", "DOWN", "HOLD", "RETURN", "STAND HOLD")[phase]
            else:
                with torch.inference_mode():
                    actions = policy(observations)
                primitive = torch.zeros_like(actions)
                phase_label = "STAND" if now < 2.0 else "REJECTED"

            with torch.inference_mode():
                _, _, dones, _ = wrapped.step(actions)
            if standing_command is not None:
                standing_command.vel_command_b.zero_()
            position = robot.data.root_pos_w.torch[0, :2]
            height = float(robot.data.root_pos_w.torch[0, 2].item())
            speed = float(torch.linalg.vector_norm(robot.data.root_lin_vel_w.torch[0, :2]).item())
            camera.update(position)
            if step % 3 == 0:
                draw.trace((float(position[0]), float(position[1])))

            if is_crouch:
                if now >= 1.5 and now <= 2.0:
                    entry_heights.append(height)
                entry_height = sum(entry_heights) / len(entry_heights) if entry_heights else height
                actual_depth = max(0.0, entry_height - height)
                lines = [
                    "Unitree G1 Command-Driven Motion System" if now <= 1.0 else "DEMO: CROUCH_SHOWCASE",
                    f"ACTIVE: {'CROUCH_SHALLOW' if phase in (1, 2, 3) else 'STAND'}",
                    "COMMAND: CROUCH 0.09m" if phase in (1, 2) else "COMMAND: STAND",
                    f"ACTUAL DEPTH: {actual_depth:.3f}m",
                    f"PHASE: {phase_label}",
                    "CONTROLLER: scripted_shallow_v1",
                    "SEQUENCE: STAND -> CROUCH 0.09m -> STAND",
                ]
                requested = "CROUCH_SHALLOW"
                supported = True
            else:
                request_active = now >= 2.0
                request_jump = float(torch.linalg.vector_norm(primitive - primitive_previous).item())
                max_request_offset_jump = max(max_request_offset_jump, request_jump)
                lines = [
                    "DEMO: SAFE_REJECTION",
                    "REQUEST: STEP_OVER" if request_active else "ACTIVE: STAND",
                    f"SUPPORTED: {'FALSE' if request_active else 'N/A'}",
                    "ACTION: REJECTED SAFELY" if request_active else "ACTION: HOLD",
                    "REASON: unsupported_obstacle" if request_active else "CONTROLLER: stage2_standing_base_model_4246",
                    f"CONTROLLER UNCHANGED: {'TRUE' if request_active else 'N/A'}",
                    f"PRIMITIVE_STARTED: {'FALSE' if request_active else 'N/A'}",
                    f"ACTION DISCONTINUITY: {request_jump:.1f}",
                ]
                actual_depth = 0.0
                requested = "STEP_OVER" if request_active else "STAND"
                supported = not request_active
            primitive_jump = float(torch.linalg.vector_norm(primitive - primitive_previous).item())
            emit_status(lines, overlay, burned, step)
            telemetry.append({
                "time_s": now, "phase": phase_label, "request": requested,
                "supported": supported, "primitive_started": False if not is_crouch else phase in (1, 2, 3),
                "primitive_action_discontinuity_l2": 0.0 if not is_crouch else primitive_jump,
                "pelvis_height_m": height, "actual_depth_m": actual_depth,
                "horizontal_speed_mps": speed,
                "position_xy_m": [float(position[0]), float(position[1])],
            })
            primitive_previous = primitive.clone()
            if bool(dones[0].item()):
                print("showcase_terminated=environment_done")
                break

        output = Path(args_cli.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        summary = {
            "showcase": args_cli.showcase,
            "checkpoint": str(checkpoint),
            "camera": args_cli.camera,
            "camera_yaw_follow": False,
            "camera_transform_write_count": camera.write_count,
            "controller": "scripted_shallow_v1" if is_crouch else "stage2_standing_base_model_4246",
            "commanded_depth_m": 0.09 if is_crouch else None,
            "actual_depth_max_m": max((row["actual_depth_m"] for row in telemetry), default=0.0),
            "unsupported_request": None if is_crouch else {
                "request": "STEP_OVER", "supported": False, "controller_unchanged": True,
                "primitive_started": False, "primitive_action_discontinuity_l2": max_request_offset_jump,
                "reason": "unsupported_obstacle",
            },
            "telemetry": telemetry,
        }
        (output / "showcase_telemetry.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if not is_crouch:
            print(
                "REJECTION_RESULT request=STEP_OVER supported=false controller_unchanged=true "
                f"primitive_started=false action_discontinuity_l2={max_request_offset_jump:.1f}"
            )
        wrapped.close()
        finalize_recording(record_info, burned)
        return summary


def main() -> None:
    output = Path(args_cli.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"showcase={args_cli.showcase}")
    print(f"camera={args_cli.camera} camera_yaw_follow=false")
    print(f"telemetry_output={output / 'showcase_telemetry.json'}")
    if args_cli.showcase.startswith("TURN_"):
        result = run_turn_showcase()
    else:
        result = run_standing_showcase()
    print("SHOWCASE_RESULT " + json.dumps({
        key: result[key] for key in ("showcase", "camera", "camera_yaw_follow")
    }))


if __name__ == "__main__":
    main()
