"""Render the adopted exp_004 H3 gait package as a labelled MP4 showcase.

The video is a simulation demonstration, not new qualification evidence.  It
loads the verified release package, runs the same routed controller and final
target-safety pipeline used by the formal evaluator, and keeps the hardware
gate visibly PROHIBITED throughout the recording.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WORKSPACE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXP_ROOT = (
    WORKSPACE
    / "experiments"
    / "mujoco"
    / "exp_004_openduckmini_safe_gait_experts"
)
SCRIPTS = EXP_ROOT / "scripts"
PACKAGE_ROOT = (
    EXP_ROOT
    / "artifacts"
    / "router_packages"
    / "exp004-safe-gait-router-h3-release-20260808-v1"
)
DEFAULT_OUTPUT = HERE / "openduckmini_h3_motion_verified_showcase_v2_final.mp4"
DEFAULT_COVER = HERE / "openduckmini_h3_motion_verified_showcase_v2_final_cover.png"
DEFAULT_MANIFEST = HERE / "video_manifest_v2_final.json"

if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from package_manifest import load_and_validate_package, sha256_file  # noqa: E402
import evaluate_routed_transitions as routed_eval  # noqa: E402
from safe_gait_experts.routed_evaluation import segment_acceptance  # noqa: E402


PACKAGE_ID = "exp004-safe-gait-router-h3-release-20260808-v1"
STATUS = "ADOPTED_SIMULATION_ONLY"
HARDWARE_GATE = "PROHIBITED"
RELEASE_EVIDENCE_SHA256 = (
    "95819b5bc1d0827a5ad779542a6f98c4aaebacf5f55a8303c0b5a14fba501674"
)
BASE_V22_SHA256 = (
    "f7a2731330cd3be52858989b021423a5f363cc4a8f9850512281da745a7617c0"
)


# Each clip uses the exact physical->policy observation mapping frozen by the
# adopted evaluator.  Clips are intentionally independent exact-home resets:
# this is a motion reel, while the 20x30 release evidence remains the source of
# transition qualification.  Six seconds is the shortest common screened
# horizon at which all twelve moving cases pass the central acceptance gate with the
# adopted controller and a 1.5 s metric warmup.
SHOWCASE_SEGMENTS: tuple[tuple[Any, ...], ...] = (
    (
        "forward",
        "前進 / FORWARD",
        (0.05, 0.0, 0.0),
        (0.10, 0.0, 0.0),
        6.0,
        "forward",
        "forward",
    ),
    ("reverse", "後進 / REVERSE", (-0.05, 0.0, 0.0), None, 6.0, "reverse", "reverse"),
    (
        "lateral_left",
        "左横歩き / LATERAL LEFT",
        (0.0, 0.06, 0.0),
        (0.0, 0.10, 0.0),
        6.0,
        "lateral_left",
        "lateral_left",
    ),
    (
        "lateral_right",
        "右横歩き / LATERAL RIGHT",
        (0.0, -0.06, 0.0),
        (0.0, -0.10, 0.0),
        6.0,
        "lateral_right",
        "lateral_right",
    ),
    (
        "yaw_left",
        "左旋回 / YAW LEFT",
        (0.0, 0.0, 0.30),
        (0.0, -0.06, 0.60),
        6.0,
        "yaw_left",
        "yaw_left",
    ),
    (
        "yaw_right",
        "右旋回 / YAW RIGHT",
        (0.0, 0.0, -0.30),
        (0.0, 0.0, -0.80),
        6.0,
        "yaw_right",
        "yaw_right",
    ),
    (
        "reverse_turn_left",
        "後進左旋回 / REVERSE TURN LEFT",
        (-0.03, 0.0, 0.20),
        None,
        6.0,
        "reverse_turn_left",
        "compound",
    ),
    (
        "reverse_turn_right",
        "後進右旋回 / REVERSE TURN RIGHT",
        (-0.04, 0.0, -0.20),
        None,
        6.0,
        "reverse_turn_right",
        "compound",
    ),
    (
        "forward_turn_left",
        "前進左旋回 / FORWARD TURN LEFT",
        (0.04, 0.0, 0.30),
        (0.08, 0.0, 0.30),
        6.0,
        "compound",
        "compound",
    ),
    (
        "forward_turn_right",
        "前進右旋回 / FORWARD TURN RIGHT",
        (0.04, 0.0, -0.22),
        (0.08, 0.0, -0.45),
        6.0,
        "compound",
        "compound",
    ),
    (
        "forward_lateral_left_turn",
        "前進左斜行旋回 / FORWARD LATERAL LEFT TURN",
        (0.04, 0.05, 0.17),
        (0.06, 0.05, 0.20),
        6.0,
        "compound",
        "compound",
    ),
    (
        "forward_lateral_right_turn",
        "前進右斜行旋回 / FORWARD LATERAL RIGHT TURN",
        (0.04, -0.03, -0.15),
        (0.06, -0.05, -0.35),
        6.0,
        "compound",
        "compound",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cover", type=Path, default=DEFAULT_COVER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Render only a short stand/forward technical check.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("width, height, and fps must be positive")
    if 500 % args.fps:
        parser.error("fps must divide the 500 Hz MuJoCo substep rate")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidate = Path(
        "/mnt/c/Windows/Fonts/meiryob.ttc"
        if bold
        else "/mnt/c/Windows/Fonts/meiryo.ttc"
    )
    if candidate.is_file():
        return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)


def make_card(
    width: int,
    height: int,
    eyebrow: str,
    title: Sequence[str],
    body: Sequence[str],
) -> np.ndarray:
    top = np.asarray((7, 15, 28), dtype=np.float64)
    bottom = np.asarray((14, 33, 52), dtype=np.float64)
    gradient = np.linspace(0.0, 1.0, height, dtype=np.float64)[:, None, None]
    pixels = np.broadcast_to(
        top[None, None, :] * (1.0 - gradient) + bottom[None, None, :] * gradient,
        (height, width, 3),
    ).astype(np.uint8).copy()
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image, "RGBA")
    margin = int(width * 0.075)
    draw.rounded_rectangle(
        (margin, int(height * 0.09), width - margin, int(height * 0.91)),
        radius=30,
        fill=(13, 29, 48, 235),
        outline=(255, 187, 64, 210),
        width=2,
    )
    draw.rectangle(
        (margin, int(height * 0.09), margin + 12, int(height * 0.91)),
        fill=(255, 187, 64, 255),
    )
    draw.text(
        (margin + 52, int(height * 0.16)),
        eyebrow,
        font=font(max(20, width // 48), bold=True),
        fill=(255, 193, 75, 255),
    )
    y = int(height * 0.28)
    for line in title:
        title_size = max(36, width // 22)
        title_font = font(title_size, bold=True)
        title_width_limit = width - 2 * margin - 105
        while (
            title_size > 28
            and draw.textbbox((0, 0), line, font=title_font)[2]
            > title_width_limit
        ):
            title_size -= 1
            title_font = font(title_size, bold=True)
        draw.text(
            (margin + 50, y),
            line,
            font=title_font,
            fill=(248, 250, 252, 255),
        )
        y += max(52, height // 10)
    y += max(12, height // 40)
    for line in body:
        draw.text(
            (margin + 52, y),
            line,
            font=font(max(20, width // 50)),
            fill=(185, 206, 226, 255),
        )
        y += max(34, height // 18)
    draw.text(
        (margin + 52, int(height * 0.84)),
        "MuJoCo simulation only  |  Hardware deployment: PROHIBITED",
        font=font(max(17, width // 62)),
        fill=(142, 168, 193, 255),
    )
    return np.asarray(image)


def append_still(writer: Any, frame: np.ndarray, seconds: float, fps: int) -> None:
    frame_count = int(round(seconds * fps))
    fade_frames = max(1, int(round(0.30 * fps)))
    black = np.zeros_like(frame)
    for index in range(frame_count):
        alpha = 1.0
        if index < fade_frames:
            alpha = (index + 1) / fade_frames
        elif index >= frame_count - fade_frames:
            alpha = (frame_count - index) / fade_frames
        blended = (
            frame.astype(np.float32) * alpha
            + black.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
        writer.append_data(blended)


class ShowcaseRecorder:
    def __init__(
        self,
        *,
        renderer: Any,
        camera: Any,
        writer: Any,
        evaluator: Any,
        schedule: Sequence[tuple[Any, ...]],
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.renderer = renderer
        self.camera = camera
        self.writer = writer
        self.evaluator = evaluator
        self.schedule = tuple(schedule)
        self.width = width
        self.height = height
        self.fps = fps
        self.render_every_substeps = 500 // fps
        self.substep_count = 0
        self.frame_count = 0
        self.segment_ends = np.cumsum([item[4] for item in self.schedule]).tolist()
        self.segment_starts = [0.0, *self.segment_ends[:-1]]
        self.segment_index = -1
        self.segment_start_position = np.zeros(3, dtype=np.float64)
        self.initial_position: np.ndarray | None = None
        self.initial_yaw_rad: float | None = None
        self.trajectory_xy: list[np.ndarray] = []

    def _segment_for_time(self, time_seconds: float) -> tuple[int, Any]:
        index = min(
            bisect_right(self.segment_ends, max(0.0, time_seconds - 1e-9)),
            len(self.schedule) - 1,
        )
        return index, self.schedule[index]

    def capture(self, data: Any) -> None:
        self.substep_count += 1
        if self.substep_count % self.render_every_substeps:
            return
        index, segment = self._segment_for_time(float(data.time))
        # Execute the full 5 s formally qualified stand transition, while
        # encoding only its first/last 0.5 s to keep the showcase concise.
        local_time = float(data.time) - self.segment_starts[index]
        duration = float(segment[4])
        del local_time, duration
        trunk = np.asarray(data.xpos[self.evaluator.trunk_body_id], dtype=np.float64)
        if self.initial_position is None:
            self.initial_position = trunk.copy()
            rotation = np.asarray(
                data.xmat[self.evaluator.trunk_body_id], dtype=np.float64
            ).reshape(3, 3)
            self.initial_yaw_rad = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            # Keep the world camera fixed at the reset point.  Tracking the
            # trunk made the superseded video hide translation.
            self.camera.lookat[:] = np.asarray(
                (trunk[0], trunk[1], 0.180), dtype=np.float64
            )
        if index != self.segment_index:
            self.segment_index = index
            self.segment_start_position = trunk.copy()
        self.trajectory_xy.append((trunk[:2] - self.segment_start_position[:2]).copy())
        self.renderer.update_scene(data, camera=self.camera)
        raw = self.renderer.render().copy()
        frame = self._overlay(raw, data, segment)
        self.writer.append_data(frame)
        self.frame_count += 1

    def _overlay(self, frame: np.ndarray, data: Any, segment: Any) -> np.ndarray:
        name, label, command, policy_command, _, route, _ = segment
        del name
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")
        top_height = 112
        draw.rectangle((0, 0, self.width, top_height), fill=(4, 12, 23, 205))
        title_size = max(24, self.width // 36)
        title_font = font(title_size, bold=True)
        title_max_width = self.width - 390
        while (
            title_size > 22
            and draw.textbbox((0, 0), label, font=title_font)[2] > title_max_width
        ):
            title_size -= 1
            title_font = font(title_size, bold=True)
        draw.text((28, 16), label, font=title_font, fill=(255, 255, 255, 255))
        draw.text(
            (30, 68),
            (
                f"physical command   vx={command[0]:+.2f} m/s   "
                f"vy={command[1]:+.2f} m/s   yaw={command[2]:+.2f} rad/s"
            ),
            font=font(max(16, self.width // 62)),
            fill=(194, 216, 238, 255),
        )
        policy_text = "same as physical" if policy_command is None else (
            f"vx={policy_command[0]:+.2f}  vy={policy_command[1]:+.2f}  "
            f"yaw={policy_command[2]:+.2f}"
        )
        draw.text(
            (30, 91),
            f"policy observation   {policy_text}",
            font=font(max(13, self.width // 78)),
            fill=(145, 188, 224, 255),
        )
        badge = f"route: {route}"
        box = draw.textbbox((0, 0), badge, font=font(max(16, self.width // 62), bold=True))
        badge_width = box[2] - box[0] + 36
        draw.rounded_rectangle(
            (self.width - badge_width - 24, 25, self.width - 24, 72),
            radius=18,
            fill=(255, 187, 64, 225),
        )
        draw.text(
            (self.width - badge_width - 6, 36),
            badge,
            font=font(max(16, self.width // 62), bold=True),
            fill=(11, 23, 38, 255),
        )

        trunk = np.asarray(data.xpos[self.evaluator.trunk_body_id], dtype=np.float64)
        segment_delta = trunk - self.segment_start_position
        total_delta = (
            np.zeros(3, dtype=np.float64)
            if self.initial_position is None
            else trunk - self.initial_position
        )
        rotation = np.asarray(
            data.xmat[self.evaluator.trunk_body_id], dtype=np.float64
        ).reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        initial_yaw = yaw if self.initial_yaw_rad is None else self.initial_yaw_rad
        yaw_delta = float(np.arctan2(np.sin(yaw - initial_yaw), np.cos(yaw - initial_yaw)))
        panel_x0 = self.width - 310
        panel_y0 = self.height - 138
        draw.rounded_rectangle(
            (panel_x0, panel_y0, self.width - 22, self.height - 28),
            radius=18,
            fill=(4, 12, 23, 185),
            outline=(151, 180, 207, 110),
            width=1,
        )
        draw.text(
            (panel_x0 + 18, panel_y0 + 14),
            "SIM DISPLACEMENT",
            font=font(max(14, self.width // 72), bold=True),
            fill=(255, 193, 75, 255),
        )
        draw.text(
            (panel_x0 + 18, panel_y0 + 48),
            f"segment  dx={segment_delta[0]:+.3f}  dy={segment_delta[1]:+.3f} m",
            font=font(max(13, self.width // 80)),
            fill=(220, 232, 244, 255),
        )
        draw.text(
            (panel_x0 + 18, panel_y0 + 76),
            f"yaw delta={np.degrees(yaw_delta):+.1f} deg",
            font=font(max(13, self.width // 80)),
            fill=(177, 201, 224, 255),
        )

        # Fixed-scale top-view trail: the origin never follows the robot.
        map_x0, map_y0, map_w, map_h = 22, self.height - 188, 270, 150
        draw.rounded_rectangle(
            (map_x0, map_y0, map_x0 + map_w, map_y0 + map_h),
            radius=16,
            fill=(4, 12, 23, 185),
            outline=(151, 180, 207, 110),
            width=1,
        )
        draw.text(
            (map_x0 + 14, map_y0 + 10),
            "WORLD TRACE / 0.50 m SPAN",
            font=font(max(11, self.width // 100), bold=True),
            fill=(255, 193, 75, 255),
        )
        origin = np.asarray((map_x0 + map_w / 2, map_y0 + map_h / 2 + 14))
        scale = min(map_w, map_h - 30) / 0.50
        draw.line((map_x0 + 10, origin[1], map_x0 + map_w - 10, origin[1]), fill=(95, 125, 153, 150), width=1)
        draw.line((origin[0], map_y0 + 36, origin[0], map_y0 + map_h - 8), fill=(95, 125, 153, 150), width=1)
        points = [
            (float(origin[0] + delta[0] * scale), float(origin[1] - delta[1] * scale))
            for delta in self.trajectory_xy
        ]
        if len(points) >= 2:
            draw.line(points, fill=(80, 220, 255, 255), width=4, joint="curve")
        if points:
            x, y = points[-1]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 187, 64, 255))

        footer_height = 38
        draw.rectangle(
            (0, self.height - footer_height, self.width, self.height),
            fill=(5, 13, 24, 225),
        )
        draw.text(
            (20, self.height - 30),
            "H3 release  |  ADOPTED_SIMULATION_ONLY  |  実機投入禁止 / HARDWARE PROHIBITED",
            font=font(max(14, self.width // 76), bold=True),
            fill=(255, 213, 125, 255),
        )
        return np.asarray(image)


def audit_showcase_result(result: dict[str, Any]) -> dict[str, Any]:
    if result["fell"]:
        raise RuntimeError("showcase simulation fell")
    if result["completed_segment_count"] != result["requested_segment_count"]:
        raise RuntimeError("showcase did not complete every segment")
    totals = {
        "segments": 0,
        "control_samples": 0,
        "physics_substeps": 0,
        "leg_joint_samples": 0,
        "qpos_limit_violations": 0,
        "target_limit_violations": 0,
        "unauthorized_target_margin_violations": 0,
        "target_slew_violations": 0,
        "nonfinite_samples": 0,
        "fall_samples": 0,
        "prohibited_route_steps": 0,
        "command_clip_events": 0,
        "maximum_qpos_excess_rad": 0.0,
        "central_motion_acceptance_passed": 0,
        "minimum_signed_progress_ratio": float("inf"),
    }
    for segment in result["segments"]:
        safety = segment["safety_audit"]
        physics = segment["physics_substep_audit"]
        routing = segment["routing"]
        if not segment["completed"] or segment["fell"]:
            raise RuntimeError(f"incomplete showcase segment: {segment['name']}")
        acceptance = segment_acceptance(segment)
        if not acceptance["passed"]:
            failed_checks = [
                name for name, passed in acceptance["checks"].items() if not passed
            ]
            raise RuntimeError(
                f"motion acceptance failed for {segment['name']}: {failed_checks}"
            )
        if physics["sample_count"] != segment["completed_physics_substeps"]:
            raise RuntimeError(f"substep audit mismatch: {segment['name']}")
        totals["segments"] += 1
        totals["central_motion_acceptance_passed"] += 1
        totals["control_samples"] += int(safety["sample_count"])
        totals["physics_substeps"] += int(physics["sample_count"])
        totals["leg_joint_samples"] += int(physics["leg_joint_sample_count"])
        totals["qpos_limit_violations"] += int(safety["qpos_limit_violations"])
        totals["qpos_limit_violations"] += int(physics["qpos_limit_violations"])
        totals["target_limit_violations"] += int(
            safety["applied_target_limit_violations"]
        )
        totals["target_limit_violations"] += int(
            safety["desired_target_margin_violations"]
        )
        totals["unauthorized_target_margin_violations"] += int(
            safety["unauthorized_applied_target_margin_violations"]
        )
        totals["target_slew_violations"] += int(safety["target_slew_violations"])
        totals["nonfinite_samples"] += int(safety["nonfinite_sample_count"])
        totals["nonfinite_samples"] += int(physics["nonfinite_state_samples"])
        totals["fall_samples"] += int(physics["height_fall_samples"])
        totals["fall_samples"] += int(physics["upright_fall_samples"])
        totals["prohibited_route_steps"] += int(routing["prohibited_expert_steps"])
        totals["command_clip_events"] += int(routing["command_clip_events"])
        totals["maximum_qpos_excess_rad"] = max(
            totals["maximum_qpos_excess_rad"],
            float(safety["maximum_qpos_excess_rad"]),
            float(physics["maximum_qpos_excess_rad"]),
        )
        command = np.asarray(segment["command"], dtype=np.float64)
        metrics = segment["metrics"]
        ratios: list[float] = []
        commanded_linear_speed = float(np.linalg.norm(command[:2]))
        if commanded_linear_speed > 0.0:
            ratios.append(
                float(metrics["projected_primary_velocity"])
                / commanded_linear_speed
            )
        if abs(float(command[2])) > 0.0:
            ratios.append(
                abs(float(metrics["mean_local_yaw_rate"])) / abs(float(command[2]))
            )
        if ratios:
            totals["minimum_signed_progress_ratio"] = min(
                float(totals["minimum_signed_progress_ratio"]), min(ratios)
            )
        if safety["head_target_peak_rad"] != 0.0:
            raise RuntimeError(f"nonzero head target: {segment['name']}")
        if safety["applied_head_action_peak"] != 0.0:
            raise RuntimeError(f"nonzero head action: {segment['name']}")
    required_zero = (
        "qpos_limit_violations",
        "target_limit_violations",
        "unauthorized_target_margin_violations",
        "target_slew_violations",
        "nonfinite_samples",
        "fall_samples",
        "prohibited_route_steps",
        "command_clip_events",
    )
    failures = {key: totals[key] for key in required_zero if totals[key] != 0}
    if failures:
        raise RuntimeError(f"showcase safety audit failed: {failures}")
    if not result["reset_qpos_audit"]["passed"]:
        raise RuntimeError("showcase reset audit failed")
    if not result["control_first_startup_audit"]["passed"]:
        raise RuntimeError("showcase startup audit failed")
    if not result["backward_exit_recovery_audit"]["passed"]:
        raise RuntimeError("showcase recovery audit failed")
    totals["reset_audit_passed"] = True
    totals["control_first_startup_audit_passed"] = True
    totals["backward_exit_recovery_audit_passed"] = True
    totals["backward_exit_recovery_events"] = int(
        result["backward_exit_recovery_audit"]["exit_event_count"]
    )
    totals["backward_exit_recovery_active_ticks"] = int(
        result["backward_exit_recovery_audit"]["active_tick_count"]
    )
    if totals["minimum_signed_progress_ratio"] == float("inf"):
        totals["minimum_signed_progress_ratio"] = None
    return totals


def build_showcase(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    package = load_and_validate_package(PACKAGE_ROOT)
    if package["package_id"] != PACKAGE_ID:
        raise ValueError("unexpected package id")
    if package["safety"]["hardware_deployment"] != HARDWARE_GATE:
        raise ValueError("hardware gate must remain PROHIBITED")
    formal_gate = package["safety"]["formal_release_gate"]
    if formal_gate["status"] != STATUS:
        raise ValueError("unexpected release status")
    if formal_gate["formal_evidence"]["sha256"] != RELEASE_EVIDENCE_SHA256:
        raise ValueError("unexpected release evidence hash")

    policy_path = PACKAGE_ROOT / "models" / "base_v22.onnx"
    packaged_scene_path = (
        PACKAGE_ROOT
        / "simulation"
        / "xmls"
        / "scene_flat_terrain_backlash_hardware_safe_calibrated.xml"
    )
    packaged_reference_path = PACKAGE_ROOT / "simulation" / "reference.pkl"
    straight_profile = PACKAGE_ROOT / "corrections" / "optimized_backward_gait.json"
    left_profile = (
        PACKAGE_ROOT / "corrections" / "optimized_backward_left_turn_gait.json"
    )
    right_profile = (
        PACKAGE_ROOT / "corrections" / "optimized_backward_right_turn_gait.json"
    )
    if sha256_file(policy_path) != BASE_V22_SHA256:
        raise ValueError("base-v22 hash mismatch")
    # The reused official evaluator discovers two initialization data files
    # relative to the scene.  Use the formally validated generated mirror for
    # that layout, after proving its scene/reference bytes are identical to the
    # packaged copies.  The executed correction profiles still come directly
    # from the verified package below.
    generated_root = EXP_ROOT / "artifacts" / "generated_playground"
    routed_eval.validate_exact_generated_assets(generated_root)
    generated_assets = routed_eval.generated_asset_paths(generated_root.resolve())
    scene_path = generated_assets["scene"]
    reference_path = generated_assets["reference"]
    if sha256_file(scene_path) != sha256_file(packaged_scene_path):
        raise ValueError("generated and packaged scene hashes differ")
    if sha256_file(reference_path) != sha256_file(packaged_reference_path):
        raise ValueError("generated and packaged reference hashes differ")
    straight_payload = json.loads(straight_profile.read_text(encoding="utf-8"))
    left_knee_extra_margin = float(
        straight_payload["composition"]["left_knee_extra_upper_margin_rad"]
    )
    if left_knee_extra_margin != 0.0125:
        raise ValueError("unexpected profile left-knee cap")

    mujoco, onnxruntime, runtime = routed_eval._load_runtime()
    policies = {role: policy_path for role in routed_eval.REQUIRED_POLICY_ROLES}
    bank = routed_eval.RoutedPolicyBank(policies, onnxruntime)
    evaluator = runtime.OfficialPolicyEvaluator(scene_path, policy_path, reference_path)
    evaluator.backward_residual_scale = 0.0
    evaluator.load_backward_profile(straight_profile)
    evaluator.load_backward_turn_profile(1, left_profile)
    evaluator.load_backward_turn_profile(-1, right_profile)
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0
    routed_eval.validate_model_contract(evaluator)
    simulator = routed_eval.RoutedSimulator(
        evaluator,
        bank,
        mujoco,
        runtime,
        leg_target_margin_rad=0.05,
        target_slew_rate_rad_s=2.0,
        diagnostic_noncontract_safety=False,
        left_knee_extra_upper_margin_rad=left_knee_extra_margin,
        formal_candidate_default=True,
    )

    chosen_segments = SHOWCASE_SEGMENTS if not args.smoke else SHOWCASE_SEGMENTS[:1]

    args.output = args.output.resolve()
    args.cover = args.cover.resolve()
    args.manifest = args.manifest.resolve()
    for path in (args.output, args.cover, args.manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}")
    partial_output = args.output.with_name(args.output.stem + ".partial.mp4")
    if partial_output.exists():
        raise FileExistsError(f"refusing stale partial output {partial_output}")

    cover = make_card(
        args.width,
        args.height,
        "OpenDuckMini / exp_004 H3 motion-verified reel",
        ("12動作すべて中央判定PASS", "H3 MOTION SHOWCASE V2"),
        (
            "正式physical→policy mapping / 各動作exact-home独立開始",
            "6.0 s motion / 1.5 s metric warmup / progress >= 30%",
            "固定世界カメラ・軌跡・実変位・yaw変化を表示",
        ),
    )
    Image.fromarray(cover).save(args.cover)

    # The formal scene ships with a conservative 640x480 offscreen buffer.
    # Enlarging only the renderer framebuffer does not alter physics state or
    # controller execution, and allows a native 720p deliverable.
    evaluator.model.vis.global_.offwidth = max(
        int(evaluator.model.vis.global_.offwidth), args.width
    )
    evaluator.model.vis.global_.offheight = max(
        int(evaluator.model.vis.global_.offheight), args.height
    )
    renderer = mujoco.Renderer(evaluator.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.20
    camera.azimuth = 135.0
    camera.elevation = -17.0
    original_step = routed_eval.apply_guarded_control_then_step_physics
    motion_results: list[dict[str, Any]] = []
    motion_summaries: list[dict[str, Any]] = []
    recorders: list[ShowcaseRecorder] = []
    try:
        with imageio.get_writer(
            partial_output,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            append_still(writer, cover, 2.8 if not args.smoke else 0.4, args.fps)
            for clip_index, clip in enumerate(chosen_segments, start=1):
                name, label, physical, policy_command, duration, route, policy_role = clip
                label_lines = tuple(part.strip() for part in label.split("/", 1))
                intro = make_card(
                    args.width,
                    args.height,
                    f"MOTION {clip_index:02d} / {len(chosen_segments):02d}",
                    (*label_lines, "INDEPENDENT EXACT-HOME RESET"),
                    (
                        f"physical vx={physical[0]:+.2f}  vy={physical[1]:+.2f}  yaw={physical[2]:+.2f}",
                        "中央acceptance判定 / fixed-world camera / all-substep safety audit",
                    ),
                )
                append_still(writer, intro, 0.55 if not args.smoke else 0.2, args.fps)

                recorder = ShowcaseRecorder(
                    renderer=renderer,
                    camera=camera,
                    writer=writer,
                    evaluator=evaluator,
                    schedule=(clip,),
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                )
                recorders.append(recorder)

                def recorded_step(*step_args: Any, **step_kwargs: Any) -> Any:
                    original_callback = step_kwargs.get("physics_substep_callback")
                    data = step_kwargs["data"]

                    def combined_callback() -> bool:
                        terminated = (
                            bool(original_callback()) if original_callback else False
                        )
                        recorder.capture(data)
                        return terminated

                    step_kwargs["physics_substep_callback"] = combined_callback
                    return original_step(*step_args, **step_kwargs)

                routed_eval.apply_guarded_control_then_step_physics = recorded_step
                schedule_case = (
                    name,
                    physical,
                    duration,
                    policy_command,
                    route,
                    policy_role,
                )
                try:
                    result = simulator.run_schedule(
                        (schedule_case,),
                        seed=args.seed,
                        joint_noise_scale=0.0,
                        initial_base_speed=0.0,
                        warmup_seconds=1.5,
                    )
                finally:
                    routed_eval.apply_guarded_control_then_step_physics = original_step
                summary = audit_showcase_result(result)
                motion_results.append(result)
                motion_summaries.append(summary)

                segment = result["segments"][0]
                metrics = segment["metrics"]
                command = np.asarray(segment["command"], dtype=np.float64)
                ratio_parts: list[str] = []
                linear_speed = float(np.linalg.norm(command[:2]))
                if linear_speed > 0.0:
                    ratio_parts.append(
                        f"linear {float(metrics['projected_primary_velocity']) / linear_speed:.3f}"
                    )
                if abs(float(command[2])) > 0.0:
                    ratio_parts.append(
                        f"yaw {abs(float(metrics['mean_local_yaw_rate'])) / abs(float(command[2])):.3f}"
                    )
                passed = make_card(
                    args.width,
                    args.height,
                    "CENTRAL MOTION ACCEPTANCE",
                    (*label_lines, "PASS"),
                    (
                        f"progress ratio: {' / '.join(ratio_parts)}   (required >= 0.300)",
                        f"{int(segment['completed_physics_substeps']):,} physics substeps / safety violations 0",
                    ),
                )
                append_still(writer, passed, 0.65 if not args.smoke else 0.2, args.fps)

            summary = {
                "segments": sum(int(item["segments"]) for item in motion_summaries),
                "central_motion_acceptance_passed": sum(
                    int(item["central_motion_acceptance_passed"])
                    for item in motion_summaries
                ),
                "control_samples": sum(
                    int(item["control_samples"]) for item in motion_summaries
                ),
                "physics_substeps": sum(
                    int(item["physics_substeps"]) for item in motion_summaries
                ),
                "leg_joint_samples": sum(
                    int(item["leg_joint_samples"]) for item in motion_summaries
                ),
                "minimum_signed_progress_ratio": min(
                    float(item["minimum_signed_progress_ratio"])
                    for item in motion_summaries
                ),
                "qpos_limit_violations": sum(
                    int(item["qpos_limit_violations"]) for item in motion_summaries
                ),
                "target_limit_violations": sum(
                    int(item["target_limit_violations"]) for item in motion_summaries
                ),
                "unauthorized_target_margin_violations": sum(
                    int(item["unauthorized_target_margin_violations"])
                    for item in motion_summaries
                ),
                "target_slew_violations": sum(
                    int(item["target_slew_violations"])
                    for item in motion_summaries
                ),
                "nonfinite_samples": sum(
                    int(item["nonfinite_samples"]) for item in motion_summaries
                ),
                "fall_samples": sum(
                    int(item["fall_samples"]) for item in motion_summaries
                ),
                "prohibited_route_steps": sum(
                    int(item["prohibited_route_steps"]) for item in motion_summaries
                ),
                "command_clip_events": sum(
                    int(item["command_clip_events"]) for item in motion_summaries
                ),
                "maximum_qpos_excess_rad": max(
                    float(item["maximum_qpos_excess_rad"])
                    for item in motion_summaries
                ),
                "reset_audits_passed": len(motion_summaries),
                "control_first_startup_audits_passed": len(motion_summaries),
                "backward_exit_recovery_audits_passed": len(motion_summaries),
            }
            close = make_card(
                args.width,
                args.height,
                "OpenDuckMini / H3 simulation result",
                (
                    f"{summary['segments']} / {len(chosen_segments)} 動作 合格",
                    "MOTION SHOWCASE COMPLETE",
                ),
                (
                    f"{summary['segments']} central acceptances / {summary['physics_substeps']:,} audited substeps",
                    f"worst progress ratio {summary['minimum_signed_progress_ratio']:.3f} >= 0.300",
                    "Falls / qpos / target / slew / route safety violations: 0",
                ),
            )
            append_still(writer, close, 3.0 if not args.smoke else 0.4, args.fps)
    except Exception:
        if partial_output.exists():
            partial_output.unlink()
        raise
    finally:
        routed_eval.apply_guarded_control_then_step_physics = original_step
        renderer.close()

    if not motion_results or not motion_summaries or not recorders:
        raise RuntimeError("showcase render produced no result")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        args.output.unlink()
    partial_output.replace(args.output)

    payload = {
        "title": "OpenDuckMini exp_004 H3 motion-verified showcase V2",
        "video": str(args.output.relative_to(WORKSPACE)),
        "video_sha256": file_sha256(args.output),
        "video_size_bytes": args.output.stat().st_size,
        "cover": str(args.cover.relative_to(WORKSPACE)),
        "cover_sha256": file_sha256(args.cover),
        "render": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "simulation_seconds": sum(float(item[4]) for item in chosen_segments),
            "title_and_close_seconds": (
                5.8 + 1.2 * len(chosen_segments)
                if not args.smoke
                else 0.8 + 0.4 * len(chosen_segments)
            ),
            "simulation_frame_count": sum(item.frame_count for item in recorders),
            "offscreen_backend": os.environ.get("MUJOCO_GL", ""),
            "camera_reference": "world_fixed_at_exact_home_reset",
        },
        "disclosure": {
            "simulation_only": True,
            "release_status": STATUS,
            "hardware_deployment": HARDWARE_GATE,
            "not_new_qualification_evidence": True,
            "independent_exact_home_reset_per_motion": True,
            "joint_noise_scale": 0.0,
            "initial_base_speed": 0.0,
            "metric_warmup_seconds": 1.5,
            "formal_policy_observation_mapping_used": True,
            "continuous_transition_claimed": False,
        },
        "package": {
            "path": str(PACKAGE_ROOT.relative_to(WORKSPACE)),
            "package_id": package["package_id"],
            "manifest_sha256": file_sha256(PACKAGE_ROOT / "package_manifest.json"),
            "formal_release_evidence_sha256": RELEASE_EVIDENCE_SHA256,
            "base_v22_sha256": BASE_V22_SHA256,
            "verified_before_render": True,
        },
        "showcase_audit": summary,
        "segments": [
            {
                "name": clip[0],
                "label": clip[1],
                "physical_command": list(clip[2]),
                "policy_observation_command": (
                    None if clip[3] is None else list(clip[3])
                ),
                "duration_seconds": clip[4],
                "metric_warmup_seconds": 1.5,
                "expected_route": clip[5],
                "expected_policy_role": clip[6],
                "central_acceptance": segment_acceptance(
                    motion_results[index]["segments"][0]
                ),
                "metrics": motion_results[index]["segments"][0]["metrics"],
            }
            for index, clip in enumerate(chosen_segments)
        ],
        "source": {
            "builder": str(Path(__file__).resolve().relative_to(WORKSPACE)),
            "builder_sha256": file_sha256(Path(__file__).resolve()),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True
            ).strip(),
        },
    }
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    payload = build_showcase(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
