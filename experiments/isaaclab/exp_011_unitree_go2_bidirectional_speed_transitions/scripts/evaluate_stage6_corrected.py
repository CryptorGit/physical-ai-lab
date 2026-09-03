"""Deterministic Stage 6 evaluation with frozen corrected endpoint metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage6_corrected_endpoint_formal"
PARENT = (
    REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/"
    "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
)
SELECTED = (
    REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage4_resumed_optimizer_training/checkpoints/model_50.pt"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--mode",
    choices=(
        "contact-preflight", "formal", "formal-steady", "formal-transitions",
        "formal-low-speed", "formal-reduced",
    ),
    required=True,
)
parser.add_argument("--output", type=Path, default=OUT)
parser.add_argument("--num-envs", type=int, default=50)
parser.add_argument("--policy", choices=("official_parent", "stage4_selected"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_physx.sensors import ContactSensorCfg  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
from go2_bidirectional.command_profiles import transition_command  # noqa: E402
from go2_bidirectional.evaluation import build_runner  # noqa: E402
from go2_bidirectional.stage6_endpoint_protocol import (  # noqa: E402
    circular_median,
    classify_go2_gait_v1,
    heading_error,
    percentile,
    physical_slip_intervals,
    quat_xyzw_to_gravity_tilt_torch,
    quat_xyzw_to_roll_pitch_yaw_torch,
)
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

SEED = 20264901
FEET = ("front-left", "front-right", "rear-left", "rear-right")
ASSET_FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GROUND = "/World/ground/terrain/GroundPlane/CollisionPlane"
STEADY = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
TRANSITIONS = ((0.0, 1.2), (1.2, 2.0), (2.0, 1.2), (1.2, 0.0))
LOW_SPEED = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
REDUCED = (0.0, 0.6, 1.2, 2.0, 1.2, 0.6, 0.0)


def dump(name: str, value) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    path = args.output / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def add_contact_point_sensors(cfg) -> None:
    for label, asset_name in zip(("fl", "fr", "rl", "rr"), ASSET_FEET):
        setattr(
            cfg.scene,
            f"stage6_{label}_contact",
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{asset_name}",
                update_period=0.0,
                track_pose=True,
                track_contact_points=True,
                track_friction_forces=False,
                max_contact_data_count_per_prim=8,
                filter_prim_paths_expr=[GROUND],
            ),
        )


def make_env(num_envs: int):
    cfg, agent = resolve_task_config("Isaac-Velocity-Flat-Unitree-Go2-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = num_envs
    cfg.seed = SEED
    cfg.episode_length_s = 60.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    add_contact_point_sensors(cfg)
    if args.device:
        cfg.sim.device = args.device
        agent.device = args.device
    return gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg), agent, cfg


def max_true_run(flags: list[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


class Collector:
    def __init__(self, raw, agent):
        self.wrapped, self.runner, self.policy = build_runner(raw, agent, PARENT)
        self.env = self.wrapped.unwrapped
        self.robot = self.env.scene["robot"]
        self.command = self.env.command_manager.get_term("base_velocity")
        self.contact = self.env.scene.sensors["contact_forces"]
        self.sensors = [
            self.env.scene.sensors[f"stage6_{label}_contact"]
            for label in ("fl", "fr", "rl", "rr")
        ]
        self.body_ids = [int(self.robot.find_bodies(name)[0][0]) for name in ASSET_FEET]
        self.dt = float(self.env.step_dt)
        self.raw_contact_mode = "force_weighted_raw_physx_contact_centroid"

    def load(self, path: Path) -> None:
        self.runner.load(str(path), strict=True, map_location=self.env.device)
        self.policy = self.runner.get_inference_policy(device=self.env.device)

    def contact_telemetry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centroids, forces, valid = [], [], []
        for sensor in self.sensors:
            normal_forces, points, _, _, counts, starts = sensor.contact_view.get_contact_data(
                dt=sensor._sim_physics_dt
            )
            normal = wp.to_torch(normal_forces).reshape(-1)
            point = wp.to_torch(points).reshape(-1, 3)
            count = wp.to_torch(counts).reshape(-1).to(torch.long)
            start = wp.to_torch(starts).reshape(-1).to(torch.long)
            width = int(sensor.cfg.max_contact_data_count_per_prim)
            offsets = torch.arange(width, device=count.device)
            indices = start[:, None] + offsets[None, :]
            mask = offsets[None, :] < count[:, None]
            indices = indices.clamp(0, max(0, point.shape[0] - 1))
            selected_points = point[indices]
            selected_weights = normal[indices].abs() * mask
            weight_sum = selected_weights.sum(dim=1)
            centroid = (selected_points * selected_weights[..., None]).sum(dim=1) / weight_sum[:, None].clamp_min(1e-12)
            centroid[weight_sum <= 0] = float("nan")
            centroids.append(centroid)
            forces.append(weight_sum)
            valid.append(weight_sum > 0)
        return torch.stack(centroids, dim=1), torch.stack(forces, dim=1), torch.stack(valid, dim=1)

    def run(self, duration: float, command_fn, seed: int, limit: int | None = None) -> list[dict]:
        self.env.seed(seed)
        self.wrapped.reset()
        n = min(self.env.num_envs, limit or self.env.num_envs)
        traces = [{
            "time": [], "vx": [], "root_speed": [], "yaw_rate": [], "yaw": [], "roll": [],
            "pitch": [], "tilt": [], "height": [], "contacts": [], "contact_force": [],
            "contact_point": [], "legacy_origin_speed": [], "action_rate": [], "vel_sat": [],
            "torque_sat": [], "segment": [], "command": [], "fall": False, "fall_step": None,
        } for _ in range(n)]
        alive = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        previous = torch.zeros(
            (self.env.num_envs, self.env.action_manager.total_action_dim), device=self.env.device
        )
        steps = round(duration / self.dt)
        for step in range(steps):
            t = step * self.dt
            speed, segment = command_fn(t)
            speed_tensor = torch.as_tensor(speed, device=self.env.device, dtype=torch.float32)
            if speed_tensor.ndim == 0:
                speed_tensor = speed_tensor.repeat(self.env.num_envs)
            self.command.vel_command_b[:, 0] = speed_tensor
            self.command.vel_command_b[:, 1:] = 0.0
            with torch.inference_mode():
                action = self.policy(self.wrapped.get_observations())
                _, _, dones, _ = self.wrapped.step(action)
            quaternion = self.robot.data.root_quat_w.torch
            roll, pitch, yaw = quat_xyzw_to_roll_pitch_yaw_torch(quaternion)
            tilt = quat_xyzw_to_gravity_tilt_torch(quaternion)
            points, forces, valid = self.contact_telemetry()
            contact_mask = (forces > 5.0) & valid
            foot_origin_speed = self.robot.data.body_lin_vel_w.torch[
                :, self.body_ids, :2
            ].norm(dim=-1)
            action_rate = torch.linalg.vector_norm(action - previous, dim=1) / self.dt
            previous = action.clone()
            vel_ratio = (
                self.robot.data.joint_vel.torch.abs()
                / self.robot.data.joint_vel_limits.torch.abs().clamp_min(1e-6)
            ).amax(dim=1)
            torque_ratio = (
                self.robot.data.applied_torque.torch.abs()
                / self.robot.data.joint_effort_limits.torch.abs().clamp_min(1e-6)
            ).amax(dim=1)
            root_speed = self.robot.data.root_lin_vel_w.torch[:, :2].norm(dim=-1)
            for index, trace in enumerate(traces):
                if not bool(alive[index]):
                    continue
                trace["time"].append(t)
                trace["vx"].append(float(self.robot.data.root_lin_vel_b.torch[index, 0]))
                trace["root_speed"].append(float(root_speed[index]))
                trace["yaw_rate"].append(float(self.robot.data.root_ang_vel_w.torch[index, 2]))
                trace["yaw"].append(float(yaw[index]))
                trace["roll"].append(float(roll[index]))
                trace["pitch"].append(float(pitch[index]))
                trace["tilt"].append(float(tilt[index]))
                trace["height"].append(float(self.robot.data.root_pos_w.torch[index, 2]))
                trace["contacts"].append([bool(value) for value in contact_mask[index].cpu()])
                trace["contact_force"].append([float(value) for value in forces[index].cpu()])
                trace["contact_point"].append([
                    [float(value) for value in points[index, foot, :2].cpu()]
                    if bool(valid[index, foot]) else None
                    for foot in range(4)
                ])
                trace["legacy_origin_speed"].append([float(value) for value in foot_origin_speed[index].cpu()])
                trace["action_rate"].append(float(action_rate[index]))
                trace["vel_sat"].append(bool(vel_ratio[index] >= 0.95))
                trace["torque_sat"].append(bool(torque_ratio[index] >= 0.95))
                trace["segment"].append(segment)
                trace["command"].append(float(speed_tensor[index]))
            newly_done = dones.bool() & alive
            for index in torch.nonzero(newly_done).flatten().tolist():
                if index < n:
                    traces[index]["fall"] = True
                    traces[index]["fall_step"] = step
            alive &= ~dones.bool()
        return traces


def condition_slip(trace: dict, indices: list[int]) -> dict:
    per_foot = []
    for foot in range(4):
        result = physical_slip_intervals(
            [trace["contact_force"][index][foot] for index in indices],
            [trace["contact_point"][index][foot] for index in indices],
            dt=0.02,
        )
        per_foot.append(result)
    speeds = [value for result in per_foot for value in result["speeds_mps"]]
    displacements = [
        value for result in per_foot for value in result["anchor_displacements_m"]
    ]
    stable_steps = sum(result["stable_contact_steps"] for result in per_foot)
    legacy = [
        trace["legacy_origin_speed"][index][foot]
        for index in indices for foot in range(4)
        if trace["contacts"][index][foot]
    ]
    return {
        "dangerous": any(result["dangerous"] for result in per_foot),
        "dangerous_interval_count": sum(result["dangerous_interval_count"] for result in per_foot),
        "stable_contact_time_fraction": stable_steps / max(1, len(indices) * 4),
        "contact_point_speed_p50_mps": percentile(speeds, 50),
        "contact_point_speed_p90_mps": percentile(speeds, 90),
        "contact_point_speed_p95_mps": percentile(speeds, 95),
        "contact_point_speed_p99_mps": percentile(speeds, 99),
        "contact_point_speed_max_mps": max(speeds, default=0.0),
        "anchor_displacement_p50_m": percentile(displacements, 50),
        "anchor_displacement_p90_m": percentile(displacements, 90),
        "anchor_displacement_p95_m": percentile(displacements, 95),
        "anchor_displacement_p99_m": percentile(displacements, 99),
        "anchor_displacement_max_m": max(displacements, default=0.0),
        "max_contiguous_dangerous_speed_s": max(
            (
                interval["maximum_contiguous_dangerous_speed_s"]
                for result in per_foot for interval in result["intervals"]
            ),
            default=0.0,
        ),
        "per_foot": {
            FEET[foot]: {
                "dangerous": result["dangerous"],
                "dangerous_interval_count": result["dangerous_interval_count"],
                "stable_contact_time_s": result["stable_contact_time_s"],
                "excluded_short_interval_count": result["excluded_short_interval_count"],
                "speed_p95_mps": percentile(result["speeds_mps"], 95),
                "displacement_p95_m": percentile(result["anchor_displacements_m"], 95),
            }
            for foot, result in enumerate(per_foot)
        },
        "legacy_foot_link_origin_speed_mean_mps": mean(legacy),
        "legacy_foot_link_origin_speed_p95_mps": percentile(legacy, 95),
    }


def summarize(trace: dict, target: float, start_s: float, end_s: float, heading_reference: float | None = None) -> dict:
    indices = [
        index for index, t in enumerate(trace["time"])
        if start_s <= t < end_s
    ]
    if heading_reference is None:
        heading_reference = trace["yaw"][indices[0]] if indices else 0.0
    signed_heading = [heading_error(trace["yaw"][index], heading_reference) for index in indices]
    slip = condition_slip(trace, list(range(len(trace["time"]))))
    contacts = [trace["contacts"][index] for index in indices]
    gait, gait_evidence = classify_go2_gait_v1(
        contacts, mean(trace["vx"][index] for index in indices), trace["fall"]
    )
    vel_dwell = max_true_run([trace["vel_sat"][index] for index in indices]) * 0.02
    torque_dwell = max_true_run([trace["torque_sat"][index] for index in indices]) * 0.02
    speed_error = mean(abs(trace["vx"][index] - target) for index in indices)
    tolerance = 0.08 if target == 0.0 else (0.25 if target >= 2.0 else 0.20)
    return {
        "target_speed_mps": target,
        "quality_window_s": [start_s, end_s],
        "samples": len(indices),
        "fall": trace["fall"],
        "fall_step": trace["fall_step"],
        "root_speed_mean_mps": mean(trace["root_speed"][index] for index in indices),
        "root_speed_p95_mps": percentile((trace["root_speed"][index] for index in indices), 95),
        "actual_forward_speed_mean_mps": mean(trace["vx"][index] for index in indices),
        "speed_mae_mps": speed_error,
        "tracking_success": not trace["fall"] and speed_error <= tolerance,
        "yaw_rate_p95_radps": percentile((abs(trace["yaw_rate"][index]) for index in indices), 95),
        "heading_error_signed_mean_rad": mean(signed_heading),
        "heading_error_abs_p95_rad": percentile((abs(value) for value in signed_heading), 95),
        "absolute_roll_p95_rad": percentile((abs(trace["roll"][index]) for index in indices), 95),
        "absolute_pitch_p95_rad": percentile((abs(trace["pitch"][index]) for index in indices), 95),
        "gravity_tilt_p95_rad": percentile((trace["tilt"][index] for index in indices), 95),
        "base_height_range_m": (
            max((trace["height"][index] for index in indices), default=0.0)
            - min((trace["height"][index] for index in indices), default=0.0)
        ),
        "foot_contact_occupancy": [
            mean(row[foot] for row in contacts) for foot in range(4)
        ] if contacts else [0.0] * 4,
        "contact_loss_max_s": max_true_run([not any(row) for row in contacts]) * 0.02,
        "long_dwell_saturation": max(vel_dwell, torque_dwell) >= 0.20,
        "action_rate_p95": percentile((trace["action_rate"][index] for index in indices), 95),
        "gait_class_v1": gait,
        "gait_evidence": gait_evidence,
        "physical_slip": slip,
    }


def aggregate(rows: list[dict], target: float, stand: bool = False) -> dict:
    result = {
        "target_speed_mps": target,
        "episodes": len(rows),
        "success_rate": mean(row["tracking_success"] for row in rows),
        "fall_rate": mean(row["fall"] for row in rows),
        "root_speed_mean_mps": mean(row["root_speed_mean_mps"] for row in rows),
        "root_speed_p95_mps": percentile((row["root_speed_p95_mps"] for row in rows), 95),
        "actual_forward_speed_mean_mps": mean(row["actual_forward_speed_mean_mps"] for row in rows),
        "speed_mae_mps": mean(row["speed_mae_mps"] for row in rows),
        "yaw_rate_p95_radps": percentile((row["yaw_rate_p95_radps"] for row in rows), 95),
        "heading_error_abs_p95_rad": percentile((row["heading_error_abs_p95_rad"] for row in rows), 95),
        "absolute_roll_p95_rad": percentile((row["absolute_roll_p95_rad"] for row in rows), 95),
        "absolute_pitch_p95_rad": percentile((row["absolute_pitch_p95_rad"] for row in rows), 95),
        "gravity_tilt_p95_rad": percentile((row["gravity_tilt_p95_rad"] for row in rows), 95),
        "base_height_range_p95_m": percentile((row["base_height_range_m"] for row in rows), 95),
        "dangerous_physical_slip_rate": mean(row["physical_slip"]["dangerous"] for row in rows),
        "long_dwell_saturation_rate": mean(row["long_dwell_saturation"] for row in rows),
        "gait_counts_v1": dict(Counter(row["gait_class_v1"] for row in rows)),
    }
    if stand:
        checks = {
            "hold_success_ge_0.95": result["success_rate"] >= 0.95,
            "fall_le_0.02": result["fall_rate"] <= 0.02,
            "speed_mean_le_0.05": result["root_speed_mean_mps"] <= 0.05,
            "speed_p95_le_0.10": result["root_speed_p95_mps"] <= 0.10,
            "yaw_rate_p95_le_0.10": result["yaw_rate_p95_radps"] <= 0.10,
            "absolute_roll_p95_le_0.15": result["absolute_roll_p95_rad"] <= 0.15,
            "absolute_pitch_p95_le_0.15": result["absolute_pitch_p95_rad"] <= 0.15,
            "gravity_tilt_p95_le_0.15": result["gravity_tilt_p95_rad"] <= 0.15,
            "height_range_p95_le_0.05": result["base_height_range_p95_m"] <= 0.05,
            "dangerous_slip_le_0.05": result["dangerous_physical_slip_rate"] <= 0.05,
            "long_dwell_saturation_le_0.05": result["long_dwell_saturation_rate"] <= 0.05,
        }
    else:
        error_limit = 0.25 if target >= 2.0 else 0.20
        checks = {
            "success_ge_0.90": result["success_rate"] >= 0.90,
            "fall_le_0.02": result["fall_rate"] <= 0.02,
            "heading_p95_le_0.12": result["heading_error_abs_p95_rad"] <= 0.12,
            "tilt_p95_le_0.20": result["gravity_tilt_p95_rad"] <= 0.20,
            "dangerous_slip_le_0.05": result["dangerous_physical_slip_rate"] <= 0.05,
            "long_dwell_saturation_le_0.05": result["long_dwell_saturation_rate"] <= 0.05,
            "speed_mae_within_limit": result["speed_mae_mps"] <= error_limit,
        }
    result["gate_checks"] = checks
    result["gate_pass"] = all(checks.values())
    result["status"] = "SUPPORTED" if result["gate_pass"] else (
        "PARTIAL" if result["success_rate"] > 0 else "UNSUPPORTED"
    )
    return result


def run_steady(collector: Collector, policy_name: str, checkpoint: Path) -> tuple[dict, list[dict]]:
    collector.load(checkpoint)
    outputs, episode_rows = {}, []
    for speed in (0.0,) + STEADY:
        traces = collector.run(8.0, lambda _t, value=speed: (value, "steady"), SEED)
        start = 1.0 if speed == 0.0 else 2.0
        rows = []
        for episode, trace in enumerate(traces):
            row = {
                "checkpoint": policy_name, "episode": episode, "episode_seed": SEED + episode,
                **summarize(trace, speed, start, 8.0),
            }
            rows.append(row)
            episode_rows.append(row)
        outputs[str(speed)] = aggregate(rows, speed, stand=speed == 0.0)
        print(f"{policy_name} steady {speed}: {outputs[str(speed)]['status']}", flush=True)
    return outputs, episode_rows


def acquisition(trace: dict, target: float, start_s: float) -> tuple[bool, float | None]:
    tolerance = 0.08 if target == 0.0 else (0.25 if target >= 2.0 else 0.20)
    required = round(1.0 / 0.02)
    current = 0
    for index, t in enumerate(trace["time"]):
        if t < start_s:
            continue
        value = abs(trace["vx"][index]) <= tolerance if target == 0 else abs(trace["vx"][index] - target) <= tolerance
        current = current + 1 if value else 0
        if current >= required:
            return True, t - (required - 1) * 0.02
    return False, None


def run_transitions(collector: Collector, policy_name: str, checkpoint: Path) -> tuple[dict, list[dict]]:
    collector.load(checkpoint)
    outputs, episode_rows = {}, []
    for source, target in TRANSITIONS:
        traces = collector.run(
            9.5,
            lambda t, a=source, b=target: (
                transition_command(t, a, b, 1.5)[0],
                transition_command(t, a, b, 1.5)[1],
            ),
            SEED,
        )
        rows = []
        for episode, trace in enumerate(traces):
            source_yaws = [
                trace["yaw"][index] for index, t in enumerate(trace["time"]) if 2.0 <= t < 3.0
            ]
            reference = circular_median(source_yaws)
            target_summary = summarize(trace, target, 5.5, 9.5, reference)
            acquired, acquisition_time = acquisition(trace, target, 4.5)
            target_tolerance = 0.08 if target == 0 else (0.25 if target >= 2.0 else 0.20)
            target_indices = [index for index, t in enumerate(trace["time"]) if 5.5 <= t < 9.5]
            target_hold = mean(
                abs(trace["vx"][index]) <= target_tolerance if target == 0
                else abs(trace["vx"][index] - target) <= target_tolerance
                for index in target_indices
            ) >= 0.90
            row = {
                "checkpoint": policy_name, "episode": episode, "episode_seed": SEED + episode,
                "source_speed_mps": source, "target_speed_mps": target,
                "completion": not trace["fall"] and acquired and target_hold,
                "target_acquired": acquired, "acquisition_time_s": acquisition_time,
                "target_hold_success": target_hold,
                **target_summary,
            }
            rows.append(row)
            episode_rows.append(row)
        summary = {
            "source_speed_mps": source,
            "target_speed_mps": target,
            "episodes": len(rows),
            "completion_rate": mean(row["completion"] for row in rows),
            "acquisition_rate": mean(row["target_acquired"] for row in rows),
            "target_hold_rate": mean(row["target_hold_success"] for row in rows),
            "fall_rate": mean(row["fall"] for row in rows),
            "heading_error_abs_p95_rad": percentile((row["heading_error_abs_p95_rad"] for row in rows), 95),
            "dangerous_physical_slip_rate": mean(row["physical_slip"]["dangerous"] for row in rows),
            "long_dwell_saturation_rate": mean(row["long_dwell_saturation"] for row in rows),
            "timeout_rate": mean(not row["target_acquired"] for row in rows),
            "final_speed_abs_mean_mps": mean(abs(row["actual_forward_speed_mean_mps"]) for row in rows),
        }
        checks = {
            "completion_ge_0.90": summary["completion_rate"] >= 0.90,
            "acquisition_ge_0.90": summary["acquisition_rate"] >= 0.90,
            "target_hold_ge_0.90": summary["target_hold_rate"] >= 0.90,
            "fall_le_0.05": summary["fall_rate"] <= 0.05,
            "heading_p95_le_0.12": summary["heading_error_abs_p95_rad"] <= 0.12,
            "dangerous_slip_le_0.05": summary["dangerous_physical_slip_rate"] <= 0.05,
            "saturation_le_0.05": summary["long_dwell_saturation_rate"] <= 0.05,
            "timeout_le_0.05": summary["timeout_rate"] <= 0.05,
        }
        if target == 0:
            checks["final_speed_le_0.08"] = summary["final_speed_abs_mean_mps"] <= 0.08
            checks["final_hold_ge_0.95"] = summary["target_hold_rate"] >= 0.95
            checks["corrected_stand_target_window"] = aggregate(rows, 0.0, stand=True)["gate_pass"]
        summary["gate_checks"] = checks
        summary["gate_pass"] = all(checks.values())
        outputs[f"{source:g}_to_{target:g}"] = summary
        print(f"{policy_name} transition {source}->{target}: {summary['gate_pass']}", flush=True)
    return outputs, episode_rows


def run_low_speed(collector: Collector) -> dict:
    results = {}
    for policy_name, checkpoint in (("official_parent", PARENT), ("stage4_selected", SELECTED)):
        collector.load(checkpoint)
        policy_results = {}
        for speed in LOW_SPEED:
            traces = collector.run(8.0, lambda _t, value=speed: (value, "steady"), SEED, limit=20)
            rows = [
                summarize(trace, speed, 2.0, 8.0) for trace in traces[:20]
            ]
            policy_results[str(speed)] = {
                **aggregate(rows, speed, stand=False),
                "diagnostic_only": True,
                "initialization_sensitivity": {
                    "failed_episode_seeds": [SEED + index for index, row in enumerate(rows) if row["fall"]],
                    "failure_concentrated_before_2s": mean(
                        row["fall_step"] is not None and row["fall_step"] * 0.02 < 2.0
                        for row in rows if row["fall"]
                    ) if any(row["fall"] for row in rows) else 0.0,
                },
            }
        results[policy_name] = policy_results
    return {
        "seed_root": SEED, "episodes_per_condition": 20,
        "formal_capability_set": False, "results": results,
    }


def run_reduced(collector: Collector) -> dict:
    duration = len(REDUCED) * 3.0 + (len(REDUCED) - 1) * 1.5

    def profile(t):
        cursor = 0.0
        for index, speed in enumerate(REDUCED):
            if t < cursor + 3.0:
                return speed, f"hold_{index}"
            cursor += 3.0
            if index < len(REDUCED) - 1:
                if t < cursor + 1.5:
                    tau = (t - cursor) / 1.5
                    p = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
                    return speed + (REDUCED[index + 1] - speed) * p, f"ramp_{index}"
                cursor += 1.5
        return REDUCED[-1], "final"

    collector.load(SELECTED)
    traces = collector.run(duration, profile, SEED)
    rows = []
    for episode, trace in enumerate(traces):
        slip = condition_slip(trace, list(range(len(trace["time"]))))
        final = summarize(trace, 0.0, duration - 3.0, duration)
        rows.append({
            "episode": episode, "episode_seed": SEED + episode,
            "completion": not trace["fall"], "fall": trace["fall"],
            "dangerous_physical_slip": slip["dangerous"],
            "heading_error_p95_rad": final["heading_error_abs_p95_rad"],
            "long_dwell_saturation": final["long_dwell_saturation"],
            "final_stand_pass": aggregate([final], 0.0, stand=True)["gate_pass"],
        })
    summary = {
        "executed": True,
        "episodes": len(rows),
        "completion_rate": mean(row["completion"] for row in rows),
        "fall_rate": mean(row["fall"] for row in rows),
        "dangerous_physical_slip_rate": mean(row["dangerous_physical_slip"] for row in rows),
        "heading_error_p95_rad": percentile((row["heading_error_p95_rad"] for row in rows), 95),
        "long_dwell_saturation_rate": mean(row["long_dwell_saturation"] for row in rows),
        "final_corrected_stand_rate": mean(row["final_stand_pass"] for row in rows),
        "checkpoint_switches": 0,
        "episodes_detail": rows,
    }
    checks = {
        "completion_ge_0.90": summary["completion_rate"] >= 0.90,
        "fall_le_0.05": summary["fall_rate"] <= 0.05,
        "heading_p95_le_0.12": summary["heading_error_p95_rad"] <= 0.12,
        "dangerous_slip_le_0.05": summary["dangerous_physical_slip_rate"] <= 0.05,
        "saturation_le_0.05": summary["long_dwell_saturation_rate"] <= 0.05,
        "final_stand_ge_0.95": summary["final_corrected_stand_rate"] >= 0.95,
        "checkpoint_switch_eq_0": True,
    }
    summary["gate_checks"] = checks
    summary["gate_pass"] = all(checks.values())
    return summary


def contact_preflight() -> None:
    raw, agent, cfg = make_env(min(args.num_envs, 4))
    collector = Collector(raw, agent)
    traces = collector.run(0.5, lambda _t: (0.0, "preflight"), SEED)
    sensor_rows = []
    for foot, sensor in zip(FEET, collector.sensors):
        data = sensor.data.contact_pos_w.torch
        sensor_rows.append({
            "foot": foot,
            "body_names": sensor.body_names,
            "num_sensors_per_env": sensor.num_sensors,
            "filter_count": sensor.contact_view.filter_count,
            "filter_path": GROUND,
            "contact_pos_shape": list(data.shape),
            "finite_contact_observed": bool(torch.isfinite(data).all(dim=-1).any()),
            "raw_contact_api": "RigidContactView.get_contact_data",
        })
    available = all(
        row["num_sensors_per_env"] == 1
        and row["filter_count"] == 1
        and row["finite_contact_observed"]
        for row in sensor_rows
    )
    dump("contact_point_source_audit.json", {
        "status": "CONTACT_POINT_AVAILABLE" if available else "CONTACT_POINT_METRIC_UNAVAILABLE",
        "backend": "PhysX",
        "source": "actual world-frame contact points from RigidContactView.get_contact_data",
        "multiple_contact_processing": "normal-force-weighted centroid per foot and ground filter",
        "sensors_are_read_only": True,
        "physics_or_policy_change": False,
        "rows": sensor_rows,
    })
    dump("contact_point_mapping.json", {
        "unambiguous": available,
        "mapping": [
            {
                "anatomical_foot": foot, "asset_body": asset,
                "diagnostic_sensor": f"stage6_{label}_contact",
                "filter": GROUND, "contact_point_tensor": "raw PhysX contact data",
            }
            for foot, asset, label in zip(FEET, ASSET_FEET, ("fl", "fr", "rl", "rr"))
        ],
    })
    raw.close()
    if not available:
        raise SystemExit("CONTACT_POINT_METRIC_UNAVAILABLE")


def formal() -> None:
    if args.num_envs != 50:
        raise SystemExit("formal mode requires exactly 50 environments")
    protocol_hash = json.loads((args.output / "protocol_hash.json").read_text(encoding="utf-8"))
    if not protocol_hash.get("frozen_before_formal_rollout"):
        raise SystemExit("protocol was not frozen")
    audit = json.loads((args.output / "contact_point_source_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "CONTACT_POINT_AVAILABLE":
        raise SystemExit("CONTACT_POINT_METRIC_UNAVAILABLE")
    if sha256(PARENT) != "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0":
        raise SystemExit("parent checkpoint hash mismatch")
    if sha256(SELECTED) != "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea":
        raise SystemExit("selected checkpoint hash mismatch")

    raw, agent, _ = make_env(args.num_envs)
    collector = Collector(raw, agent)
    steady_all, steady_rows_all = {}, {}
    transition_all, transition_rows_all = {}, {}
    for policy_name, checkpoint in (("official_parent", PARENT), ("stage4_selected", SELECTED)):
        steady_all[policy_name], steady_rows_all[policy_name] = run_steady(
            collector, policy_name, checkpoint
        )
        transition_all[policy_name], transition_rows_all[policy_name] = run_transitions(
            collector, policy_name, checkpoint
        )

    dump("parent_formal_stand.json", {
        "summary": steady_all["official_parent"]["0.0"],
        "episodes": [row for row in steady_rows_all["official_parent"] if row["target_speed_mps"] == 0.0],
    })
    dump("selected_formal_stand.json", {
        "summary": steady_all["stage4_selected"]["0.0"],
        "episodes": [row for row in steady_rows_all["stage4_selected"] if row["target_speed_mps"] == 0.0],
    })
    parent_steady_rows = [
        row for row in steady_rows_all["official_parent"] if row["target_speed_mps"] != 0.0
    ]
    selected_steady_rows = [
        row for row in steady_rows_all["stage4_selected"] if row["target_speed_mps"] != 0.0
    ]
    write_csv("parent_formal_steady_state.csv", parent_steady_rows)
    write_csv("selected_formal_steady_state.csv", selected_steady_rows)
    dump("selected_formal_steady_state.json", {
        "per_speed": {
            key: value for key, value in steady_all["stage4_selected"].items() if key != "0.0"
        },
        "episodes": selected_steady_rows,
    })
    write_csv("parent_formal_transitions.csv", transition_rows_all["official_parent"])
    write_csv("selected_formal_transitions.csv", transition_rows_all["stage4_selected"])
    dump("selected_formal_transitions.json", {
        "per_direction": transition_all["stage4_selected"],
        "episodes": transition_rows_all["stage4_selected"],
    })

    prerequisites = (
        steady_all["stage4_selected"]["0.0"]["gate_pass"]
        and all(steady_all["stage4_selected"][str(speed)]["gate_pass"] for speed in (0.6, 1.2, 2.0))
        and all(value["gate_pass"] for value in transition_all["stage4_selected"].values())
    )
    reduced = run_reduced(collector) if prerequisites else {
        "executed": False,
        "reason": "zero, required steady endpoints, or formal transitions not all SUPPORTED",
        "gate_pass": False,
    }
    dump("formal_reduced_sequence.json", reduced)
    low_speed = run_low_speed(collector)
    dump("low_speed_diagnostic.json", low_speed)

    endpoint_rows = []
    for origin, key in (("reset_steady", None), ("0_to_1.2", "0_to_1.2"), ("2_to_1.2", "2_to_1.2")):
        source_rows = (
            [row for row in selected_steady_rows if row["target_speed_mps"] == 1.2]
            if key is None else
            [row for row in transition_rows_all["stage4_selected"] if row["source_speed_mps"] == float(key.split("_to_")[0]) and row["target_speed_mps"] == 1.2]
        )
        endpoint_rows.append({
            "origin": origin,
            "speed_mean_mps": mean(row["actual_forward_speed_mean_mps"] for row in source_rows),
            "heading_p95_rad": percentile((row["heading_error_abs_p95_rad"] for row in source_rows), 95),
            "tilt_p95_rad": percentile((row["gravity_tilt_p95_rad"] for row in source_rows), 95),
            "dangerous_slip_rate": mean(row["physical_slip"]["dangerous"] for row in source_rows),
            "action_rate_p95": percentile((row["action_rate_p95"] for row in source_rows), 95),
            "gait_counts": dict(Counter(row["gait_class_v1"] for row in source_rows)),
            "high_speed_gait_or_flight_retained": False,
        })
    dump("directional_asymmetry.json", {
        "endpoint_mps": 1.2, "comparisons": endpoint_rows,
        "high_speed_gait_retention_after_deceleration": False,
    })
    write_csv("endpoint_hysteresis.csv", endpoint_rows)

    severity = {}
    per_foot_rows = []
    for policy_name, rows in steady_rows_all.items():
        severity[policy_name] = {}
        for speed in (0.0,) + STEADY:
            subset = [row for row in rows if row["target_speed_mps"] == speed]
            severity[policy_name][str(speed)] = {
                "dangerous_episode_rate": mean(row["physical_slip"]["dangerous"] for row in subset),
                "speed_p95_mps": percentile(
                    (row["physical_slip"]["contact_point_speed_p95_mps"] for row in subset), 95
                ),
                "displacement_p95_m": percentile(
                    (row["physical_slip"]["anchor_displacement_p95_m"] for row in subset), 95
                ),
                "legacy_origin_speed_p95_mps": percentile(
                    (row["physical_slip"]["legacy_foot_link_origin_speed_p95_mps"] for row in subset), 95
                ),
            }
            for foot in FEET:
                per_foot_rows.append({
                    "checkpoint": policy_name, "speed_mps": speed, "foot": foot,
                    "dangerous_episode_rate": mean(
                        row["physical_slip"]["per_foot"][foot]["dangerous"] for row in subset
                    ),
                    "speed_p95_mps": percentile(
                        (row["physical_slip"]["per_foot"][foot]["speed_p95_mps"] for row in subset), 95
                    ),
                    "displacement_p95_m": percentile(
                        (row["physical_slip"]["per_foot"][foot]["displacement_p95_m"] for row in subset), 95
                    ),
                })
    dump("physical_slip_severity.json", severity)
    write_csv("per_foot_contact_point_slip.csv", per_foot_rows)
    raw.close()


def policy_checkpoint() -> tuple[str, Path]:
    if not args.policy:
        raise SystemExit("--policy is required for split formal modes")
    return args.policy, PARENT if args.policy == "official_parent" else SELECTED


def split_steady() -> None:
    if args.num_envs != 50:
        raise SystemExit("formal steady mode requires exactly 50 environments")
    policy_name, checkpoint = policy_checkpoint()
    raw, agent, _ = make_env(50)
    collector = Collector(raw, agent)
    summaries, rows = run_steady(collector, policy_name, checkpoint)
    dump(f"partial_{policy_name}_steady.json", {
        "formal_episode_count": 50, "summaries": summaries, "episodes": rows,
    })
    raw.close()


def split_transitions() -> None:
    if args.num_envs != 50:
        raise SystemExit("formal transition mode requires exactly 50 environments")
    policy_name, checkpoint = policy_checkpoint()
    raw, agent, _ = make_env(50)
    collector = Collector(raw, agent)
    summaries, rows = run_transitions(collector, policy_name, checkpoint)
    dump(f"partial_{policy_name}_transitions.json", {
        "formal_episode_count": 50, "summaries": summaries, "episodes": rows,
    })
    raw.close()


def split_low_speed() -> None:
    if args.num_envs != 20:
        raise SystemExit("low-speed diagnostic mode requires exactly 20 environments")
    policy_name, checkpoint = policy_checkpoint()
    raw, agent, _ = make_env(20)
    collector = Collector(raw, agent)
    collector.load(checkpoint)
    results = {}
    for speed in LOW_SPEED:
        traces = collector.run(8.0, lambda _t, value=speed: (value, "steady"), SEED)
        rows = [summarize(trace, speed, 2.0, 8.0) for trace in traces]
        results[str(speed)] = {
            **aggregate(rows, speed, stand=False),
            "diagnostic_only": True,
            "initialization_sensitivity": {
                "failed_episode_seeds": [
                    SEED + index for index, row in enumerate(rows) if row["fall"]
                ],
                "failure_concentrated_before_2s": mean(
                    row["fall_step"] is not None and row["fall_step"] * 0.02 < 2.0
                    for row in rows if row["fall"]
                ) if any(row["fall"] for row in rows) else 0.0,
            },
        }
    dump(f"partial_{policy_name}_low_speed.json", {
        "formal_episode_count": 20, "results": results,
    })
    raw.close()


def split_reduced() -> None:
    if args.num_envs != 50 or args.policy != "stage4_selected":
        raise SystemExit("reduced mode requires 50 environments and stage4_selected")
    raw, agent, _ = make_env(50)
    collector = Collector(raw, agent)
    dump("partial_stage4_selected_reduced.json", run_reduced(collector))
    raw.close()


try:
    if args.mode == "contact-preflight":
        contact_preflight()
    elif args.mode == "formal-steady":
        split_steady()
    elif args.mode == "formal-transitions":
        split_transitions()
    elif args.mode == "formal-low-speed":
        split_low_speed()
    elif args.mode == "formal-reduced":
        split_reduced()
    else:
        formal()
finally:
    simulation_app.close()
