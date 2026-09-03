"""Deterministic Stage 8 low-speed heading diagnosis; no optimizer is created."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage8_low_speed_heading_diagnosis"
CHECKPOINTS = {
    "official_parent": REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt",
    "stage4_selected": REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training/checkpoints/model_50.pt",
    "stage7_selected": REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization/checkpoints/model_50.pt",
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=OUT)
parser.add_argument("--num-envs", type=int, default=50)
parser.add_argument("--aggregate-only", action="store_true")
parser.add_argument(
    "--checkpoint",
    choices=("official_parent", "stage4_selected", "stage7_selected"),
)
parser.add_argument(
    "--chunk",
    choices=(
        "steady_a", "steady_b", "steady_c",
        "low_transitions_a", "low_transitions_b",
        "anchor_transitions", "yaw_probe_02", "yaw_probe_04", "yaw_probe_06", "yaw_probe_12",
        "feedback",
        *(f"feedback_steady_{code}" for code in ("02", "03", "04", "05", "06")),
        *(f"feedback_transition_{index}" for index in range(4)),
        *(f"low_transition_{index}" for index in range(6)),
        *(f"anchor_transition_{index}" for index in range(4)),
    ),
    help="Run one durable diagnostic chunk. This keeps each Isaac process below the host job lifetime.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_physx.sensors import ContactSensorCfg  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
from go2_bidirectional.command_profiles import transition_command  # noqa: E402
from go2_bidirectional.evaluation import build_runner  # noqa: E402
from go2_bidirectional.stage6_endpoint_protocol import (  # noqa: E402
    circular_median,
    heading_error,
    physical_slip_intervals,
    quat_xyzw_to_roll_pitch_yaw_torch,
)
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

SEED = 20267901
DT = 0.02
FEET = ("FL", "FR", "RL", "RR")
ASSET_FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GROUND = "/World/ground/terrain/GroundPlane/CollisionPlane"
JOINTS = (
    "FL_hip", "FR_hip", "RL_hip", "RR_hip", "FL_thigh", "FR_thigh",
    "RL_thigh", "RR_thigh", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
)
LR_JOINT_PAIRS = (
    ("front_hip", 0, 1, -1), ("rear_hip", 2, 3, -1),
    ("front_thigh", 4, 5, 1), ("rear_thigh", 6, 7, 1),
    ("front_calf", 8, 9, 1), ("rear_calf", 10, 11, 1),
)
STEADY = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2, 2.0)
LOW_TRANSITIONS = ((0.0, 0.2), (0.0, 0.4), (0.0, 0.6), (0.6, 0.4), (0.6, 0.2), (0.6, 0.0))
ANCHOR_TRANSITIONS = ((0.0, 1.2), (1.2, 2.0), (2.0, 1.2), (1.2, 0.0))


def dump(name: str, value) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        raise TypeError(f"cannot encode {type(item)!r}")

    (args.output / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, default=encode) + "\n",
        encoding="utf-8",
    )


def write_csv(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def percentile(values, q: float) -> float:
    values = list(values)
    return float(np.percentile(values, q)) if values else 0.0


def mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def wrap(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def max_run(flags) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    result = np.empty_like(order, dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    return result


def corr(x, y, spearman=False) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    if spearman:
        x, y = ranks(x), ranks(y)
    return float(np.corrcoef(x, y)[0, 1])


def add_point_sensors(cfg) -> None:
    for label, foot in zip(("fl", "fr", "rl", "rr"), ASSET_FEET):
        setattr(cfg.scene, f"stage8_{label}_contact", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot}", update_period=0.0,
            track_pose=True, track_contact_points=True, max_contact_data_count_per_prim=8,
            filter_prim_paths_expr=[GROUND],
        ))


class Diagnostic:
    def __init__(self):
        cfg, agent = resolve_task_config("Isaac-Velocity-Flat-Unitree-Go2-v0", "rsl_rl_cfg_entry_point")
        cfg.scene.num_envs = args.num_envs
        cfg.seed = SEED
        cfg.episode_length_s = 60.0
        cfg.observations.policy.enable_corruption = False
        cfg.events.base_external_force_torque = None
        cfg.events.push_robot = None
        add_point_sensors(cfg)
        if args.device:
            cfg.sim.device = args.device
            agent.device = args.device
        self.raw = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
        self.wrapped, self.runner, self.policy = build_runner(self.raw, agent, CHECKPOINTS["official_parent"])
        self.env = self.wrapped.unwrapped
        self.robot = self.env.scene["robot"]
        self.command = self.env.command_manager.get_term("base_velocity")
        self.force_sensor = self.env.scene.sensors["contact_forces"]
        self.force_ids = [int(self.force_sensor.find_bodies(name)[0][0]) for name in ASSET_FEET]
        self.body_ids = [int(self.robot.find_bodies(name)[0][0]) for name in ASSET_FEET]
        self.point_sensors = [
            self.env.scene.sensors[f"stage8_{label}_contact"] for label in ("fl", "fr", "rl", "rr")
        ]

    def load(self, checkpoint: Path) -> None:
        self.runner.load(str(checkpoint), strict=True, map_location=self.env.device)
        self.policy = self.runner.get_inference_policy(device=self.env.device)

    def points(self):
        centroids, forces, valid = [], [], []
        for sensor in self.point_sensors:
            normal, points, _, _, counts, starts = sensor.contact_view.get_contact_data(
                dt=sensor._sim_physics_dt
            )
            normal = wp.to_torch(normal).reshape(-1)
            points = wp.to_torch(points).reshape(-1, 3)
            counts = wp.to_torch(counts).reshape(-1).long()
            starts = wp.to_torch(starts).reshape(-1).long()
            width = int(sensor.cfg.max_contact_data_count_per_prim)
            offsets = torch.arange(width, device=counts.device)
            indices = (starts[:, None] + offsets[None]).clamp(0, max(0, len(points) - 1))
            mask = offsets[None] < counts[:, None]
            weights = normal[indices].abs() * mask
            total = weights.sum(1)
            centroid = (points[indices] * weights[..., None]).sum(1) / total[:, None].clamp_min(1e-12)
            centroid[total <= 0] = float("nan")
            centroids.append(centroid[:, :2]); forces.append(total); valid.append(total > 0)
        return torch.stack(centroids, 1), torch.stack(forces, 1), torch.stack(valid, 1)

    def collect(
        self, checkpoint_name: str, family: str, source: float, target: float,
        episodes: int, duration: float, yaw_command: float = 0.0, feedback: bool = False,
    ) -> list[dict]:
        self.load(CHECKPOINTS[checkpoint_name])
        self.env.seed(SEED)
        self.wrapped.reset()
        n = episodes
        initial = {
            "quat": self.robot.data.root_quat_w.torch[:n].clone(),
            "height": self.robot.data.root_pos_w.torch[:n, 2].clone(),
            "joint_pos": self.robot.data.joint_pos.torch[:n].clone(),
            "joint_vel": self.robot.data.joint_vel.torch[:n].clone(),
        }
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=self.env.device)
        falls = torch.zeros(args.num_envs, dtype=torch.bool, device=self.env.device)
        fall_step = torch.full((args.num_envs,), -1, dtype=torch.long, device=self.env.device)
        series = defaultdict(list)
        reference = None
        for step in range(round(duration / DT)):
            t = step * DT
            if family == "transition":
                speed, phase = transition_command(t, source, target, 1.5)
            else:
                speed, phase = target, "steady"
            if feedback:
                if reference is None:
                    _, _, initial_yaw = quat_xyzw_to_roll_pitch_yaw_torch(
                        self.robot.data.root_quat_w.torch
                    )
                    reference = initial_yaw.clone()
                _, _, current_yaw = quat_xyzw_to_roll_pitch_yaw_torch(
                    self.robot.data.root_quat_w.torch
                )
                error_target_minus_current = torch.atan2(
                    torch.sin(reference - current_yaw), torch.cos(reference - current_yaw)
                )
                yaw_tensor = error_target_minus_current.clamp(-0.10, 0.10)
            else:
                yaw_tensor = torch.full(
                    (args.num_envs,), yaw_command, device=self.env.device
                )
            self.command.vel_command_b[:, 0] = speed
            self.command.vel_command_b[:, 1] = 0.0
            self.command.vel_command_b[:, 2] = yaw_tensor
            obs = self.wrapped.get_observations()
            with torch.inference_mode():
                action = self.policy(obs)
                _, _, dones, _ = self.wrapped.step(action)
            roll, pitch, yaw = quat_xyzw_to_roll_pitch_yaw_torch(
                self.robot.data.root_quat_w.torch
            )
            point, point_force, point_valid = self.points()
            contacts = (point_force > 5.0) & point_valid
            velocity_ratio = (
                self.robot.data.joint_vel.torch.abs()
                / self.robot.data.joint_vel_limits.torch.abs().clamp_min(1e-6)
            ).amax(1)
            torque_ratio = (
                self.robot.data.applied_torque.torch.abs()
                / self.robot.data.joint_effort_limits.torch.abs().clamp_min(1e-6)
            ).amax(1)
            series["time"].append(t)
            series["phase"].append(phase)
            for key, tensor in (
                ("vx", self.robot.data.root_lin_vel_b.torch[:n, 0]),
                ("yaw", yaw[:n]), ("yaw_rate", self.robot.data.root_ang_vel_w.torch[:n, 2]),
                ("roll", roll[:n]), ("pitch", pitch[:n]), ("action", action[:n]),
                ("joint_pos", self.robot.data.joint_pos.torch[:n]),
                ("joint_vel", self.robot.data.joint_vel.torch[:n]),
                ("contact", contacts[:n]), ("force", point_force[:n]),
                ("point", point[:n]), ("foot_pos", self.robot.data.body_pos_w.torch[:n, self.body_ids, :2]),
                ("alive", alive[:n]), ("yaw_command", yaw_tensor[:n]),
                ("saturation", ((velocity_ratio >= 0.95) | (torque_ratio >= 0.95))[:n]),
            ):
                series[key].append(tensor.detach().cpu())
            newly = dones.bool() & alive
            fall_step[newly] = step
            falls |= newly
            alive &= ~dones.bool()
        arrays = {
            key: torch.stack(value).numpy() for key, value in series.items()
            if key not in ("time", "phase")
        }
        iroll, ipitch, iyaw = quat_xyzw_to_roll_pitch_yaw_torch(initial["quat"])
        rows = []
        for episode in range(n):
            end = int(fall_step[episode]) if fall_step[episode] >= 0 else len(series["time"])
            end = max(end, 2)
            if family == "transition":
                reference_values = arrays["yaw"][100:min(150, end), episode]
                heading_ref = circular_median(reference_values.tolist())
                quality_start = min(round(5.5 / DT), end - 1)
            elif target == 0:
                quality_start = min(round(1.0 / DT), end - 1)
                heading_ref = float(arrays["yaw"][quality_start, episode])
            else:
                quality_start = min(round(2.0 / DT), end - 1)
                heading_ref = float(arrays["yaw"][quality_start, episode])
            indices = np.arange(quality_start, end)
            yaw = arrays["yaw"][indices, episode]
            error = wrap(yaw - heading_ref)
            times = indices * DT
            slope = float(np.polyfit(times, error, 1)[0]) if len(times) >= 2 else 0.0
            fitted = slope * (times - times[0]) + error[0] if len(times) else np.array([])
            residual = error - fitted
            actions = arrays["action"][indices, episode]
            positions = arrays["joint_pos"][indices, episode]
            velocities = arrays["joint_vel"][indices, episode]
            contacts = arrays["contact"][:end, episode].astype(bool)
            forces = arrays["force"][:end, episode]
            points = arrays["point"][:end, episode]
            foot_positions = arrays["foot_pos"][:end, episode]
            per_foot = {}
            for foot in range(4):
                point_rows = [
                    points[index, foot].tolist()
                    if contacts[index, foot] and np.isfinite(points[index, foot]).all() else None
                    for index in range(end)
                ]
                slip = physical_slip_intervals(
                    forces[:, foot].tolist(), point_rows, dt=DT
                )
                onset = np.flatnonzero(contacts[1:, foot] & ~contacts[:-1, foot]) + 1
                release = np.flatnonzero(~contacts[1:, foot] & contacts[:-1, foot]) + 1
                placements = foot_positions[onset, foot] if len(onset) else np.empty((0, 2))
                per_foot[FEET[foot]] = {
                    "duty_factor": float(contacts[:, foot].mean()),
                    "contact_duration_mean_s": mean(np.diff(np.r_[0, np.flatnonzero(np.diff(contacts[:, foot].astype(int)) != 0), end]) * DT),
                    "force_mean_n": mean(forces[contacts[:, foot], foot]),
                    "force_p95_n": percentile(forces[contacts[:, foot], foot], 95),
                    "onset_phase_mean": mean((onset * DT) % 1.0),
                    "release_phase_mean": mean((release * DT) % 1.0),
                    "foot_placement_x_mean": mean(placements[:, 0]) if len(placements) else 0.0,
                    "foot_placement_y_mean": mean(placements[:, 1]) if len(placements) else 0.0,
                    "stride_length_mean": mean(np.linalg.norm(np.diff(placements, axis=0), axis=1)) if len(placements) > 1 else 0.0,
                    "slip_speed_p95": percentile(slip["speeds_mps"], 95),
                    "stance_displacement_p95": percentile(slip["anchor_displacements_m"], 95),
                    "dangerous": slip["dangerous"],
                    "max_contiguous_slip_s": max(
                        (item["maximum_contiguous_dangerous_speed_s"] for item in slip["intervals"]),
                        default=0.0,
                    ),
                }
            left_slip = mean(per_foot[foot]["stance_displacement_p95"] for foot in ("FL", "RL"))
            right_slip = mean(per_foot[foot]["stance_displacement_p95"] for foot in ("FR", "RR"))
            left_force = mean(per_foot[foot]["force_mean_n"] for foot in ("FL", "RL"))
            right_force = mean(per_foot[foot]["force_mean_n"] for foot in ("FR", "RR"))
            action_mean = actions.mean(0) if len(actions) else np.zeros(12)
            action_amp = np.ptp(actions, axis=0) if len(actions) else np.zeros(12)
            action_rate = np.abs(np.diff(actions, axis=0)).mean(0) / DT if len(actions) > 1 else np.zeros(12)
            pair_errors = {}
            for label, left, right, sign in LR_JOINT_PAIRS:
                pair_errors[label] = {
                    "signed_mean_error": float(action_mean[left] - sign * action_mean[right]),
                    "amplitude_difference": float(action_amp[left] - action_amp[right]),
                    "action_rate_difference": float(action_rate[left] - action_rate[right]),
                    "joint_position_difference": float(positions[:, left].mean() - sign * positions[:, right].mean()),
                    "joint_velocity_rms_difference": float(np.sqrt(np.mean(velocities[:, left] ** 2)) - np.sqrt(np.mean(velocities[:, right] ** 2))),
                }
            initial_contact = arrays["contact"][0, episode].astype(bool)
            row = {
                "checkpoint": checkpoint_name, "family": family,
                "condition": f"{source:g}->{target:g}" if family == "transition" else f"{target:g}",
                "source_speed": source, "target_speed": target, "episode": episode,
                "episode_seed": SEED + episode, "fall": bool(falls[episode]),
                "initial_roll": float(iroll[episode]), "initial_pitch": float(ipitch[episode]),
                "initial_yaw": float(iyaw[episode]), "initial_height": float(initial["height"][episode]),
                "initial_joint_pos_mean": float(initial["joint_pos"][episode].mean()),
                "initial_joint_vel_norm": float(initial["joint_vel"][episode].norm()),
                "initial_contact_pattern": "".join("1" if value else "0" for value in initial_contact),
                "initial_left_force": float(forces[0, [0, 2]].sum()),
                "initial_right_force": float(forces[0, [1, 3]].sum()),
                "heading_initial_offset": float(error[0]) if len(error) else 0.0,
                "heading_final_signed": float(error[-1]) if len(error) else 0.0,
                "heading_abs_p50": percentile(np.abs(error), 50),
                "heading_abs_p90": percentile(np.abs(error), 90),
                "heading_abs_p95": percentile(np.abs(error), 95),
                "heading_abs_p99": percentile(np.abs(error), 99),
                "heading_abs_max": max(np.abs(error), default=0.0),
                "signed_yaw_drift_slope": slope, "absolute_yaw_drift_slope": abs(slope),
                "yaw_rate_mean": mean(arrays["yaw_rate"][indices, episode]),
                "yaw_rate_p50": percentile(np.abs(arrays["yaw_rate"][indices, episode]), 50),
                "yaw_rate_p90": percentile(np.abs(arrays["yaw_rate"][indices, episode]), 90),
                "yaw_rate_p95": percentile(np.abs(arrays["yaw_rate"][indices, episode]), 95),
                "yaw_rate_p99": percentile(np.abs(arrays["yaw_rate"][indices, episode]), 99),
                "oscillatory_rms": float(np.sqrt(np.mean(residual**2))) if len(residual) else 0.0,
                "rare_excursion": bool(max(np.abs(error), default=0.0) > max(0.25, 2 * percentile(np.abs(error), 90))),
                "actual_speed_mean": mean(arrays["vx"][indices, episode]),
                "speed_mae": mean(np.abs(arrays["vx"][indices, episode] - target)),
                "long_dwell_saturation": max_run(
                    arrays["saturation"][indices, episode].astype(bool).tolist()
                ) >= round(0.20 / DT),
                "yaw_command_mean": mean(arrays["yaw_command"][indices, episode]),
                "action_mean": action_mean.tolist(), "action_amplitude": action_amp.tolist(),
                "action_rate": action_rate.tolist(), "pair_action_asymmetry": pair_errors,
                "per_foot": per_foot,
                "left_slip": left_slip, "right_slip": right_slip,
                "left_right_slip_difference": left_slip - right_slip,
                "left_force": left_force, "right_force": right_force,
                "left_right_force_difference": left_force - right_force,
            }
            if family == "transition":
                phases = {
                    "source_hold": (0.0, 3.0), "ramp": (3.0, 4.5),
                    "target_acquisition": (4.5, 5.5), "target_hold": (5.5, 9.5),
                }
                row["phases"] = {}
                for label, (start_s, end_s) in phases.items():
                    ix = np.arange(round(start_s / DT), min(round(end_s / DT), end))
                    if not len(ix):
                        continue
                    phase_error = wrap(arrays["yaw"][ix, episode] - heading_ref)
                    row["phases"][label] = {
                        "heading_change": float(phase_error[-1] - phase_error[0]),
                        "yaw_rate_p95": percentile(np.abs(arrays["yaw_rate"][ix, episode]), 95),
                        "left_right_action_difference": mean(
                            np.abs(arrays["action"][ix, episode][:, [0, 2, 4, 6, 8, 10]])
                            - np.abs(arrays["action"][ix, episode][:, [1, 3, 5, 7, 9, 11]])
                        ),
                        "left_right_force_difference": mean(
                            arrays["force"][ix, episode][:, [0, 2]].sum(1)
                            - arrays["force"][ix, episode][:, [1, 3]].sum(1)
                        ),
                    }
            rows.append(row)
        print(f"STAGE8 {checkpoint_name} {family} {source}->{target} n={episodes}", flush=True)
        return rows

    def close(self):
        self.wrapped.close()


def aggregate_heading(rows: list[dict]) -> dict:
    nonfallen = [row for row in rows if not row["fall"]]
    return {
        "episodes": len(rows), "fall_rate": mean(row["fall"] for row in rows),
        "nonfallen_episodes": len(nonfallen),
        "signed_slope_mean": mean(row["signed_yaw_drift_slope"] for row in nonfallen),
        "absolute_slope_mean": mean(row["absolute_yaw_drift_slope"] for row in nonfallen),
        "yaw_rate_mean": mean(row["yaw_rate_mean"] for row in nonfallen),
        "yaw_rate_p50": percentile((row["yaw_rate_p50"] for row in nonfallen), 50),
        "yaw_rate_p90": percentile((row["yaw_rate_p90"] for row in nonfallen), 90),
        "yaw_rate_p95": percentile((row["yaw_rate_p95"] for row in nonfallen), 95),
        "yaw_rate_p99": percentile((row["yaw_rate_p99"] for row in nonfallen), 99),
        "heading_p50": percentile((row["heading_abs_p50"] for row in nonfallen), 50),
        "heading_p90": percentile((row["heading_abs_p90"] for row in nonfallen), 90),
        "heading_p95": percentile((row["heading_abs_p95"] for row in nonfallen), 95),
        "heading_p99": percentile((row["heading_abs_p99"] for row in nonfallen), 99),
        "heading_max": max((row["heading_abs_max"] for row in nonfallen), default=0.0),
        "oscillatory_rms": mean(row["oscillatory_rms"] for row in nonfallen),
        "rare_excursion_rate": mean(row["rare_excursion"] for row in nonfallen),
        "positive_final_fraction": mean(row["heading_final_signed"] > 0 for row in nonfallen),
        "negative_final_fraction": mean(row["heading_final_signed"] < 0 for row in nonfallen),
        "fall_related_heading_p95": percentile((row["heading_abs_p95"] for row in rows if row["fall"]), 95),
        "nonfall_heading_p95": percentile((row["heading_abs_p95"] for row in nonfallen), 95),
    }


def main() -> None:
    if args.num_envs != 50:
        raise SystemExit("Stage 8 requires 50 environments")
    diagnostic = None if args.aggregate_only else Diagnostic()
    if args.chunk:
        if not args.chunk.startswith(("yaw_probe", "feedback")) and not args.checkpoint:
            raise SystemExit("--checkpoint is required for this chunk")
        rows = []
        if args.chunk in ("steady_a", "steady_b", "steady_c"):
            speeds = {
                "steady_a": STEADY[:3],
                "steady_b": STEADY[3:6],
                "steady_c": STEADY[6:],
            }[args.chunk]
            for speed in speeds:
                rows.extend(diagnostic.collect(args.checkpoint, "steady", speed, speed, 50, 8.0))
        elif args.chunk in ("low_transitions_a", "low_transitions_b"):
            transitions = LOW_TRANSITIONS[:3] if args.chunk.endswith("_a") else LOW_TRANSITIONS[3:]
            for source, target in transitions:
                rows.extend(diagnostic.collect(args.checkpoint, "transition", source, target, 50, 9.5))
        elif args.chunk.startswith("low_transition_"):
            source, target = LOW_TRANSITIONS[int(args.chunk.rsplit("_", 1)[1])]
            rows.extend(diagnostic.collect(args.checkpoint, "transition", source, target, 50, 9.5))
        elif args.chunk == "anchor_transitions":
            for source, target in ANCHOR_TRANSITIONS:
                rows.extend(diagnostic.collect(args.checkpoint, "transition", source, target, 20, 9.5))
        elif args.chunk.startswith("anchor_transition_"):
            source, target = ANCHOR_TRANSITIONS[int(args.chunk.rsplit("_", 1)[1])]
            rows.extend(diagnostic.collect(args.checkpoint, "transition", source, target, 20, 9.5))
        elif args.chunk.startswith("yaw_probe"):
            speeds = ({
                "yaw_probe_02": 0.2, "yaw_probe_04": 0.4,
                "yaw_probe_06": 0.6, "yaw_probe_12": 1.2,
            }[args.chunk],)
            for speed in speeds:
                for yaw_cmd in (-0.10, -0.05, 0.0, 0.05, 0.10):
                    rows.extend(diagnostic.collect(
                        "stage7_selected", "yaw_probe", speed, speed, 20, 5.0, yaw_command=yaw_cmd
                    ))
        elif args.chunk == "feedback":
            for speed in (0.2, 0.3, 0.4, 0.5, 0.6):
                rows.extend(diagnostic.collect(
                    "stage7_selected", "steady", speed, speed, 20, 8.0, feedback=True
                ))
            for source, target in ((0.0, 0.4), (0.0, 0.6), (0.6, 0.4), (0.6, 0.0)):
                rows.extend(diagnostic.collect(
                    "stage7_selected", "transition", source, target, 20, 9.5, feedback=True
                ))
        elif args.chunk.startswith("feedback_steady_"):
            speed = {
                "02": 0.2, "03": 0.3, "04": 0.4, "05": 0.5, "06": 0.6,
            }[args.chunk.rsplit("_", 1)[1]]
            rows.extend(diagnostic.collect(
                "stage7_selected", "steady", speed, speed, 20, 8.0, feedback=True
            ))
        elif args.chunk.startswith("feedback_transition_"):
            source, target = ((0.0, 0.4), (0.0, 0.6), (0.6, 0.4), (0.6, 0.0))[
                int(args.chunk.rsplit("_", 1)[1])
            ]
            rows.extend(diagnostic.collect(
                "stage7_selected", "transition", source, target, 20, 9.5, feedback=True
            ))
        label = args.checkpoint or "stage7_selected"
        dump(f"raw_{label}_{args.chunk}.json", rows)
        diagnostic.close()
        return
    all_rows = []
    if args.aggregate_only:
        for path in sorted(args.output.glob("raw_*.json")):
            if "yaw_probe" in path.name or "feedback" in path.name:
                continue
            all_rows.extend(json.loads(path.read_text(encoding="utf-8")))
    else:
        for checkpoint in CHECKPOINTS:
            for speed in STEADY:
                all_rows.extend(diagnostic.collect(checkpoint, "steady", speed, speed, 50, 8.0))
            for source, target in LOW_TRANSITIONS:
                all_rows.extend(diagnostic.collect(checkpoint, "transition", source, target, 50, 9.5))
            for source, target in ANCHOR_TRANSITIONS:
                all_rows.extend(diagnostic.collect(checkpoint, "transition", source, target, 20, 9.5))
    by_speed, by_transition = {}, {}
    for checkpoint in CHECKPOINTS:
        by_speed[checkpoint] = {}
        for speed in STEADY:
            rows = [row for row in all_rows if row["checkpoint"] == checkpoint and row["family"] == "steady" and row["target_speed"] == speed]
            by_speed[checkpoint][str(speed)] = aggregate_heading(rows)
        by_transition[checkpoint] = {}
        for source, target in LOW_TRANSITIONS + ANCHOR_TRANSITIONS:
            key = f"{source:g}_to_{target:g}"
            rows = [row for row in all_rows if row["checkpoint"] == checkpoint and row["family"] == "transition" and row["source_speed"] == source and row["target_speed"] == target]
            by_transition[checkpoint][key] = aggregate_heading(rows)
    dump("heading_decomposition_by_speed.json", by_speed)
    dump("heading_decomposition_by_transition.json", by_transition)
    time_rows = []
    direction_rows = []
    for row in all_rows:
        time_rows.append({
            key: row[key] for key in (
                "checkpoint", "family", "condition", "episode", "episode_seed", "fall",
                "signed_yaw_drift_slope", "absolute_yaw_drift_slope", "yaw_rate_mean",
                "yaw_rate_p50", "yaw_rate_p90", "yaw_rate_p95", "yaw_rate_p99",
                "heading_abs_p50", "heading_abs_p90", "heading_abs_p95", "heading_abs_p99",
                "heading_abs_max", "oscillatory_rms", "rare_excursion",
            )
        })
        direction_rows.append({
            "checkpoint": row["checkpoint"], "family": row["family"], "condition": row["condition"],
            "episode": row["episode"], "episode_seed": row["episode_seed"], "fall": row["fall"],
            "final_signed_heading_error": row["heading_final_signed"],
            "drift_sign": "left_positive" if row["heading_final_signed"] > 0 else "right_negative",
            "initial_contact_pattern": row["initial_contact_pattern"],
        })
    write_csv("heading_time_series_summary.csv", time_rows)
    write_csv("heading_direction_by_seed.csv", direction_rows)
    signed = {}
    for checkpoint in CHECKPOINTS:
        signed[checkpoint] = {}
        for condition in sorted(set(row["condition"] for row in all_rows)):
            rows = [row for row in all_rows if row["checkpoint"] == checkpoint and row["condition"] == condition and not row["fall"]]
            if not rows:
                continue
            positive = mean(row["heading_final_signed"] > 0 for row in rows)
            signed[checkpoint][condition] = {
                "positive_fraction": positive, "negative_fraction": 1 - positive,
                "systematic_direction_80pct": positive >= 0.8 or positive <= 0.2,
                "median": percentile((row["heading_final_signed"] for row in rows), 50),
                "p05": percentile((row["heading_final_signed"] for row in rows), 5),
                "p95": percentile((row["heading_final_signed"] for row in rows), 95),
            }
    dump("signed_heading_distribution.json", signed)
    joint_rows, action_summary = [], {}
    for checkpoint in CHECKPOINTS:
        action_summary[checkpoint] = {}
        for condition in sorted(set(row["condition"] for row in all_rows)):
            rows = [row for row in all_rows if row["checkpoint"] == checkpoint and row["condition"] == condition]
            if not rows:
                continue
            pair_summary = {}
            for label, left, right, sign in LR_JOINT_PAIRS:
                values = [row["pair_action_asymmetry"][label] for row in rows]
                pair_summary[label] = {
                    key: mean(value[key] for value in values) for key in values[0]
                }
                joint_rows.append({
                    "checkpoint": checkpoint, "condition": condition, "pair": label,
                    "left_joint": JOINTS[left], "right_joint": JOINTS[right],
                    "mirror_sign": sign, **pair_summary[label],
                })
            action_summary[checkpoint][condition] = pair_summary
    dump("left_right_action_asymmetry.json", action_summary)
    write_csv("per_joint_action_asymmetry.csv", joint_rows)
    contact_summary, foot_corr_rows = {}, []
    for checkpoint in CHECKPOINTS:
        contact_summary[checkpoint] = {}
        for condition in sorted(set(row["condition"] for row in all_rows)):
            rows = [row for row in all_rows if row["checkpoint"] == checkpoint and row["condition"] == condition]
            if not rows:
                continue
            contact_summary[checkpoint][condition] = {
                "left_right_force_difference_mean": mean(row["left_right_force_difference"] for row in rows),
                "left_right_slip_difference_mean": mean(row["left_right_slip_difference"] for row in rows),
                "per_foot": {
                    foot: {
                        key: mean(row["per_foot"][foot][key] for row in rows)
                        for key in ("duty_factor", "force_mean_n", "force_p95_n", "foot_placement_y_mean", "stride_length_mean", "slip_speed_p95", "stance_displacement_p95")
                    } for foot in FEET
                },
            }
            for foot in FEET:
                foot_corr_rows.append({
                    "checkpoint": checkpoint, "condition": condition, "foot": foot,
                    "heading_vs_displacement_pearson": corr(
                        [row["heading_final_signed"] for row in rows],
                        [row["per_foot"][foot]["stance_displacement_p95"] for row in rows],
                    ),
                    "heading_vs_force_pearson": corr(
                        [row["heading_final_signed"] for row in rows],
                        [row["per_foot"][foot]["force_mean_n"] for row in rows],
                    ),
                })
    dump("left_right_contact_asymmetry.json", contact_summary)
    write_csv("per_foot_contact_heading_correlation.csv", foot_corr_rows)
    coupling = {}
    for checkpoint in CHECKPOINTS:
        rows = [row for row in all_rows if row["checkpoint"] == checkpoint and not row["fall"]]
        coupling[checkpoint] = {}
        for label, field in (
            ("slip_difference", "left_right_slip_difference"),
            ("force_difference", "left_right_force_difference"),
        ):
            x = [row[field] for row in rows]
            coupling[checkpoint][label] = {
                "heading_signed_pearson": corr(x, [row["heading_final_signed"] for row in rows]),
                "heading_signed_spearman": corr(x, [row["heading_final_signed"] for row in rows], True),
                "heading_absolute_spearman": corr(x, [row["heading_abs_p95"] for row in rows], True),
                "yaw_bias_spearman": corr(x, [row["yaw_rate_mean"] for row in rows], True),
            }
        coupling[checkpoint]["speed_conditioned"] = {}
        for speed in STEADY:
            subset = [row for row in rows if row["family"] == "steady" and row["target_speed"] == speed]
            coupling[checkpoint]["speed_conditioned"][str(speed)] = {
                "slip_heading_spearman": corr(
                    [row["left_right_slip_difference"] for row in subset],
                    [row["heading_final_signed"] for row in subset], True,
                )
            }
    dump("slip_heading_coupling.json", coupling)
    selected_rows = [row for row in all_rows if row["checkpoint"] == "stage7_selected"]
    init = {
        "initial_roll_vs_heading_spearman": corr(
            [row["initial_roll"] for row in selected_rows], [row["heading_final_signed"] for row in selected_rows], True
        ),
        "initial_pitch_vs_heading_spearman": corr(
            [row["initial_pitch"] for row in selected_rows], [row["heading_final_signed"] for row in selected_rows], True
        ),
        "initial_force_bias_vs_heading_spearman": corr(
            [row["initial_left_force"] - row["initial_right_force"] for row in selected_rows],
            [row["heading_final_signed"] for row in selected_rows], True,
        ),
        "by_initial_contact_pattern": {},
        "initial_yaw_removed_by_reference": True,
    }
    for pattern in sorted(set(row["initial_contact_pattern"] for row in selected_rows)):
        subset = [row for row in selected_rows if row["initial_contact_pattern"] == pattern]
        init["by_initial_contact_pattern"][pattern] = {
            "episodes": len(subset), "positive_drift_fraction": mean(row["heading_final_signed"] > 0 for row in subset),
            "heading_abs_p95": percentile((row["heading_abs_p95"] for row in subset), 95),
        }
    dump("heading_initialization_sensitivity.json", init)
    phase_output = {}
    for checkpoint in CHECKPOINTS:
        phase_output[checkpoint] = {}
        for key in tuple(f"{a:g}_to_{b:g}" for a, b in LOW_TRANSITIONS + ANCHOR_TRANSITIONS):
            rows = [row for row in all_rows if row["checkpoint"] == checkpoint and row["condition"] == key.replace("_to_", "->")]
            if not rows:
                continue
            phase_output[checkpoint][key] = {
                phase: {
                    metric: mean(row["phases"].get(phase, {}).get(metric, 0.0) for row in rows)
                    for metric in ("heading_change", "yaw_rate_p95", "left_right_action_difference", "left_right_force_difference")
                } for phase in ("source_hold", "ramp", "target_acquisition", "target_hold")
            }
    dump("transition_heading_phase_analysis.json", phase_output)
    # Local yaw-rate response probe.
    probe_rows = []
    if args.aggregate_only:
        for path in sorted(args.output.glob("raw_stage7_selected_yaw_probe_*.json")):
            probe_rows.extend(json.loads(path.read_text(encoding="utf-8")))
    else:
        for speed in (0.2, 0.4, 0.6, 1.2):
            for yaw_cmd in (-0.10, -0.05, 0.0, 0.05, 0.10):
                probe_rows.extend(diagnostic.collect(
                    "stage7_selected", "yaw_probe", speed, speed, 20, 5.0, yaw_command=yaw_cmd
                ))
    probe = {}
    monotonic_checks, signed_checks, safety_checks = [], [], []
    for speed in (0.2, 0.4, 0.6, 1.2):
        probe[str(speed)] = {}
        means = []
        for yaw_cmd in (-0.10, -0.05, 0.0, 0.05, 0.10):
            rows = [row for row in probe_rows if row["target_speed"] == speed and abs(row["yaw_command_mean"] - yaw_cmd) < 1e-4]
            actual = mean(row["yaw_rate_mean"] for row in rows)
            means.append(actual)
            signed_checks.append(yaw_cmd == 0 or actual * yaw_cmd > 0)
            safety_checks.append(mean(row["fall"] for row in rows) <= 0.05)
            probe[str(speed)][str(yaw_cmd)] = {
                "actual_yaw_rate_mean": actual,
                "signed_heading_change_mean": mean(row["heading_final_signed"] for row in rows),
                "fall_rate": mean(row["fall"] for row in rows),
                "speed_mae": mean(row["speed_mae"] for row in rows),
                "response_gain": actual / yaw_cmd if yaw_cmd else None,
                "left_right_slip_difference": mean(row["left_right_slip_difference"] for row in rows),
            }
        monotonic_checks.append(all(a <= b + 0.01 for a, b in zip(means, means[1:])))
    controllable = all(monotonic_checks) and mean(signed_checks) >= 0.9 and all(safety_checks)
    dump("yaw_rate_command_response.json", {
        "diagnostic_only": True, "results": probe,
        "monotonic_by_speed": monotonic_checks, "signed_response_fraction": mean(signed_checks),
        "safety_checks_pass": all(safety_checks),
    })
    yaw_class = "YAW_RATE_CONTROLLABLE" if controllable else "YAW_RATE_NOT_LOCALLY_CONTROLLABLE"
    dump("yaw_controllability_classification.json", {"classification": yaw_class})
    feedback_result = {"executed": False, "reason": "yaw-rate response not locally controllable"}
    if controllable:
        feedback_rows = []
        if args.aggregate_only:
            feedback_paths = sorted(args.output.glob("raw_stage7_selected_feedback*.json"))
            if feedback_paths:
                for feedback_path in feedback_paths:
                    feedback_rows.extend(json.loads(feedback_path.read_text(encoding="utf-8")))
            else:
                dump("fixed_heading_feedback_diagnostic.json", {
                    "executed": False,
                    "reason": "eligible and awaiting frozen feedback diagnostic rollout",
                })
                return
        else:
            for speed in (0.2, 0.3, 0.4, 0.5, 0.6):
                feedback_rows.extend(diagnostic.collect(
                    "stage7_selected", "steady", speed, speed, 20, 8.0, feedback=True
                ))
            for source, target in ((0.0, 0.4), (0.0, 0.6), (0.6, 0.4), (0.6, 0.0)):
                feedback_rows.extend(diagnostic.collect(
                    "stage7_selected", "transition", source, target, 20, 9.5, feedback=True
                ))
        comparison = {}
        gates = []
        for family, source, target in (
            [("steady", speed, speed) for speed in (0.2, 0.3, 0.4, 0.5, 0.6)]
            + [("transition", a, b) for a, b in ((0.0, 0.4), (0.0, 0.6), (0.6, 0.4), (0.6, 0.0))]
        ):
            key = f"{source:g}_to_{target:g}" if family == "transition" else f"{target:g}"
            feedback_subset = [row for row in feedback_rows if row["family"] == family and row["source_speed"] == source and row["target_speed"] == target]
            open_subset = [row for row in selected_rows if row["family"] == family and row["source_speed"] == source and row["target_speed"] == target][:20]
            fb_heading = percentile((row["heading_abs_p95"] for row in feedback_subset if not row["fall"]), 95)
            open_heading = percentile((row["heading_abs_p95"] for row in open_subset if not row["fall"]), 95)
            fb_fall, open_fall = mean(row["fall"] for row in feedback_subset), mean(row["fall"] for row in open_subset)
            fb_mae, open_mae = mean(row["speed_mae"] for row in feedback_subset), mean(row["speed_mae"] for row in open_subset)
            fb_slip = percentile((abs(row["left_right_slip_difference"]) for row in feedback_subset), 95)
            open_slip = percentile((abs(row["left_right_slip_difference"]) for row in open_subset), 95)
            checks = {
                "heading_p95_le_0.12": fb_heading <= 0.12,
                "fall_not_worse": fb_fall <= open_fall,
                "speed_mae_degradation_le_0.05": fb_mae - open_mae <= 0.05,
                "slip_not_50pct_worse": fb_slip <= 1.5 * max(open_slip, 1e-6),
                "saturation_not_worse": mean(
                    row["long_dwell_saturation"] for row in feedback_subset
                ) <= mean(row["long_dwell_saturation"] for row in open_subset),
            }
            gates.append(all(checks.values()))
            comparison[key] = {
                "open_heading_p95": open_heading, "feedback_heading_p95": fb_heading,
                "open_fall": open_fall, "feedback_fall": fb_fall,
                "open_speed_mae": open_mae, "feedback_speed_mae": fb_mae,
                "open_slip_asymmetry_p95": open_slip, "feedback_slip_asymmetry_p95": fb_slip,
                "checks": checks, "pass": all(checks.values()),
            }
        feedback_result = {
            "executed": True, "diagnostic_upper_bound_only": True,
            "controller": {"kp": 1.0, "omega_max": 0.10, "error": "wrap(reference-current yaw)"},
            "comparison": comparison, "all_conditions_pass": all(gates),
            "production_adopted": False,
        }
    dump("fixed_heading_feedback_diagnostic.json", feedback_result)
    if diagnostic is not None:
        diagnostic.close()


try:
    main()
finally:
    simulation_app.close()
