"""Deterministic Stage 9 contact-kinematics rollout; no optimizer update."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage9_contact_kinematics_heading_diagnosis"
CHECKPOINTS = {
    "official_parent": REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt",
    "stage4_selected": REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training/checkpoints/model_50.pt",
    "stage7_selected": REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization/checkpoints/model_50.pt",
}
STEADY = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2, 2.0)
LOW_TRANSITIONS = ((0.0, 0.2), (0.0, 0.4), (0.0, 0.6), (0.6, 0.4), (0.6, 0.2), (0.6, 0.0))
ANCHOR_TRANSITIONS = ((0.0, 1.2), (1.2, 2.0), (2.0, 1.2), (1.2, 0.0))
CHUNKS = (
    *(f"steady_{index}" for index in range(5)),
    *(f"low_{index}" for index in range(3)),
    "anchors_0", "anchors_1",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", choices=tuple(CHECKPOINTS), required=True)
parser.add_argument("--chunk", choices=CHUNKS, required=True)
parser.add_argument("--output", type=Path, default=OUT)
parser.add_argument("--num-envs", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_physx.sensors import ContactSensorCfg  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
from go2_bidirectional.command_profiles import transition_command  # noqa: E402
from go2_bidirectional.contact_kinematics import (  # noqa: E402
    maximum_contiguous_duration,
    stable_contact_mask,
)
from go2_bidirectional.evaluation import build_runner  # noqa: E402
from go2_bidirectional.stage6_endpoint_protocol import (  # noqa: E402
    circular_median,
    quat_xyzw_to_roll_pitch_yaw_torch,
)
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

SEED = 20268901
DT = 0.02
MU = 0.6
FEET = ("FL", "FR", "RL", "RR")
ASSET_FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GROUND = "/World/ground/terrain/GroundPlane/CollisionPlane"
SPEED_LEVELS = (0.02, 0.05, 0.10, 0.20, 0.30)
DURATION_LEVELS = (0.04, 0.10, 0.20)


def dump(path, value):
    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=encode) + "\n", encoding="utf-8")


def percentile(values, q):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else 0.0


def mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def rotate_inverse_xyzw(quaternion, vector):
    qv = -quaternion[..., :3]
    qw = quaternion[..., 3:4]
    return vector + 2.0 * (
        qw * torch.linalg.cross(qv, vector)
        + torch.linalg.cross(qv, torch.linalg.cross(qv, vector))
    )


def runs(mask):
    result, start = [], None
    for index, value in enumerate(np.r_[np.asarray(mask, dtype=bool), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index))
            start = None
    return result


class Diagnostic:
    def __init__(self):
        cfg, agent = resolve_task_config("Isaac-Velocity-Flat-Unitree-Go2-v0", "rsl_rl_cfg_entry_point")
        cfg.scene.num_envs = args.num_envs
        cfg.seed = SEED
        cfg.episode_length_s = 60.0
        cfg.observations.policy.enable_corruption = False
        cfg.events.base_external_force_torque = None
        cfg.events.push_robot = None
        for label, foot in zip(("fl", "fr", "rl", "rr"), ASSET_FEET):
            setattr(cfg.scene, f"stage9_{label}", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot}",
                update_period=0.0,
                track_pose=True,
                track_contact_points=True,
                track_friction_forces=True,
                max_contact_data_count_per_prim=16,
                filter_prim_paths_expr=[GROUND],
            ))
        if args.device:
            cfg.sim.device = args.device
            agent.device = args.device
        self.raw = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
        self.wrapped, self.runner, self.policy = build_runner(
            self.raw, agent, CHECKPOINTS[args.checkpoint]
        )
        self.env = self.wrapped.unwrapped
        self.robot = self.env.scene["robot"]
        self.command = self.env.command_manager.get_term("base_velocity")
        self.body_ids = [int(self.robot.find_bodies(name)[0][0]) for name in ASSET_FEET]
        self.sensors = [self.env.scene.sensors[f"stage9_{label}"] for label in ("fl", "fr", "rl", "rr")]

    def detailed_foot(self, sensor, body_id):
        normal_force, point, normal, separation, count, start = sensor.contact_view.get_contact_data(
            dt=sensor._sim_physics_dt
        )
        friction_force, friction_point, friction_count, friction_start = sensor.contact_view.get_friction_data(
            dt=sensor._sim_physics_dt
        )
        normal_force = wp.to_torch(normal_force).reshape(-1)
        point = wp.to_torch(point).reshape(-1, 3)
        normal = wp.to_torch(normal).reshape(-1, 3)
        separation = wp.to_torch(separation).reshape(-1)
        count = wp.to_torch(count).reshape(args.num_envs, -1)[:, 0].long()
        start = wp.to_torch(start).reshape(args.num_envs, -1)[:, 0].long()
        friction_force = wp.to_torch(friction_force).reshape(-1, 3)
        friction_point = wp.to_torch(friction_point).reshape(-1, 3)
        friction_count = wp.to_torch(friction_count).reshape(args.num_envs, -1)[:, 0].long()
        friction_start = wp.to_torch(friction_start).reshape(args.num_envs, -1)[:, 0].long()
        width = 16
        offsets = torch.arange(width, device=point.device)
        indices = (start[:, None] + offsets).clamp(0, max(0, len(point) - 1))
        mask = offsets[None] < count[:, None]
        points = point[indices]
        normals = normal[indices]
        fn = normal_force[indices] * mask
        separations = separation[indices]
        unit = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        body_pos = self.robot.data.body_pos_w.torch[:, body_id]
        body_quat = self.robot.data.body_quat_w.torch[:, body_id]
        linear = self.robot.data.body_lin_vel_w.torch[:, body_id]
        angular = self.robot.data.body_ang_vel_w.torch[:, body_id]
        radius = points - body_pos[:, None]
        surface = linear[:, None] + torch.linalg.cross(angular[:, None].expand_as(radius), radius)
        tangent = surface - (surface * unit).sum(-1, keepdim=True) * unit
        tangent_speed = tangent.norm(dim=-1) * mask
        fn_sum = fn.sum(1)
        weighted_tangent = (tangent_speed * fn).sum(1) / fn_sum.clamp_min(1e-12)
        max_tangent = tangent_speed.masked_fill(~mask, 0.0).amax(1)
        centroid = (points * fn[..., None]).sum(1) / fn_sum[:, None].clamp_min(1e-12)
        centroid[fn_sum <= 0] = float("nan")
        local_centroid = rotate_inverse_xyzw(body_quat, centroid - body_pos)
        normal_vectors = fn[..., None] * unit
        root = self.robot.data.root_com_pos_w.torch
        normal_yaw = torch.linalg.cross(points - root[:, None], normal_vectors)[..., 2]
        normal_yaw = (normal_yaw * mask).sum(1)
        fi = (friction_start[:, None] + offsets).clamp(0, max(0, len(friction_force) - 1))
        fmask = offsets[None] < friction_count[:, None]
        ft = friction_force[fi] * fmask[..., None]
        fp = friction_point[fi]
        ft_sum = ft.sum(1)
        ft_norm = ft_sum.norm(dim=-1)
        utilization = ft_norm / (MU * fn_sum).clamp_min(1e-12)
        tangential_yaw = torch.linalg.cross(fp - root[:, None], ft)[..., 2]
        tangential_yaw = (tangential_yaw * fmask).sum(1)
        separation_min = separations.masked_fill(~mask, float("inf")).amin(1)
        separation_min[~torch.isfinite(separation_min)] = float("nan")
        return {
            "contact": fn_sum > 5.0,
            "normal_force": fn_sum,
            "tangential_force": ft_norm,
            "friction_utilization": utilization,
            "tangent_mean": weighted_tangent,
            "tangent_max": max_tangent,
            "centroid": centroid,
            "local_centroid": local_centroid,
            "contact_count": count,
            "friction_count": friction_count,
            "separation": separation_min,
            "normal_yaw_moment": normal_yaw,
            "tangential_yaw_moment": tangential_yaw,
            "foot_origin_speed": linear[:, :2].norm(dim=-1),
        }

    def collect(self, family, source, target, episodes, duration):
        self.env.seed(SEED)
        self.wrapped.reset()
        n = episodes
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=self.env.device)
        falls = torch.zeros(args.num_envs, dtype=torch.bool, device=self.env.device)
        fall_step = torch.full((args.num_envs,), -1, dtype=torch.long, device=self.env.device)
        series = {key: [] for key in (
            "yaw", "yaw_rate", "yaw_accel", "vx", "root_height",
            "contact", "normal_force", "tangential_force", "friction_utilization",
            "tangent_mean", "tangent_max", "centroid", "local_centroid",
            "contact_count", "friction_count", "separation",
            "normal_yaw_moment", "tangential_yaw_moment", "foot_origin_speed",
        )}
        phases = []
        previous_yaw_rate = self.robot.data.root_ang_vel_w.torch[:, 2].clone()
        for step in range(round(duration / DT)):
            t = step * DT
            if family == "transition":
                speed, phase = transition_command(t, source, target, 1.5)
            else:
                speed, phase = target, "steady"
            phases.append(phase)
            self.command.vel_command_b[:, 0] = speed
            self.command.vel_command_b[:, 1:] = 0.0
            with torch.inference_mode():
                action = self.policy(self.wrapped.get_observations())
                _, _, dones, _ = self.wrapped.step(action)
            _, _, yaw = quat_xyzw_to_roll_pitch_yaw_torch(self.robot.data.root_quat_w.torch)
            yaw_rate = self.robot.data.root_ang_vel_w.torch[:, 2]
            yaw_accel = (yaw_rate - previous_yaw_rate) / DT
            previous_yaw_rate = yaw_rate.clone()
            detailed = [self.detailed_foot(sensor, body) for sensor, body in zip(self.sensors, self.body_ids)]
            series["yaw"].append(yaw[:n].cpu())
            series["yaw_rate"].append(yaw_rate[:n].cpu())
            series["yaw_accel"].append(yaw_accel[:n].cpu())
            series["vx"].append(self.robot.data.root_lin_vel_b.torch[:n, 0].cpu())
            series["root_height"].append(self.robot.data.root_com_pos_w.torch[:n, 2].cpu())
            for key in series:
                if key in ("yaw", "yaw_rate", "yaw_accel", "vx", "root_height"):
                    continue
                series[key].append(torch.stack([item[key][:n] for item in detailed], 1).cpu())
            newly = dones.bool() & alive
            fall_step[newly] = step
            falls |= newly
            alive &= ~dones.bool()
        arrays = {key: torch.stack(values).numpy() for key, values in series.items()}
        rows = []
        for episode in range(n):
            end = int(fall_step[episode]) if fall_step[episode] >= 0 else len(phases)
            end = max(2, end)
            if family == "transition":
                heading_ref = circular_median(arrays["yaw"][100:min(150, end), episode].tolist())
                quality_start = min(round(5.5 / DT), end - 1)
            elif target == 0:
                quality_start = min(round(1.0 / DT), end - 1)
                heading_ref = float(arrays["yaw"][quality_start, episode])
            else:
                quality_start = min(round(2.0 / DT), end - 1)
                heading_ref = float(arrays["yaw"][quality_start, episode])
            quality = np.arange(quality_start, end)
            heading = wrap(arrays["yaw"][quality, episode] - heading_ref)
            times = quality * DT
            slope = float(np.polyfit(times, heading, 1)[0]) if len(times) >= 2 else 0.0
            foot_rows = {}
            for foot, foot_name in enumerate(FEET):
                contact = arrays["contact"][:end, episode, foot].astype(bool)
                stable, boundary = stable_contact_mask(contact, 3, 2)
                stable_quality = stable.copy()
                stable_quality[:quality_start] = False
                tangent = arrays["tangent_mean"][:end, episode, foot]
                tangent_max = arrays["tangent_max"][:end, episode, foot]
                utilization = arrays["friction_utilization"][:end, episode, foot]
                fn = arrays["normal_force"][:end, episode, foot]
                ft = arrays["tangential_force"][:end, episode, foot]
                centroid = arrays["centroid"][:end, episode, foot]
                local = arrays["local_centroid"][:end, episode, foot]
                origin_speed = arrays["foot_origin_speed"][:end, episode, foot]
                legacy_anchor = np.zeros(end)
                successor_speed = np.zeros(end)
                migration = np.zeros(end)
                for start, stop in runs(stable_quality):
                    anchor = np.nanmedian(centroid[start:min(stop, start + 3)], axis=0)
                    legacy_anchor[start:stop] = np.linalg.norm(centroid[start:stop, :2] - anchor[:2], axis=1)
                    if stop - start > 1:
                        successor_speed[start + 1:stop] = np.linalg.norm(
                            np.diff(centroid[start:stop, :2], axis=0), axis=1
                        ) / DT
                        migration[start + 1:stop] = np.linalg.norm(
                            np.diff(local[start:stop], axis=0), axis=1
                        ) / DT
                diagnostic_levels = {}
                for speed_level in SPEED_LEVELS:
                    above = stable_quality & (tangent > speed_level)
                    for duration_level in DURATION_LEVELS:
                        key = f"gt_{speed_level:.2f}_ge_{duration_level:.2f}s"
                        max_duration = maximum_contiguous_duration(above, DT)
                        diagnostic_levels[key] = {
                            "occurrence": max_duration >= duration_level,
                            "stable_contact_time_fraction": mean(above[stable_quality]) if stable_quality.any() else 0.0,
                            "maximum_contiguous_duration_s": max_duration,
                        }
                values = tangent[stable_quality]
                util_values = utilization[stable_quality]
                legacy_values = legacy_anchor[stable_quality]
                rolling_candidate = stable_quality & (legacy_anchor > 0.03) & (tangent < 0.05)
                foot_rows[foot_name] = {
                    "all_contact_samples": int(contact.sum()),
                    "stable_contact_samples": int(stable_quality.sum()),
                    "boundary_samples": int(boundary.sum()),
                    "stable_contact_time_fraction": mean(stable_quality[quality]),
                    "tangent_speed_p50": percentile(values, 50),
                    "tangent_speed_p90": percentile(values, 90),
                    "tangent_speed_p95": percentile(values, 95),
                    "tangent_speed_p99": percentile(values, 99),
                    "tangent_speed_max": percentile(values, 100),
                    "tangent_max_point_p95": percentile(tangent_max[stable_quality], 95),
                    "friction_utilization_p50": percentile(util_values, 50),
                    "friction_utilization_p90": percentile(util_values, 90),
                    "friction_utilization_p95": percentile(util_values, 95),
                    "friction_utilization_p99": percentile(util_values, 99),
                    "friction_utilization_max": percentile(util_values, 100),
                    "friction_cone_exceedance_rate": mean(util_values > 1.0),
                    "normal_force_mean": mean(fn[stable_quality]),
                    "normal_force_p95": percentile(fn[stable_quality], 95),
                    "tangential_force_mean": mean(ft[stable_quality]),
                    "tangential_force_p95": percentile(ft[stable_quality], 95),
                    "normal_yaw_moment_mean": mean(arrays["normal_yaw_moment"][:end, episode, foot][stable_quality]),
                    "tangential_yaw_moment_mean": mean(arrays["tangential_yaw_moment"][:end, episode, foot][stable_quality]),
                    "net_yaw_moment_mean": mean(
                        (arrays["normal_yaw_moment"][:end, episode, foot]
                         + arrays["tangential_yaw_moment"][:end, episode, foot])[stable_quality]
                    ),
                    "legacy_anchor_displacement_p95": percentile(legacy_values, 95),
                    "legacy_successive_point_speed_p95": percentile(successor_speed[stable_quality], 95),
                    "foot_link_origin_speed_p95": percentile(origin_speed[stable_quality], 95),
                    "foot_local_contact_migration_speed_p95": percentile(migration[stable_quality], 95),
                    "rolling_candidate_sample_fraction": mean(rolling_candidate[stable_quality]) if stable_quality.any() else 0.0,
                    "contact_count_mean": mean(arrays["contact_count"][:end, episode, foot][stable_quality]),
                    "friction_point_count_mean": mean(arrays["friction_count"][:end, episode, foot][stable_quality]),
                    "separation_p05": percentile(arrays["separation"][:end, episode, foot][stable_quality], 5),
                    "diagnostic_levels": diagnostic_levels,
                }
            left = ("FL", "RL")
            right = ("FR", "RR")
            def side(metric, names):
                return mean([foot_rows[name][metric] for name in names])
            net_moment_series = (
                arrays["normal_yaw_moment"][:end, episode]
                + arrays["tangential_yaw_moment"][:end, episode]
            ).sum(1)
            row = {
                "checkpoint": args.checkpoint,
                "family": family,
                "condition": f"{source:g}->{target:g}" if family == "transition" else f"{target:g}",
                "source_speed": source,
                "target_speed": target,
                "episode": episode,
                "episode_seed": SEED + episode,
                "fall": bool(falls[episode]),
                "heading_drift_slope": slope,
                "final_signed_heading_error": float(heading[-1]) if len(heading) else 0.0,
                "heading_p95": percentile(np.abs(heading), 95),
                "yaw_rate_mean": mean(arrays["yaw_rate"][quality, episode]),
                "yaw_rate_p95": percentile(np.abs(arrays["yaw_rate"][quality, episode]), 95),
                "yaw_acceleration_mean": mean(arrays["yaw_accel"][quality, episode]),
                "speed_mean": mean(arrays["vx"][quality, episode]),
                "speed_mae": mean(np.abs(arrays["vx"][quality, episode] - target)),
                "net_contact_yaw_moment_mean": mean(net_moment_series[quality]),
                "net_contact_yaw_moment_p95": percentile(np.abs(net_moment_series[quality]), 95),
                "left_yaw_moment_mean": side("net_yaw_moment_mean", left),
                "right_yaw_moment_mean": side("net_yaw_moment_mean", right),
                "front_yaw_moment_mean": side("net_yaw_moment_mean", ("FL", "FR")),
                "rear_yaw_moment_mean": side("net_yaw_moment_mean", ("RL", "RR")),
                "left_tangent_p95": side("tangent_speed_p95", left),
                "right_tangent_p95": side("tangent_speed_p95", right),
                "left_right_tangent_difference": side("tangent_speed_p95", left) - side("tangent_speed_p95", right),
                "left_utilization_p95": side("friction_utilization_p95", left),
                "right_utilization_p95": side("friction_utilization_p95", right),
                "left_right_utilization_difference": side("friction_utilization_p95", left) - side("friction_utilization_p95", right),
                "left_tangential_force_mean": side("tangential_force_mean", left),
                "right_tangential_force_mean": side("tangential_force_mean", right),
                "left_right_tangential_force_difference": side("tangential_force_mean", left) - side("tangential_force_mean", right),
                "left_legacy_displacement_p95": side("legacy_anchor_displacement_p95", left),
                "right_legacy_displacement_p95": side("legacy_anchor_displacement_p95", right),
                "left_right_legacy_difference": side("legacy_anchor_displacement_p95", left) - side("legacy_anchor_displacement_p95", right),
                "left_right_yaw_moment_difference": side("net_yaw_moment_mean", left) - side("net_yaw_moment_mean", right),
                "feet": foot_rows,
            }
            rows.append(row)
        print(f"STAGE9 {args.checkpoint} {family} {source}->{target} n={episodes}", flush=True)
        return rows

    def close(self):
        self.wrapped.close()


def conditions_for_chunk():
    if args.chunk.startswith("steady_"):
        index = int(args.chunk.rsplit("_", 1)[1])
        return [("steady", speed, speed, 50, 8.0) for speed in STEADY[index * 2:index * 2 + 2]]
    if args.chunk.startswith("low_"):
        index = int(args.chunk.rsplit("_", 1)[1])
        return [
            ("transition", source, target, 50, 9.5)
            for source, target in LOW_TRANSITIONS[index * 2:index * 2 + 2]
        ]
    index = int(args.chunk.rsplit("_", 1)[1])
    return [
        ("transition", source, target, 20, 9.5)
        for source, target in ANCHOR_TRANSITIONS[index * 2:index * 2 + 2]
    ]


def main():
    if args.num_envs != 50:
        raise SystemExit("Stage 9 requires 50 environments")
    diagnostic = Diagnostic()
    rows = []
    for condition in conditions_for_chunk():
        rows.extend(diagnostic.collect(*condition))
    dump(args.output / f"raw_{args.checkpoint}_{args.chunk}.json", rows)
    diagnostic.close()


try:
    main()
except Exception as exc:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"error_{args.checkpoint}_{args.chunk}.txt").write_text(
        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", encoding="utf-8"
    )
    raise
finally:
    simulation_app.close()
