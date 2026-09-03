"""Validate checkpoints and formally evaluate Stage 11 with the frozen controller."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage11_tangential_slip_reduction"
PARENT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization/checkpoints/model_50.pt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("validation", "formal"), required=True)
parser.add_argument("--num-envs", type=int)
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
from go2_bidirectional.command_profiles import sequence_command, transition_command  # noqa: E402
from go2_bidirectional.evaluation import build_runner  # noqa: E402
from go2_bidirectional.phase_gated_heading import PhaseGatedFixedHeadingController, target_tolerance  # noqa: E402
from go2_bidirectional.stage6_endpoint_protocol import (  # noqa: E402
    quat_xyzw_to_gravity_tilt_torch,
    quat_xyzw_to_roll_pitch_yaw_torch,
)
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

DT = 0.02
MU = 0.6
GROUND = "/World/ground/terrain/GroundPlane/CollisionPlane"
FEET = ("FL", "FR", "RL", "RR")
ASSET_FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
VALID_STEADY = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.2, 2.0)
FORMAL_STEADY = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
TRANSITIONS = (
    (0.0, 0.2), (0.0, 0.4), (0.0, 0.6),
    (0.6, 0.4), (0.6, 0.2), (0.6, 0.0),
    (0.0, 1.2), (1.2, 2.0), (2.0, 1.2), (1.2, 0.0),
)
VALID_TRANSITIONS = (
    (0.0, 0.2), (0.0, 0.4), (0.0, 0.6), (0.6, 0.0),
    (0.0, 1.2), (1.2, 2.0), (2.0, 1.2), (1.2, 0.0),
)
SEQUENCE = (0.0, 0.6, 1.2, 2.0, 1.2, 0.6, 0.0)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def percentile(values, q):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else 0.0


def mean(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


class Evaluator:
    def __init__(self, num_envs, seed):
        cfg, self.agent = resolve_task_config(
            "Isaac-Velocity-Flat-Unitree-Go2-v0", "rsl_rl_cfg_entry_point"
        )
        cfg.scene.num_envs = num_envs
        cfg.seed = seed
        cfg.episode_length_s = 60.0
        cfg.observations.policy.enable_corruption = False
        cfg.events.base_external_force_torque = None
        cfg.events.push_robot = None
        cfg.scene.stage11_eval_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_foot", update_period=0.0,
            track_pose=True, track_contact_points=True, track_friction_forces=True,
            max_contact_data_count_per_prim=16, filter_prim_paths_expr=[GROUND],
        )
        if args.device:
            cfg.sim.device = args.device
            self.agent.device = args.device
        self.raw = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
        self.env = self.raw.unwrapped
        self.num_envs = num_envs
        self.seed = seed
        self.robot = self.env.scene["robot"]
        self.command = self.env.command_manager.get_term("base_velocity")
        self.sensor = self.env.scene.sensors["stage11_eval_contact"]
        if tuple(self.sensor.body_names) != ASSET_FEET:
            raise RuntimeError(f"foot order mismatch {self.sensor.body_names}")
        self.body_ids = [int(self.robot.find_bodies(name)[0][0]) for name in ASSET_FEET]

    def policy(self, checkpoint):
        if not hasattr(self, "wrapped"):
            self.wrapped, self.runner, policy = build_runner(self.raw, self.agent, checkpoint)
            return policy
        self.runner.load(str(checkpoint), strict=True, map_location=self.env.device)
        return self.runner.get_inference_policy(device=self.env.device)

    def contact(self):
        fn_raw, point_raw, normal_raw, _, count_raw, start_raw = self.sensor.contact_view.get_contact_data(
            dt=self.sensor._sim_physics_dt
        )
        fn = wp.to_torch(fn_raw).reshape(-1)
        point = wp.to_torch(point_raw).reshape(-1, 3)
        normal = wp.to_torch(normal_raw).reshape(-1, 3)
        count = wp.to_torch(count_raw).reshape(self.num_envs, 4, -1)[:, :, 0].long()
        start = wp.to_torch(start_raw).reshape(self.num_envs, 4, -1)[:, :, 0].long()
        offsets = torch.arange(16, device=point.device)
        indices = (start[..., None] + offsets).clamp(0, max(0, len(point) - 1))
        mask = offsets[None, None] < count[..., None]
        points, normals = point[indices], normal[indices]
        forces = fn[indices].abs() * mask
        unit = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        position = self.robot.data.body_pos_w.torch[:, self.body_ids]
        linear = self.robot.data.body_lin_vel_w.torch[:, self.body_ids]
        angular = self.robot.data.body_ang_vel_w.torch[:, self.body_ids]
        radius = points - position[:, :, None]
        surface = linear[:, :, None] + torch.linalg.cross(
            angular[:, :, None].expand_as(radius), radius
        )
        tangent = surface - (surface * unit).sum(-1, keepdim=True) * unit
        total = forces.sum(-1)
        speed = (tangent.norm(dim=-1) * forces).sum(-1) / total.clamp_min(1e-12)
        friction_raw, _, friction_count_raw, friction_start_raw = (
            self.sensor.contact_view.get_friction_data(dt=self.sensor._sim_physics_dt)
        )
        friction = wp.to_torch(friction_raw).reshape(-1, 3)
        friction_count = wp.to_torch(friction_count_raw).reshape(self.num_envs, 4, -1)[:, :, 0].long()
        friction_start = wp.to_torch(friction_start_raw).reshape(self.num_envs, 4, -1)[:, :, 0].long()
        friction_indices = (friction_start[..., None] + offsets).clamp(
            0, max(0, len(friction) - 1)
        )
        friction_mask = offsets[None, None] < friction_count[..., None]
        friction_sum = (friction[friction_indices] * friction_mask[..., None]).sum(-2)
        utilization = friction_sum.norm(dim=-1) / (MU * total).clamp_min(1e-12)
        return speed, total, utilization

    def collect(self, policy, family, source, target, group_size=None):
        self.env.seed(self.seed)
        self.wrapped.reset()
        steps = round((8.0 if family == "steady" else 9.5) / DT)
        controllers = [
            PhaseGatedFixedHeadingController("PHASE_GATED_FIXED_HEADING", family, target, DT)
            for _ in range(self.num_envs)
        ]
        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.env.device)
        falls = torch.zeros_like(alive)
        contact_age = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.env.device)
        dangerous_run = torch.zeros_like(contact_age)
        dangerous_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
        max_danger = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.env.device)
        saturation_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)
        long_saturation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
        acquisition_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)
        acquired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
        speed_samples, heading_samples, tilt_samples, tangent_samples = [[] for _ in range(self.num_envs)], [[] for _ in range(self.num_envs)], [[] for _ in range(self.num_envs)], [[] for _ in range(self.num_envs)]
        friction_samples = [[] for _ in range(self.num_envs)]
        final_within = [[] for _ in range(self.num_envs)]
        per_foot_values = [[] for _ in range(4)]
        stable_count = torch.zeros(self.num_envs, device=self.env.device)
        dangerous_count = torch.zeros(self.num_envs, device=self.env.device)
        for step in range(steps):
            t = step * DT
            if family == "steady":
                speed, phase = target, "steady"
            else:
                speed, raw_phase = transition_command(t, source, target, 1.5)
                phase = {"source_hold": "source", "ramp": "ramp", "target_hold": "target"}[raw_phase]
            _, _, yaw = quat_xyzw_to_roll_pitch_yaw_torch(self.robot.data.root_quat_w.torch)
            actual = self.robot.data.root_lin_vel_b.torch[:, 0]
            outputs = [
                controller.update(t, float(yaw[index]), float(actual[index]), phase)
                for index, controller in enumerate(controllers)
            ]
            self.command.vel_command_b[:, 0] = speed
            self.command.vel_command_b[:, 1] = 0.0
            self.command.vel_command_b[:, 2] = torch.tensor(
                [output.command for output in outputs], device=self.env.device
            )
            obs = self.wrapped.get_observations()
            with torch.inference_mode():
                action = policy(obs)
                _, _, dones, _ = self.wrapped.step(action)
            vx = self.robot.data.root_lin_vel_b.torch[:, 0]
            tilt = quat_xyzw_to_gravity_tilt_torch(self.robot.data.root_quat_w.torch)
            tangential, force, friction = self.contact()
            contact = force > 5.0
            contact_age = torch.where(contact, contact_age + 1, torch.zeros_like(contact_age))
            stable = contact & (contact_age >= 3)
            dangerous = stable & (tangential > 0.30)
            dangerous_run = torch.where(dangerous, dangerous_run + 1, torch.zeros_like(dangerous_run))
            max_danger = torch.maximum(max_danger, dangerous_run)
            dangerous_episode |= (dangerous_run >= 5).any(1)
            stable_count += stable.sum(1)
            dangerous_count += dangerous.sum(1)
            vel_ratio = (
                self.robot.data.joint_vel.torch.abs()
                / self.robot.data.joint_vel_limits.torch.abs().clamp_min(1e-6)
            ).amax(1)
            torque_ratio = (
                self.robot.data.applied_torque.torch.abs()
                / self.robot.data.joint_effort_limits.torch.abs().clamp_min(1e-6)
            ).amax(1)
            saturated = (vel_ratio >= 0.95) | (torque_ratio >= 0.95)
            saturation_run = torch.where(saturated, saturation_run + 1, torch.zeros_like(saturation_run))
            long_saturation |= saturation_run >= 10
            quality = t >= (1.5 if family == "steady" else 4.5)
            if family == "transition" and t >= 4.5:
                within = vx.abs() <= 0.08 if target == 0 else (vx - target).abs() <= target_tolerance(target)
                acquisition_run = torch.where(within, acquisition_run + 1, torch.zeros_like(acquisition_run))
                acquired |= acquisition_run >= 25
            if quality:
                for index, output in enumerate(outputs):
                    if alive[index]:
                        speed_samples[index].append(float(abs(vx[index] - target)))
                        heading_samples[index].append(abs(float(output.error)))
                        tilt_samples[index].append(float(tilt[index]))
                        tangent_samples[index].extend(tangential[index][stable[index]].cpu().tolist())
                        friction_samples[index].extend(friction[index][stable[index]].cpu().tolist())
                        if t >= steps * DT - 1.0:
                            final_within[index].append(
                                bool(abs(vx[index]) <= 0.08) if target == 0
                                else bool(abs(vx[index] - target) <= target_tolerance(target))
                            )
                for foot in range(4):
                    per_foot_values[foot].extend(tangential[:, foot][stable[:, foot]].cpu().tolist())
            newly = dones.bool() & alive
            falls |= newly
            alive &= ~dones.bool()
        episode_heading = [percentile(values, 95) for values in heading_samples]
        episode_speed = [mean(values) for values in speed_samples]
        hold = [mean(values) >= 0.90 for values in final_within]
        completion = [
            (not bool(falls[index])) and (family == "steady" or bool(acquired[index]) and hold[index])
            for index in range(self.num_envs)
        ]
        def summarize(indices):
            index_list = list(indices)
            stable_total = float(stable_count[index_list].sum())
            return {
                "family": family,
                "condition": f"{source:g}->{target:g}" if family == "transition" else f"{target:g}",
                "source": source, "target": target, "episodes": len(index_list),
                "fall_rate": float(falls[index_list].float().mean()),
                "completion_rate": mean([completion[index] for index in index_list]),
                "acquisition_rate": mean(acquired[index_list].cpu()) if family == "transition" else 1.0,
                "target_hold_rate": mean([hold[index] for index in index_list]),
                "speed_mae": mean([episode_speed[index] for index in index_list]),
                "heading_p95": percentile([episode_heading[index] for index in index_list], 95),
                "tilt_p95": percentile([percentile(tilt_samples[index], 95) for index in index_list], 95),
                "dangerous_slip_episode_rate": float(dangerous_episode[index_list].float().mean()),
                "dangerous_time_fraction": float(dangerous_count[index_list].sum()) / max(1.0, stable_total),
                "tangential_speed_p95": percentile(
                    [value for index in index_list for value in tangent_samples[index]], 95
                ),
                "friction_utilization_p95": percentile(
                    [value for index in index_list for value in friction_samples[index]], 95
                ),
                "maximum_contiguous_slip_s": float(max_danger[index_list].max()) * DT,
                "long_dwell_saturation_rate": float(long_saturation[index_list].float().mean()),
                "per_foot_tangential_p95": {
                    FEET[index]: percentile(values, 95) for index, values in enumerate(per_foot_values)
                },
            }
        if group_size:
            return [
                summarize(range(begin, min(begin + group_size, self.num_envs)))
                for begin in range(0, self.num_envs, group_size)
            ]
        return summarize(range(self.num_envs))

    def collect_sequence(self, policy, group_size=None):
        """Evaluate the integrated sequence without resetting policy or controller state mid-segment."""
        self.env.seed(self.seed)
        self.wrapped.reset()
        steps = round((3.0 + (len(SEQUENCE) - 1) * 4.5) / DT)
        controllers = [
            PhaseGatedFixedHeadingController("PHASE_GATED_FIXED_HEADING", "steady", 0.0, DT)
            for _ in range(self.num_envs)
        ]
        current_segment = 0
        segment_start = 0.0
        yaw_history = [[] for _ in range(self.num_envs)]
        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.env.device)
        falls = torch.zeros_like(alive)
        contact_age = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.env.device)
        dangerous_run = torch.zeros_like(contact_age)
        dangerous_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
        stable_count = torch.zeros(self.num_envs, device=self.env.device)
        dangerous_count = torch.zeros(self.num_envs, device=self.env.device)
        max_danger = torch.zeros_like(contact_age)
        long_saturation = torch.zeros_like(alive)
        saturation_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)
        segment_run = torch.zeros(
            self.num_envs, len(SEQUENCE) - 1, dtype=torch.long, device=self.env.device
        )
        segment_acquired = torch.zeros_like(segment_run, dtype=torch.bool)
        segment_hold = torch.zeros_like(segment_run, dtype=torch.long)
        heading_values = [[] for _ in range(self.num_envs)]
        final_speed = [[] for _ in range(self.num_envs)]
        tangent_values = [[] for _ in range(self.num_envs)]
        friction_values = [[] for _ in range(self.num_envs)]
        for step in range(steps):
            t = step * DT
            speed, segment, phase = sequence_command(t, SEQUENCE, 1.5)
            if segment != current_segment:
                current_segment = segment
                segment_start = t - 3.0
                controllers = [
                    PhaseGatedFixedHeadingController(
                        "PHASE_GATED_FIXED_HEADING", "transition", SEQUENCE[segment], DT
                    )
                    for _ in range(self.num_envs)
                ]
                for index, controller in enumerate(controllers):
                    controller.reference_samples = yaw_history[index][-round(0.5 / DT):]
            _, _, yaw = quat_xyzw_to_roll_pitch_yaw_torch(self.robot.data.root_quat_w.torch)
            actual = self.robot.data.root_lin_vel_b.torch[:, 0]
            for index in range(self.num_envs):
                yaw_history[index].append(float(yaw[index]))
            if segment == 0:
                local_time, schedule_phase = t, "steady"
            else:
                local_time = t - segment_start
                schedule_phase = "ramp" if phase == "ramp" else "target"
            outputs = [
                controller.update(local_time, float(yaw[index]), float(actual[index]), schedule_phase)
                for index, controller in enumerate(controllers)
            ]
            self.command.vel_command_b[:, 0] = speed
            self.command.vel_command_b[:, 1] = 0.0
            self.command.vel_command_b[:, 2] = torch.tensor(
                [output.command for output in outputs], device=self.env.device
            )
            observations = self.wrapped.get_observations()
            with torch.inference_mode():
                action = policy(observations)
                _, _, dones, _ = self.wrapped.step(action)
            vx = self.robot.data.root_lin_vel_b.torch[:, 0]
            tangential, force, friction = self.contact()
            contact = force > 5.0
            contact_age = torch.where(contact, contact_age + 1, torch.zeros_like(contact_age))
            stable = contact & (contact_age >= 3)
            dangerous = stable & (tangential > 0.30)
            dangerous_run = torch.where(dangerous, dangerous_run + 1, torch.zeros_like(dangerous_run))
            max_danger = torch.maximum(max_danger, dangerous_run)
            dangerous_episode |= (dangerous_run >= 5).any(1)
            stable_count += stable.sum(1)
            dangerous_count += dangerous.sum(1)
            for index in range(self.num_envs):
                tangent_values[index].extend(tangential[index][stable[index]].cpu().tolist())
                friction_values[index].extend(friction[index][stable[index]].cpu().tolist())
                heading_values[index].append(abs(float(outputs[index].error)))
            vel_ratio = (
                self.robot.data.joint_vel.torch.abs()
                / self.robot.data.joint_vel_limits.torch.abs().clamp_min(1e-6)
            ).amax(1)
            torque_ratio = (
                self.robot.data.applied_torque.torch.abs()
                / self.robot.data.joint_effort_limits.torch.abs().clamp_min(1e-6)
            ).amax(1)
            saturated = (vel_ratio >= 0.95) | (torque_ratio >= 0.95)
            saturation_run = torch.where(saturated, saturation_run + 1, torch.zeros_like(saturation_run))
            long_saturation |= saturation_run >= 10
            if segment > 0 and phase == "hold":
                target = SEQUENCE[segment]
                within = vx.abs() <= 0.08 if target == 0 else (vx - target).abs() <= target_tolerance(target)
                slot = segment - 1
                segment_run[:, slot] = torch.where(
                    within, segment_run[:, slot] + 1, torch.zeros_like(segment_run[:, slot])
                )
                segment_acquired[:, slot] |= segment_run[:, slot] >= 25
                segment_hold[:, slot] += within.long()
            if t >= steps * DT - 1.0:
                final_speed = [
                    values + [float(abs(vx[index]))] for index, values in enumerate(final_speed)
                ]
            newly = dones.bool() & alive
            falls |= newly
            alive &= ~dones.bool()
        segment_success = segment_acquired & (segment_hold >= round(0.80 * 3.0 / DT))
        rows = []
        for index in range(self.num_envs):
            rows.append({
                "episode": index,
                "sequence_completion": bool(segment_success[index].all() and not falls[index]),
                "segment_success": [bool(value) for value in segment_success[index].cpu()],
                "fall": bool(falls[index]),
                "heading_p95": percentile(heading_values[index], 95),
                "dangerous_slip": bool(dangerous_episode[index]),
                "dangerous_time_fraction": float(dangerous_count[index] / stable_count[index].clamp_min(1)),
                "tangential_speed_p95": percentile(tangent_values[index], 95),
                "friction_utilization_p95": percentile(friction_values[index], 95),
                "maximum_contiguous_slip_s": float(max_danger[index].max()) * DT,
                "long_dwell_saturation": bool(long_saturation[index]),
                "final_stand": mean(final_speed[index]) <= 0.08,
                "checkpoint_switches": 0,
            })
        def summarize(row_group):
            return {
            "episodes": len(row_group),
            "sequence_completion_rate": mean(row["sequence_completion"] for row in row_group),
            "each_segment_success_rate": [
                mean(row["segment_success"][slot] for row in row_group)
                for slot in range(len(SEQUENCE) - 1)
            ],
            "fall_rate": mean(row["fall"] for row in row_group),
            "heading_p95": percentile([row["heading_p95"] for row in row_group], 95),
            "dangerous_slip_episode_rate": mean(row["dangerous_slip"] for row in row_group),
            "dangerous_time_fraction": mean(row["dangerous_time_fraction"] for row in row_group),
            "tangential_speed_p95": percentile([row["tangential_speed_p95"] for row in row_group], 95),
            "friction_utilization_p95": percentile(
                [row["friction_utilization_p95"] for row in row_group], 95
            ),
            "maximum_contiguous_slip_s": max(row["maximum_contiguous_slip_s"] for row in row_group),
            "long_dwell_saturation_rate": mean(row["long_dwell_saturation"] for row in row_group),
            "final_stand_rate": mean(row["final_stand"] for row in row_group),
            "checkpoint_switches": 0,
        }
        if group_size:
            return [
                summarize(rows[begin:begin + group_size])
                for begin in range(0, self.num_envs, group_size)
            ]
        return summarize(rows)

    def close(self):
        self.wrapped.close()


def checkpoint_paths():
    manifest = json.loads((OUT / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    return [Path(item["path"]) for item in manifest["checkpoints"]]


def main():
    paths = checkpoint_paths() if args.mode == "validation" else []
    formal_group_size = args.num_envs or 50
    num_envs = (10 * len(paths)) if args.mode == "validation" else formal_group_size
    seed = 20270901 if args.mode == "validation" else 20271901
    evaluator = Evaluator(num_envs, seed)
    try:
        if args.mode == "validation":
            all_rows = []
            models = []
            for checkpoint in paths:
                evaluator.policy(checkpoint)
                models.append(copy.deepcopy(evaluator.runner.alg.actor).eval())

            def combined_policy(observations):
                actions = []
                for index, model in enumerate(models):
                    begin, end = index * 10, (index + 1) * 10
                    actions.append(model(observations[begin:end], stochastic_output=False))
                return torch.cat(actions, dim=0)

            conditions = [("steady", speed, speed) for speed in VALID_STEADY]
            conditions += [("transition", source, target) for source, target in VALID_TRANSITIONS]
            for family, source, target in conditions:
                group_rows = evaluator.collect(
                    combined_policy, family, source, target, group_size=10
                )
                for checkpoint, row in zip(paths, group_rows):
                    row["checkpoint"] = str(checkpoint)
                    row["checkpoint_iteration"] = (
                        int(checkpoint.stem.split("_")[-1])
                        if checkpoint.stem != "model_initial" else 0
                    )
                    all_rows.append(row)
            write_csv("validation_checkpoint_results.csv", all_rows)
            grouped = {}
            for row in all_rows:
                grouped.setdefault(row["checkpoint"], []).append(row)
            ranking = []
            for checkpoint, rows in grouped.items():
                hard = sum(
                    row["fall_rate"] <= 0.05 and row["completion_rate"] >= 0.90
                    and row["heading_p95"] <= 0.12 for row in rows
                )
                ranking.append({
                    "checkpoint": checkpoint,
                    "iteration": rows[0]["checkpoint_iteration"],
                    "hard_pass_count": hard,
                    "mean_dangerous_slip": mean(row["dangerous_slip_episode_rate"] for row in rows),
                    "mean_tangential_p95": mean(row["tangential_speed_p95"] for row in rows),
                    "mean_speed_mae": mean(row["speed_mae"] for row in rows),
                })
            ranking.sort(key=lambda row: (
                -row["hard_pass_count"], row["mean_dangerous_slip"],
                row["mean_tangential_p95"], row["mean_speed_mae"], row["iteration"],
            ))
            selected = ranking[0]
            selected["status"] = "SELECTED_PRE_FORMAL"
            selected["selection_precedence"] = "hard safety/capability, heading, slip duration/p95, tracking"
            dump("selected_checkpoint.json", selected)
        else:
            selected = json.loads((OUT / "selected_checkpoint.json").read_text(encoding="utf-8"))
            evaluator.policy(Path(selected["checkpoint"]))
            selected_model = copy.deepcopy(evaluator.runner.alg.actor).eval()
            evaluator.policy(PARENT)
            parent_model = copy.deepcopy(evaluator.runner.alg.actor).eval()
            selected_policy = lambda observations: selected_model(observations, stochastic_output=False)
            parent_policy = lambda observations: parent_model(observations, stochastic_output=False)
            models_identical = all(
                torch.equal(selected_model.state_dict()[key], parent_model.state_dict()[key])
                for key in selected_model.state_dict()
            )
            selected_steady = [
                evaluator.collect(selected_policy, "steady", speed, speed)
                for speed in FORMAL_STEADY
            ]
            selected_transitions = [
                evaluator.collect(selected_policy, "transition", a, b) for a, b in TRANSITIONS
            ]
            selected_sequence = evaluator.collect_sequence(selected_policy)
            if models_identical:
                # The pre-formal rule selected model_initial. The strict resume identity audit
                # proves that its actor is the Stage 7 parent actor; copying the paired results
                # avoids pretending that different vectorized env indices are the same seeds.
                parent_steady = copy.deepcopy(selected_steady)
                parent_transitions = copy.deepcopy(selected_transitions)
                parent_sequence = copy.deepcopy(selected_sequence)
            else:
                parent_steady = [
                    evaluator.collect(parent_policy, "steady", speed, speed)
                    for speed in FORMAL_STEADY
                ]
                parent_transitions = [
                    evaluator.collect(parent_policy, "transition", a, b) for a, b in TRANSITIONS
                ]
                parent_sequence = evaluator.collect_sequence(parent_policy)
            zero = next(row for row in selected_steady if row["target"] == 0.0)
            dump("formal_zero_results.json", zero)
            write_csv("formal_steady_state.csv", selected_steady)
            dump("formal_steady_state.json", {
                "selected": selected_steady, "parent": parent_steady,
                "models_bitwise_identical": models_identical,
                "paired_seed_handling": "single rollout copied for bitwise-identical policies"
                if models_identical else "sequential reset with identical seed",
            })
            write_csv("formal_transitions.csv", selected_transitions)
            dump("formal_transitions.json", {
                "selected": selected_transitions, "parent": parent_transitions,
                "models_bitwise_identical": models_identical,
            })
            dump("integrated_sequence_diagnostic.json", {
                "selected": selected_sequence, "parent": parent_sequence,
            })
            per_foot = []
            for selected_row, parent_row in zip(selected_steady, parent_steady):
                for foot in FEET:
                    per_foot.append({
                        "speed": selected_row["target"], "foot": foot,
                        "stage7_p95": parent_row["per_foot_tangential_p95"][foot],
                        "stage11_p95": selected_row["per_foot_tangential_p95"][foot],
                    })
            write_csv("per_foot_slip_comparison.csv", per_foot)
    finally:
        evaluator.close()


main()
simulation_app.close()
