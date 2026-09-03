"""Finite-difference and random-pose STEP_OVER authority audit on the frozen standing base."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path[:0] = [str(ROOT / "src"), str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--random-poses-per-side", type=int, default=128)
parser.add_argument("--seed", type=int, default=20260722)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser); sys.argv = [sys.argv[0]] + hydra

KEYPOINTS = {
    "toe": (0.06383880963290349, 0.0, -0.025807180037281774),
    "sole": (0.04321213238651294, 0.0, -0.025807180037281774),
    "heel": (0.022585455140122387, 0.0, -0.025807180037281774),
}
JOINTS = {"hip_pitch": (0, 1), "hip_roll": (3, 4), "knee": (11, 12),
          "ankle_pitch": (15, 16), "ankle_roll": (19, 20)}


def minimum_jerk(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    perturbations = (-0.50, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50)
    for side in ("left", "right"):
        for joint in JOINTS:
            for value in perturbations:
                candidates.append({"kind": "finite_difference", "side": side, "joint": joint, joint: value})
    generator = torch.Generator().manual_seed(args.seed)
    for side in ("left", "right"):
        for index in range(args.random_poses_per_side):
            sample = torch.rand(8, generator=generator).tolist()
            candidates.append({
                "kind": "pose_search", "side": side, "joint": f"pose_{index}",
                "hip_pitch": -1.0 + 2.0 * sample[0], "knee": -0.2 + 1.7 * sample[1],
                "ankle_pitch": -1.5 + 2.5 * sample[2], "hip_roll": -0.5 + sample[3],
                "ankle_roll": -0.5 + sample[4], "torso": -0.3 + 0.6 * sample[5],
                "support_hip_roll": -0.4 + 0.8 * sample[6], "support_ankle_roll": -0.4 + 0.8 * sample[7],
            })
    count = len(candidates)
    env_cfg, agent_cfg = resolve_task_config("Isaac-Motion-Flat-G1-Command-StepOverAudit-Eval-v0", "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = count; env_cfg.seed = args.seed
    if args.device is not None: env_cfg.sim.device = args.device
    with launch_simulation(env_cfg, args):
        raw = gym.make("Isaac-Motion-Flat-G1-Command-StepOverAudit-Eval-v0", cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions); env = raw.unwrapped
        agent_cfg.device = env.device; agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(Path(args.checkpoint).resolve(strict=True)), load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
        actor = runner.alg.actor; robot = env.scene["robot"]; contact = env.scene.sensors["contact_forces"]
        foot_ids, foot_names = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
        sensor_ids = [contact.body_names.index(name) for name in foot_names]
        all_joint_ids, _ = robot.find_joints(".*")
        offsets = torch.zeros((count, 37), device=env.device)
        for env_id, candidate in enumerate(candidates):
            lead = 0 if candidate["side"] == "left" else 1; support = 1 - lead
            for name, pair in JOINTS.items():
                offsets[env_id, pair[lead]] = float(candidate.get(name, 0.0))
            offsets[env_id, 2] = float(candidate.get("torso", 0.0))
            offsets[env_id, JOINTS["hip_roll"][support]] = float(candidate.get("support_hip_roll", 0.0))
            offsets[env_id, JOINTS["ankle_roll"][support]] = float(candidate.get("support_ankle_roll", 0.0))
        wrapped.reset(); dt = float(env.step_dt)
        settle_steps, ramp_steps, hold_steps = round(1.5 / dt), round(1.0 / dt), round(0.6 / dt)
        active = torch.ones(count, dtype=torch.bool, device=env.device)
        baseline = None; maxima = {name: torch.full((count, 2), -1e9, device=env.device) for name in KEYPOINTS}
        support_contact_all = torch.ones(count, dtype=torch.bool, device=env.device)
        both_airborne = torch.zeros(count, dtype=torch.bool, device=env.device)
        vel_sat_count = torch.zeros(count, device=env.device); torque_sat_count = vel_sat_count.clone()
        for step in range(settle_steps + ramp_steps + hold_steps):
            obs = wrapped.get_observations()
            with torch.inference_mode(): standing = actor.diagnostic_components(obs)["standing_base_action"]
            blend = 0.0 if step < settle_steps else minimum_jerk((step - settle_steps + 1) / ramp_steps)
            action = standing + offsets * blend
            _, _, dones, _ = wrapped.step(action); active &= ~dones.bool()
            pos = robot.data.body_pos_w.torch[:, foot_ids]; quat = robot.data.body_quat_w.torch[:, foot_ids]
            points = {}
            for name, value in KEYPOINTS.items():
                local = torch.tensor(value, device=env.device).expand(count, 2, 3)
                points[name] = pos + quat_apply(quat.reshape(-1, 4), local.reshape(-1, 3)).reshape(count, 2, 3)
                if step >= settle_steps:
                    maxima[name] = torch.maximum(maxima[name], points[name][..., 2])
            if step == settle_steps - 1: baseline = {name: value.clone() for name, value in points.items()}
            forces = contact.data.net_forces_w_history.torch[:, :, sensor_ids, :].norm(dim=-1).amax(dim=1) > 5.0
            lead_index = torch.tensor([0 if item["side"] == "left" else 1 for item in candidates], device=env.device)
            support_index = 1 - lead_index
            measuring_hold = step >= settle_steps + ramp_steps
            if measuring_hold:
                support_contact_all &= forces.gather(1, support_index[:, None]).squeeze(1)
                both_airborne |= ~forces.any(dim=1)
            vr = robot.data.joint_vel.torch[:, all_joint_ids].abs() / robot.data.joint_vel_limits.torch[:, all_joint_ids].abs().clamp_min(1e-6)
            tr = robot.data.applied_torque.torch[:, all_joint_ids].abs() / robot.data.joint_effort_limits.torch[:, all_joint_ids].abs().clamp_min(1e-6)
            if measuring_hold:
                vel_sat_count += (vr >= .95).any(dim=1).float()
                torque_sat_count += (tr >= .95).any(dim=1).float()
        final_pos = robot.data.body_pos_w.torch[:, foot_ids]; final_quat = robot.data.body_quat_w.torch[:, foot_ids]
        hard = robot.data.joint_pos_limits.torch[:, all_joint_ids]; q = robot.data.joint_pos.torch[:, all_joint_ids]
        default = robot.data.default_joint_pos.torch[:, all_joint_ids]
        limit = torch.maximum((q-default)/(hard[...,1]-default).clamp_min(1e-6),(default-q)/(default-hard[...,0]).clamp_min(1e-6)).abs().amax(dim=1)
        gravity = robot.data.projected_gravity_b.torch; tilt = torch.linalg.vector_norm(gravity[:, :2], dim=1)
        rows = []
        for env_id, candidate in enumerate(candidates):
            lead = 0 if candidate["side"] == "left" else 1
            row = dict(candidate)
            for name in KEYPOINTS:
                local = torch.tensor(KEYPOINTS[name], device=env.device)
                final = final_pos[env_id, lead] + quat_apply(final_quat[env_id, lead], local)
                row[f"{name}_vertical_displacement_m"] = float(maxima[name][env_id, lead] - baseline[name][env_id, lead, 2])
                row[f"{name}_forward_displacement_m"] = float(final[0] - baseline[name][env_id, lead, 0])
            row.update({"support_contact_maintained": bool(support_contact_all[env_id]), "both_feet_airborne": bool(both_airborne[env_id]),
                        "fall": not bool(active[env_id]), "tilt_rad": float(tilt[env_id]), "joint_limit_proximity": float(limit[env_id]),
                        "velocity_saturation_fraction": float(vel_sat_count[env_id] / hold_steps),
                        "torque_saturation_fraction": float(torque_sat_count[env_id] / hold_steps)})
            row["stable"] = bool(active[env_id] and support_contact_all[env_id] and not both_airborne[env_id]
                                 and vel_sat_count[env_id] / hold_steps <= .05 and torque_sat_count[env_id] / hold_steps <= .05
                                 and tilt[env_id] < .25 and limit[env_id] < .95)
            rows.append(row)
        write_csv(output / "authority.csv", rows)
        stable = [row for row in rows if row["stable"]]
        ranked = sorted(stable, key=lambda row: (row["sole_vertical_displacement_m"] + max(row["sole_forward_displacement_m"], 0.0)), reverse=True)[:20]
        summary = {"checkpoint": str(Path(args.checkpoint).resolve()), "candidate_count": count, "stable_count": len(stable),
                   "keypoints_body_m": KEYPOINTS, "obstacle": {"x_m": .32, "depth_m": .06, "width_m": 2.2, "height_m": .05},
                   "best_stable_poses": ranked}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"candidate_count": count, "stable_count": len(stable), "best": ranked[:3]}, indent=2)); raw.close()


if __name__ == "__main__": main()
