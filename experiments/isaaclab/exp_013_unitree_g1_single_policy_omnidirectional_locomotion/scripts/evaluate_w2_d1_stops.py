"""Read-only batched practical-stop boundary evaluation for W2-D1."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_d1_practical_stop_retention_diagnosis"
)
W2 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_dynamic_omnidirectional_walk_transitions"
)
CHECKPOINTS = {
    "parent": REPO / (
        "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
        "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
    ),
    "iteration_1": W2 / "checkpoints/model_1.pt",
    "iteration_5": W2 / "checkpoints/model_5.pt",
    "exp012": REPO / (
        "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/"
        "stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
    ),
}
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa
import g1_omnidirectional.tasks  # noqa
from g1_omnidirectional.policy import FrozenGaitActor
from g1_omnidirectional.yaw_calibration import calibrate_yaw
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("baseline", "hold", "ramp", "profiles", "local"), required=True)
parser.add_argument("--max-envs", type=int, default=1200)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


@dataclass(frozen=True)
class Job:
    checkpoint: str
    direction_deg: float
    source_yaw: float
    episodes: int
    ramp_s: float = 1.5
    hold_s: float = 4.0
    profile: str = "S1_DIRECT"
    source_speed: float = 0.3
    local_lambda: float = 0.0
    local_steps: int = 0


def minjerk(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(0.0, 1.0)
    return u**3 * (10.0 - 15.0 * u + 6.0 * u**2)


def jobs_for_mode(mode: str) -> list[Job]:
    jobs: list[Job] = []
    if mode == "baseline":
        for checkpoint in ("parent", "iteration_1", "iteration_5"):
            episodes = 100 if checkpoint != "iteration_1" else 50
            for direction in range(0, 360, 45):
                for yaw in (-0.3, 0.0, 0.3):
                    jobs.append(Job(checkpoint, float(direction), yaw, episodes))
        jobs.append(Job("exp012", 0.0, 0.0, 100))
    elif mode == "hold":
        for checkpoint in ("parent", "iteration_5"):
            for hold in (2, 3, 4, 5, 6, 8, 10, 12):
                for direction in range(0, 360, 45):
                    for yaw in (-0.3, 0.0, 0.3):
                        jobs.append(Job(checkpoint, float(direction), yaw, 50, hold_s=float(hold)))
    elif mode == "ramp":
        for checkpoint in ("parent", "iteration_5"):
            for ramp in (0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
                for direction in (0, 90, 180, 270):
                    for yaw in (-0.3, 0.0, 0.3):
                        jobs.append(Job(checkpoint, float(direction), yaw, 50, ramp_s=ramp, hold_s=6.0))
    elif mode == "profiles":
        for checkpoint in ("parent", "iteration_5"):
            for profile in (
                "S1_DIRECT", "S2_YAW_THEN_TRANSLATION", "S3_TRANSLATION_THEN_YAW",
                "S4_TWO_STAGE_SPEED", "S5_MAGNITUDE_ONLY_DECELERATION",
            ):
                for direction in range(0, 360, 45):
                    for yaw in (-0.3, 0.0, 0.3):
                        jobs.append(Job(checkpoint, float(direction), yaw, 50, profile=profile, hold_s=6.0))
    else:
        for direction, yaw in ((0, 0), (90, 0), (180, 0), (0, 0.3), (0, -0.3)):
            for lam in (0, .10, .25, .50, .75, 1.0):
                for steps in (1, 2, 4, 8):
                    jobs.append(Job(
                        "parent", float(direction), float(yaw), 100, hold_s=6.0,
                        profile="LOCAL_EXP012", local_lambda=lam, local_steps=steps,
                    ))
    return jobs


def expand_jobs(jobs: list[Job]) -> list[tuple[int, int]]:
    return [(index, episode) for index, job in enumerate(jobs) for episode in range(job.episodes)]


def command_at(
    job_indices: torch.Tensor, jobs: list[Job], t: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(job_indices)
    physical = torch.zeros(n, 3, device=device)
    ramp_end = torch.zeros(n, device=device)
    for ji in torch.unique(job_indices).tolist():
        mask = job_indices == ji
        job = jobs[ji]
        angle = math.radians(job.direction_deg)
        source = torch.tensor(
            [job.source_speed * math.cos(angle), job.source_speed * math.sin(angle), job.source_yaw],
            device=device,
        )
        ramp_end[mask] = 3.0 + job.ramp_s
        if t < 3.0:
            physical[mask] = source
            continue
        u = (t - 3.0) / max(job.ramp_s, 1e-6)
        if job.profile in ("S1_DIRECT", "S5_MAGNITUDE_ONLY_DECELERATION", "LOCAL_EXP012"):
            factor = 1.0 - float(minjerk(torch.tensor(u)))
            physical[mask] = source * factor
        elif job.profile == "S2_YAW_THEN_TRANSLATION":
            yaw_factor = 1.0 - float(minjerk(torch.tensor(u * 2.0)))
            trans_factor = 1.0 - float(minjerk(torch.tensor((u - .5) * 2.0)))
            physical[mask, :2] = source[:2] * trans_factor
            physical[mask, 2] = source[2] * yaw_factor
        elif job.profile == "S3_TRANSLATION_THEN_YAW":
            trans_factor = 1.0 - float(minjerk(torch.tensor(u * 2.0)))
            yaw_factor = 1.0 - float(minjerk(torch.tensor((u - .5) * 2.0)))
            physical[mask, :2] = source[:2] * trans_factor
            physical[mask, 2] = source[2] * yaw_factor
        else:
            # 0.3 -> 0.15 -> 0, with yaw going directly to zero.
            if u < .5:
                speed_factor = 1.0 - .5 * float(minjerk(torch.tensor(u * 2.0)))
            else:
                speed_factor = .5 * (1.0 - float(minjerk(torch.tensor((u - .5) * 2.0))))
            yaw_factor = 1.0 - float(minjerk(torch.tensor(u)))
            physical[mask, :2] = source[:2] * speed_factor
            physical[mask, 2] = source[2] * yaw_factor
    return physical, ramp_end


def command_at_vectors(
    source: torch.Tensor,
    ramp_s: torch.Tensor,
    profile_code: torch.Tensor,
    t: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized version used by the simulator hot loop."""
    physical = source.clone()
    ramp_end = 3.0 + ramp_s
    if t < 3.0:
        return physical, ramp_end
    u = (t - 3.0) / ramp_s.clamp_min(1e-6)
    direct = 1.0 - minjerk(u)
    physical = source * direct[:, None]
    yaw_first = profile_code == 1
    if yaw_first.any():
        physical[yaw_first, :2] = source[yaw_first, :2] * (
            1.0 - minjerk((u[yaw_first] - .5) * 2.0)
        )[:, None]
        physical[yaw_first, 2] = source[yaw_first, 2] * (
            1.0 - minjerk(u[yaw_first] * 2.0)
        )
    trans_first = profile_code == 2
    if trans_first.any():
        physical[trans_first, :2] = source[trans_first, :2] * (
            1.0 - minjerk(u[trans_first] * 2.0)
        )[:, None]
        physical[trans_first, 2] = source[trans_first, 2] * (
            1.0 - minjerk((u[trans_first] - .5) * 2.0)
        )
    two_stage = profile_code == 3
    if two_stage.any():
        uu = u[two_stage]
        speed_factor = torch.where(
            uu < .5,
            1.0 - .5 * minjerk(uu * 2.0),
            .5 * (1.0 - minjerk((uu - .5) * 2.0)),
        )
        physical[two_stage, :2] = source[two_stage, :2] * speed_factor[:, None]
        physical[two_stage, 2] = source[two_stage, 2] * (1.0 - minjerk(uu))
    return physical, ramp_end


def write_rows(name: str, rows: list[dict]) -> None:
    if rows:
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(episode_rows: list[dict], keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in episode_rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    metrics = (
        "practical_stop_success", "guard_stop_success", "translation_stop_pass",
        "yaw_stop_pass", "fall", "slip", "impact", "final_speed",
        "final_abs_yaw", "final_signed_yaw", "last2_speed", "last2_abs_yaw",
        "combined_acquisition_s", "contact_switches_last2", "mean_foot_speed_last2",
    )
    for group, rows in grouped.items():
        item = {key: value for key, value in zip(keys, group)}
        item["episodes"] = len(rows)
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            item[metric if metric.endswith("_s") else f"mean_{metric}"] = sum(values) / len(values)
        output.append(item)
    return output


def main():
    jobs = jobs_for_mode(args.mode)
    expanded = expand_jobs(jobs)
    cfg, ac = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = min(args.max_envs, len(expanded))
    cfg.episode_length_s = 20.0
    cfg.seed = 20275111
    if args.device:
        cfg.sim.device = ac.device = args.device
    all_rows: list[dict] = []
    state_rows: list[dict] = []
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=ac.clip_actions,
        )
        env = wrapped.unwrapped
        device = env.device
        actors = {
            name: FrozenGaitActor(path).to(device).eval()
            for name, path in CHECKPOINTS.items()
            if any(job.checkpoint == name for job in jobs) or args.mode == "local"
        }
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [robot.body_names.index(sensor.body_names[i]) for i in feet]
        max_envs = cfg.scene.num_envs
        for chunk_start in range(0, len(expanded), max_envs):
            chunk = expanded[chunk_start:chunk_start + max_envs]
            active = len(chunk)
            padded = chunk + [chunk[-1]] * (max_envs - active)
            job_indices = torch.tensor([item[0] for item in padded], device=device)
            episode_indices = [item[1] for item in padded]
            expanded_jobs = [jobs[item[0]] for item in padded]
            source_commands = torch.tensor([
                [
                    job.source_speed * math.cos(math.radians(job.direction_deg)),
                    job.source_speed * math.sin(math.radians(job.direction_deg)),
                    job.source_yaw,
                ]
                for job in expanded_jobs
            ], device=device)
            ramp_values = torch.tensor([job.ramp_s for job in expanded_jobs], device=device)
            hold_values = torch.tensor([job.hold_s for job in expanded_jobs], device=device)
            profile_map = {
                "S1_DIRECT": 0, "S2_YAW_THEN_TRANSLATION": 1,
                "S3_TRANSLATION_THEN_YAW": 2, "S4_TWO_STAGE_SPEED": 3,
                "S5_MAGNITUDE_ONLY_DECELERATION": 4, "LOCAL_EXP012": 0,
            }
            profile_codes = torch.tensor(
                [profile_map[job.profile] for job in expanded_jobs], device=device
            )
            checkpoint_masks = {
                checkpoint: torch.tensor(
                    [job.checkpoint == checkpoint for job in expanded_jobs], device=device
                )
                for checkpoint in set(job.checkpoint for job in expanded_jobs)
            }
            local_lambda = torch.tensor(
                [job.local_lambda for job in expanded_jobs], device=device
            )
            local_steps = torch.tensor(
                [job.local_steps for job in expanded_jobs], device=device
            )
            endpoint_begin = 3.0 + ramp_values
            endpoint_end = endpoint_begin + hold_values
            obs, _ = wrapped.reset()
            obs = obs.to(device)
            max_end = max(3.0 + jobs[ji].ramp_s + jobs[ji].hold_s for ji, _ in chunk)
            steps = math.ceil(max_end / env.step_dt)
            sum_speed = torch.zeros(max_envs, device=device)
            sum_abs_yaw = torch.zeros(max_envs, device=device)
            sum_signed_yaw = torch.zeros(max_envs, device=device)
            count = torch.zeros(max_envs, device=device)
            sum_speed_last2 = torch.zeros(max_envs, device=device)
            sum_abs_yaw_last2 = torch.zeros(max_envs, device=device)
            last2_count = torch.zeros(max_envs, device=device)
            fall = torch.zeros(max_envs, dtype=torch.bool, device=device)
            slip = fall.clone()
            impact = fall.clone()
            slip_streak = torch.zeros(max_envs, dtype=torch.long, device=device)
            first_speed = torch.full((max_envs,), float("nan"), device=device)
            first_yaw = first_speed.clone()
            first_combined = first_speed.clone()
            prev_contact = torch.zeros(max_envs, len(feet), dtype=torch.bool, device=device)
            contact_switches = torch.zeros(max_envs, device=device)
            foot_speed_sum = torch.zeros(max_envs, device=device)
            action_sum = torch.zeros(max_envs, 37, device=device)
            state_sum = torch.zeros(max_envs, 9, device=device)
            feature_count = torch.zeros(max_envs, device=device)
            for step in range(steps):
                t = step * env.step_dt
                physical, ramp_end = command_at_vectors(
                    source_commands, ramp_values, profile_codes, t
                )
                command.external_override[:, :2] = physical[:, :2]
                command.external_override[:, 2] = calibrate_yaw(physical[:, 2])
                if step == 0:
                    command._update_command()
                    obs = wrapped.get_observations().to(device)
                action = torch.zeros(max_envs, 37, device=device)
                for checkpoint, mask in checkpoint_masks.items():
                    if mask.any():
                        with torch.inference_mode():
                            action[mask] = actors[checkpoint](
                                obs["policy"][mask], torch.zeros(int(mask.sum()), device=device)
                            )
                if args.mode == "local":
                    with torch.inference_mode():
                        exp_action = actors["exp012"](
                            obs["policy"], torch.zeros(max_envs, device=device)
                        )
                    local_active = (
                        (local_lambda > 0)
                        & (t >= endpoint_begin)
                        & (t < endpoint_begin + local_steps * env.step_dt)
                    )
                    if local_active.any():
                        lam = local_lambda[local_active, None]
                        action[local_active] = (
                            (1.0 - lam) * action[local_active] + lam * exp_action[local_active]
                        )
                obs, _, done, extras = wrapped.step(action)
                obs = obs.to(device)
                timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
                fall |= done.bool() & ~timeout
                actual_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
                actual_yaw = robot.data.root_ang_vel_b[:, 2]
                force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
                contact = force > 5
                foot_speed = torch.linalg.vector_norm(
                    robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1
                )
                slipping = ((foot_speed > .55) & contact).any(-1)
                slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
                slip |= slip_streak >= 5
                impact |= force.amax(-1) > 3500
                in_endpoint = (t >= endpoint_begin) & (t < endpoint_end)
                in_last2 = (
                    (t >= torch.maximum(endpoint_begin, endpoint_end - 2.0))
                    & (t < endpoint_end)
                )
                sum_speed += torch.where(in_endpoint, actual_speed, 0)
                sum_abs_yaw += torch.where(in_endpoint, actual_yaw.abs(), 0)
                sum_signed_yaw += torch.where(in_endpoint, actual_yaw, 0)
                count += in_endpoint.float()
                sum_speed_last2 += torch.where(in_last2, actual_speed, 0)
                sum_abs_yaw_last2 += torch.where(in_last2, actual_yaw.abs(), 0)
                last2_count += in_last2.float()
                after_ramp = t >= ramp_end
                elapsed = torch.full_like(first_speed, t) - ramp_end
                speed_ok = actual_speed <= .08
                yaw_ok = actual_yaw.abs() <= .08
                first_speed = torch.where(after_ramp & speed_ok & torch.isnan(first_speed), elapsed, first_speed)
                first_yaw = torch.where(after_ramp & yaw_ok & torch.isnan(first_yaw), elapsed, first_yaw)
                first_combined = torch.where(
                    after_ramp & speed_ok & yaw_ok & torch.isnan(first_combined),
                    elapsed, first_combined,
                )
                switches = (contact != prev_contact).any(-1).float()
                contact_switches += torch.where(in_last2, switches, 0)
                foot_speed_sum += torch.where(in_last2, foot_speed.mean(-1), 0)
                prev_contact = contact
                action_sum += torch.where(in_last2[:, None], action, 0)
                state = torch.stack((
                    actual_speed, actual_yaw.abs(),
                    robot.data.projected_gravity_b[:, 0],
                    robot.data.projected_gravity_b[:, 1],
                    robot.data.root_pos_w[:, 2],
                    torch.linalg.vector_norm(robot.data.joint_vel, dim=-1),
                    torch.linalg.vector_norm(obs["policy"][:, -37:], dim=-1),
                    torch.linalg.vector_norm(action, dim=-1),
                    contact.float().mean(-1),
                ), dim=-1)
                state_sum += torch.where(in_last2[:, None], state, 0)
                feature_count += in_last2.float()
            den = count.clamp_min(1)
            den2 = last2_count.clamp_min(1)
            mean_speed = sum_speed / den
            mean_abs_yaw = sum_abs_yaw / den
            mean_signed_yaw = sum_signed_yaw / den
            last2_speed = sum_speed_last2 / den2
            last2_abs_yaw = sum_abs_yaw_last2 / den2
            safe = ~(fall | slip | impact)
            practical = (mean_speed <= .08) & (mean_abs_yaw <= .08) & safe
            guard_success = (mean_speed <= .08) & (mean_signed_yaw.abs() <= .08) & safe
            features = state_sum / feature_count.clamp_min(1)[:, None]
            mean_action = action_sum / last2_count.clamp_min(1)[:, None]
            for local in range(active):
                ji, episode = chunk[local]
                job = jobs[ji]
                row = {
                    **asdict(job),
                    "episode": episode,
                    "practical_stop_success": int(practical[local]),
                    "guard_stop_success": int(guard_success[local]),
                    "translation_stop_pass": int(mean_speed[local] <= .08),
                    "yaw_stop_pass": int(mean_abs_yaw[local] <= .08),
                    "fall": int(fall[local]), "slip": int(slip[local]), "impact": int(impact[local]),
                    "final_speed": float(mean_speed[local]),
                    "final_abs_yaw": float(mean_abs_yaw[local]),
                    "final_signed_yaw": float(mean_signed_yaw[local]),
                    "last2_speed": float(last2_speed[local]),
                    "last2_abs_yaw": float(last2_abs_yaw[local]),
                    "translation_acquisition_s": (
                        float(first_speed[local]) if torch.isfinite(first_speed[local]) else -1.0
                    ),
                    "yaw_acquisition_s": (
                        float(first_yaw[local]) if torch.isfinite(first_yaw[local]) else -1.0
                    ),
                    "combined_acquisition_s": (
                        float(first_combined[local]) if torch.isfinite(first_combined[local]) else -1.0
                    ),
                    "contact_switches_last2": float(contact_switches[local]),
                    "mean_foot_speed_last2": float(foot_speed_sum[local] / den2[local]),
                }
                all_rows.append(row)
                state_rows.append({
                    "checkpoint": job.checkpoint, "direction_deg": job.direction_deg,
                    "source_yaw": job.source_yaw, "episode": episode,
                    "success": int(practical[local]),
                    **{f"feature_{i}": float(features[local, i]) for i in range(9)},
                    **{f"action_{i}": float(mean_action[local, i]) for i in range(37)},
                })
            print(json.dumps({
                "mode": args.mode, "processed": min(chunk_start + active, len(expanded)),
                "total": len(expanded),
            }), flush=True)
        keys = [
            "checkpoint", "direction_deg", "source_yaw", "ramp_s", "hold_s",
            "profile", "source_speed", "local_lambda", "local_steps",
        ]
        summary = summarize(all_rows, keys)
        write_rows(f"_w2_d1_{args.mode}_episodes.csv", all_rows)
        write_rows(f"_w2_d1_{args.mode}_summary.csv", summary)
        if args.mode == "baseline":
            write_rows("_w2_d1_baseline_state_action.csv", state_rows)
        (OUT / f"_w2_d1_{args.mode}.json").write_text(
            json.dumps({"mode": args.mode, "jobs": len(jobs), "episodes": len(all_rows),
                        "summary": summary}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        wrapped.close()


if __name__ == "__main__":
    main()
