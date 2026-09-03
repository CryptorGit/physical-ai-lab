"""Frozen-actor W2-P1 stop-teacher recovery positive control."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_p1_practical_stop_endpoint_acquisition"
)
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
TEACHER = REPO / (
    "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/"
    "stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
)
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: F401
import g1_omnidirectional.tasks  # noqa: F401
from g1_omnidirectional.policy import FrozenGaitActor
from g1_omnidirectional.yaw_calibration import calibrate_yaw
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--max-envs", type=int, default=1600)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

SWITCHES = {"SW1_RAMP_START": 3.0, "SW2_RAMP_MID": 3.75, "SW3_ZERO_TARGET": 4.5}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def minjerk(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finalize(rows: list[dict]) -> tuple[bool, str | None]:
    """Persist results before the simulation application shuts down."""
    write_csv(OUT / "stop_teacher_recovery_positive_control.csv", rows)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["switch"], row["direction_deg"], row["source_yaw"])].append(row)
    conditions = []
    for (switch, direction, yaw), subset in sorted(grouped.items()):
        conditions.append({
            "switch": switch, "direction_deg": direction, "source_yaw": yaw,
            "episodes": len(subset),
            "success_rate": sum(r["success"] for r in subset) / len(subset),
            "mean_final_speed": sum(r["final_speed"] for r in subset) / len(subset),
            "mean_final_abs_yaw": sum(r["final_abs_yaw"] for r in subset) / len(subset),
            "fall_rate": sum(r["fall"] for r in subset) / len(subset),
            "dangerous_slip_rate": sum(r["dangerous_slip"] for r in subset) / len(subset),
            "impact_rate": sum(r["impact"] for r in subset) / len(subset),
        })
    switches = {}
    for switch in SWITCHES:
        subset = [r for r in conditions if r["switch"] == switch]
        switches[switch] = {
            "condition_pass_count": sum(r["success_rate"] >= .90 and r["fall_rate"] <= .05 for r in subset),
            "condition_count": len(subset),
            "minimum_success_rate": min(r["success_rate"] for r in subset),
            "aggregate_success_rate": sum(r["success_rate"] for r in subset) / len(subset),
            "teacher_switch_s": SWITCHES[switch],
        }
    passing = [s for s in ("SW3_ZERO_TARGET", "SW2_RAMP_MID", "SW1_RAMP_START")
               if switches[s]["condition_pass_count"] == 24]
    selected = passing[0] if passing else None
    payload = {
        "parent_sha256": digest(PARENT), "teacher_sha256": digest(TEACHER),
        "episodes": len(rows), "conditions": conditions, "switch_summary": switches,
        "selected_switch": selected, "gate_pass": selected is not None,
        "failure_classification": None if selected else "EXP013_W2_P1_STOP_TEACHER_POSITIVE_CONTROL_FAIL",
    }
    (OUT / "stop_teacher_recovery_positive_control.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "selected_teacher_switch_contract.json").write_text(
        json.dumps({
            "selected": selected,
            "selection_order": ["SW3_ZERO_TARGET", "SW2_RAMP_MID", "SW1_RAMP_START"],
            "training_authorized": selected is not None,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selected is not None, selected


def main() -> None:
    jobs = [
        (switch, float(direction), float(yaw), episode)
        for switch in SWITCHES for direction in range(0, 360, 45)
        for yaw in (-0.3, 0.0, 0.3) for episode in range(100)
    ]
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = min(args.max_envs, len(jobs))
    cfg.episode_length_s = 15.0
    cfg.seed = 20276011
    if args.device:
        cfg.sim.device = agent_cfg.device = args.device
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=agent_cfg.clip_actions,
        )
        env = wrapped.unwrapped
        device = env.device
        parent = FrozenGaitActor(PARENT).to(device).eval()
        teacher = FrozenGaitActor(TEACHER).to(device).eval()
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        feet = sensor.find_bodies(".*_ankle_roll_link")[0]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        command.external_override.zero_()
        expanded = 0
        while expanded < len(jobs):
            batch_jobs = jobs[expanded:expanded + cfg.scene.num_envs]
            count = len(batch_jobs)
            padded_jobs = batch_jobs
            if count < env.num_envs:
                padded_jobs = batch_jobs + [batch_jobs[i % count] for i in range(env.num_envs - count)]
            env_ids = torch.arange(env.num_envs, device=device)
            env.reset(env_ids=env_ids)
            obs = wrapped.get_observations().to(device)
            switch_times = torch.tensor([SWITCHES[j[0]] for j in padded_jobs], device=device)
            source = torch.zeros(env.num_envs, 3, device=device)
            for index, (_, direction, yaw, _) in enumerate(padded_jobs):
                angle = math.radians(direction)
                source[index] = torch.tensor(
                    [.3 * math.cos(angle), .3 * math.sin(angle), yaw], device=device
                )
            fall = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            slip = torch.zeros_like(fall)
            impact = torch.zeros_like(fall)
            slip_streak = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            speed_sum = torch.zeros(env.num_envs, device=device)
            abs_yaw_sum = torch.zeros(env.num_envs, device=device)
            samples = torch.zeros(env.num_envs, device=device)
            max_action = torch.zeros(env.num_envs, device=device)
            steps = int(round(10.5 / env.step_dt))
            for step in range(steps):
                time_s = step * env.step_dt
                if time_s < 3.0:
                    physical = source
                else:
                    scale = 1.0 - minjerk(torch.full(
                        (env.num_envs,), (time_s - 3.0) / 1.5, device=device
                    ))
                    physical = source * scale[:, None]
                command.external_override[:, :2] = physical[:, :2]
                command.external_override[:, 2] = calibrate_yaw(physical[:, 2])
                if step == 0:
                    command._update_command()
                    obs = wrapped.get_observations().to(device)
                with torch.inference_mode():
                    parent_action = parent(obs["policy"], torch.zeros(env.num_envs, device=device))
                    teacher_action = teacher(obs["policy"], torch.zeros(env.num_envs, device=device))
                use_teacher = time_s >= switch_times
                action = torch.where(use_teacher[:, None], teacher_action, parent_action)
                max_action = torch.maximum(max_action, action.abs().amax(-1))
                obs, _, done, extras = wrapped.step(action)
                obs = obs.to(device)
                timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
                fall |= done.bool() & ~timeout
                force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
                foot_speed = torch.linalg.vector_norm(
                    robot.data.body_lin_vel_w[:, robot.find_bodies(".*_ankle_roll_link")[0], :2], dim=-1
                )
                slipping = ((foot_speed > .55) & (force > 5)).any(-1)
                slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
                slip |= slip_streak >= 5
                impact |= force.amax(-1) > 3500
                if time_s >= 8.5:  # final recovery hold last two seconds
                    speed_sum += torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
                    abs_yaw_sum += robot.data.root_ang_vel_b[:, 2].abs()
                    samples += 1
            speed = speed_sum / samples.clamp_min(1)
            abs_yaw = abs_yaw_sum / samples.clamp_min(1)
            success = (speed <= .08) & (abs_yaw <= .08) & ~fall & ~slip & ~impact
            for index, (switch, direction, yaw, episode) in enumerate(batch_jobs):
                rows.append({
                    "switch": switch, "direction_deg": direction, "source_yaw": yaw,
                    "episode": episode, "success": int(success[index]),
                    "final_speed": float(speed[index]), "final_abs_yaw": float(abs_yaw[index]),
                    "fall": int(fall[index]), "dangerous_slip": int(slip[index]),
                    "impact": int(impact[index]), "max_abs_action": float(max_action[index]),
                    "teacher_switch_s": SWITCHES[switch], "teacher_intervention_s": 10.5 - SWITCHES[switch],
                })
            expanded += count
            print(json.dumps({"processed": expanded, "total": len(jobs)}), flush=True)
        gate_pass, selected = finalize(rows)
        print(json.dumps({"gate_pass": gate_pass, "selected": selected}), flush=True)
        wrapped.close()


if __name__ == "__main__":
    main()
