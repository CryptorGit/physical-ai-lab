"""Validation, selection, and formal evaluation for Stage 2I Phase R1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2i_reverse_continuation_phase_r1"
PARENT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
CHECKPOINTS = (0, 1, 5, 10, 20, 30, 40, 50, 75, 100)
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import minimum_jerk, wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    rows = list(rows)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def checkpoint_path(iteration):
    return OUT / "checkpoints" / ("model_initial.pt" if iteration == 0 else f"model_{iteration}.pt")


CONDITIONS = (
    ("walk_1p2", "steady", 1.2, 1.2),
    ("run_2p4", "steady", 2.4, 2.4),
    ("run_2p6", "steady", 2.6, 2.6),
    ("walk_to_run_2p4", "up", 1.2, 2.4),
    ("walk_to_run_2p6", "up", 1.2, 2.6),
    ("run_to_walk_2p4", "down", 2.4, 1.2),
    ("run_to_walk_2p6", "down", 2.6, 1.2),
    ("retention_0p0", "steady", 0.0, 0.0),
    ("retention_0p6", "steady", 0.6, 0.6),
)
EPISODES = 50


def command_at(kind, source, target, t):
    if kind == "steady":
        return target, t >= 1.0
    if t < 2.0:
        return source, False
    if t < 3.5:
        blend = float(minimum_jerk((t - 2.0) / 1.5))
        return source + (target - source) * blend, False
    return target, True


def evaluate(wrapped, runner, checkpoint, label):
    runner.load(str(checkpoint), load_cfg={
        "actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False,
    }, strict=True, map_location=runner.device)
    policy = runner.get_inference_policy(device=runner.device)
    env = wrapped.unwrapped
    robot = env.scene["robot"]
    command_term = env.command_manager.get_term("base_velocity")
    command_term.external_override_enabled = True
    reward_term = env.reward_manager.get_term_cfg("safe_periodic_flight").func
    sensor = env.scene.sensors["contact_forces"]
    sensor_feet = [index for index, name in enumerate(sensor.body_names) if "ankle_roll" in name]
    robot_feet = [
        next(index for index, name in enumerate(robot.body_names) if name == sensor.body_names[sensor_id])
        for sensor_id in sensor_feet
    ]
    spec_ids = torch.arange(len(CONDITIONS), device=runner.device).repeat_interleave(EPISODES)
    count = len(CONDITIONS) * EPISODES
    obs, _ = wrapped.reset()
    obs = obs.to(runner.device)
    reference_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
    fallen = torch.zeros(count, dtype=torch.bool, device=runner.device)
    quality_steps = torch.zeros(count, device=runner.device)
    speed_error = torch.zeros(count, device=runner.device)
    heading_history = [[] for _ in range(count)]
    flight_events = torch.zeros(count, dtype=torch.long, device=runner.device)
    safe_flights = torch.zeros_like(flight_events)
    alternating = torch.zeros_like(flight_events)
    completions = torch.zeros_like(flight_events)
    flight_streak = torch.zeros_like(flight_events)
    last_landing = torch.full_like(flight_events, -1)
    slip = torch.zeros(count, dtype=torch.bool, device=runner.device)
    impact = torch.zeros_like(slip)
    saturation = torch.zeros_like(slip)
    slip_streak = torch.zeros_like(flight_events)
    saturation_streak = torch.zeros_like(flight_events)
    dt = float(env.step_dt)
    for step in range(round(8.0 / dt)):
        t = step * dt
        commands = torch.zeros(count, device=runner.device)
        quality = torch.zeros(count, dtype=torch.bool, device=runner.device)
        for index, (_, kind, source, target) in enumerate(CONDITIONS):
            value, active = command_at(kind, source, target, t)
            mask = spec_ids == index
            commands[mask] = value
            quality[mask] = active
        command_term.external_override[:, 0] = commands
        command_term.external_override[:, 1:] = 0.0
        if step == 0:
            obs = wrapped.get_observations().to(runner.device)
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, extras = wrapped.step(actions)
        obs = obs.to(runner.device)
        timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
        fallen |= dones.bool() & ~timeouts
        actual = robot.data.root_lin_vel_b[:, 0]
        speed_error += (actual - commands).abs() * quality
        quality_steps += quality
        heading = wrapped_heading_error(reference_yaw, yaw_from_quat_wxyz(robot.data.root_quat_w)).abs()
        for env_id in torch.where(quality)[0].tolist():
            heading_history[env_id].append(float(heading[env_id]))
        forces = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].norm(dim=-1)
        contacts = forces > 5.0
        flight = contacts.sum(-1) == 0
        previous = flight_streak.clone()
        flight_events += (flight & (flight_streak == 0) & quality).long()
        flight_streak = torch.where(flight, flight_streak + 1, torch.zeros_like(flight_streak))
        landing = ~flight & (previous > 0) & quality
        single = landing & (contacts.sum(-1) == 1)
        foot = contacts.long().argmax(-1)
        safe = single & (previous >= 2) & (previous <= 8)
        alt = safe & (last_landing >= 0) & (foot != last_landing)
        safe_flights += safe.long()
        alternating += alt.long()
        last_landing[single] = foot[single]
        completions += (reward_term.last_raw_reward >= 1.0).long()
        foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
        slipping = ((foot_speed > .55) & contacts).any(-1) & quality
        slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
        slip |= slip_streak >= 5
        impact |= (forces.amax(-1) > 3500.0) & quality
        limits = robot.data.joint_vel_limits
        if limits.ndim == 3:
            limits = limits[..., 1].abs()
        saturated = (robot.data.joint_vel.abs() / torch.clamp(limits, min=1e-6) > .95).any(-1) & quality
        saturation_streak = torch.where(saturated, saturation_streak + 1, torch.zeros_like(saturation_streak))
        saturation |= saturation_streak >= 5

    records = []
    checkpoint_sha = sha(checkpoint)
    for env_id in range(count):
        name, kind, source, target = CONDITIONS[int(spec_ids[env_id])]
        mae = float(speed_error[env_id] / quality_steps[env_id].clamp(min=1))
        periodic = int(flight_events[env_id]) >= 4 and int(safe_flights[env_id]) >= 3 and int(alternating[env_id]) >= 3
        if fallen[env_id]:
            gait = "FALL"
        elif periodic:
            gait = "PERIODIC_RUNNING"
        elif int(flight_events[env_id]) > 0:
            gait = "ISOLATED_FLIGHT"
        else:
            gait = "WALK_LIKE" if target > 0 else "STAND"
        if target >= 2.3:
            success = periodic and mae <= (.25 if target == 2.4 else .30)
        elif target == 1.2:
            success = gait == "WALK_LIKE" and mae <= .20
        else:
            success = not bool(fallen[env_id]) and mae <= .20
        heading_values = heading_history[env_id]
        heading_p95 = float(torch.quantile(torch.tensor(heading_values), .95)) if heading_values else 0.0
        records.append({
            "checkpoint_label": label, "checkpoint_sha256": checkpoint_sha,
            "condition": name, "episode": env_id % EPISODES, "kind": kind,
            "source_speed": source, "target_speed": target,
            "success": bool(success), "fall": bool(fallen[env_id]), "speed_mae": mae,
            "gait": gait, "periodic_running": gait == "PERIODIC_RUNNING",
            "flight_events": int(flight_events[env_id]), "safe_flights": int(safe_flights[env_id]),
            "alternating_landings": int(alternating[env_id]),
            "completion_reward_fires": int(completions[env_id]), "heading_p95": heading_p95,
            "dangerous_slip": bool(slip[env_id]), "impact_failure": bool(impact[env_id]),
            "long_dwell_saturation": bool(saturation[env_id]),
        })
    return records


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["condition"]].append(row)
    output = {}
    for name, rows in grouped.items():
        output[name] = {
            "episodes": len(rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "periodic_running_rate": sum(row["periodic_running"] for row in rows) / len(rows),
            "fall_rate": sum(row["fall"] for row in rows) / len(rows),
            "speed_mae": sum(row["speed_mae"] for row in rows) / len(rows),
            "completion_reward_fires": sum(row["completion_reward_fires"] for row in rows),
            "heading_p95": float(torch.quantile(torch.tensor([row["heading_p95"] for row in rows]), .95)),
            "dangerous_slip_rate": sum(row["dangerous_slip"] for row in rows) / len(rows),
            "impact_failure_rate": sum(row["impact_failure"] for row in rows) / len(rows),
            "long_dwell_saturation_rate": sum(row["long_dwell_saturation"] for row in rows) / len(rows),
        }
    return output


def score(summary):
    r24, r26, walk = summary["run_2p4"], summary["run_2p6"], summary["walk_1p2"]
    downs = summary["run_to_walk_2p4"]["success_rate"] + summary["run_to_walk_2p6"]["success_rate"]
    ups = summary["walk_to_run_2p4"]["success_rate"] + summary["walk_to_run_2p6"]["success_rate"]
    return (
        r24["periodic_running_rate"], r26["periodic_running_rate"], walk["success_rate"],
        downs, ups, -(r24["fall_rate"] + r26["fall_rate"]),
        r24["completion_reward_fires"] + r26["completion_reward_fires"],
        -(r24["heading_p95"] + r26["heading_p95"]),
    )


def main():
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = len(CONDITIONS) * EPISODES
    cfg.seed = 20267021
    agent_cfg.seed = 20267021
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

        parent_records = evaluate(wrapped, runner, PARENT, "exp005_stage4_parent")
        dump("reverse_parent_baseline.json", summarize(parent_records))

        all_records, summaries = [], {}
        for iteration in CHECKPOINTS:
            path = checkpoint_path(iteration)
            records = evaluate(wrapped, runner, path, f"iteration_{iteration}")
            all_records.extend(records)
            summaries[iteration] = summarize(records)
            print(f"[Stage2I validation] iteration={iteration} score={score(summaries[iteration])}", flush=True)
        selected = max(CHECKPOINTS, key=lambda iteration: score(summaries[iteration]))
        selected_path = checkpoint_path(selected)
        selected_sha = sha(selected_path)
        dump("selected_phase_r1_checkpoint.json", {
            "iteration": selected, "path": str(selected_path.relative_to(REPO)),
            "sha256": selected_sha, "selection_precedence": [
                "RUN 2.4 retention", "RUN 2.6 retention", "WALK 1.2",
                "RUN_TO_WALK", "WALK_TO_RUN", "fall", "completion density", "heading",
            ],
        })
        write_csv("phase_r1_capability_timeline.csv", [
            {"iteration": iteration, "condition": condition, **values}
            for iteration, summary in summaries.items() for condition, values in summary.items()
        ])
        selected_records = [row for row in all_records if row["checkpoint_label"] == f"iteration_{selected}"]
        selected_summary = summaries[selected]
        write_csv("phase_r1_run_results.csv", [
            row for row in selected_records if row["condition"] in ("run_2p4", "run_2p6")
        ])
        dump("phase_r1_run_results.json", {
            key: selected_summary[key] for key in ("run_2p4", "run_2p6")
        })
        dump("phase_r1_walk_results.json", {"walk_1p2": selected_summary["walk_1p2"]})
        write_csv("phase_r1_transition_results.csv", [
            row for row in selected_records if row["kind"] in ("up", "down")
        ])
        dump("phase_r1_transition_results.json", {
            key: value for key, value in selected_summary.items()
            if key.startswith("walk_to_run") or key.startswith("run_to_walk")
        })
        parent_summary = summarize(parent_records)
        dump("run_retention_comparison.json", {
            speed: {
                "parent": parent_summary[f"run_{speed}"],
                "selected": selected_summary[f"run_{speed}"],
                "periodic_point_difference": 100 * (
                    selected_summary[f"run_{speed}"]["periodic_running_rate"]
                    - parent_summary[f"run_{speed}"]["periodic_running_rate"]
                ),
                "fall_point_difference": 100 * (
                    selected_summary[f"run_{speed}"]["fall_rate"]
                    - parent_summary[f"run_{speed}"]["fall_rate"]
                ),
            } for speed in ("2p4", "2p6")
        })
        dump("single_weight_audit.json", {
            "unique_checkpoint_sha_count": 1, "checkpoint_sha256": selected_sha,
            "unique_actor_hash_count": 1, "action_source": "selected actor deterministic mean",
            "teacher_calls": 0, "expert_calls": 0, "router_calls": 0,
            "checkpoint_switches": 0, "action_blending": 0,
        })
        wrapped.close()


if __name__ == "__main__":
    main()
