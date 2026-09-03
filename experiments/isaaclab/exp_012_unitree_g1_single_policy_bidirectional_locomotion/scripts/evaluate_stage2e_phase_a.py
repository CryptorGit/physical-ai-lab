"""Evaluate all Phase-A checkpoints, select one, and diagnose emergent gradients."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2e_phase_a_run_acquisition_preflight"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import minimum_jerk, wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = __import__("argparse").ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

CHECKPOINTS = (0, 1, 5, 10, 20, 30, 40, 50, 75, 100)
RUN_SPEEDS = (2.3, 2.4, 2.5, 2.6)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    rows = list(rows)
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_path(iteration):
    return OUT / "checkpoints" / ("model_initial.pt" if iteration == 0 else f"model_{iteration}.pt")


def q95(values):
    return float(torch.quantile(torch.tensor(values), .95)) if values else 0.0


def specs():
    rows = []
    for speed in RUN_SPEEDS:
        rows.append({"condition": f"run_{speed:.1f}", "kind": "steady", "target": speed, "episodes": 20})
    for speed in (0.0, 0.6, 1.2):
        rows.append({"condition": f"retention_{speed:.1f}", "kind": "steady", "target": speed, "episodes": 10})
    rows.append({"condition": "retention_0.6_to_0.0", "kind": "stop", "target": 0.0, "episodes": 10})
    return rows


def evaluate(wrapped, runner, checkpoint, checkpoint_iteration):
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
    sensor_feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
    robot_feet = [
        next(i for i, body in enumerate(robot.body_names) if body == sensor.body_names[sensor_id])
        for sensor_id in sensor_feet
    ]
    rowspec = specs()
    n = sum(x["episodes"] for x in rowspec)
    spec_id = torch.cat([
        torch.full((x["episodes"],), i, device=runner.device, dtype=torch.long)
        for i, x in enumerate(rowspec)
    ])
    obs, _ = wrapped.reset()
    obs = obs.to(runner.device)
    ref_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
    fallen = torch.zeros(n, dtype=torch.bool, device=runner.device)
    quality_steps = torch.zeros(n, device=runner.device)
    speed_error = torch.zeros(n, device=runner.device)
    heading_values = [[] for _ in range(n)]
    flight_events = torch.zeros(n, dtype=torch.long, device=runner.device)
    safe_flights = torch.zeros(n, dtype=torch.long, device=runner.device)
    alternating = torch.zeros(n, dtype=torch.long, device=runner.device)
    completions = torch.zeros(n, dtype=torch.long, device=runner.device)
    precursor = torch.zeros(n, dtype=torch.long, device=runner.device)
    flight_streak = torch.zeros(n, dtype=torch.long, device=runner.device)
    last_landing = torch.full((n,), -1, dtype=torch.long, device=runner.device)
    dangerous_slip_streak = torch.zeros(n, dtype=torch.long, device=runner.device)
    dangerous_slip = torch.zeros(n, dtype=torch.bool, device=runner.device)
    impact = torch.zeros(n, dtype=torch.bool, device=runner.device)
    saturation_streak = torch.zeros(n, dtype=torch.long, device=runner.device)
    saturation = torch.zeros(n, dtype=torch.bool, device=runner.device)
    max_flight = torch.zeros(n, dtype=torch.long, device=runner.device)
    dt = float(env.step_dt)
    for step in range(round(8.0 / dt)):
        t = step * dt
        command = torch.zeros(n, device=runner.device)
        quality = torch.zeros(n, dtype=torch.bool, device=runner.device)
        for index, spec in enumerate(rowspec):
            mask = spec_id == index
            if spec["kind"] == "steady":
                command[mask] = spec["target"]
                quality[mask] = t >= 1.0
            else:
                if t < 2.0:
                    value = .6
                elif t < 3.5:
                    value = .6 * (1.0 - float(minimum_jerk((t - 2.0) / 1.5)))
                else:
                    value = 0.0
                command[mask] = value
                quality[mask] = t >= 4.0
        command_term.external_override[:, 0] = command
        command_term.external_override[:, 1:] = 0.0
        if step == 0:
            obs = wrapped.get_observations().to(runner.device)
        with torch.inference_mode():
            action = policy(obs)
        obs, _, dones, extras = wrapped.step(action)
        obs = obs.to(runner.device)
        timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
        fallen |= dones.bool() & ~timeouts
        actual = robot.data.root_lin_vel_b[:, 0]
        quality_steps += quality.float()
        speed_error += (actual - command).abs() * quality.float()
        heading = wrapped_heading_error(ref_yaw, yaw_from_quat_wxyz(robot.data.root_quat_w)).abs()
        for env_id in torch.where(quality)[0].tolist():
            heading_values[env_id].append(float(heading[env_id]))
        forces = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].norm(dim=-1)
        contacts = forces > 5.0
        flight = contacts.sum(-1) == 0
        started = flight & (flight_streak == 0) & quality
        previous_streak = flight_streak.clone()
        flight_events += started.long()
        flight_streak = torch.where(flight, flight_streak + 1, torch.zeros_like(flight_streak))
        max_flight = torch.maximum(max_flight, flight_streak)
        landing = ~flight & (previous_streak > 0) & quality
        single = landing & (contacts.sum(-1) == 1)
        foot = contacts.long().argmax(-1)
        safe = single & (previous_streak >= 2) & (previous_streak <= 8)
        alt = safe & (last_landing >= 0) & (foot != last_landing)
        safe_flights += safe.long()
        alternating += alt.long()
        last_landing[single] = foot[single]
        raw_run = reward_term.last_raw_reward
        completions += (raw_run >= 1.0).long()
        precursor += ((raw_run > 0) & (raw_run < 1.0)).long()
        foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
        slipping = ((foot_speed > .55) & contacts).any(-1) & quality
        dangerous_slip_streak = torch.where(slipping, dangerous_slip_streak + 1, torch.zeros_like(dangerous_slip_streak))
        dangerous_slip |= dangerous_slip_streak >= 5
        impact |= (forces.amax(-1) > 3500.0) & quality
        limits = robot.data.joint_vel_limits
        if limits.ndim == 3:
            limits = limits[..., 1].abs()
        sat = (robot.data.joint_vel.abs() / torch.clamp(limits, min=1e-6) > .95).any(-1) & quality
        saturation_streak = torch.where(sat, saturation_streak + 1, torch.zeros_like(saturation_streak))
        saturation |= saturation_streak >= 5

    records = []
    checkpoint_sha = sha(checkpoint)
    for env_id in range(n):
        spec = rowspec[int(spec_id[env_id])]
        denom = max(1.0, float(quality_steps[env_id]))
        mae = float(speed_error[env_id] / denom)
        if spec["target"] >= 2.3:
            gait = "FALL" if fallen[env_id] else (
                "PERIODIC_RUNNING" if int(flight_events[env_id]) >= 4 and int(safe_flights[env_id]) >= 3
                and int(alternating[env_id]) >= 3 else (
                    "ISOLATED_FLIGHT" if int(flight_events[env_id]) else "IRREGULAR"))
            success = gait == "PERIODIC_RUNNING" and mae <= .25
        elif spec["target"] == 0:
            gait = "FALL" if fallen[env_id] else ("STAND" if mae <= .08 and int(flight_events[env_id]) == 0 else "IRREGULAR")
            success = gait == "STAND"
        else:
            gait = "FALL" if fallen[env_id] else ("WALK_LIKE" if mae <= .20 else "IRREGULAR")
            success = gait == "WALK_LIKE"
        records.append({
            "checkpoint_iteration": checkpoint_iteration,
            "checkpoint_sha256": checkpoint_sha, "condition": spec["condition"],
            "episode": env_id - int((spec_id[:env_id] != spec_id[env_id]).sum()),
            "target_speed": spec["target"], "success": success, "gait": gait,
            "fall": bool(fallen[env_id]), "speed_mae": mae,
            "heading_p95": q95(heading_values[env_id]), "flight_events": int(flight_events[env_id]),
            "safe_flight_events": int(safe_flights[env_id]), "alternating_landings": int(alternating[env_id]),
            "completion_reward_fires": int(completions[env_id]), "precursor_steps": int(precursor[env_id]),
            "max_flight_duration_s": int(max_flight[env_id]) * dt,
            "dangerous_slip": bool(dangerous_slip[env_id]), "impact_failure": bool(impact[env_id]),
            "long_dwell_saturation": bool(saturation[env_id]),
        })
    summary = {}
    grouped = defaultdict(list)
    for row in records:
        grouped[row["condition"]].append(row)
    for name, group in grouped.items():
        summary[name] = {
            "episodes": len(group), "success_rate": sum(x["success"] for x in group) / len(group),
            "periodic_running_rate": sum(x["gait"] == "PERIODIC_RUNNING" for x in group) / len(group),
            "fall_rate": sum(x["fall"] for x in group) / len(group),
            "speed_mae": sum(x["speed_mae"] for x in group) / len(group),
            "heading_p95": q95([x["heading_p95"] for x in group]),
            "completion_reward_fires": sum(x["completion_reward_fires"] for x in group),
            "precursor_steps": sum(x["precursor_steps"] for x in group),
            "alternating_landings": sum(x["alternating_landings"] for x in group),
            "dangerous_slip_rate": sum(x["dangerous_slip"] for x in group) / len(group),
            "impact_failure_rate": sum(x["impact_failure"] for x in group) / len(group),
            "long_dwell_saturation_rate": sum(x["long_dwell_saturation"] for x in group) / len(group),
        }
    return records, summary


def discounted_returns(reward, gamma=.99):
    output, running = torch.zeros_like(reward), torch.zeros(reward.shape[1], device=reward.device)
    for t in range(reward.shape[0] - 1, -1, -1):
        running = reward[t] + gamma * running
        output[t] = running
    return output


def gradient_diagnostic(wrapped, runner, checkpoint, iteration):
    runner.load(str(checkpoint), load_cfg={
        "actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False,
    }, strict=True, map_location=runner.device)
    env = wrapped.unwrapped
    command = env.command_manager.get_term("base_velocity")
    command.external_override_enabled = True
    command.external_override[:, 0] = 2.5
    command.external_override[:, 1:] = 0
    obs, _ = wrapped.reset()
    obs = obs.to(runner.device)
    with torch.inference_mode():
        for _ in range(250):
            obs, _, _, _ = wrapped.step(runner.alg.actor(obs, stochastic_output=False))
            obs = obs.to(runner.device)
    observations, actions, logps, component_steps = [], [], [], []
    names = list(env.reward_manager.active_terms)
    run_index = names.index("safe_periodic_flight")
    reward_term = env.reward_manager.get_term_cfg("safe_periodic_flight").func
    with torch.inference_mode():
        for _ in range(240):
            action = runner.alg.actor(obs, stochastic_output=True)
            logp = runner.alg.actor.get_output_log_prob(action)
            observations.append(obs.clone())
            actions.append(action.clone())
            logps.append(logp.clone())
            obs, _, _, _ = wrapped.step(action)
            obs = obs.to(runner.device)
            component_steps.append(env.reward_manager._step_reward.clone())
    observations = torch.stack(observations).flatten(0, 1)
    actions = torch.stack(actions).flatten(0, 1)
    old_logp = torch.stack(logps).flatten()
    components = torch.stack(component_steps)
    run_reward = components[:, :, run_index]
    completion = torch.where(run_reward >= 1.0, run_reward, 0.0)
    precursor = run_reward - completion
    base = components.sum(-1) - run_reward
    rewards = {
        "base": base, "precursor": precursor, "completion": completion,
        "run_specific": run_reward, "total": components.sum(-1),
    }
    params = list(runner.alg.actor.parameters())
    vectors, result = {}, {"iteration": iteration, "checkpoint_sha256": sha(checkpoint)}
    for name, reward in rewards.items():
        advantage = discounted_returns(reward).flatten()
        advantage -= advantage.mean()
        runner.alg.actor(observations, stochastic_output=True)
        new_logp = runner.alg.actor.get_output_log_prob(actions)
        loss = -(advantage.detach() * torch.exp(new_logp - old_logp)).mean()
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        vector = torch.cat([
            (torch.zeros_like(p) if g is None else g).reshape(-1)
            for p, g in zip(params, gradients)
        ]).detach()
        vectors[name] = vector
        result[f"{name}_gradient_norm"] = float(torch.linalg.vector_norm(vector))
    base_norm = torch.linalg.vector_norm(vectors["base"])
    total_norm = torch.linalg.vector_norm(vectors["total"])
    result.update({
        "precursor_to_base": result["precursor_gradient_norm"] / (result["base_gradient_norm"] + 1e-12),
        "completion_to_base": result["completion_gradient_norm"] / (result["base_gradient_norm"] + 1e-12),
        "run_specific_to_total": result["run_specific_gradient_norm"] / (result["total_gradient_norm"] + 1e-12),
        "completion_total_cosine": float(
            torch.dot(vectors["completion"], vectors["total"])
            / (torch.linalg.vector_norm(vectors["completion"]) * total_norm + 1e-12)),
        "completion_event_samples": int((completion > 0).sum()),
    })
    optimizer = runner.alg.optimizer
    moment_parts = []
    for parameter in params:
        state = optimizer.state.get(parameter, {})
        if "exp_avg" in state and "exp_avg_sq" in state:
            moment_parts.append((-state["exp_avg"] / (state["exp_avg_sq"].sqrt() + optimizer.param_groups[0]["eps"])).reshape(-1))
        else:
            moment_parts.append(torch.zeros_like(parameter).reshape(-1))
    adam_direction = torch.cat(moment_parts).detach()
    result["gradient_vs_adam_update_cosine"] = float(
        torch.dot(vectors["total"], adam_direction)
        / (torch.linalg.vector_norm(vectors["total"]) * torch.linalg.vector_norm(adam_direction) + 1e-12)
    )
    result["descent_vs_adam_update_cosine"] = -result["gradient_vs_adam_update_cosine"]
    result["adam_update_direction_norm"] = float(torch.linalg.vector_norm(adam_direction))
    return result


def main():
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = 120
    cfg.seed = 20266021
    cfg.episode_length_s = 20.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    agent_cfg.seed = 20266021
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    import importlib.metadata
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        all_records, results = [], {}
        timeline = list(csv.DictReader((OUT / "phase_a_run_event_timeline.csv").open(encoding="utf-8")))
        timeline_by_iter = {int(row["iteration"]): row for row in timeline}
        for iteration in CHECKPOINTS:
            checkpoint = checkpoint_path(iteration)
            records, summary = evaluate(wrapped, runner, checkpoint, iteration)
            run_groups = [summary[f"run_{speed:.1f}"] for speed in RUN_SPEEDS]
            completion = sum(x["completion_reward_fires"] for x in run_groups)
            run_samples = 4 * 20 * 7 * 50
            density = completion / run_samples
            catastrophic = (
                summary["retention_0.0"]["fall_rate"] > .30
                or summary["retention_0.6"]["success_rate"] < .50
                or summary["retention_1.2"]["success_rate"] < .50
                or summary["retention_0.6_to_0.0"]["success_rate"] < .50
            )
            score = (
                density,
                summary["run_2.4"]["periodic_running_rate"],
                summary["run_2.6"]["periodic_running_rate"],
                -sum(x["fall_rate"] for x in run_groups),
                sum(x["alternating_landings"] for x in run_groups),
                -sum(x["dangerous_slip_rate"] + x["impact_failure_rate"] + x["long_dwell_saturation_rate"] for x in run_groups),
                int(not catastrophic),
            )
            results[str(iteration)] = {
                "iteration": iteration, "checkpoint": str(checkpoint.relative_to(REPO)),
                "sha256": sha(checkpoint), "completion_density": density,
                "completion_reward_fires": completion, "catastrophic_retention_collapse": catastrophic,
                "selection_score": list(score), "conditions": summary,
                "training_iteration_events": timeline_by_iter.get(iteration),
            }
            all_records.extend(records)
            print(f"[PhaseA-eval] iteration={iteration} density={density:.6f} score={score}", flush=True)
        selected = max(CHECKPOINTS, key=lambda x: tuple(results[str(x)]["selection_score"]))
        selected_payload = results[str(selected)]
        selected_payload["selection_precedence"] = [
            "completion reward density", "completion reproducibility", "2.4 periodicity",
            "2.6 periodicity", "RUN fall", "safe-flight density", "alternating landing",
            "slip/impact/saturation", "catastrophic retention collapse",
        ]
        write_csv("phase_a_capability_results.csv", [
            row for row in all_records if row["condition"].startswith("run_")
        ])
        write_csv("phase_a_retention_diagnostic.csv", [
            row for row in all_records if row["condition"].startswith("retention_")
        ])
        dump("phase_a_evaluation_summary.json", results)
        dump("selected_phase_a_checkpoint.json", selected_payload)
        first_completion = next(
            int(row["iteration"]) for row in timeline if int(row["completion_reward_fire_count"]) > 0
        )
        first_saved_after = next(i for i in CHECKPOINTS if i >= first_completion)
        diagnostic_iterations = list(dict.fromkeys([0, first_saved_after, selected]))
        gradients = [
            gradient_diagnostic(wrapped, runner, checkpoint_path(iteration), iteration)
            for iteration in diagnostic_iterations
        ]
        dump("run_specific_gradient_after_emergence.json", {
            "first_completion_training_iteration": first_completion,
            "first_saved_checkpoint_after_emergence": first_saved_after,
            "selected_iteration": selected, "diagnostics": gradients,
            "optimizer_steps_executed": 0,
        })
        dump("adam_moment_alignment.json", {
            "interpretation": "Adam update direction is -m/(sqrt(v)+eps); both raw-gradient and descent-direction cosines are recorded.",
            "checkpoints": [{
                "iteration": x["iteration"],
                "gradient_vs_adam_update_cosine": x["gradient_vs_adam_update_cosine"],
                "descent_vs_adam_update_cosine": x["descent_vs_adam_update_cosine"],
                "adam_update_direction_norm": x["adam_update_direction_norm"],
            } for x in gradients],
            "optimizer_reset": False, "optimizer_steps_executed": 0,
        })
        wrapped.close()


if __name__ == "__main__":
    main()
