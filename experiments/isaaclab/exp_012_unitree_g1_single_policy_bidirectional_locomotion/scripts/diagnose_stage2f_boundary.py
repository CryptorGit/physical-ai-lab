"""Frozen-policy Phase-A boundary diagnosis (no optimizer step, no checkpoint write)."""

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
STAGE2E = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2e_phase_a_run_acquisition_preflight"
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2f_phase_a_boundary_diagnosis"
RAW = OUT / "raw"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = __import__("argparse").ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

CHECKPOINTS = (0, 10, 20, 40, 50, 75, 100)
SWEEP_CHECKPOINTS = (20, 40, 50, 75, 100)
SPEEDS = (2.30, 2.35, 2.40, 2.45, 2.50, 2.55, 2.60)
MODES = (("D0", 0.0), ("S025", 0.25), ("S050", 0.50), ("S100", 1.0), ("S150", 1.5))
EPISODES_PER_SPEED = 50
STEPS = 500
SEED = 20267021


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else ["status"])
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def append_csv(name, rows):
    rows = list(rows)
    if not rows:
        return
    path = OUT / name
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def append_jsonl(name, rows):
    with (RAW / name).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cp_path(iteration):
    return STAGE2E / "checkpoints" / ("model_initial.pt" if iteration == 0 else f"model_{iteration}.pt")


def q(value, probability):
    value = torch.as_tensor(value, dtype=torch.float32)
    return float(torch.quantile(value, probability)) if value.numel() else 0.0


def cosine(left, right):
    return float(torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right) + 1e-12))


def discounted(reward, gamma=0.99):
    output = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[1], device=reward.device)
    for step in range(reward.shape[0] - 1, -1, -1):
        running = reward[step] + gamma * running
        output[step] = running
    return output


def checkpoint_std(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return payload["actor_state_dict"]["distribution.std_param"].float()


def configure_runtime():
    config, agent_config = resolve_task_config(
        "Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point"
    )
    config.scene.num_envs = len(SPEEDS) * EPISODES_PER_SPEED
    config.seed = SEED
    config.episode_length_s = 20.0
    config.observations.policy.enable_corruption = False
    config.events.base_external_force_torque = None
    config.events.push_robot = None
    agent_config.seed = SEED
    if args.device:
        config.sim.device = args.device
        agent_config.device = args.device
    return config, agent_config


def load_runner(runner, checkpoint, optimizer=False):
    runner.load(str(checkpoint), load_cfg={
        "actor": True, "critic": True, "optimizer": optimizer, "iteration": optimizer, "rnd": False,
    }, strict=True, map_location=runner.device)


def landing_gate_row(env_id, step, speed_target, actual_speed, tilt, vertical_speed,
                     duration, contacts, previous_foot, official, periodic_episode=False):
    contact_count = int(contacts.sum())
    current_foot = int(torch.argmax(contacts.to(torch.int64))) if contact_count else -1
    speed_error = abs(actual_speed - speed_target)
    single = contact_count == 1
    previous_valid = previous_foot >= 0
    alternating = single and previous_valid and current_foot != previous_foot
    gates = {
        "flight_duration": 0.04 - 1e-6 <= duration <= 0.16 + 1e-6,
        "alternation": alternating,
        "speed": speed_error <= 0.30,
        "tilt": tilt <= 0.20,
        "vertical_speed": vertical_speed <= 0.50,
        "single_foot": single,
        "event_memory": previous_valid,
    }
    order = ("flight_duration", "single_foot", "event_memory", "alternation", "speed", "tilt", "vertical_speed")
    first_failure = next((name for name in order if not gates[name]), "COMPLETION_FIRE")
    return {
        "environment": env_id, "step": step, "episode_time_s": step * 0.02,
        "target_speed": speed_target, "actual_speed": actual_speed, "speed_error": speed_error,
        "flight_duration_s": duration, "landing_side": current_foot,
        "previous_landing_side": previous_foot, "contact_count": contact_count,
        "tilt": tilt, "vertical_speed": vertical_speed,
        **{f"{name}_gate": int(value) for name, value in gates.items()},
        "reward_completion": int(official), "first_failure_gate": first_failure,
        "periodic_classifier_episode": int(periodic_episode),
    }


def evaluate_sweep(runner, wrapped, checkpoint, iteration, mode_name, std_multiplier,
                   capture_gradient=False):
    load_runner(runner, checkpoint, optimizer=capture_gradient)
    env = wrapped.unwrapped
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    reward_term = env.reward_manager.get_term_cfg("safe_periodic_flight").func
    command_term = env.command_manager.get_term("base_velocity")
    command_term.external_override_enabled = True
    command_term.external_override.zero_()
    reward_names = list(env.reward_manager.active_terms)
    run_index = reward_names.index("safe_periodic_flight")
    n = wrapped.num_envs
    speed_targets = torch.repeat_interleave(
        torch.tensor(SPEEDS, device=runner.device), EPISODES_PER_SPEED
    )
    torch.manual_seed(SEED + iteration * 101)
    torch.cuda.manual_seed_all(SEED + iteration * 101)
    obs, _ = wrapped.reset()
    obs = obs.to(runner.device)
    command_term.external_override[:, 0] = speed_targets
    command_term.external_override[:, 1:] = 0
    obs = wrapped.get_observations().to(runner.device)
    reference_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
    active = torch.ones(n, dtype=torch.bool, device=runner.device)
    fallen = torch.zeros(n, dtype=torch.bool, device=runner.device)
    quality_steps = torch.zeros(n, device=runner.device)
    speed_error = torch.zeros(n, device=runner.device)
    yaw_sum = torch.zeros(n, device=runner.device)
    yaw_count = torch.zeros(n, device=runner.device)
    heading_samples = [[] for _ in range(n)]
    flight_events = torch.zeros(n, dtype=torch.long, device=runner.device)
    safe_flights = torch.zeros(n, dtype=torch.long, device=runner.device)
    alternating = torch.zeros(n, dtype=torch.long, device=runner.device)
    completions = torch.zeros(n, dtype=torch.long, device=runner.device)
    precursor = torch.zeros(n, dtype=torch.long, device=runner.device)
    flight_duration = torch.zeros(n, device=runner.device)
    maximum_flight = torch.zeros(n, device=runner.device)
    was_flight = torch.zeros(n, dtype=torch.bool, device=runner.device)
    previous_landing = torch.full((n,), -1, dtype=torch.long, device=runner.device)
    slip_streak = torch.zeros(n, dtype=torch.long, device=runner.device)
    dangerous_slip = torch.zeros(n, dtype=torch.bool, device=runner.device)
    impact = torch.zeros(n, dtype=torch.bool, device=runner.device)
    saturation_streak = torch.zeros(n, dtype=torch.long, device=runner.device)
    saturation = torch.zeros(n, dtype=torch.bool, device=runner.device)
    completion_events, landing_samples, selected_gate_rows = [], [], []
    tensor_trace = defaultdict(list) if capture_gradient else None
    joint_names = list(robot.joint_names)
    action_std = checkpoint_std(checkpoint).to(runner.device)
    robot_foot_ids = [
        next(i for i, name in enumerate(robot.body_names) if name == sensor.body_names[int(sensor_id)])
        for sensor_id in reward_term.foot_ids
    ]
    dt = float(env.step_dt)
    for step in range(STEPS):
        command_term.external_override[:, 0] = speed_targets
        command_term.external_override[:, 1:] = 0
        with torch.no_grad():
            mean_action = runner.alg.actor(obs, stochastic_output=False)
            if std_multiplier:
                full_sample = runner.alg.actor(obs, stochastic_output=True)
                action = mean_action + std_multiplier * (full_sample - mean_action)
            else:
                action = mean_action
            runner.alg.actor(obs, stochastic_output=True)
            log_probability = runner.alg.actor.get_output_log_prob(action)
        std = action_std
        zscore = (action - mean_action) / torch.clamp(std, min=1e-8)
        previous_obs = obs.detach().clone()
        previous_action = action.detach().clone()
        previous_mean = mean_action.detach().clone()
        previous_logp = log_probability.detach().clone()
        previous_zscore = zscore.detach().clone()
        obs, _, dones, extras = wrapped.step(action)
        obs = obs.to(runner.device)
        dones = dones.to(runner.device).bool()
        timeouts = extras.get("time_outs", torch.zeros_like(dones)).to(runner.device).bool()
        newly_fallen = dones & ~timeouts & active
        fallen |= newly_fallen
        active &= ~dones
        quality_mask = active if step >= 50 else torch.zeros_like(active)
        actual = robot.data.root_lin_vel_b[:, 0]
        quality_steps += quality_mask.float()
        speed_error += (actual - speed_targets).abs() * quality_mask.float()
        yaw_rate = robot.data.root_ang_vel_b[:, 2]
        yaw_sum += yaw_rate * quality_mask.float()
        yaw_count += quality_mask.float()
        heading = wrapped_heading_error(reference_yaw, yaw_from_quat_wxyz(robot.data.root_quat_w)).abs()
        for env_id in torch.where(quality_mask)[0].tolist():
            heading_samples[env_id].append(float(heading[env_id]))
        forces_history = sensor.data.net_forces_w_history[:, :, reward_term.foot_ids, :]
        contacts = forces_history.norm(dim=-1).amax(dim=1) > 1.0
        flight = contacts.sum(-1) == 0
        started = flight & ~was_flight & active
        flight_events += started.long()
        flight_duration = torch.where(flight & active, flight_duration + dt, flight_duration)
        maximum_flight = torch.maximum(maximum_flight, flight_duration)
        landing = was_flight & ~flight & active
        single = landing & (contacts.sum(-1) == 1)
        landing_foot = contacts.long().argmax(-1)
        safe = single & (flight_duration >= 0.04 - 1e-6) & (flight_duration <= 0.16 + 1e-6)
        alternate = safe & (previous_landing >= 0) & (landing_foot != previous_landing)
        safe_flights += safe.long()
        alternating += alternate.long()
        raw_run = reward_term.last_raw_reward.detach().clone()
        official_completion = raw_run >= 1.0
        completions += official_completion.long()
        precursor += ((raw_run > 0) & (raw_run < 1.0)).long()
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        vertical_speed = robot.data.root_lin_vel_b[:, 2].abs()
        landing_ids = torch.where(landing)[0].tolist()
        for env_id in landing_ids:
            maha = float(torch.linalg.vector_norm(previous_zscore[env_id]))
            row = {
                "checkpoint_iteration": iteration, "action_mode": mode_name,
                "std_multiplier": std_multiplier, "target_speed": float(speed_targets[env_id]),
                "environment": env_id, "step": step, "completion": int(official_completion[env_id]),
                "action_l2_from_mean": float(torch.linalg.vector_norm(previous_action[env_id] - previous_mean[env_id])),
                "mahalanobis_distance": maha, "max_absolute_zscore": float(previous_zscore[env_id].abs().max()),
                "zscore_semicolon": ";".join(f"{float(x):.8g}" for x in previous_zscore[env_id]),
            }
            if bool(official_completion[env_id]):
                event = {
                    **row, "episode_time_s": step * dt,
                    "actual_speed": float(actual[env_id]),
                    "policy_mean_action": [float(x) for x in previous_mean[env_id]],
                    "sampled_action": [float(x) for x in previous_action[env_id]],
                    "policy_std": [float(x) for x in std],
                    "normalized_action_deviation": [float(x) for x in previous_zscore[env_id]],
                    "action_log_probability": float(previous_logp[env_id]),
                    "flight_duration_s": float(flight_duration[env_id]),
                    "landing_side": int(landing_foot[env_id]),
                    "previous_landing_side": int(previous_landing[env_id]),
                    "tilt": float(tilt[env_id]), "vertical_speed": float(vertical_speed[env_id]),
                    "heading_error": float(heading[env_id]),
                }
                completion_events.append(event)
            if len(landing_samples) < 5000:
                landing_samples.append(row)
            if iteration == 50 and mode_name == "D0" and abs(float(speed_targets[env_id]) - 2.4) < 1e-5:
                selected_gate_rows.append(landing_gate_row(
                    env_id, step, float(speed_targets[env_id]), float(actual[env_id]), float(tilt[env_id]),
                    float(vertical_speed[env_id]), float(flight_duration[env_id]), contacts[env_id],
                    int(previous_landing[env_id]), bool(official_completion[env_id]),
                ))
        previous_landing[single] = landing_foot[single]
        flight_duration[landing] = 0.0
        was_flight.copy_(flight)
        foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_foot_ids, :2], dim=-1)
        slipping = ((foot_speed > 0.55) & contacts).any(-1) & quality_mask
        slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
        dangerous_slip |= slip_streak >= 5
        impact |= (forces_history.norm(dim=-1).amax(dim=(1, 2)) > 3500.0) & quality_mask
        limits = robot.data.joint_vel_limits
        if limits.ndim == 3:
            limits = limits[..., 1].abs()
        saturated_now = (robot.data.joint_vel.abs() / torch.clamp(limits, min=1e-6) > 0.95).any(-1) & quality_mask
        saturation_streak = torch.where(saturated_now, saturation_streak + 1, torch.zeros_like(saturation_streak))
        saturation |= saturation_streak >= 5
        if capture_gradient:
            tensor_trace["observation"].append(previous_obs.cpu())
            tensor_trace["action"].append(previous_action.cpu())
            tensor_trace["old_logp"].append(previous_logp.cpu())
            tensor_trace["reward_components"].append(env.reward_manager._step_reward.detach().cpu())
            tensor_trace["completion"].append(official_completion.detach().cpu())
            tensor_trace["landing"].append(landing.detach().cpu())
    records = []
    checkpoint_hash = sha(checkpoint)
    for env_id in range(n):
        target = float(speed_targets[env_id])
        denom = max(1.0, float(quality_steps[env_id]))
        periodic = (
            not bool(fallen[env_id]) and int(flight_events[env_id]) >= 4
            and int(safe_flights[env_id]) >= 3 and int(alternating[env_id]) >= 3
        )
        records.append({
            "checkpoint_iteration": iteration, "checkpoint_sha256": checkpoint_hash,
            "action_mode": mode_name, "std_multiplier": std_multiplier,
            "target_speed": target, "episode": env_id % EPISODES_PER_SPEED,
            "completion_events": int(completions[env_id]),
            "periodic_running": int(periodic), "fall": int(fallen[env_id]),
            "speed_mae": float(speed_error[env_id] / denom),
            "actual_yaw_rate_mean": float(yaw_sum[env_id] / max(1.0, float(yaw_count[env_id]))),
            "heading_p95": q(heading_samples[env_id], 0.95),
            "flight_events": int(flight_events[env_id]), "safe_flight_events": int(safe_flights[env_id]),
            "alternating_landings": int(alternating[env_id]), "precursor_steps": int(precursor[env_id]),
            "maximum_flight_duration_s": float(maximum_flight[env_id]),
            "dangerous_slip": int(dangerous_slip[env_id]), "impact_failure": int(impact[env_id]),
            "long_dwell_saturation": int(saturation[env_id]),
        })
    if capture_gradient:
        tensor_trace = {key: torch.stack(value) for key, value in tensor_trace.items()}
        tensor_trace["reward_names"] = reward_names
        tensor_trace["joint_names"] = joint_names
        tensor_trace["checkpoint"] = str(checkpoint)
        tensor_trace["iteration"] = iteration
        torch.save(tensor_trace, RAW / "iter50_s100_event_trace.pt")
    return records, completion_events, landing_samples, selected_gate_rows


def evaluate_heading_pair(runner, wrapped, checkpoint):
    load_runner(runner, checkpoint, optimizer=False)
    env = wrapped.unwrapped
    robot = env.scene["robot"]
    command_term = env.command_manager.get_term("base_velocity")
    command_term.external_override_enabled = True
    n = wrapped.num_envs
    targets = torch.repeat_interleave(torch.tensor(SPEEDS, device=runner.device), EPISODES_PER_SPEED)
    rows = []
    for controller in ("OFF", "ON"):
        torch.manual_seed(SEED + 9000)
        torch.cuda.manual_seed_all(SEED + 9000)
        obs, _ = wrapped.reset()
        obs = obs.to(runner.device)
        reference = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        fallen = torch.zeros(n, dtype=torch.bool, device=runner.device)
        heading_values = [[] for _ in range(n)]
        yaw_sum = torch.zeros(n, device=runner.device)
        yaw_count = torch.zeros(n, device=runner.device)
        for step in range(STEPS):
            command_term.external_override[:, 0] = targets
            if controller == "ON":
                tau = min(1.0, (step + 1) * float(env.step_dt) / 0.5)
                gate = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
                command_term.external_override[:, 2] = -0.1233 * gate
            else:
                command_term.external_override[:, 2] = 0
            command_term.external_override[:, 1] = 0
            with torch.no_grad():
                action = runner.alg.actor(obs, stochastic_output=False)
            obs, _, dones, extras = wrapped.step(action)
            obs = obs.to(runner.device)
            timeouts = extras.get("time_outs", torch.zeros_like(dones)).to(runner.device).bool()
            fallen |= dones.to(runner.device).bool() & ~timeouts
            if step >= 50:
                heading = wrapped_heading_error(reference, yaw_from_quat_wxyz(robot.data.root_quat_w)).abs()
                yaw_sum += robot.data.root_ang_vel_b[:, 2]
                yaw_count += 1
                for env_id in range(n):
                    heading_values[env_id].append(float(heading[env_id]))
        for env_id in range(n):
            rows.append({
                "controller": controller, "target_speed": float(targets[env_id]),
                "episode": env_id % EPISODES_PER_SPEED, "fall": int(fallen[env_id]),
                "actual_yaw_rate_mean": float(yaw_sum[env_id] / yaw_count[env_id]),
                "heading_p95": q(heading_values[env_id], 0.95),
            })
    return rows


def component_gradients(runner, checkpoint, trace_path):
    load_runner(runner, checkpoint, optimizer=True)
    trace = torch.load(trace_path, map_location=runner.device, weights_only=False)
    completion_mask = trace["completion"].bool()
    event_indices = torch.nonzero(completion_mask, as_tuple=False)
    if not len(event_indices):
        return {"status": "NO_STOCHASTIC_COMPLETION_EVENT"}, [], {}, {}
    width = 8
    windows = []
    for time_index, env_index in event_indices.tolist():
        start = max(0, time_index - width + 1)
        windows.append((start, time_index + 1, env_index))
    obs_parts, action_parts, logp_parts, component_parts = [], [], [], []
    for start, stop, env_index in windows:
        padding = width - (stop - start)
        indices = [start] * padding + list(range(start, stop))
        obs_parts.append(trace["observation"][indices, env_index])
        action_parts.append(trace["action"][indices, env_index])
        logp_parts.append(trace["old_logp"][indices, env_index])
        component_parts.append(trace["reward_components"][indices, env_index])
    observations = torch.stack(obs_parts, dim=1)
    actions = torch.stack(action_parts, dim=1)
    old_logp = torch.stack(logp_parts, dim=1)
    components = torch.stack(component_parts, dim=1)
    run_index = trace["reward_names"].index("safe_periodic_flight")
    run_reward = components[:, :, run_index]
    completion_reward = torch.where(run_reward >= 1.0, run_reward, 0.0)
    precursor_reward = run_reward - completion_reward
    base_reward = components.sum(-1) - run_reward
    rewards = {
        "base": base_reward, "precursor": precursor_reward, "completion": completion_reward,
        "run_specific": run_reward, "total": components.sum(-1),
    }
    actor = runner.alg.actor
    named_params = list(actor.named_parameters())
    params = [parameter for _, parameter in named_params]
    vectors, rows = {}, []
    for component, reward in rewards.items():
        advantage = discounted(reward)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        flat_obs, flat_action, flat_logp = observations.flatten(0, 1), actions.flatten(0, 1), old_logp.flatten()
        actor(flat_obs, stochastic_output=True)
        new_logp = actor.get_output_log_prob(flat_action)
        loss = -(advantage.flatten().detach() * torch.exp(new_logp - flat_logp)).mean()
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        parts = [torch.zeros_like(p) if g is None else g for p, g in zip(params, gradients)]
        vector = torch.cat([part.reshape(-1) for part in parts]).detach()
        vectors[component] = vector
        offset = 0
        for (parameter_name, parameter), gradient in zip(named_params, parts):
            layer = (
                "first_hidden" if parameter_name.startswith("mlp.0")
                else "second_hidden" if parameter_name.startswith("mlp.2")
                else "third_hidden" if parameter_name.startswith("mlp.4")
                else "output_mean_head" if parameter_name.startswith("mlp.6")
                else "std_parameter" if "std_param" in parameter_name else parameter_name
            )
            rows.append({
                "batch": "completion_event_window", "component": component, "scope": "layer",
                "name": layer, "gradient_norm": float(torch.linalg.vector_norm(gradient)),
            })
            if parameter_name == "mlp.6.weight":
                for joint, joint_name in enumerate(trace["joint_names"]):
                    rows.append({
                        "batch": "completion_event_window", "component": component, "scope": "joint",
                        "name": joint_name, "gradient_norm": float(torch.linalg.vector_norm(gradient[joint])),
                    })
            offset += parameter.numel()
    summary = {
        "status": "PASS", "completion_event_samples": int(completion_mask.sum()),
        "event_windows": len(windows), "window_steps": width,
    }
    base_norm = torch.linalg.vector_norm(vectors["base"])
    total_norm = torch.linalg.vector_norm(vectors["total"])
    for name, vector in vectors.items():
        summary[name] = {
            "gradient_norm": float(torch.linalg.vector_norm(vector)),
            "cosine_to_base": cosine(vector, vectors["base"]),
            "cosine_to_total": cosine(vector, vectors["total"]),
            "ratio_to_base": float(torch.linalg.vector_norm(vector) / (base_norm + 1e-12)),
            "ratio_to_total": float(torch.linalg.vector_norm(vector) / (total_norm + 1e-12)),
        }
    density_rows = []
    for factor in (1, 2, 4, 8):
        combined = vectors["base"] + vectors["precursor"] + factor * vectors["completion"]
        completion_vector = factor * vectors["completion"]
        density_rows.append({
            "completion_replication_factor": factor,
            "completion_gradient_to_total": float(torch.linalg.vector_norm(completion_vector) / (torch.linalg.vector_norm(combined) + 1e-12)),
            "completion_direction_projection": cosine(combined, vectors["completion"]),
            "base_completion_cosine": cosine(vectors["base"], vectors["completion"]),
            "reaches_one_percent": bool(torch.linalg.vector_norm(completion_vector) / (torch.linalg.vector_norm(combined) + 1e-12) >= .01),
        })
    optimizer = runner.alg.optimizer
    restored_parts = []
    for parameter in params:
        state = optimizer.state.get(parameter, {})
        if "exp_avg" in state and "exp_avg_sq" in state:
            restored_parts.append(
                -state["exp_avg"] / (state["exp_avg_sq"].sqrt() + optimizer.param_groups[0]["eps"])
            )
        else:
            restored_parts.append(torch.zeros_like(parameter))
    restored = torch.cat([part.reshape(-1) for part in restored_parts]).detach()
    total_gradient = vectors["total"]
    zero_moment = -total_gradient / (total_gradient.abs() + optimizer.param_groups[0]["eps"])
    sgd = -total_gradient
    update_rows = []
    for name, update in (("restored_adam", restored), ("zero_moment_adam", zero_moment), ("raw_sgd", sgd)):
        update_rows.append({
            "update_direction": name, "step_norm": float(torch.linalg.vector_norm(update)),
            "cosine_to_completion_descent": cosine(update, -vectors["completion"]),
            "cosine_to_total_descent": cosine(update, -vectors["total"]),
            "cosine_to_base_descent": cosine(update, -vectors["base"]),
        })
    return summary, rows, vectors, {"density": density_rows, "updates": update_rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((STAGE2E / "checkpoint_manifest.json").read_text(encoding="utf-8"))["checkpoints"]
    manifest_by_iteration = {int(row["phase_a_iteration"]): row for row in manifest}
    checkpoint_rows = []
    for iteration in CHECKPOINTS:
        checkpoint = cp_path(iteration)
        row = manifest_by_iteration[iteration]
        if sha(checkpoint) != row["sha256"]:
            raise RuntimeError(f"CHECKPOINT_HASH_MISMATCH:{iteration}")
        checkpoint_rows.append({
            "iteration": iteration, "path": str(checkpoint.relative_to(REPO)),
            "sha256": row["sha256"], "std_hash": row["std_hash"],
            "std_mean": float(checkpoint_std(checkpoint).mean()),
            "optimizer_hash": row.get("optimizer_hash", "checkpoint_manifest_did_not_record_optimizer_hash"),
            "adam_step": row["adam_steps"][0],
        })
    dump("runtime_checkpoint_manifest.json", checkpoint_rows)
    config, agent_config = configure_runtime()
    import importlib.metadata
    with launch_simulation(config, args):
        raw = gym.make("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", cfg=config)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_config.clip_actions)
        agent_config = handle_deprecated_rsl_rl_cfg(agent_config, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_config.to_dict(), log_dir=None, device=agent_config.device)
        sweep_rows, event_rows, landing_rows, gate_rows = [], [], [], []
        for iteration in SWEEP_CHECKPOINTS:
            checkpoint = cp_path(iteration)
            for mode_name, multiplier in MODES:
                print(f"[Stage2F] checkpoint={iteration} mode={mode_name}", flush=True)
                records, events, landings, gates = evaluate_sweep(
                    runner, wrapped, checkpoint, iteration, mode_name, multiplier,
                    capture_gradient=(iteration == 50 and mode_name == "S100"),
                )
                sweep_rows.extend(records)
                event_rows.extend(events)
                landing_rows.extend(landings)
                gate_rows.extend(gates)
                append_csv("runtime_sweep_episode_rows.csv", records)
                append_csv("runtime_completion_action_events.csv", [
                    {key: value for key, value in row.items() if not isinstance(value, list)}
                    for row in events
                ])
                append_csv("runtime_landing_action_samples.csv", landings)
                append_csv("runtime_periodic_gate_rows.csv", gates)
                append_jsonl("completion_events.jsonl", events)
        print("[Stage2F] high-speed heading diagnostic", flush=True)
        heading_rows = evaluate_heading_pair(runner, wrapped, cp_path(50))
        append_csv("runtime_heading_rows.csv", heading_rows)
        print("[Stage2F] event-conditioned gradients", flush=True)
        gradient_summary, gradient_rows, _, gradient_extra = component_gradients(
            runner, cp_path(50), RAW / "iter50_s100_event_trace.pt"
        )
        write_csv("event_conditioned_gradients.csv", gradient_rows)
        write_csv("completion_density_gradient_scaling_runtime.csv", gradient_extra.get("density", []))
        write_csv("shadow_update_direction_comparison_runtime.csv", gradient_extra.get("updates", []))
        dump("runtime_summary.json", {
            "checkpoints": checkpoint_rows, "sweep_rows": len(sweep_rows),
            "completion_events": event_rows, "gradient_summary": gradient_summary,
            "density_scaling": gradient_extra.get("density", []),
            "adam_updates": gradient_extra.get("updates", []),
        })
        dump("runtime_complete.json", {
            "status": "PASS", "production_policy_updates": 0, "new_training_checkpoints": 0,
            "raw_artifacts_git_policy": "excluded",
        })
        raw.close()


if __name__ == "__main__":
    main()
