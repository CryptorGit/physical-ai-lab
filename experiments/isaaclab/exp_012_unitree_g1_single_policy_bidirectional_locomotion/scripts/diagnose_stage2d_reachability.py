"""Stage 2D frozen-policy RUN reward reachability telemetry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2d_run_reward_reachability_preflight"
RETRY = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_retry1"
STAGE2C = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2c_multi_regime_gradient_interference"
PARENT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
STAGE3 = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_23-07-40_stage3_1024_500/model_4745.pt"
STAGE4 = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
SEED = 20265021
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

FIELDS = (
    "requested_vx", "actual_speed", "tilt", "vertical_speed", "contacts",
    "reward_raw", "active", "segment", "observation", "action", "log_prob",
)


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    rows = list(rows)
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ("status",), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows or [{"status": "NO_ROWS"}])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def minimum_jerk(t):
    t = torch.clamp(t, 0.0, 1.0)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def profile(kind, target, step, dt, device, count):
    t = torch.full((count,), step * dt, device=device)
    if kind == "direct":
        return torch.full_like(t, target), torch.full((count,), 4, device=device, dtype=torch.long)
    if kind == "ramped":
        v = torch.zeros_like(t)
        seg = torch.zeros((count,), device=device, dtype=torch.long)
        m = (t >= 1.0) & (t < 2.0)
        v[m], seg[m] = 1.2 * minimum_jerk((t[m] - 1.0) / 1.0), 1
        m = (t >= 2.0) & (t < 3.5)
        v[m], seg[m] = 1.2, 2
        m = (t >= 3.5) & (t < 5.0)
        v[m], seg[m] = 1.2 + (target - 1.2) * minimum_jerk((t[m] - 3.5) / 1.5), 3
        m = t >= 5.0
        v[m], seg[m] = target, 4
        return v, seg
    # 18 s full bidirectional profile: 0 -> .6 -> 1.2 -> target -> 1.2 -> .6 -> 0.
    schedule = (
        (0.0, 1.5, "hold"), (0.6, 1.0, "ramp"), (0.6, 1.0, "hold"),
        (1.2, 1.0, "ramp"), (1.2, 1.5, "hold"), (target, 1.5, "ramp"),
        (target, 3.0, "hold"), (1.2, 1.5, "ramp"), (1.2, 1.5, "hold"),
        (0.6, 1.0, "ramp"), (0.6, 1.0, "hold"), (0.0, 1.0, "ramp"),
        (0.0, 2.0, "hold"),
    )
    v = torch.zeros_like(t)
    seg = torch.zeros((count,), device=device, dtype=torch.long)
    cursor, previous = 0.0, 0.0
    for i, (value, duration, mode) in enumerate(schedule):
        m = (t >= cursor) & (t < cursor + duration)
        if mode == "ramp":
            v[m] = previous + (value - previous) * minimum_jerk((t[m] - cursor) / duration)
        else:
            v[m] = value
        seg[m] = i
        previous, cursor = value, cursor + duration
    return v, seg


def first_failure(gates):
    order = (
        "requested_vx_gate", "precursor_speed_gate", "tilt_gate", "vertical_speed_gate",
        "liftoff_detected", "minimum_flight_gate", "maximum_flight_gate", "landing_detected",
        "single_foot_gate", "previous_landing_valid", "alternating_landing_gate",
        "completion_speed_gate",
    )
    return next((x for x in order if not gates[x]), "COMPLETION_FIRE")


def event_row(meta, env_id, step, dt, command, speed, tilt, vz, duration, contacts, last_foot):
    contact_count = int(contacts.sum())
    landing_foot = int(torch.argmax(contacts.to(torch.int64))) if contact_count else -1
    speed_error = abs(speed - command)
    single = contact_count == 1
    previous_valid = last_foot >= 0
    alternating = single and previous_valid and landing_foot != last_foot
    gates = {
        "requested_vx_gate": command >= 2.3,
        "precursor_speed_gate": speed_error <= 1.20,
        "tilt_gate": tilt <= 0.20,
        "vertical_speed_gate": vz <= 0.50,
        "liftoff_detected": True,
        "minimum_flight_gate": duration >= 0.04 - 1e-6,
        "maximum_flight_gate": duration <= 0.16 + 1e-6,
        "landing_detected": True,
        "single_foot_gate": single,
        "previous_landing_valid": previous_valid,
        "alternating_landing_gate": alternating,
        "completion_speed_gate": speed_error <= 0.30,
    }
    official = all(gates[x] for x in (
        "requested_vx_gate", "tilt_gate", "vertical_speed_gate", "minimum_flight_gate",
        "maximum_flight_gate", "single_foot_gate", "previous_landing_valid",
        "alternating_landing_gate", "completion_speed_gate"))
    margins = {
        "speed_margin": (0.30 - speed_error) / 0.30,
        "tilt_margin": (0.20 - tilt) / 0.20,
        "vertical_speed_margin": (0.50 - vz) / 0.50,
        "flight_min_margin": (duration - 0.04) / 0.04,
        "flight_max_margin": (0.16 - duration) / 0.16,
        "single_foot_margin": 1.0 if single else -1.0,
        "alternation_margin": 1.0 if alternating else -1.0,
    }
    return {
        **meta, "environment": env_id, "step": step, "episode_time_s": step * dt,
        "requested_vx": command, "actual_speed": speed, "speed_error": speed_error,
        "tilt": tilt, "vertical_speed": vz, "flight_duration": duration,
        "contact_count": contact_count, "landing_foot": landing_foot,
        "previous_landing_foot": last_foot, "support_foot": ("left" if landing_foot == 0 else "right" if landing_foot == 1 else "double"),
        **{k: int(v) for k, v in gates.items()}, "completion_fire": int(official),
        "first_failed_gate": first_failure(gates), **margins,
        "failed_gate_count": sum(not gates[x] for x in (
            "requested_vx_gate", "tilt_gate", "vertical_speed_gate", "minimum_flight_gate",
            "maximum_flight_gate", "single_foot_gate", "previous_landing_valid",
            "alternating_landing_gate", "completion_speed_gate")),
    }


def run_condition(runner, wrapped, command_term, reward_term, checkpoint, cp_label, kind, target, stochastic=False):
    runner.load(str(checkpoint), load_cfg={
        "actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False,
    }, strict=True, map_location=runner.device)
    wrapped.reset()
    command_term.external_override_enabled = True
    command_term.external_override.zero_()
    obs = wrapped.get_observations().to(runner.device)
    n = wrapped.num_envs
    dt = float(wrapped.unwrapped.step_dt)
    steps = 900 if kind == "bidirectional" else 500
    active = torch.ones(n, dtype=torch.bool, device=runner.device)
    was_flight = torch.zeros(n, dtype=torch.bool, device=runner.device)
    duration = torch.zeros(n, device=runner.device)
    last_foot = torch.full((n,), -1, dtype=torch.long, device=runner.device)
    events, traces = [], {k: [] for k in FIELDS}
    precursor_steps = safe_steps = completion = excess = 0
    run_target_samples = 0
    target_streak = torch.zeros(n, dtype=torch.long, device=runner.device)
    max_target_streak = torch.zeros_like(target_streak)
    for step in range(steps):
        vx, seg = profile(kind, target, step, dt, runner.device, n)
        command_term.external_override[:, 0] = vx
        command_term.external_override[:, 1:] = 0.0
        if stochastic:
            actions = runner.alg.actor(obs, stochastic_output=True)
            logp = runner.alg.actor.get_output_log_prob(actions)
        else:
            with torch.inference_mode():
                actions = runner.alg.actor(obs, stochastic_output=False)
                runner.alg.actor(obs, stochastic_output=True)
                logp = runner.alg.actor.get_output_log_prob(actions)
        old_obs, old_actions, old_logp = obs.detach().clone(), actions.detach().clone(), logp.detach().clone()
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(actions.to(wrapped.unwrapped.device))
            obs, dones = obs.to(runner.device), dones.to(runner.device)
        robot = wrapped.unwrapped.scene["robot"]
        sensor = wrapped.unwrapped.scene.sensors["contact_forces"]
        forces = sensor.data.net_forces_w_history[:, :, reward_term.foot_ids, :]
        contacts = forces.norm(dim=-1).amax(dim=1) > 1.0
        in_flight = contacts.sum(dim=1) == 0
        landing = was_flight & ~in_flight & active
        duration[in_flight & active] += dt
        speed = robot.data.root_lin_vel_b[:, 0]
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        vz = robot.data.root_lin_vel_b[:, 2].abs()
        raw = reward_term.last_raw_reward.detach().clone()
        run_target = (vx >= 2.3) & active
        run_target_samples += int(run_target.sum())
        target_streak = torch.where(run_target, target_streak + 1, torch.zeros_like(target_streak))
        max_target_streak = torch.maximum(max_target_streak, target_streak)
        for env_id in torch.where(landing)[0].tolist():
            row = event_row(
                {"checkpoint": cp_label, "condition": f"{kind}_{target:.1f}", "profile": kind,
                 "target": target, "segment": int(seg[env_id])},
                env_id, step, dt, float(vx[env_id]), float(speed[env_id]), float(tilt[env_id]),
                float(vz[env_id]), float(duration[env_id]), contacts[env_id], int(last_foot[env_id]),
            )
            events.append(row)
            completion += row["completion_fire"]
        single_landing = landing & (contacts.sum(dim=1) == 1)
        landing_foot = contacts.to(torch.int64).argmax(dim=1)
        last_foot[single_landing] = landing_foot[single_landing]
        duration[landing] = 0.0
        was_flight.copy_(in_flight)
        # Raw values uniquely identify shaping/penalty in this frozen config.
        precursor_steps += int(((raw > 0) & in_flight & (duration < 0.04) & active).sum())
        safe_steps += int(((raw > 0) & in_flight & (duration >= 0.04) & active).sum())
        excess += int(((raw < 0) & active).sum())
        for key, value in (
            ("requested_vx", vx), ("actual_speed", speed), ("tilt", tilt),
            ("vertical_speed", vz), ("contacts", contacts), ("reward_raw", raw),
            ("active", active), ("segment", seg), ("observation", old_obs),
            ("action", old_actions), ("log_prob", old_logp),
        ):
            traces[key].append(value.detach().cpu())
        active &= ~dones.bool()
    traces = {k: torch.stack(v) for k, v in traces.items()}
    summary = {
        "checkpoint": cp_label, "checkpoint_sha256": sha(checkpoint),
        "condition": f"{kind}_{target:.1f}", "profile": kind, "target": target,
        "nominal_episodes": n, "steps": steps, "duration_s": steps * dt,
        "fall_rate": float((~active).float().mean()), "active_final": int(active.sum()),
        "run_target_samples": run_target_samples,
        "run_target_env_0p5s": int((max_target_streak >= 25).sum()),
        "run_target_env_1s": int((max_target_streak >= 50).sum()),
        "run_target_env_2s": int((max_target_streak >= 100).sum()),
        "max_run_target_duration_s_mean": float(max_target_streak.float().mean() * dt),
        "landing_candidates": len(events), "completion_count": completion,
        "takeoff_precursor_steps": precursor_steps, "safe_flight_reward_steps": safe_steps,
        "excess_penalty_steps": excess, "stochastic": stochastic,
    }
    return summary, events, traces


def replay_machine(trace, chunked):
    contacts = trace["contacts"]
    command, speed = trace["requested_vx"], trace["actual_speed"]
    tilt, vz = trace["tilt"], trace["vertical_speed"]
    steps, n = command.shape
    was = torch.zeros(n, dtype=torch.bool)
    duration = torch.zeros(n)
    precursor = torch.zeros(n)
    last = torch.full((n,), -1, dtype=torch.long)
    reward = torch.zeros((steps, n))
    state_hashes = []
    boundaries = set(range(24, steps, 24)) if chunked else set()
    for step in range(steps):
        # A chunk boundary is an orchestration boundary only: state is retained.
        _ = step in boundaries
        count = contacts[step].sum(-1)
        flight = count == 0
        started = flight & ~was
        precursor[started] = 0
        duration[flight] += 0.02
        common = flight & (command[step] >= 2.3) & (tilt[step] <= .2) & (vz[step] <= .5) & (duration <= .160001)
        safe_takeoff = common & ((speed[step] - command[step]).abs() <= .3) & (duration < .039999)
        sustained = common & ((speed[step] - command[step]).abs() <= 1.2) & (duration >= .039999)
        request = torch.where(sustained, .25, torch.where(safe_takeoff, .05, 0.0))
        shaped = torch.minimum(request, torch.clamp(.75 - precursor, min=0))
        precursor += shaped
        landing = was & ~flight
        foot = contacts[step].to(torch.int64).argmax(-1)
        single = landing & (count == 1)
        alt = single & (last >= 0) & (foot != last)
        completion = alt & (command[step] >= 2.3) & ((speed[step] - command[step]).abs() <= .3)
        completion &= (tilt[step] <= .2) & (vz[step] <= .5) & (duration >= .039999) & (duration <= .160001)
        excess = flight & (duration > .160001)
        reward[step] = shaped + completion.float() * 2.0 - excess.float() * .25
        last[single] = foot[single]
        duration[landing] = 0
        was.copy_(flight)
        if step in boundaries:
            h = hashlib.sha256()
            for value in (was, duration, precursor, last):
                h.update(value.contiguous().numpy().tobytes())
            state_hashes.append({"step": step, "state_hash": h.hexdigest(), "reward_count": int((reward[:step + 1] != 0).sum())})
    return reward, state_hashes


def discounted_returns(reward, gamma=.99):
    out = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[1], device=reward.device)
    for t in range(reward.shape[0] - 1, -1, -1):
        running = reward[t] + gamma * running
        out[t] = running
    return out


def gradient_components(runner, checkpoint, raw_path, reward_names):
    runner.load(str(checkpoint), load_cfg={
        "actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False,
    }, strict=True, map_location=runner.device)
    data = torch.load(raw_path, map_location=runner.device, weights_only=False)
    cohort = data["cohort"]
    idx = torch.where(cohort == 2)[0]
    obs = data["obs"][idx]
    actions, old_logp = data["actions"][idx], data["old_logp"][idx]
    components = data["reward_components"][idx]
    # Stored rollout is time-major before flattening: select RUN env indices per timestep.
    t, e = 24, 256
    components = components.reshape(t, e, -1)
    run_index = reward_names.index("safe_periodic_flight")
    run_reward = components[:, :, run_index]
    total_reward = components.sum(-1)
    base_reward = total_reward - run_reward
    completion_reward = torch.zeros_like(run_reward)
    precursor_reward = run_reward.clone()
    rewards = {
        "base": base_reward, "precursor": precursor_reward,
        "completion": completion_reward, "run_specific": run_reward, "total": total_reward,
    }
    actor_params = list(runner.alg.actor.parameters())
    vectors, rows = {}, []
    for name, reward in rewards.items():
        adv = discounted_returns(reward).reshape(-1)
        adv = adv - adv.mean()
        runner.alg.actor(obs, stochastic_output=True)
        logp = runner.alg.actor.get_output_log_prob(actions)
        loss = -(adv.detach() * torch.exp(logp - old_logp)).mean()
        grads = torch.autograd.grad(loss, actor_params, allow_unused=True)
        vec = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(actor_params, grads)]).detach()
        vectors[name] = vec
    total_norm = float(torch.linalg.vector_norm(vectors["total"]))
    base_norm = float(torch.linalg.vector_norm(vectors["base"]))
    for name, vec in vectors.items():
        norm = float(torch.linalg.vector_norm(vec))
        rows.append({
            "component": name, "gradient_norm": norm,
            "ratio_to_total": norm / (total_norm + 1e-12),
            "ratio_to_base": norm / (base_norm + 1e-12),
            "cosine_to_total": float(torch.dot(vec, vectors["total"]) / (torch.linalg.vector_norm(vec) * torch.linalg.vector_norm(vectors["total"]) + 1e-12)),
            "cosine_to_base": float(torch.dot(vec, vectors["base"]) / (torch.linalg.vector_norm(vec) * torch.linalg.vector_norm(vectors["base"]) + 1e-12)),
        })
    return rows, {k: v.detach().cpu() for k, v in vectors.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RETRY / "checkpoint_manifest.json").read_text(encoding="utf-8"))["checkpoints"]
    cp_map = {x["iteration"]: REPO / x["path"] for x in manifest}
    selected = cp_map[100]
    checks = {
        str(PARENT): "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
        str(STAGE3): "8143434c5dbb68f68414f5705dd0f40db1045c63b3e201a4e8a4c2a31e81c22e",
        str(STAGE4): "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
        str(selected): "8d8afac60cafbd4adf0b98469fab01f711f32771a40899653d962cc08a5d8143",
    }
    for path, expected in checks.items():
        if sha(path) != expected:
            raise RuntimeError(f"CHECKPOINT_HASH_MISMATCH:{path}")
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 50
    cfg.seed = SEED
    agent_cfg.seed = SEED
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        command_term = wrapped.unwrapped.command_manager.get_term("base_velocity")
        reward_names = list(wrapped.unwrapped.reward_manager.active_terms)
        reward_index = reward_names.index("safe_periodic_flight")
        reward_term = wrapped.unwrapped.reward_manager.get_term_cfg("safe_periodic_flight").func
        all_summaries, all_events, raw_traces = [], [], {}
        candidates = (("parent", PARENT), ("iter50", cp_map[50]), ("iter100", selected), ("iter300", cp_map[300]))
        conditions = (("direct", 2.4), ("direct", 2.6), ("ramped", 2.4), ("ramped", 2.6), ("bidirectional", 2.4))
        for label, checkpoint in candidates:
            for kind, target in conditions:
                print(f"[Stage2D] {label} {kind} {target}", flush=True)
                summary, events, trace = run_condition(
                    runner, wrapped, command_term, reward_term, checkpoint, label, kind, target)
                all_summaries.append(summary)
                all_events += events
                if label == "iter100" and kind == "direct":
                    raw_traces[f"{kind}_{target:.1f}"] = trace
        positive = []
        positive_traces = {}
        for label, checkpoint in (("exp005_stage3", STAGE3), ("exp005_stage4", STAGE4)):
            for target in (2.4, 2.6):
                print(f"[Stage2D] positive {label} {target}", flush=True)
                summary, events, trace = run_condition(
                    runner, wrapped, command_term, reward_term, checkpoint, label, "direct", target)
                positive.append(summary)
                all_events += events
                positive_traces[f"{label}_{target:.1f}"] = trace
        # Positive-event gradient uses a stochastic positive-control rollout.
        pos_grad_summary, pos_grad_events, pos_grad_trace = run_condition(
            runner, wrapped, command_term, reward_term, STAGE4, "exp005_stage4_gradient", "direct", 2.6, stochastic=True)
        all_events += pos_grad_events
        # Boundary equivalence is an exact offline replay of the same runtime physical trace.
        boundary = {}
        for key, trace in raw_traces.items():
            continuous, _ = replay_machine(trace, False)
            chunked, states = replay_machine(trace, True)
            boundary[key] = {
                "reward_trace_equal": bool(torch.equal(continuous, chunked)),
                "max_abs_difference": float((continuous - chunked).abs().max()),
                "continuous_fire_count": int((continuous != 0).sum()),
                "chunked_fire_count": int((chunked != 0).sum()),
                "chunk_boundaries": states,
            }
        raw_dir = OUT / "raw_traces"
        raw_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "iter100": raw_traces, "positive_controls": positive_traces,
            "positive_gradient": pos_grad_trace, "reward_names": reward_names,
        }, raw_dir / "long_horizon_traces.pt")
        gradient_rows, gradient_vectors = gradient_components(
            runner, selected, STAGE2C / "raw_rollouts/rollout_100.pt", reward_names)
        torch.save(gradient_vectors, raw_dir / "run_reward_component_gradients.pt")
        # Positive-event gradient on the stochastic Stage-4 trace.
        # Use raw run reward as a diagnostic score-function return.
        runner.load(str(STAGE4), load_cfg={
            "actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False,
        }, strict=True, map_location=runner.device)
        tr = pos_grad_trace
        obs = tr["observation"].flatten(0, 1).to(runner.device)
        actions = tr["action"].flatten(0, 1).to(runner.device)
        old_logp = tr["log_prob"].flatten().to(runner.device)
        raw_reward = tr["reward_raw"].to(runner.device)
        completion_mask = raw_reward >= 1.0
        precursor_reward = torch.where((raw_reward > 0) & ~completion_mask, raw_reward, 0.0)
        completion_reward = torch.where(completion_mask, raw_reward, 0.0)
        positive_gradient = {}
        params = list(runner.alg.actor.parameters())
        for name, rew in (("precursor", precursor_reward), ("completion", completion_reward)):
            adv = discounted_returns(rew).flatten()
            adv -= adv.mean()
            runner.alg.actor(obs, stochastic_output=True)
            logp = runner.alg.actor.get_output_log_prob(actions)
            loss = -(adv * torch.exp(logp - old_logp)).mean()
            gs = torch.autograd.grad(loss, params, allow_unused=True)
            vec = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, gs)])
            positive_gradient[name] = float(torch.linalg.vector_norm(vec))
        positive_gradient.update({
            "completion_event_count": int(completion_mask.sum()),
            "precursor_event_count": int((precursor_reward > 0).sum()),
            "positive_control": pos_grad_summary,
        })
        # Runtime contact mapping.
        sensor = wrapped.unwrapped.scene.sensors["contact_forces"]
        mapping = {
            "reward_foot_ids": [int(x) for x in reward_term.foot_ids],
            "sensor_body_names": list(sensor.body_names),
            "resolved_reward_foot_names": [sensor.body_names[int(i)] for i in reward_term.foot_ids],
            "contact_force_threshold_n": 1.0,
            "history_reduction": "norm then amax over history",
            "ordering": "index 0=left, index 1=right" if "left" in sensor.body_names[int(reward_term.foot_ids[0])].lower() else "MAPPING_REVIEW_REQUIRED",
        }
        emit = {
            "summaries": all_summaries, "events": all_events, "positive": positive,
            "boundary": boundary, "mapping": mapping, "gradient_rows": gradient_rows,
            "positive_gradient": positive_gradient, "reward_names": reward_names,
            "positive_gradient_summary": pos_grad_summary,
        }
        dump("runtime_diagnostic_summary.json", emit)
        write_csv("run_event_gate_cascade.csv", all_events)
        write_csv("run_reward_component_gradients.csv", gradient_rows)
        print("[Stage2D] collection complete", flush=True)
        wrapped.close()


if __name__ == "__main__":
    main()
