"""D17 read-only START source geometry and causal diagnostics.

This runner never writes a checkpoint and never updates a persistent policy.
All optimizer probes live in memory and all trajectory searches optimize action
sequences, not policy parameters.
"""
from __future__ import annotations

import argparse, copy, csv, hashlib, importlib.util, json, math, sys
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d17_start_source_and_causality_audit"
RAW = OUT / "raw"
D16 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d16_dedicated_start_specialist"
D6 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher"
DT = 0.02


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d16 = load_module("d16_for_d17", HERE.parent / "run_phase2_d16_train.py")
d15, d3, d6 = d16.d15, d16.d3, d16.d6
from g1_explicit_motion_mode.contract import MotionMode, minimum_jerk
from g1_omnidirectional.policy import FrozenGaitActor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ff(x):
    return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)


def select_pool(pool, indices):
    idx = torch.tensor(indices, dtype=torch.long)
    return {k: v[idx].clone() for k, v in pool["snapshot"].items()}


def support_features(world, n):
    force = world.sensor.data.net_forces_w_history[:n, -1, world.sf, :].norm(dim=-1)
    contact = force > 5.0
    return contact.float(), force


def physical_features(world, n):
    contact, _ = support_features(world, n)
    return torch.cat((
        world.robot.data.root_lin_vel_b[:n], world.robot.data.root_ang_vel_b[:n],
        world.robot.data.projected_gravity_b[:n], world.robot.data.joint_pos[:n],
        world.robot.data.joint_vel[:n], world.env.action_manager.prev_action[:n], contact,
    ), dim=1)


def component_slices():
    return {
        "base_velocity": [0, 3], "base_angular_velocity": [3, 6],
        "projected_gravity": [6, 9], "joint_position": [9, 46],
        "joint_velocity": [46, 83], "previous_action": [83, 120],
        "contact_support": [120, 122], "constructed_122d_physical_state": [0, 122],
    }


def restore_source(world, pool, picks):
    n = len(picks)
    idx = torch.tensor(picks, dtype=torch.long)
    snap = {}
    for key, value in pool["snapshot"].items():
        selected = value[idx]
        if n < world.env.num_envs:
            selected = torch.cat((selected, selected[-1:].expand(world.env.num_envs - n, *selected.shape[1:])), 0)
        snap[key] = selected.to(world.device)
    world.restore_snapshot(snap)
    ids = torch.arange(n, device=world.device)
    world.state.target_mode[ids] = int(MotionMode.WALK)
    world.state.previous_target_mode[ids] = int(MotionMode.STAND)
    world.state.time_since_mode_change_s[ids] = 0
    world.state.ramp_progress[ids] = 0
    world.state.physical_command[ids] = 0
    world.state.previous_physical_command[ids] = 0
    world.term.external_override[ids, :3] = 0
    world.term._update_command()
    world.env.sim.forward()


def set_command(world, step, ramp_s, n):
    denom = max(1, round(ramp_s / DT))
    progress = torch.full((world.env.num_envs,), min(1.0, step / denom), device=world.device)
    target = torch.zeros(world.env.num_envs, 3, device=world.device)
    target[:, 0] = 0.3
    physical = target * minimum_jerk(progress)[:, None]
    world.state.advance(physical, progress, 0 if step == 0 else DT)
    d6.set_command(world, physical)
    return target[:n]


def gate_for(obs, duration):
    t = obs[:, 139] * 3.0
    if duration == "FULL":
        return torch.ones_like(t)
    duration = float(duration)
    decay_start = 0.5
    width = max(DT, duration - decay_start)
    decay = 1 - minimum_jerk(((t - decay_start) / width).clamp(0, 1))
    return torch.where(t <= decay_start, torch.ones_like(t), torch.where(t < duration, decay, torch.zeros_like(t)))


def final_action(policy, obs, gate_duration, use_r40):
    base = policy.base_action(obs)
    if not use_r40:
        return base, torch.zeros_like(base)
    raw = policy.residual.net(obs)
    residual = policy.bound * torch.tanh(raw)
    delta = gate_for(obs, gate_duration)[:, None] * residual
    action = torch.where(delta == 0, base, (base + delta).clamp(-1, 1))
    return action, residual


def nearest_distance(features, basin, mean, std):
    q = (features - mean) / std
    b = (basin - mean) / std
    best = torch.full((len(q),), float("inf"), device=q.device)
    index = torch.zeros(len(q), dtype=torch.long, device=q.device)
    for start in range(0, len(b), 2048):
        d = torch.cdist(q, b[start:start + 2048])
        v, i = d.min(1)
        use = v < best
        best[use], index[use] = v[use], i[use] + start
    return best, index


def safety_step(world, n, done, extras, streaks):
    return d15.safety(world, n, done, extras, streaks)[:6]


def rollout(world, policy, source_pool, picks, basin, bmean, bstd, *, use_r40=True,
            ramp_s=0.5, previous="P0", gate_duration=1.5, horizon=75, hold=False,
            basin_current_actions=None):
    n = len(picks)
    restore_source(world, source_pool, picks)
    source_feat = physical_features(world, n)
    _, nearest = nearest_distance(source_feat, basin, bmean, bstd)
    nearest_prev = basin[nearest, 83:120]
    nearest_current = nearest_prev if basin_current_actions is None else basin_current_actions[nearest]
    if previous == "P1":
        world.env.action_manager._prev_action[:n] = 0
    elif previous == "P2":
        world.env.action_manager._prev_action[:n] = nearest_prev
    elif previous == "P3":
        world.env.action_manager._prev_action[:n] = nearest_current
    check_steps = {0, 1, 2, 4, 8, 12, 16, 20, 25, 32, 50, 74}
    streak = torch.zeros(n, dtype=torch.long, device=world.device)
    completed = torch.full((n,), -1, dtype=torch.long, device=world.device)
    safety_flags = [torch.zeros(n, dtype=torch.bool, device=world.device) for _ in range(6)]
    safety_streaks = [torch.zeros(n, dtype=torch.long, device=world.device) for _ in range(3)]
    traces, yaw_series, action_series, contact_series = [], [], [], []
    first_action = None
    for step in range(horizon):
        target = set_command(world, step, ramp_s, n)
        obs = world.obs()
        with torch.inference_mode():
            if hold:
                action = policy.hold.mean(obs); residual = torch.zeros_like(action)
            else:
                action, residual = final_action(policy, obs, gate_duration, use_r40)
        if first_action is None:
            first_action = action[:n].clone()
        _, _, done, extras = world.wrapped.step(action)
        flags = safety_step(world, n, done, extras, safety_streaks)
        for dst, src in zip(safety_flags, flags): dst |= src
        vel = world.robot.data.root_lin_vel_b[:n, :2]
        yaw = world.robot.data.root_ang_vel_b[:n, 2]
        good = ((vel - target[:, :2]).norm(dim=1) <= 0.12) & (yaw.abs() <= 0.10)
        streak = torch.where(good, streak + 1, torch.zeros_like(streak))
        new = (completed < 0) & (streak >= 25) & ((step - 24) < 75)
        completed[new] = step
        contact, _ = support_features(world, n)
        yaw_series.append(yaw.detach().cpu()); action_series.append(action[:n].detach().cpu()); contact_series.append(contact.detach().cpu())
        if step in check_steps:
            feat = physical_features(world, n)
            bd, _ = nearest_distance(feat, basin, bmean, bstd)
            traces.append({
                "step": step, "basin_distance_mean": ff(bd.mean()), "basin_distance_p95": ff(torch.quantile(bd, .95)),
                "forward_velocity_mean": ff(vel[:, 0].mean()), "yaw_mean": ff(yaw.mean()),
                "yaw_abs_p95": ff(torch.quantile(yaw.abs(), .95)), "residual_l2_mean": ff(residual[:n].norm(dim=1).mean()),
                "fall_rate": ff(safety_flags[0].float().mean()), "slip_rate": ff(safety_flags[1].float().mean()),
                "torque_saturation_rate": ff(safety_flags[4].float().mean()),
            })
    safe = ~(safety_flags[0] | safety_flags[1] | safety_flags[2] | safety_flags[3] | safety_flags[4] | safety_flags[5])
    acquired = (completed >= 0) & safe
    yaws = torch.stack(yaw_series)
    contacts = torch.stack(contact_series)
    actions = torch.stack(action_series)
    source_d, _ = nearest_distance(source_feat, basin, bmean, bstd)
    final_d, _ = nearest_distance(physical_features(world, n), basin, bmean, bstd)
    sign_changes = ((yaws[1:] * yaws[:-1]) < 0).sum(0)
    contact_change = (contacts[1:] != contacts[:-1]).any(2)
    yaw_spike = yaws[1:].abs() > (yaws[:-1].abs() + .1)
    contact_spike_fraction = ff((yaw_spike & contact_change).float().sum() / yaw_spike.float().sum().clamp_min(1))
    return {
        "episodes": n, "acquisition": ff(acquired.float().mean()), "yaw_acquisition": ff((yaws[-25:].abs().amax(0) <= .1).float().mean()),
        "fall": ff(safety_flags[0].float().mean()), "dangerous_slip": ff(safety_flags[1].float().mean()),
        "impact": ff(safety_flags[2].float().mean()), "velocity_saturation": ff(safety_flags[3].float().mean()),
        "torque_saturation": ff(safety_flags[4].float().mean()), "nan_inf": ff(safety_flags[5].float().mean()),
        "source_basin_distance_mean": ff(source_d.mean()), "minimum_basin_distance_mean": min(x["basin_distance_mean"] for x in traces),
        "final_basin_distance_mean": ff(final_d.mean()), "basin_reduction_fraction": ff((1 - final_d / source_d.clamp_min(1e-6)).mean()),
        "first_action_l2_from_previous": ff((first_action - world.env.action_manager.prev_action[:n]).norm(dim=1).mean()),
        "yaw_mean": ff(yaws.mean()), "yaw_abs_p95": ff(torch.quantile(yaws.abs(), .95)),
        "yaw_sign_change_mean": ff(sign_changes.float().mean()), "contact_conditioned_yaw_spike_fraction": contact_spike_fraction,
        "left_right_action_asymmetry": ff((actions[:, :, :6].mean((0, 1)) - actions[:, :, 6:12].mean((0, 1))).abs().mean()),
        "trace": traces,
    }


def collect_basin(world, d6_pool, policy):
    active = [i for i, ok in enumerate(d6_pool["w_move_acquired"]) if ok][:64]
    snap = {k: v[active].to(world.device) for k, v in d6_pool["snapshot"].items()}
    world.restore_snapshot(snap)
    gait = torch.zeros(world.env.num_envs, device=world.device)
    states, actions = [], []
    for step in range(160):
        physical = torch.zeros(world.env.num_envs, 3, device=world.device); physical[:, 0] = .3
        d6.set_command(world, physical)
        obs124 = world.env.observation_manager.compute()["policy"]
        with torch.inference_mode(): action = policy.base(obs124[:, :123], gait)
        world.wrapped.step(action)
        states.append(physical_features(world, 64).cpu()); actions.append(action[:64].cpu())
    return torch.cat(states), torch.cat(actions), active


def geometry(source, basin, basin_actions):
    mean, std = basin.mean(0), basin.std(0).clamp_min(1e-4)
    dist, nn = nearest_distance(source, basin, mean, std)
    rows = []
    slices = component_slices()
    for i in range(len(source)):
        row = {"source_index": i, "nearest_state_index": int(nn[i]), "nearest_state_distance": ff(dist[i])}
        for name, (a, b) in slices.items(): row[name + "_distance"] = ff(((source[i, a:b] - basin[nn[i], a:b]) / std[a:b]).norm())
        sa, ba = source[i, 83:120], basin_actions[nn[i]]
        row.update({"nearest_action_l2": ff((sa - ba).norm()), "nearest_action_cosine": ff(torch.nn.functional.cosine_similarity(sa[None], ba[None])),
                    "contact_mismatch": bool((source[i, 120:122] != basin[nn[i], 120:122]).any()),
                    "support_foot_mismatch": bool((source[i, 120:122].argmax() != basin[nn[i], 120:122].argmax()))})
        rows.append(row)
    return mean, std, rows


def temporary_probes(world, source_pool, picks, basin, mean, std, initial_policy, initial_critic):
    restore_source(world, source_pool, picks)
    set_command(world, 0, .5, len(picks))
    obs = world.obs()
    _, data, terms, _, _ = d16.collect(world, initial_policy, initial_critic, obs, [0], 20279401, steps=75)
    params = list(initial_policy.residual.parameters())
    windows = {"W0": (0, 10), "W1": (10, 25), "W2": (25, 50), "W3": (50, 75)}
    temporal = {}
    term_map = {"velocity": "velocity_tracking", "yaw": "yaw_tracking", "safety": "upright_safety", "regularization": "regularization"}
    for wn, (a, b) in windows.items():
        temporal[wn] = {}
        for out_name, term_name in term_map.items():
            ret = d16.returns(terms[term_name][a:b], data["done"][a:b])
            dist = initial_policy.residual.dist(data["obs"][a:b].flatten(0, 1))
            loss = -(dist.log_prob(data["raw"][a:b].flatten(0, 1)).sum(1) * ret.flatten().detach()).mean()
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            temporal[wn][out_name + "_gradient_norm"] = math.sqrt(sum(ff((x if x is not None else torch.zeros_like(p)).square().sum()) for x, p in zip(g, params)))
        temporal[wn]["yaw_abs_mean"] = ff(world.robot.data.root_ang_vel_b[:len(picks), 2].abs().mean())
    probes = {}
    groups = {"U_ALL": list(terms), "U_VEL": ["velocity_tracking"], "U_YAW": ["yaw_tracking"],
              "U_TRACK": ["velocity_tracking", "yaw_tracking", "upright_safety"],
              "U_NO_REG": [k for k in terms if k != "regularization"]}
    for name, selected in groups.items():
        clone = copy.deepcopy(initial_policy)
        opt = torch.optim.Adam(clone.residual.parameters(), lr=1.5e-5)
        reward = sum(terms[k] for k in selected)
        ret = d16.returns(reward, data["done"])
        dist = clone.residual.dist(data["obs"].flatten(0, 1))
        loss = -(dist.log_prob(data["raw"].flatten(0, 1)).sum(1) * ret.flatten().detach()).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(clone.residual.parameters(), 10); opt.step()
        probes[name] = rollout(world, clone, source_pool, picks, basin, mean, std, use_r40=True, horizon=25)
    return temporal, probes


def cem_search(world, source_pool, picks, policy, basin, mean, std, horizon, bound, seed):
    # A global five-step-piecewise sequence is deliberately tested first: it is
    # transferable by construction and cannot memorize recipe IDs.
    torch.manual_seed(seed)
    segments = max(5, horizon // 5)
    candidates, repeats = 16, 4
    search_picks = [picks[i % len(picks)] for i in range(candidates * repeats)]
    mu = torch.zeros(segments, 37, device=world.device); sigma = torch.full_like(mu, .20)
    history = []
    for iteration in range(8):
        seq = (mu[None] + sigma[None] * torch.randn(candidates, segments, 37, device=world.device)).clamp(-bound, bound)
        restore_source(world, source_pool, search_picks)
        scores = torch.zeros(candidates * repeats, device=world.device)
        bad = torch.zeros_like(scores, dtype=torch.bool)
        streaks = [torch.zeros_like(scores, dtype=torch.long) for _ in range(3)]
        for step in range(horizon):
            target = set_command(world, step, .5, len(search_picks)); obs = world.obs()
            with torch.inference_mode(): base = policy.base_action(obs)
            residual = seq.repeat_interleave(repeats, 0)[:, min(segments - 1, step * segments // horizon)]
            action = (base + residual).clamp(-1, 1)
            _, _, done, extras = world.wrapped.step(action)
            flags = safety_step(world, len(search_picks), done, extras, streaks); bad |= flags[0] | flags[1] | flags[2] | flags[4] | flags[5]
        feat = physical_features(world, len(search_picks)); bd, _ = nearest_distance(feat, basin, mean, std)
        vel = world.robot.data.root_lin_vel_b[:len(search_picks), :2]; yaw = world.robot.data.root_ang_vel_b[:len(search_picks), 2]
        scores = -bd - 20 * (vel[:, 0] - .3).abs() - 20 * yaw.abs() - 100 * bad.float()
        cs = scores.view(candidates, repeats).mean(1); elite = cs.topk(4).indices
        mu = seq[elite].mean(0); sigma = seq[elite].std(0).clamp(.02, .3)
        history.append({"iteration": iteration, "best_score": ff(cs.max()), "mean_score": ff(cs.mean())})
    # Apply the best transferable sequence to 32 distinct sources, then frozen W_MOVE.
    test = picks[:32]; restore_source(world, source_pool, test); streak = torch.zeros(32, dtype=torch.long, device=world.device); complete = torch.zeros(32, dtype=torch.bool, device=world.device)
    bad = torch.zeros(32, dtype=torch.bool, device=world.device); ss = [torch.zeros(32, dtype=torch.long, device=world.device) for _ in range(3)]
    source_d, _ = nearest_distance(physical_features(world, 32), basin, mean, std)
    for step in range(75):
        target = set_command(world, step, .5, 32); obs = world.obs()
        with torch.inference_mode(): base = policy.base_action(obs)
        residual = mu[min(segments - 1, step * segments // horizon)] if step < horizon else torch.zeros(37, device=world.device)
        action = (base + residual).clamp(-1, 1) if step < horizon else base
        _, _, done, extras = world.wrapped.step(action); flags = safety_step(world, 32, done, extras, ss); bad |= flags[0] | flags[1] | flags[2] | flags[4] | flags[5]
        vel = world.robot.data.root_lin_vel_b[:32, :2]; yaw = world.robot.data.root_ang_vel_b[:32, 2]
        good = ((vel - target[:, :2]).norm(dim=1) <= .12) & (yaw.abs() <= .10); streak = torch.where(good, streak + 1, torch.zeros_like(streak)); complete |= streak >= 25
        if step == 24: d25, _ = nearest_distance(physical_features(world, 32), basin, mean, std)
    final_d, _ = nearest_distance(physical_features(world, 32), basin, mean, std)
    safe = ~bad; success = safe & complete & (d25 <= .5 * source_d)
    return {"horizon_steps": horizon, "bound": bound, "snapshots": 32, "success_rate": ff(success.float().mean()),
            "safe_rate": ff(safe.float().mean()), "walk_acquisition": ff(complete.float().mean()),
            "basin_entry_50pct": ff((d25 <= .5 * source_d).float().mean()), "fall_or_safety_failure": ff(bad.float().mean()),
            "history": history, "sequence_l2": ff(mu.norm()), "sequence_max_abs": ff(mu.abs().max())}


def main():
    parser = argparse.ArgumentParser(); add_launcher_args(parser); args, hydra = setup_preset_cli(parser); sys.argv = [sys.argv[0], *hydra]
    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 64; cfg.seed = 20279401; cfg.episode_length_s = 20.; cfg.observations.policy.enable_corruption = False; cfg.events.base_external_force_torque = None; cfg.events.push_robot = None
    if args.device: cfg.sim.device = agent.device = args.device
    train = torch.load(D16 / "raw/train_start_snapshots.pt", map_location="cpu", weights_only=False)
    val = torch.load(D16 / "raw/validation_start_snapshots.pt", map_location="cpu", weights_only=False)
    cp0 = torch.load(D16 / "raw/checkpoints/model_000.pt", map_location="cpu", weights_only=False)
    cp40 = torch.load(D16 / "raw/checkpoints/model_040.pt", map_location="cpu", weights_only=False)
    d6pool = torch.load(D6 / "raw/snapshots/selected/train_batch_00.pt", map_location="cpu", weights_only=False)
    valid = [i for i, x in enumerate(train["valid"]) if x][:64]
    RAW.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        world = d16.StartWorld(wrapped, d3.load_resets(), train)
        p40 = d16.StartPolicy(cp40["residual_bound"]).to(world.device); p40.residual.load_state_dict(cp40["residual_state_dict"]); p40.eval()
        p0 = d16.StartPolicy(cp0["residual_bound"]).to(world.device); p0.residual.load_state_dict(cp0["residual_state_dict"]); p0.eval()
        p40.hold = d3.initialize("P0_STAND_PARENT", world.device)[0].eval(); p0.hold = p40.hold
        basin, basin_actions, basin_ids = collect_basin(world, d6pool, p40)
        restore_source(world, train, valid); source = physical_features(world, 64).detach()
        basin_device = basin.to(world.device)
        mean, std, geom = geometry(source, basin_device, basin_actions.to(world.device))
        diagnostic_basin = basin_device[::5]
        first = {
            "B0_DIRECT_WMOVE": rollout(world, p40, train, valid, diagnostic_basin, mean, std, use_r40=False),
            "R40": rollout(world, p40, train, valid, diagnostic_basin, mean, std),
            "H_SHOLD": rollout(world, p40, train, valid, diagnostic_basin, mean, std, hold=True),
        }
        ramps = {}
        for name, seconds in (("C0", .5), ("C1", DT), ("C2", .1), ("C3", .25), ("C4", 1.0)):
            ramps[name] = {"ramp_seconds": seconds, "B0": rollout(world, p40, train, valid, diagnostic_basin, mean, std, use_r40=False, ramp_s=seconds), "R40": rollout(world, p40, train, valid, diagnostic_basin, mean, std, ramp_s=seconds)}
        previous = {name: {"B0": rollout(world, p40, train, valid, diagnostic_basin, mean, std, use_r40=False, previous=name), "R40": rollout(world, p40, train, valid, diagnostic_basin, mean, std, previous=name)} for name in ("P0", "P1", "P2", "P3")}
        gates = {name: rollout(world, p40, train, valid, diagnostic_basin, mean, std, gate_duration=duration, horizon=150) for name, duration in (("G15", 1.5), ("G20", 2.0), ("G30", 3.0), ("GFULL", "FULL"))}
        critic = d16.Critic().to(world.device); critic.load_state_dict(cp0["critic_state_dict"])
        temporal, probes = temporary_probes(world, train, valid, diagnostic_basin, mean, std, p0, critic)
        searches = [cem_search(world, train, valid[:32], p40, diagnostic_basin, mean, std, 25, .5, 20279411)]
        if searches[-1]["success_rate"] < .8: searches.append(cem_search(world, train, valid[:32], p40, diagnostic_basin, mean, std, 50, .5, 20279412))
        if searches[-1]["success_rate"] < .8: searches.append(cem_search(world, train, valid[:32], p40, diagnostic_basin, mean, std, 25, .75, 20279413))
        # One-time validation confirmation of the selected diagnostic result (R40/C0).
        val_valid = [i for i, x in enumerate(val["valid"]) if x]
        validation = []
        for start in range(0, len(val_valid), 64): validation.append(rollout(world, p40, val, val_valid[start:start+64], diagnostic_basin, mean, std))
        out = {
            "identities": {"B0_sha256": d16.WMOVE_SHA, "R0_sha256": sha(D16 / "raw/checkpoints/model_000.pt"), "R40_path": str((D16 / "raw/checkpoints/model_040.pt").relative_to(REPO)).replace("\\", "/"), "R40_sha256": sha(D16 / "raw/checkpoints/model_040.pt"), "H_sha256": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"},
            "source_indices": valid, "source_recipe_ids": [train["recipes"][i] for i in valid], "source_features": source.cpu().tolist(),
            "basin": {"states": len(basin), "source_artifact": str((D6 / "raw/snapshots/selected/train_batch_00.pt").relative_to(REPO)).replace("\\", "/"), "source_indices": basin_ids, "feature_dimension": basin.shape[1]},
            "geometry": geom, "first_divergence": first, "command_ramps": ramps, "previous_action": previous, "gate_duration": gates,
            "temporal_gradient": temporal, "temporary_probes": probes, "reachability": searches, "validation_confirmation": validation,
            "persistent_policy_updates": 0, "new_checkpoint": 0,
        }
        dump(RAW / "audit_results.json", out)
        print(json.dumps({"basin_states": len(basin), "R40_acquisition": first["R40"]["acquisition"], "searches": [{"h":x["horizon_steps"],"b":x["bound"],"success":x["success_rate"]} for x in searches]}, indent=2), flush=True)
        wrapped.close()


if __name__ == "__main__": main()
