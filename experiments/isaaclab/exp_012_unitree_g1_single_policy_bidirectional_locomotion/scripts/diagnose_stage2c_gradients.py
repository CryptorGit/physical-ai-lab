"""Stage 2C: frozen-checkpoint multi-regime gradient interference diagnosis."""

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
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2c_multi_regime_gradient_interference"
RETRY = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_retry1"
MANIFEST = RETRY / "checkpoint_manifest.json"
TIMELINE = RETRY / "capability_training_timeline.csv"
COHORTS = ("ZERO_HOLD", "WALK_STEADY", "RUN_HOLD", "BIDIRECTIONAL_SEQUENCE")
WEIGHTS = torch.tensor((0.2, 0.2, 0.2, 0.4))
SEED = 20264021
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--warmup-steps", type=int, default=360)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]
ZERO_HOLD, WALK_STEADY, RUN_HOLD, BIDIRECTIONAL_SEQUENCE = range(4)


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows, fields=None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else ("status",)
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows or [{"status": "NO_ROWS"}])


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(*xs):
    h = hashlib.sha256()
    for x in xs:
        a = x.detach().contiguous().cpu().numpy()
        if isinstance(a, dict):
            for key in sorted(a):
                h.update(str(key).encode())
                value = np.asarray(a[key])
                h.update(str(value.dtype).encode())
                h.update(np.asarray(value.shape, dtype=np.int64).tobytes())
                h.update(value.tobytes())
        else:
            h.update(str(a.dtype).encode())
            h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
            h.update(a.tobytes())
    return h.hexdigest()


def qstats(x):
    x = x.detach().float()
    q = torch.quantile(x, torch.tensor((.01, .05, .5, .95, .99), device=x.device))
    return {
        "mean": float(x.mean()), "std": float(x.std(unbiased=False)),
        "positive_rate": float((x > 0).float().mean()), "negative_rate": float((x < 0).float().mean()),
        "p1": float(q[0]), "p5": float(q[1]), "p50": float(q[2]),
        "p95": float(q[3]), "p99": float(q[4]),
        "absolute_p95": float(torch.quantile(x.abs(), .95)),
        "absolute_p99": float(torch.quantile(x.abs(), .99)),
    }


def flat_grad(loss, params, retain=False):
    grads = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grads)
    ])


def cosine(a, b):
    den = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return float(torch.dot(a, b) / den) if den > 0 else 0.0


def pair_metrics(a, b):
    dot = float(torch.dot(a, b))
    na, nb = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b))
    nz = (a != 0) | (b != 0)
    signs = torch.sign(a[nz]) == torch.sign(b[nz])
    neg = (a[nz] * b[nz]) < 0
    return {
        "dot_product": dot, "cosine": cosine(a, b),
        "gradient_norm_ratio": na / (nb + 1e-12),
        "coordinate_sign_agreement": float(signs.float().mean()) if nz.any() else 1.0,
        "negative_coordinate_fraction": float(neg.float().mean()) if nz.any() else 0.0,
    }


def distribution(alg, obs, actions):
    alg.actor(obs, stochastic_output=True)
    logp = alg.actor.get_output_log_prob(actions)
    entropy = alg.actor.output_entropy
    value = alg.critic(obs).squeeze(-1)
    return logp, entropy, value


def losses(alg, data, idx, advantage):
    logp, entropy, value = distribution(alg, data["obs"][idx], data["actions"][idx])
    ratio = torch.exp(logp - data["old_logp"][idx])
    unclipped = -advantage * ratio
    clipped = -advantage * ratio.clamp(1.0 - alg.clip_param, 1.0 + alg.clip_param)
    actor = torch.maximum(unclipped, clipped).mean()
    old_v, returns = data["values"][idx], data["returns"][idx]
    vclip = old_v + (value - old_v).clamp(-alg.clip_param, alg.clip_param)
    critic = torch.maximum((value - returns).square(), (vclip - returns).square()).mean()
    return actor, entropy.mean(), critic


def layer_slices(actor):
    labels = []
    offset = 0
    for name, p in actor.named_parameters():
        if name.startswith("mlp.0"):
            layer = "first_hidden"
        elif name.startswith("mlp.2"):
            layer = "second_hidden"
        elif name.startswith("mlp.4"):
            layer = "third_hidden"
        elif name.startswith("mlp.6"):
            layer = "output_mean_head"
        elif "std" in name:
            layer = "std_parameter"
        else:
            layer = "other"
        labels.append((name, layer, offset, offset + p.numel(), tuple(p.shape)))
        offset += p.numel()
    return labels


def set_exact_cohorts(term):
    ids = torch.arange(term.num_envs, device=term.device)
    term.cohort[:256] = ZERO_HOLD
    term.cohort[256:512] = WALK_STEADY
    term.cohort[512:768] = RUN_HOLD
    term.cohort[768:] = BIDIRECTIONAL_SEQUENCE
    term._resample_command(ids)


def collect(runner, wrapped, command_term, reward_index):
    runner.alg.storage.clear()
    obs = wrapped.get_observations().to(runner.device)
    reward_events = defaultdict(float)
    gait_counts = defaultdict(int)
    cohort_steps = []
    sequence_steps = []
    command_steps = []
    reward_component_steps = []
    falls = []
    contacts = []
    for _ in range(runner.cfg["num_steps_per_env"]):
        with torch.inference_mode():
            actions = runner.alg.act(obs)
            obs, rewards, dones, extras = wrapped.step(actions.to(wrapped.unwrapped.device))
            obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
            runner.alg.process_env_step(obs, rewards, dones, extras)
        cohort_steps.append(command_term.cohort.detach().clone())
        sequence_steps.append(command_term.segment_index.detach().clone())
        command_steps.append(command_term.vel_command_b.detach().clone())
        reward_component_steps.append(wrapped.unwrapped.reward_manager._step_reward.detach().clone())
        falls.append(dones.detach().clone())
        if reward_index is not None:
            rr = wrapped.unwrapped.reward_manager._step_reward[:, reward_index].detach()
            reward_events["nonzero"] += float((rr != 0).sum())
            reward_events["positive"] += float((rr > 0).sum())
            reward_events["negative"] += float((rr < 0).sum())
            reward_events["sum"] += float(rr.sum())
        sensor = wrapped.unwrapped.scene.sensors.get("contact_forces")
        if sensor is not None:
            force = sensor.data.net_forces_w_history[:, -1].norm(dim=-1)
            contacts.append((force > 5).detach().clone())
        vx = command_term.vel_command_b[:, 0]
        run = vx >= 2.3
        if run.any():
            gait_counts["run_command_samples"] += int(run.sum())
            gait_counts["run_fall_samples"] += int((dones & run).sum())
    with torch.inference_mode():
        runner.alg.compute_returns(obs)
    s = runner.alg.storage
    old_mean, old_std = s.distribution_params
    data = {
        "obs": s.observations.flatten(0, 1).detach().clone(),
        "actions": s.actions.flatten(0, 1).detach().clone(),
        "old_mean": old_mean.flatten(0, 1).detach().clone(),
        "old_std": old_std.flatten(0, 1).detach().clone(),
        "old_logp": s.actions_log_prob.flatten(0, 1).squeeze(-1).detach().clone(),
        "values": s.values.flatten(0, 1).squeeze(-1).detach().clone(),
        "returns": s.returns.flatten(0, 1).squeeze(-1).detach().clone(),
        "raw_adv": (s.returns - s.values).flatten(0, 1).squeeze(-1).detach().clone(),
        "stored_adv": s.advantages.flatten(0, 1).squeeze(-1).detach().clone(),
        "cohort": torch.stack(cohort_steps).flatten().detach().clone(),
        "segment": torch.stack(sequence_steps).flatten().detach().clone(),
        "command": torch.stack(command_steps).flatten(0, 1).detach().clone(),
        "reward_components": torch.stack(reward_component_steps).flatten(0, 1).detach().clone(),
        "fall": torch.stack(falls).flatten().detach().clone(),
    }
    data["global_adv"] = (data["raw_adv"] - data["raw_adv"].mean()) / (data["raw_adv"].std() + 1e-8)
    data["contacts"] = torch.stack(contacts).flatten(0, 1) if contacts else torch.empty(0, device=runner.device)
    return data, dict(reward_events), dict(gait_counts)


def checkpoint_analysis(runner, wrapped, command_term, checkpoint, reward_index, joint_names):
    runner.load(checkpoint["path_abs"], load_cfg={
        "actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False,
    }, strict=True, map_location=runner.device)
    wrapped.reset()
    set_exact_cohorts(command_term)
    # Natural stepping only: populate physically consistent command/gait phases.
    obs = wrapped.get_observations().to(runner.device)
    with torch.inference_mode():
        for _ in range(args.warmup_steps):
            actions = runner.alg.actor(obs, stochastic_output=False)
            obs, _, _, _ = wrapped.step(actions.to(wrapped.unwrapped.device))
            obs = obs.to(runner.device)
    data, events, gait = collect(runner, wrapped, command_term, reward_index)
    actor_params = list(runner.alg.actor.parameters())
    critic_params = list(runner.alg.critic.parameters())
    layers = layer_slices(runner.alg.actor)
    global_adv = data["global_adv"]
    gradient_variants = {}
    actor_grads, entropy_grads, critic_grads = {}, {}, {}
    adv_rows, norm_rows = [], []
    cohort_masks = {}
    for ci, name in enumerate(COHORTS):
        idx = torch.where(data["cohort"] == ci)[0]
        cohort_masks[name] = idx
        raw = data["raw_adv"][idx]
        variants = {
            "global": global_adv[idx],
            "cohort_local": (raw - raw.mean()) / (raw.std() + 1e-8),
            "none": raw,
        }
        for variant, adv in variants.items():
            actor, entropy, critic = losses(runner.alg, data, idx, adv)
            ag = flat_grad(actor, actor_params, retain=True).detach()
            eg = flat_grad(-runner.alg.entropy_coef * entropy, actor_params, retain=True).detach()
            cg = flat_grad(runner.alg.value_loss_coef * critic, critic_params).detach()
            gradient_variants[(name, variant)] = ag
            if variant == "global":
                actor_grads[name], entropy_grads[name], critic_grads[name] = ag, eg, cg
            adv_rows.append({"iteration": checkpoint["iteration"], "cohort": name, "normalization": variant, **qstats(adv)})
        norm_rows.append({
            "iteration": checkpoint["iteration"], "cohort": name,
            "actor_surrogate_norm": float(torch.linalg.vector_norm(actor_grads[name])),
            "entropy_std_norm": float(torch.linalg.vector_norm(entropy_grads[name])),
            "critic_value_norm": float(torch.linalg.vector_norm(critic_grads[name])),
            "combined_ppo_norm": math.hypot(
                float(torch.linalg.vector_norm(actor_grads[name] + entropy_grads[name])),
                float(torch.linalg.vector_norm(critic_grads[name])),
            ),
            "finite": bool(torch.isfinite(actor_grads[name]).all() and torch.isfinite(critic_grads[name]).all()),
        })
    combined = sum(float(WEIGHTS[i]) * actor_grads[n] for i, n in enumerate(COHORTS))
    pair = {}
    for i, a in enumerate(COHORTS):
        for b in COHORTS[i + 1:]:
            pair[f"{a}__{b}"] = pair_metrics(actor_grads[a], actor_grads[b])
    pair["normalization_comparison"] = {}
    for variant in ("global", "cohort_local", "none"):
        pair["normalization_comparison"][variant] = {}
        for i, a in enumerate(COHORTS):
            for b in COHORTS[i + 1:]:
                pair["normalization_comparison"][variant][f"{a}__{b}"] = pair_metrics(
                    gradient_variants[(a, variant)], gradient_variants[(b, variant)])
    projections = []
    for i, name in enumerate(COHORTS):
        g = actor_grads[name]
        projections.append({
            "iteration": checkpoint["iteration"], "cohort": name,
            "raw_projection": float(torch.dot(g, combined)),
            "normalized_projection": cosine(g, combined),
            "angle_to_combined_deg": math.degrees(math.acos(max(-1.0, min(1.0, cosine(g, combined))))),
            "weighted_contribution_norm": float(WEIGHTS[i] * torch.linalg.vector_norm(g)),
        })
    # Four 1536-sample sub-batches per cohort preserve the formal 6144-sample
    # diagnostic cohort while exposing conflict prevalence rather than one mean.
    mb = {}
    mb_grads = {}
    for name in COHORTS:
        idx = cohort_masks[name]
        chunks = torch.chunk(idx, 4)
        mb_grads[name] = []
        for chunk in chunks:
            actor, _, _ = losses(runner.alg, data, chunk, global_adv[chunk])
            mb_grads[name].append(flat_grad(actor, actor_params).detach())
    for i, a in enumerate(COHORTS):
        for b in COHORTS[i + 1:]:
            cs = [cosine(x, y) for x in mb_grads[a] for y in mb_grads[b]]
            t = torch.tensor(cs)
            mb[f"{a}__{b}"] = {
                "comparison_count": len(cs), "negative_cosine_rate": float((t < 0).float().mean()),
                "cosine_median": float(t.median()), "cosine_p10": float(torch.quantile(t, .1)),
                "cosine_p90": float(torch.quantile(t, .9)),
            }
    layer_rows = []
    for _, layer, start, end, _ in layers:
        pass
    for layer in dict.fromkeys(x[1] for x in layers):
        spans = [(s, e) for _, l, s, e, _ in layers if l == layer]
        def take(g):
            return torch.cat([g[s:e] for s, e in spans])
        for name in COHORTS:
            lg = take(actor_grads[name])
            layer_rows.append({
                "iteration": checkpoint["iteration"], "layer": layer, "cohort": name,
                "gradient_norm": float(torch.linalg.vector_norm(lg)),
                "combined_projection": float(torch.dot(lg, take(combined))),
                "cosine_to_run": cosine(lg, take(actor_grads["RUN_HOLD"])),
                "negative_coordinate_fraction_vs_run": pair_metrics(lg, take(actor_grads["RUN_HOLD"]))["negative_coordinate_fraction"],
            })
    # Output-head row/bias attribution.
    named = list(runner.alg.actor.named_parameters())
    output_weight = next((n for n, p in named if n.startswith("mlp.6") and p.ndim == 2), None)
    output_bias = next((n for n, p in named if n.startswith("mlp.6") and p.ndim == 1), None)
    offsets = {n: (s, e, shape) for n, _, s, e, shape in layers}
    joint_rows = []
    if output_weight and output_bias:
        ws, we, wshape = offsets[output_weight]
        bs, be, _ = offsets[output_bias]
        for ji, joint in enumerate(joint_names):
            vectors = {}
            for name in COHORTS:
                w = actor_grads[name][ws:we].reshape(wshape)[ji]
                b = actor_grads[name][bs:be][ji:ji + 1]
                vectors[name] = torch.cat((w, b))
            joint_rows.append({
                "iteration": checkpoint["iteration"], "joint_index": ji, "joint_name": joint,
                **{f"{n}_norm": float(torch.linalg.vector_norm(vectors[n])) for n in COHORTS},
                "run_zero_cosine": cosine(vectors["RUN_HOLD"], vectors["ZERO_HOLD"]),
                "run_walk_cosine": cosine(vectors["RUN_HOLD"], vectors["WALK_STEADY"]),
                "run_sequence_cosine": cosine(vectors["RUN_HOLD"], vectors["BIDIRECTIONAL_SEQUENCE"]),
                "run_combined_projection": float(torch.dot(vectors["RUN_HOLD"],
                    sum(float(WEIGHTS[i]) * vectors[n] for i, n in enumerate(COHORTS)))),
            })
    critic = {}
    for name, idx in cohort_masks.items():
        v = data["values"][idx]
        ret = data["returns"][idx]
        err = ret - v
        ev = 1.0 - float(err.var() / (ret.var() + 1e-8))
        critic[name] = {
            "sample_count": int(idx.numel()), "value_mean": float(v.mean()), "return_mean": float(ret.mean()),
            "value_error_mean": float(err.mean()), "value_error_abs_mean": float(err.abs().mean()),
            "explained_variance": ev, "advantage": qstats(data["raw_adv"][idx]),
            "fall_advantage_mean": float(data["raw_adv"][idx][data["fall"][idx]].mean())
            if data["fall"][idx].any() else None,
        }
    manifest = {
        "iteration": checkpoint["iteration"], "checkpoint_sha256": checkpoint["sha256"],
        "sample_count": int(data["actions"].shape[0]), "samples_per_cohort": {
            n: int(cohort_masks[n].numel()) for n in COHORTS},
        "rollout_hash": tensor_hash(data["obs"], data["actions"], data["old_logp"], data["returns"], data["cohort"]),
        "yaw_command_nonzero": int(torch.count_nonzero(command_term.vel_command_b[:, 2])),
        "external_controller": "OFF", "checkpoint_switches": 0, "teacher_expert_calls": 0,
        "warmup_steps": args.warmup_steps, "natural_physics_steps_only": True,
    }
    return {
        "data": data, "actor_grads": actor_grads, "combined": combined,
        "manifest": manifest, "adv_rows": adv_rows, "norm_rows": norm_rows,
        "pair": pair, "projections": projections, "minibatch": mb,
        "layer_rows": layer_rows, "joint_rows": joint_rows, "critic": critic,
        "reward_events": events, "gait": gait,
    }


def shadow_cross_effect(runner, checkpoint, result):
    data = result["data"]
    masks = {n: torch.where(data["cohort"] == i)[0] for i, n in enumerate(COHORTS)}
    baseline = {}
    with torch.no_grad():
        for name, idx in masks.items():
            a, e, c = losses(runner.alg, data, idx, data["global_adv"][idx])
            baseline[name] = float(a + runner.alg.value_loss_coef * c - runner.alg.entropy_coef * e)
    rows = []
    sources = (*COHORTS, "COMBINED")
    for source in sources:
        runner.load(checkpoint["path_abs"], load_cfg={
            "actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False,
        }, strict=True, map_location=runner.device)
        runner.alg.optimizer.zero_grad()
        if source == "COMBINED":
            total = 0.0
            for i, name in enumerate(COHORTS):
                idx = masks[name]
                a, e, c = losses(runner.alg, data, idx, data["global_adv"][idx])
                total = total + float(WEIGHTS[i]) * (
                    a + runner.alg.value_loss_coef * c - runner.alg.entropy_coef * e)
        else:
            idx = masks[source]
            a, e, c = losses(runner.alg, data, idx, data["global_adv"][idx])
            total = a + runner.alg.value_loss_coef * c - runner.alg.entropy_coef * e
        total.backward()
        runner.alg.optimizer.step()
        with torch.no_grad():
            for target, idx in masks.items():
                a, e, c = losses(runner.alg, data, idx, data["global_adv"][idx])
                after = float(a + runner.alg.value_loss_coef * c - runner.alg.entropy_coef * e)
                rows.append({
                    "iteration": checkpoint["iteration"], "update_source": source,
                    "target_cohort": target, "loss_before": baseline[target],
                    "loss_after": after, "loss_change": after - baseline[target],
                    "improves_target": after < baseline[target],
                })
    runner.load(checkpoint["path_abs"], load_cfg={
        "actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False,
    }, strict=True, map_location=runner.device)
    return rows


def emit_outputs(manifests, hash_rows, adv_rows, norm_rows, projections, layer_rows,
                 joint_rows, mb_rows, event_rows, pair_json, critic_json, cross, reward_index):
    dump("cohort_rollout_manifests.json", {"status": "PASS", "rollouts": manifests})
    dump("cohort_rollout_hashes.json", {"canonical": "float tensor raw bytes; fixed field order", "hashes": hash_rows})
    dump("pairwise_gradient_matrices.json", pair_json)
    dump("critic_advantage_diagnosis.json", critic_json)
    dump("run_reward_event_reachability.json", {
        "reward_term": "safe_periodic_flight", "source_index": reward_index,
        "counts_by_checkpoint": event_rows,
        "note": "Counts are raw diagnostic rollout events after physically stepped warm-up.",
    })
    write_csv("advantage_statistics.csv", adv_rows)
    write_csv("gradient_norms_by_checkpoint.csv", norm_rows)
    write_csv("combined_gradient_projections.csv", projections)
    write_csv("minibatch_conflict_statistics.csv", mb_rows)
    write_csv("layerwise_gradient_conflicts.csv", layer_rows)
    write_csv("jointwise_gradient_conflicts.csv", joint_rows)
    write_csv("run_reward_event_timeline.csv", event_rows)
    for iteration, rows in cross.items():
        write_csv(f"one_step_cross_effect_matrix_{'initial' if iteration == 0 else 'iter' + str(iteration)}.csv", rows)
    torch.save({
        "version": 1, "seed": SEED, "checkpoint_rollout_hashes": hash_rows,
        "pairwise": pair_json, "projections": projections,
    }, OUT / "raw_diagnostic_snapshot.pt")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["checkpoints"]
    for row in manifest:
        row["path_abs"] = str((REPO / row["path"]).resolve())
        if file_hash(row["path_abs"]) != row["sha256"]:
            raise RuntimeError(f"CHECKPOINT_HASH_MISMATCH:{row['iteration']}")
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1024
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
        joint_names = list(wrapped.unwrapped.scene["robot"].joint_names)
        reward_names = list(wrapped.unwrapped.reward_manager.active_terms)
        reward_index = reward_names.index("safe_periodic_flight") if "safe_periodic_flight" in reward_names else None
        cross = {}
        manifests, hash_rows, adv_rows, norm_rows, projections = [], [], [], [], []
        layer_rows, joint_rows, mb_rows, event_rows = [], [], [], []
        pair_json, critic_json = {}, {}
        for checkpoint in manifest:
            print(f"[Stage2C] checkpoint {checkpoint['iteration']}", flush=True)
            result = checkpoint_analysis(runner, wrapped, command_term, checkpoint, reward_index, joint_names)
            raw_dir = OUT / "raw_gradients"
            raw_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "iteration": checkpoint["iteration"],
                "checkpoint_sha256": checkpoint["sha256"],
                "actor_surrogate": {k: v.detach().cpu() for k, v in result["actor_grads"].items()},
                "combined_actor": result["combined"].detach().cpu(),
            }, raw_dir / f"gradients_{checkpoint['iteration']}.pt")
            raw_rollout_dir = OUT / "raw_rollouts"
            raw_rollout_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "iteration": checkpoint["iteration"], "checkpoint_sha256": checkpoint["sha256"],
                **{k: v.detach().cpu() for k, v in result["data"].items()},
                "reward_component_names": reward_names,
                "gait_state_contract": "contact-state trace plus formal performance timeline; no result-fitted classifier",
            }, raw_rollout_dir / f"rollout_{checkpoint['iteration']}.pt")
            if checkpoint["iteration"] in (0, 100, 300):
                cross[checkpoint["iteration"]] = shadow_cross_effect(runner, checkpoint, result)
            manifests.append(result["manifest"])
            hash_rows.append({
                "iteration": checkpoint["iteration"], "checkpoint_sha256": checkpoint["sha256"],
                "rollout_sha256": result["manifest"]["rollout_hash"],
            })
            adv_rows += result["adv_rows"]
            norm_rows += result["norm_rows"]
            projections += result["projections"]
            layer_rows += result["layer_rows"]
            joint_rows += result["joint_rows"]
            pair_json[str(checkpoint["iteration"])] = result["pair"]
            critic_json[str(checkpoint["iteration"])] = result["critic"]
            for pair, value in result["minibatch"].items():
                mb_rows.append({"iteration": checkpoint["iteration"], "pair": pair, **value})
            event_rows.append({
                "iteration": checkpoint["iteration"], **result["reward_events"], **result["gait"],
            })
            # Raw observation and gradient tensors are deliberately ephemeral.
            del result
            torch.cuda.empty_cache()
        # Persist while the Isaac application is still alive. Its shutdown path
        # terminates the standalone process on this Windows runtime.
        emit_outputs(
            manifests, hash_rows, adv_rows, norm_rows, projections, layer_rows,
            joint_rows, mb_rows, event_rows, pair_json, critic_json, cross, reward_index,
        )
        print("[Stage2C] collection complete", flush=True)
        wrapped.close()


if __name__ == "__main__":
    main()
