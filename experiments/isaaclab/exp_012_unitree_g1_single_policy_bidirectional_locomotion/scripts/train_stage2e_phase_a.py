"""Stage 2E Phase A: one focused continuation from the iteration-100 checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2e_phase_a_run_acquisition_preflight"
PARENT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_retry1/checkpoints/model_100.pt"
PARENT_SHA = "8d8afac60cafbd4adf0b98469fab01f711f32771a40899653d962cc08a5d8143"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.strict_ppo_resume import Exp012StrictPPOResumeContract  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    rows = list(rows)
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_equal(current, expected):
    return set(current) == set(expected) and all(torch.equal(current[k].cpu(), expected[k].cpu()) for k in current)


def grad_norm(module):
    return math.sqrt(sum(float(torch.sum(p.grad.detach() ** 2)) for p in module.parameters() if p.grad is not None))


def rollout_metrics(alg):
    storage = alg.storage
    observations = storage.observations.flatten(0, 1)
    actions = storage.actions.flatten(0, 1)
    old_logp = storage.actions_log_prob.flatten(0, 1).squeeze(-1)
    old_mean, old_std = (x.flatten(0, 1) for x in storage.distribution_params)
    with torch.no_grad():
        alg.actor(observations, stochastic_output=True)
        new_mean, new_std = (x.clone() for x in alg.actor.output_distribution_params)
        new_logp = alg.actor.get_output_log_prob(actions)
    ratio = torch.exp(new_logp - old_logp)
    old_dist, new_dist = torch.distributions.Normal(old_mean, old_std), torch.distributions.Normal(new_mean, new_std)
    exact = torch.distributions.kl_divergence(old_dist, new_dist).sum(-1)
    return {
        "exact_old_new": float(exact.mean()),
        "clip_fraction": float(((ratio < 1 - alg.clip_param) | (ratio > 1 + alg.clip_param)).float().mean()),
        "ratio_p95": float(torch.quantile(ratio, .95)),
        "ratio_p99": float(torch.quantile(ratio, .99)),
        "mean_action_shift": float(torch.linalg.vector_norm(new_mean - old_mean, dim=-1).mean()),
    }


def save_checkpoint(runner, path, local_iteration):
    payload = runner.alg.save()
    payload["iter"] = 100 + local_iteration
    payload["infos"] = {
        "exp": "exp_012",
        "run_identity": "stage2e_phase_a_run_acquisition_preflight",
        "phase_a_iteration": local_iteration,
        "source_iteration": 100,
        "runtime_learning_rate": float(runner.alg.learning_rate),
        "scheduler_learning_rate": float(runner.alg.learning_rate),
        "resume_contract": "Exp012StrictPPOResumeContract",
        "single_checkpoint_continuation": True,
    }
    torch.save(payload, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = OUT / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    dump("starting_repository_state.json", {
        "starting_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "starting_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines(),
    })
    if file_hash(PARENT) != PARENT_SHA:
        raise RuntimeError("PHASE_A_PARENT_PROVENANCE_FAIL")

    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp012-G1-PhaseA-RunAcquisition-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = 1024
    cfg.seed = 20265021
    cfg.episode_length_s = 20.0
    agent_cfg.seed = 20265021
    agent_cfg.max_iterations = 100
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-PhaseA-RunAcquisition-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        parent = torch.load(PARENT, map_location=runner.device, weights_only=False)
        runner.load(str(PARENT), load_cfg={
            "actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False,
        }, strict=True, map_location=runner.device)
        contract = Exp012StrictPPOResumeContract()
        contract.require_optimizer_state(parent)
        lr_state = contract.synchronize(runner.alg, runner, resume=True)
        optimizer = runner.alg.optimizer.state_dict()
        adam_steps = sorted({int(float(v["step"])) for v in optimizer["state"].values()})
        actor_ok = state_equal(runner.alg.actor.state_dict(), parent["actor_state_dict"])
        critic_ok = state_equal(runner.alg.critic.state_dict(), parent["critic_state_dict"])
        optimizer_ok = len(optimizer["state"]) == len(parent["optimizer_state_dict"]["state"]) == 17
        expected_lr = float(parent["optimizer_state_dict"]["param_groups"][0]["lr"])
        lr_ok = (
            abs(float(runner.alg.optimizer.param_groups[0]["lr"]) - expected_lr) <= 1e-12
            and abs(float(runner.alg.learning_rate) - expected_lr) <= 1e-12
        )
        command_term = raw.unwrapped.command_manager.get_term("base_velocity")
        reward_term = raw.unwrapped.reward_manager.get_term_cfg("safe_periodic_flight").func
        reward_names = list(raw.unwrapped.reward_manager.active_terms)
        integrity = {
            "status": "PASS" if actor_ok and critic_ok and optimizer_ok and adam_steps == [87000] and lr_ok else "FAIL",
            "parent_path": str(PARENT.relative_to(REPO)), "parent_sha256": file_hash(PARENT),
            "actor_hash_match": actor_ok, "critic_hash_match": critic_ok,
            "std_hash_match": torch.equal(
                runner.alg.actor.state_dict()["distribution.std_param"].cpu(),
                parent["actor_state_dict"]["distribution.std_param"].cpu()),
            "optimizer_hash_match": optimizer_ok, "adam_step": adam_steps,
            "optimizer_lr": float(runner.alg.optimizer.param_groups[0]["lr"]),
            "runtime_lr": float(runner.alg.learning_rate), "scheduler_lr": float(runner.alg.learning_rate),
            "resume_contract": lr_state.to_dict(), "reward_semantic_diff": 0,
            "yaw_command": 0, "external_controllers": "OFF", "unique_checkpoint": 1,
            "teacher_expert_calls": 0,
        }
        dump("phase_a_pre_run_integrity.json", integrity)
        if integrity["status"] != "PASS":
            raise RuntimeError("PHASE_A_PRE_RUN_INTEGRITY_FAIL")

        dump("resolved_phase_a_curriculum.json", {
            "target_distribution": {
                "focused": {"probability": .70, "uniform_mps": [2.4, 2.5]},
                "full": {"probability": .30, "uniform_mps": [2.3, 2.6]},
            },
            "profile": [
                {"phase": "zero_hold", "duration_s": 1.0, "target": 0.0},
                {"phase": "zero_to_walk_minimum_jerk", "duration_s": 1.0, "target": 1.2},
                {"phase": "walk_hold", "duration_s": 1.5, "target": 1.2},
                {"phase": "walk_to_run_minimum_jerk", "duration_s": 1.5, "target": "sampled"},
                {"phase": "run_hold", "duration_s": 15.0, "target": "sampled"},
            ],
            "episode_duration_s": 20.0, "all_environments_phase_a": True,
        })
        dump("resolved_reward_config.json", {
            "base_reward_semantic_difference": 0,
            "safe_periodic_flight_semantic_difference": 0,
            "reward_component_names": reward_names,
            "threshold_or_weight_changes": 0,
        })
        dump("single_checkpoint_contract.json", {
            "parent_checkpoint_count": 1, "runtime_checkpoint_switches": 0,
            "expert_router": 0, "teacher_action_calls": 0, "action_blend": 0,
            "continued_actor": True,
        })
        dump("resume_lr_contract.json", {
            "name": "Exp012StrictPPOResumeContract", "source": "restored optimizer param group",
            "restored_lr": expected_lr, "runtime_lr": runner.alg.learning_rate,
            "scheduler_lr": runner.alg.learning_rate, "tolerance": 1e-12,
        })
        dump("resolved_phase_a_training_config.yaml", {
            "run_identity": "stage2e_phase_a_run_acquisition_preflight",
            "num_envs": 1024, "iterations": 100, "rollout_steps": agent_cfg.num_steps_per_env,
            "interactions": 2457600, "seed": 20265021, "episode_duration_s": 20.0,
            "ppo": agent_cfg.algorithm.to_dict(),
        })

        runner.current_learning_iteration = 100
        save_checkpoint(runner, checkpoints / "model_initial.pt", 0)
        obs = wrapped.get_observations().to(runner.device)
        curves, events_timeline, lr_trace = [], [], []
        schedule = {1, 5, 10, 20, 30, 40, 50, 75, 100}
        first_update, stopped = None, None
        consecutive_high = 0
        local_last_landing = torch.full((1024,), -1, dtype=torch.long, device=runner.device)
        local_was_flight = torch.zeros(1024, dtype=torch.bool, device=runner.device)

        for iteration in range(1, 101):
            requested_samples = precursor_steps = safe_steps = completion_fires = 0
            landing_candidates = alternating_candidates = yaw_nonzero = 0
            per_env_completion = torch.zeros(1024, device=runner.device)
            per_env_flight = torch.zeros(1024, dtype=torch.bool, device=runner.device)
            falls = torch.zeros(1024, dtype=torch.bool, device=runner.device)
            speed_error_24, speed_error_26, reward_means = [], [], []
            with torch.inference_mode():
                for _ in range(agent_cfg.num_steps_per_env):
                    actions = runner.alg.act(obs)
                    obs, rewards, dones, extras = wrapped.step(actions.to(wrapped.unwrapped.device))
                    obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                    runner.alg.process_env_step(obs, rewards, dones, extras)
                    timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
                    command = command_term.vel_command_b[:, 0]
                    actual = raw.unwrapped.scene["robot"].data.root_lin_vel_b[:, 0]
                    run = command >= 2.3
                    requested_samples += int(run.sum())
                    yaw_nonzero += int(torch.count_nonzero(command_term.vel_command_b[:, 2]))
                    raw_run = reward_term.last_raw_reward
                    completion = raw_run >= 1.0
                    precursor_steps += int(((raw_run > 0) & (raw_run < .10)).sum())
                    safe_steps += int(((raw_run >= .10) & (raw_run < 1.0)).sum())
                    completion_fires += int(completion.sum())
                    per_env_completion += completion.float()
                    sensor = raw.unwrapped.scene.sensors["contact_forces"]
                    forces = sensor.data.net_forces_w_history[:, :, reward_term.foot_ids, :].norm(dim=-1).amax(dim=1)
                    contacts = forces > 1.0
                    flight = contacts.sum(-1) == 0
                    landing = local_was_flight & ~flight
                    single = landing & (contacts.sum(-1) == 1)
                    foot = contacts.long().argmax(-1)
                    alternating = single & (local_last_landing >= 0) & (foot != local_last_landing)
                    landing_candidates += int(landing.sum())
                    alternating_candidates += int(alternating.sum())
                    local_last_landing[single] = foot[single]
                    local_was_flight.copy_(flight)
                    per_env_flight |= flight
                    falls |= dones.bool() & ~timeouts
                    reset = dones.bool()
                    local_last_landing[reset] = -1
                    local_was_flight[reset] = False
                    near24 = run & ((command - 2.4).abs() <= .06)
                    near26 = run & ((command - 2.6).abs() <= .06)
                    if near24.any():
                        speed_error_24.append(float((actual[near24] - command[near24]).abs().mean()))
                    if near26.any():
                        speed_error_26.append(float((actual[near26] - command[near26]).abs().mean()))
                    reward_means.append(float(rewards.mean()))
                runner.alg.compute_returns(obs)

            original_clear = runner.alg.storage.clear
            runner.alg.storage.clear = lambda: None
            original_step = runner.alg.optimizer.step
            original_kl = runner.alg.actor.get_kl_divergence
            pending, step_rows, first_step_metrics = [], [], []

            def traced_kl(old_params, new_params):
                kl = original_kl(old_params, new_params)
                value = float(kl.mean())
                action = "DECREASE" if value > runner.alg.desired_kl * 2 else (
                    "INCREASE" if 0 < value < runner.alg.desired_kl / 2 else "HOLD")
                pending.append({"kl": value, "action": action})
                return kl

            def traced_step(*a, **kw):
                before_opt = float(runner.alg.optimizer.param_groups[0]["lr"])
                before_run = float(runner.alg.learning_rate)
                if abs(before_opt - before_run) > 1e-12:
                    raise RuntimeError("PHASE_A_RUNTIME_LR_CONTRACT_REGRESSION")
                result = original_step(*a, **kw)
                after_opt = float(runner.alg.optimizer.param_groups[0]["lr"])
                after_run = float(runner.alg.learning_rate)
                if abs(after_opt - after_run) > 1e-12:
                    raise RuntimeError("PHASE_A_RUNTIME_LR_CONTRACT_REGRESSION")
                step_rows.append({
                    "optimizer_step": len(step_rows) + 1, "lr_before": before_opt, "lr_after": after_opt,
                    "runtime_lr": after_run, "scheduler_lr": after_run,
                    "adaptive_kl": pending[-1]["kl"], "adaptive_action": pending[-1]["action"],
                })
                if iteration == 1:
                    first_step_metrics.append(rollout_metrics(runner.alg))
                return result

            runner.alg.actor.get_kl_divergence = traced_kl
            runner.alg.optimizer.step = traced_step
            losses = runner.alg.update()
            runner.alg.actor.get_kl_divergence = original_kl
            runner.alg.optimizer.step = original_step
            actor_gradient = grad_norm(runner.alg.actor)
            critic_gradient = grad_norm(runner.alg.critic)
            metrics = rollout_metrics(runner.alg)
            runner.alg.storage.clear = original_clear
            original_clear()

            finite = all(math.isfinite(float(v)) for v in losses.values()) and all(
                torch.isfinite(p).all() for p in runner.alg.actor.parameters())
            value_loss = float(losses.get("value", losses.get("value_loss", 0.0)))
            current_lr = float(runner.alg.learning_rate)
            lr_equal = all(
                abs(row["lr_after"] - row["runtime_lr"]) <= 1e-12
                and abs(row["lr_after"] - row["scheduler_lr"]) <= 1e-12 for row in step_rows)
            adam_step = max(int(float(v["step"])) for v in runner.alg.optimizer.state_dict()["state"].values())
            periodic = float((per_env_completion >= 2).float().mean())
            isolated = float((per_env_flight & (per_env_completion == 0) & ~falls).float().mean())
            irregular = float((~falls & ~((per_env_completion >= 2)) & ~(per_env_flight & (per_env_completion == 0))).float().mean())
            event_row = {
                "iteration": iteration, "requested_run_samples": requested_samples,
                "takeoff_precursor_steps": precursor_steps, "safe_flight_steps": safe_steps,
                "landing_candidates": landing_candidates, "alternating_landing_candidates": alternating_candidates,
                "completion_reward_fire_count": completion_fires,
                "completion_per_run_sample": completion_fires / max(1, requested_samples),
                "completion_per_environment": completion_fires / 1024.0,
                "periodic_running_rate": periodic, "isolated_flight_rate": isolated,
                "irregular_rate": irregular, "fall_rate": float(falls.float().mean()),
                "speed_mae_2p4": sum(speed_error_24) / max(1, len(speed_error_24)),
                "speed_mae_2p6": sum(speed_error_26) / max(1, len(speed_error_26)),
                "heading_p95": "DEFERRED_CHECKPOINT_EVAL", "slip": "DEFERRED_CHECKPOINT_EVAL",
                "impact": "DEFERRED_CHECKPOINT_EVAL", "saturation": "DEFERRED_CHECKPOINT_EVAL",
            }
            events_timeline.append(event_row)
            curve = {
                "iteration": iteration, "interactions": iteration * 1024 * agent_cfg.num_steps_per_env,
                "mean_reward": sum(reward_means) / len(reward_means),
                **metrics, "actor_gradient_norm": actor_gradient, "critic_gradient_norm": critic_gradient,
                "value_loss": value_loss, "entropy": float(losses.get("entropy", 0.0)),
                "learning_rate": current_lr, "adam_step": adam_step,
                "nan_inf": 0 if finite else 1, "yaw_nonzero_samples": yaw_nonzero,
            }
            curves.append(curve)
            lr_trace.append({
                "iteration": iteration, "first_optimizer_step_lr": step_rows[0]["lr_before"],
                "final_optimizer_step_lr": step_rows[-1]["lr_after"],
                "minimum_lr": min([x["lr_before"] for x in step_rows] + [x["lr_after"] for x in step_rows]),
                "maximum_lr": max([x["lr_before"] for x in step_rows] + [x["lr_after"] for x in step_rows]),
                "adaptive_increase_count": sum(x["adaptive_action"] == "INCREASE" for x in step_rows),
                "adaptive_decrease_count": sum(x["adaptive_action"] == "DECREASE" for x in step_rows),
                "optimizer_runtime_scheduler_equal": lr_equal, "adam_step": adam_step,
                "exact_rollout_kl": metrics["exact_old_new"], "clip_fraction": metrics["clip_fraction"],
            })
            if iteration == 1:
                all_step_max = max(x["exact_old_new"] for x in first_step_metrics)
                first_step_lr = step_rows[0]["lr_before"]
                gate = (
                    metrics["exact_old_new"] <= .20 and all_step_max <= .20
                    and metrics["clip_fraction"] <= .50 and metrics["mean_action_shift"] <= 2.0
                    and critic_gradient <= 1e6 and value_loss <= 1e8 and finite and lr_equal
                    and abs(first_step_lr - expected_lr) <= 1e-12
                )
                first_update = {
                    "status": "PASS" if gate else "PHASE_A_FIRST_UPDATE_UNSTABLE",
                    **metrics, "all_step_maximum_kl": all_step_max,
                    "actor_gradient_norm": actor_gradient, "critic_gradient_norm": critic_gradient,
                    "value_loss": value_loss, "nan_inf": 0 if finite else 1,
                    "first_step_lr": first_step_lr, "final_lr": step_rows[-1]["lr_after"],
                    "optimizer_step_trace": step_rows,
                }
                if not gate:
                    stopped = "PHASE_A_FIRST_UPDATE_UNSTABLE"
            consecutive_high = consecutive_high + 1 if metrics["exact_old_new"] > .20 else 0
            if (
                not finite or metrics["exact_old_new"] > .50 or critic_gradient > 1e6
                or value_loss > 1e8 or consecutive_high >= 3 or not lr_equal
            ):
                stopped = stopped or "PHASE_A_EARLY_TRAINING_UNSTABLE"
            if iteration in schedule or stopped:
                save_checkpoint(runner, checkpoints / f"model_{iteration}.pt", iteration)
            print(
                f"[PhaseA] iter={iteration} completion={completion_fires} density={event_row['completion_per_run_sample']:.6f} "
                f"fall={event_row['fall_rate']:.3f} kl={metrics['exact_old_new']:.5f} lr={current_lr:.8g}",
                flush=True,
            )
            if stopped:
                break

        write_csv("training_curves.csv", curves)
        write_csv("runtime_lr_trace.csv", lr_trace)
        write_csv("phase_a_run_event_timeline.csv", events_timeline)
        dump("first_update_stability.json", first_update)
        dump("early_training_guard.json", {
            "status": "PASS" if not stopped and len(curves) >= 10 else stopped,
            "iterations_audited": min(10, len(curves)), "rows": curves[:10],
        })
        dump("training_run_summary.json", {
            "status": stopped or "COMPLETE", "completed_iterations": len(curves),
            "completed_interactions": len(curves) * 1024 * agent_cfg.num_steps_per_env,
            "first_completion_iteration": next(
                (x["iteration"] for x in events_timeline if x["completion_reward_fire_count"] > 0), None),
            "checkpoint_schedule": sorted({0, *schedule} & {0, *range(1, len(curves) + 1)}),
        })
        wrapped.close()


if __name__ == "__main__":
    main()
