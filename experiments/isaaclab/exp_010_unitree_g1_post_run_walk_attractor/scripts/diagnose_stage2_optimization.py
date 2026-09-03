"""Stage 2 read-only PPO stability diagnostics for exp_010 Pilot 1."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Normal
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
STAGE1 = REPO / "results/exp_010_unitree_g1_post_run_walk_attractor/stage1_post_run_walk_pilot1"
OUT = REPO / "results/exp_010_unitree_g1_post_run_walk_attractor/stage2_optimization_stability_preflight"
CFG_PATH = EXP / "configs/stage0_post_run_walk_pilot1.yaml"
STAGE1_SCRIPT = EXP / "scripts/execute_post_run_walk_pilot1.py"
STAGE8C_SCRIPT = REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/scripts/execute_stage8c_pilot1.py"
EXPECTED = {
    "stand": ("logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt", "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"),
    "stw": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt", "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e"),
    "walk": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt", "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa"),
    "run": ("logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt", "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266"),
    "wtr": ("results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt", "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0"),
    "model10": ("results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution/checkpoints/model_10.pt", "f54ead0da2a192e238e1fd6dbcb48670fb785f7ef7e7766c64d0dfbf06eba263"),
}

sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]
import g1_command_skills.tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
import isaaclab_tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_run_expert, load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_run_observation  # noqa: E402
from g1_walk_centered.tasks.stage7r_action import RunToWalkTransitionActor152, WalkToRunTransitionActor152  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402
from post_run_walk.actor import PostRunWalkExpert152  # noqa: E402
from post_run_walk.contract import PostRunWalkContractState, update_contract  # noqa: E402
from post_run_walk.reward import post_run_walk_reward as production_reward  # noqa: E402


def write_json(name: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mj(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0, 1)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 3:
        return 0.0
    x, y = ranks(left), ranks(right)
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - xm) ** 2 for a in x) * sum((b - ym) ** 2 for b in y)
    )
    return numerator / denominator if denominator else 0.0


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    return float((left @ right) / denominator) if float(denominator) else 0.0


def extract_function(path: Path, enclosing: str, name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == enclosing)
    function = next(
        node for node in ast.walk(owner)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class SourceState:
    pass


def build_models(cfg: dict, device: torch.device, paths: dict[str, Path]):
    stand = load_walk_expert(paths["stand"], device=device)
    stw = load_walk_expert(paths["stw"], device=device)
    walk = load_walk_expert(paths["walk"], device=device)
    run = load_run_expert(paths["run"], device=device)
    wtr = WalkToRunTransitionActor152(run.actor).to(device)
    wtr.load_state_dict(torch.load(paths["wtr"], map_location=device, weights_only=False)["actor"], strict=True)
    wtr.eval()
    source_actor = RunToWalkTransitionActor152(run.actor).to(device)
    source_actor.load_state_dict(
        torch.load(paths["model10"], map_location=device, weights_only=False)["actor"],
        strict=True,
    )
    source_actor.eval()
    for parameter in source_actor.parameters():
        parameter.requires_grad_(False)
    actor = PostRunWalkExpert152(source_actor).to(device)
    critic = nn.Sequential(
        nn.Linear(152, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 1),
    ).to(device)
    log_std = nn.Parameter(torch.zeros(37, device=device))
    return stand, stw, walk, run, wtr, source_actor, actor, critic, log_std


parser = __import__("argparse").ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main() -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    paths = {name: (REPO / relative).resolve() for name, (relative, _) in EXPECTED.items()}
    before_hashes = {name: file_sha(path) for name, path in paths.items()}
    if not all(before_hashes[name] == EXPECTED[name][1] for name in EXPECTED):
        raise RuntimeError("protected checkpoint hash mismatch")
    manifests = json.loads((STAGE1 / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    checkpoints = [(Path(item["path"]).stem, REPO / item["path"], item["iteration"]) for item in manifests]

    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    task_cfg.scene.num_envs = cfg["experiment"]["physical_envs"]
    task_cfg.seed = 20270501
    task_cfg.episode_length_s = 40.0
    task_cfg.sim.device = cfg["experiment"]["device"]
    args.device = cfg["experiment"]["device"]

    segment_rows: list[dict] = []
    gradient_rows: list[dict] = []
    gradient_summaries: list[dict] = []
    critic_summaries: list[dict] = []
    mean_policy_rows: list[dict] = []
    captured_by_checkpoint: dict[str, dict] = {}

    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg),
            clip_actions=agent_cfg.clip_actions,
        )
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        dt = float(env.step_dt)
        stand, stw, walk, run, wtr, source_actor, actor, critic, log_std = build_models(cfg, device, paths)
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        joints, joint_names = robot.find_joints(".*")
        joint_groups = {
            "hip": [i for i, name in enumerate(joint_names) if "hip" in name],
            "knee": [i for i, name in enumerate(joint_names) if "knee" in name],
            "ankle_pitch": [i for i, name in enumerate(joint_names) if "ankle_pitch" in name],
            "ankle_roll": [i for i, name in enumerate(joint_names) if "ankle_roll" in name],
            "lower_body": [i for i, name in enumerate(joint_names) if any(key in name for key in ("hip", "knee", "ankle"))],
        }
        common_namespace = {
            "cfg": cfg, "wrapped": wrapped, "device": device, "dt": dt,
            "stand": stand, "stw": stw, "walk": walk, "run": run, "wtr": wtr,
            "source_actor": source_actor, "actor": actor, "critic": critic,
            "log_std": log_std, "robot": robot, "command_term": command_term,
            "sensor": sensor, "feet": feet, "sensor_feet": sensor_feet,
            "joints": joints, "torch": torch, "math": math, "Counter": Counter,
            "MotionCommand": MotionCommand,
            "canonical_state_from_legacy_observation": canonical_state_from_legacy_observation,
            "to_run_observation": to_run_observation, "mj": mj,
            "SourceState": SourceState, "Normal": Normal,
            "PostRunWalkContractState": PostRunWalkContractState,
            "update_contract": update_contract,
            "failure_counts": Counter(),
        }
        prepare_source = extract_function(STAGE8C_SCRIPT, "main", "prepare_source", common_namespace)
        graph_background = extract_function(STAGE8C_SCRIPT, "main", "graph_background", common_namespace)
        common_namespace["prepare_source"] = prepare_source
        common_namespace["graph_background"] = graph_background

        reference_obs = None
        reference_initial_mean = None
        reference_previous_mean = None
        reference_previous_label = None

        for checkpoint_index, (label, checkpoint_path, iteration) in enumerate(checkpoints):
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            actor.load_state_dict(payload["actor"], strict=True)
            critic.load_state_dict(payload["critic"], strict=True)
            log_std.data.copy_(payload["log_std"].to(device))

            captured_terms: list[dict[str, torch.Tensor]] = []

            def capture_reward(**kwargs):
                total, terms = production_reward(**kwargs)
                captured_terms.append({name: value.detach() for name, value in terms.items()})
                return total, terms

            common_namespace["post_run_walk_reward"] = capture_reward
            rollout = extract_function(STAGE1_SCRIPT, "main", "rollout", common_namespace)
            source = prepare_source(20270510 + checkpoint_index, 64, balanced=True)
            if source is None:
                raise RuntimeError(f"source formation failed for {label}")
            torch.manual_seed(20270610 + checkpoint_index)
            result = rollout(source, deterministic=False)
            records = result["records"]
            if len(records) != len(captured_terms):
                raise RuntimeError("reward capture and record count mismatch")

            observations = torch.cat([record["observation"] for record in records])
            actions = torch.cat([record["action"] for record in records])
            old_log = torch.cat([record["log_prob"] for record in records])
            values = torch.cat([record["value"] for record in records])
            returns = torch.cat([record["return"] for record in records])
            advantages = torch.cat([record["advantage"] for record in records])
            normalized_advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
            env_ids = torch.cat([record["env"] for record in records])
            source_speed = result["source_speed"]
            source_phase = result["source_phase"]
            count = torch.zeros(64, device=device)
            total_return = torch.zeros(64, device=device)
            advantage_sum = torch.zeros(64, device=device)
            speed_score_sum = torch.zeros(64, device=device)
            heading_score_sum = torch.zeros(64, device=device)
            term_sums = {name: torch.zeros(64, device=device) for name in cfg["reward"]}
            offset = 0
            for record, term_values in zip(records, captured_terms):
                ids = record["env"]
                size = len(ids)
                record_advantage = normalized_advantages[offset:offset + size]
                count.index_add_(0, ids, torch.ones(size, device=device))
                total_return.index_add_(0, ids, record["reward"])
                advantage_sum.index_add_(0, ids, record_advantage)
                speed_score_sum.index_add_(
                    0, ids, torch.exp(-((record["observation"][:, 0] - 1.2) / 0.30) ** 2)
                )
                heading = torch.atan2(record["observation"][:, 123 + 12], record["observation"][:, 123 + 13]).abs()
                heading_score_sum.index_add_(0, ids, torch.exp(-((heading / 0.12) ** 2)))
                for name, term in term_values.items():
                    term_sums[name].index_add_(0, ids, term)
                offset += size
            mean_advantage = advantage_sum / count.clamp_min(1)
            mean_speed_score = speed_score_sum / count.clamp_min(1)
            mean_heading_score = heading_score_sum / count.clamp_min(1)
            outcomes = result["outcomes"]
            streak_score = (result["max_valid_dwell"] / 8.0).clamp(0, 1)
            progress = (
                mean_speed_score
                + (~outcomes["periodic_run"]).float()
                + (~outcomes["excessive_flight"]).float()
                + streak_score
                + mean_heading_score
                + (~outcomes["saturation"]).float()
                + (~outcomes["slip"]).float()
                + (~outcomes["fall"]).float()
            ) / 8.0
            safety_failure = (
                outcomes["fall"] | outcomes["slip"] | outcomes["impact"]
                | outcomes["saturation"] | outcomes["excessive_flight"]
            )
            for local in range(64):
                segment_rows.append({
                    "checkpoint": label,
                    "iteration": iteration,
                    "segment": local,
                    "source_speed_mps": float(source_speed[local]),
                    "source_phase": {0: "flight", 1: "left", 2: "right", 3: "double", -1: "not_reached"}[int(source_phase[local])],
                    "return": float(total_return[local]),
                    "normalized_advantage_mean": float(mean_advantage[local]),
                    "diagnostic_progress": float(progress[local]),
                    "maximum_stable_low_speed_streak_seconds": float(result["max_valid_dwell"][local]),
                    "safety_failure": bool(safety_failure[local]),
                    "periodic_run": bool(outcomes["periodic_run"][local]),
                    "excessive_flight": bool(outcomes["excessive_flight"][local]),
                    "saturation": bool(outcomes["saturation"][local]),
                    **{f"reward_{name}": float(term_sums[name][local]) for name in term_sums},
                })

            subset = min(len(observations), 8192)
            obs_grad = observations[:subset]
            action_grad = actions[:subset]
            old_log_grad = old_log[:subset]
            adv_grad = normalized_advantages[:subset]
            mean = actor(obs_grad)
            distribution = Normal(mean, log_std.exp().expand_as(mean))
            new_log = distribution.log_prob(action_grad).sum(-1)
            ratio = (new_log - old_log_grad).exp()
            policy_loss = -torch.minimum(
                ratio * adv_grad,
                ratio.clamp(0.8, 1.2) * adv_grad,
            ).mean()
            entropy_loss = -cfg["ppo"]["entropy_coefficient"] * distribution.entropy().sum(-1).mean()
            g_policy = torch.autograd.grad(policy_loss, log_std, retain_graph=True)[0]
            g_entropy = torch.autograd.grad(entropy_loss, log_std, retain_graph=True)[0]
            g_total = g_policy + g_entropy
            actor_grads = torch.autograd.grad(
                policy_loss,
                tuple(actor.parameters()),
                retain_graph=False,
                allow_unused=True,
            )
            mean_grad_norm = math.sqrt(sum(float(gradient.square().sum()) for gradient in actor_grads if gradient is not None))
            entropy_fraction = float(g_entropy.norm() / (g_policy.norm() + g_entropy.norm() + 1e-12))
            gradient_summaries.append({
                "checkpoint": label,
                "iteration": iteration,
                "std_min": float(log_std.exp().min()),
                "std_mean": float(log_std.exp().mean()),
                "std_max": float(log_std.exp().max()),
                "g_std_policy_norm": float(g_policy.norm()),
                "g_std_entropy_norm": float(g_entropy.norm()),
                "g_std_total_norm": float(g_total.norm()),
                "policy_entropy_cosine": cosine(g_policy, g_entropy),
                "entropy_contribution_fraction": entropy_fraction,
                "mean_action_network_gradient_norm": mean_grad_norm,
                "action_entropy": float(distribution.entropy().sum(-1).mean()),
                "deterministic_mean_action_norm": float(mean.norm(dim=1).mean()),
                "sampled_action_norm": float(action_grad.norm(dim=1).mean()),
                "action_rate": float((action_grad - obs_grad[:, 86:123]).norm(dim=1).mean()),
            })
            for joint, name in enumerate(joint_names):
                gradient_rows.append({
                    "checkpoint": label,
                    "iteration": iteration,
                    "joint_index": joint,
                    "joint_name": name,
                    "std": float(log_std.exp()[joint]),
                    "g_std_policy": float(g_policy[joint]),
                    "g_std_entropy": float(g_entropy[joint]),
                    "g_std_total": float(g_total[joint]),
                })

            residual = returns - values
            explained_variance = 1.0 - float(residual.var() / returns.var().clamp_min(1e-8))
            safety_steps = safety_failure[env_ids]
            progress_repeated = progress[env_ids]
            top_threshold = torch.quantile(progress_repeated, 0.9)
            bottom_threshold = torch.quantile(progress_repeated, 0.1)
            critic_summaries.append({
                "checkpoint": label,
                "iteration": iteration,
                "explained_variance": explained_variance,
                "value_mean": float(values.mean()),
                "value_std": float(values.std()),
                "return_mean": float(returns.mean()),
                "return_std": float(returns.std()),
                "return_p95": float(torch.quantile(returns, 0.95)),
                "return_p99": float(torch.quantile(returns, 0.99)),
                "advantage_mean": float(advantages.mean()),
                "advantage_std": float(advantages.std()),
                "advantage_p95": float(torch.quantile(advantages, 0.95)),
                "advantage_p99": float(torch.quantile(advantages, 0.99)),
                "positive_advantage_rate": float((advantages > 0).float().mean()),
                "safety_failure_advantage_mean": float(normalized_advantages[safety_steps].mean()) if safety_steps.any() else 0.0,
                "progress_top_advantage_mean": float(normalized_advantages[progress_repeated >= top_threshold].mean()),
                "progress_bottom_advantage_mean": float(normalized_advantages[progress_repeated <= bottom_threshold].mean()),
                "finite": bool(torch.isfinite(values).all() and torch.isfinite(returns).all() and torch.isfinite(advantages).all()),
            })

            if reference_obs is None:
                reference_obs = observations[:4096].detach().clone()
                with torch.no_grad():
                    reference_initial_mean = actor(reference_obs).detach().clone()
                    reference_previous_mean = reference_initial_mean
                    reference_previous_label = label
            with torch.no_grad():
                current_mean = actor(reference_obs)
            shift_initial = current_mean - reference_initial_mean
            shift_previous = current_mean - reference_previous_mean
            mean_policy_rows.append({
                "checkpoint": label,
                "iteration": iteration,
                "previous_checkpoint": reference_previous_label,
                "action_l2_shift_from_initial": float(shift_initial.norm(dim=1).mean()),
                "action_l2_shift_from_previous": float(shift_previous.norm(dim=1).mean()),
                "mean_policy_kl_from_initial_fixed_std_0_2": float(0.5 * (shift_initial / 0.2).square().sum(1).mean()),
                **{
                    f"{group}_shift_from_initial": float(shift_initial[:, indices].norm(dim=1).mean())
                    for group, indices in joint_groups.items()
                },
            })
            reference_previous_mean = current_mean.detach().clone()
            reference_previous_label = label
            captured_by_checkpoint[label] = {
                "observations": observations.detach().cpu(),
                "actions": actions.detach().cpu(),
                "advantages": normalized_advantages.detach().cpu(),
            }
        # launch_simulation closes the app when the context exits, so durable
        # diagnostic outputs are flushed before wrapped.close().
        write_csv("diagnostic_segments.csv", segment_rows)
        write_csv("per_joint_std_gradient.csv", gradient_rows)
        write_json("log_std_gradient_decomposition.json", gradient_summaries)
        write_json("critic_advantage_audit.json", critic_summaries)
        write_json("mean_policy_update_audit.json", mean_policy_rows)
        write_json(
            "protected_hashes.json",
            {
                "before": before_hashes,
                "after": {name: file_sha(path) for name, path in paths.items()},
                "all_source_experts_unchanged": all(
                    file_sha(paths[name]) == before_hashes[name] for name in paths
                ),
                "pilot1_files_modified": False,
                "diagnostic_clone_only": True,
                "optimizer_steps": 0,
                "pilot2_executed": False,
            },
        )
        wrapped.close()

    write_csv("diagnostic_segments.csv", segment_rows)
    write_csv("per_joint_std_gradient.csv", gradient_rows)
    write_json("log_std_gradient_decomposition.json", gradient_summaries)
    write_json("critic_advantage_audit.json", critic_summaries)
    write_json("mean_policy_update_audit.json", mean_policy_rows)
    write_json(
        "protected_hashes.json",
        {
            "before": before_hashes,
            "after": {name: file_sha(path) for name, path in paths.items()},
            "all_source_experts_unchanged": all(file_sha(paths[name]) == before_hashes[name] for name in paths),
            "pilot1_files_modified": False,
            "diagnostic_clone_only": True,
            "optimizer_steps": 0,
            "pilot2_executed": False,
        },
    )


if __name__ == "__main__":
    main()
