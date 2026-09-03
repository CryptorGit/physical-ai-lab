"""Common-initial-condition search for safety-bounded STOP fixed feedback."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import gymnasium as gym
import torch
from tensordict import TensorDict


SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from g1_command_skills.fixed_feedback import StopFeedbackConfig, StopFixedFeedbackController  # noqa: E402
from g1_command_skills.models import G1CommandResidualActor  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--candidate-count", type=int, default=40)
parser.add_argument("--search-seed", type=int, default=20260721)
parser.add_argument("--env-seed", type=int, default=42)
parser.add_argument("--shortlist-from", type=Path)
parser.add_argument("--shortlist-count", type=int, default=5)
parser.add_argument("--refine-spike", action="store_true")
parser.add_argument("--controller-candidates", type=Path)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def as_tensordict(observation) -> TensorDict:
    if isinstance(observation, TensorDict):
        return observation
    first = next(iter(observation.values()))
    return TensorDict(observation, batch_size=[first.shape[0]])


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(round((len(ordered) - 1) * q / 100.0)), len(ordered) - 1)]


def config(name: str, **overrides) -> dict:
    values = {
        "name": name,
        "k_heading": 0.08,
        "k_yaw_rate": 0.04,
        "alpha": 1.0,
        "max_delta_per_step": 1.0,
        "braking_scale": 1.0,
        "hold_scale": 1.0,
        "double_support_scale": 1.0,
        "single_support_scale": 1.0,
        "flight_scale": 1.0,
        "yaw_soft_threshold": float("inf"),
        "yaw_hard_threshold": float("inf"),
        "hard_guard_mode": "zero",
    }
    values.update(overrides)
    return values


def candidate_grid() -> list[dict]:
    baselines = [
        config("parent", k_heading=0.0, k_yaw_rate=0.0),
        config("current"),
        config("smoothing_only", alpha=0.35),
        config("slew_only", max_delta_per_step=0.005),
        config("contact_only", single_support_scale=0.5, flight_scale=0.0),
        config("spike_only", yaw_soft_threshold=1.5, yaw_hard_threshold=4.0),
    ]
    rng = random.Random(args_cli.search_seed)
    combinations = [
        (kh, ky, alpha, slew, brake, single, hard, mode)
        for kh in (0.05, 0.065, 0.08, 0.095)
        for ky in (0.03, 0.05, 0.07, 0.09)
        for alpha in (0.35, 0.5)
        for slew in (0.0025, 0.005)
        for brake in (0.5, 0.75, 1.0)
        for single in (0.25, 0.5, 0.75)
        for hard in (2.5, 4.0)
        for mode in ("zero", "damping_only")
    ]
    rng.shuffle(combinations)
    sampled = []
    for index, (kh, ky, alpha, slew, brake, single, hard, mode) in enumerate(
        combinations[: max(args_cli.candidate_count - len(baselines), 0)]
    ):
        sampled.append(config(
            f"safe_{index:02d}", k_heading=kh, k_yaw_rate=ky, alpha=alpha,
            max_delta_per_step=slew, braking_scale=brake, single_support_scale=single,
            flight_scale=0.0, yaw_soft_threshold=1.5, yaw_hard_threshold=hard,
            hard_guard_mode=mode,
        ))
    return baselines + sampled


def load_candidates() -> list[dict]:
    if args_cli.controller_candidates:
        loaded = json.loads(args_cli.controller_candidates.resolve(strict=True).read_text(encoding="utf-8"))
        return loaded["candidates"] if isinstance(loaded, dict) else loaded
    if args_cli.refine_spike:
        fixed = dict(
            k_heading=0.095, k_yaw_rate=0.09, alpha=0.5, max_delta_per_step=0.0025,
            braking_scale=1.0, single_support_scale=0.5, flight_scale=0.0,
        )
        candidates = [config("parent", k_heading=0.0, k_yaw_rate=0.0), config("current")]
        candidates.append(config("safe_05", **fixed, yaw_soft_threshold=1.5, yaw_hard_threshold=4.0))
        for soft, hard in ((0.8, 1.5), (1.0, 2.0), (1.2, 2.5), (1.5, 2.5)):
            for mode in ("zero", "damping_only"):
                candidates.append(config(
                    f"refine_s{soft:.1f}_h{hard:.1f}_{mode}", **fixed,
                    yaw_soft_threshold=soft, yaw_hard_threshold=hard, hard_guard_mode=mode,
                ))
        candidates.append(config(
            "refine_s1.0_h2.0_damping_slew15", **{**fixed, "max_delta_per_step": 0.0015},
            yaw_soft_threshold=1.0, yaw_hard_threshold=2.0, hard_guard_mode="damping_only",
        ))
        return candidates
    if not args_cli.shortlist_from:
        return candidate_grid()
    report = json.loads(args_cli.shortlist_from.resolve(strict=True).read_text(encoding="utf-8"))
    names = ["parent", "current"] + report["shortlist"][: args_cli.shortlist_count]
    by_name = {candidate["name"]: candidate for candidate in report["candidates"]}
    return [{key: value for key, value in by_name[name].items() if key in config(name)} for name in dict.fromkeys(names)]


def make_actor(observations: TensorDict, device: str) -> G1CommandResidualActor:
    actor = G1CommandResidualActor(
        observations, {"actor": ["policy"]}, "actor", 37,
        hidden_dims=[256, 128, 128], activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[], train_stop_correction=False,
    ).to(device)
    checkpoint = torch.load(args_cli.checkpoint.resolve(strict=True), map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    actor.configure_stop_fixed_feedback(0.0, 0.0)
    return actor


def synchronize_initial_conditions(unwrapped, term, candidate_count: int, episodes: int) -> None:
    """Clone each parent episode state into the matching candidate environments."""
    robot = unwrapped.scene["robot"]
    origins = unwrapped.scene.env_origins
    reference = torch.arange(episodes, device=unwrapped.device, dtype=torch.long)
    targets = torch.arange(candidate_count * episodes, device=unwrapped.device, dtype=torch.long)
    sources = targets.remainder(episodes)
    local_position = robot.data.root_pos_w.torch[reference] - origins[reference]
    root_pose = torch.cat((local_position[sources] + origins[targets], robot.data.root_quat_w.torch[reference][sources]), dim=1)
    root_velocity = torch.cat((robot.data.root_lin_vel_w.torch[reference][sources], robot.data.root_ang_vel_w.torch[reference][sources]), dim=1)
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=targets)
    robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=targets)
    robot.write_joint_state_to_sim(
        robot.data.joint_pos.torch[reference][sources], robot.data.joint_vel.torch[reference][sources], env_ids=targets
    )
    for name, value in vars(term).items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != unwrapped.num_envs:
            continue
        cloned = value[reference][sources].clone()
        if name in {"target_position_w", "path_origin_w"}:
            cloned = cloned - origins[reference][sources, :2] + origins[targets, :2]
        value[targets] = cloned
    unwrapped.action_manager._action[targets] = unwrapped.action_manager.action[reference][sources]
    unwrapped.action_manager._prev_action[targets] = unwrapped.action_manager.prev_action[reference][sources]
    unwrapped.episode_length_buf[targets] = unwrapped.episode_length_buf[reference][sources]
    unwrapped.sim.forward()


def acceptable(candidate: dict, parent: dict, current: dict) -> bool:
    return (
        candidate["fall_rate"] <= min(parent["fall_rate"], current["fall_rate"])
        and candidate["saturation_failure_rate"] <= min(
            parent["saturation_failure_rate"], current["saturation_failure_rate"]
        )
        and candidate["hold_success_rate"] >= max(parent["hold_success_rate"], current["hold_success_rate"])
        and candidate["position_error_m"] <= 0.5
        and candidate["hold_end_speed_mps"] <= 0.2
        and candidate["yaw_rate_abs_p99_rps"] <= current["yaw_rate_abs_p99_rps"]
        and candidate["yaw_rate_abs_max_rps"] <= current["yaw_rate_abs_max_rps"]
    )


def rank_key(item: dict) -> tuple:
    return (
        item["fall_rate"], item["saturation_failure_rate"], -item["success_rate"],
        item["heading_p95_rad"], item["yaw_rate_abs_p99_rps"], item["yaw_rate_abs_max_rps"],
        -item["hold_success_rate"], item["position_error_m"], item["hold_end_speed_mps"],
        item["parent_action_deviation_mean"],
    )


def main() -> None:
    candidates = load_candidates()
    episodes = args_cli.episodes
    num_envs = len(candidates) * episodes
    cfg, _ = resolve_task_config("Isaac-Motion-Flat-G1-Command-Stop-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = num_envs
    cfg.seed = args_cli.env_seed
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    with launch_simulation(cfg, args_cli):
        env = gym.make("Isaac-Motion-Flat-G1-Command-Stop-Eval-v0", cfg=cfg)
        unwrapped = env.unwrapped
        device = unwrapped.device
        placeholder = TensorDict({"policy": torch.zeros(num_envs, 152, device=device)}, batch_size=[num_envs])
        actor = make_actor(placeholder, device)
        observations, _ = env.reset()
        term = unwrapped.command_manager.get_term("base_velocity")
        synchronize_initial_conditions(unwrapped, term, len(candidates), episodes)
        observations = as_tensordict(unwrapped.observation_manager.compute())
        robot = unwrapped.scene["robot"]
        contact = unwrapped.scene.sensors["contact_forces"]
        _, foot_names = robot.find_bodies(".*_ankle_roll_link")
        foot_ids = [contact.body_names.index(name) for name in foot_names]
        ankle_ids, _ = robot.find_joints(".*_ankle_.*_joint")
        all_joint_ids, _ = robot.find_joints(".*")
        controllers = []
        for candidate in candidates:
            values = {key: value for key, value in candidate.items() if key != "name"}
            controllers.append(StopFixedFeedbackController(episodes, 37, device, StopFeedbackConfig(**values)))
        accum = [{
            "stop_steps": 0, "heading": [], "yaw": [], "position": 0.0, "speed": 0.0,
            "hold": 0.0, "hold_end_speed": 0.0, "braking_end_speed": 0.0,
            "joint_sat": 0.0, "ankle_sat": 0.0, "deviation": [],
            "support_norms": {"double": [], "single": [], "flight": []},
            "spike_count": 0, "hard_count": 0, "slew_count": 0,
        } for _ in range(num_envs)]
        records = []
        finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        while not bool(finished.all().item()):
            policy = observations["policy"]
            stop_mask = term.skill_id == 1
            forces = contact.data.net_forces_w_history.torch[:, :, foot_ids, :]
            support_count = (forces.norm(dim=-1).amax(dim=1) > 1.0).sum(dim=1)
            with torch.inference_mode():
                actions = actor(observations)
            corrections = torch.zeros_like(actions)
            diagnostics = {}
            for candidate_id, controller in enumerate(controllers):
                sl = slice(candidate_id * episodes, (candidate_id + 1) * episodes)
                correction, diag = controller.step(
                    policy[sl], stop_mask[sl], term.stop_hold_progress[sl], support_count[sl]
                )
                corrections[sl] = correction
                diagnostics[candidate_id] = diag
            actions = actions + corrections
            observations, _, terminated, truncated, _ = env.step(actions)
            observations = as_tensordict(observations)
            done = terminated | truncated
            joint_ratio = robot.data.joint_vel.torch.abs() / robot.data.joint_vel_limits.torch.abs().clamp_min(1.0e-6)
            ankle_ratio = robot.data.applied_torque.torch[:, ankle_ids].abs() / robot.data.joint_effort_limits.torch[:, ankle_ids].abs().clamp_min(1.0e-6)
            for env_id in range(num_envs):
                if finished[env_id]:
                    continue
                candidate_id = env_id // episodes
                local_id = env_id % episodes
                diag = diagnostics[candidate_id]
                if bool(stop_mask[env_id].item()) and not bool(done[env_id].item()):
                    item = accum[env_id]
                    item["stop_steps"] += 1
                    item["heading"].append(abs(float(term.heading_error[env_id].item())))
                    yaw = float(robot.data.root_ang_vel_b.torch[env_id, 2].item())
                    item["yaw"].append(yaw)
                    item["position"] = float(torch.linalg.vector_norm(term.target_displacement_b[env_id]).item())
                    item["speed"] = float(torch.linalg.vector_norm(robot.data.root_lin_vel_b.torch[env_id, :2]).item())
                    hold = float(term.stop_hold_progress[env_id].item())
                    if hold <= 0.0:
                        item["braking_end_speed"] = item["speed"]
                    else:
                        item["hold_end_speed"] = item["speed"]
                    item["hold"] = max(item["hold"], hold)
                    item["joint_sat"] += float((joint_ratio[env_id, all_joint_ids] >= 0.95).float().mean().item())
                    item["ankle_sat"] += float((ankle_ratio[env_id] >= 0.95).float().mean().item())
                    norm = float(diag["feedback_norm"][local_id].item())
                    item["deviation"].append(norm)
                    support = int(diag["support_count"][local_id].item())
                    item["support_norms"]["double" if support >= 2 else "single" if support == 1 else "flight"].append(norm)
                    item["spike_count"] += int(diag["spike_guard_active"][local_id].item())
                    item["hard_count"] += int(diag["hard_guard_active"][local_id].item())
                    item["slew_count"] += int(diag["slew_limiter_active"][local_id].item())
                if bool(done[env_id].item()):
                    item = accum[env_id]
                    count = max(item["stop_steps"], 1)
                    yaw_abs = [abs(value) for value in item["yaw"]]
                    fall = bool(terminated[env_id].item() and stop_mask[env_id].item())
                    joint_sat = item["joint_sat"] / count
                    ankle_sat = item["ankle_sat"] / count
                    saturation = joint_sat > 0.05 or ankle_sat > 0.20
                    hold_success = item["hold"] >= 1.0
                    hold_speed = item["hold_end_speed"] if item["hold"] > 0.0 else item["speed"]
                    heading_mean = sum(item["heading"]) / count
                    success = not fall and not saturation and hold_success and heading_mean <= 0.12 and item["position"] <= 0.5 and hold_speed <= 0.2
                    records.append({
                        "candidate": candidates[candidate_id]["name"], "common_episode_index": local_id,
                        "fall": fall, "saturation_failure": saturation, "success": success,
                        "heading_mean_rad": heading_mean, "heading_p95_rad": percentile(item["heading"], 95),
                        "heading_max_rad": max(item["heading"], default=0.0),
                        "yaw_rate_signed_mean_rps": sum(item["yaw"]) / count,
                        "yaw_rate_abs_mean_rps": sum(yaw_abs) / count,
                        "yaw_rate_abs_p95_rps": percentile(yaw_abs, 95), "yaw_rate_abs_p99_rps": percentile(yaw_abs, 99),
                        "yaw_rate_abs_max_rps": max(yaw_abs, default=0.0),
                        "yaw_rate_over_1_5_fraction": sum(value > 1.5 for value in yaw_abs) / count,
                        "yaw_rate_over_2_5_fraction": sum(value > 2.5 for value in yaw_abs) / count,
                        "yaw_rate_over_4_0_fraction": sum(value > 4.0 for value in yaw_abs) / count,
                        "hold_success": hold_success, "position_error_m": item["position"],
                        "hold_end_speed_mps": hold_speed, "braking_end_speed_mps": item["braking_end_speed"],
                        "parent_action_deviation_mean": sum(item["deviation"]) / max(len(item["deviation"]), 1),
                        "parent_action_deviation_p95": percentile(item["deviation"], 95),
                        "parent_action_deviation_max": max(item["deviation"], default=0.0),
                        **{f"{support}_support_feedback_norm": sum(values) / max(len(values), 1) for support, values in item["support_norms"].items()},
                        "spike_guard_count": item["spike_count"], "hard_guard_count": item["hard_count"],
                        "slew_limiter_count": item["slew_count"],
                    })
                    finished[env_id] = True

        summaries = []
        boolean_fields = ("fall", "saturation_failure", "success", "hold_success")
        numeric_fields = [key for key in records[0] if key not in {"candidate", "common_episode_index", *boolean_fields}]
        for candidate in candidates:
            rows = [row for row in records if row["candidate"] == candidate["name"]]
            summary = {**candidate, "episodes": len(rows)}
            for field in boolean_fields:
                summary[field.replace("fall", "fall_rate").replace("saturation_failure", "saturation_failure_rate").replace("success", "success_rate").replace("hold_success_rate", "hold_success_rate")] = sum(row[field] for row in rows) / len(rows)
            for field in numeric_fields:
                summary[field] = sum(float(row[field]) for row in rows) / len(rows)
            summaries.append(summary)
        parent = next(item for item in summaries if item["name"] == "parent")
        current = next(item for item in summaries if item["name"] == "current")
        eligible = [item for item in summaries if item["name"] not in {"parent", "current"} and acceptable(item, parent, current)]
        eligible.sort(key=rank_key)
        shortlist = [item["name"] for item in eligible[:5]]
        report = {
            "checkpoint": str(args_cli.checkpoint.resolve()), "candidate_count": len(candidates),
            "episodes_per_candidate": episodes, "common_initial_conditions": True,
            "env_seed": args_cli.env_seed, "search_seed": args_cli.search_seed,
            "common_episode_indices": list(range(episodes)),
            "ranking": "fall,saturation,-success,heading_p95,yaw_p99,yaw_max,-hold,position,hold_speed,deviation",
            "candidates": summaries, "shortlist": shortlist,
            "best_candidate": shortlist[0] if shortlist else None,
        }
        args_cli.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        csv_path = args_cli.output.resolve().with_suffix(".episodes.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))
        env.close()


if __name__ == "__main__":
    main()
