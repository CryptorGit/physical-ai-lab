"""Short parallel deterministic +/-45/90 TURN evaluation without learning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parent.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes-per-category", type=int, choices=range(5, 11), default=5)
parser.add_argument("--saved-run-summary")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", required=True)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

RUN, TURN = 0, 2


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(round((len(values) - 1) * fraction), len(values) - 1)]


def category(angle: float) -> str:
    side = "left" if angle >= 0.0 else "right"
    degrees = 45 if abs(abs(math.degrees(angle)) - 45.0) <= 1.0 else 90
    return f"{side}_{degrees}"


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).resolve(strict=True)
    output = (REPOSITORY_ROOT / args_cli.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    count = 4 * args_cli.episodes_per_category
    task = "Isaac-Motion-Flat-G1-Command-TurnFull-Eval-v0"
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = count
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        raw_env = gym.make(task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        unwrapped = raw_env.unwrapped
        agent_cfg.device = unwrapped.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        )
        actor = runner.alg.actor
        term = unwrapped.command_manager.get_term("base_velocity")
        wrapped.reset()

        active = torch.ones(count, dtype=torch.bool, device=unwrapped.device)
        previous_skill = term.skill_id.clone()
        turn_seen = torch.zeros_like(active)
        turn_ended = torch.zeros_like(active)
        commanded = torch.zeros(count, device=unwrapped.device)
        accumulated = torch.zeros(count, device=unwrapped.device)
        final_error = torch.full((count,), float("inf"), device=unwrapped.device)
        completion_time = torch.full((count,), float("nan"), device=unwrapped.device)
        turn_start_step = torch.zeros(count, dtype=torch.long, device=unwrapped.device)
        fallen = torch.zeros_like(active)
        post_heading: list[list[float]] = [[] for _ in range(count)]
        post_lateral: list[list[float]] = [[] for _ in range(count)]
        trace: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        step = 0
        max_steps = math.ceil(float(env_cfg.episode_length_s) / float(unwrapped.step_dt)) + 5

        while active.any() and step < max_steps:
            observations = wrapped.get_observations()
            source = observations["policy"]
            current_skill = term.skill_id.clone()
            for env_id in active.nonzero(as_tuple=False).flatten().tolist():
                skill_id = int(current_skill[env_id].item())
                key = "RUN"
                if skill_id == TURN:
                    key = category(float(term.commanded_turn_angle_rad[env_id].item()))
                trace[key]["legacy_vx_col9"].append(float(source[env_id, 9].item()))
                trace[key]["legacy_vy_col10"].append(float(source[env_id, 10].item()))
                trace[key]["legacy_yaw_rate_col11"].append(float(source[env_id, 11].item()))
                trace[key]["new_heading_sin_col12"].append(float(source[env_id, 123 + 12].item()))
                trace[key]["new_heading_cos_col13"].append(float(source[env_id, 123 + 13].item()))
                trace[key]["new_col14_target_x_or_commanded_turn_angle"].append(float(source[env_id, 123 + 14].item()))
                trace[key]["new_col15_target_y_or_accumulated_yaw"].append(float(source[env_id, 123 + 15].item()))
            with torch.inference_mode():
                actions = actor(observations)
                _, _, dones, infos = wrapped.step(actions)
            step += 1
            now = term.skill_id.clone()
            entered = active & (previous_skill != TURN) & (now == TURN)
            if entered.any():
                turn_seen[entered] = True
                commanded[entered] = term.commanded_turn_angle_rad[entered]
                turn_start_step[entered] = step

            in_turn = active & (now == TURN)
            if in_turn.any():
                current_error = torch.abs(term.commanded_turn_angle_rad - term.actual_accumulated_yaw_rad)
                reached = in_turn & torch.isnan(completion_time) & (current_error <= 0.12)
                completion_time[reached] = (
                    (step - turn_start_step[reached]).float() * float(unwrapped.step_dt)
                )

            exited = active & (previous_skill == TURN) & (now == RUN)
            if exited.any():
                turn_ended[exited] = True
                accumulated[exited] = term.actual_accumulated_yaw_rad[exited]
                final_error[exited] = torch.abs(commanded[exited] - accumulated[exited])

            post = active & turn_ended & (now == RUN)
            for env_id in post.nonzero(as_tuple=False).flatten().tolist():
                post_heading[env_id].append(abs(float(term.heading_error[env_id].item())))
                post_lateral[env_id].append(abs(float(term.path_lateral_error[env_id].item())))

            done_active = active & dones.bool()
            if done_active.any():
                timeout_tensor = infos.get("time_outs") if isinstance(infos, dict) else None
                if timeout_tensor is None:
                    fallen[done_active] = True
                else:
                    fallen[done_active] = ~timeout_tensor[done_active].bool()
                active[done_active] = False
            previous_skill = now

        records = []
        for env_id in range(count):
            angle = float(commanded[env_id].item())
            heading_mean = mean(post_heading[env_id])
            lateral_p95 = percentile(post_lateral[env_id], 0.95)
            recovery = bool(
                turn_ended[env_id].item()
                and not fallen[env_id].item()
                and heading_mean <= 0.12
                and lateral_p95 <= 0.75
            )
            error = float(final_error[env_id].item())
            success = bool(error <= 0.12 and recovery)
            completion = float(completion_time[env_id].item())
            records.append({
                "episode": env_id,
                "category": category(angle),
                "commanded_turn_angle_rad": angle,
                "actual_accumulated_yaw_rad": float(accumulated[env_id].item()),
                "final_turn_angle_error_rad": error,
                "turn_completion_time_s": completion if math.isfinite(completion) else 0.0,
                "turn_success": success,
                "straight_recovery_success": recovery,
                "post_turn_heading_error_rad": heading_mean,
                "post_turn_path_lateral_error_m": lateral_p95,
                "fall": bool(fallen[env_id].item()),
            })
        write_csv(output / "episodes.csv", records)

        buckets = {}
        for name in ("left_45", "right_45", "left_90", "right_90"):
            selected = [record for record in records if record["category"] == name]
            buckets[name] = {
                "count": len(selected),
                "success_rate": mean([float(record["turn_success"]) for record in selected]),
                "commanded_turn_angle_rad": mean([record["commanded_turn_angle_rad"] for record in selected]),
                "actual_accumulated_yaw_rad": mean([record["actual_accumulated_yaw_rad"] for record in selected]),
                "final_turn_angle_error_rad": mean([record["final_turn_angle_error_rad"] for record in selected]),
                "straight_recovery_success_rate": mean([float(record["straight_recovery_success"]) for record in selected]),
                "fall_rate": mean([float(record["fall"]) for record in selected]),
            }

        saved_run_retention = None
        if args_cli.saved_run_summary:
            run_summary = json.loads(Path(args_cli.saved_run_summary).resolve(strict=True).read_text(encoding="utf-8"))
            saved_run_retention = float(run_summary["skills"]["RUN"]["success_rate"])
        command_trace = {
            key: {
                field: {
                    "mean": mean(values),
                    "min": min(values, default=0.0),
                    "max": max(values, default=0.0),
                }
                for field, values in fields.items()
            }
            for key, fields in trace.items()
        }
        full_candidate = bool(
            all(bucket["success_rate"] >= 0.90 for bucket in buckets.values())
            and mean([float(record["fall"]) for record in records]) <= 0.05
            and (saved_run_retention is None or saved_run_retention >= 0.95)
        )
        summary = {
            "checkpoint": str(checkpoint),
            "task": task,
            "deterministic": True,
            "parallel_envs": count,
            "episodes_per_category": args_cli.episodes_per_category,
            "category_results": buckets,
            "overall_turn_success_rate": mean([float(record["turn_success"]) for record in records]),
            "straight_recovery_success_rate": mean([float(record["straight_recovery_success"]) for record in records]),
            "fall_rate": mean([float(record["fall"]) for record in records]),
            "saved_run_retention_success_rate": saved_run_retention,
            "actual_episode_command_trace": command_trace,
            "decision": "full_turn_candidate_no_additional_training" if full_candidate else "additional_turn_work_required",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        raw_env.close()


if __name__ == "__main__":
    main()
