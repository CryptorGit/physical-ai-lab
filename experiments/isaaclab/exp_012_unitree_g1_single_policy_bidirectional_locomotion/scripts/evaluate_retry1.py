"""Validation, fixed checkpoint selection, and formal evaluation for retry 1.

The evaluator is deliberately controller-free: the policy observes a zero yaw-rate
command at every step.  Validation and formal evaluation use disjoint fixed seeds.
"""

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
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import minimum_jerk, wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=("validation", "formal"), required=True)
parser.add_argument("--output", type=Path, required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

CHECKPOINT_ITERS = (0, 1, 10, 25, 50, 75, 100, 150, 200, 250, 300)
STEADY = (0.0, 0.6, 0.8, 1.0, 1.2, 2.4, 2.6)
TRANSITIONS = ((0.0, 0.6), (0.6, 1.2), (1.2, 2.4), (1.2, 2.6),
               (2.4, 1.2), (2.6, 1.2), (1.2, 0.6), (0.6, 0.0))


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def q95(x: torch.Tensor) -> float:
    return float(torch.quantile(x.float(), 0.95)) if x.numel() else 0.0


def command_profile(kind: str, a: float, b: float, t: float, formal: bool) -> float:
    if kind == "steady":
        return a
    if kind == "transition":
        if t < 2.0:
            return a
        if t < 3.5:
            return a + (b - a) * float(minimum_jerk((t - 2.0) / 1.5))
        return b
    if not formal:
        points = ((0.0, 0.0), (1.5, 0.6), (3.5, 1.2), (6.0, 2.6),
                  (10.5, 1.2), (13.5, 0.6), (16.0, 0.0), (18.0, 0.0))
    else:
        # Hold durations are 2 s for walk/stand, 3 s for run/final stand;
        # walk ramps are 1 s and run-boundary ramps are 1.5 s.
        points = (
            (0.0, 0.0), (2.0, 0.0), (3.0, 0.6), (5.0, 0.6),
            (6.0, 1.2), (8.0, 1.2), (9.5, 2.4), (12.5, 2.4),
            (14.0, 2.6), (17.0, 2.6), (18.5, 2.4), (21.5, 2.4),
            (23.0, 1.2), (25.0, 1.2), (26.0, 0.6), (28.0, 0.6),
            (29.0, 0.0), (32.0, 0.0),
        )
    for (ta, va), (tb, vb) in zip(points, points[1:]):
        if ta <= t < tb:
            # Equal endpoints are holds; all other intervals are smooth ramps.
            return va if va == vb else va + (vb - va) * float(minimum_jerk((t - ta) / (tb - ta)))
    return points[-1][1]


def condition_specs(episodes: int, formal: bool) -> list[dict]:
    specs = []
    for speed in STEADY:
        specs.append({"name": f"steady_{speed:.1f}", "kind": "steady", "a": speed, "b": speed,
                      "horizon": 8.0, "episodes": episodes})
    for source, target in TRANSITIONS:
        specs.append({"name": f"transition_{source:.1f}_to_{target:.1f}", "kind": "transition",
                      "a": source, "b": target, "horizon": 8.0, "episodes": episodes})
    specs.append({"name": "integrated_sequence", "kind": "sequence", "a": 0.0, "b": 0.0,
                  "horizon": 32.0 if formal else 18.0, "episodes": episodes})
    return specs


def episode_label(target: float, fallen: bool, speed_mae: float, flight_events: int,
                  safe_flights: int, alternating: int) -> str:
    if fallen:
        return "FALL"
    periodic = flight_events >= 4 and safe_flights >= 3 and alternating >= 3
    if target >= 2.3:
        return "PERIODIC_RUNNING" if periodic else ("ISOLATED_FLIGHT" if flight_events else "WALK_LIKE")
    if target == 0.0 and speed_mae <= 0.08 and flight_events == 0:
        return "STAND"
    return "PERIODIC_RUNNING" if periodic else ("WALK_LIKE" if speed_mae <= 0.25 else "IRREGULAR")


def evaluate_checkpoint(wrapped, runner, checkpoint: Path, specs: list[dict], episodes: int,
                        seed: int, formal: bool) -> tuple[list[dict], dict]:
    runner.load(str(checkpoint.resolve()), strict=True, map_location=runner.device)
    policy = runner.get_inference_policy(device=runner.device)
    env = wrapped.unwrapped
    robot = env.scene["robot"]
    term = env.command_manager.get_term("base_velocity")
    term.external_override_enabled = True
    dt = float(env.step_dt)
    n = len(specs) * episodes
    device = runner.device
    spec_index = torch.arange(n, device=device) // episodes
    horizon = torch.tensor([specs[int(i)]["horizon"] for i in spec_index.cpu()], device=device)

    obs, _ = wrapped.reset()
    obs = obs.to(device)
    ref_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
    fallen = torch.zeros(n, dtype=torch.bool, device=device)
    steps = torch.zeros(n, device=device)
    speed_abs_sum = torch.zeros(n, device=device)
    speed_signed_sum = torch.zeros(n, device=device)
    heading_values: list[list[float]] = [[] for _ in range(n)]
    yaw_rate_sum = torch.zeros(n, device=device)
    speed_sum = torch.zeros(n, device=device)
    flight_steps = torch.zeros(n, device=device)
    double_support_steps = torch.zeros(n, device=device)
    flight_events = torch.zeros(n, dtype=torch.long, device=device)
    safe_flights = torch.zeros(n, dtype=torch.long, device=device)
    alternating_landings = torch.zeros(n, dtype=torch.long, device=device)
    flight_streak = torch.zeros(n, dtype=torch.long, device=device)
    flight_event_eligible = torch.zeros(n, dtype=torch.bool, device=device)
    last_landing = torch.full((n,), -1, dtype=torch.long, device=device)
    dangerous_slip_streak = torch.zeros(n, dtype=torch.long, device=device)
    dangerous_slip = torch.zeros(n, dtype=torch.bool, device=device)
    impact_failure = torch.zeros(n, dtype=torch.bool, device=device)
    saturation_streak = torch.zeros(n, dtype=torch.long, device=device)
    long_saturation = torch.zeros(n, dtype=torch.bool, device=device)
    tilt_values: list[list[float]] = [[] for _ in range(n)]
    segment_error: dict[str, torch.Tensor] = {}
    segment_count: dict[str, torch.Tensor] = {}

    sensor = env.scene.sensors["contact_forces"]
    sensor_feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
    robot_feet = []
    for idx in sensor_feet:
        name = sensor.body_names[idx]
        matches = [i for i, body_name in enumerate(robot.body_names) if body_name == name]
        robot_feet.append(matches[0] if matches else -1)
    robot_feet = [x for x in robot_feet if x >= 0]

    max_steps = math.ceil(max(x["horizon"] for x in specs) / dt)
    for step in range(max_steps):
        t = step * dt
        command = torch.zeros(n, device=device)
        quality = torch.zeros(n, dtype=torch.bool, device=device)
        for index, spec in enumerate(specs):
            mask = spec_index == index
            value = command_profile(spec["kind"], spec["a"], spec["b"], t, formal)
            command[mask] = value
            if spec["kind"] == "steady":
                quality[mask] = t >= 2.0
            elif spec["kind"] == "transition":
                quality[mask] = t >= 4.0
            else:
                quality[mask] = t >= 1.0
        active = t < horizon
        quality &= active
        term.external_override[:, 0] = command
        term.external_override[:, 1] = 0.0
        term.external_override[:, 2] = 0.0
        if step == 0:
            obs = wrapped.get_observations().to(device)
        with torch.inference_mode():
            action = policy(obs)
        obs, _, dones, _ = wrapped.step(action)
        obs = obs.to(device)
        fallen |= dones.bool() & active
        yaw = yaw_from_quat_wxyz(robot.data.root_quat_w)
        actual = robot.data.root_lin_vel_b[:, 0]
        yaw_rate = robot.data.root_ang_vel_b[:, 2]
        gravity = robot.data.projected_gravity_b
        tilt = torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0))
        abs_error = (actual - command).abs()
        signed_error = actual - command
        qf = quality.float()
        steps += qf
        speed_abs_sum += abs_error * qf
        speed_signed_sum += signed_error * qf
        speed_sum += actual * qf
        yaw_rate_sum += yaw_rate * qf
        heading = wrapped_heading_error(ref_yaw, yaw).abs()
        for env_id in torch.where(quality)[0].tolist():
            heading_values[env_id].append(float(heading[env_id]))
            tilt_values[env_id].append(float(tilt[env_id]))

        forces = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].norm(dim=-1)
        contacts = forces > 5.0
        in_flight = contacts.sum(dim=1) == 0
        prior_streak = flight_streak.clone()
        landed = (~in_flight) & (prior_streak > 0) & active
        flight_started = in_flight & (flight_streak == 0) & active
        flight_event_eligible = torch.where(flight_started, quality, flight_event_eligible)
        flight_events += (flight_started & quality).long()
        flight_streak = torch.where(in_flight & active, flight_streak + 1, torch.zeros_like(flight_streak))
        eligible_landing = landed & flight_event_eligible & quality
        safe_flights += (eligible_landing & (prior_streak >= 2) & (prior_streak <= 8)).long()
        landing_side = contacts.long().argmax(dim=1)
        alternating_landings += (
            eligible_landing & (last_landing >= 0) & (landing_side != last_landing)
        ).long()
        last_landing = torch.where(eligible_landing, landing_side, last_landing)
        flight_event_eligible = torch.where(landed, torch.zeros_like(flight_event_eligible),
                                            flight_event_eligible)
        flight_steps += in_flight.float() * qf
        double_support_steps += (contacts.sum(dim=1) >= 2).float() * qf
        impact_failure |= (forces.amax(dim=1) > 3500.0) & quality

        if robot_feet:
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
            slipping = ((foot_speed > 0.55) & contacts[:, :len(robot_feet)]).any(dim=1) & quality
            dangerous_slip_streak = torch.where(slipping, dangerous_slip_streak + 1,
                                                torch.zeros_like(dangerous_slip_streak))
            dangerous_slip |= dangerous_slip_streak >= 5

        velocity_limits = getattr(robot.data, "joint_vel_limits", None)
        if velocity_limits is not None:
            if velocity_limits.ndim == 3:
                velocity_limits = velocity_limits[..., 1].abs()
            saturated = (robot.data.joint_vel.abs() / torch.clamp(velocity_limits, min=1e-6) > 0.95).any(dim=1)
            saturation_streak = torch.where(saturated & quality, saturation_streak + 1,
                                            torch.zeros_like(saturation_streak))
            long_saturation |= saturation_streak >= 5

        if formal:
            # Segment-level tracking for the integrated sequence.
            seq_mask = spec_index == len(specs) - 1
            if seq_mask.any() and t < specs[-1]["horizon"]:
                key = f"{command_profile('sequence', 0.0, 0.0, t, True):.1f}"
                segment_error.setdefault(key, torch.zeros(n, device=device))
                segment_count.setdefault(key, torch.zeros(n, device=device))
                segment_error[key] += abs_error * seq_mask.float()
                segment_count[key] += seq_mask.float()

    records = []
    checkpoint_hash = sha(checkpoint)
    for env_id in range(n):
        spec = specs[int(spec_index[env_id])]
        count = max(float(steps[env_id]), 1.0)
        speed_mae = float(speed_abs_sum[env_id] / count)
        target = float(spec["b"] if spec["kind"] == "transition" else spec["a"])
        if spec["kind"] == "sequence":
            target = 0.0
        gait = episode_label(target, bool(fallen[env_id]), speed_mae, int(flight_events[env_id]),
                             int(safe_flights[env_id]), int(alternating_landings[env_id]))
        heading_tensor = torch.tensor(heading_values[env_id])
        tilt_tensor = torch.tensor(tilt_values[env_id])
        record = {
            "condition": spec["name"], "kind": spec["kind"], "episode": env_id % episodes,
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "target_speed": target, "fall": bool(fallen[env_id]), "speed_mae": speed_mae,
            "actual_speed_mean": float(speed_sum[env_id] / count),
            "signed_speed_error": float(speed_signed_sum[env_id] / count),
            "yaw_bias": float(yaw_rate_sum[env_id] / count),
            "heading_p95": q95(heading_tensor), "tilt_p95": q95(tilt_tensor),
            "flight_fraction": float(flight_steps[env_id] / count),
            "flight_events": int(flight_events[env_id]), "safe_flight_events": int(safe_flights[env_id]),
            "alternating_landings": int(alternating_landings[env_id]),
            "final_double_support_fraction": float(double_support_steps[env_id] / count),
            "dangerous_slip": bool(dangerous_slip[env_id]), "impact_failure": bool(impact_failure[env_id]),
            "long_dwell_saturation": bool(long_saturation[env_id]), "gait": gait,
            "yaw_command_nonzero": 0, "checkpoint_switch": 0, "expert_action_calls": 0,
        }
        records.append(record)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)
    summaries = {}
    for name, rows in grouped.items():
        spec = next(x for x in specs if x["name"] == name)
        target = float(spec["b"] if spec["kind"] == "transition" else spec["a"])
        if spec["kind"] == "sequence":
            target = 0.0
        periodic = sum(
            x["gait"] == "PERIODIC_RUNNING" and not x["fall"] and x["heading_p95"] <= 0.12
            for x in rows
        ) / len(rows)
        walk_like = sum(
            x["gait"] in ("WALK_LIKE", "STAND") and not x["fall"] and x["speed_mae"] <= 0.20
            and x["heading_p95"] <= 0.12
            for x in rows
        ) / len(rows)
        # Reset settling can contain a one-step pre-quality-window contact gap.
        # The formal STAND flight gate is evaluated in the quality window, so use
        # flight_fraction rather than the all-episode onset counter here.
        stand = sum(
            not x["fall"] and abs(x["actual_speed_mean"]) <= 0.05 and x["speed_mae"] <= 0.05
            and x["heading_p95"] <= 0.12 and x["tilt_p95"] <= 0.15
            and x["flight_fraction"] == 0.0 and x["final_double_support_fraction"] >= 0.95
            for x in rows
        ) / len(rows)
        fall_rate = sum(x["fall"] for x in rows) / len(rows)
        speed_mae = sum(x["speed_mae"] for x in rows) / len(rows)
        heading_p95 = q95(torch.tensor([x["heading_p95"] for x in rows]))
        slip_rate = sum(x["dangerous_slip"] for x in rows) / len(rows)
        saturation_rate = sum(x["long_dwell_saturation"] for x in rows) / len(rows)
        impact_rate = sum(x["impact_failure"] for x in rows) / len(rows)
        if spec["kind"] == "steady":
            if target == 0.0:
                success = stand
            elif target >= 2.3:
                success = periodic
            else:
                success = walk_like
        elif spec["kind"] == "transition":
            if target == 0.0:
                success = sum((not x["fall"]) and x["actual_speed_mean"] <= 0.08 and
                              x["final_double_support_fraction"] >= 0.95 for x in rows) / len(rows)
            elif target >= 2.3:
                success = periodic
            else:
                success = walk_like
        else:
            success = sum((not x["fall"]) and x["speed_mae"] <= 0.25 for x in rows) / len(rows)
        summaries[name] = {
            "episodes": len(rows), "success_rate": success, "periodic_running_rate": periodic,
            "walk_like_rate": walk_like, "stand_rate": stand, "fall_rate": fall_rate,
            "speed_mae": speed_mae, "heading_p95": heading_p95,
            "dangerous_slip_rate": slip_rate, "long_dwell_saturation_rate": saturation_rate,
            "impact_failure_rate": impact_rate, "yaw_bias": sum(x["yaw_bias"] for x in rows) / len(rows),
        }
    return records, summaries


def validation_score(summary: dict) -> tuple:
    seq = summary["integrated_sequence"]
    stand = summary["steady_0.0"]
    walks = [summary[f"steady_{x:.1f}"] for x in (0.6, 0.8, 1.0, 1.2)]
    runs = [summary[f"steady_{x:.1f}"] for x in (2.4, 2.6)]
    down = [summary["transition_2.4_to_1.2"], summary["transition_2.6_to_1.2"]]
    stop = summary["transition_0.6_to_0.0"]
    all_rows = list(summary.values())
    return (
        int(seq["success_rate"] >= 0.9),
        stand["success_rate"],
        sum(x["success_rate"] for x in walks) / 4,
        sum(x["success_rate"] for x in runs) / 2,
        sum(x["success_rate"] for x in down) / 2,
        stop["success_rate"],
        -sum(x["fall_rate"] for x in all_rows),
        -sum(x["heading_p95"] for x in all_rows),
        -sum(x["long_dwell_saturation_rate"] for x in all_rows),
        -sum(x["dangerous_slip_rate"] + x["impact_failure_rate"] for x in all_rows),
    )


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = 10 if args.phase == "validation" else 50
    seed = 20262021 if args.phase == "validation" else 20263021
    formal = args.phase == "formal"
    specs = condition_specs(episodes, formal)
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = len(specs) * episodes
    cfg.seed = seed
    cfg.episode_length_s = 35.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    agent_cfg.seed = seed
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    import importlib.metadata
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        if not formal:
            rows, results = [], {}
            for iteration in CHECKPOINT_ITERS:
                checkpoint_name = "model_initial.pt" if iteration == 0 else f"model_{iteration}.pt"
                checkpoint = args.output / "checkpoints" / checkpoint_name
                records, summary = evaluate_checkpoint(wrapped, runner, checkpoint, specs, episodes, seed, False)
                score = validation_score(summary)
                results[str(iteration)] = {"iteration": iteration, "checkpoint": str(checkpoint),
                                           "sha256": sha(checkpoint), "score": list(score),
                                           "conditions": summary}
                rows.extend({"iteration": iteration, **record} for record in records)
                print(f"[validation] iteration={iteration} score={score}", flush=True)
            best_iteration = max(CHECKPOINT_ITERS, key=lambda x: tuple(results[str(x)]["score"]))
            selected = results[str(best_iteration)]
            selected["selection_precedence"] = [
                "full-sequence hard-gate pass count", "STAND retention", "WALK retention",
                "RUN periodicity", "RUN_TO_WALK success", "WALK_TO_STAND success",
                "fall", "heading", "long-dwell saturation", "slip / impact",
            ]
            with (args.output / "validation_checkpoint_results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            dump(args.output / "validation_checkpoint_results.json", results)
            dump(args.output / "selected_checkpoint.json", selected)
        else:
            selected = json.loads((args.output / "selected_checkpoint.json").read_text(encoding="utf-8"))
            checkpoint = Path(selected["checkpoint"])
            records, summary = evaluate_checkpoint(wrapped, runner, checkpoint, specs, episodes, seed, True)
            dump(args.output / "formal_all_episode_records.json", records)
            dump(args.output / "formal_all_summary.json", summary)
        wrapped.close()


if __name__ == "__main__":
    main()
