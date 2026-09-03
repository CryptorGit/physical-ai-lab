"""Closed-loop endpoint, transition, and integrated-sequence evaluation for Stage 2Q."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration"
DEFAULT_CHECKPOINT = OUT / "raw/selected_stage2q_student.pt"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("endpoints", "transitions", "final", "stochastic_final"), required=True)
parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
parser.add_argument("--gui", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

ENDPOINTS = (
    ("STAND_0P0", 0.0, 0.0, 8.0), ("WALK_0P6", .6, 0.0, 10.0),
    ("WALK_0P8", .8, 0.0, 10.0), ("WALK_1P0", 1.0, 0.0, 10.0),
    ("WALK_1P2", 1.2, 0.0, 10.0), ("RUN_1P2", 1.2, 1.0, 10.0),
    ("RUN_2P4", 2.4, 1.0, 10.0), ("RUN_2P6", 2.6, 1.0, 10.0),
)
TRANSITIONS = (
    "STAND_TO_WALK", "WALK_TO_RUN", "RUN_ACCELERATION",
    "RUN_DECELERATION", "RUN_TO_WALK", "WALK_TO_STAND",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimum_jerk(x):
    x = torch.clamp(x, 0.0, 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def ramp(t, start, duration, left, right, device):
    return left + (right - left) * minimum_jerk(torch.tensor((t - start) / duration, device=device))


class Policy(nn.Module):
    def __init__(self, path, stochastic):
        super().__init__()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["actor_state_dict"]
        self.first_base_weight = nn.Parameter(state["first_base_weight"], requires_grad=False)
        self.first_gait_column = nn.Parameter(state["first_gait_column"], requires_grad=False)
        self.first_bias = nn.Parameter(state["first_bias"], requires_grad=False)
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.hidden.load_state_dict(OrderedDict(
            (key.removeprefix("hidden."), value)
            for key, value in state.items() if key.startswith("hidden.")
        ))
        self.register_buffer("log_std_walk", state["distribution.log_std_walk"])
        self.register_buffer("log_std_run", state["distribution.log_std_run"])
        self.stochastic = stochastic

    def forward(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        mean = self.hidden(first + gait[:, None] * self.first_gait_column.T)
        if not self.stochastic:
            return mean
        log_std = (1 - gait[:, None]) * self.log_std_walk + gait[:, None] * self.log_std_run
        return mean + torch.randn_like(mean) * torch.exp(log_std)


def schedule(mode, condition, t, device):
    if mode == "endpoints":
        _, speed, gait, _ = ENDPOINTS[condition]
        return torch.tensor(speed, device=device), torch.tensor(gait, device=device), 0
    if mode == "transitions":
        name = TRANSITIONS[condition]
        if name == "STAND_TO_WALK":
            speed = ramp(t, 1, 1, 0, .6, device) if t < 2 else ramp(t, 2, 1, .6, 1.2, device) if t < 3 else torch.tensor(1.2, device=device)
            return speed if t >= 1 else torch.tensor(0., device=device), torch.tensor(0., device=device), int(t >= 4)
        if name == "WALK_TO_RUN":
            gait = ramp(t, 5, 2, 0, 1, device) if 5 <= t < 7 else torch.tensor(float(t >= 7), device=device)
            return torch.tensor(1.2, device=device), gait, int(t >= 7)
        if name == "RUN_ACCELERATION":
            speed = ramp(t, 3, 1.5, 1.2, 2.4, device) if 3 <= t < 4.5 else torch.tensor(1.2 if t < 3 else 2.4, device=device)
            return speed, torch.tensor(1., device=device), int(t >= 4.5)
        if name == "RUN_DECELERATION":
            speed = ramp(t, 3, 1.5, 2.4, 1.2, device) if 3 <= t < 4.5 else torch.tensor(2.4 if t < 3 else 1.2, device=device)
            return speed, torch.tensor(1., device=device), int(t >= 4.5)
        if name == "RUN_TO_WALK":
            gait = 1 - ramp(t, 5, 2, 0, 1, device) if 5 <= t < 7 else torch.tensor(float(t < 5), device=device)
            return torch.tensor(1.2, device=device), gait, int(t >= 7)
        speed = (
            torch.tensor(1.2, device=device) if t < 2 else
            ramp(t, 2, 1, 1.2, .6, device) if t < 3 else
            ramp(t, 3, 1, .6, 0, device) if t < 4 else torch.tensor(0., device=device)
        )
        return speed, torch.tensor(0., device=device), int(t >= 5)
    # Exact final sequence, with four diagnostic physical segments.
    if t < 3:
        return torch.tensor(0., device=device), torch.tensor(0., device=device), 0
    if t < 4:
        return ramp(t, 3, 1, 0, .6, device), torch.tensor(0., device=device), 1
    if t < 5:
        return ramp(t, 4, 1, .6, 1.2, device), torch.tensor(0., device=device), 1
    if t < 7:
        return torch.tensor(1.2, device=device), torch.tensor(0., device=device), 1
    if t < 9:
        return torch.tensor(1.2, device=device), ramp(t, 7, 2, 0, 1, device), 1
    if t < 10.5:
        return ramp(t, 9, 1.5, 1.2, 2.4, device), torch.tensor(1., device=device), 2
    if t < 14.5:
        return torch.tensor(2.4, device=device), torch.tensor(1., device=device), 2
    if t < 16:
        return ramp(t, 14.5, 1.5, 2.4, 1.2, device), torch.tensor(1., device=device), 2
    if t < 17:
        return torch.tensor(1.2, device=device), torch.tensor(1., device=device), 2
    if t < 19:
        return torch.tensor(1.2, device=device), 1 - ramp(t, 17, 2, 0, 1, device), 2
    if t < 21:
        return torch.tensor(1.2, device=device), torch.tensor(0., device=device), 3
    if t < 22:
        return ramp(t, 21, 1, 1.2, .6, device), torch.tensor(0., device=device), 3
    if t < 23:
        return ramp(t, 22, 1, .6, 0, device), torch.tensor(0., device=device), 3
    return torch.tensor(0., device=device), torch.tensor(0., device=device), 4


def main():
    checkpoint = Path(args.checkpoint).resolve()
    stochastic = args.mode == "stochastic_final"
    mode = "final" if stochastic else args.mode
    episodes = 1 if args.gui else 50 if stochastic else 100
    condition_count = len(ENDPOINTS) if mode == "endpoints" else len(TRANSITIONS) if mode == "transitions" else 1
    count = episodes * condition_count
    duration = 10.0 if mode != "final" else 28.0
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = count
    cfg.episode_length_s = duration
    cfg.seed = 20269031 if not stochastic else 20269032
    agent_cfg.seed = cfg.seed
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        device = runner.device
        policy = Policy(checkpoint, stochastic).to(device).eval()
        env = wrapped.unwrapped
        robot = env.scene["robot"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        sensor = env.scene.sensors["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [next(i for i, name in enumerate(robot.body_names) if name == sensor.body_names[j]) for j in feet]
        condition = torch.arange(count, device=device) // episodes
        obs, _ = wrapped.reset()
        obs = obs.to(device)
        steps = round(duration / float(env.step_dt))
        fallen = torch.zeros(count, dtype=torch.bool, device=device)
        speed_error = torch.zeros(count, device=device)
        actual_speed_sum = torch.zeros(count, device=device)
        flight_streak = torch.zeros(count, dtype=torch.long, device=device)
        segment_steps = torch.zeros((5, count), device=device)
        segment_flight = torch.zeros_like(segment_steps)
        segment_events = torch.zeros((5, count), dtype=torch.long, device=device)
        segment_safe = torch.zeros_like(segment_events)
        segment_alt = torch.zeros_like(segment_events)
        segment_double = torch.zeros_like(segment_steps)
        segment_speed_abs = torch.zeros_like(segment_steps)
        last_landing = torch.full((count,), -1, dtype=torch.long, device=device)
        heading_trace = []
        slip_streak = torch.zeros(count, dtype=torch.long, device=device)
        dangerous_slip = torch.zeros(count, dtype=torch.bool, device=device)
        impact = torch.zeros_like(dangerous_slip)
        saturation_streak = torch.zeros_like(slip_streak)
        saturation = torch.zeros_like(dangerous_slip)
        completion_fires = torch.zeros(count, dtype=torch.long, device=device)
        reward_term = env.reward_manager.get_term_cfg("safe_periodic_flight").func
        reference_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        guides = None
        if args.gui:
            try:
                import omni.isaac.debug_draw._debug_draw as debug_draw
                guides = debug_draw.acquire_debug_draw_interface()
                starts, ends, colors, widths = [], [], [], []
                for x in range(-5, 71):
                    starts.append((float(x), -1.5, .012))
                    ends.append((float(x), 1.5, .012))
                    colors.append((.25, .72, .90, 1.) if x % 5 == 0 else (.55, .60, .64, 1.))
                    widths.append(5. if x % 5 == 0 else 2.)
                guides.draw_lines(starts, ends, colors, widths)
            except Exception as error:
                print(f"[FinalSequence] floor-guide fallback: {error}")
        for step in range(steps):
            t = step * float(env.step_dt)
            speeds = torch.empty(count, device=device)
            gaits = torch.empty(count, device=device)
            segments = torch.empty(count, dtype=torch.long, device=device)
            for c in range(condition_count):
                speed, gait, segment = schedule(mode, c, t, device)
                mask = condition == c
                speeds[mask], gaits[mask], segments[mask] = speed, gait, segment
            command.external_override[:, 0] = speeds
            command.external_override[:, 1:] = 0
            if step == 0:
                obs = wrapped.get_observations().to(device)
            with torch.inference_mode():
                action = policy(obs["policy"], gaits)
            obs, _, dones, extras = wrapped.step(action)
            obs = obs.to(device)
            timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
            fallen |= dones.bool() & ~timeout
            actual = robot.data.root_lin_vel_b[:, 0]
            speed_error += (actual - speeds).abs()
            actual_speed_sum += actual.abs()
            forces = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
            contacts = forces > 5
            in_flight = contacts.sum(-1) == 0
            previous = flight_streak.clone()
            takeoff = in_flight & (flight_streak == 0)
            flight_streak = torch.where(in_flight, flight_streak + 1, torch.zeros_like(flight_streak))
            landing = ~in_flight & (previous > 0)
            single = landing & (contacts.sum(-1) == 1)
            foot = contacts.long().argmax(-1)
            safe = single & (previous >= 2) & (previous <= 8)
            alt = safe & (last_landing >= 0) & (foot != last_landing)
            last_landing[single] = foot[single]
            for segment in range(5):
                mask = segments == segment
                segment_steps[segment, mask] += 1
                segment_flight[segment, mask] += in_flight[mask].float()
                segment_events[segment, mask] += takeoff[mask].long()
                segment_safe[segment, mask] += safe[mask].long()
                segment_alt[segment, mask] += alt[mask].long()
                segment_double[segment, mask] += (contacts[mask].sum(-1) == 2).float()
                segment_speed_abs[segment, mask] += actual[mask].abs()
            completion_fires += (reward_term.last_raw_reward >= 1).long()
            heading_trace.append(wrapped_heading_error(reference_yaw, yaw_from_quat_wxyz(robot.data.root_quat_w)).abs().cpu())
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
            slipping = ((foot_speed > .55) & contacts).any(-1)
            slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
            dangerous_slip |= slip_streak >= 5
            impact |= forces.amax(-1) > 3500
            limits = robot.data.joint_vel_limits
            if limits.ndim == 3:
                limits = limits[..., 1].abs()
            saturated_now = (robot.data.joint_vel.abs() / limits.clamp_min(1e-6) > .95).any(-1)
            saturation_streak = torch.where(saturated_now, saturation_streak + 1, torch.zeros_like(saturation_streak))
            saturation |= saturation_streak >= 5
            if args.gui and step % 25 == 0:
                contact_text = "DOUBLE" if bool(contacts[0].all()) else "FLIGHT" if not bool(contacts[0].any()) else "LEFT" if bool(contacts[0, 0]) else "RIGHT"
                gait_state = "RUN" if bool(in_flight[0]) and float(gaits[0]) > .5 else "WALK" if float(speeds[0]) > .08 else "STAND"
                print(" | ".join((
                    "EXP_012 G1 FINAL SINGLE-POLICY SEQUENCE", "MODE FinalSequence",
                    f"TARGET SPEED {float(speeds[0]):.2f}", f"ACTUAL SPEED {float(actual[0]):+.3f}",
                    f"GAIT COMMAND {float(gaits[0]):.3f}", f"GAIT STATE {gait_state}",
                    f"FLIGHT {bool(in_flight[0])}", f"CONTACTS {contact_text}",
                    f"HEADING {float(heading_trace[-1][0]):.3f}", f"SLIP {bool(dangerous_slip[0])}",
                    f"IMPACT {bool(impact[0])}", f"SATURATION {bool(saturation[0])}",
                    f"FALL {bool(fallen[0])}", f"CHECKPOINT SHA {sha(checkpoint)[:12]}",
                    "UNIQUE CHECKPOINT COUNT 1",
                )))
                root = robot.data.root_pos_w[0].detach().cpu()
                env.sim.set_camera_view(
                    eye=(float(root[0] - 3.8), float(root[1] - 4.0), 2.3),
                    target=(float(root[0] + .5), float(root[1]), .9),
                )
        heading = torch.stack(heading_trace)
        rows = []
        for env_id in range(count):
            c = int(condition[env_id])
            name = ENDPOINTS[c][0] if mode == "endpoints" else TRANSITIONS[c] if mode == "transitions" else "FINAL_SEQUENCE"
            segment_metrics = {}
            for segment in range(5):
                n = max(float(segment_steps[segment, env_id]), 1)
                flight_fraction = float(segment_flight[segment, env_id] / n)
                periodic = (
                    int(segment_events[segment, env_id]) >= 4 and
                    int(segment_safe[segment, env_id]) >= 3 and
                    int(segment_alt[segment, env_id]) >= 3
                )
                segment_metrics[str(segment)] = {
                    "flight_fraction": flight_fraction, "periodic": periodic,
                    "double_support_fraction": float(segment_double[segment, env_id] / n),
                    "speed_abs_mean": float(segment_speed_abs[segment, env_id] / n),
                }
            if mode == "endpoints":
                speed, gait = ENDPOINTS[c][1], ENDPOINTS[c][2]
                metric = segment_metrics["0"]
                if speed == 0:
                    success = metric["flight_fraction"] == 0 and metric["double_support_fraction"] >= .95
                    gait_label = "STAND"
                elif gait == 0:
                    success = metric["flight_fraction"] < .10
                    gait_label = "WALK_LIKE" if success else "PERIODIC_RUNNING" if metric["periodic"] else "IRREGULAR"
                else:
                    success = metric["periodic"]
                    gait_label = "PERIODIC_RUNNING" if success else "IRREGULAR"
            elif mode == "transitions":
                metric = segment_metrics["1"]
                target_run = name in ("WALK_TO_RUN", "RUN_ACCELERATION", "RUN_DECELERATION")
                target_stand = name == "WALK_TO_STAND"
                success = metric["periodic"] if target_run else (
                    metric["flight_fraction"] == 0 and metric["double_support_fraction"] >= .95
                    if target_stand else metric["flight_fraction"] < .10
                )
                gait_label = "TARGET_ACQUIRED" if success else "TARGET_FAIL"
            else:
                stand0 = segment_metrics["0"]["flight_fraction"] == 0
                walk0 = segment_metrics["1"]["flight_fraction"] < .10
                run = segment_metrics["2"]["periodic"]
                walk1 = segment_metrics["3"]["flight_fraction"] < .10
                stand1 = (
                    segment_metrics["4"]["flight_fraction"] == 0 and
                    segment_metrics["4"]["double_support_fraction"] >= .95 and
                    segment_metrics["4"]["speed_abs_mean"] <= .08
                )
                success = stand0 and walk0 and run and walk1 and stand1 and not bool(fallen[env_id])
                gait_label = "SEQUENCE_COMPLETE" if success else "SEQUENCE_FAIL"
            rows.append({
                "condition": name, "episode": env_id % episodes, "success": bool(success),
                "gait_classification": gait_label, "fall": bool(fallen[env_id]),
                "speed_mae": float(speed_error[env_id] / steps),
                "final_speed": float(robot.data.root_lin_vel_b[env_id, 0].abs()),
                "heading_p95": float(torch.quantile(heading[:, env_id], .95)),
                "dangerous_slip": bool(dangerous_slip[env_id]), "impact_failure": bool(impact[env_id]),
                "long_dwell_saturation": bool(saturation[env_id]),
                "completion_reward_fires": int(completion_fires[env_id]),
                "segment_metrics": json.dumps(segment_metrics, sort_keys=True),
            })
        grouped = {}
        for row in rows:
            grouped.setdefault(row["condition"], []).append(row)
        summary = {}
        for name, values in grouped.items():
            summary[name] = {
                "episodes": len(values), "success_rate": sum(row["success"] for row in values) / len(values),
                "fall_rate": sum(row["fall"] for row in values) / len(values),
                "speed_mae": sum(row["speed_mae"] for row in values) / len(values),
                "final_speed_mean": sum(row["final_speed"] for row in values) / len(values),
                "heading_p95_mean": sum(row["heading_p95"] for row in values) / len(values),
                "dangerous_slip_rate": sum(row["dangerous_slip"] for row in values) / len(values),
                "impact_failure_rate": sum(row["impact_failure"] for row in values) / len(values),
                "long_dwell_saturation_rate": sum(row["long_dwell_saturation"] for row in values) / len(values),
                "completion_reward_fires": sum(row["completion_reward_fires"] for row in values),
            }
        prefix = "candidate_stochastic_sequence" if stochastic else (
            "closed_loop_endpoint" if mode == "endpoints" else "transition_results" if mode == "transitions" else "final_integrated_sequence"
        )
        with (OUT / f"{prefix}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (OUT / f"{prefix}.json").write_text(json.dumps({
            "mode": mode, "stochastic": stochastic, "checkpoint_sha256": sha(checkpoint),
            "summary": summary, "teacher_calls": 0, "expert_calls": 0, "router_calls": 0,
            "checkpoint_switches": 0, "action_blends": 0, "unique_checkpoint_count": 1,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if guides is not None:
            guides.clear_lines()
        wrapped.close()


if __name__ == "__main__":
    main()
