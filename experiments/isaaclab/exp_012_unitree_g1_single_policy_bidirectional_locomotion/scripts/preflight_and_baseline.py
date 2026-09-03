"""Parent yaw-response preflight and Stage 1 diagnostic baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import minimum_jerk, wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(values):
    x = torch.cat(values)
    return {"mean": float(x.mean()), "p95": float(torch.quantile(x, .95)), "max": float(x.max())}


def main():
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 20
    cfg.seed = 20261020
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    agent_cfg.seed = 20261020
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(args.checkpoint.resolve()), strict=True, map_location=runner.device)
        policy = runner.get_inference_policy(device=runner.device)
        env = raw.unwrapped
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        dt = float(env.step_dt)

        def rollout(profile, seconds=8.0, heading=True):
            term.external_override_enabled = True
            obs, _ = wrapped.reset()
            obs = obs.to(runner.device)
            ref = None
            speed_err, heading_err, yaw_rates, actuals = [], [], [], []
            fall = torch.zeros(20, dtype=torch.bool, device=runner.device)
            flight_steps = torch.zeros(20, device=runner.device)
            sensor = env.scene.sensors["contact_forces"]
            foot_ids = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
            for step in range(round(seconds / dt)):
                t = step * dt
                vx = profile(t)
                yaw = yaw_from_quat_wxyz(robot.data.root_quat_w)
                if ref is None and t >= 1.0:
                    ref = yaw.clone()
                gate = 0.0 if t < 1.0 else float(minimum_jerk((t - 1.0) / .5))
                yaw_cmd = torch.zeros_like(yaw)
                if heading and ref is not None:
                    yaw_cmd = gate * torch.clamp(wrapped_heading_error(ref, yaw), -.1, .1)
                term.external_override[:, 0] = vx
                term.external_override[:, 1] = 0.0
                term.external_override[:, 2] = yaw_cmd
                with torch.inference_mode():
                    action = policy(obs)
                obs, _, dones, _ = wrapped.step(action)
                obs = obs.to(runner.device)
                fall |= dones.bool()
                actual = robot.data.root_lin_vel_b[:, 0]
                if t >= 2.0:
                    speed_err.append((actual - vx).abs().detach().clone())
                    actuals.append(actual.detach().clone())
                    yaw_rates.append(robot.data.root_ang_vel_b[:, 2].abs().detach().clone())
                    if ref is not None:
                        heading_err.append(wrapped_heading_error(ref, yaw).abs().detach().clone())
                if foot_ids:
                    force = sensor.data.net_forces_w_history[:, -1, foot_ids, :].norm(dim=-1)
                    flight_steps += (~(force > 5.0).any(dim=1)).float()
            return {
                "episodes": 20, "fall_rate": float(fall.float().mean()),
                "speed_mae": float(torch.cat(speed_err).mean()), "actual_speed_mean": float(torch.cat(actuals).mean()),
                "heading_p95": float(torch.quantile(torch.cat(heading_err), .95)) if heading_err else 0.0,
                "yaw_rate_p95": float(torch.quantile(torch.cat(yaw_rates), .95)),
                "flight_fraction": float((flight_steps / round(seconds / dt)).mean()),
            }

        preflight = {}
        for speed in (0.0, .6, 1.2):
            rows = []
            for yaw_command in (-.1, -.05, 0., .05, .1):
                term.external_override_enabled = True
                obs, _ = wrapped.reset()
                obs = obs.to(runner.device)
                rates, errors = [], []
                fell = torch.zeros(20, dtype=torch.bool, device=runner.device)
                for _ in range(round(5.0 / dt)):
                    term.external_override[:, 0] = speed
                    term.external_override[:, 1] = 0.0
                    term.external_override[:, 2] = yaw_command
                    with torch.inference_mode():
                        action = policy(obs)
                    obs, _, dones, _ = wrapped.step(action)
                    obs = obs.to(runner.device)
                    rates.append(robot.data.root_ang_vel_b[:, 2].detach().clone())
                    errors.append((robot.data.root_lin_vel_b[:, 0] - speed).abs().detach().clone())
                    fell |= dones.bool()
                rows.append({"command": yaw_command, "actual_yaw_rate": float(torch.cat(rates).mean()),
                             "speed_mae": float(torch.cat(errors).mean()), "fall_rate": float(fell.float().mean())})
            actual = [r["actual_yaw_rate"] for r in rows]
            signs = all((r["command"] == 0 or r["command"] * r["actual_yaw_rate"] > 0) for r in rows)
            monotonic = all(a <= b + .01 for a, b in zip(actual, actual[1:]))
            preflight[str(speed)] = {"rows": rows, "sign_pass": signs, "monotonic_pass": monotonic}
        preflight_pass = all(v["sign_pass"] and v["monotonic_pass"] and
                             max(r["fall_rate"] for r in v["rows"]) == 0 and
                             max(r["speed_mae"] for r in v["rows"]) - v["rows"][2]["speed_mae"] <= .05
                             for v in preflight.values())
        dump("heading_response_preflight.json", {
            "status": "PASS" if preflight_pass else "G1_YAW_RATE_NOT_LOCALLY_CONTROLLABLE",
            "conditions": preflight, "episodes_per_condition": 20,
        })
        if not preflight_pass:
            dump("gate.json", {"status": "FAIL", "classification": "G1_YAW_RATE_NOT_LOCALLY_CONTROLLABLE"})
            wrapped.close()
            return
        steady = {str(s): rollout(lambda t, s=s: s) for s in (0., .6, .8, 1., 1.2, 2.4, 2.6)}
        transitions = {}
        for source, target in ((0., .6), (.6, 1.2), (1.2, 2.4), (1.2, 2.6),
                               (2.4, 1.2), (2.6, 1.2), (1.2, .6), (.6, 0.)):
            def profile(t, source=source, target=target):
                if t < 2.0: return source
                if t < 3.5: return source + (target - source) * float(minimum_jerk((t - 2.0) / 1.5))
                return target
            transitions[f"{source}->{target}"] = rollout(profile, 8.0)
        seq_points = [(0., 0.), (1.5, .6), (3.5, 1.2), (6., 2.6), (10.5, 1.2), (13.5, .6), (16., 0.)]
        def sequence(t):
            for (ta, va), (tb, vb) in zip(seq_points, seq_points[1:]):
                if ta <= t < tb:
                    return va + (vb - va) * float(minimum_jerk((t - ta) / max(tb - ta, 1e-9)))
            return 0.0
        full = rollout(sequence, 18.0)
        dump("parent_baseline_results.json", {
            "status": "COMPLETE", "episodes_per_condition": 20,
            "steady": steady, "transitions": transitions, "full_sequence": full,
            "diagnostic_only": True, "checkpoint_switch": 0,
        })
        wrapped.close()


if __name__ == "__main__":
    main()
