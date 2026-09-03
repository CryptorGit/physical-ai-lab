"""Isolated clean W2 early-guard evaluator."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_dynamic_omnidirectional_walk_transitions"
)
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa
import g1_omnidirectional.tasks  # noqa
from g1_omnidirectional.policy import FrozenGaitActor
from g1_omnidirectional.yaw_calibration import calibrate_yaw
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--tag", required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def minjerk(x):
    x = max(0.0, min(1.0, x))
    return x**3 * (10 - 15*x + 6*x*x)


def main():
    specs = []
    for d in (i * 22.5 for i in range(16)):
        specs.append(("ZERO", d, .3, 0., 0., 20))
    specs += [("FWD", 0., .6, 0., 0., 20), ("FWD", 0., 1.2, 0., 0., 20)]
    for d in range(0, 360, 45):
        for y in (-.3, 0., .3):
            specs.append(("STATIC", float(d), .3, y, y, 20))
    for d in range(0, 360, 45):
        specs.append(("START", float(d), 0., 0., .0, 20))
        specs.append(("STOP", float(d), .3, 0., .0, 20))
    total = sum(row[-1] for row in specs)
    cfg, ac = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = total
    cfg.episode_length_s = 10.0
    cfg.seed = 20275091
    if args.device:
        cfg.sim.device = ac.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=ac.clip_actions,
        )
        env = wrapped.unwrapped
        actor = FrozenGaitActor(args.checkpoint).to(env.device).eval()
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        cond_ids = []
        for index, spec in enumerate(specs):
            cond_ids += [index] * spec[-1]
        cond_ids = torch.tensor(cond_ids, device=env.device)
        obs, _ = wrapped.reset()
        obs = obs.to(env.device)
        n = total
        sums = {key: torch.zeros(n, device=env.device) for key in ("vec", "dir", "yawerr", "yaw", "speed")}
        endpoint_steps = torch.zeros(n, device=env.device)
        fall = torch.zeros(n, dtype=torch.bool, device=env.device)
        slip = fall.clone()
        impact = fall.clone()
        slip_streak = torch.zeros(n, dtype=torch.long, device=env.device)
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [robot.body_names.index(sensor.body_names[i]) for i in feet]
        for step in range(round(9.0 / env.step_dt)):
            t = step * env.step_dt
            vx = torch.zeros(n, device=env.device)
            vy = torch.zeros(n, device=env.device)
            yaw = torch.zeros(n, device=env.device)
            endpoint = torch.zeros(n, dtype=torch.bool, device=env.device)
            for index, (kind, deg, source_speed, source_yaw, target_yaw, _) in enumerate(specs):
                mask = cond_ids == index
                angle = math.radians(deg)
                if kind in ("ZERO", "FWD", "STATIC"):
                    speed = source_speed
                    y = target_yaw
                    endpoint[mask] = t >= 1.0
                elif kind == "START":
                    u = minjerk((t - 3.0) / 1.5)
                    speed = .3 * u if t >= 3 else 0.
                    y = 0.
                    endpoint[mask] = t >= 4.5
                else:
                    u = minjerk((t - 3.0) / 1.5)
                    speed = source_speed * (1-u) if t >= 3 else source_speed
                    y = source_yaw * (1-u) if t >= 3 else source_yaw
                    endpoint[mask] = t >= 4.5
                vx[mask] = speed * math.cos(angle)
                vy[mask] = speed * math.sin(angle)
                yaw[mask] = y
            command.external_override[:, 0] = vx
            command.external_override[:, 1] = vy
            command.external_override[:, 2] = calibrate_yaw(yaw)
            if step == 0:
                command._update_command()
                obs = wrapped.get_observations().to(env.device)
            with torch.inference_mode():
                action = actor(obs["policy"], torch.zeros(n, device=env.device))
            obs, _, done, extras = wrapped.step(action)
            obs = obs.to(env.device)
            timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
            fall |= done.bool() & ~timeout
            actual = robot.data.root_lin_vel_b[:, :2]
            actual_yaw = robot.data.root_ang_vel_b[:, 2]
            force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
            contact = force > 5
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
            slipping = ((foot_speed > .55) & contact).any(-1)
            slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
            slip |= slip_streak >= 5
            impact |= force.amax(-1) > 3500
            vector = torch.linalg.vector_norm(actual - torch.stack((vx, vy), 1), dim=-1)
            target_angle = torch.atan2(vy, vx)
            actual_angle = torch.atan2(actual[:, 1], actual[:, 0])
            direction = torch.atan2(torch.sin(actual_angle-target_angle),
                                    torch.cos(actual_angle-target_angle)).abs() * 180/math.pi
            values = {
                "vec": vector, "dir": direction, "yawerr": (actual_yaw-yaw).abs(),
                "yaw": actual_yaw, "speed": torch.linalg.vector_norm(actual, dim=-1),
            }
            for key, value in values.items():
                sums[key] += torch.where(endpoint, value, 0.)
            endpoint_steps += endpoint.float()
        rows = []
        for index, (kind, deg, source_speed, _, target_yaw, episodes) in enumerate(specs):
            mask = cond_ids == index
            den = endpoint_steps[mask].clamp_min(1)
            metrics = {key: value[mask] / den for key, value in sums.items()}
            safe = ~(fall[mask] | slip[mask] | impact[mask])
            if kind == "STOP":
                ok = safe & (metrics["speed"] <= .08) & (metrics["yaw"].abs() <= .08)
            elif kind in ("ZERO", "FWD"):
                ok = safe & (metrics["vec"] <= .20) & (metrics["dir"] <= 20) & (metrics["yaw"].abs() <= .20)
            else:
                yaw_ok = metrics["yawerr"] <= .20
                if abs(target_yaw) > .01:
                    yaw_ok &= metrics["yaw"] * target_yaw > 0
                ok = safe & (metrics["vec"] <= .25) & (metrics["dir"] <= 25) & yaw_ok
            rows.append({
                "kind": kind, "direction_deg": deg, "source_speed": source_speed,
                "yaw_target": target_yaw, "success_rate": float(ok.float().mean()),
                "fall_rate": float(fall[mask].float().mean()),
            })
        zero_pass = sum(row["success_rate"] >= .9 for row in rows if row["kind"] == "ZERO")
        static_pass = sum(row["success_rate"] >= .9 for row in rows if row["kind"] == "STATIC")
        fwd = [row for row in rows if row["kind"] == "FWD"]
        start_stop = [row for row in rows if row["kind"] in ("START", "STOP")]
        result = {
            "zero_yaw_pass_directions": zero_pass,
            "forward_0p6_success": fwd[0]["success_rate"],
            "forward_1p2_success": fwd[1]["success_rate"],
            "static_moving_turn_pass": static_pass,
            "start_stop_success": sum(r["success_rate"] for r in start_stop) / len(start_stop),
            "fall_rate": float(fall.float().mean()),
            "dangerous_slip_rate": float(slip.float().mean()),
            "impact_rate": float(impact.float().mean()),
            "rows": rows,
            "seed": 20275091,
        }
        (OUT / f"_guard_{args.tag}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        wrapped.close()


if __name__ == "__main__":
    main()
