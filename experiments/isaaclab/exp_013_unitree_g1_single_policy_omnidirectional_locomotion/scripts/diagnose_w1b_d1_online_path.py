"""Reproduce the W1B online guard path without training or optimizer construction."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_d1_yaw_translation_interference_diagnosis"
PARENT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
ITER1 = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk/checkpoints/model_1.pt"
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(HERE.parent))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks_w1b  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


class W1AVecEnv:
    def __init__(self, base):
        self.base = base
        for name in ("num_envs", "device", "max_episode_length", "num_actions"):
            setattr(self, name, getattr(base, name))
        self.gait = torch.zeros(self.num_envs, device=self.device)
        self.command = self.base.unwrapped.command_manager.get_term("base_velocity")

    @property
    def unwrapped(self):
        return self.base.unwrapped

    def seed(self, value):
        return self.base.seed(value)

    def reset(self):
        obs, extras = self.base.reset()
        return torch.cat((obs["policy"], self.gait[:, None]), -1), extras

    def get_observations(self):
        obs = self.base.get_observations()
        return torch.cat((obs["policy"], self.gait[:, None]), -1)

    def step(self, actions):
        obs, rewards, dones, extras = self.base.step(actions)
        return torch.cat((obs["policy"], self.gait[:, None]), -1), rewards, dones, extras

    def close(self):
        self.base.close()


def main():
    cfg, acfg = resolve_task_config("Isaac-Exp013-G1-W1B-YawWalk-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1024
    cfg.seed = acfg.seed = 20274021
    if args.device:
        cfg.sim.device = acfg.device = args.device
    with launch_simulation(cfg, args):
        base = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-W1B-YawWalk-v0", cfg=cfg),
                                  clip_actions=acfg.clip_actions)
        wrapped = W1AVecEnv(base)
        env, device = wrapped.unwrapped, wrapped.device
        command = wrapped.command
        command.external_override_enabled = True
        robot, sensor = env.scene["robot"], env.scene.sensors["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [robot.body_names.index(sensor.body_names[i]) for i in feet]
        conditions = [(.3, i * 22.5, 0) for i in range(16)] + [
            (.6, 0, 0), (1.2, 0, 0), (0, 0, -.3), (0, 0, .3), (.3, 0, -.3), (.3, 0, .3)]
        index = torch.arange(1024, device=device) % len(conditions)
        vx = torch.zeros(1024, device=device); vy = vx.clone(); yaw = vx.clone()
        for i, (speed, degrees, yaw_cmd) in enumerate(conditions):
            mask = index == i
            vx[mask] = speed * math.cos(math.radians(degrees))
            vy[mask] = speed * math.sin(math.radians(degrees))
            yaw[mask] = yaw_cmd
        actors = {"parent": FrozenGaitActor(PARENT).to(device).eval(),
                  "iteration1": FrozenGaitActor(ITER1).to(device).eval()}
        all_rows = []

        def evaluate(actor_name, path, warm):
            wrapped.seed(20274021)
            obs, _ = wrapped.reset()
            if warm:
                wrapped.command.set_training_iteration(1)
                wrapped.command._resample_command(torch.arange(1024, device=device))
                obs = wrapped.get_observations().to(device)
                for _ in range(24):
                    with torch.inference_mode():
                        action = actors[actor_name](obs[:, :123], obs[:, 123])
                    obs, _, _, _ = wrapped.step(action)
                    obs = obs.to(device)
                obs, _ = wrapped.reset()
            obs = obs.to(device)
            steps = round(8 / env.step_dt)
            vec = torch.zeros(1024, device=device); direction = vec.clone()
            actual_yaw = vec.clone(); yaw_error = vec.clone(); flight = vec.clone()
            fall = torch.zeros(1024, dtype=torch.bool, device=device)
            slip = fall.clone(); impact = fall.clone(); streak = torch.zeros(1024, dtype=torch.long, device=device)
            for step in range(steps):
                command.external_override[:, 0] = vx
                command.external_override[:, 1] = vy
                command.external_override[:, 2] = yaw
                if step == 0:
                    command._update_command()
                    obs = wrapped.get_observations().to(device)
                with torch.inference_mode():
                    action = actors[actor_name](obs[:, :123], obs[:, 123])
                obs, _, done, extra = wrapped.step(action)
                obs = obs.to(device)
                actual = robot.data.root_lin_vel_b[:, :2]
                az = robot.data.root_ang_vel_b[:, 2]
                vec += torch.linalg.vector_norm(actual - torch.stack((vx, vy), 1), dim=-1)
                direction += torch.atan2(torch.sin(torch.atan2(actual[:, 1], actual[:, 0]) - torch.atan2(vy, vx)),
                                         torch.cos(torch.atan2(actual[:, 1], actual[:, 0]) - torch.atan2(vy, vx))).abs() * 180 / math.pi
                actual_yaw += az; yaw_error += (az - yaw).abs()
                force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
                contact = force > 5
                flight += (contact.sum(-1) == 0).float()
                foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
                sliding = ((foot_speed > .55) & contact).any(-1)
                streak = torch.where(sliding, streak + 1, torch.zeros_like(streak))
                slip |= streak >= 5; impact |= force.amax(-1) > 3500
                timeout = extra.get("time_outs", torch.zeros_like(done)).bool()
                fall |= done.bool() & ~timeout
            vec /= steps; direction /= steps; actual_yaw /= steps; yaw_error /= steps; flight /= steps
            for i, (speed, degrees, yaw_cmd) in enumerate(conditions):
                mask = index == i
                safe = ~fall[mask] & ~slip[mask] & ~impact[mask] & (flight[mask] < .1)
                if speed == 0:
                    ok = safe & (actual_yaw[mask] * yaw_cmd > 0)
                elif yaw_cmd == 0:
                    ok = safe & (vec[mask] <= .2) & (direction[mask] <= 20) & (actual_yaw[mask].abs() <= .2)
                else:
                    ok = safe & (vec[mask] <= .25) & (direction[mask] <= 25) & \
                         (yaw_error[mask] <= .2) & (actual_yaw[mask] * yaw_cmd > 0)
                all_rows.append({
                    "path": path, "checkpoint": actor_name, "condition_index": i, "direction_deg": degrees,
                    "commanded_speed": speed, "yaw_cmd": yaw_cmd, "episodes": int(mask.sum()),
                    "success_rate": float(ok.float().mean()), "gate_pass": float(ok.float().mean()) >= .9,
                    "vector_mae": float(vec[mask].mean()), "direction_error": float(direction[mask].mean()),
                    "actual_yaw": float(actual_yaw[mask].mean()), "yaw_mae": float(yaw_error[mask].mean()),
                    "fall_rate": float(fall[mask].float().mean()), "slip_rate": float(slip[mask].float().mean()),
                    "flight_fraction": float(flight[mask].mean()),
                })

        for actor_name in ("parent", "iteration1"):
            evaluate(actor_name, "E2_online_evaluator_fresh_process", False)
            evaluate(actor_name, "E1_exact_online_contract_after_rollout_no_update", True)
        fields = list(all_rows[0])
        with (OUT / "_raw_online_path_parity.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(all_rows)
        (OUT / "_raw_online_path_parity.json").write_text(
            json.dumps({"rows": all_rows, "ppo_updates": 0, "checkpoint_writes": 0}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        wrapped.close()


if __name__ == "__main__":
    main()
