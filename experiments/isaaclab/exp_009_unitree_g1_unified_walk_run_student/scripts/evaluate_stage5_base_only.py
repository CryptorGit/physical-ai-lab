"""Small closed-loop diagnostic for existing Stage 5 base candidates.

All candidates start from the same valid WALK@1.2 occupancy.  This is a
diagnostic cross-regime stress test, not a formal capability evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import gymnasium as gym

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage5_base_action_manifold_compatibility"
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
]

import g1_command_skills.tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
import isaaclab_tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_run_expert, load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_run_observation  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def minimum_jerk(x):
    x = x.clamp(0, 1)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def main() -> None:
    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    candidates = ["A_WALK", "B_RUN_INTERNAL_BASE", "C_RUN_FULL"]
    commands = [0.6, 1.2, 2.4, 2.8]
    episodes = 20
    assignments = [(candidate, command) for candidate in candidates for command in commands for _ in range(episodes)]
    n = len(assignments)
    task_cfg.scene.num_envs = n
    task_cfg.seed = 20270201
    task_cfg.episode_length_s = 24.0
    task_cfg.sim.device = "cuda:0"
    args.device = "cuda:0"
    OUT.mkdir(parents=True, exist_ok=True)
    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg),
            clip_actions=agent_cfg.clip_actions,
        )
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        dt = float(env.step_dt)
        walk = load_walk_expert(
            REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/"
            "2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
            device=device,
        )
        run = load_run_expert(
            REPO / "logs/rsl_rl/physical_ai_g1_command_skills/"
            "2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt",
            device=device,
        )
        stand = load_walk_expert(
            REPO / "logs/rsl_rl/physical_ai_g1_flat_run/"
            "2026-07-17_21-40-39_stage2_1024_750/model_4246.pt", device=device,
        )
        stw = load_walk_expert(
            REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/"
            "2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt",
            device=device,
        )
        for model in (walk.actor, run.actor, stand.actor, stw.actor):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        joints, _ = robot.find_joints(".*")
        targets = torch.tensor([item[1] for item in assignments], device=device)
        candidate_id = torch.tensor([candidates.index(item[0]) for item in assignments], device=device)

        def observe():
            legacy = wrapped.get_observations()["policy"]
            return canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)

        def motion(speed, heading, gain=0.8):
            error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
            yaw = (gain * error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-1.5, 1.5)
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0] = speed
            command_term.vel_command_b[:, 2] = yaw
            return MotionCommand(speed, heading, target_yaw_rate_radps=yaw), error

        wrapped.reset()
        heading = robot.data.heading_w.torch.clone()
        phase = torch.zeros(n, dtype=torch.long, device=device)
        elapsed = torch.zeros(n, device=device)
        good = torch.zeros(n, device=device)
        switches = torch.zeros(n, dtype=torch.long, device=device)
        previous_support = torch.zeros(n, dtype=torch.long, device=device)
        ready = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(round(14.0 / dt)):
            state = observe()
            source_speed = torch.where(phase < 2, torch.zeros(n, device=device), torch.where(
                phase == 2, 1.2 * minimum_jerk(elapsed / 1.5), torch.full((n,), 1.2, device=device)
            ))
            command, error = motion(source_speed, heading)
            with torch.no_grad():
                a0, a1, a2 = stand(state, command), stw(state, command), walk(state, command)
                action = torch.where((phase < 2)[:, None], a0, torch.where((phase == 2)[:, None], a1, a2))
                _, _, dones, _ = wrapped.step(action)
            force = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :].norm(dim=-1).amax(1)
            contacts = force > 5
            support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
            fall = dones.bool()
            reset = torch.nonzero(fall).flatten()
            if len(reset):
                phase[reset], elapsed[reset], good[reset], switches[reset], previous_support[reset] = 0, 0, 0, 0, 0
                heading[reset] = robot.data.heading_w.torch[reset]
            settled = (robot.data.root_lin_vel_b.torch[:, 0].abs() < 0.1) & contacts.all(1) & ~fall
            good = torch.where((phase == 0) & settled, good + dt, torch.where(phase == 0, 0, good))
            advance = (phase == 0) & (good >= 0.4)
            phase[advance], elapsed[advance], good[advance] = 1, 0, 0
            advance = (phase == 1) & (elapsed >= 0.8)
            phase[advance], elapsed[advance] = 2, 0
            changed = (support != previous_support) & ((support == 1) | (support == 2)) & (phase == 2)
            switches[changed] += 1
            acquire = (phase == 2) & ((robot.data.root_lin_vel_b.torch[:, 0] - 1.2).abs() <= 0.2) & (error.abs() <= 0.12) & (switches >= 2) & ~fall
            good = torch.where(acquire, good + dt, torch.where(phase == 2, 0, good))
            advance = (phase == 2) & (good >= 0.4)
            phase[advance], elapsed[advance], good[advance] = 3, 0, 0
            valid = (phase == 3) & ((robot.data.root_lin_vel_b.torch[:, 0] - 1.2).abs() <= 0.2) & (error.abs() <= 0.12) & ~fall
            good = torch.where(valid, good + dt, torch.where(phase == 3, 0, good))
            ready |= good >= 0.4
            previous_support = support
            elapsed += dt
            if bool(ready.all()):
                break
        if not bool(ready.all()):
            raise RuntimeError(f"valid WALK source shortfall: {int(ready.sum())}/{n}")

        alive = ready.clone()
        fall_any = torch.zeros(n, dtype=torch.bool, device=device)
        slip_any = torch.zeros_like(fall_any)
        saturation_any = torch.zeros_like(fall_any)
        impact_over_steps = torch.zeros(n, device=device)
        slip_dwell = torch.zeros(n, device=device)
        effort_dwell = torch.zeros(n, len(joints), device=device)
        flight_steps = torch.zeros(n, device=device)
        speed_sum = torch.zeros(n, device=device)
        heading_max = torch.zeros(n, device=device)
        periodic_landings = torch.zeros(n, device=device)
        previous_contacts = torch.zeros(n, 2, dtype=torch.bool, device=device)
        hold_good = torch.zeros(n, device=device)
        for step in range(round(5.0 / dt)):
            speed = 1.2 + (targets - 1.2) * minimum_jerk(torch.full((n,), step * dt / 1.4, device=device))
            state = observe()
            command, error = motion(speed, heading, 1.0)
            with torch.no_grad():
                walk_action = walk(state, command)
                wrapped_run = TensorDict({"policy": to_run_observation(state, command, route="RUN")}, batch_size=[n])
                components = run.actor.diagnostic_components(wrapped_run)
                action = walk_action.clone()
                action[candidate_id == 1] = components["running_base_action"][candidate_id == 1]
                action[candidate_id == 2] = components["action_mean"][candidate_id == 2]
                _, _, dones, _ = wrapped.step(action)
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(1) > 5
            landing = contacts & ~previous_contacts
            periodic_landings += landing.any(1)
            previous_contacts = contacts
            foot_speed = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
            slip = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(1) > 0.8
            effort = robot.data.applied_torque.torch[:, joints].abs() / robot.data.joint_effort_limits.torch[:, joints].abs().clamp_min(1e-6)
            effort_dwell = torch.where(effort >= 0.95, effort_dwell + dt, torch.zeros_like(effort_dwell))
            saturation = (effort_dwell >= 0.20).any(1)
            slip_dwell = torch.where(slip, slip_dwell + dt, torch.zeros_like(slip_dwell))
            dangerous_slip = slip_dwell >= 0.20
            impact = forces[:, :, :, 2].abs().mean(1).amax(1) > 3500.0
            fall = dones.bool()
            actual = robot.data.root_lin_vel_b.torch[:, 0]
            speed_ok = (actual - targets).abs() <= 0.25
            hold_good = torch.where(speed_ok & ~fall, hold_good + dt, torch.zeros_like(hold_good))
            fall_any |= fall
            slip_any |= dangerous_slip
            saturation_any |= saturation
            impact_over_steps += impact.float()
            alive &= ~fall
            flight_steps += (~contacts.any(1)).float()
            speed_sum += actual
            heading_max = torch.maximum(heading_max, error.abs())

        rows = {}
        for candidate_index, candidate in enumerate(candidates):
            rows[candidate] = {}
            for target in commands:
                mask = (candidate_id == candidate_index) & torch.isclose(targets, torch.tensor(target, device=device))
                hold = (hold_good[mask] >= 1.0) & alive[mask]
                periodic = (periodic_landings[mask] >= 4) & alive[mask]
                gait = periodic if target >= 2.4 else hold
                rows[candidate][f"{target:.1f}"] = {
                    "episodes": int(mask.sum()), "hold_success": float(hold.float().mean()),
                    "gait_classification_success": float(gait.float().mean()),
                    "actual_speed_mean_mps": float((speed_sum[mask] / round(5.0 / dt)).mean()),
                    "fall_rate": float(fall_any[mask].float().mean()),
                    "heading_max_mean_rad": float(heading_max[mask].mean()),
                    "dangerous_slip_proxy_rate": float(slip_any[mask].float().mean()),
                    "impact_failure_rate": float(((impact_over_steps[mask] / round(5.0 / dt)) > 0.05).float().mean()),
                    "saturation_proxy_rate": float(saturation_any[mask].float().mean()),
                    "flight_fraction_mean": float((flight_steps[mask] / round(5.0 / dt)).mean()),
                }
        payload = {
            "diagnostic_only": True, "formal_capability_claim": False,
            "source": "valid frozen graph WALK@1.2 occupancy", "episodes_per_candidate_command": episodes,
            "candidates": rows,
            "D_WALK_TO_RUN_ENDPOINTS": {
                "status": "not_executed", "reason": "phase anchors are not a steady-state base controller",
            },
            "optimizer_updates": 0, "teacher_gradients": 0,
        }
        (OUT / "base_only_closed_loop_diagnostic.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    from tensordict import TensorDict
    main()
