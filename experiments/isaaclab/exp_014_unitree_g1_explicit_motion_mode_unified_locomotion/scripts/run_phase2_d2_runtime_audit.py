"""Read-only runtime probes for EXP014 Phase 2-D2.

No optimizer, training API, dataset writer, or checkpoint writer is present.
Each invocation recreates the registered reset distribution in a fresh process.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
BASE = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
OUT = BASE / "phase_2_d2_specialist_s_action_contract_parity"
RAW = OUT / "raw"
STAGE2Q = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
STAND = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"

sys.path[:0] = [
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    str(REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"),
    str(EXP / "src"),
]
import isaaclab_tasks  # noqa: F401,E402
import g1_omnidirectional.tasks  # noqa: F401,E402
import g1_single_policy.tasks  # noqa: F401,E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from g1_explicit_motion_mode.contract import ExplicitMotionModeCommand, build_observation_141  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

DT = 0.02
RUNS = (
    "exp014_stage2q_baseline", "exp014_stage2q_prev_first", "exp014_stage2q_gait1",
    "exp014_stand_candidate", "dual_hold", "dual_stop",
    "reference_stand", "reference_moving_stop",
)


class StandActor(nn.Sequential):
    def __init__(self, checkpoint: Path, device: torch.device):
        super().__init__(nn.Linear(123, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37))
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.load_state_dict({k.removeprefix("mlp."): v for k, v in payload["actor_state_dict"].items() if k.startswith("mlp.")})
        self.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def vector(value: torch.Tensor) -> str:
    return json.dumps([round(float(x), 9) for x in value.detach().cpu().flatten()])


def safe_attr(obj, name: str):
    value = getattr(obj, name, None)
    if torch.is_tensor(value):
        return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS, required=True)
    add_launcher_args(parser)
    args, hydra = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]

    reference = args.run.startswith("reference_")
    task = "Isaac-Exp012-G1-Reverse-PhaseR1-v0" if reference else "Isaac-Exp013-G1-DirectionalBaseline-v0"
    # The original transition evaluator instantiated six 100-episode condition
    # blocks; WALK_TO_STAND occupied environments 500:600.  Recreate that RNG
    # topology exactly and report only the registered block.
    n = 100 if args.run == "reference_stand" else 600 if args.run == "reference_moving_stop" else 680
    seed = 20269031 if reference else 20260803
    cfg, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.episode_length_s = 16.0
    cfg.seed = seed
    cfg.observations.policy.enable_corruption = False
    if not reference:
        cfg.events.base_external_force_torque = None
        cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = agent.device = args.device
    RAW.mkdir(parents=True, exist_ok=True)

    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        dev = env.device
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        sf = sensor.find_bodies(".*_ankle_roll_link")[0]
        rf = robot.find_bodies(".*_ankle_roll_link")[0]
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        ids = torch.arange(n, device=dev)
        stage2q = FrozenGaitActor(STAGE2Q).to(dev).eval()
        stand = StandActor(STAND, dev)
        gait_value = 1.0 if args.run == "exp014_stage2q_gait1" else 0.0
        gait = torch.full((n,), gait_value, device=dev)
        state = ExplicitMotionModeCommand.zeros(n, device=dev)
        lifecycle = []

        def command(speed: torch.Tensor | float = 0.0) -> None:
            term.external_override.zero_()
            term.external_override[:, 0] = speed
            term._update_command()

        def snapshot(stage: str, obs123: torch.Tensor | None = None, action: torch.Tensor | None = None) -> None:
            force = sensor.data.net_forces_w_history[:, -1, sf, :].norm(dim=-1)
            reset_buf = safe_attr(env, "reset_buf")
            timeout_buf = safe_attr(env, "time_out_buf")
            current_action = env.action_manager.action
            previous_action = env.action_manager.prev_action
            for i in range(n):
                lifecycle.append({
                    "recipe_id": i,
                    "stage": stage,
                    "root_pose": vector(torch.cat((robot.data.root_pos_w[i], robot.data.root_quat_w[i]))),
                    "root_velocity": vector(torch.cat((robot.data.root_lin_vel_w[i], robot.data.root_ang_vel_w[i]))),
                    "joint_position": vector(robot.data.joint_pos[i]),
                    "joint_velocity": vector(robot.data.joint_vel[i]),
                    "base_linear_velocity": vector(robot.data.root_lin_vel_b[i]),
                    "base_angular_velocity": vector(robot.data.root_ang_vel_b[i]),
                    "projected_gravity": vector(robot.data.projected_gravity_b[i]),
                    "physical_command": vector(term.external_override[i, :3]),
                    "actor_command": vector(term.command[i, :3]),
                    "legacy_gait": gait_value,
                    "current_action": vector(current_action[i]),
                    "previous_action": vector(previous_action[i]),
                    "episode_length": int(env.episode_length_buf[i]),
                    "reset_buffer": None if reset_buf is None else bool(reset_buf[i]),
                    "timeout_buffer": None if timeout_buf is None else bool(timeout_buf[i]),
                    "command_resampling_timer": None if not hasattr(term, "time_left") else float(term.time_left[i]),
                    "heading_target": None if not hasattr(term, "heading_target") else float(term.heading_target[i]),
                    "contact_state": vector(force[i]),
                    "observation_123": None if obs123 is None else vector(obs123[i]),
                    "policy_action": None if action is None else vector(action[i]),
                    "observation_history": "ABSENT_IN_123D_CONTRACT",
                })

        # Gym construction performs its manager reset; capture it before the explicit
        # reset used by the registered exp014 collector.
        snapshot("T_PRE_RESET")
        env.reset(env_ids=ids)
        snapshot("T_POST_RESET")
        command(0.0)
        snapshot("T_COMMAND_ZERO")
        obs = wrapped.get_observations().to(dev)
        obs123 = obs["policy"]
        snapshot("T_OBSERVATION_0", obs123)

        # R0 original and R1 current wrapper both consume the same 123-D manager
        # observation and append gait only inside the identical first-layer algebra.
        r0_124 = torch.cat((obs123, gait[:, None]), dim=1)
        r1_141 = build_observation_141(obs123, state)
        r1_124 = r1_141[:, :124]
        with torch.inference_mode():
            r0_action = stage2q(r0_124[:, :123], r0_124[:, 123])
            r1_action = stage2q(r1_124[:, :123], r1_124[:, 123])
        parity = {
            "observation_max_abs_difference": float((r0_124 - r1_124).abs().max()),
            "observation_bitwise_equal": bool(torch.equal(r0_124, r1_124)),
            "mean_action_max_abs_difference": float((r0_action - r1_action).abs().max()),
            "mean_action_bitwise_equal": bool(torch.equal(r0_action, r1_action)),
            "action_l2_max": float(torch.linalg.vector_norm(r0_action - r1_action, dim=1).max()),
            "action_cosine_min": float(torch.nn.functional.cosine_similarity(r0_action, r1_action).min()),
            "r0_observation_hash": tensor_hash(r0_124),
            "r1_observation_hash": tensor_hash(r1_124),
            "r0_action_hash": tensor_hash(r0_action),
            "r1_action_hash": tensor_hash(r1_action),
        }

        if args.run == "exp014_stage2q_prev_first":
            env.action_manager._prev_action.copy_(r0_action)
            env.action_manager._action.copy_(r0_action)
            obs = wrapped.get_observations().to(dev)
            obs123 = obs["policy"]

        fall = torch.zeros(n, dtype=torch.bool, device=dev)
        slip = fall.clone(); impact = fall.clone(); saturation = fall.clone()
        slip_streak = torch.zeros(n, dtype=torch.long, device=dev)
        sat_streak = slip_streak.clone()
        limits = robot.data.joint_vel_limits
        limits = limits[..., 1].abs() if limits.ndim == 3 else limits
        speeds = []; yaws = []; contacts_trace = []; first_actions = []; first_contacts = []
        boundary_compare = None
        steps = 500 if reference else 200 if args.run in ("dual_hold", "dual_stop") else 100
        stand_policy = args.run in ("exp014_stand_candidate", "dual_hold", "dual_stop")

        for step in range(steps):
            if args.run == "reference_moving_stop":
                t = step * DT
                if t < 2.0:
                    speed = 1.2
                elif t < 3.0:
                    x = t - 2.0; speed = 1.2 + (0.6 - 1.2) * (10*x**3 - 15*x**4 + 6*x**5)
                elif t < 4.0:
                    x = t - 3.0; speed = 0.6 * (1 - (10*x**3 - 15*x**4 + 6*x**5))
                else:
                    speed = 0.0
                command(speed)
            else:
                command(0.0)
            if not reference:
                state.advance(torch.zeros(n, 3, device=dev), torch.ones(n, device=dev), DT)
            obs = wrapped.get_observations().to(dev); obs123 = obs["policy"]
            if step == 100 and args.run in ("dual_hold", "dual_stop"):
                with torch.inference_mode():
                    hold_action = stand(obs123)
                    stop_action = stage2q(obs123, torch.zeros(n, device=dev))
                delta = hold_action - stop_action
                boundary_compare = {
                    "samples": n,
                    "action_l2_mean": float(torch.linalg.vector_norm(delta, dim=1).mean()),
                    "action_l2_max": float(torch.linalg.vector_norm(delta, dim=1).max()),
                    "cosine_mean": float(torch.nn.functional.cosine_similarity(hold_action, stop_action).mean()),
                    "cosine_min": float(torch.nn.functional.cosine_similarity(hold_action, stop_action).min()),
                    "material_conflict_rate": float(((torch.linalg.vector_norm(delta, dim=1) >= .5) | (torch.nn.functional.cosine_similarity(hold_action, stop_action) <= .98)).float().mean()),
                    "hold_action_hash": tensor_hash(hold_action), "stop_action_hash": tensor_hash(stop_action),
                }
            with torch.inference_mode():
                if stand_policy and not (args.run == "dual_stop" and step >= 100):
                    action = stand(obs123)
                else:
                    action = stage2q(obs123, gait)
            if step < 4:
                first_actions.append(action.detach().cpu())
                contact_now = sensor.data.net_forces_w_history[:, -1, sf, :].norm(dim=-1) > 5
                first_contacts.append(contact_now.detach().cpu())
            if step == 0: snapshot("T_ACTION_0", obs123, action)
            if step == 1: snapshot("T_ACTION_1", obs123, action)
            obs, _, done, extras = wrapped.step(action); obs = obs.to(dev)
            if step == 0: snapshot("T_STEP_1")
            if step == 1: snapshot("T_STEP_2")
            if step == 2: snapshot("T_STEP_3")
            if step == 3: snapshot("T_STEP_4")
            timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
            fall |= done.bool() & ~timeout
            force = sensor.data.net_forces_w_history[:, -1, sf, :].norm(dim=-1)
            contact = force > 5
            feet_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, rf, :2], dim=-1)
            bad = ((feet_speed > .55) & contact).any(1)
            slip_streak = torch.where(bad, slip_streak + 1, torch.zeros_like(slip_streak)); slip |= slip_streak >= 5
            impact |= force.amax(1) > 3500
            ratio = robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(1)
            sat_streak = torch.where(ratio > .95, sat_streak + 1, torch.zeros_like(sat_streak)); saturation |= sat_streak >= 5
            speeds.append(torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1).detach().cpu())
            yaws.append(robot.data.root_ang_vel_b[:, 2].abs().detach().cpu())
            contacts_trace.append(contact.sum(1).detach().cpu())

        speed_t = torch.stack(speeds); yaw_t = torch.stack(yaws); contact_t = torch.stack(contacts_trace)
        window = (
            slice(100, None) if args.run in ("dual_hold", "dual_stop")
            else slice(400, None) if args.run == "reference_moving_stop"
            else slice(None)
        )
        speed_mean_episode = speed_t[window].mean(0); yaw_mean_episode = yaw_t[window].mean(0)
        practical = (speed_mean_episode <= .08) & (yaw_mean_episode <= .08) & ~fall.cpu() & ~slip.cpu() & ~impact.cpu()
        strict = (contact_t[window].eq(0).float().mean(0) == 0) & (contact_t[window].eq(2).float().mean(0) >= .95) & ~fall.cpu()
        eval_ids = torch.arange(500, 600) if args.run == "reference_moving_stop" else torch.arange(n)
        speed_eval = speed_mean_episode[eval_ids]; yaw_eval = yaw_mean_episode[eval_ids]
        practical_eval = practical[eval_ids]; strict_eval = strict[eval_ids]
        fall_eval = fall.cpu()[eval_ids]; slip_eval = slip.cpu()[eval_ids]
        impact_eval = impact.cpu()[eval_ids]; saturation_eval = saturation.cpu()[eval_ids]
        result = {
            "run": args.run, "task": task, "seed": seed, "episodes": len(eval_ids), "simulator_environments": n, "control_steps": steps,
            "policy": "stand_candidate" if stand_policy else "stage2q",
            "gait": gait_value,
            "metrics": {
                "practical_stand": float(practical_eval.float().mean()), "strict_stand": float(strict_eval.float().mean()),
                "fall": float(fall_eval.float().mean()), "dangerous_slip": float(slip_eval.float().mean()),
                "impact": float(impact_eval.float().mean()), "long_dwell_saturation": float(saturation_eval.float().mean()),
                "speed_mean": float(speed_eval.mean()), "speed_p95": float(torch.quantile(speed_eval, .95)),
                "absolute_yaw_mean": float(yaw_eval.mean()), "absolute_yaw_p95": float(torch.quantile(yaw_eval, .95)),
            },
            "first_four_action_hash": tensor_hash(torch.stack(first_actions)),
            "first_four_contact_hash": tensor_hash(torch.stack(first_contacts).to(torch.uint8)),
            "same_state_parity": parity,
            "boundary_compare": boundary_compare,
            "initial_previous_action_hash": tensor_hash(lifecycle[n * 3] and env.action_manager.prev_action.detach().cpu()) if False else None,
        }
        (RAW / f"{args.run}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.run == "exp014_stage2q_baseline":
            with (RAW / "reset_lifecycle.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(lifecycle[0])); writer.writeheader(); writer.writerows(lifecycle)
            (RAW / "reset_lifecycle.json").write_text(json.dumps({
                "rows": len(lifecycle), "recipes": n,
                "stages": ["T_PRE_RESET", "T_POST_RESET", "T_COMMAND_ZERO", "T_OBSERVATION_0", "T_ACTION_0", "T_STEP_1", "T_ACTION_1", "T_STEP_2", "T_STEP_3", "T_STEP_4"],
                "command_zero_before_observation": True, "sensor_read_before_action": True,
                "observation_history": "absent", "same_state_parity": parity,
            }, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
