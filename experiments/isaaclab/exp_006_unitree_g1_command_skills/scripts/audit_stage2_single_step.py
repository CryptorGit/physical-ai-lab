"""Audit contact-gated single steps from the frozen Stage-2 velocity actor."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [str(EXP / "src"), str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
from g1_command_skills.single_step_option import SingleStepPhase, minimum_jerk  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--mode", choices=("sweep", "lead_search", "lead_validate", "validate"), default="sweep")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--vx", type=float, default=0.6)
parser.add_argument("--yaw", type=float, default=0.0)
parser.add_argument("--duration", type=float, default=0.6)
parser.add_argument("--desired-lead", choices=("natural", "left", "right"), default="natural")
parser.add_argument("--seed", type=int, default=20260722)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra

KEYPOINTS = {
    "toe": (0.06383880963290349, 0.0, -0.025807180037281774),
    "sole": (0.04321213238651294, 0.0, -0.025807180037281774),
    "heel": (0.022585455140122387, 0.0, -0.025807180037281774),
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(round((len(values) - 1) * q / 100.0), len(values) - 1)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_conditions() -> list[dict]:
    if args.mode == "validate":
        return [{"vx": args.vx, "yaw": args.yaw, "duration_s": args.duration,
                 "desired_lead": args.desired_lead, "replicate": i} for i in range(args.episodes)]
    if args.mode == "lead_search":
        values = itertools.product((-0.15, -0.08, 0.0, 0.08, 0.15), range(args.episodes))
        return [{"vx": args.vx, "yaw": yaw, "duration_s": args.duration,
                 "desired_lead": "natural", "replicate": replicate}
                for yaw, replicate in values]
    if args.mode == "lead_validate":
        conditions = []
        for desired, yaw in (("left", -0.08), ("right", 0.08)):
            conditions.extend({"vx": args.vx, "yaw": yaw, "duration_s": args.duration,
                               "desired_lead": desired, "replicate": replicate}
                              for replicate in range(args.episodes))
        return conditions
    # Broad enough to expose the actor's walking onset without claiming that a
    # fixed-duration pulse itself defines a step.  Contact events terminate it.
    values = itertools.product((0.15, 0.25, 0.40, 0.60, 0.80, 1.00, 1.30, 1.60, 2.00),
                               (0.30, 0.50, 0.70, 0.90), range(4))
    return [{"vx": vx, "yaw": 0.0, "duration_s": duration, "desired_lead": "natural",
             "replicate": replicate} for vx, duration, replicate in values]


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    conditions = make_conditions()
    n = len(conditions)
    env_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = n
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device

    with launch_simulation(env_cfg, args):
        raw = gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env = raw.unwrapped
        agent_cfg.device = env.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(str(checkpoint), load_cfg={"actor": True, "critic": False, "optimizer": False,
                                               "iteration": False, "rnd": False})
        policy = runner.get_inference_policy(device=env.device)
        robot = env.scene["robot"]
        command = env.command_manager.get_term("base_velocity")
        contact = env.scene.sensors["contact_forces"]
        foot_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_ids = [contact.body_names.index(name) for name in foot_names]
        all_joint_ids, _ = robot.find_joints(".*")
        ankle_ids, _ = robot.find_joints(".*ankle.*")
        wrapped.reset()

        dt = float(env.step_dt)
        settle_need = max(1, round(0.4 / dt))
        recovery_need = max(1, round(0.4 / dt))
        stand_need = max(1, round(0.8 / dt))
        phase = torch.full((n,), int(SingleStepPhase.SETTLE), dtype=torch.long, device=env.device)
        phase_step = torch.zeros(n, dtype=torch.long, device=env.device)
        settle_streak = torch.zeros(n, dtype=torch.long, device=env.device)
        recovery_streak = torch.zeros(n, dtype=torch.long, device=env.device)
        stand_streak = torch.zeros(n, dtype=torch.long, device=env.device)
        active = torch.ones(n, dtype=torch.bool, device=env.device)
        fall = torch.zeros(n, dtype=torch.bool, device=env.device)
        dangerous_support = torch.zeros(n, dtype=torch.bool, device=env.device)
        flight_run = torch.zeros(n, dtype=torch.long, device=env.device)
        max_flight_run = torch.zeros(n, dtype=torch.long, device=env.device)
        lead = torch.full((n,), -1, dtype=torch.long, device=env.device)
        liftoff_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
        touchdown_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
        off_count = torch.zeros(n, dtype=torch.long, device=env.device)
        extra_liftoffs = torch.zeros(n, dtype=torch.long, device=env.device)
        failure = ["" for _ in range(n)]
        baseline_root_x = torch.zeros(n, device=env.device)
        baseline_points = {name: torch.zeros((n, 2, 3), device=env.device) for name in KEYPOINTS}
        max_points = {name: torch.full((n, 2, 3), -math.inf, device=env.device) for name in KEYPOINTS}
        touchdown_points = {name: torch.zeros((n, 3), device=env.device) for name in KEYPOINTS}
        torque_sat_steps = torch.zeros(n, device=env.device)
        velocity_sat_steps = torch.zeros(n, device=env.device)
        observed_steps = torch.zeros(n, device=env.device)
        max_impact = torch.zeros(n, device=env.device)
        max_slip = torch.zeros(n, device=env.device)
        max_action_rate = torch.zeros(n, device=env.device)
        final_speed = torch.full((n,), math.inf, device=env.device)
        previous_action = torch.zeros((n, wrapped.num_actions), device=env.device)
        step_rows: list[dict] = []
        max_steps = round(8.0 / dt)

        for sim_step in range(max_steps):
            command.vel_command_b.zero_()
            for env_id, condition in enumerate(conditions):
                if not active[env_id]:
                    continue
                current = SingleStepPhase(int(phase[env_id]))
                if current in (SingleStepPhase.COMMAND_STEP, SingleStepPhase.SWING):
                    age = float(phase_step[env_id]) * dt
                    ramp = minimum_jerk(age / 0.15)
                    tail = minimum_jerk(max(float(condition["duration_s"]) - age, 0.0) / 0.15)
                    command.vel_command_b[env_id, 0] = float(condition["vx"]) * min(ramp, tail)
                    command.vel_command_b[env_id, 2] = float(condition["yaw"]) * min(ramp, tail)
            applied_command = command.vel_command_b.clone()
            observations = wrapped.get_observations()
            with torch.inference_mode():
                actions = policy(observations)
                _, _, dones, _ = wrapped.step(actions)
            # Re-assert evaluator ownership after command-manager updates.
            command.vel_command_b.zero_()

            force_history = contact.data.net_forces_w_history.torch[:, :, sensor_ids, :]
            forces = force_history.norm(dim=-1).amax(dim=1)
            contacts = forces > 5.0
            count = contacts.sum(dim=1)
            speed = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
            vertical = robot.data.root_lin_vel_w.torch[:, 2].abs()
            gravity = robot.data.projected_gravity_b.torch
            roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
            pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1].square() + gravity[:, 2].square()))
            pos = robot.data.body_pos_w.torch[:, foot_ids]
            quat = robot.data.body_quat_w.torch[:, foot_ids]
            points = {}
            for name, local_value in KEYPOINTS.items():
                local = torch.tensor(local_value, device=env.device).expand(n, 2, 3)
                points[name] = pos + quat_apply(quat.reshape(-1, 4), local.reshape(-1, 3)).reshape(n, 2, 3)
            vr = robot.data.joint_vel.torch[:, all_joint_ids].abs() / robot.data.joint_vel_limits.torch[:, all_joint_ids].abs().clamp_min(1e-6)
            tr = robot.data.applied_torque.torch[:, all_joint_ids].abs() / robot.data.joint_effort_limits.torch[:, all_joint_ids].abs().clamp_min(1e-6)
            slip = robot.data.body_lin_vel_w.torch[:, foot_ids, :2].norm(dim=-1) * contacts
            action_rate = (actions - previous_action).abs().amax(dim=1)
            previous_action.copy_(actions)

            for env_id, condition in enumerate(conditions):
                if not active[env_id]:
                    continue
                current = SingleStepPhase(int(phase[env_id]))
                measuring = current != SingleStepPhase.SETTLE
                if measuring:
                    observed_steps[env_id] += 1
                    torque_sat_steps[env_id] += float((tr[env_id, ankle_ids] >= 0.95).any())
                    velocity_sat_steps[env_id] += float((vr[env_id] >= 0.95).any())
                max_impact[env_id] = torch.maximum(max_impact[env_id], forces[env_id].max())
                max_slip[env_id] = torch.maximum(max_slip[env_id], slip[env_id].max())
                max_action_rate[env_id] = torch.maximum(max_action_rate[env_id], action_rate[env_id])
                if measuring and count[env_id] == 0:
                    flight_run[env_id] += 1
                    max_flight_run[env_id] = torch.maximum(max_flight_run[env_id], flight_run[env_id])
                    if flight_run[env_id] * dt > 0.04:
                        dangerous_support[env_id] = True
                else:
                    flight_run[env_id] = 0
                safe = speed[env_id] <= 0.08 and vertical[env_id] <= 0.05 and abs(roll[env_id]) <= 0.10 and abs(pitch[env_id]) <= 0.10
                if current == SingleStepPhase.SETTLE:
                    settle_streak[env_id] = settle_streak[env_id] + 1 if safe and count[env_id] == 2 else 0
                    if settle_streak[env_id] >= settle_need:
                        baseline_root_x[env_id] = robot.data.root_pos_w.torch[env_id, 0]
                        for name in KEYPOINTS:
                            baseline_points[name][env_id] = points[name][env_id]
                        phase[env_id] = int(SingleStepPhase.COMMAND_STEP)
                        phase_step[env_id] = 0
                    elif phase_step[env_id] * dt > 2.0:
                        failure[env_id] = "standing_settle_failure"; active[env_id] = False
                elif current in (SingleStepPhase.COMMAND_STEP, SingleStepPhase.SWING):
                    single = int(count[env_id]) == 1
                    if current == SingleStepPhase.COMMAND_STEP and single:
                        swing_id = 0 if not bool(contacts[env_id, 0]) else 1
                        lead[env_id] = swing_id
                        liftoff_step[env_id] = sim_step
                        phase[env_id] = int(SingleStepPhase.SWING)
                        phase_step[env_id] = 0
                    if int(phase[env_id]) == int(SingleStepPhase.SWING):
                        foot = int(lead[env_id])
                        for name in KEYPOINTS:
                            max_points[name][env_id, foot] = torch.maximum(max_points[name][env_id, foot], points[name][env_id, foot])
                        if not bool(contacts[env_id, foot]):
                            off_count[env_id] += 1
                        elif off_count[env_id] >= 2:
                            touchdown_step[env_id] = sim_step
                            for name in KEYPOINTS:
                                touchdown_points[name][env_id] = points[name][env_id, foot]
                            phase[env_id] = int(SingleStepPhase.DOUBLE_SUPPORT_RECOVERY)
                            phase_step[env_id] = 0
                    if current == SingleStepPhase.COMMAND_STEP and phase_step[env_id] * dt > float(condition["duration_s"]) + 1.0:
                        failure[env_id] = "step_liftoff_failure"; active[env_id] = False
                    elif current == SingleStepPhase.SWING and phase_step[env_id] * dt > 1.5:
                        failure[env_id] = "touchdown_failure"; active[env_id] = False
                elif current == SingleStepPhase.DOUBLE_SUPPORT_RECOVERY:
                    if int(count[env_id]) == 1:
                        extra_liftoffs[env_id] += 1
                    recovery_streak[env_id] = recovery_streak[env_id] + 1 if safe and count[env_id] == 2 else 0
                    if recovery_streak[env_id] >= recovery_need:
                        phase[env_id] = int(SingleStepPhase.STAND_HOLD); phase_step[env_id] = 0
                    elif phase_step[env_id] * dt > 2.0:
                        failure[env_id] = "double_support_recovery_failure"; active[env_id] = False
                elif current == SingleStepPhase.STAND_HOLD:
                    stand_streak[env_id] = stand_streak[env_id] + 1 if safe and count[env_id] == 2 else 0
                    if stand_streak[env_id] >= stand_need:
                        final_speed[env_id] = speed[env_id]
                        active[env_id] = False
                    elif phase_step[env_id] * dt > 1.5:
                        failure[env_id] = "zero_command_recovery_failure"; active[env_id] = False
                if bool(dones[env_id]):
                    fall[env_id] = True; failure[env_id] = "fall"; active[env_id] = False
                step_rows.append({"episode": env_id, "sim_step": sim_step, "time_s": sim_step * dt,
                                  "phase": SingleStepPhase(int(phase[env_id])).name, "command_vx": float(condition["vx"]),
                                  "applied_command_vx": float(applied_command[env_id, 0]),
                                  "contacts_left": bool(contacts[env_id, 0]), "contacts_right": bool(contacts[env_id, 1]),
                                  "actual_speed_mps": float(speed[env_id]), "actor_action_norm": float(actions[env_id].norm()),
                                  "action_rate_max": float(action_rate[env_id])})
                phase_step[env_id] += 1
            if not bool(active.any()):
                break

        rows = []
        for env_id, condition in enumerate(conditions):
            foot = int(lead[env_id])
            lead_name = "left" if foot == 0 else "right" if foot == 1 else "none"
            desired_ok = condition["desired_lead"] == "natural" or condition["desired_lead"] == lead_name
            # At most 40 ms of contact debounce is tolerated after touchdown;
            # a longer renewed single-support interval is a second gait step,
            # not a recovered single-step option.
            one_step_only = int(extra_liftoffs[env_id]) <= max(1, round(0.04 / dt))
            successful = (touchdown_step[env_id] >= 0 and stand_streak[env_id] >= stand_need and desired_ok and one_step_only
                          and not fall[env_id] and not dangerous_support[env_id]
                          and torque_sat_steps[env_id] / observed_steps[env_id].clamp_min(1) <= 0.05
                          and velocity_sat_steps[env_id] / observed_steps[env_id].clamp_min(1) <= 0.05)
            row = dict(condition)
            derived_failure = ""
            if not desired_ok:
                derived_failure = "desired_lead_phase_unavailable"
            elif not one_step_only:
                derived_failure = "double_support_recovery_failure"
            elif not successful:
                derived_failure = "single_step_unavailable"
            row.update({"episode": env_id, "natural_lead": lead_name, "desired_lead_success": desired_ok,
                        "single_step_success": bool(successful), "failure_class": failure[env_id] or derived_failure,
                        "liftoff_time_s": float(liftoff_step[env_id]) * dt if liftoff_step[env_id] >= 0 else None,
                        "touchdown_time_s": float(touchdown_step[env_id]) * dt if touchdown_step[env_id] >= 0 else None,
                        "pelvis_forward_displacement_m": float(robot.data.root_pos_w.torch[env_id, 0] - baseline_root_x[env_id]),
                        "fall": bool(fall[env_id]), "dangerous_support_loss": bool(dangerous_support[env_id]),
                        "maximum_both_feet_airborne_s": float(max_flight_run[env_id]) * dt,
                        "extra_single_support_steps_after_touchdown": int(extra_liftoffs[env_id]),
                        "final_horizontal_speed_mps": float(final_speed[env_id]),
                        "ankle_torque_saturation_fraction": float(torque_sat_steps[env_id] / observed_steps[env_id].clamp_min(1)),
                        "joint_velocity_saturation_fraction": float(velocity_sat_steps[env_id] / observed_steps[env_id].clamp_min(1)),
                        "max_foot_slip_mps": float(max_slip[env_id]), "max_impact_n": float(max_impact[env_id]),
                        "max_action_rate": float(max_action_rate[env_id])})
            for name in KEYPOINTS:
                if foot >= 0:
                    row[f"{name}_max_height_m"] = float(max_points[name][env_id, foot, 2] - baseline_points[name][env_id, foot, 2])
                    row[f"{name}_max_forward_m"] = float(max_points[name][env_id, foot, 0] - baseline_points[name][env_id, foot, 0])
                    row[f"{name}_touchdown_forward_m"] = float(touchdown_points[name][env_id, 0] - baseline_points[name][env_id, foot, 0])
                else:
                    row[f"{name}_max_height_m"] = row[f"{name}_max_forward_m"] = row[f"{name}_touchdown_forward_m"] = 0.0
            row["stride_length_m"] = row["sole_touchdown_forward_m"]
            rows.append(row)
        write_csv(output / "episodes.csv", rows)
        write_csv(output / "steps.csv", step_rows)
        successful = [row for row in rows if row["single_step_success"]]
        summary = {"checkpoint": str(checkpoint), "mode": args.mode, "seed": args.seed, "episodes": n,
                   "contact_event_gated": True, "production_step_over_unchanged_fail_closed": True,
                   "single_step_success_rate": mean([float(row["single_step_success"]) for row in rows]),
                   "left_lead_count": sum(row["natural_lead"] == "left" for row in rows),
                   "right_lead_count": sum(row["natural_lead"] == "right" for row in rows),
                   "no_liftoff_count": sum(row["natural_lead"] == "none" for row in rows),
                   "fall_rate": mean([float(row["fall"]) for row in rows]),
                   "dangerous_support_loss_rate": mean([float(row["dangerous_support_loss"]) for row in rows]),
                   "saturation_failure_rate": mean([float(row["ankle_torque_saturation_fraction"] > .05 or row["joint_velocity_saturation_fraction"] > .05) for row in rows]),
                   "stride_mean_m": mean([row["stride_length_m"] for row in successful]),
                   "stride_p95_m": percentile([row["stride_length_m"] for row in successful], 95),
                   "max_safe_forward_reach_m": max([row["sole_max_forward_m"] for row in successful], default=0.0),
                   "toe_clearance_mean_m": mean([row["toe_max_height_m"] for row in successful]),
                   "sole_clearance_mean_m": mean([row["sole_max_height_m"] for row in successful]),
                   "heel_clearance_mean_m": mean([row["heel_max_height_m"] for row in successful]),
                   "final_speed_mean_mps": mean([row["final_horizontal_speed_mps"] for row in successful]),
                   "best_conditions": sorted(successful, key=lambda row: (row["stride_length_m"], -row["max_action_rate"]), reverse=True)[:20]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key != "best_conditions"}, indent=2))
        raw.close()


if __name__ == "__main__":
    main()
