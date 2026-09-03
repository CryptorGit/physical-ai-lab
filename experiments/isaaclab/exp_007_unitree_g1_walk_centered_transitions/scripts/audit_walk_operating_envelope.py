"""Stage 2B limited WALK envelope and fixed-heading audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [str(EXP / "src"), str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_walk_observation  # noqa: E402
from g1_walk_centered.heading_controller import FixedHeadingConfig, fixed_heading_yaw_rate, wrap_angle  # noqa: E402
from g1_walk_centered.stand_walk_controller import Phase, velocity_command  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

EXPECTED_SHA = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
STAGE2A_COMMIT = "048af59"
TASK = "Isaac-Velocity-Flat-G1-Run-Eval-v0"
SWEEP_SPEEDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
CANDIDATES = (
    ("H075_D005_L050", .75, .05, .50),
    ("H100_D005_L050", 1.00, .05, .50),
    ("H100_D010_L050", 1.00, .10, .50),
    ("H125_D010_L050", 1.25, .10, .50),
)
RAMP_DURATION_S = 2.0
SPEED_SUSTAIN_S = 2.0
VELOCITY_SATURATION_FRACTION = .05
ANKLE_TORQUE_SATURATION_FRACTION = .20
FOOT_SLIP_MEAN_LIMIT_MPS = .55

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--mode", choices=("candidates", "sweep", "compare", "transition"), required=True)
parser.add_argument("--heading-mode", choices=("ZeroYaw", "FixedTarget"), default="FixedTarget")
parser.add_argument("--seed", type=int, default=20260725)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * q / 100), len(ordered) - 1)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "liftoff_rate": mean([r["liftoff"] for r in rows]),
        "sustained_walk_rate": mean([r["sustained_walk"] for r in rows]),
        "actual_speed_mean_mps": mean([r["actual_speed_mean_mps"] for r in rows]),
        "actual_speed_p95_mps": mean([r["actual_speed_p95_mps"] for r in rows]),
        "speed_error_mean_mps": mean([r["speed_error_mean_mps"] for r in rows]),
        "heading_error_p95_rad": mean([r["heading_error_p95_rad"] for r in rows]),
        "lateral_drift_max_m": mean([r["lateral_drift_max_m"] for r in rows]),
        "double_support_fraction": mean([r["double_support_fraction"] for r in rows]),
        "single_support_fraction": mean([r["single_support_fraction"] for r in rows]),
        "flight_fraction": mean([r["flight_fraction"] for r in rows]),
        "fall_rate": mean([r["fall"] for r in rows]),
        "saturation_failure_rate": mean([r["saturation_failure"] for r in rows]),
        "velocity_saturation_fraction": mean([r["velocity_saturation_fraction"] for r in rows]),
        "ankle_pitch_torque_saturation_fraction": mean([r["ankle_pitch_torque_saturation_fraction"] for r in rows]),
        "ankle_roll_torque_saturation_fraction": mean([r["ankle_roll_torque_saturation_fraction"] for r in rows]),
        "knee_velocity_utilization_max": max([r["knee_velocity_utilization_max"] for r in rows], default=0.0),
        "pelvis_vertical_range_m": mean([r["pelvis_vertical_range_m"] for r in rows]),
        "foot_slip_mean_mps": mean([r["foot_slip_mean_mps"] for r in rows]),
        "stand_recovery_rate": mean([r["stand_recovery"] for r in rows]),
        "action_rate_p99": mean([r["action_rate_p99"] for r in rows]),
        "yaw_rate_p95_radps": mean([r["yaw_rate_p95_radps"] for r in rows]),
        "yaw_reversal_frequency_hz": mean([r["yaw_reversal_frequency_hz"] for r in rows]),
    }


def assignments(output: Path):
    if args.mode == "candidates":
        items = [(speed, candidate) for candidate in CANDIDATES for speed in (.8, 1.2) for _ in range(3)]
        return [x[0] for x in items], [x[1] for x in items]
    selected = json.loads((output / "selected_heading_controller.json").read_text())
    cfg = (selected["id"], selected["k_heading"], selected["k_yaw_rate"], selected["yaw_rate_limit_radps"])
    if args.mode == "sweep":
        return [speed for speed in SWEEP_SPEEDS for _ in range(5)], [cfg] * 60
    if args.mode == "compare":
        return [speed for speed in (.8, 1.0, 1.2) for _ in range(5)], [cfg] * 15
    safe = json.loads((output / "safe_upper_limit_summary.json").read_text())["transition_pilot_speeds_mps"]
    return [speed for speed in safe for _ in range(10)], [cfg] * (10 * len(safe))


def classify(row: dict) -> str:
    if row["saturation_failure"]:
        return "saturation-dominated walking"
    if row["fall"] or row["excessive_flight"] or row["heading_error_p95_rad"] > .12 or row["lateral_drift_max_m"] > .50:
        return "unstable walking"
    if not row["liftoff"]:
        return "no liftoff"
    if not row["sustained_walk"]:
        return "isolated step"
    return "sustained walking"


def write_final_classification(output: Path, transition_summary: dict) -> None:
    safe = json.loads((output / "safe_upper_limit_summary.json").read_text())
    comparison = json.loads((output / "zero_yaw_vs_heading_feedback.json").read_text())
    zero = comparison["zero_yaw"]["overall"]
    fixed = comparison["fixed_heading"]["overall"]
    heading_improved = fixed["heading_error_p95_rad"] < zero["heading_error_p95_rad"]
    destabilized = (
        fixed["fall_rate"] > zero["fall_rate"]
        or fixed["saturation_failure_rate"] > zero["saturation_failure_rate"]
        or fixed["foot_slip_mean_mps"] > 1.25 * max(zero["foot_slip_mean_mps"], 1e-6)
        or fixed["yaw_reversal_frequency_hz"] > 1.5 * max(zero["yaw_reversal_frequency_hz"], 1e-6)
    )
    safe_speeds = safe["safe_speeds_mps"]
    pilot_ok = transition_summary.get("overall", {}).get("sustained_walk_rate", 0.0) >= .90 and transition_summary.get("overall", {}).get("stand_recovery_rate", 0.0) >= .90
    if heading_improved and destabilized:
        classification = "HEADING_CONTROLLER_UNSTABLE"
    elif not safe_speeds or not pilot_ok:
        classification = "WALK_EXPERT_RETRAIN_REQUIRED"
    elif min(safe_speeds) <= .8 and max(safe_speeds) >= 1.2:
        classification = "CONTROLLER_LIMITATION_RESOLVED"
    else:
        classification = "LIMITED_OPERATING_ENVELOPE"
    payload = {
        "classification": classification,
        "eligible_for_stage2c": classification in ("CONTROLLER_LIMITATION_RESOLVED", "LIMITED_OPERATING_ENVELOPE"),
        "safe_candidate_speeds_mps": safe_speeds,
        "transition_pilot_speeds_mps": safe["transition_pilot_speeds_mps"],
        "heading_feedback_improved_heading": heading_improved,
        "heading_feedback_destabilized_other_metrics": destabilized,
        "transition_pilot": transition_summary,
        "capability_manifest_updated": False,
        "stage2a_gate_changed": False,
    }
    (output / "classification.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    before_hash = sha256(checkpoint)
    if before_hash != EXPECTED_SHA:
        raise RuntimeError(f"checkpoint hash mismatch: {before_hash}")
    output = Path(args.output)
    output = output if output.is_absolute() else REPO / output
    output.mkdir(parents=True, exist_ok=True)
    stage2a = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage2_stand_walk"
    stage2a_reference = {
        "status": "FROZEN_NEGATIVE_BASELINE",
        "commit": subprocess.check_output(["git", "rev-parse", STAGE2A_COMMIT], cwd=REPO, text=True).strip(),
        "files": {
            name: {"path": str((stage2a / name).relative_to(REPO)), "sha256": sha256(stage2a / name)}
            for name in ("gate.json", "formal_summary.json", "episodes.csv", "selected_controller.json")
        },
        "metrics": {"full_sequence": "0/50", "stand_to_walk": "17/50", "walk_hold": "0/50", "walk_to_stand": "17/50", "fall": "1/50", "saturation_failure": "14/50"},
        "modified_by_stage2b": False,
    }
    (output / "stage2a_reference.json").write_text(json.dumps(stage2a_reference, indent=2) + "\n")

    speeds, configs = assignments(output)
    if args.mode == "transition" and not speeds:
        skipped = {"status": "SKIPPED", "reason": "No safe continuous operating range from steady sweep", "episodes": 0}
        (output / "transition_pilot_summary.json").write_text(json.dumps(skipped, indent=2) + "\n")
        write_csv(output / "transition_pilot_episodes.csv", [], ["episode", "target_speed_mps", "status"])
        write_final_classification(output, skipped)
        return
    n = len(speeds)
    rng = random.Random(args.seed)
    initial_holds = [rng.uniform(.8, 1.2) for _ in range(n)]
    walk_holds = [3.0 if args.mode != "transition" else rng.uniform(2.5, 4.0) for _ in range(n)]
    final_holds = [2.0 if args.mode != "transition" else rng.uniform(4.0, 5.0) for _ in range(n)]
    cfg, agent = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.episode_length_s = 28.0
    if args.device is not None:
        cfg.sim.device = args.device

    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        expert = load_walk_expert(checkpoint, device=env.device)
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        all_joints, joint_names = robot.find_joints(".*")
        ankle_pitch, _ = robot.find_joints(".*ankle_pitch.*")
        ankle_roll, _ = robot.find_joints(".*ankle_roll.*")
        knees, _ = robot.find_joints(".*knee.*")
        wrapped.reset()
        device = env.device
        dt = float(env.step_dt)
        speed_t = torch.tensor(speeds, device=device)
        phase = torch.full((n,), int(Phase.INITIAL_STAND_SETTLE), dtype=torch.long, device=device)
        elapsed = torch.zeros(n, device=device)
        streak = torch.zeros(n, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        fallen = torch.zeros_like(done)
        acquired = torch.zeros_like(done)
        target_heading = robot.data.heading_w.torch.clone()
        origin_xy = robot.data.root_pos_w.torch[:, :2].clone()
        previous = torch.zeros(n, 37, device=device)
        traces = [
            {key: [] for key in ("phase", "cmd", "legacy_cmd", "yaw_cmd", "legacy_yaw", "vx", "heading_error", "lateral", "contact", "slip_left", "slip_right", "vel_ratio", "ankle_pitch", "ankle_roll", "knee", "height", "action_rate", "yaw_rate")}
            for _ in range(n)
        ]
        adapter_exact = True
        finite = True
        for _ in range(round(26.0 / dt)):
            vx_cmd = velocity_command(phase, elapsed, speed_t, torch.full_like(speed_t, RAMP_DURATION_S))
            current_heading = robot.data.heading_w.torch
            current_yaw_rate = robot.data.root_ang_vel_b.torch[:, 2]
            yaw_cmd = torch.zeros(n, device=device)
            heading_error = wrap_angle(target_heading - current_heading)
            if args.heading_mode == "FixedTarget":
                for index, spec in enumerate(configs):
                    controller = FixedHeadingConfig(spec[1], spec[2], spec[3])
                    yaw_cmd[index], heading_error[index] = tuple(
                        value.squeeze(0)
                        for value in fixed_heading_yaw_rate(
                            target_heading[index:index + 1],
                            current_heading[index:index + 1],
                            current_yaw_rate[index:index + 1],
                            controller,
                        )
                    )
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0] = vx_cmd
            command_term.vel_command_b[:, 2] = yaw_cmd
            legacy = wrapped.get_observations()["policy"]
            state = canonical_state_from_legacy_observation(legacy, heading_w_rad=current_heading)
            motion = MotionCommand(vx_cmd, torch.zeros_like(vx_cmd), target_yaw_rate_radps=yaw_cmd)
            adapter_exact &= bool(torch.equal(legacy, to_walk_observation(state, motion)))
            with torch.inference_mode():
                action = expert(state, motion)
                _, _, terminal, _ = wrapped.step(action)
            finite &= bool(torch.isfinite(action).all() and torch.isfinite(legacy).all())
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            contact_code = contacts[:, 0].long() + 2 * contacts[:, 1].long()
            support = contacts.sum(dim=1)
            vx = robot.data.root_lin_vel_b.torch[:, 0]
            horizontal = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
            vz = robot.data.root_lin_vel_w.torch[:, 2].abs()
            gravity = robot.data.projected_gravity_b.torch
            roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
            pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2))
            slips = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
            velocity_ratio = robot.data.joint_vel.torch[:, all_joints].abs() / robot.data.joint_vel_limits.torch[:, all_joints].abs().clamp_min(1e-6)
            torque_ratio = robot.data.applied_torque.torch[:, all_joints].abs() / robot.data.joint_effort_limits.torch[:, all_joints].abs().clamp_min(1e-6)
            action_rate = torch.linalg.vector_norm(action - previous, dim=1) / dt
            previous = action.clone()
            safe = (horizontal <= .08) & (vz <= .05) & (roll.abs() <= .10) & (pitch.abs() <= .10) & (support == 2)
            speed_good = (vx >= torch.maximum(torch.full_like(speed_t, .20), .75 * speed_t)) & ((vx - speed_t).abs() <= .20)
            walk_good = speed_good & (heading_error.abs() <= .12) & (roll.abs() <= .20) & (pitch.abs() <= .20)
            for i in range(n):
                if done[i]:
                    continue
                p = int(phase[i])
                trace = traces[i]
                delta_xy = robot.data.root_pos_w.torch[i, :2] - origin_xy[i]
                cross_track = -torch.sin(target_heading[i]) * delta_xy[0] + torch.cos(target_heading[i]) * delta_xy[1]
                values = {
                    "phase": p, "cmd": float(vx_cmd[i]), "legacy_cmd": float(legacy[i, 9]), "yaw_cmd": float(yaw_cmd[i]), "legacy_yaw": float(legacy[i, 11]), "vx": float(vx[i]), "heading_error": abs(float(heading_error[i])),
                    "lateral": abs(float(cross_track)), "contact": int(contact_code[i]), "slip_left": float(slips[i, 0]) if contacts[i, 0] else 0.0, "slip_right": float(slips[i, 1]) if contacts[i, 1] else 0.0,
                    "vel_ratio": float(velocity_ratio[i].amax()), "ankle_pitch": float(torque_ratio[i, ankle_pitch].amax()), "ankle_roll": float(torque_ratio[i, ankle_roll].amax()), "knee": float(velocity_ratio[i, knees].amax()),
                    "height": float(robot.data.root_pos_w.torch[i, 2]), "action_rate": float(action_rate[i]), "yaw_rate": float(current_yaw_rate[i]),
                }
                for key, value in values.items():
                    trace[key].append(value)
                if terminal[i]:
                    fallen[i] = True
                    done[i] = True
                    continue
                if p == 0:
                    streak[i] = streak[i] + 1 if safe[i] else 0
                    if streak[i] * dt >= .4:
                        target_heading[i] = current_heading[i]
                        phase[i] = 1
                        elapsed[i] = 0
                        streak[i] = 0
                    elif elapsed[i] >= 2.0:
                        done[i] = True
                elif p == 1 and elapsed[i] >= initial_holds[i]:
                    phase[i] = 2
                    elapsed[i] = 0
                elif p == 2 and elapsed[i] >= RAMP_DURATION_S:
                    phase[i] = 3
                    elapsed[i] = 0
                elif p == 3:
                    streak[i] = streak[i] + 1 if walk_good[i] else 0
                    if streak[i] * dt >= .4:
                        acquired[i] = True
                        phase[i] = 4
                        elapsed[i] = 0
                        streak[i] = 0
                    elif elapsed[i] >= 3.0:
                        phase[i] = 5
                        elapsed[i] = 0
                elif p == 4 and elapsed[i] >= walk_holds[i]:
                    phase[i] = 5
                    elapsed[i] = 0
                elif p == 5 and elapsed[i] >= RAMP_DURATION_S:
                    phase[i] = 6
                    elapsed[i] = 0
                elif p == 6:
                    streak[i] = streak[i] + 1 if safe[i] else 0
                    if streak[i] * dt >= .4:
                        phase[i] = 7
                        elapsed[i] = 0
                        streak[i] = 0
                    elif elapsed[i] >= 3.0:
                        done[i] = True
                elif p == 7 and elapsed[i] >= final_holds[i]:
                    done[i] = True
                elapsed[i] += dt
            if bool(done.all()):
                break

        rows = []
        for i, trace in enumerate(traces):
            ids = [index for index, p in enumerate(trace["phase"]) if p in (3, 4)]
            values = lambda key: [trace[key][index] for index in ids]
            vx_values = values("vx")
            contacts = values("contact")
            support_switches = sum(a != b for a, b in zip(contacts, contacts[1:]))
            liftoff = any(code != 3 for code in contacts)
            speed_good_fraction = mean([v >= max(.20, .75 * speeds[i]) and abs(v - speeds[i]) <= .20 for v in vx_values])
            velocity_sat = mean([v >= .95 for v in values("vel_ratio")])
            ankle_pitch_sat = mean([v >= .95 for v in values("ankle_pitch")])
            ankle_roll_sat = mean([v >= .95 for v in values("ankle_roll")])
            saturation = velocity_sat > VELOCITY_SATURATION_FRACTION or max(ankle_pitch_sat, ankle_roll_sat) > ANKLE_TORQUE_SATURATION_FRACTION
            heading_p95 = percentile(values("heading_error"), 95)
            lateral = max(values("lateral"), default=0.0)
            flight = mean([code == 0 for code in contacts])
            slips_left = values("slip_left")
            slips_right = values("slip_right")
            slip_mean = mean(slips_left + slips_right)
            sustained = bool(len(ids) * dt >= SPEED_SUSTAIN_S and liftoff and support_switches >= 2 and speed_good_fraction >= .80 and heading_p95 <= .12 and lateral <= .50 and not fallen[i] and not saturation and flight <= .25 and slip_mean <= FOOT_SLIP_MEAN_LIMIT_MPS)
            final_ids = [index for index, p in enumerate(trace["phase"]) if p == 7]
            final_speed = mean([abs(trace["vx"][index]) for index in final_ids])
            final_double = bool(final_ids and trace["contact"][final_ids[-1]] == 3)
            stand_recovery = bool(final_ids and len(final_ids) * dt >= final_holds[i] - dt and final_speed <= .08 and final_double and not fallen[i])
            yaw_values = values("yaw_rate")
            yaw_reversals = sum(a * b < 0.0 and abs(a - b) > .02 for a, b in zip(yaw_values, yaw_values[1:]))
            row = {
                "episode": i, "mode": args.mode, "heading_mode": args.heading_mode, "controller_id": configs[i][0], "target_speed_mps": speeds[i], "ramp_duration_s": RAMP_DURATION_S,
                "liftoff": liftoff, "support_switches": support_switches, "sustained_walk": sustained, "acquired": bool(acquired[i]), "actual_speed_mean_mps": mean(vx_values), "actual_speed_p95_mps": percentile(vx_values, 95),
                "speed_error_mean_mps": mean([abs(v - speeds[i]) for v in vx_values]), "speed_good_fraction": speed_good_fraction, "heading_error_mean_rad": mean(values("heading_error")), "heading_error_p95_rad": heading_p95,
                "lateral_drift_max_m": lateral, "double_support_fraction": mean([code == 3 for code in contacts]), "single_support_fraction": mean([code in (1, 2) for code in contacts]), "flight_fraction": flight,
                "foot_slip_left_mean_mps": mean(slips_left), "foot_slip_right_mean_mps": mean(slips_right), "foot_slip_mean_mps": slip_mean, "velocity_saturation_fraction": velocity_sat,
                "ankle_pitch_torque_saturation_fraction": ankle_pitch_sat, "ankle_roll_torque_saturation_fraction": ankle_roll_sat, "knee_velocity_utilization_max": max(values("knee"), default=0.0),
                "pelvis_vertical_range_m": max(values("height"), default=0.0) - min(values("height"), default=0.0), "excessive_flight": flight > .25, "saturation_failure": saturation, "fall": bool(fallen[i]),
                "stand_recovery": stand_recovery, "final_speed_mps": final_speed, "final_double_support": final_double, "action_rate_p99": percentile(values("action_rate"), 99), "yaw_rate_p95_radps": percentile([abs(v) for v in yaw_values], 95),
                "yaw_reversal_frequency_hz": yaw_reversals / max(len(yaw_values) * dt, dt), "generated_legacy_vx_match": all(abs(trace["cmd"][k] - trace["legacy_cmd"][k]) < 1e-7 for k in range(len(trace["cmd"]))),
                "generated_legacy_yaw_match": all(abs(trace["yaw_cmd"][k] - trace["legacy_yaw"][k]) < 1e-7 for k in range(len(trace["yaw_cmd"]))),
            }
            row["classification"] = classify(row)
            rows.append(row)

        routing = {
            "checkpoint_sha256_match": before_hash == EXPECTED_SHA, "expert_weights_unchanged": sha256(checkpoint) == before_hash, "adapter_bitwise_every_step": adapter_exact,
            "generated_legacy_commands_match": all(r["generated_legacy_vx_match"] and r["generated_legacy_yaw_match"] for r in rows), "finite": finite,
            "absolute_world_xy_in_policy_observation": False, "target_heading_external_controller_state_only": True, "active_expert": "stage2_model_4246_only",
            "run_expert_loaded": False, "run_contribution_bitwise_zero": True, "bridge_contribution_bitwise_zero": True, "scripted_offset_bitwise_zero": True,
            "action_dimension": 37, "action_order_dimension": len(joint_names), "contact_sensor_update_period_s": float(sensor.cfg.update_period),
        }
        (output / "routing_preflight.json").write_text(json.dumps(routing, indent=2) + "\n")
        (output / "checkpoint_provenance.json").write_text(json.dumps({"path": str(checkpoint.relative_to(REPO)), "sha256_before": before_hash, "sha256_after": sha256(checkpoint), "copied": False, "modified": False}, indent=2) + "\n")

        if args.mode == "candidates":
            summaries = {}
            for candidate in CANDIDATES:
                candidate_rows = [r for r in rows if r["controller_id"] == candidate[0]]
                summaries[candidate[0]] = {"parameters": {"k_heading": candidate[1], "k_yaw_rate": candidate[2], "yaw_rate_limit_radps": candidate[3]}, "overall": summarize(candidate_rows), "per_speed": {str(speed): summarize([r for r in candidate_rows if r["target_speed_mps"] == speed]) for speed in (.8, 1.2)}}
            ranked = sorted(CANDIDATES, key=lambda c: (-summaries[c[0]]["overall"]["sustained_walk_rate"], summaries[c[0]]["overall"]["fall_rate"], summaries[c[0]]["overall"]["saturation_failure_rate"], summaries[c[0]]["overall"]["heading_error_p95_rad"], summaries[c[0]]["overall"]["lateral_drift_max_m"], summaries[c[0]]["overall"]["yaw_reversal_frequency_hz"], summaries[c[0]]["overall"]["action_rate_p99"]))
            chosen = ranked[0]
            definitions = {"speed_tracking": "vx >= max(0.20, 0.75*command) and abs(vx-command) <= 0.20 m/s for >=80% of >=2 s window", "heading_error_p95_max_rad": .12, "lateral_drift_max_m": .50, "minimum_support_switches": 2, "flight_fraction_max": .25, "velocity_saturation_fraction_max": VELOCITY_SATURATION_FRACTION, "ankle_torque_saturation_fraction_max": ANKLE_TORQUE_SATURATION_FRACTION, "contact_foot_slip_mean_max_mps": FOOT_SLIP_MEAN_LIMIT_MPS}
            (output / "heading_controller_candidates.json").write_text(json.dumps({"status": "DIAGNOSTIC_ONLY", "episodes_per_candidate_speed": 3, "candidates": summaries, "selection_order": [c[0] for c in ranked]}, indent=2) + "\n")
            safe_candidate = summaries[chosen[0]]["overall"]["sustained_walk_rate"] > 0.0 and summaries[chosen[0]]["overall"]["saturation_failure_rate"] <= .05
            selected = {"status": "FROZEN_BEFORE_SWEEP" if safe_candidate else "FROZEN_DIAGNOSTIC_LEAST_BAD_NOT_SAFE", "id": chosen[0], "k_heading": chosen[1], "k_yaw_rate": chosen[2], "yaw_rate_limit_radps": chosen[3], "selection_warning": None if safe_candidate else "All limited candidates failed the sustained-WALK/saturation criteria; this controller is retained only to map the envelope.", "target_heading_capture": "initial STAND settle completion", "equation": "clamp(K_heading*wrap(target-current)-K_yaw_rate*current_yaw_rate,+/-limit)", "not_a_turn_command": True, "ramp_duration_s": RAMP_DURATION_S, "success_definition": definitions}
            (output / "selected_heading_controller.json").write_text(json.dumps(selected, indent=2) + "\n")
            write_csv(output / "heading_candidate_episodes.csv", rows)
        elif args.mode == "sweep":
            per_speed = {str(speed): summarize([r for r in rows if r["target_speed_mps"] == speed]) for speed in SWEEP_SPEEDS}
            classifications = []
            safe_speeds = []
            for speed in SWEEP_SPEEDS:
                speed_rows = [r for r in rows if r["target_speed_mps"] == speed]
                counts = Counter(r["classification"] for r in speed_rows)
                dominant = counts.most_common(1)[0][0]
                summary = per_speed[str(speed)]
                safe = summary["fall_rate"] <= .05 and summary["saturation_failure_rate"] <= .05 and summary["heading_error_p95_rad"] <= .12 and summary["sustained_walk_rate"] >= .90
                if safe:
                    safe_speeds.append(speed)
                classifications.append({"target_speed_mps": speed, "dominant_classification": dominant, "no_liftoff": counts["no liftoff"], "isolated_step": counts["isolated step"], "sustained_walking": counts["sustained walking"], "unstable_walking": counts["unstable walking"], "saturation_dominated_walking": counts["saturation-dominated walking"], "safe_gate": safe})
            maximum_stand = max((speed for speed in SWEEP_SPEEDS if per_speed[str(speed)]["liftoff_rate"] < .5), default=None)
            minimum_reliable = min((speed for speed in SWEEP_SPEEDS if per_speed[str(speed)]["sustained_walk_rate"] >= .90), default=None)
            longest = []
            current = []
            for speed in SWEEP_SPEEDS:
                if speed in safe_speeds:
                    current.append(speed)
                else:
                    if len(current) > len(longest):
                        longest = current
                    current = []
            if len(current) > len(longest):
                longest = current
            transition_speeds = longest if len(longest) >= 2 else []
            (output / "steady_speed_sweep.json").write_text(json.dumps({"audit_only": True, "heading_controller": json.loads((output / "selected_heading_controller.json").read_text()), "episodes_per_speed": 5, "per_speed": per_speed}, indent=2) + "\n")
            write_csv(output / "steady_speed_episodes.csv", rows)
            write_csv(output / "command_response_classification.csv", classifications)
            (output / "dead_zone_summary.json").write_text(json.dumps({"maximum_command_that_remains_stand_mps": maximum_stand, "criterion": "liftoff rate <50% in five diagnostic episodes", "minimum_command_that_reliably_produces_walk_mps": minimum_reliable, "reliable_criterion": "sustained WALK >=90%", "not_a_capability_claim": True}, indent=2) + "\n")
            (output / "safe_upper_limit_summary.json").write_text(json.dumps({"safe_speeds_mps": safe_speeds, "maximum_safe_walk_speed_mps": max(safe_speeds, default=None), "criteria": {"fall_rate_max": .05, "saturation_failure_rate_max": .05, "heading_error_p95_max_rad": .12, "stable_walk_hold_min": .90}, "transition_pilot_speeds_mps": transition_speeds}, indent=2) + "\n")
            failure_counts = {"classification": dict(Counter(r["classification"] for r in rows)), "fall": sum(r["fall"] for r in rows), "saturation_failure": sum(r["saturation_failure"] for r in rows), "stand_recovery_failure": sum(not r["stand_recovery"] for r in rows)}
            (output / "failure_counts.json").write_text(json.dumps(failure_counts, indent=2) + "\n")
        elif args.mode == "compare":
            name = "comparison_zero_yaw_episodes.csv" if args.heading_mode == "ZeroYaw" else "comparison_fixed_heading_episodes.csv"
            write_csv(output / name, rows)
            if args.heading_mode == "FixedTarget":
                zero_rows = list(csv.DictReader((output / "comparison_zero_yaw_episodes.csv").open(encoding="utf-8")))
                bool_keys = ("liftoff", "sustained_walk", "fall", "saturation_failure", "stand_recovery")
                for row in zero_rows:
                    for key in bool_keys:
                        row[key] = row[key].lower() == "true"
                    for key in ("actual_speed_mean_mps", "actual_speed_p95_mps", "speed_error_mean_mps", "heading_error_p95_rad", "lateral_drift_max_m", "double_support_fraction", "single_support_fraction", "flight_fraction", "velocity_saturation_fraction", "ankle_pitch_torque_saturation_fraction", "ankle_roll_torque_saturation_fraction", "knee_velocity_utilization_max", "pelvis_vertical_range_m", "foot_slip_mean_mps", "action_rate_p99", "yaw_rate_p95_radps", "yaw_reversal_frequency_hz", "target_speed_mps"):
                        row[key] = float(row[key])
                comparison = {"seed": args.seed, "paired_speed_order": True, "zero_yaw": {"overall": summarize(zero_rows), "per_speed": {str(s): summarize([r for r in zero_rows if r["target_speed_mps"] == s]) for s in (.8, 1.0, 1.2)}}, "fixed_heading": {"overall": summarize(rows), "per_speed": {str(s): summarize([r for r in rows if r["target_speed_mps"] == s]) for s in (.8, 1.0, 1.2)}}}
                (output / "zero_yaw_vs_heading_feedback.json").write_text(json.dumps(comparison, indent=2) + "\n")
        else:
            write_csv(output / "transition_pilot_episodes.csv", rows)
            per_speed = {str(speed): summarize([r for r in rows if r["target_speed_mps"] == speed]) for speed in sorted(set(speeds))}
            transition_summary = {"status": "PILOT_ONLY_NOT_FORMAL", "episodes": len(rows), "overall": summarize(rows), "per_speed": per_speed}
            (output / "transition_pilot_summary.json").write_text(json.dumps(transition_summary, indent=2) + "\n")
            write_final_classification(output, transition_summary)
        wrapped.close()


if __name__ == "__main__":
    main()
