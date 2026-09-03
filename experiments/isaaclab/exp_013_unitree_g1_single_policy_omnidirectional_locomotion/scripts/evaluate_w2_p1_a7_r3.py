"""Fresh-process ReplayV2 evaluator for A7-R3 rescue and authorization."""
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
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
M0 = BASE / "phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
TEACHER = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path[:0] = [
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    str(EXP / "src"),
]
import isaaclab_tasks  # noqa: F401,E402
import g1_omnidirectional.tasks  # noqa: F401,E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--split", choices=("validation", "heldout"), required=True)
parser.add_argument("--direction", type=float, required=True)
parser.add_argument("--speed", type=float, required=True)
parser.add_argument("--yaw", type=float, required=True)
parser.add_argument("--episodes", type=int, required=True)
parser.add_argument("--group", default="start_matrix")
parser.add_argument("--output", required=True)
parser.add_argument("--diagnostic-output")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]

N = 1024
ROLL_IN = 150
RAMP = 75
MOVING_HOLD = 200
CONTROL_DT = 0.02


def minimum_jerk(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def first_sustained(pass_trace: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first window end index and whether it exists for [time, env]."""
    streak = torch.zeros(pass_trace.shape[1], dtype=torch.long, device=pass_trace.device)
    first = torch.full_like(streak, -1)
    for step in range(pass_trace.shape[0]):
        streak = torch.where(pass_trace[step], streak + 1, torch.zeros_like(streak))
        hit = (streak >= width) & (first < 0)
        first[hit] = step - width + 1
    return first, first >= 0


cfg, agent_cfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = N
cfg.episode_length_s = 12.0
cfg.seed = 20278501
cfg.observations.policy.enable_corruption = False
if args.device:
    cfg.sim.device = agent_cfg.device = args.device
masks = json.loads((M0 / "a7_environment_masks.json").read_text(encoding="utf-8"))["batches"][str(args.batch)]
split_mask_cpu = torch.tensor(masks[f"{args.split}_mask"], dtype=torch.bool)
active_ids_cpu = torch.nonzero(split_mask_cpu).flatten()[: args.episodes]
active_cpu = torch.zeros(N, dtype=torch.bool)
active_cpu[active_ids_cpu] = True
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)

with launch_simulation(cfg, args):
    wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent_cfg.clip_actions)
    env = wrapped.unwrapped
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    sensor_feet = sensor.find_bodies(".*_ankle_roll_link")[0]
    robot_feet = robot.find_bodies(".*_ankle_roll_link")[0]
    command_term = env.command_manager.get_term("base_velocity")
    command_term.external_override_enabled = True
    teacher = FrozenGaitActor(TEACHER).to(env.device).eval()
    gait = torch.zeros(N, device=env.device)
    env_ids = torch.arange(N, device=env.device)
    limits = robot.data.joint_vel_limits
    limits = limits[..., 1].abs() if limits.ndim == 3 else limits

    # ReplayV2: the policy object is not constructed until the full-batch stop identity exists.
    for _batch_id in range(args.batch + 1):
        env.reset(env_ids=env_ids)
        command_term.external_override.zero_()
        command_term._update_command()
        observations = wrapped.get_observations().to(env.device)
        for _ in range(ROLL_IN):
            with torch.inference_mode():
                teacher_action = teacher(observations["policy"], gait)
            observations, _, _, _ = wrapped.step(teacher_action)
            observations = observations.to(env.device)

    policy = FrozenGaitActor(Path(args.policy)).to(env.device).eval()
    active = active_cpu.to(env.device)
    target = torch.zeros(N, 3, device=env.device)
    radians = math.radians(args.direction)
    target[active, 0] = args.speed * math.cos(radians)
    target[active, 1] = args.speed * math.sin(radians)
    target[active, 2] = args.yaw

    fall = torch.zeros(N, dtype=torch.bool, device=env.device)
    slip = fall.clone()
    impact = fall.clone()
    saturation = fall.clone()
    slip_streak = torch.zeros(N, dtype=torch.long, device=env.device)
    saturation_streak = slip_streak.clone()
    endpoint_count = torch.zeros(N, dtype=torch.long, device=env.device)
    endpoint_velocity = torch.zeros(N, 2, device=env.device)
    endpoint_vector_error = torch.zeros(N, device=env.device)
    endpoint_yaw = torch.zeros(N, device=env.device)
    endpoint_yaw_error = torch.zeros(N, device=env.device)
    component_traces: dict[str, list[torch.Tensor]] = {name: [] for name in ("translation", "direction", "yaw", "gait_safety", "combined")}
    diagnostic_rows: list[dict[str, object]] = []
    diagnostic_yaw_errors: list[torch.Tensor] = []
    diagnostic_contact_phases: list[torch.Tensor] = []
    previous_mean_action: torch.Tensor | None = None

    for step in range(RAMP + MOVING_HOLD):
        alpha = minimum_jerk(torch.tensor(step / RAMP, device=env.device))
        physical = target * alpha
        actor_command = physical.clone()
        actor_command[:, 2] = torch.where(actor_command[:, 2] > 0, actor_command[:, 2] * 1.5, actor_command[:, 2])
        command_term.external_override.zero_()
        command_term.external_override[active] = actor_command[active]
        command_term._update_command()
        observations = wrapped.get_observations().to(env.device)
        with torch.inference_mode():
            policy_action = policy(observations["policy"], gait)
            housekeeping_action = teacher(observations["policy"], gait)
            action = torch.where(active[:, None], policy_action, housekeeping_action)
        observations, _, done, extras = wrapped.step(action)
        observations = observations.to(env.device)
        timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
        fall |= done.bool() & ~timeout
        force = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].norm(dim=-1)
        contact = force > 5
        foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
        bad_slip = ((foot_speed > 0.55) & contact).any(1)
        slip_streak = torch.where(bad_slip, slip_streak + 1, torch.zeros_like(slip_streak))
        slip |= slip_streak >= 5
        impact |= force.amax(1) > 3500
        velocity_ratio = robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(1)
        saturation_streak = torch.where(velocity_ratio > 0.95, saturation_streak + 1, torch.zeros_like(saturation_streak))
        saturation |= saturation_streak >= 5

        actual = robot.data.root_lin_vel_b[:, :2]
        actual_yaw = robot.data.root_ang_vel_b[:, 2]
        vector_error = torch.linalg.vector_norm(actual - physical[:, :2], dim=1)
        actual_speed = torch.linalg.vector_norm(actual, dim=1)
        target_speed = torch.linalg.vector_norm(physical[:, :2], dim=1)
        target_angle = torch.atan2(physical[:, 1], physical[:, 0])
        actual_angle = torch.atan2(actual[:, 1], actual[:, 0])
        direction_error = torch.atan2(torch.sin(actual_angle - target_angle), torch.cos(actual_angle - target_angle)).abs() * 180 / math.pi
        translation_pass = torch.where(target_speed < 1e-8, actual_speed <= 0.08, vector_error <= 0.25)
        direction_pass = torch.where(target_speed < 1e-8, torch.ones_like(translation_pass), direction_error <= 25)
        yaw_pass = torch.where(
            physical[:, 2].abs() < 1e-8,
            actual_yaw.abs() <= 0.20,
            (torch.sign(actual_yaw) == torch.sign(physical[:, 2])) & ((actual_yaw - physical[:, 2]).abs() <= 0.20),
        )
        gait_safety_pass = contact.any(1) & ~fall & ~slip & ~impact
        combined = translation_pass & direction_pass & yaw_pass & gait_safety_pass
        if args.diagnostic_output:
            selected = active
            contact_phase = contact[:, 0].long() + 2 * contact[:, 1].long()
            mean_action = policy_action[selected].mean(0)
            action_derivative = torch.zeros_like(mean_action) if previous_mean_action is None else mean_action - previous_mean_action
            previous_mean_action = mean_action.clone()
            diagnostic_yaw_errors.append((actual_yaw[selected] - physical[selected, 2]).detach().cpu())
            diagnostic_contact_phases.append(contact_phase[selected].detach().cpu())
            diagnostic_rows.append({
                "control_step": step, "time_s": step * CONTROL_DT,
                "physical_vx_target": float(physical[selected, 0].mean()), "physical_vy_target": float(physical[selected, 1].mean()),
                "physical_yaw_target": float(physical[selected, 2].mean()), "actor_yaw_input": float(actor_command[selected, 2].mean()),
                "actual_vx": float(actual[selected, 0].mean()), "actual_vy": float(actual[selected, 1].mean()),
                "actual_speed": float(actual_speed[selected].mean()), "translation_vector_mae": float(vector_error[selected].mean()),
                "direction_error_deg": float(direction_error[selected].mean()), "actual_yaw_rate": float(actual_yaw[selected].mean()),
                "yaw_mae": float((actual_yaw[selected] - physical[selected, 2]).abs().mean()),
                "yaw_sign_correct": float((torch.sign(actual_yaw[selected]) == torch.sign(physical[selected, 2])).float().mean()),
                "translation_pass": float(translation_pass[selected].float().mean()), "direction_pass": float(direction_pass[selected].float().mean()),
                "yaw_pass": float(yaw_pass[selected].float().mean()), "gait_safety_pass": float(gait_safety_pass[selected].float().mean()),
                "combined_pass": float(combined[selected].float().mean()),
                "left_contact": float(contact[selected, 0].float().mean()), "right_contact": float(contact[selected, 1].float().mean()),
                **{f"action_{joint:02d}": float(value) for joint, value in enumerate(mean_action)},
                **{f"action_derivative_{joint:02d}": float(value) for joint, value in enumerate(action_derivative)},
            })
        if step >= RAMP:
            for key, value in (
                ("translation", translation_pass), ("direction", direction_pass), ("yaw", yaw_pass),
                ("gait_safety", gait_safety_pass), ("combined", combined),
            ):
                component_traces[key].append(value.clone())
        if step >= 175:
            endpoint_count += active
            endpoint_velocity += actual * active[:, None]
            endpoint_vector_error += vector_error * active
            endpoint_yaw += actual_yaw * active
            endpoint_yaw_error += (actual_yaw - target[:, 2]).abs() * active

    traces = {key: torch.stack(value) for key, value in component_traces.items()}
    active_ids = torch.nonzero(active).flatten()
    combined = traces["combined"][:, active_ids]
    acquisition_combined = combined[:150]
    acquisition_yaw = traces["yaw"][:150, active_ids]
    longest = torch.zeros(len(active_ids), dtype=torch.long, device=env.device)
    streak = torch.zeros_like(longest)
    resets = torch.zeros_like(longest)
    for frame in acquisition_yaw:
        resets += (~frame) & (streak > 0)
        streak = torch.where(frame, streak + 1, torch.zeros_like(streak))
        longest = torch.maximum(longest, streak)

    endpoint_count = endpoint_count.clamp_min(1)
    mean_velocity = endpoint_velocity / endpoint_count[:, None]
    mean_yaw = endpoint_yaw / endpoint_count
    mean_vector_error = endpoint_vector_error / endpoint_count
    mean_yaw_error = endpoint_yaw_error / endpoint_count
    mean_speed = torch.linalg.vector_norm(mean_velocity, dim=1)
    mean_angle = torch.atan2(mean_velocity[:, 1], mean_velocity[:, 0])
    target_angle = torch.atan2(target[:, 1], target[:, 0])
    mean_direction_error = torch.atan2(torch.sin(mean_angle - target_angle), torch.cos(mean_angle - target_angle)).abs() * 180 / math.pi
    target_speed = torch.linalg.vector_norm(target[:, :2], dim=1)
    endpoint_translation = torch.where(target_speed < 1e-8, mean_speed <= 0.08, (mean_vector_error <= 0.25) & (mean_direction_error <= 25))
    endpoint_yaw_ok = torch.where(target[:, 2].abs() < 1e-8, mean_yaw.abs() <= 0.20, (torch.sign(mean_yaw) == torch.sign(target[:, 2])) & (mean_yaw_error <= 0.20))
    endpoint_ok = endpoint_translation & endpoint_yaw_ok & ~fall & ~slip & ~impact & ~saturation

    row: dict[str, object] = {
        "group": args.group, "direction": args.direction, "speed": args.speed, "yaw": args.yaw,
        "episodes": len(active_ids), "endpoint_success": float(endpoint_ok[active_ids].float().mean()),
        "yaw_timer_resets": float(resets.float().mean()), "longest_yaw_pass_s": float(longest.float().mean() * CONTROL_DT),
        "yaw_mae": float(mean_yaw_error[active_ids].mean()), "fall_rate": float(fall[active_ids].float().mean()),
        "dangerous_slip_rate": float(slip[active_ids].float().mean()), "impact_rate": float(impact[active_ids].float().mean()),
        "saturation_rate": float(saturation[active_ids].float().mean()),
    }
    for width, label in ((5, "0p10"), (8, "0p15"), (10, "0p20"), (13, "0p25")):
        first, passed = first_sustained(acquisition_combined, width)
        row[f"acquisition_{label}"] = float(passed.float().mean())
        valid = first[passed].float() * CONTROL_DT
        row[f"acquisition_{label}_median_s"] = float(valid.median()) if len(valid) else None
        row[f"acquisition_{label}_p95_s"] = float(torch.quantile(valid, 0.95)) if len(valid) else None
    for component, trace in traces.items():
        first, passed = first_sustained(trace[:150, active_ids], 10)
        row[f"{component}_sustained_0p20"] = float(passed.float().mean())
        row[f"{component}_final_hold_pass_fraction"] = float(trace[-100:, active_ids].float().mean())

    component_passes = {component: first_sustained(trace[:150, active_ids], 10)[1] for component, trace in traces.items()}
    combined_0p10 = first_sustained(traces["combined"][:150, active_ids], 5)[1]
    attribution = []
    for local_index, env_id in enumerate(active_ids.tolist()):
        if component_passes["combined"][local_index]: category = "PASS"
        else:
            failed = [name for name in ("translation", "direction", "yaw", "gait_safety") if not component_passes[name][local_index]]
            if failed == ["translation"]: category = "TRANSLATION_VECTOR_LIMIT"
            elif failed == ["direction"]: category = "DIRECTION_LIMIT"
            elif failed == ["yaw"]: category = "YAW_RATE_OSCILLATION" if combined_0p10[local_index] else "YAW_SIGN_LIMIT"
            elif failed == ["gait_safety"]: category = "GAIT_LIMIT" if not (fall[env_id] or slip[env_id] or impact[env_id]) else "SAFETY_LIMIT"
            elif not failed and combined_0p10[local_index]: category = "SUSTAINED_WINDOW_ONLY"
            else: category = "MULTIPLE_COMPONENTS"
        attribution.append({"episode_index": local_index, "environment_id": env_id, "category": category})

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    output.with_suffix(".json").write_text(json.dumps({"policy": args.policy, "batch": args.batch, "split": args.split, "row": row, "episode_attribution": attribution}, indent=2) + "\n", encoding="utf-8")
    if args.diagnostic_output:
        diagnostic_path = Path(args.diagnostic_output)
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        with diagnostic_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(diagnostic_rows[0])); writer.writeheader(); writer.writerows(diagnostic_rows)
        yaw_error_tensor = torch.stack(diagnostic_yaw_errors)[RAMP:]
        phase_tensor = torch.stack(diagnostic_contact_phases)[RAMP:]
        mean_signal = yaw_error_tensor.mean(1); centered = mean_signal - mean_signal.mean()
        spectrum = torch.fft.rfft(centered); frequencies = torch.fft.rfftfreq(len(centered), d=CONTROL_DT)
        dominant_index = int(spectrum.abs()[1:].argmax()) + 1 if len(spectrum) > 1 else 0
        phase_rows = []
        total_variance = float(yaw_error_tensor.var().clamp_min(1e-12))
        weighted_between = 0.0
        overall_mean = float(yaw_error_tensor.mean())
        for phase in range(4):
            phase_mask = phase_tensor == phase
            values = yaw_error_tensor[phase_mask]
            phase_mean = float(values.mean()) if len(values) else None
            phase_pass = float((values.abs() <= 0.20).float().mean()) if len(values) else None
            phase_rows.append({"contact_phase": phase, "samples": int(phase_mask.sum()), "yaw_error_mean": phase_mean, "yaw_pass_probability": phase_pass})
            if len(values): weighted_between += len(values) * (phase_mean - overall_mean) ** 2
        transitions = (phase_tensor[1:] != phase_tensor[:-1]).float().sum(0)
        diagnostic_json = {
            "condition": {"direction": args.direction, "speed": args.speed, "yaw": args.yaw},
            "phase_rows": phase_rows, "dominant_oscillation_frequency_hz": float(frequencies[dominant_index]),
            "stride_frequency_hz": float(transitions.mean() / (2 * MOVING_HOLD * CONTROL_DT)),
            "phase_locking_strength": float(weighted_between / (yaw_error_tensor.numel() * total_variance)),
            "joint_groups": {"legs": list(range(18)), "waist": list(range(18, 21)), "torso_arms": list(range(21, 33)), "hands": list(range(33, 37))},
        }
        diagnostic_path.with_suffix(".json").write_text(json.dumps(diagnostic_json, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2), flush=True)
    wrapped.close()
