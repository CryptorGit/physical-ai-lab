"""Isaac Lab worker for the one-time D14 D11R replacement evaluation.

The worker never owns durable state.  It requests an EPISODE_STARTED commit from
the parent before stepping physics and emits one result record per episode over
stdout.  The parent acknowledges each durable result before the worker proceeds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
D7 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation"
D13R = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d13r_seed_contract_correction"
SEALED = D13R / "d11r_sealed_payload.bin"
S1 = D7 / "raw/bc_checkpoints/s1_step_30000.pt"
EXPECTED_SEALED_SHA = "c6ef724da6fcafb25eb5c7d6a7b0b1ade17deb5cd4051a7fa16172c9465b9cfa"
DT = 0.02


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d10 = load_module("d10_d14", HERE.parent / "run_phase2_d10_frozen.py")
d6, d3, s1mod = d10.d6, d10.d3, d10.s1mod
from g1_explicit_motion_mode.contract import MotionMode, minimum_jerk  # noqa: E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def emit(kind: str, value: dict, wait_for_ack: bool = False) -> None:
    print("D14_IPC:" + json.dumps({"kind": kind, "value": value}, sort_keys=True, separators=(",", ":")), flush=True)
    if wait_for_ack:
        reply = sys.stdin.readline().strip()
        if reply != "D14_ACK":
            raise RuntimeError(f"parent persistence acknowledgement missing: {reply!r}")


def condition(entry: dict) -> dict:
    return {
        "condition_id": int(entry["condition_id"]),
        "kind": entry["condition_kind"],
        "direction_deg": float(entry["direction"]),
        "speed": float(entry["speed"]),
        "yaw": float(entry["yaw"]),
        "switch_time_s": float(entry["stop_timing"]),
    }


def command_matrix(entries: list[dict], count: int, device: torch.device) -> torch.Tensor:
    target = torch.zeros(count, 3, device=device)
    for index, entry in enumerate(entries):
        spec = entry["condition"]
        angle = math.radians(spec["direction_deg"])
        target[index] = torch.tensor(
            [spec["speed"] * math.cos(angle), spec["speed"] * math.sin(angle), spec["yaw"]],
            device=device,
        )
    return target


def generate_moving_start(world, walk, entries: list[dict], stop_timing: float) -> dict:
    """D11-family W_MOVE acquisition with the sealed STOP-timing perturbation."""
    active = len(entries)
    recipe_indices = [int(item["recipe_seed"]) % 680 for item in entries]
    padded = recipe_indices + [recipe_indices[-1]] * (world.env.num_envs - active)
    world.restore(torch.tensor(padded, device=world.device))
    device = world.device
    world.state.request(torch.full((world.env.num_envs,), int(MotionMode.WALK), device=device))
    target = command_matrix(entries, world.env.num_envs, device)
    gait = torch.zeros(world.env.num_envs, device=device)
    fall = torch.zeros(active, dtype=torch.bool, device=device)
    slip = fall.clone()
    slip_streak = torch.zeros(active, dtype=torch.long, device=device)
    stable_speed, stable_yaw = [], []
    # The D11 family reaches full command at step 75.  The sealed 0.45/0.50/0.55 s
    # value controls the result-blind steady-motion dwell before STOP_REQUEST.
    total_steps = 75 + int(math.ceil(stop_timing / DT))
    for step in range(total_steps):
        progress = torch.full((world.env.num_envs,), min(1.0, step / 75), device=device)
        physical = target * minimum_jerk(progress)[:, None]
        world.state.advance(physical, progress, DT)
        d6.set_command(world, physical)
        base = world.env.observation_manager.compute()["policy"]
        with torch.inference_mode():
            action = walk(base, gait)
        _, _, done, extras = world.wrapped.step(action)
        timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
        fall |= done[:active].bool() & ~timeout[:active]
        force = world.sensor.data.net_forces_w_history[:active, -1, world.sf, :].norm(dim=-1)
        contact = force > 5
        foot_speed = world.robot.data.body_lin_vel_w[:active, world.rf, :2].norm(dim=-1)
        bad_slip = ((foot_speed > 0.55) & contact).any(1)
        slip_streak = torch.where(bad_slip, slip_streak + 1, torch.zeros_like(slip_streak))
        slip |= slip_streak >= 5
        if step >= 75:
            stable_speed.append(world.robot.data.root_lin_vel_b[:active, :2].detach().cpu())
            stable_yaw.append(world.robot.data.root_ang_vel_b[:active, 2].detach().cpu())
    speed = torch.stack(stable_speed)
    yaw = torch.stack(stable_yaw)
    valid, invalid_reason = [], []
    for index, entry in enumerate(entries):
        expected = target[index].cpu()
        translation_ok = float((speed[:, index] - expected[:2]).norm(dim=1).mean()) <= 0.18
        yaw_ok = float((yaw[:, index] - expected[2]).abs().mean()) <= 0.18
        ok = bool(translation_ok and yaw_ok and not fall[index] and not slip[index])
        valid.append(ok)
        if fall[index]:
            invalid_reason.append("FALL_DURING_MOVING_START")
        elif slip[index]:
            invalid_reason.append("SLIP_DURING_MOVING_START")
        elif not translation_ok:
            invalid_reason.append("MOVING_TRANSLATION_NOT_ACQUIRED")
        elif not yaw_ok:
            invalid_reason.append("MOVING_YAW_NOT_ACQUIRED")
        else:
            invalid_reason.append(None)
    snapshot = {key: value.detach().cpu() for key, value in world.snapshot().items()}
    state_hashes = []
    for index in range(active):
        state_hashes.append(tensor_sha(*(snapshot[key][index:index + 1] for key in sorted(snapshot))))
    return {
        "snapshot": snapshot,
        "entries": entries,
        "active": active,
        "w_move_acquired": valid,
        "moving_invalid_reason": invalid_reason,
        "state_hashes": state_hashes,
        "recipe_indices": recipe_indices,
    }


def evaluate_stop(world, payload: dict, student, hold) -> list[dict]:
    d6.restore_payload(world, payload)
    active = payload["active"]
    entries = payload["entries"]
    device = world.device
    target = command_matrix(entries, world.env.num_envs, device)
    valid = torch.tensor(payload["w_move_acquired"], device=device)
    world.state.request(torch.full((world.env.num_envs,), int(MotionMode.STAND), device=device))
    streak = torch.zeros(active, dtype=torch.long, device=device)
    completion = torch.full((active,), -1, dtype=torch.long, device=device)
    first_entry = torch.full((active,), -1, dtype=torch.long, device=device)
    reexit = torch.zeros(active, dtype=torch.long, device=device)
    previous_good = torch.zeros(active, dtype=torch.bool, device=device)
    fall = torch.zeros(active, dtype=torch.bool, device=device)
    slip = fall.clone(); impact = fall.clone(); vsat = fall.clone(); tsat = fall.clone(); nonfinite = fall.clone()
    fall_onset = torch.full((active,), 999, dtype=torch.long, device=device)
    slip_onset = fall_onset.clone(); impact_onset = fall_onset.clone(); vsat_onset = fall_onset.clone(); tsat_onset = fall_onset.clone(); nan_onset = fall_onset.clone()
    slip_streak = torch.zeros(active, dtype=torch.long, device=device)
    velocity_streak = slip_streak.clone(); torque_streak = slip_streak.clone()
    speed, yaw = [], []
    handoff_l2 = torch.zeros(active, device=device); handoff_cos = torch.ones(active, device=device); joint_jump = torch.zeros(active, device=device)
    root_discontinuity = torch.zeros(active, dtype=torch.bool, device=device)
    contact_buffer_corruption = root_discontinuity.clone(); contact_change = root_discontinuity.clone(); handoff_failure = root_discontinuity.clone()
    previous_root = world.robot.data.root_pos_w[:active].clone()
    previous_contact = world.sensor.data.net_forces_w_history[:active, -1, world.sf, :].norm(dim=-1) > 5
    stop_action_at_completion = torch.zeros(active, 37, device=device)
    for step in range(200):
        progress = torch.full((world.env.num_envs,), min(1.0, step / 25), device=device)
        physical = target * (1 - minimum_jerk(progress))[:, None]
        world.state.advance(physical, progress, 0.0 if step == 0 else DT)
        d6.set_command(world, physical)
        observation = world.obs()
        with torch.inference_mode():
            student_action = student.mean(observation)
            hold_action = hold.mean(observation)
        newly_handoff = completion == step - 1
        if newly_handoff.any():
            delta = hold_action[:active] - stop_action_at_completion
            handoff_l2[newly_handoff] = delta[newly_handoff].norm(dim=1)
            handoff_cos[newly_handoff] = F.cosine_similarity(hold_action[:active][newly_handoff], stop_action_at_completion[newly_handoff])
            joint_jump[newly_handoff] = (delta[newly_handoff] * 0.5).norm(dim=1)
        action = student_action.clone()
        action[:active] = torch.where((completion >= 0)[:, None], hold_action[:active], student_action[:active])
        _, _, done, extras = world.wrapped.step(action)
        timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
        new_fall = done[:active].bool() & ~timeout[:active]
        force = world.sensor.data.net_forces_w_history[:active, -1, world.sf, :].norm(dim=-1)
        contact = force > 5
        foot_speed = world.robot.data.body_lin_vel_w[:active, world.rf, :2].norm(dim=-1)
        bad_slip = ((foot_speed > 0.55) & contact).any(1)
        slip_streak = torch.where(bad_slip, slip_streak + 1, torch.zeros_like(slip_streak))
        new_slip = slip_streak >= 5
        new_impact = force.amax(1) > 3500
        velocity_ratio = world.robot.data.joint_vel[:active].abs().div(world.limits[:active].clamp_min(1e-6)).amax(1)
        effort = world.robot.data.joint_effort_limits[:active].abs().clamp_min(1e-6)
        torque_ratio = world.robot.data.applied_torque[:active].abs().div(effort).amax(1)
        velocity_streak = torch.where(velocity_ratio > 0.95, velocity_streak + 1, torch.zeros_like(velocity_streak))
        torque_streak = torch.where(torque_ratio > 0.95, torque_streak + 1, torch.zeros_like(torque_streak))
        new_vsat = velocity_streak >= 5; new_tsat = torque_streak >= 5
        finite = torch.isfinite(action[:active]).all(1) & torch.isfinite(world.robot.data.root_state_w[:active]).all(1) & torch.isfinite(world.robot.data.joint_pos[:active]).all(1)
        new_nan = ~finite
        for flag, new, onset in ((fall, new_fall, fall_onset), (slip, new_slip, slip_onset), (impact, new_impact, impact_onset), (vsat, new_vsat, vsat_onset), (tsat, new_tsat, tsat_onset), (nonfinite, new_nan, nan_onset)):
            first = new & ~flag
            onset[first] = step
            flag |= new
        root_delta = (world.robot.data.root_pos_w[:active] - previous_root).norm(dim=1)
        root_discontinuity |= newly_handoff & (root_delta > 1.0)
        contact_change |= newly_handoff & ((contact != previous_contact).any(1))
        contact_buffer_corruption |= newly_handoff & ~torch.isfinite(force).all(1)
        handoff_failure |= newly_handoff & (new_fall | new_slip | new_impact)
        current_speed = world.robot.data.root_lin_vel_b[:active, :2].norm(dim=1)
        current_yaw = world.robot.data.root_ang_vel_b[:active, 2].abs()
        good = (current_speed <= 0.08) & (current_yaw <= 0.08)
        entering = good & (first_entry < 0)
        first_entry[entering] = step
        reexit += (previous_good & ~good & (first_entry >= 0)).long()
        previous_good = good
        streak = torch.where(good, streak + 1, torch.zeros_like(streak))
        newly_complete = (completion < 0) & (streak >= 25) & ((step - 24) < 75)
        completion[newly_complete] = step
        stop_action_at_completion[newly_complete] = student_action[:active][newly_complete]
        speed.append(current_speed.detach().cpu()); yaw.append(current_yaw.detach().cpu())
        previous_root = world.robot.data.root_pos_w[:active].clone(); previous_contact = contact.clone()
    speed = torch.stack(speed); yaw = torch.stack(yaw)
    rows = []
    for index, entry in enumerate(entries):
        complete = int(completion[index])
        acquisition_step = complete - 24 if complete >= 0 else None
        stop_safety = complete >= 0 and all(int(onset[index]) > complete for onset in (fall_onset, slip_onset, impact_onset, vsat_onset, tsat_onset, nan_onset))
        acquired = bool(complete >= 0 and stop_safety)
        hold_success = False
        hold_mean_speed = hold_p95_speed = hold_mean_yaw = hold_p95_yaw = None
        if acquired and complete + 101 <= 200:
            hold_speed = speed[complete + 1:complete + 101, index]
            hold_yaw = yaw[complete + 1:complete + 101, index]
            hold_mean_speed = float(hold_speed.mean()); hold_p95_speed = float(torch.quantile(hold_speed, 0.95))
            hold_mean_yaw = float(hold_yaw.mean()); hold_p95_yaw = float(torch.quantile(hold_yaw, 0.95))
            hold_safe = all(int(onset[index]) > complete + 100 for onset in (fall_onset, slip_onset, impact_onset, vsat_onset, tsat_onset, nan_onset))
            hold_success = bool(hold_safe and hold_mean_speed <= 0.08 and hold_p95_speed <= 0.12 and hold_mean_yaw <= 0.08 and hold_p95_yaw <= 0.12)
        moving_valid = bool(valid[index])
        joint = bool(acquired and hold_success)
        if nonfinite[index]: primary = "NON_FINITE"
        elif not moving_valid: primary = "MOVING_START_INVALID"
        elif complete < 0: primary = "STOP_ACQUISITION_FAILURE"
        elif not stop_safety:
            primary = "SAFETY_FAILURE_DURING_STOP"
        elif not acquired: primary = "STOP_CONFIRMATION_FAILURE"
        elif root_discontinuity[index] or contact_buffer_corruption[index] or handoff_failure[index]: primary = "S1_TO_SHOLD_HANDOFF_FAILURE"
        elif not hold_success: primary = "STAND_HOLD_FAILURE"
        else: primary = "PASS"
        rows.append({
            "episode_id": entry["episode_id"], "condition_id": int(entry["condition"]["condition_id"]),
            "direction": float(entry["condition"]["direction_deg"]), "yaw": float(entry["condition"]["yaw"]),
            "speed": float(entry["condition"]["speed"]), "stop_timing": float(entry["condition"]["switch_time_s"]),
            "recipe_id": entry["sealed_recipe_id"], "reset_recipe_index": int(entry["recipe_index"]),
            "snapshot_id": entry["snapshot_id"], "state_hash_at_stop_request": payload["state_hashes"][index],
            "moving_start_valid": moving_valid, "moving_start_invalid_reason": payload["moving_invalid_reason"][index],
            "stop_acquisition": acquired, "acquisition_step": acquisition_step,
            "acquisition_time_s": None if acquisition_step is None else acquisition_step * DT,
            "confirmation_success": bool(complete >= 0), "confirmation_step": None if complete < 0 else complete,
            "threshold_reexit_count": int(reexit[index]), "stand_hold_success": hold_success if acquired else None,
            "joint_success": joint, "end_to_end_success": bool(moving_valid and joint), "primary_failure": primary,
            "fall": bool(fall[index]), "dangerous_slip": bool(slip[index]), "impact": bool(impact[index]),
            "velocity_saturation": bool(vsat[index]), "torque_saturation": bool(tsat[index]), "nan_inf": bool(nonfinite[index]),
            "hold_mean_speed": hold_mean_speed, "hold_p95_speed": hold_p95_speed,
            "hold_mean_yaw": hold_mean_yaw, "hold_p95_yaw": hold_p95_yaw,
            "speed_mean_stop_window": float(speed[:100, index].mean()), "speed_p95_stop_window": float(torch.quantile(speed[:100, index], 0.95)),
            "yaw_mean_stop_window": float(yaw[:100, index].mean()), "yaw_p95_stop_window": float(torch.quantile(yaw[:100, index], 0.95)),
            "handoff_action_l2": float(handoff_l2[index]), "handoff_action_cosine": float(handoff_cos[index]),
            "joint_target_jump_rad_l2": float(joint_jump[index]), "root_state_discontinuity": bool(root_discontinuity[index]),
            "contact_state_changed_at_handoff": bool(contact_change[index]), "contact_buffer_corruption": bool(contact_buffer_corruption[index]),
            "handoff_new_safety_failure": bool(handoff_failure[index]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args, hydra = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]
    if file_sha(SEALED) != EXPECTED_SEALED_SHA:
        raise RuntimeError("sealed payload identity changed after parent pre-open")
    payload = json.loads(SEALED.read_bytes())
    requested = set(json.loads(os.environ["D14_EPISODE_IDS_JSON"]))
    selected = [item for item in payload["episodes"] if item["replacement_episode_id"] in requested]
    if len(selected) != len(requested):
        raise RuntimeError("requested episode identity mismatch")
    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 256; cfg.seed = 1940027935; cfg.episode_length_s = 20.0
    cfg.observations.policy.enable_corruption = False; cfg.events.base_external_force_torque = None; cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = agent.device = args.device
    checkpoint = torch.load(S1, map_location="cpu", weights_only=False)
    resets = d3.load_resets(); severity = torch.zeros(680)
    context_ok = False
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        world = d3.StandWorld(wrapped, resets, severity)
        walk = FrozenGaitActor(d3.WMOVE).to(world.device).eval()
        student = s1mod.S1().to(world.device).eval(); student.load_state_dict(checkpoint["actor_state_dict"])
        hold = d3.initialize("P0_STAND_PARENT", world.device)[0].eval()
        for stop_timing in (0.45, 0.50, 0.55):
            group = [item for item in selected if abs(float(item["stop_timing"]) - stop_timing) < 1e-9]
            for offset in range(0, len(group), world.env.num_envs):
                raw_entries = group[offset:offset + world.env.num_envs]
                entries = [{
                    **item, "episode_id": item["replacement_episode_id"], "sealed_recipe_id": item["recipe_id"],
                    "recipe_index": int(item["recipe_seed"]) % 680, "condition": condition(item),
                } for item in raw_entries]
                ids = [item["episode_id"] for item in entries]
                emit("START_REQUEST", {"episode_ids": ids, "stop_timing": stop_timing}, wait_for_ack=True)
                moving = generate_moving_start(world, walk, entries, stop_timing)
                rows = evaluate_stop(world, moving, student, hold)
                for row in rows:
                    emit("RESULT", row, wait_for_ack=True)
        wrapped.close()
        context_ok = True
    emit("WORKER_FINISHED", {"episodes": len(selected), "simulation_context_teardown": "PASS" if context_ok else "FAIL"})


if __name__ == "__main__":
    main()
