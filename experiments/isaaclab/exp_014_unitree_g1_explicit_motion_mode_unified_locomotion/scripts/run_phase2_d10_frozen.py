"""Frozen S1 teacher-free stop validation on fixed formal and local snapshots."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]
D6RAW = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher/raw"
D7RAW = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation/raw"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d10_s1_stop_closed_loop"; RAW = OUT / "raw"
S1_PATH = D7RAW / "bc_checkpoints/s1_step_30000.pt"; DT = .02


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


d6 = load_module("d6", HERE.parent / "run_phase2_d6_audit.py"); d3 = d6.d3
s1mod = load_module("s1mod", HERE.parent / "run_phase2_d7_s1_bc.py")
from g1_explicit_motion_mode.contract import MotionMode, minimum_jerk  # noqa: E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


def dump(path, value): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
def digest_tensor(h, tensor): h.update(tensor.detach().contiguous().cpu().numpy().tobytes())


def evaluate_batch(world, payload, student, hold, group):
    d6.restore_payload(world, payload); n = payload["active"]; entries = payload["entries"]; device = world.device
    snapshot_hash = payload.get("snapshot_hash") or d6.sha_bytes(*(value[:n] for value in payload["snapshot"].values()))
    target = d6.command_matrix(entries, world.env.num_envs, device); valid = torch.tensor(payload["w_move_acquired"], device=device)
    world.state.request(torch.full((world.env.num_envs,), int(MotionMode.STAND), device=device)); streak = torch.zeros(n, dtype=torch.long, device=device); completion = torch.full((n,), -1, dtype=torch.long, device=device)
    fall = torch.zeros(n, dtype=torch.bool, device=device); slip = fall.clone(); impact = fall.clone(); vsat = fall.clone(); tsat = fall.clone(); nan = fall.clone(); slip_streak = torch.zeros(n, dtype=torch.long, device=device); vstreak = slip_streak.clone(); tstreak = slip_streak.clone()
    speed, yaw, actions, supports = [], [], [], []; jump = torch.zeros(n, device=device); cosine = torch.ones(n, device=device); joint_jump = torch.zeros(n, device=device); root_discontinuity = torch.zeros(n, dtype=torch.bool, device=device); contact_discontinuity = torch.zeros(n, dtype=torch.bool, device=device); handoff_safety = torch.zeros(n, dtype=torch.bool, device=device)
    obs_hash = hashlib.sha256(); action_hash = hashlib.sha256(); previous_root = world.robot.data.root_pos_w[:n].clone(); previous_contact = world.sensor.data.net_forces_w_history[:n, -1, world.sf, :].norm(dim=-1) > 5
    for step in range(200):
        progress = torch.full((world.env.num_envs,), min(1., step / 25), device=device); physical = target * (1 - minimum_jerk(progress))[:, None]
        world.state.advance(physical, progress, 0 if step == 0 else DT); d6.set_command(world, physical); obs = world.obs()
        with torch.inference_mode(): student_action = student.mean(obs); hold_action = hold.mean(obs)
        newly = completion == step - 1
        if newly.any():
            delta = hold_action[:n] - student_action[:n]; jump[newly] = delta[newly].norm(dim=1); cosine[newly] = F.cosine_similarity(hold_action[:n][newly], student_action[:n][newly]); joint_jump[newly] = (delta[newly] * .5).norm(dim=1)
        action = student_action.clone(); action[:n] = torch.where((completion >= 0)[:, None], hold_action[:n], student_action[:n]); digest_tensor(obs_hash, obs[:n]); digest_tensor(action_hash, action[:n]); actions.append(action[:n].detach().cpu())
        _, _, done, extras = world.wrapped.step(action); timeout = extras.get("time_outs", torch.zeros_like(done)).bool(); new_fall = done[:n].bool() & ~timeout[:n]; fall |= new_fall
        force = world.sensor.data.net_forces_w_history[:n, -1, world.sf, :].norm(dim=-1); contact = force > 5; feet = world.robot.data.body_lin_vel_w[:n, world.rf, :2].norm(dim=-1); bad = ((feet > .55) & contact).any(1); slip_streak = torch.where(bad, slip_streak + 1, torch.zeros_like(slip_streak)); new_slip = slip_streak >= 5; slip |= new_slip; new_impact = force.amax(1) > 3500; impact |= new_impact
        velocity_ratio = world.robot.data.joint_vel[:n].abs().div(world.limits[:n].clamp_min(1e-6)).amax(1); effort = world.robot.data.joint_effort_limits[:n].abs().clamp_min(1e-6); torque_ratio = world.robot.data.applied_torque[:n].abs().div(effort).amax(1); vstreak = torch.where(velocity_ratio > .95, vstreak + 1, torch.zeros_like(vstreak)); tstreak = torch.where(torque_ratio > .95, tstreak + 1, torch.zeros_like(tstreak)); vsat |= vstreak >= 5; tsat |= tstreak >= 5
        finite = torch.isfinite(action[:n]).all(1) & torch.isfinite(world.robot.data.root_state_w[:n]).all(1) & torch.isfinite(world.robot.data.joint_pos[:n]).all(1); nan |= ~finite
        root_delta = (world.robot.data.root_pos_w[:n] - previous_root).norm(dim=1); if_handoff = newly; root_discontinuity |= if_handoff & (root_delta > 1.0); contact_discontinuity |= if_handoff & ((contact != previous_contact).any(1)); handoff_safety |= if_handoff & (new_fall | new_slip | new_impact)
        s = world.robot.data.root_lin_vel_b[:n, :2].norm(dim=1); y = world.robot.data.root_ang_vel_b[:n, 2].abs(); good = (s <= .08) & (y <= .08); streak = torch.where(good, streak + 1, torch.zeros_like(streak)); new = (completion < 0) & (streak >= 25) & ((step - 24) < 75); completion[new] = step
        speed.append(s.cpu()); yaw.append(y.cpu()); supports.append(contact.sum(1).cpu()); previous_root = world.robot.data.root_pos_w[:n].clone(); previous_contact = contact.clone()
    speed = torch.stack(speed); yaw = torch.stack(yaw); supports = torch.stack(supports); actions = torch.stack(actions); rows = []
    for j, entry in enumerate(entries):
        comp = int(completion[j]); safe = comp >= 0 and not bool(fall[j] or slip[j] or impact[j] or vsat[j] or tsat[j] or nan[j]); stop = bool(safe); hold_ok = False; hold_mean_speed = hold_p95_speed = hold_mean_yaw = hold_p95_yaw = None
        if stop and comp + 101 <= 200:
            hs = speed[comp + 1:comp + 101, j]; hy = yaw[comp + 1:comp + 101, j]; hold_mean_speed = float(hs.mean()); hold_p95_speed = float(torch.quantile(hs, .95)); hold_mean_yaw = float(hy.mean()); hold_p95_yaw = float(torch.quantile(hy, .95)); hold_ok = bool(hold_mean_speed <= .08 and hold_mean_yaw <= .08 and hold_p95_speed <= .12 and hold_p95_yaw <= .12 and not bool(fall[j] or slip[j] or impact[j] or vsat[j] or tsat[j] or nan[j]))
        condition = entry["condition"]; cid = condition.get("formal_condition_id", condition["condition_id"]); rows.append({"group": group, "condition_id": cid, "variant": condition.get("variant", -1), "snapshot_id": entry.get("snapshot_id", entry.get("episode_id", "")), "recipe_id": entry["recipe_id"], "snapshot_hash": snapshot_hash, "moving_start_valid": bool(valid[j]), "stop_acquisition": stop, "conditional_hold": hold_ok if stop else None, "joint_success": bool(stop and hold_ok), "end_to_end_success": bool(valid[j] and stop and hold_ok), "acquisition_step": comp - 24 if comp >= 0 else None, "confirmation_step": comp if comp >= 0 else None, "fall": bool(fall[j]), "dangerous_slip": bool(slip[j]), "impact": bool(impact[j]), "velocity_saturation": bool(vsat[j]), "torque_saturation": bool(tsat[j]), "nan_inf": bool(nan[j]), "handoff_action_l2": float(jump[j]), "handoff_action_cosine": float(cosine[j]), "joint_target_jump_rad_l2": float(joint_jump[j]), "root_state_discontinuity": bool(root_discontinuity[j]), "contact_continuity_change": bool(contact_discontinuity[j]), "handoff_new_safety_failure": bool(handoff_safety[j]), "hold_mean_speed": hold_mean_speed, "hold_p95_speed": hold_p95_speed, "hold_mean_yaw": hold_mean_yaw, "hold_p95_yaw": hold_p95_yaw, "speed_trajectory": speed[:, j].tolist(), "yaw_trajectory": yaw[:, j].tolist(), "support_trajectory": supports[:, j].tolist(), "action_hash": d6.sha_bytes(actions[:, j])})
    return rows, {"snapshot_hash": snapshot_hash, "observation_hash": obs_hash.hexdigest(), "action_hash": action_hash.hexdigest(), "active": n}


def summarize(rows, local=False):
    valid = [r for r in rows if r["moving_start_valid"]]; acquired = [r for r in valid if r["stop_acquisition"]]; rate = lambda key, data=valid: sum(bool(r[key]) for r in data) / max(1, len(data)); groups = {}
    for cid in range(34):
        data = [r for r in valid if r["condition_id"] == cid]; groups[str(cid)] = {"episodes": len(data), "joint_success": rate("joint_success", data), "stop_acquisition": rate("stop_acquisition", data)}
    summary = {"episodes": len(rows), "moving_start_valid": len(valid), "moving_start_validity": len(valid) / len(rows), "conditional_stop_success_given_valid_start": rate("stop_acquisition"), "conditional_hold": sum(bool(r["conditional_hold"]) for r in acquired) / max(1, len(acquired)), "joint_success": rate("joint_success"), "end_to_end_success": rate("end_to_end_success", rows), "minimum_condition_joint_success": min(x["joint_success"] for x in groups.values()), "fall": rate("fall"), "dangerous_slip": rate("dangerous_slip"), "impact": rate("impact"), "velocity_saturation": rate("velocity_saturation"), "torque_saturation": rate("torque_saturation"), "nan_inf": rate("nan_inf"), "physical_discontinuity": sum(r["root_state_discontinuity"] for r in rows), "handoff_new_safety_failure": sum(r["handoff_new_safety_failure"] for r in rows), "condition_groups": groups}
    if local: summary["gate_pass"] = summary["joint_success"] >= .85 and summary["minimum_condition_joint_success"] > 0
    else: summary["gate_pass"] = summary["conditional_stop_success_given_valid_start"] >= .95 and summary["conditional_hold"] >= .95 and summary["joint_success"] >= .90 and summary["fall"] <= .02 and summary["dangerous_slip"] <= .05 and summary["impact"] <= .05 and max(summary["velocity_saturation"], summary["torque_saturation"]) <= .05 and summary["minimum_condition_joint_success"] >= .80 and summary["physical_discontinuity"] == 0 and summary["handoff_new_safety_failure"] == 0
    return summary


def run_once(world, student, hold):
    formal_rows, local_rows, hashes = [], [], []
    for path in sorted(D6RAW.glob("validation_snapshot_batch_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False); rows, identity = evaluate_batch(world, payload, student, hold, "formal"); formal_rows.extend(rows); hashes.append(identity)
    payload = torch.load(D7RAW / "r4_local_payload.pt", map_location="cpu", weights_only=False); local_rows, identity = evaluate_batch(world, payload, student, hold, "local"); hashes.append(identity)
    return {"formal_rows": formal_rows, "local_rows": local_rows, "formal": summarize(formal_rows), "local": summarize(local_rows, True), "hashes": hashes}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-name", default="frozen_results.json"); parser.add_argument("--repeat", type=int, default=1); parser.add_argument("--separate-scenes", action="store_true"); add_launcher_args(parser); args, hydra = setup_preset_cli(parser); sys.argv = [sys.argv[0], *hydra]
    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point"); cfg.scene.num_envs = 476; cfg.seed = 20279301; cfg.episode_length_s = 20.; cfg.observations.policy.enable_corruption = False; cfg.events.base_external_force_torque = None; cfg.events.push_robot = None
    if args.device: cfg.sim.device = agent.device = args.device
    payload = torch.load(S1_PATH, map_location="cpu", weights_only=False); resets = d3.load_resets(); severity = torch.zeros(680); RAW.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        runs = []
        scene_count = args.repeat if args.separate_scenes else 1
        for scene in range(scene_count):
            wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions); world = d3.StandWorld(wrapped, resets, severity); student = s1mod.S1().to(world.device).eval(); student.load_state_dict(payload["actor_state_dict"]); hold = d3.initialize("P0_STAND_PARENT", world.device)[0].eval()
            runs.extend([run_once(world, student, hold) for _ in range(1 if args.separate_scenes else args.repeat)]); wrapped.close()
        result = {"checkpoint": str(S1_PATH.relative_to(REPO)).replace("\\", "/"), "checkpoint_sha256": sha(S1_PATH), "architecture": payload["architecture"], "step": payload["step"], "runs": runs}; dump(RAW / args.output_name, result)
        print(json.dumps({"output": args.output_name, "separate_scenes": args.separate_scenes, "runs": [{"formal": r["formal"], "local": r["local"]} for r in runs]}, indent=2))


if __name__ == "__main__": main()
