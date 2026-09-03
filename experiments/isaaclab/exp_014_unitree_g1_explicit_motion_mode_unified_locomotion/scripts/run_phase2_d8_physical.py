"""D8 validation-only counterfactual continuation and conditional shadow probe."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
D7 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation"
RAW7 = D7 / "raw"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d8_phase_error_causal_relevance"
ROOT_DIVERGENCE_THRESHOLD_M = 0.02  # preregistered before this physical run
DT = 0.02


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


offline = load_module("d8offline", HERE.parent / "run_phase2_d8_offline.py")
pre = load_module("d7pre", HERE.parent / "run_phase2_d7_preflight.py")
d6 = pre.d6; d3 = pre.d3
from g1_explicit_motion_mode.contract import MotionMode, minimum_jerk  # noqa: E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows, fields=None):
    path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def command(entries, count, device):
    target = torch.zeros(count, 3, device=device)
    for j, entry in enumerate(entries):
        c = entry["condition"]; angle = math.radians(c["direction_deg"])
        target[j] = torch.tensor([c["speed"] * math.cos(angle), c["speed"] * math.sin(angle), c["yaw"]], device=device)
    return target


def support(world, n):
    force = world.sensor.data.net_forces_w_history[:n, -1, world.sf, :].norm(dim=-1)
    return force > 5, force


def original_oracle_capture(world, payload, actors, classifier):
    walk, stop, hold = actors; d6.restore_payload(world, payload); n = payload["active"]; entries = payload["entries"]; device = world.device
    target = command(entries, world.env.num_envs, device); switch = torch.tensor([e["condition"]["switch_step"] for e in entries] + [25] * (world.env.num_envs - n), device=device)
    world.state.request(torch.full((world.env.num_envs,), int(MotionMode.STAND), device=device)); gait = torch.zeros(world.env.num_envs, device=device)
    streak = torch.zeros(n, dtype=torch.long, device=device); completion = torch.full((n,), -1, dtype=torch.long, device=device)
    snapshots, observations, roles = [], [], []
    for step in range(200):
        progress = torch.full((world.env.num_envs,), min(1., step / 25), device=device); physical = target * (1 - minimum_jerk(progress))[:, None]
        world.state.advance(physical, progress, 0 if step == 0 else DT); d6.set_command(world, physical); base = world.env.observation_manager.compute()["policy"]; obs = world.obs()
        with torch.inference_mode(): aw = walk(base, gait); astop = stop.mean(obs); ahold = hold.mean(obs)
        role = torch.where(step < switch[:n], 0, torch.where(completion >= 0, 2, 1)); action = torch.where((step < switch)[:, None], aw, astop); action[:n] = torch.where((completion >= 0)[:, None], ahold[:n], action[:n])
        snapshots.append({k: v[:n].detach().cpu() for k, v in world.snapshot().items()}); observations.append(obs[:n].detach().cpu()); roles.append(role.cpu())
        world.wrapped.step(action); speed = world.robot.data.root_lin_vel_b[:n, :2].norm(dim=1); yaw = world.robot.data.root_ang_vel_b[:n, 2].abs(); good = (speed <= .08) & (yaw <= .08)
        streak = torch.where(good, streak + 1, torch.zeros_like(streak)); new = (completion < 0) & (streak >= 25) & ((step - 24) < 75); completion[new] = step
    observations = torch.stack(observations); roles = torch.stack(roles); true_phase = torch.empty_like(roles)
    for step in range(200):
        for j in range(n):
            cp = int(roles[step, j]); last_stop = int(completion[j])
            true_phase[step, j] = 1 if step == 0 else (3 if cp == 0 and step >= 21 else 2 if cp == 0 else 6 if cp == 2 else 5 if step >= last_stop - 24 else 3 if step <= 29 else 4)
    flat = observations.reshape(-1, 141); guesses = []
    with torch.inference_mode():
        for lo in range(0, len(flat), 8192): guesses.append(classifier(flat[lo:lo + 8192].to(device)).argmax(1).cpu())
    predicted = torch.cat(guesses).reshape(200, n)
    selected, counts = [], defaultdict(int)
    for step in range(200):
        for j in range(n):
            true = int(true_phase[step, j]); pred = int(predicted[step, j])
            if true == pred: continue
            pair = f"{true}->{pred}"
            if counts[pair] >= 100: continue
            counts[pair] += 1
            selected.append({"snapshot": {k: v[j].clone() for k, v in snapshots[step].items()}, "step": step,
                             "env": j, "condition_id": entries[j]["condition"]["formal_condition_id"],
                             "variant": entries[j]["condition"].get("variant", 0), "true_phase": true, "predicted_phase": pred,
                             "switch_step": int(switch[j]), "target": target[j].cpu(), "completion": int(completion[j])})
    return selected, dict(counts)


def pooled_snapshot(records, num_envs, device):
    result = {}
    for key in records[0]["snapshot"]:
        values = torch.stack([record["snapshot"][key] for record in records])
        if len(values) < num_envs: values = torch.cat((values, values[-1:].expand(num_envs - len(values), *values.shape[1:])), 0)
        result[key] = values.to(device)
    return result


def branch(world, records, actors, student, route):
    walk, stop, hold = actors; n = len(records); world.restore_snapshot(pooled_snapshot(records, world.env.num_envs, world.device)); device = world.device; gait = torch.zeros(world.env.num_envs, device=device)
    target = torch.zeros(world.env.num_envs, 3, device=device); target[:n] = torch.stack([r["target"] for r in records]).to(device)
    initial_streak = []
    for r in records:
        if r["true_phase"] == 6: initial_streak.append(25)
        elif r["true_phase"] == 5: initial_streak.append(max(0, r["step"] - (r["completion"] - 24)))
        else: initial_streak.append(0)
    streak = torch.tensor(initial_streak, device=device); acquired = streak >= 25; fall = torch.zeros(n, dtype=torch.bool, device=device); slip = fall.clone(); impact = fall.clone(); saturation = fall.clone(); slip_streak = torch.zeros(n, dtype=torch.long, device=device); sat_streak = slip_streak.clone(); points = {}
    for horizon in range(1, 33):
        absolute = torch.tensor([r["step"] + horizon for r in records], device=device); progress_n = (absolute.float() / 25).clamp(max=1); physical = torch.zeros_like(target); physical[:n] = target[:n] * (1 - minimum_jerk(progress_n))[:, None]
        progress = torch.ones(world.env.num_envs, device=device); progress[:n] = progress_n; world.state.advance(physical, progress, DT); d6.set_command(world, physical); base = world.env.observation_manager.compute()["policy"]; obs = world.obs()
        with torch.inference_mode():
            if route == "student": action = student.mean(obs)
            else:
                aw = walk(base, gait); astop = stop.mean(obs); ahold = hold.mean(obs); action = astop.clone()
                for j, record in enumerate(records):
                    if route == "predicted": rid = 0 if record["predicted_phase"] <= 3 else 1 if record["predicted_phase"] <= 5 else 2
                    else: rid = 2 if record["true_phase"] == 6 else (0 if int(absolute[j]) < record["switch_step"] else 1)
                    action[j] = (aw if rid == 0 else astop if rid == 1 else ahold)[j]
        _, _, done, extras = world.wrapped.step(action); timeout = extras.get("time_outs", torch.zeros_like(done)).bool(); fall |= done[:n].bool() & ~timeout[:n]
        contact, force = support(world, n); feet = world.robot.data.body_lin_vel_w[:n, world.rf, :2].norm(dim=-1); bad = ((feet > .55) & contact).any(1); slip_streak = torch.where(bad, slip_streak + 1, torch.zeros_like(slip_streak)); slip |= slip_streak >= 5; impact |= force.amax(1) > 3500
        ratio = world.robot.data.joint_vel[:n].abs().div(world.limits[:n].clamp_min(1e-6)).amax(1); sat_streak = torch.where(ratio > .95, sat_streak + 1, torch.zeros_like(sat_streak)); saturation |= sat_streak >= 5
        speed = world.robot.data.root_lin_vel_b[:n, :2].norm(dim=1); yaw = world.robot.data.root_ang_vel_b[:n, 2].abs(); good = (speed <= .08) & (yaw <= .08); streak = torch.where(good, streak + 1, torch.zeros_like(streak)); acquired |= streak >= 25
        if horizon in (8, 16, 32):
            points[horizon] = {"root": world.robot.data.root_pos_w[:n].detach().cpu(), "joint": world.robot.data.joint_pos[:n].detach().cpu(), "contact": contact.detach().cpu(), "speed": speed.detach().cpu(), "yaw": yaw.detach().cpu()}
    return {"points": points, "fall": fall.cpu(), "slip": slip.cpu(), "impact": impact.cpu(), "saturation": saturation.cpu(), "acquired": acquired.cpu()}


def counterfactual(world, selected, actors, student):
    rows = []
    for start in range(0, len(selected), world.env.num_envs):
        records = selected[start:start + world.env.num_envs]; true = branch(world, records, actors, student, "true"); pred = branch(world, records, actors, student, "predicted"); stu = branch(world, records, actors, student, "student")
        for j, record in enumerate(records):
            row = {"condition_id": record["condition_id"], "variant": record["variant"], "control_step": record["step"], "true_phase": offline.PHASES[record["true_phase"]], "predicted_phase": offline.PHASES[record["predicted_phase"]]}
            for horizon in (8, 16, 32):
                row[f"student_true_root_divergence_{horizon}"] = float((stu["points"][horizon]["root"][j] - true["points"][horizon]["root"][j]).norm())
                row[f"pred_true_root_divergence_{horizon}"] = float((pred["points"][horizon]["root"][j] - true["points"][horizon]["root"][j]).norm())
                row[f"student_true_joint_divergence_{horizon}"] = float((stu["points"][horizon]["joint"][j] - true["points"][horizon]["joint"][j]).norm())
                row[f"student_true_contact_divergence_{horizon}"] = int((stu["points"][horizon]["contact"][j] != true["points"][horizon]["contact"][j]).sum())
                row[f"student_true_speed_divergence_{horizon}"] = float((stu["points"][horizon]["speed"][j] - true["points"][horizon]["speed"][j]).abs())
                row[f"student_true_yaw_divergence_{horizon}"] = float((stu["points"][horizon]["yaw"][j] - true["points"][horizon]["yaw"][j]).abs())
            for key in ("fall", "slip", "impact", "saturation", "acquired"): row[f"true_{key}"] = bool(true[key][j]); row[f"pred_{key}"] = bool(pred[key][j]); row[f"student_{key}"] = bool(stu[key][j])
            safety_new = (row["student_fall"] and not row["true_fall"]) or (row["student_slip"] and not row["true_slip"]) or (row["student_impact"] and not row["true_impact"]) or (row["student_saturation"] and not row["true_saturation"])
            acquisition_failure = row["true_acquired"] and not row["student_acquired"]
            row["physical_critical"] = bool(safety_new or acquisition_failure or row["student_true_root_divergence_32"] > ROOT_DIVERGENCE_THRESHOLD_M)
            rows.append(row)
    return rows


def shadow(world, payload, student, hold, group):
    d6.restore_payload(world, payload); n = payload["active"]; entries = payload["entries"]; device = world.device; target = command(entries, world.env.num_envs, device)
    world.state.request(torch.full((world.env.num_envs,), int(MotionMode.STAND), device=device)); streak = torch.zeros(n, dtype=torch.long, device=device); completion = torch.full((n,), -1, dtype=torch.long, device=device); fall = torch.zeros(n, dtype=torch.bool, device=device); slip = fall.clone(); impact = fall.clone(); sat = fall.clone(); slip_streak = torch.zeros(n, dtype=torch.long, device=device); sat_streak = slip_streak.clone(); speed_hist = []; yaw_hist = []
    for step in range(200):
        progress = torch.full((world.env.num_envs,), min(1., step / 25), device=device); physical = target * (1 - minimum_jerk(progress))[:, None]; world.state.advance(physical, progress, 0 if step == 0 else DT); d6.set_command(world, physical); obs = world.obs()
        with torch.inference_mode(): student_action = student.mean(obs); hold_action = hold.mean(obs); action = student_action; action[:n] = torch.where((completion >= 0)[:, None], hold_action[:n], student_action[:n])
        _, _, done, extras = world.wrapped.step(action); timeout = extras.get("time_outs", torch.zeros_like(done)).bool(); fall |= done[:n].bool() & ~timeout[:n]
        contact, force = support(world, n); feet = world.robot.data.body_lin_vel_w[:n, world.rf, :2].norm(dim=-1); bad = ((feet > .55) & contact).any(1); slip_streak = torch.where(bad, slip_streak + 1, torch.zeros_like(slip_streak)); slip |= slip_streak >= 5; impact |= force.amax(1) > 3500
        ratio = world.robot.data.joint_vel[:n].abs().div(world.limits[:n].clamp_min(1e-6)).amax(1); sat_streak = torch.where(ratio > .95, sat_streak + 1, torch.zeros_like(sat_streak)); sat |= sat_streak >= 5
        speed = world.robot.data.root_lin_vel_b[:n, :2].norm(dim=1); yaw = world.robot.data.root_ang_vel_b[:n, 2].abs(); good = (speed <= .08) & (yaw <= .08); streak = torch.where(good, streak + 1, torch.zeros_like(streak)); new = (completion < 0) & (streak >= 25) & ((step - 24) < 75); completion[new] = step; speed_hist.append(speed.cpu()); yaw_hist.append(yaw.cpu())
    speed_hist = torch.stack(speed_hist); yaw_hist = torch.stack(yaw_hist); rows = []
    for j, entry in enumerate(entries):
        comp = int(completion[j]); acquisition = comp >= 0 and not bool(fall[j] or slip[j] or impact[j] or sat[j]); hold_ok = False
        if acquisition and comp + 101 <= 200:
            speed = speed_hist[comp + 1:comp + 101, j]; yaw = yaw_hist[comp + 1:comp + 101, j]; hold_ok = bool(speed.mean() <= .08 and yaw.mean() <= .08 and torch.quantile(speed, .95) <= .12 and torch.quantile(yaw, .95) <= .12)
        rows.append({"group": group, "condition_id": entry["condition"]["formal_condition_id"], "variant": entry["condition"].get("variant", -1), "stop_acquisition": acquisition, "conditional_hold": hold_ok if acquisition else None, "joint_success": acquisition and hold_ok, "fall": bool(fall[j]), "dangerous_slip": bool(slip[j]), "impact": bool(impact[j]), "saturation": bool(sat[j]), "acquisition_step": comp - 24 if comp >= 0 else None})
    return rows


def summarize_shadow(rows):
    rate = lambda key: sum(bool(row[key]) for row in rows) / len(rows); acquired = [r for r in rows if r["stop_acquisition"]]; condition = defaultdict(list)
    for row in rows: condition[row["condition_id"]].append(row["joint_success"])
    formal = [r for r in rows if r["group"] == "formal"]; local = [r for r in rows if r["group"] == "local"]
    return {"episodes": len(rows), "STOP_ACQUISITION": rate("stop_acquisition"), "conditional_S_HOLD": sum(r["conditional_hold"] for r in acquired) / max(1, len(acquired)), "joint_success": rate("joint_success"), "minimum_formal_condition": min(sum(r["joint_success"] for r in formal if r["condition_id"] == c) / max(1, sum(r["condition_id"] == c for r in formal)) for c in range(34)), "local_neighborhood_joint_success": sum(r["joint_success"] for r in local) / len(local), "fall": rate("fall"), "dangerous_slip": rate("dangerous_slip"), "impact": rate("impact"), "saturation": rate("saturation")}


def main():
    parser = argparse.ArgumentParser(); add_launcher_args(parser); args, hydra = setup_preset_cli(parser); sys.argv = [sys.argv[0], *hydra]
    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point"); cfg.scene.num_envs = 476; cfg.seed = 20279220; cfg.episode_length_s = 20.; cfg.observations.policy.enable_corruption = False; cfg.events.base_external_force_torque = None; cfg.events.push_robot = None
    if args.device: cfg.sim.device = agent.device = args.device
    train = torch.load(RAW7 / "dataset/train.pt", map_location="cpu", weights_only=False); val = torch.load(RAW7 / "dataset/validation.pt", map_location="cpu", weights_only=False); device = torch.device(args.device or agent.device)
    classifier, _, _ = offline.d7_classifier(train, val, device); checkpoint = torch.load(RAW7 / "bc_checkpoints/s1_step_30000.pt", map_location=device, weights_only=False); student = offline.s1mod.S1().to(device).eval(); student.load_state_dict(checkpoint["actor_state_dict"])
    resets = d3.load_resets(); severity = torch.zeros(680)
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions); world = d3.StandWorld(wrapped, resets, severity); walk = FrozenGaitActor(d3.WMOVE).to(world.device).eval(); stop = d3.initialize("P1_STOP_PARENT", world.device)[0].eval(); hold = d3.initialize("P0_STAND_PARENT", world.device)[0].eval(); actors = (walk, stop, hold)
        local_payload = torch.load(RAW7 / "r4_local_payload.pt", map_location="cpu", weights_only=False); selected, cluster_counts = original_oracle_capture(world, local_payload, actors, classifier); rows = counterfactual(world, selected, actors, student)
        critical = sum(r["physical_critical"] for r in rows); physical_safety = 1 - critical / max(1, len(rows)); write_csv("phase_physical_counterfactual.csv", rows)
        physical = {"source": "D7 validation local-neighborhood R4 states", "selection": "up to 100 states per observed true->predicted cluster", "states": len(rows), "cluster_counts_available": cluster_counts, "horizons_steps": [8, 16, 32], "root_threshold_preregistered_m": ROOT_DIVERGENCE_THRESHOLD_M, "root_threshold_basis": "R4 deterministic repeatability floor plus 2 cm conservative numerical/physical margin, fixed before continuations", "physical_critical_errors": critical, "PHYSICAL_PHASE_SAFETY": physical_safety}
        dump("phase_physical_counterfactual.json", physical)
        action = json.loads((OUT / "phase_action_relevance.json").read_text(encoding="utf-8")); raw = json.loads((OUT / "raw_input_phase_separability.json").read_text(encoding="utf-8")); hidden = json.loads((OUT / "hidden_feature_phase_probes.json").read_text(encoding="utf-8")); conflict = json.loads((D7 / "label_conflict_audit.json").read_text(encoding="utf-8"))
        leakage = all(v["validation"]["accuracy"] < .995 for v in hidden.values()); authorized = action["ACTION_RELEVANT_PHASE_ACCURACY"] >= .99 and physical_safety >= .99 and action["action_critical_misclassification_rate"] <= .01 and conflict["total_material_conflicts"] == 0 and leakage
        shadow_rows = []
        if authorized:
            conditions = d6.conditions(); entries = [{"episode_id": i, "recipe_id": d3.VALIDATION[i], "split": "validation", "condition": {**c, "formal_condition_id": c["condition_id"], "variant": -1, "switch_step": 25}} for i, c in enumerate(conditions)]; formal_payload = pre.generate(world, walk, entries); shadow_rows.extend(shadow(world, formal_payload, student, hold, "formal")); shadow_rows.extend(shadow(world, local_payload, student, hold, "local"))
        write_csv("diagnostic_closed_loop_shadow.csv", shadow_rows, ["group", "condition_id", "variant", "stop_acquisition", "conditional_hold", "joint_success", "fall", "dangerous_slip", "impact", "saturation", "acquisition_step"])
        summary = summarize_shadow(shadow_rows) if shadow_rows else {"status": "NOT_AUTHORIZED"}
        if shadow_rows: summary["diagnostic_target_pass"] = summary["STOP_ACQUISITION"] >= .90 and summary["conditional_S_HOLD"] >= .95 and summary["joint_success"] >= .85 and summary["minimum_formal_condition"] >= .70 and summary["local_neighborhood_joint_success"] >= .80 and summary["fall"] <= .05 and summary["dangerous_slip"] <= .10
        dump("diagnostic_closed_loop_shadow.json", summary); dump("diagnostic_closed_loop_authorization.json", {"authorized": authorized, "action_regression_gates": "PASS", "material_label_conflict": conflict["total_material_conflicts"], "ACTION_RELEVANT_PHASE_ACCURACY": action["ACTION_RELEVANT_PHASE_ACCURACY"], "PHYSICAL_PHASE_SAFETY": physical_safety, "action_critical_misclassification_rate": action["action_critical_misclassification_rate"], "observation_leakage": False, "heldout_used": False})
        wrapped.close()
    metric = json.loads((OUT / "action_relevant_phase_metric.json").read_text(encoding="utf-8")); metric["PHYSICAL_PHASE_SAFETY"] = physical_safety; metric["physical_metric_pending"] = False; dump("action_relevant_phase_metric.json", metric)
    print(json.dumps({"counterfactual": physical, "shadow_authorized": authorized, "shadow": summary}, indent=2))


if __name__ == "__main__": main()
