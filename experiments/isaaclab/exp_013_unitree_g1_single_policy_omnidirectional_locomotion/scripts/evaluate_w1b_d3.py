"""Fresh-process dynamic-yaw diagnostics for the frozen W1B-R2 actor."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
EXP = HERE.parent.parent
OUT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d3_dynamic_yaw_transition_boundary_diagnosis"
)
CHECKPOINT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
sys.path.insert(0, str(ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(ROOT / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from g1_omnidirectional.yaw_calibration import calibrate_yaw  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=("step", "ramp", "dwell", "profile", "history", "backward", "random", "variance"))
parser.add_argument("--profile-dwell", type=float, default=.5)
parser.add_argument("--debug", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def minjerk(value):
    value = max(0.0, min(1.0, float(value)))
    return value**3 * (10 - 15 * value + 6 * value**2)


def smooth_calibration(value):
    if value <= -.1:
        gain = 1.0
    elif value >= .1:
        gain = 1.5
    else:
        blend = minjerk((value + .1) / .2)
        gain = 1.0 + .5 * blend
    return value * gain


def direction_name(angle):
    return "PURE" if angle is None else f"D{int(angle):03d}"


def spec(name, angle, speed, initial, final, episodes, transition="ramp", ramp=2.0, dwell=0.0, profile="C1", history=None):
    return {
        "name": name, "angle": angle, "speed": speed, "initial": initial, "final": final,
        "episodes": episodes, "transition": transition, "ramp": ramp, "dwell": dwell,
        "profile": profile, "history": history,
        "duration": 10.0 if transition == "step" else 4.0 + ramp + dwell + (ramp if transition == "dwell" else 0.0) + 6.0,
    }


def specs_for_mode():
    core = [(None, 0.0), (0, .3), (45, .3), (90, .3), (135, .3), (180, .3), (225, .3), (270, .3), (315, .3)]
    focus = [(None, 0.0), (0, .3), (90, .3), (180, .3), (270, .3)]
    if args.mode == "step":
        changes = [(0, .3), (0, -.3), (.3, 0), (-.3, 0), (-.3, .3), (.3, -.3),
                   (.15, .3), (.3, .15), (-.15, -.3), (-.3, -.15)]
        return [spec(f"STEP_{direction_name(a)}_{x:+.2f}_{y:+.2f}", a, s, x, y, 50, "step", 0)
                for a, s in core for x, y in changes]
    if args.mode == "ramp":
        return [spec(f"RAMP_{direction_name(a)}_{x:+.1f}_{y:+.1f}_{r:.2f}s", a, s, x, y, 50, "ramp", r)
                for a, s in focus for x, y in ((-.3, .3), (.3, -.3))
                for r in (.25, .5, 1, 1.5, 2, 3, 4)]
    if args.mode == "dwell":
        return [spec(f"DWELL_{direction_name(a)}_{x:+.1f}_{y:+.1f}_{d:.2f}s", a, s, x, y, 50, "dwell", 1, d)
                for a, s in focus for x, y in ((-.3, .3), (.3, -.3))
                for d in (0, .1, .25, .5, .75, 1, 1.5, 2)]
    if args.mode == "profile":
        return [spec(f"PROFILE_{p}_{direction_name(a)}_{x:+.1f}_{y:+.1f}", a, s, x, y, 100,
                     "dwell" if p == "C4" else "ramp", 1 if p == "C4" else 2,
                     args.profile_dwell if p == "C4" else 0, p)
                for p in ("C1", "C2", "C3", "C4") for a, s in focus
                for x, y in ((-.3, .3), (.3, -.3))]
    if args.mode == "history":
        out = []
        for a, s in focus:
            for final in (.3, -.3):
                histories = {
                    "STATIC": final,
                    "SAME_SIGN": math.copysign(.15, final),
                    "OPPOSITE_SIGN": -final,
                    "ZERO": 0.0,
                }
                for history, initial in histories.items():
                    transition = "static" if history == "STATIC" else "ramp"
                    out.append(spec(f"HISTORY_{history}_{direction_name(a)}_{final:+.1f}", a, s, initial, final,
                                    100, transition, 2, history=history))
        return out
    if args.mode == "backward":
        changes = [(0, .3), (0, -.3), (.3, 0), (-.3, 0), (-.3, .3), (.3, -.3)]
        return [spec(f"BACK_S{s:.1f}_{x:+.1f}_{y:+.1f}_{r:.1f}s", 180, s, x, y, 50, "ramp", r)
                for s in (.1, .2, .3, .4) for x, y in changes for r in (.5, 1, 2, 3, 4)]
    if args.mode == "variance":
        return [spec(f"VAR_BATCH_{batch:03d}", 0, 1.2, 0, 0, 50, "static", 0)
                for batch in range(100)]
    return [{"name": "RANDOM_60S", "episodes": 50, "duration": 60.0, "transition": "random",
             "profile": "C1", "angle": None, "speed": None, "initial": 0.0, "final": 0.0,
             "ramp": 0.0, "dwell": 0.0, "history": None}]


def command_at(item, time_s, episode):
    if item["transition"] == "random":
        generator = torch.Generator().manual_seed(20283021 + episode)
        durations = 3 + 2 * torch.rand(20, generator=generator)
        cumulative = torch.cumsum(durations, 0)
        segment = int(torch.searchsorted(cumulative, torch.tensor(time_s)).clamp(max=19))
        angle = float(torch.rand(20, generator=generator)[segment] * 2 * math.pi)
        speed = float(torch.rand(20, generator=generator)[segment] * .4)
        target = float(torch.rand(20, generator=generator)[segment] * .8 - .4)
        return speed * math.cos(angle), speed * math.sin(angle), target, float(calibrate_yaw(target)), segment
    angle = 0 if item["angle"] is None else math.radians(item["angle"])
    vx, vy = item["speed"] * math.cos(angle), item["speed"] * math.sin(angle)
    initial, final = item["initial"], item["final"]
    if item["transition"] in ("static",):
        physical = final
        actor = float(calibrate_yaw(physical))
    elif item["transition"] == "step":
        physical = initial if time_s < 4 else final
        actor = float(calibrate_yaw(physical))
    elif item["transition"] == "dwell":
        if time_s < 4:
            physical = initial
        elif time_s < 5:
            physical = initial * (1 - minjerk(time_s - 4))
        elif time_s < 5 + item["dwell"]:
            physical = 0.0
        elif time_s < 6 + item["dwell"]:
            physical = final * minjerk(time_s - 5 - item["dwell"])
        else:
            physical = final
        actor = float(calibrate_yaw(physical))
    else:
        blend = minjerk((time_s - 4) / item["ramp"])
        physical = initial + (final - initial) * blend
        if item["profile"] == "C2":
            actor = float(calibrate_yaw(initial)) + (float(calibrate_yaw(final)) - float(calibrate_yaw(initial))) * blend
        elif item["profile"] == "C3":
            actor = smooth_calibration(physical)
        else:
            actor = float(calibrate_yaw(physical))
    return vx, vy, physical, actor, 0


def chunks(items, capacity=1024):
    current, count = [], 0
    for item in items:
        if current and count + item["episodes"] > capacity:
            yield current
            current, count = [], 0
        current.append(item)
        count += item["episodes"]
    if current:
        yield current


def summarize(group):
    row = {key: group[0].get(key) for key in (
        "condition", "direction_deg", "speed", "initial_yaw", "final_yaw", "transition",
        "ramp_duration", "zero_dwell", "profile", "history",
    )}
    row["episodes"] = len(group)
    for key in ("whole_sign_fraction", "final_hold_sign_fraction", "sign_acquisition_delay",
                "response_delay", "time_constant", "settling_time", "overshoot",
                "steady_state_error", "integrated_absolute_yaw_error", "vector_mae",
                "final_vector_mae", "direction_error", "translation_degradation",
                "actor_input_max_derivative", "actor_input_max_acceleration"):
        values = [float(value[key]) for value in group]
        row[key] = sum(values) / len(values)
    for key in ("success", "fall", "dangerous_slip", "impact", "saturation"):
        row[key + "_rate"] = sum(bool(value[key]) for value in group) / len(group)
    contacts = defaultdict(int)
    for value in group:
        contacts[value["transition_contact_state"]] += 1
    row["contact_state_counts"] = dict(contacts)
    row["gate_pass"] = row["success_rate"] >= .9 and row["fall_rate"] <= .05
    return row


def main():
    cfg, acfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1024
    cfg.episode_length_s = 65.0
    cfg.seed = acfg.seed = 20283021
    if args.device:
        cfg.sim.device = acfg.device = args.device
    specs = specs_for_mode()
    if args.debug:
        specs = [{**specs[0], "episodes": 1}]
    all_rows, all_traces, all_vectors = [], {}, {}
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
                                     clip_actions=acfg.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        actor = FrozenGaitActor(CHECKPOINT).to(device).eval()
        robot, sensor = env.scene["robot"], env.scene.sensors["contact_forces"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [robot.body_names.index(sensor.body_names[i]) for i in feet]
        joint_names = list(robot.joint_names)
        for chunk_index, batch in enumerate(chunks(specs)):
            count = sum(item["episodes"] for item in batch)
            wrapped.seed(20283021 + chunk_index)
            obs, _ = wrapped.reset()
            obs = obs["policy"].to(device)
            item_ids, episode_ids = [], []
            for index, item in enumerate(batch):
                item_ids.extend([index] * item["episodes"])
                episode_ids.extend(range(item["episodes"]))
            ids = torch.tensor(item_ids, device=device)
            active = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            active[:count] = True
            histories = [{key: [] for key in ("target", "actor", "actual", "vx", "vy", "vec", "direction", "contact")} for _ in range(count)]
            fall = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            dangerous = fall.clone(); impact = fall.clone(); saturation = fall.clone()
            slip_streak = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            sat_streak = slip_streak.clone()
            transition_obs = torch.zeros(count, obs.shape[-1], device=device)
            transition_action = torch.zeros(count, 37, device=device)
            transition_previous = torch.zeros_like(transition_action)
            captured = torch.zeros(count, dtype=torch.bool, device=device)
            previous_action = torch.zeros(env.num_envs, 37, device=device)
            max_steps = round(max(item["duration"] for item in batch) / env.step_dt)
            for step in range(max_steps):
                time_s = step * env.step_dt
                vx = torch.zeros(env.num_envs, device=device); vy = torch.zeros_like(vx)
                target = torch.zeros_like(vx); actor_yaw = torch.zeros_like(vx)
                valid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
                for env_id in range(count):
                    item = batch[int(ids[env_id])]
                    if time_s < item["duration"]:
                        vx[env_id], vy[env_id], target[env_id], actor_yaw[env_id], _ = command_at(
                            item, time_s, episode_ids[env_id]
                        )
                        valid[env_id] = True
                command.external_override[:, 0] = vx
                command.external_override[:, 1] = vy
                command.external_override[:, 2] = actor_yaw
                if step == 0:
                    command._update_command()
                    obs = wrapped.get_observations()["policy"].to(device)
                with torch.inference_mode():
                    action = actor(obs, torch.zeros(env.num_envs, device=device))
                capture_now = active[:count] & ~captured & (time_s >= 4.0)
                if capture_now.any():
                    transition_obs[capture_now] = obs[:count][capture_now]
                    transition_action[capture_now] = action[:count][capture_now]
                    transition_previous[capture_now] = previous_action[:count][capture_now]
                    captured[capture_now] = True
                previous_action.copy_(action)
                obs, _, done, extras = wrapped.step(action)
                obs = obs["policy"].to(device)
                measure = active & valid
                actual_v = robot.data.root_lin_vel_b[:, :2]
                actual_yaw = robot.data.root_ang_vel_b[:, 2]
                force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
                contact = force > 5
                foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
                sliding = ((foot_speed > .55) & contact).any(-1)
                slip_streak = torch.where(sliding, slip_streak + 1, torch.zeros_like(slip_streak))
                dangerous |= (slip_streak >= 5) & measure
                impact |= (force.amax(-1) > 3500) & measure
                limits = robot.data.joint_vel_limits
                limits = limits[..., 1].abs() if limits.ndim == 3 else limits
                ratio = robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(-1)
                sat_streak = torch.where(ratio > .95, sat_streak + 1, torch.zeros_like(sat_streak))
                saturation |= (sat_streak >= 5) & measure
                timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
                fall |= done.bool() & ~timeout & measure
                vector_error = torch.linalg.vector_norm(actual_v - torch.stack((vx, vy), -1), dim=-1)
                cmd_angle = torch.atan2(vy, vx); act_angle = torch.atan2(actual_v[:, 1], actual_v[:, 0])
                direction_error = torch.atan2(torch.sin(act_angle - cmd_angle), torch.cos(act_angle - cmd_angle)).abs() * 180 / math.pi
                for env_id in range(count):
                    if not valid[env_id]:
                        continue
                    c = contact[env_id]
                    contact_name = "double_support" if bool(c.all()) else (
                        "left_support" if bool(c[0]) else ("right_support" if len(c) > 1 and bool(c[1]) else "flight")
                    )
                    h = histories[env_id]
                    h["target"].append(float(target[env_id])); h["actor"].append(float(actor_yaw[env_id]))
                    h["actual"].append(float(actual_yaw[env_id]))
                    h["vx"].append(float(actual_v[env_id, 0])); h["vy"].append(float(actual_v[env_id, 1]))
                    h["vec"].append(float(vector_error[env_id])); h["direction"].append(float(direction_error[env_id]))
                    h["contact"].append(contact_name)
            dt = env.step_dt
            for env_id in range(count):
                item = batch[int(ids[env_id])]
                h = histories[env_id]
                n = len(h["target"]); transition_index = min(round(4 / dt), n - 1)
                final_start = min(round((4 + item["ramp"] + item["dwell"] + (item["ramp"] if item["transition"] == "dwell" else 0)) / dt), n - 1)
                final_window = range(max(final_start, n - round(4 / dt)), n)
                final_sign = 1 if item["final"] > 0 else (-1 if item["final"] < 0 else 0)
                sign_ok = lambda value: True if final_sign == 0 else value * final_sign > .05
                acquisition = next((i for i in range(transition_index, n) if sign_ok(h["actual"][i])), n - 1)
                delta = item["final"] - item["initial"]
                response_level = item["initial"] + .1 * delta
                tau_level = item["initial"] + .632 * delta
                crossed = lambda value, level: value >= level if delta >= 0 else value <= level
                response = next((i for i in range(transition_index, n) if crossed(h["actual"][i], response_level)), n - 1)
                tau = next((i for i in range(transition_index, n) if crossed(h["actual"][i], tau_level)), n - 1)
                threshold = .15 if item["angle"] is None else .20
                hold_steps = max(1, round(.5 / dt))
                settling = n - 1
                for i in range(final_start, max(final_start + 1, n - hold_steps)):
                    if max(abs(h["actual"][j] - item["final"]) for j in range(i, min(i + hold_steps, n))) <= threshold:
                        settling = i
                        break
                whole_sign = sum((a * t > 0) or abs(t) < 1e-6 for a, t in zip(h["actual"], h["target"])) / n
                final_sign_fraction = sum(sign_ok(h["actual"][i]) for i in final_window) / len(final_window)
                actor_d = [(h["actor"][i] - h["actor"][i - 1]) / dt for i in range(1, n)]
                actor_dd = [(actor_d[i] - actor_d[i - 1]) / dt for i in range(1, len(actor_d))]
                safe = not bool(fall[env_id] or dangerous[env_id] or impact[env_id] or saturation[env_id])
                final_vec = sum(h["vec"][i] for i in final_window) / len(final_window)
                final_dir = sum(h["direction"][i] for i in final_window) / len(final_window)
                final_success = safe and final_sign_fraction >= .95
                if args.mode == "variance":
                    final_success = safe and final_vec <= .20 and final_dir <= 20
                transition_contact = h["contact"][transition_index]
                first_switch = next((h["contact"][i] for i in range(transition_index + 1, n)
                                     if h["contact"][i] != transition_contact), "none")
                row = {
                    "condition": item["name"], "episode": episode_ids[env_id],
                    "direction_deg": item["angle"], "speed": item["speed"],
                    "initial_yaw": item["initial"], "final_yaw": item["final"],
                    "transition": item["transition"], "ramp_duration": item["ramp"],
                    "zero_dwell": item["dwell"], "profile": item["profile"], "history": item["history"],
                    "whole_sign_fraction": whole_sign,
                    "final_hold_sign_fraction": final_sign_fraction,
                    "sign_acquisition_delay": max(0, (acquisition - transition_index) * dt),
                    "response_delay": max(0, (response - transition_index) * dt),
                    "time_constant": max(0, (tau - transition_index) * dt),
                    "settling_time": max(0, (settling - final_start) * dt),
                    "overshoot": max(max((a - item["final"]) * final_sign for a in h["actual"][final_start:]), 0) if final_sign else max(abs(a) for a in h["actual"][final_start:]),
                    "steady_state_error": sum(abs(h["actual"][i] - item["final"]) for i in final_window) / len(final_window),
                    "integrated_absolute_yaw_error": sum(abs(a - t) for a, t in zip(h["actual"], h["target"])) * dt,
                    "vector_mae": sum(h["vec"]) / n,
                    "final_vector_mae": final_vec,
                    "direction_error": sum(h["direction"]) / n,
                    "translation_degradation": max(h["vec"]),
                    "actor_input_max_derivative": max(map(abs, actor_d), default=0),
                    "actor_input_max_acceleration": max(map(abs, actor_dd), default=0),
                    "transition_contact_state": transition_contact,
                    "first_support_switch": first_switch,
                    "fall": bool(fall[env_id]), "dangerous_slip": bool(dangerous[env_id]),
                    "impact": bool(impact[env_id]), "saturation": bool(saturation[env_id]),
                    "success": final_success,
                }
                all_rows.append(row)
            for index, item in enumerate(batch):
                members = [env_id for env_id in range(count) if int(ids[env_id]) == index]
                if not members:
                    continue
                max_n = max(len(histories[i]["target"]) for i in members)
                trace = []
                for step in range(max_n):
                    valid_members = [i for i in members if step < len(histories[i]["target"])]
                    trace.append({
                        "time": step * dt,
                        **{key: sum(histories[i][key][step] for i in valid_members) / len(valid_members)
                           for key in ("target", "actor", "actual", "vx", "vy", "vec")},
                    })
                all_traces[item["name"]] = trace
                all_vectors[item["name"]] = {
                    "observation": transition_obs[members].mean(0).cpu().tolist(),
                    "action": transition_action[members].mean(0).cpu().tolist(),
                    "previous_action": transition_previous[members].mean(0).cpu().tolist(),
                    "contact_histogram": dict(defaultdict(int)),
                }
        grouped = defaultdict(list)
        for row in all_rows:
            grouped[row["condition"]].append(row)
        summary = [summarize(group) for group in grouped.values()]
        names = {
            "step": "dynamic_yaw_step_response", "ramp": "yaw_ramp_duration_boundary",
            "dwell": "yaw_zero_dwell_boundary", "profile": "dynamic_command_profile_comparison",
            "history": "yaw_history_dependence", "backward": "backward_dynamic_yaw_boundary",
            "random": "random_command_dynamic_trace", "variance": "forward_1p2_variance_raw",
        }
        base = names[args.mode]
        write_csv(base + ".csv", summary)
        dump(base + ".json", {
            "mode": args.mode, "checkpoint_sha256": "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d",
            "calibration": "MonotonicPositiveYawCalibrationV1", "training_updates": 0,
            "rows": summary, "episode_rows": all_rows, "mean_traces": all_traces,
            "transition_vectors": all_vectors, "joint_names": joint_names,
            "episode_traces": (
                [{"episode": i, **histories[i]} for i in range(count)]
                if args.mode == "random" else []
            ),
        })
        print(json.dumps({"mode": args.mode, "conditions": len(summary), "episodes": len(all_rows)}, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
