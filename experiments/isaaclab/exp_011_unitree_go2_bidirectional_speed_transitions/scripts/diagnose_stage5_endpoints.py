"""Paired, deterministic Stage 5 endpoint telemetry with no optimizer updates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage5_endpoint_failure_diagnosis"
PARENT = (
    REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/"
    "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
)
SELECTED = (
    REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage4_resumed_optimizer_training/checkpoints/model_50.pt"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=OUT)
parser.add_argument("--num-envs", type=int, default=50)
parser.add_argument("--skip-transitions", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
from go2_bidirectional.contact_analysis import resolve_foot_mapping  # noqa: E402
from go2_bidirectional.evaluation import build_runner  # noqa: E402
from go2_bidirectional.gait_classifier import classify  # noqa: E402
from go2_bidirectional.metrics import mean, percentile, wrap_angle  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

SEED = 20263901
SPEEDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2, 2.0)
THRESHOLDS = (0.10, 0.20, 0.30, 0.50)
FEET = ("front-left", "front-right", "rear-left", "rear-right")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rpy(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def support_entropy(states: list[int]) -> float:
    counts = Counter(states)
    total = len(states)
    return -sum((count / total) * math.log2(count / total) for count in counts.values()) if total else 0.0


def max_contiguous(flags: list[bool], dt: float) -> float:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best * dt


def make_env():
    cfg, agent = resolve_task_config(
        "Isaac-Velocity-Flat-Unitree-Go2-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = args.num_envs
    cfg.seed = SEED
    cfg.episode_length_s = 12.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = args.device
    return gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg), agent, cfg


def interval_stats(contact: list[bool], positions: list[list[float]], dt: float) -> dict:
    intervals = []
    start = None
    for index, flag in enumerate(contact + [False]):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            end = index
            points = positions[start:end]
            steps = [
                math.dist(points[i - 1][:2], points[i][:2]) / dt
                for i in range(1, len(points))
            ]
            inner = steps[2:-2] if len(steps) > 4 else []
            displacement = math.dist(points[0][:2], points[-1][:2]) if len(points) > 1 else 0.0
            intervals.append({
                "start": start, "end": end, "duration_s": (end - start) * dt,
                "displacement_m": displacement, "speed_including_boundary": steps,
                "speed_excluding_two_boundary_steps": inner,
            })
            start = None
    return {
        "intervals": intervals,
        "displacements": [row["displacement_m"] for row in intervals],
        "durations": [row["duration_s"] for row in intervals],
        "speeds_including": [value for row in intervals for value in row["speed_including_boundary"]],
        "speeds_excluding": [
            value for row in intervals for value in row["speed_excluding_two_boundary_steps"]
        ],
    }


def diagnostic_reference_gait(contacts: list[list[bool]], speed: float, fall: bool) -> str:
    if fall:
        return "FALL"
    columns = list(map(list, zip(*contacts)))
    duty = [mean(column) for column in columns]
    diag_opposition = mean(
        (a == d and b == c and a != b) for a, b, c, d in contacts
    )
    single = mean(sum(row) in (1, 3) for row in contacts)
    if speed <= 0.08 and min(duty) > 0.90:
        return "STAND_LIKE"
    if min(duty) > 0.85 and speed <= 0.5:
        return "STAND_LIKE_STEPPING"
    if single > 0.25:
        return "CRAWL_WALK_LIKE"
    if diag_opposition > 0.35:
        return "TROT_LIKE"
    return "IRREGULAR"


class DiagnosticRunner:
    def __init__(self, raw, agent):
        self.wrapped, self.runner, self.policy = build_runner(raw, agent, PARENT)
        self.env = self.wrapped.unwrapped
        self.robot = self.env.scene["robot"]
        self.sensor = self.env.scene.sensors["contact_forces"]
        self.command = self.env.command_manager.get_term("base_velocity")
        self.mapping = resolve_foot_mapping(self.robot, self.sensor)
        self.body_ids = [row["robot_body_index"] for row in self.mapping]
        self.sensor_ids = [row["contact_sensor_index"] for row in self.mapping]
        self.dt = float(self.env.step_dt)
        self.frame_rows = []
        self.event_rows = []
        self.failure_timelines = []

    def load(self, path: Path):
        self.runner.load(str(path), strict=True, map_location=self.env.device)
        self.policy = self.runner.get_inference_policy(device=self.env.device)

    def run_speed(self, checkpoint_name: str, speed: float) -> list[dict]:
        self.env.seed(SEED)
        self.wrapped.reset()
        n = self.env.num_envs
        traces = [{
            "vx": [], "vy": [], "yaw_rate": [], "yaw": [], "roll": [], "pitch": [], "tilt": [],
            "height": [], "contacts5": [], "contacts1": [], "force": [], "foot_pos": [],
            "foot_vel": [], "root_relative_vel": [], "official_raw": [], "action_norm": [],
            "action_rate": [], "support": [], "fall": False, "fall_step": None,
            "initial_joint": None, "initial_quat": None,
        } for _ in range(n)]
        previous_action = torch.zeros((n, 12), device=self.env.device)
        alive = torch.ones(n, dtype=torch.bool, device=self.env.device)
        initial_yaw = None
        previous_contact = torch.zeros((n, 4), dtype=torch.bool, device=self.env.device)
        pending_events = []
        boundary_history = []
        for step in range(round(8.0 / self.dt)):
            self.command.vel_command_b[:, 0] = speed
            self.command.vel_command_b[:, 1:] = 0.0
            with torch.inference_mode():
                action = self.policy(self.wrapped.get_observations())
                _, _, dones, _ = self.wrapped.step(action)
            quat_xyzw = self.robot.data.root_quat_w.torch
            quat = torch.roll(quat_xyzw, shifts=1, dims=-1)
            roll, pitch, yaw = rpy(quat)
            if initial_yaw is None:
                initial_yaw = yaw.clone()
            gravity = quat_apply_inverse(
                quat_xyzw, torch.tensor((0.0, 0.0, -1.0), device=quat.device).repeat(n, 1)
            )
            tilt = torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0))
            forces = self.sensor.data.net_forces_w_history.torch[:, :, self.sensor_ids, :]
            force_norm = forces.norm(dim=-1).amax(dim=1)
            contact5, contact1 = force_norm > 5.0, force_norm > 1.0
            foot_pos = self.robot.data.body_pos_w.torch[:, self.body_ids, :]
            foot_vel = self.robot.data.body_lin_vel_w.torch[:, self.body_ids, :]
            root_pos = self.robot.data.root_pos_w.torch
            root_lin = self.robot.data.root_lin_vel_w.torch
            root_ang = self.robot.data.root_ang_vel_w.torch
            point_velocity = root_lin[:, None, :] + torch.cross(
                root_ang[:, None, :].expand_as(foot_pos), foot_pos - root_pos[:, None, :], dim=-1
            )
            relative_world = foot_vel - point_velocity
            relative_root = quat_apply_inverse(
                quat_xyzw[:, None, :].expand(-1, 4, -1).reshape(-1, 4),
                relative_world.reshape(-1, 3),
            ).reshape(n, 4, 3)
            planar = foot_vel[:, :, :2].norm(dim=-1)
            official_raw = (planar * contact1).sum(dim=1)
            action_rate = torch.linalg.vector_norm(action - previous_action, dim=1) / self.dt
            action_norm = torch.linalg.vector_norm(action, dim=1)
            previous_action = action.clone()
            if step < 20 and speed in (0.0, 0.4, 1.2, 2.0):
                for foot in range(4):
                    self.frame_rows.append({
                        "checkpoint": checkpoint_name, "speed_mps": speed, "episode": 0,
                        "step": step, "foot": FEET[foot],
                        "world_pos": foot_pos[0, foot].cpu().tolist(),
                        "world_velocity": foot_vel[0, foot].cpu().tolist(),
                        "root_relative_velocity_root_frame": relative_root[0, foot].cpu().tolist(),
                        "root_world_linear_velocity": root_lin[0].cpu().tolist(),
                        "root_world_angular_velocity": root_ang[0].cpu().tolist(),
                        "contact_force_n": float(force_norm[0, foot]),
                        "contact": bool(contact5[0, foot]),
                        "air_time_s": float(self.sensor.data.current_air_time.torch[0, self.sensor_ids[foot]]),
                        "last_contact_time_s": float(self.sensor.data.last_contact_time.torch[0, self.sensor_ids[foot]]),
                    })
            boundary_snapshot = []
            for env_index in range(min(2, n)):
                for foot_index in range(4):
                    boundary_snapshot.append({
                        "checkpoint": checkpoint_name, "speed_mps": speed,
                        "episode": env_index, "foot": FEET[foot_index], "step": step,
                        "contact_force_n": float(force_norm[env_index, foot_index]),
                        "foot_height_m": float(foot_pos[env_index, foot_index, 2]),
                        "world_foot_speed_mps": float(planar[env_index, foot_index]),
                        "contact": bool(contact5[env_index, foot_index]),
                        "existing_slip_flag": bool(
                            contact5[env_index, foot_index] and planar[env_index, foot_index] > 0.55
                        ),
                    })
            for pending in pending_events:
                offset = step - pending["event_step"]
                if 1 <= offset <= 5:
                    for snapshot in boundary_snapshot:
                        if (
                            snapshot["episode"] == pending["episode"]
                            and snapshot["foot"] == pending["foot"]
                        ):
                            pending["rows"].append({
                                **snapshot, "event": pending["event"], "step_offset": offset,
                            })
            completed = [pending for pending in pending_events if step - pending["event_step"] >= 5]
            for pending in completed:
                self.event_rows.extend(pending["rows"])
                pending_events.remove(pending)
            changed = contact5 != previous_contact
            for env_index, foot_index in torch.nonzero(changed).cpu().tolist():
                if env_index >= 2 or len(self.event_rows) >= 2000:
                    continue
                kind = "onset" if bool(contact5[env_index, foot_index]) else "release"
                rows = []
                for old_snapshot in boundary_history[-5:]:
                    for snapshot in old_snapshot:
                        if snapshot["episode"] == env_index and snapshot["foot"] == FEET[foot_index]:
                            rows.append({
                                **snapshot, "event": kind, "step_offset": snapshot["step"] - step,
                            })
                current = next(
                    snapshot for snapshot in boundary_snapshot
                    if snapshot["episode"] == env_index and snapshot["foot"] == FEET[foot_index]
                )
                rows.append({**current, "event": kind, "step_offset": 0})
                pending_events.append({
                    "event_step": step, "event": kind, "episode": env_index,
                    "foot": FEET[foot_index], "rows": rows,
                })
            previous_contact = contact5.clone()
            boundary_history.append(boundary_snapshot)
            boundary_history = boundary_history[-5:]
            for index, trace in enumerate(traces):
                if not bool(alive[index]):
                    continue
                if trace["initial_joint"] is None:
                    trace["initial_joint"] = self.robot.data.joint_pos.torch[index].cpu().tolist()
                    trace["initial_quat"] = quat_xyzw[index].cpu().tolist()
                trace["vx"].append(float(self.robot.data.root_lin_vel_b.torch[index, 0]))
                trace["vy"].append(float(self.robot.data.root_lin_vel_b.torch[index, 1]))
                trace["yaw_rate"].append(float(self.robot.data.root_ang_vel_b.torch[index, 2]))
                trace["yaw"].append(abs(wrap_angle(float(yaw[index] - initial_yaw[index]))))
                trace["roll"].append(float(roll[index])); trace["pitch"].append(float(pitch[index]))
                trace["tilt"].append(float(tilt[index])); trace["height"].append(float(root_pos[index, 2]))
                c5 = [bool(v) for v in contact5[index].cpu()]
                trace["contacts5"].append(c5)
                trace["contacts1"].append([bool(v) for v in contact1[index].cpu()])
                trace["force"].append(force_norm[index].cpu().tolist())
                trace["foot_pos"].append(foot_pos[index].cpu().tolist())
                trace["foot_vel"].append(foot_vel[index].cpu().tolist())
                trace["root_relative_vel"].append(relative_root[index].cpu().tolist())
                trace["official_raw"].append(float(official_raw[index]))
                trace["action_norm"].append(float(action_norm[index]))
                trace["action_rate"].append(float(action_rate[index]))
                trace["support"].append(sum((1 << foot) for foot, value in enumerate(c5) if value))
            newly_done = dones.bool() & alive
            for index in torch.nonzero(newly_done).flatten().tolist():
                traces[index]["fall"] = True
                traces[index]["fall_step"] = step
            alive &= ~dones.bool()
        for pending in pending_events:
            self.event_rows.extend(pending["rows"])
        return [self.summarize(checkpoint_name, speed, index, trace) for index, trace in enumerate(traces)]

    def summarize(self, checkpoint: str, speed: float, episode: int, trace: dict) -> dict:
        foot_intervals = [
            interval_stats(
                [row[foot] for row in trace["contacts5"]],
                [row[foot] for row in trace["foot_pos"]],
                self.dt,
            )
            for foot in range(4)
        ]
        world_speeds = [
            math.hypot(row[foot][0], row[foot][1])
            for row, contact in zip(trace["foot_vel"], trace["contacts5"])
            for foot in range(4) if contact[foot]
        ]
        root_relative = [
            math.hypot(row[foot][0], row[foot][1])
            for row, contact in zip(trace["root_relative_vel"], trace["contacts5"])
            for foot in range(4) if contact[foot]
        ]
        threshold_stats = {}
        total_steps = len(trace["contacts5"])
        contact_steps = sum(sum(row) for row in trace["contacts5"])
        for threshold in THRESHOLDS:
            flags = [
                any(
                    contact[foot] and math.hypot(velocity[foot][0], velocity[foot][1]) > threshold
                    for foot in range(4)
                )
                for velocity, contact in zip(trace["foot_vel"], trace["contacts5"])
            ]
            per_foot = [
                mean(
                    contact[foot] and math.hypot(velocity[foot][0], velocity[foot][1]) > threshold
                    for velocity, contact in zip(trace["foot_vel"], trace["contacts5"])
                )
                for foot in range(4)
            ]
            flagged_contact_steps = sum(
                contact[foot] and math.hypot(velocity[foot][0], velocity[foot][1]) > threshold
                for velocity, contact in zip(trace["foot_vel"], trace["contacts5"])
                for foot in range(4)
            )
            threshold_stats[str(threshold)] = {
                "occurrence": any(flags), "time_fraction": mean(flags),
                "contact_time_fraction": flagged_contact_steps / max(contact_steps, 1),
                "max_contiguous_duration_s": max_contiguous(flags, self.dt),
                "per_foot_time_fraction": per_foot,
            }
        settle_start = min(round(2.0 / self.dt), max(len(trace["height"]) - 1, 0))
        settle = slice(settle_start, None)
        settle_roll = trace["roll"][settle]; settle_pitch = trace["pitch"][settle]
        median_roll = percentile(settle_roll, 50); median_pitch = percentile(settle_pitch, 50)
        nominal_deviation = [
            math.hypot(r - median_roll, p - median_pitch)
            for r, p in zip(settle_roll, settle_pitch)
        ]
        gait, evidence = classify(trace["contacts5"], abs(mean(trace["vx"])), trace["fall"])
        reference_gait = diagnostic_reference_gait(
            trace["contacts5"], abs(mean(trace["vx"])), trace["fall"]
        )
        if speed == 0.4 and trace["fall"]:
            fall_step = trace["fall_step"]
            self.failure_timelines.append({
                "checkpoint": checkpoint, "episode": episode, "seed": SEED + episode,
                "initial_joint_state": trace["initial_joint"], "initial_root_quaternion": trace["initial_quat"],
                "fall_step": fall_step, "fall_time_s": fall_step * self.dt,
                "yaw_drift_onset_step": next((i for i, value in enumerate(trace["yaw"]) if value > 0.12), None),
                "lateral_drift_onset_step": next((i for i, value in enumerate(trace["vy"]) if abs(value) > 0.2), None),
                "slip_onset_step": next((
                    i for i, value in enumerate(trace["foot_vel"])
                    if any(
                        trace["contacts5"][i][foot] and math.hypot(value[foot][0], value[foot][1]) > 0.55
                        for foot in range(4)
                    )
                ), None),
                "contact_sequence_to_failure": trace["support"][:fall_step + 1],
                "yaw_to_failure": trace["yaw"][:fall_step + 1],
                "lateral_velocity_to_failure": trace["vy"][:fall_step + 1],
                "action_norm_to_failure": trace["action_norm"][:fall_step + 1],
            })
        return {
            "checkpoint": checkpoint, "speed_mps": speed, "episode": episode,
            "seed": SEED + episode, "fall": trace["fall"], "fall_step": trace["fall_step"],
            "actual_speed_mean_mps": mean(trace["vx"]),
            "lateral_speed_abs_mean_mps": mean(abs(value) for value in trace["vy"]),
            "yaw_drift_p95_rad": percentile(trace["yaw"], 95),
            "yaw_drift_max_rad": max(trace["yaw"], default=0.0),
            "roll_abs_p95_rad": percentile((abs(v) for v in trace["roll"]), 95),
            "pitch_abs_p95_rad": percentile((abs(v) for v in trace["pitch"]), 95),
            "gravity_tilt_p95_rad": percentile(trace["tilt"], 95),
            "base_height_range_m": max(trace["height"]) - min(trace["height"]),
            "settle_height_range_m": max(trace["height"][settle]) - min(trace["height"][settle]),
            "settle_median_roll_rad": median_roll, "settle_median_pitch_rad": median_pitch,
            "settle_nominal_tilt_deviation_p95_rad": percentile(nominal_deviation, 95),
            "root_speed_mean_mps": mean(math.hypot(vx, vy) for vx, vy in zip(trace["vx"], trace["vy"])),
            "yaw_rate_abs_p95_radps": percentile((abs(v) for v in trace["yaw_rate"]), 95),
            "contact_occupancy": [mean(row[foot] for row in trace["contacts5"]) for foot in range(4)],
            "contact_loss_fraction": mean(not any(row) for row in trace["contacts5"]),
            "support_state_entropy": support_entropy(trace["support"]),
            "gait": gait, "reference_gait": reference_gait, "gait_evidence": evidence,
            "action_norm_mean": mean(trace["action_norm"]),
            "action_rate_p95": percentile(trace["action_rate"], 95),
            "existing_slip_mean_mps": mean(
                max(
                    [math.hypot(velocity[foot][0], velocity[foot][1]) for foot in range(4) if contact[foot]]
                    or [0.0]
                )
                for velocity, contact in zip(trace["foot_vel"], trace["contacts5"])
            ),
            "existing_dangerous_slip": mean(
                max(
                    [math.hypot(velocity[foot][0], velocity[foot][1]) for foot in range(4) if contact[foot]]
                    or [0.0]
                )
                for velocity, contact in zip(trace["foot_vel"], trace["contacts5"])
            ) > 0.55,
            "official_feet_slide_raw_mean": mean(trace["official_raw"]),
            "official_feet_slide_per_foot_mean": [
                mean(
                    math.hypot(velocity[foot][0], velocity[foot][1]) * contact[foot]
                    for velocity, contact in zip(trace["foot_vel"], trace["contacts1"])
                )
                for foot in range(4)
            ],
            "official_feet_slide_weighted_mean": 0.0,
            "world_stance_speed_mean_mps": mean(world_speeds),
            "world_stance_speed_p95_mps": percentile(world_speeds, 95),
            "root_relative_foot_speed_mean_mps": mean(root_relative),
            "physical_intervals": foot_intervals,
            "threshold_stats": threshold_stats,
        }

    def run_transition(self, checkpoint_name: str, source: float, target: float) -> dict:
        self.env.seed(SEED); self.wrapped.reset()
        alive = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        acquired = torch.zeros_like(alive); held = torch.zeros(self.env.num_envs, device=self.env.device)
        falls = torch.zeros_like(alive)
        for step in range(round(9.5 / self.dt)):
            t = step * self.dt
            if t < 3.0:
                command = source
            elif t < 4.5:
                tau = (t - 3.0) / 1.5
                p = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
                command = source + (target - source) * p
            else:
                command = target
            self.command.vel_command_b[:, 0] = command; self.command.vel_command_b[:, 1:] = 0.0
            with torch.inference_mode():
                action = self.policy(self.wrapped.get_observations())
                _, _, dones, _ = self.wrapped.step(action)
            if t >= 4.5:
                actual = self.robot.data.root_lin_vel_b.torch[:, 0]
                in_band = actual.abs() <= 0.08 if target == 0 else (actual - target).abs() <= (0.2 if target < 2 else 0.25)
                held = torch.where(in_band & alive, held + self.dt, torch.zeros_like(held))
                acquired |= held >= 1.0
            falls |= dones.bool() & alive; alive &= ~dones.bool()
        return {
            "checkpoint": checkpoint_name, "transition": f"{source}->{target}",
            "episodes": self.env.num_envs, "fall_rate": float(falls.float().mean()),
            "target_acquisition_rate": float(acquired.float().mean()),
            "completion_rate": float((~falls & acquired).float().mean()),
        }


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    raw, agent, cfg = make_env()
    diagnostic = DiagnosticRunner(raw, agent)
    checkpoints = {"official_parent": PARENT, "stage4_selected": SELECTED}
    all_rows = []
    transition_rows = []
    for name, checkpoint in checkpoints.items():
        diagnostic.load(checkpoint)
        for speed in SPEEDS:
            all_rows.extend(diagnostic.run_speed(name, speed))
            print(f"stage5_progress checkpoint={name} speed={speed}", flush=True)
        if not args.skip_transitions:
            for source, target in ((0, 0.4), (0.4, 0), (0, 1.2), (1.2, 0), (1.2, 2), (2, 1.2)):
                transition_rows.append(diagnostic.run_transition(name, source, target))
                print(f"stage5_progress checkpoint={name} transition={source}->{target}", flush=True)
    dump(args.output / "raw_episode_summaries.json", all_rows)
    write_csv(args.output / "per_foot_frame_comparison.csv", diagnostic.frame_rows)
    write_csv(args.output / "contact_event_examples.csv", diagnostic.event_rows)
    dump(args.output / "low_speed_failure_timeline.json", {
        "failures": diagnostic.failure_timelines,
        "selection": "all failed 0.4 m/s episodes in ascending fixed seed order",
    })
    write_csv(args.output / "low_speed_failure_examples.csv", [
        {key: value for key, value in row.items() if not isinstance(value, list)}
        for row in diagnostic.failure_timelines
    ])
    if not args.skip_transitions:
        dump(args.output / "transition_diagnostic.json", transition_rows)
    dump(args.output / "runtime_contract.json", {
        "environment_id": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "num_envs": args.num_envs, "seed_root": SEED, "dt": diagnostic.dt,
        "physics_dt": float(cfg.sim.dt), "decimation": int(cfg.decimation),
        "mapping": diagnostic.mapping, "parent_sha256": sha256(PARENT),
        "selected_sha256": sha256(SELECTED), "optimizer_updates": 0,
    })
    diagnostic.wrapped.close()


try:
    main()
except Exception:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "runtime_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    raise
finally:
    simulation_app.close()
