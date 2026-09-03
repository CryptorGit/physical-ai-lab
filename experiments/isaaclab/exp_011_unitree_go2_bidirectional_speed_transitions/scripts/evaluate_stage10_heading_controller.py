"""Paired Stage 10 command-controller evaluation with a frozen Stage 7 policy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage10_phase_gated_fixed_heading"
CHECKPOINT = (
    REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage7_low_speed_gait_stabilization/checkpoints/model_50.pt"
)
CONTROLLERS = ("C0", "C1", "C2")
STEADY = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2, 2.0)
LOW = ((0.0, 0.2), (0.0, 0.4), (0.0, 0.6), (0.6, 0.4), (0.6, 0.2), (0.6, 0.0))
ANCHOR = ((0.0, 1.2), (1.2, 2.0), (2.0, 1.2), (1.2, 0.0))
SEQUENCE = (0.0, 0.6, 1.2, 2.0, 1.2, 0.6, 0.0)
CHUNKS = (
    *(f"steady_{index}" for index in range(5)),
    *(f"low_{index}" for index in range(3)),
    "anchor_0", "anchor_1", "sequence",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--controller", choices=CONTROLLERS, required=True)
parser.add_argument("--chunk", choices=CHUNKS, required=True)
parser.add_argument("--output", type=Path, default=OUT)
parser.add_argument("--num-envs", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_physx.sensors import ContactSensorCfg  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
from go2_bidirectional.command_profiles import sequence_command, transition_command  # noqa: E402
from go2_bidirectional.contact_kinematics import stable_contact_mask  # noqa: E402
from go2_bidirectional.evaluation import build_runner  # noqa: E402
from go2_bidirectional.phase_gated_heading import (  # noqa: E402
    PhaseGatedFixedHeadingController,
    target_tolerance,
)
from go2_bidirectional.stage6_endpoint_protocol import (  # noqa: E402
    circular_median,
    quat_xyzw_to_gravity_tilt_torch,
    quat_xyzw_to_roll_pitch_yaw_torch,
)
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

SEED = 20269901
DT = 0.02
FEET = ("FL", "FR", "RL", "RR")
ASSET_FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GROUND = "/World/ground/terrain/GroundPlane/CollisionPlane"
MODE = {
    "C0": "OPEN_LOOP",
    "C1": "ALWAYS_ON_FIXED_HEADING",
    "C2": "PHASE_GATED_FIXED_HEADING",
}


def dump(path: Path, value) -> None:
    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=encode) + "\n", encoding="utf-8")


def mean(values) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def percentile(values, q) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else 0.0


def wrap(values):
    return np.arctan2(np.sin(values), np.cos(values))


def max_run(flags) -> int:
    best = current = 0
    for value in flags:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


class Runtime:
    def __init__(self):
        if args.num_envs != 50:
            raise SystemExit("Stage 10 requires exactly 50 environments")
        cfg, agent = resolve_task_config("Isaac-Velocity-Flat-Unitree-Go2-v0", "rsl_rl_cfg_entry_point")
        cfg.scene.num_envs = args.num_envs
        cfg.seed = SEED
        cfg.episode_length_s = 60.0
        cfg.observations.policy.enable_corruption = False
        cfg.events.base_external_force_torque = None
        cfg.events.push_robot = None
        for label, foot in zip(("fl", "fr", "rl", "rr"), ASSET_FEET):
            setattr(cfg.scene, f"stage10_{label}", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot}",
                update_period=0.0,
                track_pose=True,
                track_contact_points=True,
                track_friction_forces=True,
                max_contact_data_count_per_prim=16,
                filter_prim_paths_expr=[GROUND],
            ))
        if args.device:
            cfg.sim.device = args.device
            agent.device = args.device
        self.raw = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
        self.wrapped, self.runner, self.policy = build_runner(self.raw, agent, CHECKPOINT)
        self.env = self.wrapped.unwrapped
        self.robot = self.env.scene["robot"]
        self.command = self.env.command_manager.get_term("base_velocity")
        self.body_ids = [int(self.robot.find_bodies(name)[0][0]) for name in ASSET_FEET]
        self.sensors = [self.env.scene.sensors[f"stage10_{name}"] for name in ("fl", "fr", "rl", "rr")]

    def contact_telemetry(self):
        tangents, forces, moments = [], [], []
        root = self.robot.data.root_com_pos_w.torch
        for sensor, body_id in zip(self.sensors, self.body_ids):
            fn_raw, point_raw, normal_raw, _, count_raw, start_raw = sensor.contact_view.get_contact_data(
                dt=sensor._sim_physics_dt
            )
            friction_raw, friction_point_raw, friction_count_raw, friction_start_raw = (
                sensor.contact_view.get_friction_data(dt=sensor._sim_physics_dt)
            )
            fn_raw = wp.to_torch(fn_raw).reshape(-1)
            point_raw = wp.to_torch(point_raw).reshape(-1, 3)
            normal_raw = wp.to_torch(normal_raw).reshape(-1, 3)
            counts = wp.to_torch(count_raw).reshape(args.num_envs, -1)[:, 0].long()
            starts = wp.to_torch(start_raw).reshape(args.num_envs, -1)[:, 0].long()
            offsets = torch.arange(16, device=point_raw.device)
            indices = (starts[:, None] + offsets).clamp(0, max(0, len(point_raw) - 1))
            mask = offsets[None] < counts[:, None]
            points = point_raw[indices]
            normals = normal_raw[indices]
            unit = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            fn = fn_raw[indices].abs() * mask
            radius = points - self.robot.data.body_pos_w.torch[:, body_id, None, :]
            angular = self.robot.data.body_ang_vel_w.torch[:, body_id, None, :].expand_as(radius)
            surface = self.robot.data.body_lin_vel_w.torch[:, body_id, None, :] + torch.linalg.cross(
                angular, radius, dim=-1
            )
            tangent = surface - (surface * unit).sum(-1, keepdim=True) * unit
            total = fn.sum(1)
            tangent_mean = (tangent.norm(dim=-1) * fn).sum(1) / total.clamp_min(1e-12)
            tangent_mean[total <= 5.0] = 0.0
            normal_force = fn[..., None] * unit
            moment = torch.linalg.cross(points - root[:, None, :], normal_force, dim=-1)[..., 2].sum(1)

            friction = wp.to_torch(friction_raw).reshape(-1, 3)
            friction_point = wp.to_torch(friction_point_raw).reshape(-1, 3)
            fcounts = wp.to_torch(friction_count_raw).reshape(args.num_envs, -1)[:, 0].long()
            fstarts = wp.to_torch(friction_start_raw).reshape(args.num_envs, -1)[:, 0].long()
            findices = (fstarts[:, None] + offsets).clamp(0, max(0, len(friction) - 1))
            fmask = offsets[None] < fcounts[:, None]
            fforce = friction[findices] * fmask[..., None]
            fpoint = friction_point[findices]
            moment += torch.linalg.cross(fpoint - root[:, None, :], fforce, dim=-1)[..., 2].sum(1)
            tangents.append(tangent_mean)
            forces.append(total)
            moments.append(moment)
        return torch.stack(tangents, 1), torch.stack(forces, 1), torch.stack(moments, 1)

    def collect(self, family: str, source: float, target: float) -> list[dict]:
        self.env.seed(SEED)
        self.wrapped.reset()
        duration = 8.0 if family == "steady" else 9.5
        steps = round(duration / DT)
        controllers = [
            PhaseGatedFixedHeadingController(MODE[args.controller], family, target, DT)
            for _ in range(args.num_envs)
        ]
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=self.env.device)
        falls = torch.zeros_like(alive)
        fall_step = torch.full((args.num_envs,), -1, dtype=torch.long, device=self.env.device)
        acquisition_count = np.zeros(args.num_envs, dtype=int)
        acquisition_step = np.full(args.num_envs, -1, dtype=int)
        trace = defaultdict(list)

        for step in range(steps):
            t = step * DT
            if family == "steady":
                speed, schedule_phase = target, "steady"
            else:
                speed, raw_phase = transition_command(t, source, target, 1.5)
                schedule_phase = {"source_hold": "source", "ramp": "ramp", "target_hold": "target"}[raw_phase]
            _, _, yaw = quat_xyzw_to_roll_pitch_yaw_torch(self.robot.data.root_quat_w.torch)
            actual = self.robot.data.root_lin_vel_b.torch[:, 0]
            outputs = [
                controller.update(t, float(yaw[index]), float(actual[index]), schedule_phase)
                for index, controller in enumerate(controllers)
            ]
            yaw_command = torch.tensor(
                [output.command for output in outputs], device=self.env.device, dtype=torch.float32
            )
            self.command.vel_command_b[:, 0] = speed
            self.command.vel_command_b[:, 1] = 0.0
            self.command.vel_command_b[:, 2] = yaw_command
            obs = self.wrapped.get_observations()
            with torch.inference_mode():
                action = self.policy(obs)
                _, _, dones, _ = self.wrapped.step(action)
            roll, pitch, yaw_after = quat_xyzw_to_roll_pitch_yaw_torch(self.robot.data.root_quat_w.torch)
            tilt = quat_xyzw_to_gravity_tilt_torch(self.robot.data.root_quat_w.torch)
            tangent, force, yaw_moment = self.contact_telemetry()
            velocity_ratio = (
                self.robot.data.joint_vel.torch.abs()
                / self.robot.data.joint_vel_limits.torch.abs().clamp_min(1e-6)
            ).amax(1)
            torque_ratio = (
                self.robot.data.applied_torque.torch.abs()
                / self.robot.data.joint_effort_limits.torch.abs().clamp_min(1e-6)
            ).amax(1)
            if family == "transition" and t >= 4.5:
                now = self.robot.data.root_lin_vel_b.torch[:, 0].detach().cpu().numpy()
                within = np.abs(now) <= 0.08 if target == 0 else np.abs(now - target) <= target_tolerance(target)
                acquisition_count = np.where(within, acquisition_count + 1, 0)
                newly_acquired = (acquisition_step < 0) & (acquisition_count >= round(0.5 / DT))
                acquisition_step[newly_acquired] = step
            for key, tensor in (
                ("vx", self.robot.data.root_lin_vel_b.torch[:, 0]),
                ("yaw", yaw_after), ("yaw_rate", self.robot.data.root_ang_vel_w.torch[:, 2]),
                ("tilt", tilt), ("yaw_command", yaw_command),
                ("tangent", tangent), ("contact_force", force), ("yaw_moment", yaw_moment),
                ("saturation", (velocity_ratio >= 0.95) | (torque_ratio >= 0.95)),
                ("alive", alive),
            ):
                trace[key].append(tensor.detach().cpu())
            trace["gate"].append(np.asarray([output.gate for output in outputs], dtype=float))
            trace["raw"].append(np.asarray([output.raw_command for output in outputs], dtype=float))
            trace["error"].append(np.asarray([output.error for output in outputs], dtype=float))
            trace["phase"].append([output.phase.value for output in outputs])
            trace["reference"].append(np.asarray([
                output.reference if output.reference is not None else np.nan for output in outputs
            ], dtype=float))
            newly = dones.bool() & alive
            fall_step[newly] = step
            falls |= newly
            alive &= ~dones.bool()

        arrays = {
            key: torch.stack(values).numpy()
            for key, values in trace.items()
            if key not in {"gate", "raw", "error", "phase", "reference"}
        }
        for key in ("gate", "raw", "error", "reference"):
            arrays[key] = np.stack(trace[key])
        rows = []
        for episode in range(args.num_envs):
            end = int(fall_step[episode]) if fall_step[episode] >= 0 else steps
            end = max(2, end)
            yaw_values = arrays["yaw"][:end, episode]
            if args.controller == "C1":
                reference = float(yaw_values[0])
            elif family == "steady":
                reference = circular_median(yaw_values[25:min(51, end)].tolist())
            else:
                reference = circular_median(yaw_values[125:min(151, end)].tolist())
            error = wrap(yaw_values - reference)
            quality_start = min(round((1.5 if family == "steady" else 4.5) / DT), end - 1)
            quality = np.arange(quality_start, end)
            target_indices = np.arange(min(round(4.5 / DT), end - 1), end)
            final_indices = np.arange(max(0, end - round(1.0 / DT)), end)
            hold_within = (
                np.abs(arrays["vx"][final_indices, episode]) <= 0.08
                if target == 0
                else np.abs(arrays["vx"][final_indices, episode] - target) <= target_tolerance(target)
            )
            stable_tangent = []
            stable_moment = []
            for foot in range(4):
                stable, _ = stable_contact_mask(
                    arrays["contact_force"][:end, episode, foot] > 5.0,
                    minimum_steps=3, boundary_steps=2,
                )
                stable_tangent.extend(arrays["tangent"][:end, episode, foot][stable].tolist())
                stable_moment.extend(arrays["yaw_moment"][:end, episode, foot][stable].tolist())
            gate_values = arrays["gate"][:end, episode]
            active_indices = np.flatnonzero(gate_values >= 1.0 - 1e-9)
            saturation_flags = arrays["saturation"][quality, episode].astype(bool)
            slope = float(np.polyfit(quality * DT, error[quality], 1)[0]) if len(quality) >= 2 else 0.0
            row = {
                "controller": args.controller,
                "family": family,
                "condition": f"{source:g}->{target:g}" if family == "transition" else f"{target:g}",
                "source_speed": source, "target_speed": target,
                "episode": episode, "episode_seed": SEED + episode,
                "fall": bool(falls[episode]),
                "speed_mae": mean(abs(arrays["vx"][quality, episode] - target)),
                "heading_p50": percentile(abs(error[quality]), 50),
                "heading_p90": percentile(abs(error[quality]), 90),
                "heading_p95": percentile(abs(error[quality]), 95),
                "heading_p99": percentile(abs(error[quality]), 99),
                "signed_heading_slope": slope,
                "yaw_rate_mean": mean(arrays["yaw_rate"][quality, episode]),
                "yaw_rate_p95": percentile(abs(arrays["yaw_rate"][quality, episode]), 95),
                "feedback_activation_time": float(active_indices[0] * DT) if len(active_indices) else None,
                "feedback_duty_fraction": mean(gate_values > 0),
                "yaw_command_p95": percentile(abs(arrays["yaw_command"][:end, episode]), 95),
                "yaw_command_max": float(np.max(abs(arrays["yaw_command"][:end, episode]))),
                "gravity_tilt_p95": percentile(arrays["tilt"][quality, episode], 95),
                "long_dwell_saturation": max_run(saturation_flags) >= round(0.20 / DT),
                "saturation_fraction": mean(saturation_flags),
                "tangential_speed_p95": percentile(stable_tangent, 95),
                "contact_yaw_moment_mean": mean(stable_moment),
                "contact_yaw_moment_p95": percentile(abs(np.asarray(stable_moment)), 95),
                "heading_reference": reference,
                "command_sign_changes": int(np.sum(np.diff(np.sign(arrays["yaw_command"][:end, episode])) != 0)),
                "feedback_started_before_target_acquisition": (
                    bool(len(active_indices) and acquisition_step[episode] >= 0 and active_indices[0] < acquisition_step[episode])
                    if family == "transition" else False
                ),
                "acquisition": bool(acquisition_step[episode] >= 0) if family == "transition" else True,
                "acquisition_time": (
                    float(acquisition_step[episode] * DT) if acquisition_step[episode] >= 0 else None
                ),
                "target_hold": bool(mean(hold_within) >= 0.90),
                "completion": (
                    bool(acquisition_step[episode] >= 0 and mean(hold_within) >= 0.90)
                    if family == "transition" else not bool(falls[episode])
                ),
                "timeout": bool(family == "transition" and acquisition_step[episode] < 0),
                "ramp_heading_change": (
                    float(wrap(error[min(end - 1, 224)] - error[min(end - 1, 150)]))
                    if family == "transition" else 0.0
                ),
                "target_hold_heading_change": (
                    float(wrap(error[-1] - error[min(end - 1, 225)]))
                    if family == "transition" else 0.0
                ),
                "state_entries": controllers[episode].entries,
            }
            phase_names = [trace["phase"][index][episode] for index in range(end)]
            row["state_trace"] = []
            for phase_name in dict.fromkeys(phase_names):
                ix = np.flatnonzero(np.asarray(phase_names) == phase_name)
                row["state_trace"].append({
                    "phase": phase_name,
                    "entry_time": float(ix[0] * DT),
                    "exit_time": float((ix[-1] + 1) * DT),
                    "heading_reference": reference,
                    "heading_error_mean": mean(error[ix]),
                    "heading_error_p95": percentile(abs(error[ix]), 95),
                    "gate_mean": mean(arrays["gate"][ix, episode]),
                    "gate_min": float(np.min(arrays["gate"][ix, episode])),
                    "gate_max": float(np.max(arrays["gate"][ix, episode])),
                    "raw_yaw_command_mean": mean(arrays["raw"][ix, episode]),
                    "final_yaw_command_mean": mean(arrays["yaw_command"][ix, episode]),
                })
            if family == "transition":
                activation_start = int(active_indices[0]) if len(active_indices) else end
                acquisition_end = (
                    int(acquisition_step[episode]) if acquisition_step[episode] >= 0 else end
                )
                phase_ranges = {
                    "source_hold": (0, min(round(3.0 / DT), end)),
                    "speed_ramp": (min(round(3.0 / DT), end), min(round(4.5 / DT), end)),
                    "target_acquisition": (min(round(4.5 / DT), end), min(acquisition_end, end)),
                    "feedback_activation": (
                        min(acquisition_end, end),
                        min(activation_start + round(0.5 / DT), end),
                    ),
                    "active_target_hold": (
                        min(activation_start + round(0.5 / DT), end),
                        end,
                    ),
                }
                row["phases"] = {}
                for phase_name, (phase_start, phase_end) in phase_ranges.items():
                    ix = np.arange(phase_start, phase_end)
                    if len(ix) == 0:
                        row["phases"][phase_name] = {"samples": 0}
                        continue
                    row["phases"][phase_name] = {
                        "samples": len(ix),
                        "heading_change": float(wrap(error[ix[-1]] - error[ix[0]])),
                        "yaw_rate_mean": mean(arrays["yaw_rate"][ix, episode]),
                        "yaw_rate_p95": percentile(abs(arrays["yaw_rate"][ix, episode]), 95),
                        "speed_error": mean(abs(arrays["vx"][ix, episode] - target)),
                        "fall": bool(falls[episode] and fall_step[episode] < phase_end),
                        "contact_yaw_moment": mean(arrays["yaw_moment"][ix, episode].sum(1)),
                        "tangential_relative_motion": percentile(
                            arrays["tangent"][ix, episode].reshape(-1), 95
                        ),
                    }
                if len(active_indices):
                    before = np.arange(max(0, active_indices[0] - 10), active_indices[0])
                    after = np.arange(active_indices[0], min(end, active_indices[0] + 10))
                    row["speed_change_after_feedback"] = mean(
                        arrays["vx"][after, episode]
                    ) - mean(arrays["vx"][before, episode])
                else:
                    row["speed_change_after_feedback"] = None
            rows.append(row)
        print(f"STAGE10 {args.controller} {family} {source:g}->{target:g} n=50", flush=True)
        return rows

    def collect_sequence(self) -> list[dict]:
        # Sequence uses the same controller law per command segment. A source reference is
        # frozen from the last 0.5 s before each ramp; controller state never touches policy state.
        self.env.seed(SEED)
        self.wrapped.reset()
        duration = 3.0 + (len(SEQUENCE) - 1) * 4.5
        steps = round(duration / DT)
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=self.env.device)
        falls = torch.zeros_like(alive)
        fall_step = torch.full((args.num_envs,), -1, dtype=torch.long, device=self.env.device)
        controllers = [
            PhaseGatedFixedHeadingController(MODE[args.controller], "steady", 0.0, DT)
            for _ in range(args.num_envs)
        ]
        current_segment = 0
        segment_local_start = 0.0
        yaw_history = [[] for _ in range(args.num_envs)]
        trace = defaultdict(list)
        segment_acquisition = np.zeros((args.num_envs, len(SEQUENCE) - 1), dtype=bool)
        segment_acquisition_count = np.zeros((args.num_envs, len(SEQUENCE) - 1), dtype=int)
        for step in range(steps):
            t = step * DT
            speed, segment, phase = sequence_command(t, SEQUENCE, 1.5)
            if segment != current_segment:
                current_segment = segment
                segment_local_start = t - 3.0
                controllers = [
                    PhaseGatedFixedHeadingController(
                        MODE[args.controller], "transition", SEQUENCE[segment], DT
                    ) for _ in range(args.num_envs)
                ]
                for index, controller in enumerate(controllers):
                    samples = yaw_history[index][-round(0.5 / DT):]
                    controller.reference_samples = list(samples)
            _, _, yaw = quat_xyzw_to_roll_pitch_yaw_torch(self.robot.data.root_quat_w.torch)
            actual = self.robot.data.root_lin_vel_b.torch[:, 0]
            for index in range(args.num_envs):
                yaw_history[index].append(float(yaw[index]))
            if segment == 0:
                local_time = t
                schedule_phase = "steady"
            else:
                local_time = t - segment_local_start
                schedule_phase = "ramp" if phase == "ramp" else "target"
            outputs = [
                controller.update(local_time, float(yaw[index]), float(actual[index]), schedule_phase)
                for index, controller in enumerate(controllers)
            ]
            if segment > 0 and phase == "hold":
                actual_np = actual.detach().cpu().numpy()
                target_now = SEQUENCE[segment]
                within = (
                    np.abs(actual_np) <= 0.08
                    if target_now == 0
                    else np.abs(actual_np - target_now) <= target_tolerance(target_now)
                )
                slot = segment - 1
                segment_acquisition_count[:, slot] = np.where(
                    within, segment_acquisition_count[:, slot] + 1, 0
                )
                segment_acquisition[:, slot] |= (
                    segment_acquisition_count[:, slot] >= round(0.5 / DT)
                )
            yaw_command = torch.tensor([output.command for output in outputs], device=self.env.device)
            self.command.vel_command_b[:, 0] = speed
            self.command.vel_command_b[:, 1] = 0.0
            self.command.vel_command_b[:, 2] = yaw_command
            obs = self.wrapped.get_observations()
            with torch.inference_mode():
                action = self.policy(obs)
                _, _, dones, _ = self.wrapped.step(action)
            _, _, yaw_after = quat_xyzw_to_roll_pitch_yaw_torch(self.robot.data.root_quat_w.torch)
            trace["vx"].append(self.robot.data.root_lin_vel_b.torch[:, 0].detach().cpu())
            trace["yaw"].append(yaw_after.detach().cpu())
            trace["yaw_command"].append(yaw_command.detach().cpu())
            trace["gate"].append(np.asarray([output.gate for output in outputs]))
            trace["segment"].append(segment)
            newly = dones.bool() & alive
            fall_step[newly] = step
            falls |= newly
            alive &= ~dones.bool()
        vx = torch.stack(trace["vx"]).numpy()
        yaw = torch.stack(trace["yaw"]).numpy()
        yaw_command = torch.stack(trace["yaw_command"]).numpy()
        gate = np.stack(trace["gate"])
        rows = []
        for episode in range(args.num_envs):
            end = int(fall_step[episode]) if fall_step[episode] >= 0 else steps
            end = max(2, end)
            reference = float(yaw[0, episode])
            error = wrap(yaw[:end, episode] - reference)
            segment_success = []
            for index in range(1, len(SEQUENCE)):
                hold_start = round((3.0 + (index - 1) * 4.5 + 1.5) / DT)
                hold_end = min(round((3.0 + index * 4.5) / DT), end)
                if hold_end <= hold_start:
                    segment_success.append(False)
                    continue
                target = SEQUENCE[index]
                within = (
                    abs(vx[hold_start:hold_end, episode]) <= 0.08
                    if target == 0
                    else abs(vx[hold_start:hold_end, episode] - target) <= target_tolerance(target)
                )
                segment_success.append(bool(segment_acquisition[episode, index - 1] and mean(within) >= 0.80))
            final = np.arange(max(0, end - round(1.0 / DT)), end)
            rows.append({
                "controller": args.controller,
                "family": "sequence",
                "condition": "0->0.6->1.2->2.0->1.2->0.6->0",
                "episode": episode, "episode_seed": SEED + episode,
                "fall": bool(falls[episode]),
                "sequence_completion": bool(all(segment_success) and not falls[episode]),
                "segment_success": segment_success,
                "heading_p95": percentile(abs(error), 95),
                "speed_mae": mean([
                    abs(vx[index, episode] - sequence_command(index * DT, SEQUENCE, 1.5)[0])
                    for index in range(end)
                ]),
                "feedback_duty_fraction": mean(gate[:end, episode] > 0),
                "yaw_command_p95": percentile(abs(yaw_command[:end, episode]), 95),
                "final_stand": bool(mean(abs(vx[final, episode]) <= 0.08) >= 0.95),
                "checkpoint_switches": 0,
            })
        print(f"STAGE10 {args.controller} sequence n=50", flush=True)
        return rows

    def close(self):
        self.wrapped.close()


def main():
    runtime = Runtime()
    try:
        rows = []
        if args.chunk.startswith("steady_"):
            pair = int(args.chunk.rsplit("_", 1)[1])
            for speed in STEADY[pair * 2:pair * 2 + 2]:
                rows.extend(runtime.collect("steady", speed, speed))
        elif args.chunk.startswith("low_"):
            pair = int(args.chunk.rsplit("_", 1)[1])
            for source, target in LOW[pair * 2:pair * 2 + 2]:
                rows.extend(runtime.collect("transition", source, target))
        elif args.chunk.startswith("anchor_"):
            pair = int(args.chunk.rsplit("_", 1)[1])
            for source, target in ANCHOR[pair * 2:pair * 2 + 2]:
                rows.extend(runtime.collect("transition", source, target))
        else:
            rows.extend(runtime.collect_sequence())
        dump(args.output / f"raw_{args.controller}_{args.chunk}.json", rows)
    finally:
        runtime.close()


try:
    main()
finally:
    simulation_app.close()
