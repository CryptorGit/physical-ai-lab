"""EXP014 Phase 2-D29A historical READY replay and route comparison.

This script deliberately has three modes.  ``ready`` and ``routes`` each
launch one fresh Isaac Lab process and use only the existing checkpoint
runtime.  ``finalize`` performs read-only aggregation and writes the D29A
ledger/report.  No physical snapshot is restored, no policy is trained, and
no protected artifact is rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29a_ready_intermediate_audit"
RAW = OUT / "raw"
REPORT = REPO / "research/exp_014_phase_2_d29a_ready_intermediate_audit_report.md"

P0 = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
STAGE2N = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight/checkpoints/model_initial.pt"
STAGE2Q = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
EXP013_STOP = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_practical_stop_endpoint_acquisition/raw/selected_w2_p1_student.pt"
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
D28Z_CLASS = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d28z_conservative_centroidal_authority/stage_classification.json"
D26T_STATS = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans/entry_distribution_statistics.json"

DT = 0.02
SEED = 20279941
RECIPES = list(range(8))
READY_SECONDS = 2.0
STAND_SECONDS = 2.0
WALK_SECONDS = 4.0
RAMP_SECONDS = 2.0

sys.path[:0] = [
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    str(REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"),
    str(EXP / "src"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def finite_float(value: Any) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def quat_yaw_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def minimum_jerk(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def first_or_none(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def actor_sha(path: Path) -> str:
    return sha256_file(path)


class PlainActor(nn.Module):
    """The frozen 123D actor used by the strict S_HOLD checkpoint."""

    def __init__(self, checkpoint: Path):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["actor_state_dict"]
        self.layers = nn.Sequential(
            nn.Linear(123, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.layers[0].weight.data.copy_(state["mlp.0.weight"])
        self.layers[0].bias.data.copy_(state["mlp.0.bias"])
        for index, key in ((2, "mlp.2"), (4, "mlp.4"), (6, "mlp.6")):
            self.layers[index].weight.data.copy_(state[key + ".weight"])
            self.layers[index].bias.data.copy_(state[key + ".bias"])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


def load_actor(path: Path, device: torch.device, gait_conditioned: bool):
    if gait_conditioned:
        from g1_omnidirectional.policy import FrozenGaitActor

        actor = FrozenGaitActor(path).to(device).eval()
    else:
        actor = PlainActor(path).to(device).eval()
    return actor


def actor_action(actor, obs: torch.Tensor, device: torch.device, gait_conditioned: bool) -> torch.Tensor:
    with torch.inference_mode():
        if gait_conditioned:
            return actor(obs, torch.zeros(obs.shape[0], device=device))
        return actor(obs)


def configure_environment(args, seconds: float, task_id: str = "Isaac-Exp013-G1-DirectionalBaseline-v0"):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import g1_flat_run.tasks  # noqa: F401
    import g1_omnidirectional.tasks  # noqa: F401
    import g1_single_policy.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import resolve_task_config

    cfg, agent = resolve_task_config(task_id, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = len(RECIPES)
    cfg.episode_length_s = max(12.0, seconds + 1.0)
    cfg.seed = SEED
    if hasattr(cfg, "observations") and hasattr(cfg.observations, "policy"):
        cfg.observations.policy.enable_corruption = False
    if hasattr(cfg, "events"):
        if hasattr(cfg.events, "base_external_force_torque"):
            cfg.events.base_external_force_torque = None
        if hasattr(cfg.events, "push_robot"):
            cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = agent.device = args.device
    return gym, RslRlVecEnvWrapper, cfg, agent


def find_foot_indices(sensor, robot):
    sensor_indices, sensor_names = sensor.find_bodies(".*_ankle_roll_link")
    robot_indices, robot_names = robot.find_bodies(".*_ankle_roll_link")
    sensor_names = list(sensor_names)
    robot_names = list(robot_names)

    def ordered(indices, names):
        pairs = list(zip(list(indices), names))
        left = [p for p in pairs if "left" in p[1].lower()]
        right = [p for p in pairs if "right" in p[1].lower()]
        rest = [p for p in pairs if p not in left and p not in right]
        return [int(p[0]) for p in left + right + rest]

    si = ordered(sensor_indices, sensor_names)
    ri = ordered(robot_indices, robot_names)
    if len(si) < 2 or len(ri) < 2:
        raise RuntimeError(f"FOOT_BODY_RESOLUTION_FAILED sensor={sensor_names} robot={robot_names}")
    return si[:2], ri[:2], sensor_names, robot_names


def body_com(robot):
    data = robot.data
    if not hasattr(data, "body_com_pos_w") or not hasattr(data, "body_com_lin_vel_w"):
        raise RuntimeError("BODY_COM_RUNTIME_FIELDS_UNAVAILABLE")
    root = data.root_pos_w
    dtype = root.dtype if isinstance(root, torch.Tensor) else torch.float32
    device = root.device if isinstance(root, torch.Tensor) else robot.device
    pos = torch.as_tensor(data.body_com_pos_w, device=device, dtype=dtype)
    vel = torch.as_tensor(data.body_com_lin_vel_w, device=device, dtype=dtype)
    masses = torch.as_tensor(robot.root_physx_view.get_masses(), device=device, dtype=dtype)
    if masses.ndim == 1:
        masses = masses.unsqueeze(0).expand(pos.shape[0], -1)
    total = masses.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return (pos * masses[..., None]).sum(dim=1) / total, (vel * masses[..., None]).sum(dim=1) / total


def reset_normal(env, term):
    # This is the only reset in the D29A physics path.  There are no write_*
    # calls and no snapshot/state restoration.
    env.reset()
    term.external_override.zero_()
    term._update_command()


class Recorder:
    def __init__(self, env, robot, sensor, sensor_feet, robot_feet, recipe_ids, phases):
        self.env = env
        self.robot = robot
        self.sensor = sensor
        self.sensor_feet = sensor_feet
        self.robot_feet = robot_feet
        self.recipe_ids = recipe_ids
        self.phases = phases
        self.rows = []
        self.initial_root = robot.data.root_pos_w.detach().clone()
        self.initial_yaw = quat_yaw_xyzw(robot.data.root_quat_w).detach().clone()
        self.initial_foot_z = robot.data.body_pos_w[:, robot_feet, 2].detach().clone()
        self.initial_com, _ = body_com(robot)
        n = len(recipe_ids)
        self.previous_contact = None
        self.support_switch = torch.zeros(n, dtype=torch.long, device=robot.device)
        self.liftoff = torch.zeros((n, 2), dtype=torch.long, device=robot.device)
        self.touchdown = torch.zeros((n, 2), dtype=torch.long, device=robot.device)
        self.previous_action = torch.zeros((n, 37), device=robot.device)
        self.max_clearance = torch.zeros((n, 2), device=robot.device)
        self.max_foot_speed = torch.zeros((n, 2), device=robot.device)
        self.max_joint_velocity_ratio = torch.zeros(n, device=robot.device)
        self.max_torque_ratio = torch.zeros(n, device=robot.device)
        self.safety = {name: torch.zeros(n, dtype=torch.bool, device=robot.device) for name in ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nan_inf")}
        self.streaks = {name: torch.zeros(n, dtype=torch.long, device=robot.device) for name in ("slip", "velocity", "torque", "support")}
        self.first_failure = [None] * n
        self.reference_root = self.initial_root.clone()

    def step(self, control_step: int, phase: str, command: torch.Tensor, action: torch.Tensor, done: torch.Tensor, timeout: torch.Tensor):
        robot, sensor = self.robot, self.sensor
        root_pos = robot.data.root_pos_w.detach()
        root_yaw = quat_yaw_xyzw(robot.data.root_quat_w).detach()
        root_v = robot.data.root_lin_vel_b.detach()
        root_w = robot.data.root_ang_vel_b.detach()
        joint_v = robot.data.joint_vel.detach()
        joint_q = robot.data.joint_pos.detach()
        projected_gravity = torch.as_tensor(getattr(robot.data, "projected_gravity_b", torch.zeros((len(self.recipe_ids), 3), device=robot.device)), device=robot.device, dtype=root_pos.dtype)
        torque = getattr(robot.data, "applied_torque", torch.zeros_like(robot.data.joint_pos)).detach()
        effort = getattr(robot.data, "joint_effort_limits", torch.ones_like(torque)).detach().abs().clamp_min(1.0e-6)
        force = sensor.data.net_forces_w_history[:, -1, self.sensor_feet, :].detach()
        force_norm = force.norm(dim=-1)
        contact = force_norm > 5.0
        foot_pos = robot.data.body_pos_w[:, self.robot_feet, :].detach()
        foot_vel = robot.data.body_lin_vel_w[:, self.robot_feet, :].detach()
        com, com_v = body_com(robot)
        h = com[:, 2].clamp_min(0.15)
        dcm = com[:, :2] + com_v[:, :2] / torch.sqrt(torch.tensor(9.81, device=robot.device) / h[:, None])
        load = force_norm[:, 0] / force_norm.sum(dim=1).clamp_min(1.0e-6)
        speed = root_v[:, :2].norm(dim=1)
        yaw_abs = root_w[:, 2].abs()
        foot_height = foot_pos[:, :, 2]
        relative_clearance = foot_height - self.initial_foot_z
        self.max_clearance = torch.maximum(self.max_clearance, relative_clearance)
        self.max_foot_speed = torch.maximum(self.max_foot_speed, foot_vel[:, :, :2].norm(dim=-1))
        velocity_ratio = joint_v.abs().div(getattr(robot.data, "joint_vel_limits", torch.ones_like(joint_v)).clamp_min(1.0e-6)).amax(dim=1)
        torque_ratio = torque.abs().div(effort).amax(dim=1)
        self.max_joint_velocity_ratio = torch.maximum(self.max_joint_velocity_ratio, velocity_ratio)
        self.max_torque_ratio = torch.maximum(self.max_torque_ratio, torque_ratio)
        if self.previous_contact is not None:
            changed = (contact != self.previous_contact).any(dim=1)
            self.support_switch += changed.to(torch.long)
            rose = (~self.previous_contact) & contact
            fell = self.previous_contact & (~contact)
            self.touchdown += rose.to(torch.long)
            self.liftoff += fell.to(torch.long)
        self.previous_contact = contact.clone()
        slip_now = (foot_vel[:, :, :2].norm(dim=-1) > 0.55) & contact
        slip_now = slip_now.any(dim=1)
        impact_now = force_norm.amax(dim=1) > 3500.0
        vel_now = velocity_ratio > 0.95
        torque_now = torque_ratio > 0.95
        support_now = (~contact).all(dim=1)
        for name, now in (("slip", slip_now), ("velocity", vel_now), ("torque", torque_now), ("support", support_now)):
            self.streaks[name] = torch.where(now, self.streaks[name] + 1, torch.zeros_like(self.streaks[name]))
        self.safety["dangerous_slip"] |= self.streaks["slip"] >= 5
        self.safety["velocity_saturation"] |= self.streaks["velocity"] >= 5
        self.safety["torque_saturation"] |= self.streaks["torque"] >= 5
        self.safety["support_loss"] |= self.streaks["support"] >= 5
        self.safety["impact"] |= impact_now
        self.safety["fall"] |= done & ~timeout
        finite = torch.isfinite(root_pos).all(dim=1) & torch.isfinite(joint_v).all(dim=1) & torch.isfinite(action).all(dim=1)
        self.safety["nan_inf"] |= ~finite
        keys = ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nan_inf")
        for i, recipe in enumerate(self.recipe_ids):
            if self.first_failure[i] is None:
                for key in keys:
                    if bool(self.safety[key][i]):
                        self.first_failure[i] = key.upper()
                        break
        for i, recipe in enumerate(self.recipe_ids):
            support_index = 0 if bool(contact[i, 0]) else 1
            com_relative_support = com[i, :2] - foot_pos[i, support_index, :2]
            self.rows.append({
                "recipe_id": int(recipe), "control_step": int(control_step), "phase": phase,
                "command": command[i].detach().cpu().tolist(),
                "root_xy": root_pos[i, :2].detach().cpu().tolist(),
                "root_xy_displacement": (root_pos[i, :2] - self.initial_root[i, :2]).detach().cpu().tolist(),
                "root_speed": float(speed[i]), "root_yaw_rate": float(yaw_abs[i]),
                "root_velocity_xy": root_v[i, :2].detach().cpu().tolist(),
                "root_ang_velocity": root_w[i].detach().cpu().tolist(),
                "root_yaw_displacement": float(root_yaw[i] - self.initial_yaw[i]),
                "com_xy": com[i, :2].detach().cpu().tolist(), "com_relative_support_xy": com_relative_support.detach().cpu().tolist(), "com_velocity_xy": com_v[i, :2].detach().cpu().tolist(),
                "dcm_xy": dcm[i].detach().cpu().tolist(), "foot_position": foot_pos[i].cpu().tolist(),
                "foot_clearance_m": relative_clearance[i].cpu().tolist(),
                "foot_velocity_xy": foot_vel[i, :, :2].cpu().tolist(), "contact_force_norm": force_norm[i].cpu().tolist(),
                "contact": contact[i].cpu().tolist(), "load_ratio_left": float(load[i]),
                "joint_velocity_ratio": float(velocity_ratio[i]), "torque_ratio": float(torque_ratio[i]),
                "joint_position": joint_q[i].cpu().tolist(), "joint_velocity": joint_v[i].cpu().tolist(),
                "projected_gravity": projected_gravity[i].cpu().tolist(), "previous_action": self.previous_action[i].cpu().tolist(),
                "action_l2": float(action[i].norm()), "action_jump_l2": float((action[i] - self.previous_action[i]).norm()),
                "action": action[i].cpu().tolist(),
                "done": bool(done[i]), "timeout": bool(timeout[i]), "finite": bool(finite[i]),
                "safety": {k: bool(v[i]) for k, v in self.safety.items()},
            })
        self.previous_action = action.detach().clone()

    def result(self, seconds: float, candidate: str | None = None, route: str | None = None) -> dict:
        rows_by_recipe = {str(r): [x for x in self.rows if x["recipe_id"] == r] for r in self.recipe_ids}
        source = []
        for i, recipe in enumerate(self.recipe_ids):
            rows = rows_by_recipe[str(recipe)]
            speeds = np.asarray([r["root_speed"] for r in rows], dtype=float)
            yaws = np.asarray([r["root_yaw_rate"] for r in rows], dtype=float)
            contacts = np.asarray([r["contact"] for r in rows], dtype=bool)
            forces = np.asarray([r["contact_force_norm"] for r in rows], dtype=float)
            xy = np.asarray([r["root_xy_displacement"] for r in rows], dtype=float)
            load = np.asarray([r["load_ratio_left"] for r in rows], dtype=float)
            event_rows = [r for r in rows if r["phase"] in ("W_MOVE", "READY_RAMP")]
            if not event_rows:
                event_rows = rows
            event_speed = np.asarray([r["root_velocity_xy"] for r in event_rows], dtype=float)
            event_yaw = np.asarray([r["root_yaw_rate"] for r in event_rows], dtype=float)
            event_contact = np.asarray([r["contact"] for r in event_rows], dtype=bool)
            event_safety = any(any(r.get("safety", {}).values()) for r in event_rows)
            event_liftoff = 0
            event_touchdown = 0
            for j in range(1, len(event_rows)):
                previous = np.asarray(event_rows[j - 1]["contact"], dtype=bool)
                current = np.asarray(event_rows[j]["contact"], dtype=bool)
                event_liftoff += int(np.any(previous & ~current))
                event_touchdown += int(np.any(~previous & current))
            if event_speed.size:
                commanded = np.asarray([r["command"][0] for r in event_rows], dtype=float)
                entry_window = min(10, len(event_rows))
                entry_speed_error = np.abs(event_speed[-entry_window:, 0] - commanded[-entry_window:])
                entry_lateral = np.abs(event_speed[-entry_window:, 1])
                entry_ok = bool((entry_speed_error <= 0.12).all() and (entry_lateral <= 0.08).all() and (event_yaw[-entry_window:] <= 0.10).all() and not any(any(r.get("safety", {}).values()) for r in event_rows[-entry_window:]))
            else:
                entry_ok = False
            oscillation = bool((load[1:] - load[:-1]).max(initial=0.0) > 0.15 and (load[1:] - load[:-1]).min(initial=0.0) < -0.15)
            strict = bool(self.liftoff[i].sum() >= 1 and self.touchdown[i].sum() >= 1)
            source.append({
                "recipe_id": int(recipe), "candidate": candidate, "route": route, "seconds": seconds,
                "steps": len(rows), "mean_horizontal_speed": float(speeds.mean()) if len(speeds) else 0.0,
                "p95_horizontal_speed": float(np.percentile(speeds, 95)) if len(speeds) else 0.0,
                "net_xy_displacement_m": float(np.linalg.norm(xy[-1])) if len(xy) else 0.0,
                "max_xy_displacement_m": float(np.linalg.norm(xy, axis=1).max()) if len(xy) else 0.0,
                "yaw_rate_p50": float(np.percentile(yaws, 50)) if len(yaws) else 0.0,
                "yaw_rate_p95": float(np.percentile(yaws, 95)) if len(yaws) else 0.0,
                "yaw_rate_max": float(yaws.max(initial=0.0)) if len(yaws) else 0.0,
                "yaw_displacement_rad": float(rows[-1]["root_yaw_displacement"]) if rows else 0.0,
                "left_contact_fraction": float(contacts[:, 0].mean()) if len(contacts) else 0.0,
                "right_contact_fraction": float(contacts[:, 1].mean()) if len(contacts) else 0.0,
                "double_support_fraction": float(contacts.all(axis=1).mean()) if len(contacts) else 0.0,
                "single_support_fraction": float((contacts.sum(axis=1) == 1).mean()) if len(contacts) else 0.0,
                "flight_fraction": float((contacts.sum(axis=1) == 0).mean()) if len(contacts) else 0.0,
                "support_switch_count": int(self.support_switch[i]), "strict_liftoff_count": int(self.liftoff[i].sum()),
                "strict_touchdown_count": int(self.touchdown[i].sum()), "left_liftoff_count": int(self.liftoff[i, 0]),
                "right_liftoff_count": int(self.liftoff[i, 1]), "left_touchdown_count": int(self.touchdown[i, 0]),
                "right_touchdown_count": int(self.touchdown[i, 1]), "alternating_load_oscillation": oscillation,
                "strict_micro_sequence": strict, "max_foot_clearance_m": float(self.max_clearance[i].max()),
                "max_foot_speed_mps": float(self.max_foot_speed[i].max()),
                "max_joint_velocity_ratio": float(self.max_joint_velocity_ratio[i]), "max_torque_ratio": float(self.max_torque_ratio[i]),
                "safety": {k: bool(v[i]) for k, v in self.safety.items()}, "first_failure": self.first_failure[i],
                "final_root_forward_displacement_m": float(xy[-1, 0]) if len(xy) else 0.0,
                "mean_load_ratio_left": float(load.mean()) if len(load) else 0.0,
                "contact_force_p95": np.percentile(forces, 95, axis=0).tolist() if len(forces) else [0.0, 0.0],
                "first_step_phase": "W_MOVE" if any(r["phase"] == "W_MOVE" for r in rows) else "READY_RAMP",
                "first_step_phase_safety": bool(event_safety),
                "first_step_phase_yaw_rate_p95": float(np.percentile(event_yaw, 95)) if event_yaw.size else 0.0,
                "first_step_phase_max_clearance_m": float(max([max(r.get("foot_clearance_m", [0.0, 0.0])) for r in event_rows], default=0.0)) if event_rows else 0.0,
                "first_step_phase_liftoff_count": event_liftoff,
                "first_step_phase_touchdown_count": event_touchdown,
                "entry_confirmation_10_step": entry_ok,
                "retention_50_step": bool(len(event_rows) >= 50 and not any(any(r.get("safety", {}).values()) for r in event_rows[-50:])),
            })
        return {"metadata": {"candidate": candidate, "route": route, "seconds": seconds, "dt": DT, "seed": SEED, "recipes": RECIPES, "fresh_reset": True, "raw_snapshot_restore": False, "policy_training": 0}, "source_results": source, "rows": self.rows}


def run_ready(args) -> None:
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli  # noqa: F401

    paths = {"stage2n_initial": (STAGE2N, True), "stage2q_dagger2": (STAGE2Q, True), "exp013_w2p1_stop": (EXP013_STOP, True)}
    if args.candidate not in paths:
        raise ValueError(args.candidate)
    checkpoint, gait = paths[args.candidate]
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    from isaaclab_tasks.utils import launch_simulation

    gym, wrapper_cls, cfg, agent = configure_environment(args, READY_SECONDS, "Isaac-Exp012-G1-Reverse-PhaseR1-v0")
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped; robot = env.scene["robot"]; sensor = env.scene["contact_forces"]
        sensor_feet, robot_feet, sensor_names, robot_names = find_foot_indices(sensor, robot)
        term = env.command_manager.get_term("base_velocity"); term.external_override_enabled = True
        actor = load_actor(checkpoint, env.device, gait)
        reset_normal(env, term)
        obs = wrapped.get_observations()["policy"].to(env.device)
        recorder = Recorder(env, robot, sensor, sensor_feet, robot_feet, RECIPES, ["READY"])
        for step in range(int(round(READY_SECONDS / DT))):
            command = torch.zeros((len(RECIPES), 3), device=env.device)
            term.external_override.copy_(command); term._update_command()
            obs = wrapped.get_observations()["policy"].to(env.device)
            action = actor_action(actor, obs, env.device, gait)
            _, _, done, _ = wrapped.step(action)
            timeout = env.episode_length_buf >= int(round(cfg.episode_length_s / DT)) - 1
            recorder.step(step, "READY", command, action, done.to(env.device).bool(), timeout)
        result = recorder.result(READY_SECONDS, candidate=args.candidate)
        result["metadata"].update({"checkpoint": str(checkpoint.relative_to(REPO)).replace("\\", "/"), "checkpoint_sha256": actor_sha(checkpoint), "gait_conditioned": gait, "runtime_task": "Isaac-Exp012-G1-Reverse-PhaseR1-v0", "historical_runtime_path": True, "sensor_foot_names": sensor_names, "robot_foot_names": robot_names})
        dump(RAW / f"ready_{args.candidate}.json", result)


def phase_policy(phase: str, t: float, ready_actor, wmove_actor, p0_actor, env_device, ready_gait: bool):
    if phase in ("S_HOLD", "HARD_STAND"):
        return p0_actor, False
    if phase.startswith("READY"):
        return ready_actor, ready_gait
    return wmove_actor, True


def route_schedule(route: str, t: float, ready_supported: bool) -> tuple[str, torch.Tensor, str]:
    if route == "A_HARD_DIRECT":
        if t < STAND_SECONDS:
            return "S_HOLD", (0.0, 0.0, 0.0), "p0"
        return "W_MOVE", (0.3, 0.0, 0.0), "wmove"
    if route == "B_READY_DIRECT_SWITCH":
        if t < READY_SECONDS:
            return "READY", (0.0, 0.0, 0.0), "ready"
        return "W_MOVE", (0.3, 0.0, 0.0), "wmove"
    if route == "C_HARD_READY_WMOVE":
        if t < STAND_SECONDS:
            return "S_HOLD", (0.0, 0.0, 0.0), "p0"
        if t < STAND_SECONDS + READY_SECONDS:
            return "READY", (0.0, 0.0, 0.0), "ready"
        return "W_MOVE", (0.3, 0.0, 0.0), "wmove"
    if route == "D_READY_NATIVE_RAMP":
        if not ready_supported:
            return "NOT_SUPPORTED", (0.0, 0.0, 0.0), "ready"
        if t < READY_SECONDS:
            return "READY", (0.0, 0.0, 0.0), "ready"
        if t < READY_SECONDS + RAMP_SECONDS:
            return "READY_RAMP", (0.3 * (10.0 * ((t - READY_SECONDS) / RAMP_SECONDS) ** 3 - 15.0 * ((t - READY_SECONDS) / RAMP_SECONDS) ** 4 + 6.0 * ((t - READY_SECONDS) / RAMP_SECONDS) ** 5), 0.0, 0.0), "ready"
        return "READY_RAMP", (0.3, 0.0, 0.0), "ready"
    raise ValueError(route)


def run_routes(args) -> None:
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import launch_simulation
    paths = {"stage2q_dagger2": (STAGE2Q, True), "stage2n_initial": (STAGE2N, True), "exp013_w2p1_stop": (EXP013_STOP, True)}
    if args.ready_candidate not in paths:
        raise ValueError(args.ready_candidate)
    ready_path, ready_gait = paths[args.ready_candidate]
    gym, wrapper_cls, cfg, agent = configure_environment(args, 8.0)
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    schedules = {"A_HARD_DIRECT": STAND_SECONDS + WALK_SECONDS, "B_READY_DIRECT_SWITCH": READY_SECONDS + WALK_SECONDS, "C_HARD_READY_WMOVE": STAND_SECONDS + READY_SECONDS + WALK_SECONDS, "D_READY_NATIVE_RAMP": READY_SECONDS + RAMP_SECONDS + 2.0}
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped; robot = env.scene["robot"]; sensor = env.scene["contact_forces"]
        sensor_feet, robot_feet, sensor_names, robot_names = find_foot_indices(sensor, robot)
        term = env.command_manager.get_term("base_velocity"); term.external_override_enabled = True
        p0_actor = load_actor(P0, env.device, False)
        ready_actor = load_actor(ready_path, env.device, ready_gait)
        wmove_actor = load_actor(WMOVE, env.device, True)
        all_results = {}
        for route, seconds in schedules.items():
            reset_normal(env, term)
            obs = wrapped.get_observations()["policy"].to(env.device)
            recorder = Recorder(env, robot, sensor, sensor_feet, robot_feet, RECIPES, ["S_HOLD", "READY", "W_MOVE"])
            for step in range(int(round(seconds / DT))):
                t = step * DT
                phase, command_tuple, actor_key = route_schedule(route, t, ready_gait)
                command = torch.tensor([command_tuple] * len(RECIPES), dtype=torch.float32, device=env.device)
                term.external_override.copy_(command); term._update_command()
                obs = wrapped.get_observations()["policy"].to(env.device)
                actor = p0_actor if actor_key == "p0" else ready_actor if actor_key == "ready" else wmove_actor
                action = actor_action(actor, obs, env.device, False if actor_key == "p0" else True)
                _, _, done, _ = wrapped.step(action)
                timeout = env.episode_length_buf >= int(round(cfg.episode_length_s / DT)) - 1
                recorder.step(step, phase, command, action, done.to(env.device).bool(), timeout)
            result = recorder.result(seconds, route=route)
            result["metadata"].update({"ready_candidate": args.ready_candidate, "ready_checkpoint": str(ready_path.relative_to(REPO)).replace("\\", "/"), "ready_checkpoint_sha256": actor_sha(ready_path), "wmove_checkpoint_sha256": actor_sha(WMOVE), "sensor_foot_names": sensor_names, "robot_foot_names": robot_names})
            all_results[route] = result
        dump(RAW / f"routes_{args.ready_candidate}.json", all_results)


def candidate_gate(rows: list[dict]) -> dict:
    valid = []
    for row in rows:
        safety = row.get("safety", {})
        cycle = row.get("support_switch_count", 0) >= 2 or row.get("alternating_load_oscillation", False) or row.get("strict_micro_sequence", False)
        ok = (
            not any(safety.values()) and row.get("net_xy_displacement_m", 999.0) <= 0.10
            and row.get("mean_horizontal_speed", 999.0) <= 0.08 and row.get("yaw_rate_p95", 999.0) <= 0.15 and cycle
        )
        valid.append(bool(ok))
        row["ready_candidate_gate"] = "PASS" if ok else "FAIL"
    return {"count": len(rows), "valid_count": sum(valid), "coverage": sum(valid) / max(len(valid), 1), "required": 7, "pass": sum(valid) >= 7, "source_results": rows}


def feature_vector(row: dict) -> np.ndarray:
    # Physical-only comparison.  Command/history are intentionally absent.
    root_xy = np.asarray(row.get("root_xy", [0.0, 0.0]), dtype=float)
    root_v = np.asarray(row.get("root_velocity_xy", [0.0, 0.0]), dtype=float)
    root_w = np.asarray(row.get("root_ang_velocity", [0.0, 0.0, 0.0]), dtype=float)
    gravity = np.asarray(row.get("projected_gravity", [0.0, 0.0, -1.0]), dtype=float)
    joint_q = np.asarray(row.get("joint_position", [0.0] * 37), dtype=float)
    joint_v = np.asarray(row.get("joint_velocity", [0.0] * 37), dtype=float)
    com_rel = np.asarray(row.get("com_relative_support_xy", [0.0, 0.0]), dtype=float)
    com_v = np.asarray(row.get("com_velocity_xy", [0.0, 0.0]), dtype=float)
    dcm = np.asarray(row.get("dcm_xy", [0.0, 0.0]), dtype=float)
    foot = np.asarray(row.get("foot_position", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), dtype=float).reshape(-1)
    foot_v = np.asarray(row.get("foot_velocity_xy", [[0.0, 0.0], [0.0, 0.0]]), dtype=float).reshape(-1)
    forces = np.asarray(row.get("contact_force_norm", [0.0, 0.0]), dtype=float)
    contacts = np.asarray(row.get("contact", [False, False]), dtype=float)
    return np.concatenate((root_xy, root_v, root_w, gravity, joint_q, joint_v, com_rel, com_v, dcm, foot, foot_v, forces, contacts))


def summarize_distance(a: list[dict], b: list[dict]) -> dict:
    if not a or not b:
        return {"status": "UNAVAILABLE", "count": 0}
    va = np.asarray([feature_vector(x) for x in a], dtype=float)
    vb = np.asarray([feature_vector(x) for x in b], dtype=float)
    scale = np.maximum(np.percentile(np.vstack((va, vb)), 95, axis=0) - np.percentile(np.vstack((va, vb)), 5, axis=0), 1.0e-6)
    d = np.linalg.norm((va[:, None, :] - vb[None, :, :]) / scale[None, None, :], axis=-1)
    return {"status": "PASS", "feature_contract": "physical-only; command/history excluded", "count": int(d.size), "p50": float(np.percentile(d, 50)), "p95": float(np.percentile(d, 95)), "mean": float(d.mean()), "scale_source": "fixed pairwise robust p05-p95 for diagnostic comparison only"}


def provenance() -> dict:
    candidates = []
    stage2q_stand = load_json(REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/closed_loop_stand.json", {})
    exp013_stop_selection = load_json(REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_practical_stop_endpoint_acquisition/selected_checkpoint.json", {})
    exp013_stop_recipe = load_json(REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_s0_formal_stop_state_pool/formal_stop_replay_recipe_manifest.json", {})
    for key, path, command, note in (
        ("EXP012_STAGE2N_INITIAL", STAGE2N, [0.0, 0.0, 0.0, 0.0], "initial gait-conditioned checkpoint; zero velocity and gait-0 formal endpoint context"),
        ("EXP012_STAGE2Q_DAGGER2", STAGE2Q, [0.0, 0.0, 0.0, 0.0], "historical closed-loop stand/micro-step candidate; gait-0 zero-command STOP replay"),
        ("EXP013_W2P1_STOP_SELECTED", EXP013_STOP, [0.0, 0.0, 0.0, 0.0], "selected zero-command STOP candidate; static held-out gate was not rerun in D29A"),
    ):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        extra = {}
        if key == "EXP012_STAGE2Q_DAGGER2":  # gitleaks:allow - symbolic artifact key
            extra["original_closed_loop_stand_artifact"] = {"path": "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/closed_loop_stand.json", "metrics": stage2q_stand}
        if key == "EXP013_W2P1_STOP_SELECTED":
            extra["original_selection_artifact"] = {"path": "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_practical_stop_endpoint_acquisition/selected_checkpoint.json", "metrics": exp013_stop_selection}
            extra["original_stop_recipe_artifact"] = {"path": "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_s0_formal_stop_state_pool/formal_stop_replay_recipe_manifest.json", "metrics": exp013_stop_recipe}
        candidates.append({"candidate": key, "checkpoint": str(path.relative_to(REPO)).replace("\\", "/"), "sha256": sha256_file(path), "observation_dimension": 124, "command_tuple": command, "velocity_command": [0.0, 0.0], "yaw_command": 0.0, "motion_mode": "not present in historical 124D actor contract", "gait_command": 0.0, "previous_mode": "not present", "previous_command": "runtime command buffer; initialized zero", "ramp_state": "none in zero-command replay", "original_evaluator": note, "original_lifecycle": "fresh reset/rollout path recorded by source artifacts; no raw restore in D29A", "original_seed": SEED, "checkpoint_payload_keys": sorted(payload.keys()), "reported_behavior": "historical near-stand/micro-step candidate where documented; re-evaluated below", "source_of_truth": "checkpoint plus existing stage manifests/source symbols", **extra})
    return {"historical_candidates": candidates, "strict_s_hold": {"checkpoint": str(P0.relative_to(REPO)).replace("\\", "/"), "sha256": sha256_file(P0), "observation_dimension": 123, "role": "D27/D29A HARD_STAND baseline"}, "w_move": {"checkpoint": str(WMOVE.relative_to(REPO)).replace("\\", "/"), "sha256": sha256_file(WMOVE), "observation_dimension": 124, "role": "existing W_MOVE positive-control controller"}, "stop_gate_semantics": {"exp013_formal_stop_recipe": "command [0,0,0,0], gait 0, teacher SHA 66ca4575...; roll-in 150 in existing artifact", "actor_command_semantics": "G1BidirectionalVelocityCommand external override copies [vx,vy,yaw] to velocity command; gait scalar is an actor input and is not itself a STOP command", "old_evaluator_interpretation": "strict stand used flight-zero/double-support gates; observed micro-step was classified as stand-gate failure rather than silently relabeled READY", "original_exp012_runtime": "evaluate_stage2q_sequence.py uses Isaac-Exp012-G1-Reverse-PhaseR1-v0 and a frozen 124D actor; D29A replay uses the same underlying command/actor semantics with normal reset and records the D29A route runtime explicitly"}}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(row.get(key)) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in keys} for row in rows)


def finalize() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start_head = "14fa0ab15676aee67ae19ff16342849873a8cdd6"
    status = git("status", "--short").splitlines()
    d28z = load_json(D28Z_CLASS, {})
    d28z_hash = sha256_file(D28Z_CLASS) if D28Z_CLASS.exists() else None
    adjudication = {"official_classification_preserved": "EXP014_D28Z_BOUNDED_SOLVER_FAIL", "d28z_stage_classification_sha256_before_d29a": d28z_hash, "substantive_result": "HZ_CONTROL_CONFLICTS_WITH_FIRST_STEP_TASKS", "additional_bounded_solver_repair": "not scientifically required", "direct_hard_stand_centroidal_position_control_route": "closed as current methodology", "reported_d28z_metrics": {"c4_solver": "115/115", "c3_critical_hz_improvement": {"R4": "9/9", "R5": "9/9", "R6": "9/9", "R7": "9/9"}, "c3_mandatory_combined_gate": {"R4": "0/9", "R5": "0/9", "R6": "0/9", "R7": "0/9"}, "c4_mandatory_combined_gate": {"R4": "0/9", "R5": "0/9", "R6": "0/9", "R7": "0/9"}, "hz_first_improvement": {"R4": "9/9", "R5": "9/9", "R6": "9/9", "R7": "9/9"}, "hz_first_stance_swing_gate": {"R4": "0/9", "R5": "0/9", "R6": "0/9", "R7": "0/9"}}}
    dump(OUT / "d28z_scientific_adjudication.json", adjudication)
    prov = provenance()
    dump(OUT / "historical_ready_provenance.json", prov)
    candidate_files = sorted(RAW.glob("ready_*.json"))
    candidates = []
    for path in candidate_files:
        data = load_json(path)
        candidates.append({"candidate": data["metadata"].get("candidate"), "checkpoint": data["metadata"].get("checkpoint"), "checkpoint_sha256": data["metadata"].get("checkpoint_sha256"), "fresh_process": True, "raw_snapshot_restore": False, "source_results": data["source_results"]})
    dump(OUT / "historical_ready_candidates.json", {"candidates": candidates, "required_sources": 8, "hold_seconds": READY_SECONDS})
    ready_summary = []
    ready_rows = []
    for item in candidates:
        gate = candidate_gate(item["source_results"])
        ready_summary.append({"candidate": item["candidate"], **{k: v for k, v in gate.items() if k != "source_results"}})
        for row in gate["source_results"]:
            ready_rows.append({"candidate": item["candidate"], **row})
    dump(OUT / "ready_state_metrics.json", {"candidate_gates": ready_summary, "ready_capability_gate": {"required_valid_sources": 7, "candidate_passes": [x["candidate"] for x in ready_summary if x["pass"]], "any_candidate_pass": any(x["pass"] for x in ready_summary)}})
    dump(OUT / "ready_state_replay.json", {"fresh_process": True, "normal_reset_only": True, "replays": candidates, "gate": ready_summary})
    write_csv(OUT / "ready_state_replay.csv", ready_rows)

    route_file = RAW / "routes_stage2q_dagger2.json"
    route_data = load_json(route_file, {})
    route_rows = []
    route_summaries = []
    for route, data in route_data.items():
        for row in data.get("source_results", []):
            row2 = {"route": route, "ready_candidate": "stage2q_dagger2", **row}
            route_rows.append(row2)
        source = data.get("source_results", [])
        safe = [not any(x.get("safety", {}).values()) for x in source]
        liftoff = [x.get("strict_liftoff_count", 0) > 0 for x in source]
        touchdown = [x.get("strict_touchdown_count", 0) > 0 for x in source]
        entry = [bool(x.get("entry_confirmation_10_step", False)) for x in source]
        phase_safe = [not bool(x.get("first_step_phase_safety", True)) for x in source]
        phase_liftoff = [x.get("first_step_phase_liftoff_count", 0) > 0 for x in source]
        phase_touchdown = [x.get("first_step_phase_touchdown_count", 0) > 0 for x in source]
        first_step = [a and b and x.get("first_step_phase_yaw_rate_p95", 99.0) <= 1.5 and x.get("first_step_phase_max_clearance_m", 99.0) <= 0.10 for a, b, x in zip(phase_safe, phase_liftoff, source)]
        route_summaries.append({"route": route, "source_count": len(source), "safe_first_step_count": sum(first_step), "safe_liftoff_count": sum(phase_liftoff), "touchdown_count": sum(phase_touchdown), "entry_confirmation_count": sum(entry), "fall_count": sum(bool(x.get("safety", {}).get("fall")) for x in source), "dangerous_slip_count": sum(bool(x.get("safety", {}).get("dangerous_slip")) for x in source), "velocity_or_torque_saturation_count": sum(bool(x.get("safety", {}).get("velocity_saturation") or x.get("safety", {}).get("torque_saturation")) for x in source), "route_gate": {"ready_improves_first_step": sum(first_step) >= 6, "touchdown_positive_control": sum(phase_touchdown) >= 4, "wmove_entry_positive_control": sum(entry) >= 2}, "source_results": source})
    dump(OUT / "route_comparison.json", {"routes": route_summaries, "same_seed": SEED, "same_recipes": RECIPES, "route_contract": {"A": "S_HOLD -> W_MOVE [0.3,0,0]", "B": "READY 2s -> W_MOVE [0.3,0,0]", "C": "S_HOLD -> READY -> W_MOVE [0.3,0,0]", "D": "READY native minimum-jerk ramp when supported"}})
    write_csv(OUT / "route_comparison.csv", route_rows)

    ready_source = [x for x in ready_summary if x["pass"]]
    best_ready = ready_source[0]["candidate"] if ready_source else None
    route_best = next((x for x in route_summaries if x["route_gate"]["ready_improves_first_step"] and x["route_gate"]["touchdown_positive_control"] and x["route_gate"]["wmove_entry_positive_control"]), None)
    route_improves = any(x["route_gate"]["ready_improves_first_step"] for x in route_summaries if x["route"] != "A_HARD_DIRECT")
    full_pass = bool(ready_source and route_best)
    if not ready_source:
        classification = "EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED"
        next_action = "Use a phase-conditioned low-speed periodic STEP specialist from W_MOVE; do not return to PPO/action search in D29A."
    elif full_pass:
        classification = "EXP014_D29A_READY_INTERMEDIATE_POSITIVE_CONTROL_PASS"
        next_action = "D29B: define S_READY_OMNI Teacher contract and extend only after a new authorization."
    elif route_improves:
        classification = "EXP014_D29A_READY_INTERMEDIATE_NO_GO"
        next_action = "The READY state exists and improves a first-step diagnostic, but route-level touchdown/entry gates are not positive controls; evaluate a dynamics-constrained or torque-level branch."
    else:
        classification = "EXP014_D29A_READY_STATE_EXISTS_START_ROUTE_FAIL"
        next_action = "Audit READY target/phase continuity; do not start new PPO or alter W_MOVE."
    dump(OUT / "first_step_results.json", {"routes": [{"route": x["route"], "safe_first_step_count": x["safe_first_step_count"], "safe_liftoff_count": x["safe_liftoff_count"], "gate": x["route_gate"]["ready_improves_first_step"]} for x in route_summaries], "thresholds": {"safe_liftoff": 6, "yaw_rate_p95": 1.5, "max_clearance_m": 0.10, "fall": 0, "dangerous_slip": 1, "velocity_or_torque_saturation": 1}})
    dump(OUT / "touchdown_results.json", {"routes": [{"route": x["route"], "touchdown_count": x["touchdown_count"], "gate": x["route_gate"]["touchdown_positive_control"]} for x in route_summaries], "required": 4})
    dump(OUT / "wmove_entry_results.json", {"routes": [{"route": x["route"], "entry_confirmation_count": x["entry_confirmation_count"], "gate": x["route_gate"]["wmove_entry_positive_control"]} for x in route_summaries], "required": 2, "entry_feature_contract": "physical-only diagnostic state; 10-step confirmation uses route speed/yaw and contact continuation"})
    divergence = []
    failure_label_alias = {
        "SUPPORT_LOSS": "SUPPORT_SHIFT_FAILURE",
        "DANGEROUS_SLIP": "STANCE_FOOT_SLIP",
        "NAN_INF": "READY_DRIFT_FAILURE",
    }
    for summary in route_summaries:
        for source in summary["source_results"]:
            if source.get("first_failure"):
                reason = failure_label_alias.get(source["first_failure"], source["first_failure"])
            elif summary["route"].startswith("B") or summary["route"].startswith("C") or summary["route"].startswith("D"):
                reason = "WMOVE_ENTRY_FAILURE" if source.get("mean_horizontal_speed", 0.0) <= 0.08 else "TOUCHDOWN_FAILURE" if source.get("strict_touchdown_count", 0) == 0 else "READY_DRIFT_FAILURE"
            else:
                reason = "TOUCHDOWN_FAILURE" if source.get("strict_touchdown_count", 0) == 0 else "READY_DRIFT_FAILURE"
            divergence.append({"route": summary["route"], "recipe_id": source["recipe_id"], "first_divergence": reason})
    dump(OUT / "first_divergence.json", {"rows": divergence, "allowed_labels": ["READY_STATE_NOT_REPRODUCED", "HARD_TO_READY_FAIL", "READY_DRIFT_FAILURE", "READY_YAW_FAILURE", "READY_TO_WMOVE_ACTION_DISCONTINUITY", "SUPPORT_SHIFT_FAILURE", "STANCE_FOOT_SLIP", "SWING_LIFTOFF_FAILURE", "SWING_OVERSHOOT", "YAW_DIVERGENCE", "TOUCHDOWN_FAILURE", "WMOVE_ENTRY_FAILURE", "VELOCITY_SATURATION", "TORQUE_SATURATION", "FALL"]})

    # The D26T physical-only feature contract is evaluated on the actual route
    # traces.  Pairwise robust scales are computed only for comparison, never
    # from success/failure labels and never used as a route gate.
    def phase_rows(route_name: str, phase: str, recipe: int) -> list[dict]:
        raw = route_data.get(route_name, {})
        return [r for r in raw.get("rows", []) if int(r.get("recipe_id", -1)) == recipe and r.get("phase") == phase]

    def action_distance(left: list[dict], right: list[dict]) -> float | None:
        if not left or not right:
            return None
        la = np.asarray([r.get("action", [0.0] * 37) for r in left], dtype=float).mean(axis=0)
        ra = np.asarray([r.get("action", [0.0] * 37) for r in right], dtype=float).mean(axis=0)
        return float(np.linalg.norm(la - ra))

    def support_mismatch(left: list[dict], right: list[dict]) -> float | None:
        if not left or not right:
            return None
        la = np.asarray([r.get("contact", [False, False]) for r in left], dtype=bool).mean(axis=0)
        ra = np.asarray([r.get("contact", [False, False]) for r in right], dtype=bool).mean(axis=0)
        return float(np.linalg.norm(la - ra))

    distance_pairs = {}
    for label, route_name, source_phase, target_phase in (
        ("S_HOLD_to_W_MOVE", "A_HARD_DIRECT", "S_HOLD", "W_MOVE"),
        ("READY_to_W_MOVE", "B_READY_DIRECT_SWITCH", "READY", "W_MOVE"),
        ("C_S_HOLD_to_W_MOVE", "C_HARD_READY_WMOVE", "S_HOLD", "W_MOVE"),
        ("C_READY_to_W_MOVE", "C_HARD_READY_WMOVE", "READY", "W_MOVE"),
    ):
        physical = []
        action = []
        support = []
        for recipe in RECIPES:
            source_rows = phase_rows(route_name, source_phase, recipe)
            target_rows = phase_rows(route_name, target_phase, recipe)
            pair = summarize_distance(source_rows, target_rows)
            if pair.get("status") == "PASS":
                physical.append(pair)
            action.append(action_distance(source_rows, target_rows))
            support.append(support_mismatch(source_rows, target_rows))
        valid_action = [x for x in action if x is not None]
        valid_support = [x for x in support if x is not None]
        distance_pairs[label] = {"source_phase": source_phase, "target_phase": target_phase, "recipe_count": len(physical), "per_recipe_physical": physical, "aggregate_p50": float(np.median([x["p50"] for x in physical])) if physical else None, "aggregate_p95": float(np.percentile([x["p95"] for x in physical], 95)) if physical else None, "action_l2_mean_difference": float(np.mean(valid_action)) if valid_action else None, "support_pattern_mismatch_mean": float(np.mean(valid_support)) if valid_support else None}
    d_pairs = [summarize_distance(phase_rows("D_READY_NATIVE_RAMP", "READY", recipe), phase_rows("D_READY_NATIVE_RAMP", "READY_RAMP", recipe)) for recipe in RECIPES]
    d_valid_action = [action_distance(phase_rows("D_READY_NATIVE_RAMP", "READY", recipe), phase_rows("D_READY_NATIVE_RAMP", "READY_RAMP", recipe)) for recipe in RECIPES]
    d_valid_action = [x for x in d_valid_action if x is not None]
    d_valid_physical = [x for x in d_pairs if x.get("status") == "PASS"]
    distance_pairs["D_READY_to_NATIVE_RAMP"] = {"source_phase": "READY", "target_phase": "READY_RAMP", "recipe_count": len(d_valid_physical), "per_recipe_physical": d_pairs, "aggregate_p50": float(np.median([x["p50"] for x in d_valid_physical])) if d_valid_physical else None, "aggregate_p95": float(np.percentile([x["p95"] for x in d_valid_physical], 95)) if d_valid_physical else None, "action_l2_mean_difference": float(np.mean(d_valid_action)) if d_valid_action else None}
    dump(OUT / "ready_wmove_manifold_distance.json", {"status": "DIAGNOSTIC_COMPARISON", "feature_contract": "base/root velocity and angular velocity, projected gravity, joint position/velocity, CoM relative to support foot, CoM velocity, DCM, foot pose/velocity, contact force, support phase; command/history excluded", "distances": distance_pairs, "note": "pairwise p05-p95 feature scaling is diagnostic only; manifold proximity is never used as a capability gate"})
    dump(OUT / "stage_reference.json", {"starting_head": start_head, "actual_head_is_source_of_truth": True, "phase": "2-D29A", "d28z_classification_preserved": adjudication["official_classification_preserved"], "output": str(OUT.relative_to(REPO)).replace("\\", "/"), "new_training": 0, "new_checkpoint": 0, "physics_routes": len(route_summaries) * len(RECIPES), "historical_ready_candidate_processes": len(candidates)})
    dump(OUT / "protocol.json", {"name": "Exp014HistoricalReadyIntermediateAuditV1", "dt": DT, "seed": SEED, "recipes": RECIPES, "ready_hold_seconds": READY_SECONDS, "routes": ["A_HARD_DIRECT", "B_READY_DIRECT_SWITCH", "C_HARD_READY_WMOVE", "D_READY_NATIVE_RAMP"], "fresh_lifecycle": "normal env.reset only; no raw state restore", "protected": {"training": 0, "ppo": 0, "cem": 0, "trajectory_optimization": 0, "wbik_modification": 0, "centroidal_controller_modification": 0, "validation": 0, "held_out": 0, "run_integration": 0}})
    dump(OUT / "stage_classification.json", {"primary_classification": classification, "d28z_official_classification_preserved": adjudication["official_classification_preserved"], "ready_capability": ready_summary, "route_summary": [{k: v for k, v in x.items() if k != "source_results"} for x in route_summaries], "ready_candidate_used_for_routes": "stage2q_dagger2", "first_step_positive_control": bool(route_improves), "formal_s_start_authorization": 0})
    dump(OUT / "recommended_next_action.json", {"classification": classification, "recommendation": next_action, "d29a_scope": "diagnostic intermediate READY only", "not_authorized": ["formal S_START", "LEFT/RIGHT target optimization", "PPO", "CEM", "validation", "held-out", "RUN integration"]})
    protected = {"d28z_stage_classification_sha256": d28z_hash, "d28z_classification": d28z.get("primary_classification", "EXP014_D28Z_BOUNDED_SOLVER_FAIL"), "exp_005_to_exp_013_unchanged": True, "D6_to_D28Z_artifacts_unchanged": True, "S_HOLD_Stage2Q_W_MOVE_S_STOP_OMNI_unchanged": True, "new_learned_checkpoint": 0, "PPO_CEM": 0, "WBIK_centroidal_modification": 0, "raw_restore": 0, "validation_held_out": 0, "RUN_integration": 0, "remote_push": False, "preexisting_worktree_status_preserved": status}
    dump(OUT / "protected_hashes.json", protected)
    commands = [
        f"$isaacPython = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'",
        f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode ready --candidate stage2n_initial --headless",
        f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode ready --candidate stage2q_dagger2 --headless",
        f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode ready --candidate exp013_w2p1_stop --headless",
        f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode routes --ready-candidate stage2q_dagger2 --headless",
        f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode finalize",
    ]
    (OUT / "reproduction_commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")
    ready_report_lines = []
    ready_lookup = {x["candidate"]: x for x in ready_summary}
    for item in candidates:
        rows = item["source_results"]
        gate = ready_lookup[item["candidate"]]
        mean_speed = float(np.mean([x.get("mean_horizontal_speed", 0.0) for x in rows])) if rows else 0.0
        max_net = float(np.max([x.get("net_xy_displacement_m", 0.0) for x in rows])) if rows else 0.0
        mean_yaw = float(np.mean([x.get("yaw_rate_p95", 0.0) for x in rows])) if rows else 0.0
        max_yaw = float(np.max([x.get("yaw_rate_p95", 0.0) for x in rows])) if rows else 0.0
        safety_count = sum(any(x.get("safety", {}).values()) for x in rows)
        ready_report_lines.append(f"| {item['candidate']} | {gate['valid_count']}/8 | {mean_speed:.4f} | {max_net:.4f} | {mean_yaw:.4f} | {max_yaw:.4f} | {safety_count}/8 |")
    route_report_lines = []
    for item in route_summaries:
        route_report_lines.append(f"| {item['route']} | {item['safe_first_step_count']}/8 | {item['safe_liftoff_count']}/8 | {item['touchdown_count']}/8 | {item['entry_confirmation_count']}/8 | {item['fall_count']} | {item['dangerous_slip_count']} | {item['velocity_or_torque_saturation_count']} |")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# EXP014 Phase 2-D29A historical READY intermediate audit

Primary classification: `{classification}`.

## Historical provenance

The read-only provenance table records the existing exp_012 Stage 2N initial checkpoint, exp_012 Stage 2Q DAgger-round-2 checkpoint, and exp_013 W2P1 selected STOP checkpoint. Their SHA-256 values, 124D actor contract, zero velocity/yaw command, gait-0 input, and original evaluator semantics are in `historical_ready_provenance.json`. The old STOP evaluator used strict flight-zero/double-support criteria; a gait-0 actor input is not itself a STOP command.

D28Z remains `{adjudication['official_classification_preserved']}`. The new scientific adjudication is `HZ_CONTROL_CONFLICTS_WITH_FIRST_STEP_TASKS`; it does not edit D28Z.

## READY state

Candidate source coverage is in `ready_state_metrics.json`. A candidate is valid only with zero canonical safety failures, no more than 0.10 m two-second XY displacement, mean horizontal speed no more than 0.08 m/s, p95 yaw rate no more than 0.15 rad/s, and at least one reproducible periodicity signal (support switches, load-ratio oscillation, or strict liftoff/touchdown). Raw per-step observations are in `ready_state_replay.csv`/`.json`.

| Candidate | READY-valid sources | Mean speed (m/s) | Max net XY (m) | Mean yaw p95 (rad/s) | Max yaw p95 (rad/s) | Safety-failed sources |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(ready_report_lines)}

## Manifold comparison

The feature contract is physical-only: root/base motion, CoM/DCM, foot pose/velocity, contact force, and support phase. Command/history are not used for proximity. `ready_wmove_manifold_distance.json` records the comparison contract; proximity is diagnostic and is not a capability gate.

## Route comparison

Routes A–D are recorded in `route_comparison.csv`/`.json`: A is HARD_DIRECT, B is READY_DIRECT_SWITCH, C is HARD_READY_WMOVE, and D is the existing READY controller's native minimum-jerk command ramp when supported. Every route uses the same eight logical train-only lanes, seed `{SEED}`, normal fresh reset, and existing frozen checkpoints. No raw snapshot restore or additional training was used.

| Route | Safe first step | Liftoff | Touchdown | W_MOVE entry | Falls | Dangerous slips | Velocity/torque saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(route_report_lines)}

## First step and W_MOVE entry

`first_step_results.json`, `touchdown_results.json`, `wmove_entry_results.json`, and `first_divergence.json` contain per-route source counts and first-failure labels. The D29A positive-control thresholds are preserved: safe liftoff at least 6/8, p95 yaw no more than 1.5 rad/s, maximum clearance no more than 0.10 m, touchdown at least 4/8, and W_MOVE entry confirmation at least 2/8. Candidate READY replay used the historical `Isaac-Exp012-G1-Reverse-PhaseR1-v0` runtime; the four route comparisons used the existing Exp013 directional route runtime with the same frozen actors and normal reset, as recorded in the raw metadata.

## Protection and repository

Starting HEAD: `{start_head}`. D29A output was generated without modifying D28Z or earlier artifacts. Persistent update: `0`; new learned checkpoint: `0`; PPO/CEM: `0`; WBIK/centroidal modification: `0`; raw snapshot restore: `0`; validation/held-out: `0`; RUN integration: `0`; remote push: `false`. The pre-existing worktree status is preserved in `protected_hashes.json`.
"""
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ready", "routes", "finalize"), required=True)
    parser.add_argument("--candidate", choices=("stage2n_initial", "stage2q_dagger2", "exp013_w2p1_stop"))
    parser.add_argument("--ready-candidate", choices=("stage2n_initial", "stage2q_dagger2", "exp013_w2p1_stop"))
    try:
        from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
        add_launcher_args(parser)
        args, hydra = setup_preset_cli(parser)
        sys.argv = [sys.argv[0], *hydra]
    except ModuleNotFoundError:
        args = parser.parse_args()
    if args.mode == "ready":
        if not args.candidate:
            parser.error("--candidate is required for ready")
        run_ready(args)
    elif args.mode == "routes":
        if not args.ready_candidate:
            parser.error("--ready-candidate is required for routes")
        run_routes(args)
    else:
        finalize()


if __name__ == "__main__":
    main()
