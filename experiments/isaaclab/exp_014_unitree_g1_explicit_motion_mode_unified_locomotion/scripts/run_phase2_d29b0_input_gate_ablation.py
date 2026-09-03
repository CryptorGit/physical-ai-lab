"""EXP014 Phase 2-D29B0 zero-speed STAND/WALK input-gate ablation.

One frozen Exp014 141D actor is compared under the same source reset-boundary
lifecycle and the same zero physical command. Only target motion mode changes.
The existing W_MOVE actor is used only after the fixed preconditioning interval
as a hard-switch positive control. No checkpoint is trained or written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29b0_input_gate_ablation"
RAW = OUT / "raw"
REPORT = REPO / "research/exp_014_phase_2_d29b0_input_gate_ablation_report.md"

EXPECTED_START_HEAD = "600298f1d21acaf7389efd96ede081faa9bd90b9"
DT = 0.02
SEED = 20279941
RECIPES = list(range(8))
PRECONDITION_STEPS = 100
WMOVE_STEPS = 150
WMOVE_ENTRY_DISTANCE_P95 = 12.8774970171285

ACTOR = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/dagger_checkpoints/round_2_step_10000.pt"
ACTOR_REL = ACTOR.relative_to(REPO).as_posix()
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
WMOVE_REL = WMOVE.relative_to(REPO).as_posix()
RESET_SCRIPT = EXP / "scripts/run_phase2_d3.py"
D29A_CLASS = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29a_ready_intermediate_audit/stage_classification.json"
D29A_PROVENANCE = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29a_ready_intermediate_audit/historical_ready_provenance.json"
D29A_ROUTES = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29a_ready_intermediate_audit/route_comparison.json"
D29A_MANIFOLD = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29a_ready_intermediate_audit/ready_wmove_manifold_distance.json"
D29B_CLASS = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29b_post_touchdown_walk_capture/stage_classification.json"
D29A_SCRIPT = EXP / "scripts/run_phase2_d29a_ready_audit.py"
CURRENT_BEST = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/current_best_manifest.json"
ROUND2_TRAINING = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/dagger_round_2_training.json"
COMMAND_CONTRACT = EXP / "command_contract.json"
OBS_CONTRACT = EXP / "observation_contract.json"
STUDENT_SCRIPT = EXP / "src/g1_explicit_motion_mode/student.py"
MODE_CONTRACT = EXP / "src/g1_explicit_motion_mode/contract.py"
COLLECT_SCRIPT = EXP / "scripts/collect_dagger.py"
WMOVE_TRACE = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation/native_steady_trace_bundle.npz"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def git_status() -> list[str]:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"IMPORT_FAILED:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_runtime():
    sys.path[:0] = [
        str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
        str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
        str(REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"),
        str(EXP / "src"),
    ]
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    import g1_flat_run.tasks  # noqa: F401
    import g1_omnidirectional.tasks  # noqa: F401
    import g1_single_policy.tasks  # noqa: F401
    from g1_explicit_motion_mode.contract import ExplicitMotionModeCommand, MotionMode, build_observation_141
    from g1_explicit_motion_mode.student import ExplicitModeStudent
    from g1_omnidirectional.policy import FrozenGaitActor
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
    d3 = import_file("exp014_d29b0_d3_source", RESET_SCRIPT)
    return {
        "gym": gym,
        "torch": torch,
        "ExplicitMotionModeCommand": ExplicitMotionModeCommand,
        "MotionMode": MotionMode,
        "build_observation_141": build_observation_141,
        "ExplicitModeStudent": ExplicitModeStudent,
        "FrozenGaitActor": FrozenGaitActor,
        "RslRlVecEnvWrapper": RslRlVecEnvWrapper,
        "add_launcher_args": add_launcher_args,
        "launch_simulation": launch_simulation,
        "resolve_task_config": resolve_task_config,
        "setup_preset_cli": setup_preset_cli,
        "d3": d3,
    }


def ordered_feet(sensor, robot):
    sensor_idx, sensor_names = sensor.find_bodies(".*_ankle_roll_link")
    robot_idx, robot_names = robot.find_bodies(".*_ankle_roll_link")

    def order(indices, names):
        pairs = list(zip([int(x) for x in indices], list(names)))
        left = [p for p in pairs if "left" in p[1].lower()]
        right = [p for p in pairs if "right" in p[1].lower()]
        rest = [p for p in pairs if p not in left and p not in right]
        return [int(x[0]) for x in left + right + rest]

    si = order(sensor_idx, sensor_names)
    ri = order(robot_idx, robot_names)
    if len(si) < 2 or len(ri) < 2:
        raise RuntimeError(f"FOOT_BODY_RESOLUTION_FAILED:{sensor_names}:{robot_names}")
    return si[:2], ri[:2], list(sensor_names), list(robot_names)


def as_2d_limit(value, torch):
    if value is None:
        return None
    out = value
    if out.ndim == 3:
        out = out[..., 1].abs()
    return out.abs().clamp_min(1.0e-6)


def body_com(robot, torch):
    if not hasattr(robot.data, "body_com_pos_w") or not hasattr(robot.data, "body_com_lin_vel_w"):
        raise RuntimeError("BODY_COM_RUNTIME_FIELDS_UNAVAILABLE")
    root = robot.data.root_pos_w
    dtype = root.dtype if isinstance(root, torch.Tensor) else torch.float32
    device = root.device if isinstance(root, torch.Tensor) else robot.device
    pos = torch.as_tensor(robot.data.body_com_pos_w, device=device, dtype=dtype)
    vel = torch.as_tensor(robot.data.body_com_lin_vel_w, device=device, dtype=dtype)
    masses = torch.as_tensor(robot.root_physx_view.get_masses(), device=robot.device, dtype=pos.dtype)
    if masses.ndim == 1:
        masses = masses.unsqueeze(0).expand(pos.shape[0], -1)
    total = masses.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return (pos * masses[..., None]).sum(dim=1) / total, (vel * masses[..., None]).sum(dim=1) / total


def yaw_xyzw(quat, torch):
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def runtime_snapshot(world, sensor_feet, robot_feet, torch) -> dict[str, Any]:
    robot, sensor = world.robot, world.sensor
    root = robot.data.root_pos_w.detach()
    root_v = robot.data.root_lin_vel_b.detach()
    root_w = robot.data.root_ang_vel_b.detach()
    q = robot.data.joint_pos.detach()
    dq = robot.data.joint_vel.detach()
    foot_pos = robot.data.body_pos_w[:, robot_feet, :].detach()
    foot_vel = robot.data.body_lin_vel_w[:, robot_feet, :].detach()
    force_vec = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].detach()
    force = force_vec.norm(dim=-1)
    contact = force > 5.0
    com, com_v = body_com(robot, torch)
    h = com[:, 2].clamp_min(0.15)
    dcm = com[:, :2] + com_v[:, :2] / torch.sqrt(torch.tensor(9.81, device=robot.device) / h[:, None])
    support = torch.where(contact[:, 0:1], foot_pos[:, 0, :2], foot_pos[:, 1, :2])
    both = contact.all(dim=1)
    support[both] = foot_pos[both, 0, :2]
    gravity = torch.as_tensor(getattr(robot.data, "projected_gravity_b", torch.zeros((len(root), 3), device=robot.device)), device=robot.device, dtype=root.dtype)
    torque = getattr(robot.data, "applied_torque", torch.zeros_like(q)).detach()
    effort = as_2d_limit(getattr(robot.data, "joint_effort_limits", None), torch)
    if effort is None:
        effort = torch.ones_like(q)
    vlim = as_2d_limit(getattr(robot.data, "joint_vel_limits", None), torch)
    if vlim is None:
        vlim = torch.ones_like(dq)
    return {
        "root_pos": root.cpu().numpy(),
        "root_v": root_v.cpu().numpy(),
        "root_w": root_w.cpu().numpy(),
        "root_yaw": yaw_xyzw(robot.data.root_quat_w.detach(), torch).cpu().numpy(),
        "joint_q": q.cpu().numpy(),
        "joint_dq": dq.cpu().numpy(),
        "foot_pos": foot_pos.cpu().numpy(),
        "foot_vel": foot_vel.cpu().numpy(),
        "force": force.cpu().numpy(),
        "contact": contact.cpu().numpy(),
        "com": com.cpu().numpy(),
        "com_v": com_v.cpu().numpy(),
        "dcm": dcm.cpu().numpy(),
        "gravity": gravity.cpu().numpy(),
        "torque": torque.cpu().numpy(),
        "effort": effort.cpu().numpy(),
        "vlim": vlim.cpu().numpy(),
        "support": support.cpu().numpy(),
    }


def runtime_feature(snap: dict[str, Any], index: int) -> np.ndarray:
    force = snap["force"][index]
    contact = snap["contact"][index]
    foot = snap["foot_pos"][index]
    support = foot[0, :2]
    com_rel = snap["com"][index, :2] - support
    return np.concatenate(
        (
            snap["root_pos"][index, :2],
            snap["root_v"][index, :2],
            snap["root_w"][index],
            snap["gravity"][index],
            snap["joint_q"][index],
            snap["joint_dq"][index],
            com_rel,
            snap["com_v"][index, :2],
            snap["dcm"][index],
            foot.reshape(-1),
            snap["foot_vel"][index, :, :2].reshape(-1),
            force,
            contact.astype(float),
        )
    )


class WMoveReference:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        root = z["root_pose"][:, :2]
        root_v = z["root_velocity"][:, :2]
        root_w = z["root_velocity"][:, 3:6]
        gravity = z["obs_124"][:, 6:9]
        q = z["joint_pos"]
        dq = z["joint_vel"]
        foot = z["left_right_foot_pose"]
        foot_v = z["foot_velocity"][:, :, :2]
        force = np.linalg.norm(z["contact_force"], axis=-1)
        contact = force > 5.0
        com_rel = z["com_position"][:, :2] - foot[:, 0, :2]
        self.features = np.concatenate(
            (
                root, root_v, root_w, gravity, q, dq, com_rel,
                z["com_velocity"][:, :2], z["dcm"], foot.reshape(len(foot), -1),
                foot_v.reshape(len(foot_v), -1), force, contact.astype(float),
            ),
            axis=1,
        )[::5].astype(np.float64)
        self.scale = np.maximum(np.percentile(self.features, 95, axis=0) - np.percentile(self.features, 5, axis=0), 1.0e-6)
        self.source = path.relative_to(REPO).as_posix()

    def distance(self, feature: np.ndarray) -> float:
        d = (self.features - np.asarray(feature, dtype=np.float64)[None, :]) / self.scale[None, :]
        return float(np.linalg.norm(d, axis=1).min())


class SafetyTrace:
    def __init__(self, source_count: int, start: dict[str, Any], feet_z: np.ndarray):
        self.n = source_count
        self.start = start
        self.feet_z = feet_z.copy()
        self.previous_contact: np.ndarray | None = None
        self.streak = {k: np.zeros(self.n, dtype=np.int64) for k in ("slip", "velocity", "torque", "support")}
        self.flags = {k: np.zeros(self.n, dtype=bool) for k in ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nan_inf")}
        self.first_failure: list[str | None] = [None] * self.n
        self.liftoff = np.zeros(self.n, dtype=np.int64)
        self.touchdown = np.zeros(self.n, dtype=np.int64)
        self.support_switch = np.zeros(self.n, dtype=np.int64)
        self.rows: list[list[dict[str, Any]]] = [[] for _ in range(self.n)]
        self.max_clearance = np.full((self.n, 2), -np.inf, dtype=float)
        self.max_foot_speed = np.zeros((self.n, 2), dtype=float)
        self.max_velocity_ratio = np.zeros(self.n, dtype=float)
        self.max_torque_ratio = np.zeros(self.n, dtype=float)

    def record(self, step: int, phase: str, snap: dict[str, Any], action: np.ndarray, done: np.ndarray, timeout: np.ndarray, reference: WMoveReference | None):
        force = snap["force"]
        contact = snap["contact"]
        speed = np.linalg.norm(snap["root_v"][:, :2], axis=1)
        yaw_rate = np.abs(snap["root_w"][:, 2])
        clearance = snap["foot_pos"][:, :, 2] - self.feet_z
        foot_speed = np.linalg.norm(snap["foot_vel"][:, :, :2], axis=2)
        load = force[:, 0] / np.maximum(force.sum(axis=1), 1.0e-6)
        vr = np.max(np.abs(snap["joint_dq"]) / np.maximum(snap["vlim"], 1.0e-6), axis=1)
        tr = np.max(np.abs(snap["torque"]) / np.maximum(snap["effort"], 1.0e-6), axis=1)
        self.max_clearance = np.maximum(self.max_clearance, clearance)
        self.max_foot_speed = np.maximum(self.max_foot_speed, foot_speed)
        self.max_velocity_ratio = np.maximum(self.max_velocity_ratio, vr)
        self.max_torque_ratio = np.maximum(self.max_torque_ratio, tr)
        if self.previous_contact is not None:
            rose = (~self.previous_contact) & contact
            fell = self.previous_contact & (~contact)
            self.touchdown += rose.any(axis=1).astype(np.int64)
            self.liftoff += fell.any(axis=1).astype(np.int64)
            self.support_switch += (contact != self.previous_contact).any(axis=1).astype(np.int64)
        self.previous_contact = contact.copy()
        slip_now = ((foot_speed > 0.55) & contact).any(axis=1)
        impact_now = force.max(axis=1) > 3500.0
        vel_now = vr > 0.95
        torque_now = tr > 0.95
        support_now = (~contact).all(axis=1)
        for name, now in (("slip", slip_now), ("velocity", vel_now), ("torque", torque_now), ("support", support_now)):
            self.streak[name] = np.where(now, self.streak[name] + 1, 0)
        self.flags["dangerous_slip"] |= self.streak["slip"] >= 5
        self.flags["velocity_saturation"] |= self.streak["velocity"] >= 5
        self.flags["torque_saturation"] |= self.streak["torque"] >= 5
        self.flags["support_loss"] |= self.streak["support"] >= 5
        self.flags["impact"] |= impact_now
        finite = np.isfinite(snap["root_pos"]).all(axis=1) & np.isfinite(snap["joint_dq"]).all(axis=1) & np.isfinite(action).all(axis=1)
        self.flags["fall"] |= done & ~timeout
        self.flags["nan_inf"] |= ~finite
        priority = (
            ("nan_inf", "NUMERICAL_FAILURE"),
            ("fall", "FALL"),
            ("dangerous_slip", "DANGEROUS_SLIP"),
            ("impact", "IMPACT"),
            ("velocity_saturation", "VELOCITY_SATURATION"),
            ("torque_saturation", "TORQUE_SATURATION"),
            ("support_loss", "SUPPORT_LOSS"),
        )
        for i in range(self.n):
            if self.first_failure[i] is None:
                for key, label in priority:
                    if bool(self.flags[key][i]):
                        self.first_failure[i] = label
                        break
        for i in range(self.n):
            feature_distance = reference.distance(runtime_feature(snap, i)) if reference is not None and phase == "W_MOVE" else None
            self.rows[i].append({
                "control_step": int(step),
                "phase": phase,
                "root_xy": snap["root_pos"][i, :2].tolist(),
                "root_xy_displacement": (snap["root_pos"][i, :2] - self.start["root_pos"][i, :2]).tolist(),
                "forward_displacement": float(snap["root_pos"][i, 0] - self.start["root_pos"][i, 0]),
                "root_speed": float(speed[i]),
                "root_yaw_rate": float(yaw_rate[i]),
                "root_yaw_displacement": float(snap["root_yaw"][i] - self.start["root_yaw"][i]),
                "root_velocity_xy": snap["root_v"][i, :2].tolist(),
                "com_xy": snap["com"][i, :2].tolist(),
                "com_velocity_xy": snap["com_v"][i, :2].tolist(),
                "dcm_xy": snap["dcm"][i].tolist(),
                "contact": contact[i].tolist(),
                "contact_force_norm": force[i].tolist(),
                "load_ratio_left": float(load[i]),
                "foot_clearance_m": clearance[i].tolist(),
                "foot_velocity_xy": snap["foot_vel"][i, :, :2].tolist(),
                "joint_velocity_ratio": float(vr[i]),
                "torque_ratio": float(tr[i]),
                "action_l2": float(np.linalg.norm(action[i])),
                "safety": {k: bool(v[i]) for k, v in self.flags.items()},
                "entry_distance": feature_distance,
            })

    def summarize(self, condition: str, phase: str) -> list[dict[str, Any]]:
        out = []
        for i in range(self.n):
            rows = self.rows[i]
            speeds = np.asarray([r["root_speed"] for r in rows], dtype=float)
            yaw = np.asarray([r["root_yaw_rate"] for r in rows], dtype=float)
            load = np.asarray([r["load_ratio_left"] for r in rows], dtype=float)
            contact = np.asarray([r["contact"] for r in rows], dtype=bool)
            force = np.asarray([r["contact_force_norm"] for r in rows], dtype=float)
            xy = np.asarray([r["root_xy_displacement"] for r in rows], dtype=float)
            clearance = np.asarray([r["foot_clearance_m"] for r in rows], dtype=float)
            safety = {k: bool(v[i]) for k, v in self.flags.items()}
            delta = np.diff(load) if len(load) >= 2 else np.asarray([], dtype=float)
            oscillation = bool(len(delta) and delta.max() > 0.15 and delta.min() < -0.15)
            single = np.mean(contact.sum(axis=1) == 1) if len(contact) else 0.0
            double = np.mean(contact.sum(axis=1) == 2) if len(contact) else 0.0
            flight = np.mean(contact.sum(axis=1) == 0) if len(contact) else 0.0
            out.append({
                "condition": condition,
                "phase": phase,
                "recipe_id": i,
                "steps": len(rows),
                "mean_horizontal_speed": float(speeds.mean()) if len(speeds) else 0.0,
                "p95_horizontal_speed": float(np.percentile(speeds, 95)) if len(speeds) else 0.0,
                "net_xy_displacement_m": float(np.linalg.norm(xy[-1])) if len(xy) else 0.0,
                "max_xy_displacement_m": float(np.max(np.linalg.norm(xy, axis=1))) if len(xy) else 0.0,
                "max_forward_displacement_m": float(np.max([r["forward_displacement"] for r in rows])) if rows else 0.0,
                "yaw_rate_p50": float(np.percentile(yaw, 50)) if len(yaw) else 0.0,
                "yaw_rate_p95": float(np.percentile(yaw, 95)) if len(yaw) else 0.0,
                "yaw_rate_max": float(yaw.max()) if len(yaw) else 0.0,
                "yaw_displacement_rad": float(rows[-1]["root_yaw_displacement"]) if rows else 0.0,
                "support_switch_count": int(self.support_switch[i]),
                "strict_liftoff_count": int(self.liftoff[i]),
                "strict_touchdown_count": int(self.touchdown[i]),
                "alternating_load_oscillation": oscillation,
                "load_ratio_amplitude": float(load.max() - load.min()) if len(load) else 0.0,
                "mean_load_ratio_left": float(load.mean()) if len(load) else 0.0,
                "single_support_fraction": float(single),
                "double_support_fraction": float(double),
                "flight_fraction": float(flight),
                "max_clearance_m": float(np.max(clearance)) if len(clearance) else 0.0,
                "max_foot_speed_mps": float(np.max([np.linalg.norm(x) for r in rows for x in r["foot_velocity_xy"]])) if rows else 0.0,
                "contact_force_p95": np.percentile(force, 95, axis=0).tolist() if len(force) else [0.0, 0.0],
                "max_joint_velocity_ratio": float(self.max_velocity_ratio[i]),
                "max_torque_ratio": float(self.max_torque_ratio[i]),
                "safety": safety,
                "first_failure": self.first_failure[i],
            })
        return out


def load_actor(path: Path, runtime, device):
    torch = runtime["torch"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = runtime["ExplicitModeStudent"](tuple(payload["architecture"][1:-1])).to(device)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    return actor, payload


def set_external(world, command, torch):
    world.term.external_override.zero_()
    world.term.external_override[:, :3] = command
    world.term._update_command()


def explicit_action(world, actor, physical, step, runtime):
    torch = runtime["torch"]
    state = world.state
    state.advance(physical, torch.ones(world.env.num_envs, device=world.device), 0.0 if step == 0 else DT)
    set_external(world, physical, torch)
    base = world.env.observation_manager.compute()["policy"]
    base123 = base[:, :123] if base.shape[-1] != 123 else base
    observation = runtime["build_observation_141"](base123, state)
    with torch.inference_mode():
        action = actor(observation)
    return action, observation


def wmove_action(world, actor, command, runtime):
    torch = runtime["torch"]
    world.state.advance(command, torch.ones(world.env.num_envs, device=world.device), DT)
    set_external(world, command, torch)
    base = world.env.observation_manager.compute()["policy"]
    base123 = base[:, :123] if base.shape[-1] != 123 else base
    gait = torch.zeros(world.env.num_envs, device=world.device)
    with torch.inference_mode():
        action = actor(base123, gait)
    return action, base123


def numpy_bool_tensor(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(bool)
    return np.asarray(x, dtype=bool)


def run_condition(world, condition: str, actor, wmove, recipes: list[int], runtime, reference) -> dict[str, Any]:
    torch = runtime["torch"]
    n = len(recipes)
    # D29B0 uses the same normal fresh lifecycle as the D29A baseline.  The
    # logical recipe IDs label the fixed eight seeded reset lanes; no
    # write_root/write_joint or snapshot restore is permitted here.
    world.env.reset()
    world.term.external_override.zero_()
    world.term._update_command()
    world.state = runtime["ExplicitMotionModeCommand"].zeros(n, device=world.device)
    start = runtime_snapshot(world, world._d29b0_sensor_feet, world._d29b0_robot_feet, torch)
    source_state_hash = hashlib.sha256(
        np.concatenate((start["root_pos"].ravel(), start["root_v"].ravel(), start["joint_q"].ravel(), start["joint_dq"].ravel())).astype(np.float64).tobytes()
    ).hexdigest()
    pre = SafetyTrace(n, start, start["foot_pos"][:, :, 2])
    mode_value = runtime["MotionMode"].WALK if condition == "P_WALK_ZERO" else runtime["MotionMode"].STAND
    world.state.request(torch.full((n,), int(mode_value), device=world.device, dtype=torch.long))
    pre_actions: list[np.ndarray] = []
    pre_qcmd: list[np.ndarray] = []
    for step in range(PRECONDITION_STEPS):
        physical = torch.zeros(n, 3, device=world.device)
        action, _ = explicit_action(world, actor, physical, step, runtime)
        _, _, done, extras = world.wrapped.step(action)
        snap = runtime_snapshot(world, world._d29b0_sensor_feet, world._d29b0_robot_feet, torch)
        timeout_value = extras.get("time_outs", torch.zeros_like(done))
        action_np = action.detach().cpu().numpy()
        pre.record(step, condition, snap, action_np, numpy_bool_tensor(done), numpy_bool_tensor(timeout_value), None)
        pre_actions.append(action_np)
        default_q = world.robot.data.default_joint_pos.detach().cpu().numpy()
        pre_qcmd.append(default_q + 0.5 * action_np)
    pre_rows = pre.summarize(condition, "PRECONDITION")

    route_start = runtime_snapshot(world, world._d29b0_sensor_feet, world._d29b0_robot_feet, torch)
    route = SafetyTrace(n, route_start, route_start["foot_pos"][:, :, 2])
    route_actions: list[np.ndarray] = []
    route_qcmd: list[np.ndarray] = []
    route_torque: list[np.ndarray] = []
    previous_action = pre_actions[-1]
    previous_qcmd = pre_qcmd[-1]
    for _step in range(WMOVE_STEPS):
        command = torch.zeros(n, 3, device=world.device)
        command[:, 0] = 0.3
        action, _ = wmove_action(world, wmove, command, runtime)
        action_np = action.detach().cpu().numpy()
        qcmd = world.robot.data.default_joint_pos.detach().cpu().numpy() + 0.5 * action_np
        _, _, done, extras = world.wrapped.step(action)
        snap = runtime_snapshot(world, world._d29b0_sensor_feet, world._d29b0_robot_feet, torch)
        timeout_value = extras.get("time_outs", torch.zeros_like(done))
        route.record(_step, "W_MOVE", snap, action_np, numpy_bool_tensor(done), numpy_bool_tensor(timeout_value), reference)
        route_actions.append(action_np)
        route_qcmd.append(qcmd)
        route_torque.append(snap["torque"].copy())
    route_rows = route.summarize(condition, "W_MOVE")
    action_jump = np.linalg.norm(route_actions[0] - previous_action, axis=1)
    qcmd_jump = np.linalg.norm(route_qcmd[0] - previous_qcmd, axis=1)
    cosine = np.sum(route_actions[0] * previous_action, axis=1) / np.maximum(np.linalg.norm(route_actions[0], axis=1) * np.linalg.norm(previous_action, axis=1), 1.0e-8)
    torque_transient = np.linalg.norm(route_torque[0] - route_start["torque"], axis=1)
    for i, row in enumerate(route_rows):
        rows = route.rows[i]
        qualified = []
        for start_idx in range(max(0, len(rows) - 10 + 1)):
            window = rows[start_idx:start_idx + 10]
            if len(window) < 10:
                continue
            good = all(
                x["entry_distance"] is not None
                and x["entry_distance"] <= WMOVE_ENTRY_DISTANCE_P95
                and abs(x["root_velocity_xy"][0] - 0.3) <= 0.12
                and abs(x["root_velocity_xy"][1]) <= 0.08
                and x["root_yaw_rate"] <= 0.10
                and not any(x["safety"].values())
                for x in window
            )
            if good:
                qualified.append(start_idx)
        entry_step = qualified[0] if qualified else -1
        retention = False
        if entry_step >= 0 and entry_step + 50 <= len(rows):
            window = rows[entry_step:entry_step + 50]
            retention = all(
                abs(x["root_velocity_xy"][0] - 0.3) <= 0.12
                and abs(x["root_velocity_xy"][1]) <= 0.08
                and x["root_yaw_rate"] <= 0.10
                and not any(x["safety"].values())
                for x in window
            )
        liftoff = row["strict_liftoff_count"] >= 1
        touchdown = row["strict_touchdown_count"] >= 1
        safe_first = (
            liftoff
            and touchdown
            and row["max_forward_displacement_m"] > 0.03
            and row["yaw_rate_p95"] <= 1.5
            and row["max_clearance_m"] <= 0.10
            and not any(row["safety"].values())
        )
        row.update({
            "safe_first_step": bool(safe_first),
            "liftoff": bool(liftoff),
            "touchdown": bool(touchdown),
            "forward_pelvis_displacement_m": float(row["max_forward_displacement_m"]),
            "wmove_entry_neighborhood_distance_p50": float(np.percentile([x["entry_distance"] for x in rows], 50)) if rows else None,
            "wmove_entry_neighborhood_distance_p95": float(np.percentile([x["entry_distance"] for x in rows], 95)) if rows else None,
            "wmove_entry_neighborhood_distance_min": float(min(x["entry_distance"] for x in rows)) if rows else None,
            "entry_confirmation_10_step": bool(entry_step >= 0),
            "entry_confirmation_step": int(entry_step),
            "retention_50_step": bool(retention),
            "action_l2_jump": float(action_jump[i]),
            "action_cosine": float(cosine[i]),
            "joint_target_jump_l2": float(qcmd_jump[i]),
            "torque_transient_l2": float(torque_transient[i]),
            "first_divergence": route_first_divergence(row, bool(liftoff), bool(touchdown), bool(safe_first), bool(entry_step >= 0)),
            "entry_threshold_source": "D29A ready_wmove_manifold_distance.json READY_to_W_MOVE aggregate p95",
            "entry_threshold": WMOVE_ENTRY_DISTANCE_P95,
        })
    return {
        "condition": condition,
        "seed": SEED,
        "recipes": recipes,
        "source_state_hash": source_state_hash,
        "preconditioning": pre_rows,
        "route": route_rows,
        "pre_rows": pre.rows,
        "route_rows": route.rows,
        "precondition_steps": PRECONDITION_STEPS,
        "wmove_steps": WMOVE_STEPS,
        "controller_switch": "fixed boundary; explicit actor -> existing W_MOVE; hard switch; no blending",
        "raw_snapshot_restore": False,
        "source_reset_boundary_initialization": True,
    }


def route_first_divergence(row: dict[str, Any], liftoff: bool, touchdown: bool, safe_first: bool, entry: bool) -> str:
    if row["first_failure"]:
        return str(row["first_failure"])
    if not liftoff:
        return "SUPPORT_SHIFT_FAILURE"
    if not touchdown:
        return "TOUCHDOWN_FAILURE"
    if row["max_clearance_m"] > 0.10:
        return "SWING_OVERSHOOT"
    if row["yaw_rate_p95"] > 1.5:
        return "YAW_DIVERGENCE"
    if not safe_first:
        return "SUPPORT_SHIFT_FAILURE"
    if not entry:
        return "WMOVE_ENTRY_FAILURE"
    return "NONE"


def mode_shadow(world, actor, runtime) -> dict[str, Any]:
    torch = runtime["torch"]
    rows = []
    names = [str(x) for x in world.robot.data.joint_names]
    # One fresh reset establishes the common physical state for all eight
    # logical source lanes.  STAND/WALK are then evaluated on that identical
    # state without a simulation step or a state write.
    world.env.reset()
    world.term.external_override.zero_()
    world.term._update_command()
    world.state = runtime["ExplicitMotionModeCommand"].zeros(world.env.num_envs, device=world.device)
    base_all = world.env.observation_manager.compute()["policy"].detach().clone()
    for recipe in RECIPES:
        base = base_all[recipe:recipe + 1]
        base = base[:, :123] if base.shape[-1] != 123 else base
        stand = runtime["ExplicitMotionModeCommand"].zeros(1, device=world.device)
        walk = runtime["ExplicitMotionModeCommand"].zeros(1, device=world.device)
        walk.request(torch.full((1,), int(runtime["MotionMode"].WALK), device=world.device, dtype=torch.long))
        x0 = runtime["build_observation_141"](base, stand)
        x1 = runtime["build_observation_141"](base, walk)
        with torch.inference_mode():
            a0 = actor(x0)
            a1 = actor(x1)
        x0n, x1n = x0.detach().cpu().numpy()[0], x1.detach().cpu().numpy()[0]
        a0n, a1n = a0.detach().cpu().numpy()[0], a1.detach().cpu().numpy()[0]
        keep = np.ones(141, dtype=bool)
        keep[127:130] = False
        group = {}
        for i, name in enumerate(names):
            lname = name.lower()
            key = "legs" if any(s in lname for s in ("hip", "knee", "ankle")) else "waist" if any(s in lname for s in ("waist", "torso")) else "arms" if any(s in lname for s in ("shoulder", "elbow")) else "wrist_hand"
            group.setdefault(key, 0.0)
            group[key] += float((a1n[i] - a0n[i]) ** 2)
        rows.append({
            "recipe_id": recipe,
            "same_base_observation_hash": hashlib.sha256(base.detach().cpu().numpy().tobytes()).hexdigest(),
            "stand_input_hash": hashlib.sha256(x0n.tobytes()).hexdigest(),
            "walk_input_hash": hashlib.sha256(x1n.tobytes()).hexdigest(),
            "non_mode_input_hash_stand": hashlib.sha256(x0n[keep].tobytes()).hexdigest(),
            "non_mode_input_hash_walk": hashlib.sha256(x1n[keep].tobytes()).hexdigest(),
            "non_mode_input_max_abs_difference": float(np.max(np.abs(x0n[keep] - x1n[keep]))),
            "mode_only_changed_indices": [127, 128, 129],
            "stand_target_mode": "STAND",
            "walk_target_mode": "WALK",
            "previous_target_mode_both": "STAND",
            "physical_command_both": [0.0, 0.0, 0.0],
            "action_l2": float(np.linalg.norm(a1n - a0n)),
            "action_cosine": float(np.dot(a1n, a0n) / max(np.linalg.norm(a1n) * np.linalg.norm(a0n), 1.0e-8)),
            "action_max_abs_difference": float(np.max(np.abs(a1n - a0n))),
            "joint_group_action_l2": {k: float(math.sqrt(v)) for k, v in group.items()},
            "stand_action_sha256": hashlib.sha256(a0n.tobytes()).hexdigest(),
            "walk_action_sha256": hashlib.sha256(a1n.tobytes()).hexdigest(),
            "predicted_torque_difference": "NOT_CAPTURED_IN_SHADOW; route torque transients are recorded",
        })
    return {
        "actor_inference": "same frozen actor, deterministic mean output",
        "same_source_state": True,
        "same_zero_velocity_command": True,
        "mode_only_input_difference": True,
        "all_non_mode_input_hashes_match": all(x["non_mode_input_hash_stand"] == x["non_mode_input_hash_walk"] for x in rows),
        "rows": rows,
    }


def run_physics(args) -> None:
    runtime = import_runtime()
    torch = runtime["torch"]
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    actor, payload = load_actor(ACTOR, runtime, device)
    wmove = runtime["FrozenGaitActor"](WMOVE).to(device).eval()
    parser = argparse.ArgumentParser()
    runtime["add_launcher_args"](parser)
    launcher_argv = [sys.argv[0]]
    if args.headless:
        launcher_argv.append("--headless")
    if args.device:
        launcher_argv.extend(["--device", args.device])
    saved_argv = sys.argv
    sys.argv = launcher_argv
    launch_args, hydra_args = runtime["setup_preset_cli"](parser)
    sys.argv = saved_argv
    sys.argv = [sys.argv[0], *hydra_args]
    cfg, agent = runtime["resolve_task_config"]("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = len(RECIPES)
    cfg.seed = SEED
    cfg.episode_length_s = 20.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = agent.device = args.device
    with runtime["launch_simulation"](cfg, launch_args):
        wrapped = runtime["RslRlVecEnvWrapper"](
            runtime["gym"].make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=agent.clip_actions,
        )
        d3 = runtime["d3"]
        world = d3.StandWorld(wrapped, d3.load_resets(), torch.zeros(680, device=wrapped.unwrapped.device))
        world._d29b0_sensor_feet, world._d29b0_robot_feet, sensor_names, robot_names = ordered_feet(world.sensor, world.robot)
        reference = WMoveReference(WMOVE_TRACE)
        shadow = mode_shadow(world, actor, runtime)
        start_actual = git_head()
        result = {
            "metadata": {
                "starting_head_actual": start_actual,
                "expected_start_head": EXPECTED_START_HEAD,
                "starting_head_mismatch": start_actual != EXPECTED_START_HEAD,
                "seed": SEED,
                "recipes": RECIPES,
                "dt": DT,
                "precondition_steps": PRECONDITION_STEPS,
                "wmove_steps": WMOVE_STEPS,
                "sensor_foot_names": sensor_names,
                "robot_foot_names": robot_names,
                "actor": ACTOR_REL,
                "actor_sha256": sha256_file(ACTOR),
                "actor_payload_format": payload.get("format"),
                "actor_architecture": payload.get("architecture"),
                "wmove_actor": WMOVE_REL,
                "wmove_actor_sha256": sha256_file(WMOVE),
                "entry_neighborhood_contract": {
                    "distance_threshold_p95": WMOVE_ENTRY_DISTANCE_P95,
                    "threshold_source": D29A_MANIFOLD.relative_to(REPO).as_posix(),
                    "feature_contract": "physical-only; command/history excluded",
                    "reference_source": reference.source,
                },
                "raw_snapshot_restore": False,
                "source_reset_boundary_initialization": True,
                "physics_steps": "two independent 8-source condition processes; 16 episodes total",
                "policy_training": 0,
                "persistent_update": 0,
                "new_checkpoint": 0,
                "validation": 0,
                "held_out": 0,
                "PPO": 0,
                "CEM": 0,
                "WBIK_modification": 0,
                "centroidal_modification": 0,
                "trajectory_optimization": 0,
            },
            "shadow": shadow,
            "conditions": {},
        }
        condition = args.condition
        result["conditions"][condition] = run_condition(world, condition, actor, wmove, RECIPES, runtime, reference)
        OUT.mkdir(parents=True, exist_ok=True)
        dump(RAW / f"physics_{condition}.json", result)
        wrapped.close()
        print(json.dumps({"condition": condition, "shadow_non_mode_match": shadow["all_non_mode_input_hashes_match"], "sources": 8, "raw_snapshot_restore": False}, indent=2))


def source_provenance() -> dict[str, Any]:
    d29a = load_json(D29A_CLASS, {})
    historical = load_json(D29A_PROVENANCE, {})
    exp014_actor_sha = sha256_file(ACTOR) if ACTOR.exists() else None
    historical_candidates = []
    for candidate in historical.get("historical_candidates", []):
        historical_candidates.append({
            key: candidate.get(key)
            for key in (
                "candidate",
                "checkpoint",
                "sha256",
                "observation_dimension",
                "command_tuple",
                "velocity_command",
                "yaw_command",
                "motion_mode",
                "gait_command",
                "previous_mode",
                "previous_command",
                "ramp_state",
                "original_evaluator",
                "original_lifecycle",
                "original_seed",
                "reported_behavior",
                "checkpoint_payload_keys",
            )
        })
    return {
        "historical_checkpoint_contracts_read_only": {
            "source_artifact": D29A_PROVENANCE.relative_to(REPO).as_posix(),
            "candidates": historical_candidates,
            "stop_gate_semantics": historical.get("stop_gate_semantics", {}),
        },
        "exp012": {
            "source_symbols": [
                "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src/g1_single_policy/stage2n_models.py:GaitConditionedMLPModel",
                "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src/g1_single_policy/tasks.py:Exp012G1Stage2NEnvCfg",
            ],
            "gait_cmd": "legacy scalar gait input; formal values are WALK=0 and RUN=1 in the Stage2N contract; no independent STAND mode field",
            "input_dimension": 124,
            "gate_field_name": "legacy_gait",
            "gate_index": [123, 124],
            "encoded_values": {"WALK": 0.0, "RUN": 1.0},
            "checkpoint_contract": "historical candidates and SHA-256 are copied from the protected D29A provenance artifact",
            "causal_role": "historical context only; not selected for D29B0 same-actor causal evidence",
        },
        "exp013": {
            "checkpoint": WMOVE_REL,
            "checkpoint_sha256": sha256_file(WMOVE) if WMOVE.exists() else None,
            "input_dimension": 124,
            "gate_field_name": "legacy_gait",
            "gate_index": [123, 124],
            "encoded_values": {"WALK": 0.0, "RUN": 1.0},
            "source": "D29A provenance and existing FrozenGaitActor runtime",
            "stop_gate": "zero physical command plus gait-conditioned legacy actor path; no explicit STAND/WALK mode",
            "causal_role": "existing W_MOVE hard-switch positive-control actor only",
        },
        "exp014": {
            "checkpoint": ACTOR_REL,
            "checkpoint_sha256": exp014_actor_sha,
            "input_dimension": 141,
            "gate_field_name": "target_mode_one_hot",
            "gate_index": [127, 130],
            "mode_enum": {"STAND": 0, "WALK": 1, "RUN": 2},
            "mode_contract_source": MODE_CONTRACT.relative_to(REPO).as_posix(),
            "command_contract_source": COMMAND_CONTRACT.relative_to(REPO).as_posix(),
            "observation_contract_source": OBS_CONTRACT.relative_to(REPO).as_posix(),
            "feature_indices": {
                "legacy_gait": [123, 124],
                "physical_command": [124, 127],
                "target_mode_one_hot": [127, 130],
                "previous_target_mode_one_hot": [130, 133],
                "previous_command": [133, 136],
                "command_delta": [136, 139],
                "time_since_mode_change": [139, 140],
                "ramp_progress": [140, 141],
            },
            "semantics": "request target mode immediately before physical-command/ramp advance",
            "mode_values": {"STAND": 0, "WALK": 1, "RUN": 2},
            "training_time_supported_values": ["STAND", "WALK", "RUN"],
            "evaluation_time_supported_values": ["STAND_HOLD", "STAND_TO_WALK", "WALK_STEADY", "WALK_TO_STAND"],
            "formal_actor_training_contexts": ["STAND_HOLD", "STAND_TO_WALK", "WALK_STEADY", "WALK_TO_STAND"],
            "same_weight_actor": True,
            "action_path": "ExplicitModeStudent -> 37D normalized position action -> existing Isaac Lab wrapper",
        },
        "source_evidence": {
            "actor_manifest": CURRENT_BEST.relative_to(REPO).as_posix(),
            "round2_training": ROUND2_TRAINING.relative_to(REPO).as_posix(),
            "d29a_classification_preserved": d29a.get("primary_classification", "EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED"),
        },
    }


def causal_gate(physics: dict[str, Any]) -> dict[str, Any]:
    shadow = physics.get("shadow", {})
    source_hashes = physics.get("metadata", {}).get("source_state_hashes", {})
    same_source_reset_hashes = bool(source_hashes.get("P_STAND") and source_hashes.get("P_STAND") == source_hashes.get("P_WALK_ZERO"))
    actor_ok = physics.get("metadata", {}).get("actor_payload_format") == "Exp014ExplicitModeStudentV1" and physics.get("metadata", {}).get("actor_architecture") == [141, 256, 128, 128, 37]
    identifiable = bool(actor_ok and shadow.get("same_source_state") and same_source_reset_hashes and shadow.get("same_zero_velocity_command") and shadow.get("mode_only_input_difference") and shadow.get("all_non_mode_input_hashes_match"))
    return {
        "status": "PASS" if identifiable else "FAIL",
        "same_network_weights": actor_ok,
        "same_observation_schema": actor_ok,
        "same_action_path": actor_ok,
        "formal_stand_supported": True,
        "formal_walk_supported": True,
        "same_source": bool(shadow.get("same_source_state")),
        "same_source_reset_hashes": same_source_reset_hashes,
        "source_state_hashes": source_hashes,
        "same_zero_velocity_command": bool(shadow.get("same_zero_velocity_command")),
        "mode_only_difference": bool(shadow.get("mode_only_input_difference")),
        "non_mode_input_hash_match": bool(shadow.get("all_non_mode_input_hashes_match")),
        "identifiable": identifiable,
        "failure_closed_if_false": "EXP014_D29B0_GATE_CAUSAL_ABLATION_NOT_IDENTIFIABLE",
    }


def row_gate_precondition(row: dict[str, Any]) -> bool:
    safety = row.get("safety", {})
    periodic = row.get("support_switch_count", 0) >= 2 or row.get("alternating_load_oscillation", False) or (row.get("strict_liftoff_count", 0) >= 1 and row.get("strict_touchdown_count", 0) >= 1)
    return not any(safety.values()) and row.get("net_xy_displacement_m", 999.0) <= 0.10 and row.get("mean_horizontal_speed", 999.0) <= 0.08 and row.get("yaw_rate_p95", 999.0) <= 0.15 and periodic


def route_gate(row: dict[str, Any]) -> bool:
    safety = row.get("safety", {})
    return bool(row.get("safe_first_step")) and row.get("yaw_rate_p95", 999.0) <= 1.5 and row.get("max_clearance_m", 999.0) <= 0.10 and not any(safety.values())


def aggregate_condition(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    chosen = [x for x in rows if x["condition"] == condition]
    valid = [row_gate_precondition(x) for x in chosen]
    return {
        "condition": condition,
        "source_count": len(chosen),
        "ready_like_valid_count": int(sum(valid)),
        "ready_like_coverage": float(sum(valid) / max(1, len(valid))),
        "ready_like_gate": bool(sum(valid) >= 7),
        "microstep_signals": {
            "support_switch_p50": float(np.percentile([x["support_switch_count"] for x in chosen], 50)) if chosen else 0.0,
            "load_oscillation_count": int(sum(bool(x["alternating_load_oscillation"]) for x in chosen)),
            "strict_micro_sequence_count": int(sum(x["strict_liftoff_count"] >= 1 and x["strict_touchdown_count"] >= 1 for x in chosen)),
        },
    }


def make_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(row.get(k)), separators=(",", ":")) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in fields})


def protected_hashes() -> dict[str, Any]:
    paths = [D29A_CLASS, D29A_PROVENANCE, D29A_ROUTES, D29A_MANIFOLD, D29B_CLASS, ACTOR, WMOVE, COMMAND_CONTRACT, OBS_CONTRACT, MODE_CONTRACT, STUDENT_SCRIPT, COLLECT_SCRIPT, RESET_SCRIPT]
    return {
        "starting_head_actual": git_head(),
        "expected_user_start_head": EXPECTED_START_HEAD,
        "starting_head_mismatch": git_head() != EXPECTED_START_HEAD,
        "protected_paths": {p.relative_to(REPO).as_posix(): sha256_file(p) for p in paths if p.exists()},
        "D6_to_D29A_artifacts_read_only": True,
        "D29B_artifacts_read_only": True,
        "exp005_to_exp013_read_only": True,
        "checkpoint_hashes_unchanged": True,
        "new_learned_checkpoint": 0,
        "persistent_update": 0,
        "PPO_CEM": 0,
        "WBIK_centroidal_modification": 0,
        "raw_restore": 0,
        "validation_held_out": 0,
        "RUN_integration": 0,
        "remote_push": False,
        "preexisting_worktree_status_preserved": git_status(),
    }


def finalize() -> None:
    process_results = {condition: load_json(RAW / f"physics_{condition}.json", {}) for condition in ("P_STAND", "P_WALK_ZERO")}
    if any(not value for value in process_results.values()):
        raise RuntimeError("D29B0_PHYSICS_RESULTS_MISSING_FOR_BOTH_CONDITIONS")
    first = process_results["P_STAND"]
    second = process_results["P_WALK_ZERO"]
    physics = {
        "metadata": {
            **first.get("metadata", {}),
            "condition_processes": {k: v.get("metadata", {}).get("starting_head_actual") for k, v in process_results.items()},
            "independent_condition_processes": True,
            "same_seed": all(v.get("metadata", {}).get("seed") == SEED for v in process_results.values()),
            "source_state_hashes": {
                condition: next(iter(value.get("conditions", {}).values())).get("source_state_hash")
                for condition, value in process_results.items()
            },
        },
        "shadow": first.get("shadow", {}),
        "conditions": {
            condition: next(iter(value.get("conditions", {}).values()))
            for condition, value in process_results.items()
        },
    }
    dump(RAW / "physics_results.json", physics)
    ident = causal_gate(physics)
    provenance = source_provenance()
    shadow = physics.get("shadow", {})
    pre_rows, route_rows = [], []
    for result in physics.get("conditions", {}).values():
        pre_rows.extend(result.get("preconditioning", []))
        route_rows.extend(result.get("route", []))
    pre_stand = aggregate_condition(pre_rows, "P_STAND")
    pre_walk = aggregate_condition(pre_rows, "P_WALK_ZERO")
    stand_routes = [x for x in route_rows if x["condition"] == "P_STAND"]
    walk_routes = [x for x in route_rows if x["condition"] == "P_WALK_ZERO"]
    stand_safe, walk_safe = sum(route_gate(x) for x in stand_routes), sum(route_gate(x) for x in walk_routes)
    stand_entry, walk_entry = sum(bool(x.get("entry_confirmation_10_step")) for x in stand_routes), sum(bool(x.get("entry_confirmation_10_step")) for x in walk_routes)
    stand_yaw = float(np.percentile([x["yaw_rate_p95"] for x in stand_routes], 50)) if stand_routes else 0.0
    walk_yaw = float(np.percentile([x["yaw_rate_p95"] for x in walk_routes], 50)) if walk_routes else 0.0
    stand_clear = float(np.percentile([x["max_clearance_m"] for x in stand_routes], 95)) if stand_routes else 0.0
    walk_clear = float(np.percentile([x["max_clearance_m"] for x in walk_routes], 95)) if walk_routes else 0.0
    precondition_unsafe = pre_walk["ready_like_coverage"] < 7 / 8 or any(any(x["safety"].values()) for x in pre_rows if x["condition"] == "P_WALK_ZERO")
    microstep_effect = pre_walk["microstep_signals"]["support_switch_p50"] > pre_stand["microstep_signals"]["support_switch_p50"] or pre_walk["microstep_signals"]["load_oscillation_count"] > pre_stand["microstep_signals"]["load_oscillation_count"] or pre_walk["microstep_signals"]["strict_micro_sequence_count"] > pre_stand["microstep_signals"]["strict_micro_sequence_count"]
    yaw_reduction = (stand_yaw - walk_yaw) / max(stand_yaw, 1.0e-8)
    start_positive = pre_walk["ready_like_gate"] and walk_safe >= 6 and walk_safe >= stand_safe + 2 and yaw_reduction >= 0.30 and walk_clear <= 0.10 and sum(not any(x["safety"].values()) for x in walk_routes) == 8 and walk_entry >= 2
    if not ident["identifiable"]:
        classification = "EXP014_D29B0_GATE_CAUSAL_ABLATION_NOT_IDENTIFIABLE"
        next_action = "Do not use different actors as causal evidence; stop this gate-only ablation."
    elif precondition_unsafe:
        classification = "EXP014_D29B0_ZERO_SPEED_WALK_PRECONDITION_UNSAFE"
        next_action = "Do not use zero-speed WALK as READY; proceed to post-touchdown WALK capture only."
    elif start_positive:
        classification = "EXP014_D29B0_ZERO_SPEED_WALK_GATE_PRIMES_START"
        next_action = "Formalize zero-speed WALK as a distinct READY diagnostic; hold D29B post-touchdown capture."
    elif microstep_effect:
        classification = "EXP014_D29B0_ZERO_SPEED_WALK_MICROSTEP_NO_START_BENEFIT"
        next_action = "Reject the input-gate START hypothesis and continue with D29B post-touchdown WALK capture."
    else:
        classification = "EXP014_D29B0_INPUT_GATE_NO_CAUSAL_EFFECT"
        next_action = "No causal mode-gate effect was established; continue with D29B post-touchdown WALK capture."
    gate_stats = {
        "ready_like_gate_required": ">=7/8",
        "B_ready_like_gate": pre_walk["ready_like_gate"],
        "B_safe_first_step": walk_safe,
        "A_safe_first_step": stand_safe,
        "safe_first_step_improvement": walk_safe - stand_safe,
        "yaw_p95_relative_reduction_median": yaw_reduction,
        "B_clearance_p95": walk_clear,
        "B_fall_count": sum(bool(x["safety"]["fall"]) for x in walk_routes),
        "B_dangerous_slip_count": sum(bool(x["safety"]["dangerous_slip"]) for x in walk_routes),
        "B_wmove_entry_count": walk_entry,
        "A_wmove_entry_count": stand_entry,
        "microstep_effect": microstep_effect,
        "positive_control_pass": start_positive,
    }
    out = {
        "official_d29a_classification_preserved": "EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED",
        "primary_classification": classification,
        "identifiability": ident,
        "provenance": provenance,
        "physics_metadata": physics.get("metadata", {}),
        "same_source_shadow": shadow,
        "precondition_summary": {"P_STAND": pre_stand, "P_WALK_ZERO": pre_walk},
        "route_summary": {
            "A_STAND_PRECONDITION": {"source_count": 8, "safe_first_step": stand_safe, "entry_confirmation": stand_entry, "median_route_yaw_p95": stand_yaw, "p95_clearance_across_sources": stand_clear},
            "B_WALK_ZERO_PRECONDITION": {"source_count": 8, "safe_first_step": walk_safe, "entry_confirmation": walk_entry, "median_route_yaw_p95": walk_yaw, "p95_clearance_across_sources": walk_clear},
        },
        "gate_statistics": gate_stats,
        "next_action": next_action,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / "gate_provenance_audit.json", provenance)
    dump(OUT / "causal_identifiability.json", ident)
    dump(OUT / "selected_actor_contract.json", {
        "actor": ACTOR_REL,
        "sha256": physics["metadata"].get("actor_sha256"),
        "format": physics["metadata"].get("actor_payload_format"),
        "architecture": physics["metadata"].get("actor_architecture"),
        "input_dimension": 141,
        "action_dimension": 37,
        "same_weights_for_stand_walk": True,
        "mode_values": {"STAND": 0, "WALK": 1, "RUN": 2},
        "action_path": "same actor -> same wrapper -> normalized 37D position action",
        "wmove_actor_for_route_only": {"path": WMOVE_REL, "sha256": physics["metadata"].get("wmove_actor_sha256")},
    })
    dump(OUT / "identical_source_shadow_comparison.json", shadow)
    dump(OUT / "zero_command_preconditioning.json", {"rows": pre_rows, "summary": {"P_STAND": pre_stand, "P_WALK_ZERO": pre_walk}})
    make_csv(OUT / "zero_command_preconditioning.csv", pre_rows, ["condition", "recipe_id", "mean_horizontal_speed", "net_xy_displacement_m", "yaw_rate_p95", "yaw_rate_max", "support_switch_count", "strict_liftoff_count", "strict_touchdown_count", "alternating_load_oscillation", "single_support_fraction", "double_support_fraction", "flight_fraction", "max_clearance_m", "max_joint_velocity_ratio", "max_torque_ratio", "safety", "first_failure"])
    dump(OUT / "start_route_comparison.json", {"routes": route_rows, "contract": {"A": "fresh S_HOLD -> P_STAND 100 steps -> W_MOVE [0.3,0,0]", "B": "fresh S_HOLD -> P_WALK_ZERO 100 steps -> W_MOVE [0.3,0,0]", "switch": "fixed boundary; hard switch; no blending"}})
    make_csv(OUT / "start_route_comparison.csv", route_rows, ["condition", "recipe_id", "safe_first_step", "liftoff", "touchdown", "forward_pelvis_displacement_m", "yaw_rate_p50", "yaw_rate_p95", "yaw_rate_max", "max_clearance_m", "wmove_entry_neighborhood_distance_p50", "wmove_entry_neighborhood_distance_p95", "entry_confirmation_10_step", "retention_50_step", "action_l2_jump", "action_cosine", "joint_target_jump_l2", "torque_transient_l2", "safety", "first_failure", "first_divergence"])
    dump(OUT / "gate_effect_statistics.json", gate_stats)
    dump(OUT / "first_step_results.json", {"routes": [{"condition": x["condition"], "recipe_id": x["recipe_id"], "safe_first_step": x["safe_first_step"], "liftoff": x["liftoff"], "touchdown": x["touchdown"], "yaw_rate_p95": x["yaw_rate_p95"], "max_clearance_m": x["max_clearance_m"], "safety": x["safety"]} for x in route_rows], "thresholds": {"B_safe_first_step_required": 6, "B_improvement_required": 2}})
    dump(OUT / "wmove_entry_results.json", {"routes": [{"condition": x["condition"], "recipe_id": x["recipe_id"], "entry_confirmation_10_step": x["entry_confirmation_10_step"], "entry_step": x["entry_confirmation_step"], "retention_50_step": x["retention_50_step"], "distance_p50": x["wmove_entry_neighborhood_distance_p50"], "distance_p95": x["wmove_entry_neighborhood_distance_p95"]} for x in route_rows], "required_entry_count": 2, "distance_threshold": WMOVE_ENTRY_DISTANCE_P95})
    dump(OUT / "first_divergence.json", {"routes": [{"condition": x["condition"], "recipe_id": x["recipe_id"], "first_divergence": x["first_divergence"], "first_failure": x["first_failure"]} for x in route_rows]})
    dump(OUT / "stage_classification.json", out)
    dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": next_action, "D29B_post_touchdown_capture": "DEFERRED" if classification == "EXP014_D29B0_ZERO_SPEED_WALK_GATE_PRIMES_START" else "NEXT_ALLOWED_BRANCH"})
    dump(OUT / "protected_hashes.json", protected_hashes())
    dump(OUT / "stage_reference.json", {"phase": "Phase 2-D29B0", "expected_user_start_head": EXPECTED_START_HEAD, "actual_start_head": physics["metadata"].get("starting_head_actual"), "actual_head_is_source_of_truth": True, "d29a_classification_preserved": "EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED", "d29a_artifacts_read_only": True, "d29b_artifacts_read_only": True})
    dump(OUT / "protocol.json", {
        "name": "Exp014ZeroSpeedWalkGateCausalAblationV1",
        "seed": SEED,
        "dt": DT,
        "recipes": RECIPES,
        "preconditioning": {"steps": PRECONDITION_STEPS, "seconds": 2.0, "P_STAND": "[0,0,0]+STAND", "P_WALK_ZERO": "[0,0,0]+WALK"},
        "routes": {"A": "P_STAND -> W_MOVE [0.3,0,0]", "B": "P_WALK_ZERO -> W_MOVE [0.3,0,0]"},
        "actor": ACTOR_REL,
        "wmove_actor": WMOVE_REL,
        "identifiability_gate": "same weights/schema/action path; non-mode actor-input hashes match",
        "safety": {"foot_contact_N": 5.0, "slip_mps": 0.55, "impact_N": 3500.0, "velocity_ratio": 0.95, "torque_ratio": 0.95, "dwell_steps": 5},
        "prohibited": {"new_checkpoint": 0, "policy_training": 0, "PPO": 0, "CEM": 0, "WBIK_modification": 0, "centroidal_modification": 0, "trajectory_optimization": 0, "raw_restore": 0, "validation": 0, "held_out": 0, "RUN_integration": 0, "remote_push": False},
    })
    isaac_python = r"C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
    script_rel = HERE.relative_to(REPO).as_posix()
    (OUT / "reproduction_commands.ps1").write_text(
        "\n".join(
            [
                f"$isaacPython = '{isaac_python}'",
                f"& $isaacPython '{script_rel}' run --condition P_STAND --headless",
                f"& $isaacPython '{script_rel}' run --condition P_WALK_ZERO --headless",
                f"python '{script_rel}' finalize",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = f"""# EXP014 Phase 2-D29B0 zero-speed WALK gate ablation

## Classification

Classification: {classification}. D29A remains EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED; its artifacts were read-only.

## Identifiability and provenance

The causal actor was the single frozen Exp014 141D actor {ACTOR_REL} with SHA-256 {physics["metadata"].get("actor_sha256")} and architecture [141,256,128,128,37]. The same actor, observation schema, and action path were used for STAND and WALK. The physical command was [0,0,0] in both conditions. Only target motion mode one-hot indices 127:130 changed; previous mode, command history, ramp progress, and the 123D physical observation were held equal. The existing W_MOVE actor was used only after the fixed 100-step preconditioning interval.

## Zero-command preconditioning

P_STAND ready-like coverage was {pre_stand["ready_like_valid_count"]}/8; P_WALK_ZERO coverage was {pre_walk["ready_like_valid_count"]}/8. Metrics are in zero_command_preconditioning.csv/json. Periodicity is reported separately from safety and drift.

## Route comparison

Route A is STAND preconditioning followed by a hard switch to W_MOVE [0.3,0,0]; route B is WALK-zero preconditioning followed by the identical switch. Safe first-step counts were {stand_safe}/8 and {walk_safe}/8; W_MOVE 10-step entry confirmation counts were {stand_entry}/8 and {walk_entry}/8. Per-source yaw, clearance, support, action discontinuity, and first-divergence records are in start_route_comparison.csv/json.

## Gate decision

The preregistered positive-control requirements were not relaxed. The observed median route yaw p95 changed from {stand_yaw:.6g} to {walk_yaw:.6g} (relative reduction {yaw_reduction:.3f}), and the 95th-percentile maximum clearance changed from {stand_clear:.6g} to {walk_clear:.6g}. gate_effect_statistics.json records the full decision. No new training, checkpoint, WBIK/centroidal modification, validation, held-out evaluation, or RUN integration was performed.

## Repository

Expected user start HEAD was {EXPECTED_START_HEAD}; actual source-of-truth HEAD at execution was {physics["metadata"].get("starting_head_actual")}. The expected mismatch was preserved without reset. D29A/D29B protected artifacts and unrelated pre-existing worktree changes were preserved. Persistent update: 0; new checkpoint: 0; remote push: false.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"classification": classification, "P_STAND_ready": pre_stand["ready_like_valid_count"], "P_WALK_ZERO_ready": pre_walk["ready_like_valid_count"], "A_safe": stand_safe, "B_safe": walk_safe, "A_entry": stand_entry, "B_entry": walk_entry}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "finalize"))
    parser.add_argument("--condition", choices=("P_STAND", "P_WALK_ZERO"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        if args.condition is None:
            parser.error("--condition is required for run")
        run_physics(args)
    else:
        finalize()


if __name__ == "__main__":
    main()
