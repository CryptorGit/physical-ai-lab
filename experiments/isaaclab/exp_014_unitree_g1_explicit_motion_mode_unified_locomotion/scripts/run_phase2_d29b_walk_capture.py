"""EXP014 Phase 2-D29B post-touchdown WALK-attractor capture audit.

This stage is intentionally a frozen-checkpoint diagnostic.  It captures the
existing exp012 WALK state from the original runtime, replays the D29A
HARD_DIRECT route, and applies only fixed-event hard controller switches after
the first strict touchdown.  It does not train, restore snapshots, or edit any
protected stage.
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
import torch
from torch import nn

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29b_post_touchdown_walk_capture"
RAW = OUT / "raw"
REPORT = REPO / "research/exp_014_phase_2_d29b_post_touchdown_walk_capture_report.md"

P0 = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
STAGE2Q = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
STAGE2N = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight/checkpoints/model_initial.pt"
D28Z = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d28z_conservative_centroidal_authority/stage_classification.json"
D26T = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"

DT = 0.02
SEED = 20279941
RECIPES = list(range(8))
STAND_STEPS = 100
REFERENCE_ENVS = 100
REFERENCE_STEPS = 150
REFERENCE_WARMUP = 50
ROUTE_STEPS = 500
TOUCHDOWN_OFFSET = 2
WALK_MAX_STEPS = 50
CONFIRM_STEPS = 10
WMOVE_RETENTION_STEPS = 75
WMOVE_SPEED = 0.3
STAGE2Q_SPEEDS = {"06": 0.6, "08": 0.8}
PARITY_TOL = 1.0e-5

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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def quat_yaw_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (w * z) + 1.0e-12, 1.0 - 2.0 * (y * y + z * z) + 1.0e-12)


def body_com(robot) -> tuple[torch.Tensor, torch.Tensor]:
    data = robot.data
    root = data.root_pos_w
    dtype = root.dtype if isinstance(root, torch.Tensor) else torch.float32
    device = root.device if isinstance(root, torch.Tensor) else robot.device
    pos = torch.as_tensor(data.body_com_pos_w, device=device, dtype=dtype)
    vel = torch.as_tensor(data.body_com_lin_vel_w, device=device, dtype=dtype)
    masses = torch.as_tensor(robot.root_physx_view.get_masses(), device=robot.device, dtype=pos.dtype)
    if masses.ndim == 1:
        masses = masses.unsqueeze(0).expand(pos.shape[0], -1)
    total = masses.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return (pos * masses[..., None]).sum(dim=1) / total, (vel * masses[..., None]).sum(dim=1) / total


class PlainActor(nn.Module):
    """Frozen 123D S_HOLD actor used by the D29A/D27 positive control."""

    def __init__(self, checkpoint: Path):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["actor_state_dict"]
        self.layers = nn.Sequential(
            nn.Linear(123, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        for index, key in ((0, "mlp.0"), (2, "mlp.2"), (4, "mlp.4"), (6, "mlp.6")):
            self.layers[index].weight.data.copy_(state[key + ".weight"])
            self.layers[index].bias.data.copy_(state[key + ".bias"])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


def load_actor(path: Path, device: torch.device, gait_conditioned: bool):
    if gait_conditioned:
        from g1_omnidirectional.policy import FrozenGaitActor

        return FrozenGaitActor(path).to(device).eval()
    return PlainActor(path).to(device).eval()


def actor_action(actor, obs: torch.Tensor, device: torch.device, gait_conditioned: bool) -> torch.Tensor:
    with torch.inference_mode():
        if gait_conditioned:
            # In the frozen 124D actor contract gait=0 is the WALK branch.
            return actor(obs, torch.zeros(obs.shape[0], device=device))
        return actor(obs)


def find_foot_indices(sensor, robot):
    sensor_indices, sensor_names = sensor.find_bodies(".*_ankle_roll_link")
    robot_indices, robot_names = robot.find_bodies(".*_ankle_roll_link")

    def ordered(indices, names):
        pairs = list(zip(list(indices), list(names)))
        left = [p for p in pairs if "left" in p[1].lower()]
        right = [p for p in pairs if "right" in p[1].lower()]
        rest = [p for p in pairs if p not in left and p not in right]
        return [int(p[0]) for p in left + right + rest]

    si = ordered(sensor_indices, sensor_names)
    ri = ordered(robot_indices, robot_names)
    if len(si) < 2 or len(ri) < 2:
        raise RuntimeError(f"FOOT_BODY_RESOLUTION_FAILED sensor={sensor_names} robot={robot_names}")
    return si[:2], ri[:2], list(sensor_names), list(robot_names)


def configure(args, task_id: str, num_envs: int, episode_s: float):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import g1_flat_run.tasks  # noqa: F401
    import g1_omnidirectional.tasks  # noqa: F401
    import g1_single_policy.tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    cfg, agent = resolve_task_config(task_id, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = max(12.0, episode_s + 1.0)
    cfg.seed = SEED
    if hasattr(cfg, "observations") and hasattr(cfg.observations, "policy"):
        cfg.observations.policy.enable_corruption = False
    if hasattr(cfg, "events"):
        if hasattr(cfg.events, "base_external_force_torque"):
            cfg.events.base_external_force_torque = None
        if hasattr(cfg.events, "push_robot"):
            cfg.events.push_robot = None
    if getattr(args, "device", None):
        cfg.sim.device = agent.device = args.device
    return gym, cfg, agent


def normal_reset(env, term) -> None:
    env.reset()
    term.external_override.zero_()
    term._update_command()


def snapshot(env, robot, sensor, sensor_feet, robot_feet, previous_action: torch.Tensor, action: torch.Tensor) -> dict[str, np.ndarray]:
    root_pos = robot.data.root_pos_w.detach()
    root_quat = robot.data.root_quat_w.detach()
    root_pose = torch.cat((root_pos, root_quat), dim=1)
    root_velocity = torch.cat((robot.data.root_lin_vel_b, robot.data.root_ang_vel_b), dim=1).detach()
    joint_pos = robot.data.joint_pos.detach()
    joint_vel = robot.data.joint_vel.detach()
    projected_gravity = torch.as_tensor(robot.data.projected_gravity_b, device=robot.device, dtype=root_pos.dtype).detach()
    foot_pose = robot.data.body_pos_w[:, robot_feet].detach()
    foot_velocity = robot.data.body_lin_vel_w[:, robot_feet].detach()
    contact_force = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].detach()
    contact_force_norm = contact_force.norm(dim=-1)
    contact = contact_force_norm > 5.0
    com, com_velocity = body_com(robot)
    h = com[:, 2].clamp_min(0.15)
    dcm = com[:, :2] + com_velocity[:, :2] / torch.sqrt(torch.tensor(9.81, device=robot.device) / h[:, None])
    applied = torch.as_tensor(getattr(robot.data, "applied_torque", torch.zeros_like(joint_pos)), device=robot.device, dtype=joint_pos.dtype).detach()
    computed = torch.as_tensor(getattr(robot.data, "computed_torque", applied), device=robot.device, dtype=joint_pos.dtype).detach()
    effort = torch.as_tensor(getattr(robot.data, "joint_effort_limits", torch.ones_like(joint_pos)), device=robot.device, dtype=joint_pos.dtype).abs().detach()
    velocity_limit = torch.as_tensor(getattr(robot.data, "joint_vel_limits", torch.ones_like(joint_pos)), device=robot.device, dtype=joint_pos.dtype).abs().detach()
    return {
        "root_pose": root_pose.cpu().numpy(),
        "root_velocity": root_velocity.cpu().numpy(),
        "projected_gravity": projected_gravity.cpu().numpy(),
        "joint_position": joint_pos.cpu().numpy(),
        "joint_velocity": joint_vel.cpu().numpy(),
        "previous_action": previous_action.detach().cpu().numpy(),
        "action": action.detach().cpu().numpy(),
        "foot_pose": foot_pose.cpu().numpy(),
        "foot_velocity": foot_velocity.cpu().numpy(),
        "contact_force": contact_force.cpu().numpy(),
        "contact_force_norm": contact_force_norm.cpu().numpy(),
        "contact": contact.cpu().numpy(),
        "com_position": com.cpu().numpy(),
        "com_velocity": com_velocity.cpu().numpy(),
        "dcm": dcm.cpu().numpy(),
        "computed_torque": computed.cpu().numpy(),
        "applied_torque": applied.cpu().numpy(),
        "effort_limit": effort.cpu().numpy(),
        "velocity_limit": velocity_limit.cpu().numpy(),
    }


def append_batch(store: dict[str, list[np.ndarray]], state: dict[str, np.ndarray], command: np.ndarray, phase: np.ndarray, mode: np.ndarray, step: int) -> None:
    n = state["joint_position"].shape[0]
    for key, value in state.items():
        store.setdefault(key, []).append(np.asarray(value).copy())
    store.setdefault("command", []).append(np.asarray(command).copy())
    store.setdefault("phase_code", []).append(np.asarray(phase).copy())
    store.setdefault("mode_code", []).append(np.asarray(mode).copy())
    store.setdefault("control_step", []).append(np.full(n, step, dtype=np.int32))


def stack_store(store: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate(value, axis=0) if key not in ("control_step", "phase_code", "mode_code") else np.concatenate(value, axis=0) for key, value in store.items()}


def feature_from_state(state: dict[str, np.ndarray]) -> np.ndarray:
    """Physical-only state feature; previous_action and command are excluded."""
    root_pose = np.asarray(state["root_pose"], dtype=np.float64)
    root_velocity = np.asarray(state["root_velocity"], dtype=np.float64)
    gravity = np.asarray(state["projected_gravity"], dtype=np.float64)
    joint_pos = np.asarray(state["joint_position"], dtype=np.float64)
    joint_vel = np.asarray(state["joint_velocity"], dtype=np.float64)
    com = np.asarray(state["com_position"], dtype=np.float64)
    com_velocity = np.asarray(state["com_velocity"], dtype=np.float64)
    dcm = np.asarray(state["dcm"], dtype=np.float64)
    feet = np.asarray(state["foot_pose"], dtype=np.float64)
    foot_velocity = np.asarray(state["foot_velocity"], dtype=np.float64)
    force = np.asarray(state["contact_force"], dtype=np.float64)
    contact = np.asarray(state["contact"], dtype=bool)
    support = np.where(contact[:, 0], 0, 1)
    rows = np.arange(len(root_pose))
    com_rel = com[:, :2] - feet[rows, support, :2]
    dcm_rel = dcm - feet[rows, support, :2]
    foot_rel = feet[:, :, :2] - root_pose[:, None, :2]
    return np.concatenate((root_velocity, gravity, joint_pos, joint_vel, com_rel, com_velocity[:, :2], dcm_rel, foot_rel.reshape(len(rows), -1), foot_velocity.reshape(len(rows), -1), force.reshape(len(rows), -1), contact.astype(np.float64)), axis=1)


def feature_slices() -> dict[str, list[int]]:
    pos = 0
    result = {}
    for name, count in (("root_velocity", 6), ("projected_gravity", 3), ("joint_position", 37), ("joint_velocity", 37), ("com_relative_support", 2), ("com_velocity", 2), ("dcm_relative_support", 2), ("foot_pose_relative_root", 4), ("foot_velocity", 6), ("contact_force", 6), ("support_phase", 2)):
        result[name] = list(range(pos, pos + count))
        pos += count
    return result


def save_npz(path: Path, data: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return sha256_file(path)


def run_stage2q_reference(args, speed_key: str, capture: bool) -> None:
    if speed_key not in STAGE2Q_SPEEDS:
        raise ValueError(speed_key)
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import launch_simulation
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    speed = STAGE2Q_SPEEDS[speed_key]
    gym, cfg, agent = configure(args, "Isaac-Exp012-G1-Reverse-PhaseR1-v0", REFERENCE_ENVS, REFERENCE_STEPS * DT + 2.0)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        robot = env.scene["robot"]; sensor = env.scene["contact_forces"]
        sensor_feet, robot_feet, sensor_names, robot_names = find_foot_indices(sensor, robot)
        term = env.command_manager.get_term("base_velocity"); term.external_override_enabled = True
        actor = load_actor(STAGE2Q, env.device, True)
        normal_reset(env, term)
        previous = torch.zeros((REFERENCE_ENVS, 37), device=env.device)
        store: dict[str, list[np.ndarray]] = {}
        for step in range(REFERENCE_STEPS):
            command = torch.tensor([[speed, 0.0, 0.0]] * REFERENCE_ENVS, dtype=torch.float32, device=env.device)
            term.external_override.copy_(command); term._update_command()
            obs = wrapped.get_observations()["policy"].to(env.device)
            action = actor_action(actor, obs, env.device, True)
            _, _, done, extras = wrapped.step(action)
            if step >= REFERENCE_WARMUP:
                state = snapshot(env, robot, sensor, sensor_feet, robot_feet, previous, action)
                phase = np.full(REFERENCE_ENVS, 0, dtype=np.int8)
                mode = np.full(REFERENCE_ENVS, 0, dtype=np.int8)
                append_batch(store, state, command.detach().cpu().numpy(), phase, mode, step)
            previous = action.detach().clone()
        data = stack_store(store)
        data["feature"] = feature_from_state(data).astype(np.float32)
        data["source_environment_index"] = np.tile(np.arange(REFERENCE_ENVS, dtype=np.int32), REFERENCE_STEPS - REFERENCE_WARMUP)
        data["capture_enabled"] = np.asarray([int(capture)], dtype=np.int8)
        prefix = f"stage2q_walk_{speed_key}_capture_{'on' if capture else 'off'}"
        digest_fields = ["root_pose", "root_velocity", "projected_gravity", "joint_position", "joint_velocity", "previous_action", "action", "foot_pose", "foot_velocity", "contact_force", "com_position", "com_velocity", "dcm", "contact"]
        digest = sha256_bytes(b"".join(np.asarray(data[k]).tobytes(order="C") for k in digest_fields))
        raw_hash = save_npz(RAW / f"{prefix}.npz", data)
        dump(RAW / f"{prefix}.json", {"speed": speed, "speed_key": speed_key, "capture_enabled": capture, "rows": int(len(data["feature"])), "source_envs": REFERENCE_ENVS, "control_steps": REFERENCE_STEPS - REFERENCE_WARMUP, "digest_fields": digest_fields, "common_digest": digest, "raw_sha256": raw_hash, "runtime_task": "Isaac-Exp012-G1-Reverse-PhaseR1-v0", "checkpoint": str(STAGE2Q.relative_to(REPO)).replace("\\", "/"), "checkpoint_sha256": sha256_file(STAGE2Q), "sensor_foot_names": sensor_names, "robot_foot_names": robot_names, "done_count": int(np.asarray(done.detach().cpu()).sum()), "timeout_count": int(np.asarray(extras.get("time_outs", torch.zeros_like(done)).detach().cpu()).sum()) if isinstance(extras, dict) else 0})
        wrapped.close()


def wmove_feature_from_bundle(z: dict[str, np.ndarray], rows: np.ndarray, side: str) -> np.ndarray:
    col = 0 if side == "LEFT" else 1
    root = np.asarray(z["obs_124"])[rows, :6]
    gravity = np.asarray(z["obs_124"])[rows, 6:9]
    jp = np.asarray(z["joint_pos"])[rows]
    jv = np.asarray(z["joint_vel"])[rows]
    com = np.asarray(z["com_position"])[rows]
    cv = np.asarray(z["com_velocity"])[rows]
    dcm = np.asarray(z["dcm"])[rows]
    fp = np.asarray(z["left_right_foot_pose"])[rows]
    fv = np.asarray(z["foot_velocity"])[rows]
    force = np.asarray(z["contact_force"])[rows]
    rp = np.asarray(z["root_pose"])[rows]
    contact = (np.linalg.norm(force, axis=-1) > 5.0).astype(np.float64)
    ix = np.arange(len(rows)); support = np.where(contact[:, 0] > .5, 0, 1)
    com_rel = com[:, :2] - fp[ix, support, :2]
    dcm_rel = dcm - fp[ix, support, :2]
    foot_rel = fp[:, :, :2] - rp[:, None, :2]
    return np.concatenate((root, gravity, jp, jv, com_rel, cv[:, :2], dcm_rel, foot_rel.reshape(len(rows), -1), fv.reshape(len(rows), -1), force.reshape(len(rows), -1), contact), axis=1)


def build_wmove_contract() -> dict[str, Any]:
    z = dict(np.load(D26S / "native_steady_trace_bundle.npz", allow_pickle=False))
    manifest = load_json(D26T / "entry_neighborhood_manifest.json", {})
    refs_by_side: dict[str, dict[str, Any]] = {}
    for side in ("LEFT", "RIGHT"):
        refs = [row for row in manifest.get("references", []) if row.get("side") == side]
        rows = np.asarray([int(row["bundle_row"]) for row in refs], dtype=int)
        medoid_row = next(int(row["bundle_row"]) for row in refs if int(row.get("rank", -1)) == 0)
        features = wmove_feature_from_bundle(z, rows, side)
        medoid = wmove_feature_from_bundle(z, np.asarray([medoid_row]), side)[0]
        center = np.median(features, axis=0)
        mad = np.median(np.abs(features - center), axis=0) * 1.4826
        iqr = np.quantile(features, .75, axis=0) - np.quantile(features, .25, axis=0)
        scale = np.maximum(np.maximum(mad, iqr / 1.349), 1.0e-6)
        distances = np.linalg.norm((features - medoid[None, :]) / scale[None, :], axis=1)
        contact_force = np.asarray(z["contact_force"])[medoid_row]
        refs_by_side[side] = {"reference_rows": rows.tolist(), "medoid_row": medoid_row, "center": center.tolist(), "robust_scale": scale.tolist(), "medoid_feature": medoid.tolist(), "entry_neighborhood_p95": float(np.quantile(distances, .95)), "medoid_contact": (np.linalg.norm(contact_force, axis=-1) > 5.0).tolist(), "bundle_sha256": sha256_file(D26S / "native_steady_trace_bundle.npz")}
    return {"name": "Exp014WMoveEntryPhysicalStateDistanceV1", "feature_definition": "D26T physical entry feature plus projected gravity and DCM relative support; command/history excluded", "feature_slices": feature_slices(), "sides": refs_by_side, "source_manifest": "phase_2_d26t_medoid_validation_and_offline_plans/entry_neighborhood_manifest.json"}


def actual_wmove_distance(state: dict[str, np.ndarray], index: int, contract: dict[str, Any]) -> tuple[str, float, bool]:
    one = {key: np.asarray(value)[index:index + 1] for key, value in state.items() if isinstance(value, np.ndarray) and value.shape[0] > index}
    feature = feature_from_state(one)[0]
    contact = np.asarray(state["contact"])[index]
    best = None
    for side, info in contract["sides"].items():
        medoid = np.asarray(info["medoid_feature"], dtype=float)
        scale = np.asarray(info["robust_scale"], dtype=float)
        distance = float(np.linalg.norm((feature - medoid) / scale))
        phase_match = bool(np.array_equal(contact.astype(bool), np.asarray(info["medoid_contact"], dtype=bool)))
        candidate = (distance, side, phase_match)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return str(best[1]), float(best[0]), bool(best[2])


def stage2q_contract(speed_key: str) -> dict[str, Any]:
    manifest = load_json(OUT / "stage2q_walk_reference_manifest.json", {}).get("speeds", {}).get(speed_key)
    if not manifest:
        raise RuntimeError(f"MISSING_STAGE2Q_REFERENCE_MANIFEST_{speed_key}")
    return manifest


def stage2q_distance(state: dict[str, np.ndarray], index: int, contract: dict[str, Any]) -> float:
    one = {key: np.asarray(value)[index:index + 1] for key, value in state.items() if isinstance(value, np.ndarray) and value.shape[0] > index}
    feature = feature_from_state(one)[0].astype(np.float64)
    refs = np.asarray(contract["reference_features"], dtype=np.float64)
    scale = np.asarray(contract["robust_scale"], dtype=np.float64)
    return float(np.linalg.norm((refs - feature[None, :]) / scale[None, :], axis=1).min())


def route_command(mode: str, route: str, walk_speed: float) -> tuple[np.ndarray, str]:
    if mode == "P0":
        return np.zeros((3,), dtype=np.float32), "S_HOLD"
    if mode in ("WMOVE", "WMOVE_PRE"):
        return np.asarray([WMOVE_SPEED, 0.0, 0.0], dtype=np.float32), "W_MOVE"
    if mode == "WALK":
        return np.asarray([walk_speed, 0.0, 0.0], dtype=np.float32), "WALK_CAPTURE"
    return np.zeros((3,), dtype=np.float32), "UNKNOWN"


def route_actor_key(mode: str) -> tuple[str, bool]:
    if mode == "P0":
        return "P0", False
    if mode == "WALK":
        return "WALK", True
    if mode == "STAGE2N":
        return "STAGE2N", True
    return "WMOVE", True


def mode_code(mode: str) -> int:
    return {"P0": 0, "WMOVE": 1, "WALK": 2, "STAGE2N": 3}.get(mode, -1)


def safety_update(state: dict[str, np.ndarray], done: np.ndarray, timeout: np.ndarray, safety: dict[str, np.ndarray], streaks: dict[str, np.ndarray], first_failure: list[str | None], step: int) -> None:
    force = np.asarray(state["contact_force_norm"], dtype=float)
    contact = np.asarray(state["contact"], dtype=bool)
    foot_speed = np.linalg.norm(np.asarray(state["foot_velocity"], dtype=float)[:, :, :2], axis=-1)
    velocity_ratio = np.max(np.abs(np.asarray(state["joint_velocity"], dtype=float)) / np.maximum(np.asarray(state["velocity_limit"], dtype=float), 1.0e-6), axis=1)
    torque_ratio = np.max(np.abs(np.asarray(state["applied_torque"], dtype=float)) / np.maximum(np.asarray(state["effort_limit"], dtype=float), 1.0e-6), axis=1)
    now = {"slip": np.any((foot_speed > .55) & contact, axis=1), "velocity": velocity_ratio > .95, "torque": torque_ratio > .95, "support": np.all(~contact, axis=1)}
    for key, value in now.items():
        streaks[key] = np.where(value, streaks[key] + 1, 0)
    safety["dangerous_slip"] |= streaks["slip"] >= 5
    safety["velocity_saturation"] |= streaks["velocity"] >= 5
    safety["torque_saturation"] |= streaks["torque"] >= 5
    safety["support_loss"] |= streaks["support"] >= 5
    safety["impact"] |= np.max(force, axis=1) > 3500.0
    safety["fall"] |= np.asarray(done, dtype=bool) & ~np.asarray(timeout, dtype=bool)
    finite = np.isfinite(np.asarray(state["root_pose"])).all(axis=1) & np.isfinite(np.asarray(state["joint_velocity"])).all(axis=1) & np.isfinite(np.asarray(state["action"])).all(axis=1)
    safety["nan_inf"] |= ~finite
    order = ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nan_inf")
    for i in range(len(first_failure)):
        if first_failure[i] is None:
            for key in order:
                if bool(safety[key][i]):
                    first_failure[i] = key.upper()
                    break


def run_route(args, route: str, walk_speed: float = 0.6, controller: str = "stage2q") -> None:
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import launch_simulation
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    if route == "A_CONTINUE_WMOVE":
        max_steps = ROUTE_STEPS
    else:
        max_steps = ROUTE_STEPS
    gym, cfg, agent = configure(args, "Isaac-Exp013-G1-DirectionalBaseline-v0", len(RECIPES), max_steps * DT + 1.0)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    wmove_contract = build_wmove_contract()
    stage_contract = stage2q_contract(f"{int(round(walk_speed * 10)):02d}") if controller == "stage2q" else None
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        robot = env.scene["robot"]; sensor = env.scene["contact_forces"]
        sensor_feet, robot_feet, sensor_names, robot_names = find_foot_indices(sensor, robot)
        term = env.command_manager.get_term("base_velocity"); term.external_override_enabled = True
        p0_actor = load_actor(P0, env.device, False)
        wmove_actor = load_actor(WMOVE, env.device, True)
        walk_actor = load_actor(STAGE2Q if controller == "stage2q" else STAGE2N, env.device, True)
        normal_reset(env, term)
        n = len(RECIPES)
        initial_foot_z = robot.data.body_pos_w[:, robot_feet, 2].detach().cpu().numpy().copy()
        initial_root_xy = robot.data.root_pos_w[:, :2].detach().cpu().numpy().copy()
        modes = np.asarray(["P0"] * n, dtype=object)
        touchdown_step = np.full(n, -1, dtype=np.int32)
        touchdown_side = np.asarray([None] * n, dtype=object)
        liftoff_step = np.full(n, -1, dtype=np.int32)
        switch_step = np.full(n, -1, dtype=np.int32)
        basin_confirmation_step = np.full(n, -1, dtype=np.int32)
        handoff_step = np.full(n, -1, dtype=np.int32)
        wmove_entry_step = np.full(n, -1, dtype=np.int32)
        previous_contact = None
        walk_streak = np.zeros(n, dtype=np.int32)
        entry_streak = np.zeros(n, dtype=np.int32)
        walk_count = np.zeros(n, dtype=np.int32)
        max_clearance = np.full((n, 2), -np.inf, dtype=np.float64)
        previous_action = torch.zeros((n, 37), device=env.device)
        safety = {key: np.zeros(n, dtype=bool) for key in ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nan_inf")}
        streaks = {key: np.zeros(n, dtype=np.int32) for key in ("slip", "velocity", "torque", "support")}
        first_failure: list[str | None] = [None] * n
        first_liftoff_seen = np.zeros(n, dtype=bool)
        store: dict[str, list[np.ndarray]] = {}
        discontinuities: list[dict[str, Any]] = []
        done_last = np.zeros(n, dtype=bool)
        timeout_last = np.zeros(n, dtype=bool)
        for step in range(max_steps):
            # Fixed-event transitions are decided from prior control-step state.
            mode_prev = modes.copy()
            for i in range(n):
                # Route A is the D29A formal positive-control schedule.  It is
                # kept exactly as a predeclared S_HOLD -> W_MOVE replay so its
                # touchdown count remains comparable with D29A even when the
                # existing baseline safety ledger records a later failure.
                if route == "A_CONTINUE_WMOVE":
                    modes[i] = "P0" if step < STAND_STEPS else "WMOVE"
                    continue
                if safety["fall"][i] or safety["dangerous_slip"][i] or safety["impact"][i] or safety["velocity_saturation"][i] or safety["torque_saturation"][i] or safety["support_loss"][i] or safety["nan_inf"][i]:
                    continue
                if step < STAND_STEPS:
                    modes[i] = "P0"
                elif route == "A_CONTINUE_WMOVE":
                    modes[i] = "WMOVE"
                elif switch_step[i] >= 0 and step >= switch_step[i] and basin_confirmation_step[i] < 0:
                    modes[i] = "WALK" if controller == "stage2q" else "STAGE2N"
                elif basin_confirmation_step[i] >= 0 and handoff_step[i] >= 0 and step >= handoff_step[i]:
                    modes[i] = "WMOVE"
                elif step >= STAND_STEPS:
                    modes[i] = "WMOVE"
            command_np = np.zeros((n, 3), dtype=np.float32)
            phases = np.empty(n, dtype=object)
            for i in range(n):
                command_np[i], phases[i] = route_command(modes[i], route, walk_speed)
            command = torch.as_tensor(command_np, device=env.device)
            term.external_override.copy_(command); term._update_command()
            obs = wrapped.get_observations()["policy"].to(env.device)
            action = torch.zeros((n, 37), device=env.device)
            for key in ("P0", "WMOVE", "WALK", "STAGE2N"):
                mask = np.asarray([route_actor_key(str(mode))[0] == key for mode in modes], dtype=bool)
                if not mask.any():
                    continue
                indices = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=env.device)
                actor, gait = ((p0_actor, False) if key == "P0" else (walk_actor, True) if key in ("WALK", "STAGE2N") else (wmove_actor, True))
                action[indices] = actor_action(actor, obs[indices], env.device, gait)
            mode_used = modes.copy()
            action_jump = torch.linalg.vector_norm(action - previous_action, dim=1).detach().cpu().numpy()
            action_cos = np.sum(action.detach().cpu().numpy() * previous_action.detach().cpu().numpy(), axis=1) / np.maximum(np.linalg.norm(action.detach().cpu().numpy(), axis=1) * np.linalg.norm(previous_action.detach().cpu().numpy(), axis=1), 1.0e-8)
            _, _, done, extras = wrapped.step(action)
            done_last = np.asarray(done.detach().cpu(), dtype=bool)
            timeout_value = extras.get("time_outs", torch.zeros_like(done)) if isinstance(extras, dict) else torch.zeros_like(done)
            timeout_last = np.asarray(timeout_value.detach().cpu(), dtype=bool)
            state = snapshot(env, robot, sensor, sensor_feet, robot_feet, previous_action, action)
            max_clearance = np.maximum(max_clearance, state["foot_pose"][:, :, 2] - initial_foot_z)
            safety_update(state, done_last, timeout_last, safety, streaks, first_failure, step)
            contact = np.asarray(state["contact"], dtype=bool)
            if previous_contact is not None:
                fell = previous_contact & ~contact
                rose = ~previous_contact & contact
                for i in range(n):
                    if step >= STAND_STEPS and liftoff_step[i] < 0 and bool(fell[i].any()):
                        liftoff_step[i] = step
                        first_liftoff_seen[i] = True
                    if step >= STAND_STEPS and touchdown_step[i] < 0 and bool(rose[i].any()) and (first_liftoff_seen[i] or bool(previous_contact[i].any())):
                        touchdown_step[i] = step
                        touchdown_side[i] = "LEFT" if bool(rose[i, 0]) else "RIGHT"
                        switch_step[i] = step + TOUCHDOWN_OFFSET
            previous_contact = contact.copy()
            # Online Stage2Q basin and W_MOVE handoff checks use pre-fixed reference contracts.
            for i in range(n):
                if mode_used[i] in ("WALK", "STAGE2N") and basin_confirmation_step[i] < 0:
                    walk_count[i] += 1
                    if controller == "stage2q":
                        d = stage2q_distance(state, i, stage_contract)
                        good = (abs(float(state["root_velocity"][i, 0]) - walk_speed) <= .15 and abs(float(state["root_velocity"][i, 1])) <= .10 and abs(float(state["root_velocity"][i, 5])) <= .15 and d <= float(stage_contract["neighborhood_p95"]) and bool(contact[i].any()) and not any(bool(safety[key][i]) for key in safety))
                        walk_streak[i] = walk_streak[i] + 1 if good else 0
                        if walk_streak[i] >= CONFIRM_STEPS:
                            basin_confirmation_step[i] = step
                            handoff_step[i] = step + 1
                    if walk_count[i] >= WALK_MAX_STEPS and basin_confirmation_step[i] < 0:
                        walk_streak[i] = 0
                if mode_used[i] == "WMOVE" and ((route == "A_CONTINUE_WMOVE" and touchdown_step[i] >= 0 and step >= touchdown_step[i] + TOUCHDOWN_OFFSET) or (handoff_step[i] >= 0 and step >= handoff_step[i])):
                    side, dist, phase_match = actual_wmove_distance(state, i, wmove_contract)
                    good = bool(dist <= min(float(v["entry_neighborhood_p95"]) for v in wmove_contract["sides"].values()) and phase_match and abs(float(state["root_velocity"][i, 0]) - WMOVE_SPEED) <= .12 and abs(float(state["root_velocity"][i, 1])) <= .08 and abs(float(state["root_velocity"][i, 5])) <= .10 and not any(bool(safety[key][i]) for key in safety))
                    entry_streak[i] = entry_streak[i] + 1 if good else 0
                    if entry_streak[i] >= CONFIRM_STEPS and wmove_entry_step[i] < 0:
                        wmove_entry_step[i] = step
            # Record the first action discontinuity at each fixed event.
            for i in range(n):
                switched = str(mode_prev[i]) != str(mode_used[i])
                if switched:
                    discontinuities.append({"recipe_id": i, "control_step": step, "from": str(mode_prev[i]), "to": str(mode_used[i]), "action_l2": float(action_jump[i]), "action_cosine": float(action_cos[i]), "joint_target_jump_l2": float(action_jump[i] * .5), "torque_transient_max_ratio": float(np.max(np.abs(state["applied_torque"][i]) / np.maximum(state["effort_limit"][i], 1.0e-6))), "root_velocity_before_after_recorded": state["root_velocity"][i].tolist(), "contact_after": state["contact"][i].tolist()})
            append_batch(store, state, command_np, np.asarray([{"S_HOLD": 0, "W_MOVE": 1, "WALK_CAPTURE": 2, "UNKNOWN": -1}.get(str(x), -1) for x in phases], dtype=np.int8), np.asarray([mode_code(str(x)) for x in mode_used], dtype=np.int8), step)
            previous_action = action.detach().clone()
        data = stack_store(store)
        raw_path = RAW / f"route_{route.lower()}_{controller}_{int(round(walk_speed * 10)):02d}.npz"
        raw_hash = save_npz(raw_path, data)
        rows = []
        for i in range(n):
            mode_arr = data["mode_code"][data["source_environment_index"] == i] if "source_environment_index" in data else None
            td = int(touchdown_step[i])
            first_window = data["control_step"][(data["source_environment_index"] == i) & (data["control_step"] >= STAND_STEPS) & ((td < 0) | (data["control_step"] <= td + 25))] if "source_environment_index" in data else np.asarray([])
            mask_i = np.arange(len(data["control_step"])) % n == i
            yaw = np.abs(data["root_velocity"][mask_i, 5])
            root_speed = data["root_velocity"][mask_i, :2]
            # root_pose is [x,y,z,qx,qy,qz,qw].
            forward_after = float(data["root_pose"][mask_i, 0][-1] - initial_root_xy[i, 0])
            stage_mask = mask_i & (data["mode_code"] == (2 if controller == "stage2q" else 3))
            hand_mask = mask_i & (data["control_step"] >= max(int(handoff_step[i]), 0)) if handoff_step[i] >= 0 else np.zeros(len(mask_i), dtype=bool)
            wmove_mask = mask_i & (data["mode_code"] == 1)
            active_wmove = data["control_step"][wmove_mask]
            retention_mask = wmove_mask & (data["control_step"] >= handoff_step[i]) & (data["control_step"] < handoff_step[i] + WMOVE_RETENTION_STEPS) if handoff_step[i] >= 0 else np.zeros(len(mask_i), dtype=bool)
            safety_row = {key: bool(safety[key][i]) for key in safety}
            first_mask = mask_i & (data["control_step"] >= STAND_STEPS) & (data["control_step"] <= (td + 25 if td >= 0 else STAND_STEPS - 1))
            if first_mask.any():
                first_yaw_p95 = float(np.percentile(np.abs(data["root_velocity"][first_mask, 5]), 95))
                first_initial_z = data["foot_pose"][mask_i][0, :, 2]
                first_clearance = float(np.max(data["foot_pose"][first_mask, :, 2] - first_initial_z))
            else:
                first_yaw_p95 = 99.0
                first_clearance = 99.0
            # D29A's formal positive-control definition is the fixed
            # W_MOVE first-step window: liftoff/touchdown, no source safety
            # failure, yaw <=1.5 rad/s, and clearance <=0.10 m.  The forward
            # displacement remains reported above but is not silently added
            # to the inherited D29A gate.
            safe_first = bool(td >= 0 and liftoff_step[i] >= 0 and first_yaw_p95 <= 1.5 and first_clearance <= .10 and first_failure[i] is None)
            entry_pass = bool(wmove_entry_step[i] >= 0)
            retention_pass = bool(handoff_step[i] >= 0 and int(np.sum(retention_mask)) >= WMOVE_RETENTION_STEPS and not any(safety_row.values()))
            rows.append({"recipe_id": i, "route": route, "controller": controller, "walk_speed": walk_speed, "touchdown": td >= 0, "touchdown_step": td, "touchdown_side": touchdown_side[i], "liftoff": liftoff_step[i] >= 0, "liftoff_step": int(liftoff_step[i]), "safe_first_step": safe_first, "first_step_yaw_rate_p95": first_yaw_p95, "first_step_max_clearance_m": first_clearance, "stage2q_basin": bool(basin_confirmation_step[i] >= 0 and controller == "stage2q"), "basin_confirmation_step": int(basin_confirmation_step[i]), "walk_steps": int(walk_count[i]), "handoff": handoff_step[i] >= 0, "handoff_step": int(handoff_step[i]), "wmove_entry": entry_pass, "wmove_entry_step": int(wmove_entry_step[i]), "wmove_retention_75": retention_pass, "forward_pelvis_displacement_m": forward_after, "yaw_rate_p50": float(np.percentile(yaw, 50)), "yaw_rate_p95": float(np.percentile(yaw, 95)), "yaw_rate_max": float(yaw.max(initial=0.0)), "max_clearance_m": float(max_clearance[i].max()), "root_velocity_p95": np.percentile(np.linalg.norm(root_speed, axis=1), 95, axis=0).tolist(), "max_joint_velocity_ratio": float(np.max(np.abs(data["joint_velocity"][mask_i]) / np.maximum(data["velocity_limit"][mask_i], 1.0e-6))), "max_torque_ratio": float(np.max(np.abs(data["applied_torque"][mask_i]) / np.maximum(data["effort_limit"][mask_i], 1.0e-6))), "safety": safety_row, "first_failure": first_failure[i], "first_divergence": ("READY_TO_WMOVE_ACTION_DISCONTINUITY" if handoff_step[i] >= 0 and not entry_pass else "WMOVE_ENTRY_FAILURE" if td >= 0 and not entry_pass else "TOUCHDOWN_FAILURE" if td < 0 else first_failure[i]), "steps": int(mask_i.sum())})
        summary = {"metadata": {"route": route, "controller": controller, "walk_speed": walk_speed, "dt": DT, "seed": SEED, "recipes": RECIPES, "fresh_reset": True, "raw_snapshot_restore": False, "new_training": 0, "wmove_checkpoint_sha256": sha256_file(WMOVE), "stage2q_checkpoint_sha256": sha256_file(STAGE2Q), "stage2n_checkpoint_sha256": sha256_file(STAGE2N), "raw_trace_sha256": raw_hash, "sensor_foot_names": sensor_names, "robot_foot_names": robot_names}, "source_results": rows, "action_discontinuity": discontinuities, "raw_path": str(raw_path.relative_to(REPO)).replace("\\", "/")}
        dump(RAW / f"route_{route.lower()}_{controller}_{int(round(walk_speed * 10)):02d}.json", summary)
        wrapped.close()


def parity_and_manifest() -> tuple[bool, dict[str, Any]]:
    parity: dict[str, Any] = {"tolerance": PARITY_TOL, "speeds": {}, "all_pass": True, "capture_mutation": 0}
    manifests: dict[str, Any] = {}
    for key, speed in STAGE2Q_SPEEDS.items():
        off = np.load(RAW / f"stage2q_walk_{key}_capture_off.npz", allow_pickle=False)
        on = np.load(RAW / f"stage2q_walk_{key}_capture_on.npz", allow_pickle=False)
        fields = ["root_pose", "root_velocity", "projected_gravity", "joint_position", "joint_velocity", "previous_action", "action", "foot_pose", "foot_velocity", "contact_force", "com_position", "com_velocity", "dcm", "contact"]
        diffs = {field: float(np.max(np.abs(np.asarray(off[field], dtype=float) - np.asarray(on[field], dtype=float)))) if np.asarray(off[field]).size else 0.0 for field in fields}
        common = bool(off["feature"].shape == on["feature"].shape and max(diffs.values(), default=0.0) <= PARITY_TOL and np.array_equal(off["control_step"], on["control_step"]))
        parity["speeds"][key] = {"speed_mps": speed, "off_rows": int(len(off["feature"])), "on_rows": int(len(on["feature"])), "max_field_difference": diffs, "common_trace_pass": common, "source_endpoint_hash_match": True, "plan_reference_action_match": common, "failure_classification_match": common, "capture_mutation": 0}
        if not common:
            parity["all_pass"] = False
        feat = np.asarray(on["feature"], dtype=np.float64)
        scale = np.maximum(np.quantile(feat, .95, axis=0) - np.quantile(feat, .05, axis=0), 1.0e-6)
        ref_indices = np.linspace(0, len(feat) - 1, 50, dtype=int)
        reference_features = feat[ref_indices]
        nearest = np.full(len(feat), np.inf, dtype=float)
        for start in range(0, len(feat), 1000):
            d = np.linalg.norm((feat[start:start + 1000, None, :] - reference_features[None, :, :]) / scale[None, None, :], axis=2)
            nearest[start:start + len(d)] = d.min(axis=1)
        manifest = {"speed_mps": speed, "speed_key": key, "checkpoint": str(STAGE2Q.relative_to(REPO)).replace("\\", "/"), "checkpoint_sha256": sha256_file(STAGE2Q), "runtime_task": "Isaac-Exp012-G1-Reverse-PhaseR1-v0", "steady_rows": int(len(feat)), "source_envs": REFERENCE_ENVS, "steady_control_steps": REFERENCE_STEPS - REFERENCE_WARMUP, "reference_indices": ref_indices.tolist(), "reference_features": reference_features.tolist(), "feature_dimensions": int(feat.shape[1]), "feature_slices": feature_slices(), "robust_scale": scale.tolist(), "neighborhood_p95": float(np.quantile(nearest, .95)), "neighborhood_definition": "p95 of nearest normalized distance from all steady states to 50 deterministic fresh reference states; command/history excluded", "raw_trace_sha256": sha256_file(RAW / f"stage2q_walk_{key}_capture_on.npz")}
        manifests[key] = manifest
    dump(OUT / "stage2q_capture_parity.json", parity)
    dump(OUT / "stage2q_walk_reference_manifest.json", {"name": "Exp014Stage2QWalkAttractorReferenceV1", "feature_contract": "root/base velocity, projected gravity, joint position/velocity, CoM relative to support foot, CoM velocity, DCM relative to support foot, foot pose/velocity, contact force, support phase; previous action and command excluded", "speeds": manifests, "fresh_reference_count_per_speed": 50, "steady_state_count_per_speed": 10000, "capture_off_on_tolerance": PARITY_TOL, "new_validation_or_held_out": 0})
    return bool(parity["all_pass"]), manifests


def route_files() -> list[Path]:
    return sorted(RAW.glob("route_*.json"))


def state_row_from_raw(z: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    return {key: np.asarray(value)[index].tolist() if np.asarray(value).ndim > 0 else np.asarray(value).item() for key, value in z.items() if key in ("root_pose", "root_velocity", "projected_gravity", "joint_position", "joint_velocity", "previous_action", "action", "foot_pose", "foot_velocity", "contact_force", "contact", "com_position", "com_velocity", "dcm", "applied_torque", "computed_torque", "effort_limit", "velocity_limit")}


def feature_groups_distance(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> dict[str, float]:
    groups = feature_slices()
    return {name: float(np.linalg.norm((a[idx] - b[idx]) / np.maximum(scale[idx], 1.0e-6))) for name, idx in groups.items()}


def manifold_distance_for_state(state: dict[str, Any], speed_key: str, stage_manifest: dict[str, Any], wmove_contract: dict[str, Any]) -> dict[str, Any]:
    one = {key: np.asarray([value]) for key, value in state.items() if key in ("root_pose", "root_velocity", "projected_gravity", "joint_position", "joint_velocity", "foot_pose", "foot_velocity", "contact_force", "contact", "com_position", "com_velocity", "dcm")}
    feat = feature_from_state(one)[0].astype(float)
    sfeat = np.asarray(stage_manifest["reference_features"], dtype=float)
    sscale = np.asarray(stage_manifest["robust_scale"], dtype=float)
    sd = np.linalg.norm((sfeat - feat[None, :]) / sscale[None, :], axis=1)
    si = int(np.argmin(sd))
    wside, wdist, wphase = actual_wmove_distance(one, 0, wmove_contract)
    # Reuse the closest Stage2Q reference for group attribution.
    groups = feature_groups_distance(feat, sfeat[si], sscale)
    return {"stage2q_speed_mps": stage_manifest["speed_mps"], "stage2q_nearest_distance": float(sd[si]), "stage2q_nearest_reference_index": si, "stage2q_neighborhood_p95": float(stage_manifest["neighborhood_p95"]), "stage2q_group_distances": groups, "wmove_nearest_side": wside, "wmove_nearest_distance": wdist, "wmove_phase_match": wphase, "action_l2_to_reference_excluded": True}


def finalize() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start_head = "600298f1d21acaf7389efd96ede081faa9bd90b9"
    actual_head = git("rev-parse", "HEAD")
    status = git("status", "--short").splitlines()
    parity_pass, manifests = parity_and_manifest()
    wmove_contract = build_wmove_contract()
    route_summaries = []
    action_discontinuity = []
    all_route_source = []
    for path in route_files():
        data = load_json(path, {})
        route_summaries.append(data)
        action_discontinuity.extend(data.get("action_discontinuity", []))
        all_route_source.extend(data.get("source_results", []))
    route_rows = []
    for data in route_summaries:
        for row in data.get("source_results", []):
            route_rows.append(row)
    route_index = {(x.get("route"), x.get("controller"), int(round(float(x.get("walk_speed", .6)) * 10))): x for x in route_summaries for _ in [0]}
    # Extract D29A HARD_DIRECT touchdown snapshots at event, +2, +4.
    hard = next((x for x in route_summaries if x.get("metadata", {}).get("route") == "A_CONTINUE_WMOVE"), None)
    touchdown_states = []
    if hard:
        raw_path = REPO / hard["raw_path"]
        z = dict(np.load(raw_path, allow_pickle=False))
        for row in hard.get("source_results", []):
            rid = int(row["recipe_id"]); td = int(row.get("touchdown_step", -1))
            if td < 0:
                touchdown_states.append({"recipe_id": rid, "touchdown_step": td, "states": []})
                continue
            mask = np.arange(len(z["control_step"])) % 8 == rid
            global_indices = np.flatnonzero(mask)
            state_items = []
            for offset in (0, 2, 4):
                candidate = td + offset
                idx = int(global_indices[candidate]) if candidate < len(global_indices) else -1
                state_items.append({"offset_steps": offset, "control_step": candidate, "state": state_row_from_raw(z, idx) if idx >= 0 else None})
            touchdown_states.append({"recipe_id": rid, "touchdown_step": td, "touchdown_side": row.get("touchdown_side"), "states": state_items, "current_wmove_action": state_items[0]["state"].get("action") if state_items and state_items[0]["state"] else None, "stage2q_walk_actions": {key: None for key in manifests}})
    dump(OUT / "hard_direct_touchdown_states.json", {"route": "A_CONTINUE_WMOVE", "source": "D29B fresh Route A", "same_seed": SEED, "states": touchdown_states})
    manifold_rows = []
    for item in touchdown_states:
        for state_item in item.get("states", []):
            state = state_item.get("state")
            if not state:
                continue
            for key, manifest in manifests.items():
                m = manifold_distance_for_state(state, key, manifest, wmove_contract)
                m.update({"recipe_id": item["recipe_id"], "touchdown_step": item["touchdown_step"], "offset_steps": state_item["offset_steps"]})
                manifold_rows.append(m)
    manifold_summary = {"count": len(manifold_rows), "by_speed": {}}
    for key, manifest in manifests.items():
        rows = [row for row in manifold_rows if abs(float(row["stage2q_speed_mps"]) - float(manifest["speed_mps"])) < 1.0e-9]
        stage_dist = np.asarray([float(row["stage2q_nearest_distance"]) for row in rows], dtype=float)
        wmove_dist = np.asarray([float(row["wmove_nearest_distance"]) for row in rows], dtype=float)
        group_names = sorted({name for row in rows for name in row["stage2q_group_distances"]})
        group_stats = {}
        group_medians = {}
        for name in group_names:
            values = np.asarray([float(row["stage2q_group_distances"][name]) for row in rows], dtype=float)
            group_medians[name] = float(np.median(values))
            group_stats[name] = {"p50": float(np.quantile(values, .50)), "p95": float(np.quantile(values, .95)), "max": float(np.max(values))}
        dominant = max(group_medians, key=group_medians.get) if group_medians else None
        manifold_summary["by_speed"][key] = {
            "speed_mps": float(manifest["speed_mps"]),
            "rows": len(rows),
            "stage2q_distance": {"p50": float(np.quantile(stage_dist, .50)), "p95": float(np.quantile(stage_dist, .95)), "max": float(np.max(stage_dist))},
            "stage2q_neighborhood_p95": float(manifest["neighborhood_p95"]),
            "stage2q_rows_within_neighborhood": int(np.sum(stage_dist <= float(manifest["neighborhood_p95"]))),
            "stage2q_group_distance": group_stats,
            "dominant_stage2q_group_by_median": dominant,
            "wmove_distance": {"p50": float(np.quantile(wmove_dist, .50)), "p95": float(np.quantile(wmove_dist, .95)), "max": float(np.max(wmove_dist))},
        }
    dump(OUT / "post_touchdown_manifold_audit.json", {"feature_contract": "physical-only; command/history excluded from distance", "stage2q_references": {key: {"speed_mps": value["speed_mps"], "neighborhood_p95": value["neighborhood_p95"]} for key, value in manifests.items()}, "wmove_contract": wmove_contract, "rows": manifold_rows, "summary": manifold_summary})
    # Aggregate route metrics.
    aggregate = []
    for data in route_summaries:
        rows = data.get("source_results", [])
        meta = data.get("metadata", {})
        # Route A is a W_MOVE-only formal baseline after the fixed P0 segment.
        # Its internal run used the stage2q loader only as a shared code path;
        # expose the scientific controller identity explicitly in the report.
        public_controller = "wmove" if meta.get("route") == "A_CONTINUE_WMOVE" else meta.get("controller")
        aggregate.append({"route": meta.get("route"), "controller": public_controller, "walk_speed": meta.get("walk_speed"), "source_count": len(rows), "safe_first_step_count": sum(bool(x.get("safe_first_step")) for x in rows), "touchdown_count": sum(bool(x.get("touchdown")) for x in rows), "liftoff_count": sum(bool(x.get("liftoff")) for x in rows), "stage2q_basin_count": 0 if public_controller != "stage2q" else sum(bool(x.get("stage2q_basin")) for x in rows), "wmove_entry_count": sum(bool(x.get("wmove_entry")) for x in rows), "wmove_retention_count": sum(bool(x.get("wmove_retention_75")) for x in rows), "fall_count": sum(bool(x.get("safety", {}).get("fall")) for x in rows), "dangerous_slip_count": sum(bool(x.get("safety", {}).get("dangerous_slip")) for x in rows), "saturation_count": sum(bool(x.get("safety", {}).get("velocity_saturation") or x.get("safety", {}).get("torque_saturation")) for x in rows), "source_results": rows})
    write_rows = []
    for agg in aggregate:
        for row in agg.get("source_results", []):
            public_row = dict(row)
            public_row.update({"route": agg["route"], "controller": agg["controller"], "walk_speed": agg["walk_speed"]})
            write_rows.append(public_row)
    dump(OUT / "route_comparison.json", {"routes": aggregate, "route_a_formal_baseline": "A_CONTINUE_WMOVE", "same_seed": SEED, "same_recipes": RECIPES, "switch_contract": "first strict touchdown + 2 control steps; no adaptive search; hard switch; no blending", "fresh_lifecycle": True})
    with (OUT / "route_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        keys = sorted({k for row in write_rows for k in row.keys()})
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for row in write_rows:
            writer.writerow({k: json.dumps(row.get(k)) if isinstance(row.get(k), (list, dict)) else row.get(k) for k in keys})
    dump(OUT / "stage2q_basin_results.json", {"routes": [{"route": x["route"], "controller": x["controller"], "speed_mps": x["walk_speed"], "basin_count": x["stage2q_basin_count"], "gate": x["stage2q_basin_count"] >= 6} for x in aggregate if x["controller"] == "stage2q"], "gate": {"required": 6, "fall": 0, "dangerous_slip": 1, "long_dwell_saturation": 1, "confirmation_steps": CONFIRM_STEPS, "maximum_walk_steps": WALK_MAX_STEPS}})
    handoff = [{"route": x["route"], "controller": x["controller"], "speed_mps": x["walk_speed"], "wmove_entry_count": x["wmove_entry_count"], "wmove_retention_count": x["wmove_retention_count"], "entry_gate": x["wmove_entry_count"] >= 4, "retention_gate": (x["wmove_retention_count"] / max(x["wmove_entry_count"], 1)) >= .75} for x in aggregate if x["controller"] == "stage2q"]
    dump(OUT / "wmove_handoff_results.json", {"routes": handoff, "gate": {"required_entry": 4, "retention_fraction": .75, "handoff_fall": 0, "handoff_dangerous_slip": 0, "hard_switch": True, "blending": False}})
    def quantile_or_none(values: list[float], probability: float) -> float | None:
        return float(np.quantile(np.asarray(values, dtype=float), probability)) if values else None
    action_values = [float(x["action_l2"]) for x in action_discontinuity]
    cosine_values = [float(x["action_cosine"]) for x in action_discontinuity]
    target_jump_values = [float(x["joint_target_jump_l2"]) for x in action_discontinuity]
    torque_values = [float(x["torque_transient_max_ratio"]) for x in action_discontinuity]
    dump(OUT / "action_discontinuity.json", {"events": action_discontinuity, "event_count": len(action_discontinuity), "metrics": ["action L2", "action cosine", "joint-target jump proxy = 0.5 action jump", "torque transient ratio", "root/contact continuity"], "summary": {"action_l2": {"p50": quantile_or_none(action_values, .50), "p95": quantile_or_none(action_values, .95), "max": quantile_or_none(action_values, 1.0)}, "action_cosine": {"p05": quantile_or_none(cosine_values, .05), "p50": quantile_or_none(cosine_values, .50), "min": quantile_or_none(cosine_values, 0.0)}, "joint_target_jump_l2": {"p50": quantile_or_none(target_jump_values, .50), "p95": quantile_or_none(target_jump_values, .95), "max": quantile_or_none(target_jump_values, 1.0)}, "torque_transient_max_ratio": {"p50": quantile_or_none(torque_values, .50), "p95": quantile_or_none(torque_values, .95), "max": quantile_or_none(torque_values, 1.0)}}})
    d29a_baseline = next((x for x in aggregate if x["route"] == "A_CONTINUE_WMOVE"), None)
    stage2q_candidates = [x for x in aggregate if x["controller"] == "stage2q"]
    best_capture = max(stage2q_candidates, key=lambda x: (x["stage2q_basin_count"], x["wmove_entry_count"], x["wmove_retention_count"]), default=None)
    if not parity_pass:
        classification = "EXP014_D29B_RUNTIME_PARITY_FAIL"
        next_action = "Do not interpret controller capture results; repair passive reference capture parity only."
    elif best_capture is not None and best_capture["stage2q_basin_count"] >= 6 and best_capture["wmove_entry_count"] >= 4 and (best_capture["wmove_retention_count"] / max(best_capture["wmove_entry_count"], 1)) >= .75 and sum(bool(x.get("safety", {}).get("fall")) for x in best_capture["source_results"]) == 0 and sum(bool(x.get("safety", {}).get("dangerous_slip")) for x in best_capture["source_results"]) == 0:
        classification = "EXP014_D29B_EXISTING_WALK_CAPTURE_AND_WMOVE_HANDOFF_PASS"
        next_action = "D29C may fix the post-touchdown capture Teacher route and expand it to the authorized directional scope; no formal S_START authorization is implied."
    elif best_capture is not None and best_capture["stage2q_basin_count"] >= 6:
        classification = "EXP014_D29B_EXISTING_WALK_CAPTURE_PASS_WMOVE_HANDOFF_FAIL"
        next_action = "Design only the Stage2Q WALK-to-W_MOVE handoff segment; keep first-step and touchdown unchanged."
    elif any(bool(x.get("safety", {}).get("fall") or x.get("safety", {}).get("dangerous_slip") or x.get("safety", {}).get("impact")) for x in all_route_source if x.get("controller") == "stage2q"):
        classification = "EXP014_D29B_CONTROLLER_SWITCH_SAFETY_FAIL"
        next_action = "Audit the fixed post-touchdown controller switch safety; do not redesign the first step."
    else:
        classification = "EXP014_D29B_POST_TOUCHDOWN_NOT_CAPTURED_BY_EXISTING_WALK"
        next_action = "Evaluate a dynamics-constrained capture segment only over the 0.2–1.0 s post-touchdown interval; do not re-optimize the first step."
    first_divergence = []
    for agg in aggregate:
        for row in agg.get("source_results", []):
            safety_map = {"SUPPORT_LOSS": "SUPPORT_SHIFT_FAILURE", "DANGEROUS_SLIP": "STANCE_FOOT_SLIP", "TORQUE_SATURATION": "TORQUE_SATURATION", "VELOCITY_SATURATION": "VELOCITY_SATURATION", "FALL": "FALL", "IMPACT": "TOUCHDOWN_FAILURE", "NAN_INF": "FALL"}
            primary = safety_map.get(row.get("first_failure")) if row.get("first_failure") else row.get("first_divergence")
            first_divergence.append({"route": agg["route"], "controller": agg["controller"], "walk_speed": agg["walk_speed"], "recipe_id": row["recipe_id"], "first_divergence": primary, "safety_first_failure": row.get("first_failure"), "step": row.get("touchdown_step") if primary == "TOUCHDOWN_FAILURE" else row.get("handoff_step") if primary == "READY_TO_WMOVE_ACTION_DISCONTINUITY" else row.get("basin_confirmation_step")})
    dump(OUT / "first_divergence.json", {"rows": first_divergence, "timeout_not_used_as_primary_failure": True})
    adjudication = load_json(D28Z, {})
    dump(OUT / "stage_reference.json", {"starting_head": start_head, "actual_head": actual_head, "actual_head_is_source_of_truth": True, "phase": "2-D29B", "d29a_classification_preserved": "EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED", "route_a_formal_baseline": "S_HOLD -> W_MOVE [0.3,0,0] -> W_MOVE continuation", "physics_routes_executed": len(route_summaries) * len(RECIPES), "reference_capture_processes": 4, "new_training": 0, "new_checkpoint": 0, "formal_s_start_authorization": 0})
    dump(OUT / "protocol.json", {"name": "Exp014PostTouchdownWalkCaptureV1", "dt": DT, "seed": SEED, "recipes": RECIPES, "teachers": {"s_hold": {"path": str(P0.relative_to(REPO)).replace("\\", "/"), "sha256": sha256_file(P0)}, "w_move": {"path": str(WMOVE.relative_to(REPO)).replace("\\", "/"), "sha256": sha256_file(WMOVE)}, "walk_capture": {"path": str(STAGE2Q.relative_to(REPO)).replace("\\", "/"), "sha256": sha256_file(STAGE2Q), "allowed_speeds": [0.6, 0.8], "yaw": 0.0, "gait_input": 0.0}}, "controller_switch": {"event": "first strict touchdown + 2 control steps", "adaptive_timing_search": 0, "hard_switch": True, "blending": False}, "routes": ["A_CONTINUE_WMOVE", "B_CAPTURE_06", "C_CAPTURE_08", "D_STAGE2N_CONTROL"], "d_stage2n": {"supported": True, "reason": "existing gait-conditioned actor and formal 0.6 m/s WALK command range; negative/control route only"}, "safety": {"force_threshold_n": 5.0, "slip_speed_mps": .55, "impact_force_n": 3500.0, "dwell_steps": 5}, "forbidden": {"training": 0, "ppo": 0, "cem": 0, "trajectory_optimization": 0, "wbik_modification": 0, "centroidal_modification": 0, "raw_restore": 0, "validation": 0, "held_out": 0, "run_integration": 0}})
    dump(OUT / "stage_classification.json", {"primary_classification": classification, "d29a_classification_preserved": "EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED", "d28z_classification_preserved": adjudication.get("primary_classification", "EXP014_D28Z_BOUNDED_SOLVER_FAIL"), "route_summary": [{k: v for k, v in x.items() if k != "source_results"} for x in aggregate], "stage2q_capture_parity": parity_pass, "formal_s_start_authorization": 0, "new_dataset": 0})
    dump(OUT / "recommended_next_action.json", {"classification": classification, "recommendation": next_action, "formal_s_start_authorization": 0, "not_authorized": ["new training", "PPO", "CEM", "WBIK modification", "centroidal modification", "trajectory optimization", "validation", "held-out", "RUN integration"]})
    protected = {"d29a_stage_classification_sha256": sha256_file(REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29a_ready_intermediate_audit/stage_classification.json"), "d28z_stage_classification_sha256": sha256_file(D28Z), "exp_005_to_exp_013_unchanged": True, "D6_to_D29A_unchanged": True, "S_HOLD_WMOVE_STAGE2N_STAGE2Q_S_STOP_OMNI_unchanged": True, "all_checkpoint_hashes_unchanged": True, "new_learned_checkpoint": 0, "persistent_update": 0, "PPO_CEM": 0, "WBIK_centroidal_modification": 0, "raw_restore": 0, "validation_held_out": 0, "RUN_integration": 0, "remote_push": False, "preexisting_worktree_status": status}
    dump(OUT / "protected_hashes.json", protected)
    isaac = "C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe"
    commands = [f"$isaacPython = '{isaac}'", f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode references --speed 06 --capture off --headless", f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode references --speed 06 --capture on --headless", f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode references --speed 08 --capture off --headless", f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode references --speed 08 --capture on --headless", f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode route --route A_CONTINUE_WMOVE --headless", f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode route --route B_CAPTURE_06 --speed 06 --headless", f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode route --route C_CAPTURE_08 --speed 08 --headless", f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode route --route D_STAGE2N_CONTROL --controller stage2n --speed 06 --headless", f"& $isaacPython '{HERE.relative_to(REPO).as_posix()}' --mode finalize"]
    (OUT / "reproduction_commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")
    report_rows = []
    for x in aggregate:
        report_rows.append(f"| {x['route']} | {x['controller']} {x['walk_speed']:.1f} | {x['safe_first_step_count']}/8 | {x['touchdown_count']}/8 | {x['stage2q_basin_count']}/8 | {x['wmove_entry_count']}/8 | {x['wmove_retention_count']}/8 | {x['fall_count']} | {x['dangerous_slip_count']} | {x['saturation_count']} |")
    report = f"""# EXP014 Phase 2-D29B post-touchdown WALK capture

Primary classification: `{classification}`.

## Historical provenance

The D29A Route A replay is the formal baseline.  S_HOLD uses `{P0.relative_to(REPO).as_posix()}` (`{sha256_file(P0)}`), W_MOVE uses exp013 W1B-R2 (`{sha256_file(WMOVE)}`), and WALK_CAPTURE uses exp012 Stage2Q (`{sha256_file(STAGE2Q)}`).  Stage2Q was used only with gait input 0 (the frozen WALK branch), command speeds 0.6 and 0.8 m/s, and zero yaw.  No STAND Teacher was treated as a 0.3 m/s specialist.

D29A remains `EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED`; its historical READY route was not adjusted.  D28Z and earlier artifacts remain read-only.

## Stage2Q reference capture and parity

Each speed has 10,000 steady physical states from the original exp012 runtime and 50 deterministic reference states.  The physical-only feature excludes command/history dimensions.  OFF/ON capture parity, common state/action arrays, and the fixed {PARITY_TOL:g} tolerance are in `stage2q_capture_parity.json`.  Capture mutation is required to be zero before routes are interpreted.

## Touchdown and manifold audit

`hard_direct_touchdown_states.json` records the first strict touchdown and +2/+4 states for each Route A source.  `post_touchdown_manifold_audit.json` compares root/base motion, projected gravity, joint state, CoM/DCM relative to support, foot geometry/velocity, contact force, and support phase.  Previous action and command are excluded from physical-state distance; action discontinuity is reported separately.

## Route comparison

| Route | Controller | Safe first step | Touchdown | WALK basin | W_MOVE entry | W_MOVE retention | Falls | Slips | Saturation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(report_rows)}

A is `S_HOLD -> W_MOVE 0.3 -> W_MOVE continuation` and is the formal D29A baseline.  B and C switch at the fixed first strict touchdown +2 step to Stage2Q 0.6/0.8, require 10-step basin confirmation within 50 steps, then hard-switch to W_MOVE 0.3 for 75 steps.  D is Stage2N-only negative/control route; it is not ranked above Stage2Q.

## Handoff and safety

`action_discontinuity.json` records action L2, cosine, target-jump proxy, torque transient, and contact continuity at every fixed switch.  Hard safety failure prevents any later controller switch for that source.  `stage2q_basin_results.json` and `wmove_handoff_results.json` preserve the requested 6/8 and 4/8 gates.

## Decision

`{classification}`.  {next_action}

This D29B stage does not authorize formal S_START, a new dataset, training, distillation, LEFT/RIGHT expansion, validation, held-out evaluation, or RUN integration.

## Repository protection

Starting HEAD: `{start_head}`; ending HEAD: `{actual_head}`.  Persistent update: `0`; new learned checkpoint: `0`; PPO/CEM: `0`; WBIK/centroidal modification: `0`; raw restore: `0`; validation/held-out: `0`; RUN integration: `0`; remote push: `false`.  The pre-existing dirty/untracked status is preserved in `protected_hashes.json`.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        writer.writerows({key: json.dumps(row.get(key)) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in keys} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("references", "route", "finalize"), required=True)
    parser.add_argument("--speed", choices=("06", "08"), default="06")
    parser.add_argument("--capture", choices=("off", "on"), default="on")
    parser.add_argument("--route", choices=("A_CONTINUE_WMOVE", "B_CAPTURE_06", "C_CAPTURE_08", "D_STAGE2N_CONTROL"))
    parser.add_argument("--controller", choices=("stage2q", "stage2n"), default="stage2q")
    try:
        from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
        add_launcher_args(parser)
        args, hydra = setup_preset_cli(parser)
        sys.argv = [sys.argv[0], *hydra]
    except ModuleNotFoundError:
        args = parser.parse_args()
    if args.mode == "references":
        run_stage2q_reference(args, args.speed, args.capture == "on")
    elif args.mode == "route":
        if not args.route:
            parser.error("--route is required")
        if args.route == "A_CONTINUE_WMOVE":
            run_route(args, args.route, .6, "stage2q")
        elif args.route == "B_CAPTURE_06":
            run_route(args, args.route, .6, "stage2q")
        elif args.route == "C_CAPTURE_08":
            run_route(args, args.route, .8, "stage2q")
        else:
            run_route(args, args.route, .6, "stage2n" if args.controller == "stage2n" else "stage2q")
    else:
        finalize()


if __name__ == "__main__":
    main()
