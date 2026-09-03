"""Phase 2-D26U fresh S_HOLD identity-complete source capture.

This runner deliberately keeps the D26U capture separate from D26T.  It uses
the frozen Exp013 reset-recipe lifecycle and the frozen P0 S_HOLD parent.  The
capture hook only clones state to CPU; it does not write simulator, actuator,
command, observation, or lifecycle state.  Two fresh passes (capture OFF and
capture ON) are compared before the identity-complete bundle is persisted.

No raw snapshot restore, policy update, physics START, PPO, CEM, validation,
held-out, RUN integration, or checkpoint write is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26u_fresh_source_and_offline_execution"

DT = 0.02
# Exp014FreshS_HOLDSourceLifecycleV2 uses the fixed D24D train-only seed.
SEED = 20279941
RECIPES = list(range(8))
N_ENVS = 16
CONFIRMATION_STEPS = 50
ADDITIONAL_HOLD_STEPS = 50
MAX_STEPS = 320


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# D3 supplies the frozen reset-recipe lifecycle and local task registration.
d3 = load_module("exp014_d3_runtime_d26u", HERE.parent / "run_phase2_d3.py")

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import (  # noqa: E402
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def tensor_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous()


def numpy_value(value) -> np.ndarray:
    if torch.is_tensor(value) or (hasattr(value, "detach") and hasattr(value, "cpu")):
        return value.detach().cpu().contiguous().numpy()
    return np.asarray(value)


def hash_value(digest: "hashlib._Hash", name: str, value) -> None:
    digest.update(name.encode("utf-8"))
    array = numpy_value(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())


def hash_values(values: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        hash_value(digest, name, values[name])
    return digest.hexdigest()


def as_batch(value, n: int, dtype=torch.float32) -> torch.Tensor:
    """Convert a runtime value to an [N,...] tensor without changing it."""
    if value is None:
        raise RuntimeError("missing runtime value")
    result = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=dtype)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.shape[0] == n:
        return result
    if result.shape[0] == 1:
        return result.expand((n,) + tuple(result.shape[1:]))
    raise RuntimeError(f"unexpected batch shape {tuple(result.shape)} for N={n}")


def get_jacobians(world) -> torch.Tensor:
    raw = world.robot.root_physx_view.get_jacobians()
    if torch.is_tensor(raw):
        result = raw
    else:
        import warp as wp

        result = wp.to_torch(raw)
    result = result.detach()
    if result.ndim != 4 or tuple(result.shape[1:]) != (44, 6, 43):
        raise RuntimeError(f"unexpected PhysX Jacobian shape {tuple(result.shape)}")
    return result


def com_from_body(data, masses: torch.Tensor, n: int):
    body_com = data.body_com_pos_w[:n]
    body_com_velocity = data.body_com_lin_vel_w[:n]
    mass = masses[:n].to(body_com.device, dtype=body_com.dtype)
    total = mass.sum(dim=1).clamp_min(1.0e-9)
    com = (body_com * mass[..., None]).sum(dim=1) / total[:, None]
    com_velocity = (body_com_velocity * mass[..., None]).sum(dim=1) / total[:, None]
    return com, com_velocity, total


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    result = q.clone()
    result[..., :3] = -result[..., :3]
    return result


def quat_rotate_inverse(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    # IsaacLab uses xyzw quaternions in the frozen Exp013 contract.
    qv = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    qq = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    result = quat_multiply(quat_multiply(quat_conjugate(qq), qv), qq)
    return result[..., 1:]


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def action_actor(world, actor, obs: torch.Tensor):
    with torch.inference_mode():
        action = actor.mean(obs)
    return action


def runtime_arrays(world, obs: torch.Tensor, action: torch.Tensor, previous_action: torch.Tensor, masses: torch.Tensor, streaks: dict[str, torch.Tensor], flags: dict[str, torch.Tensor], done: torch.Tensor | None = None, timeout: torch.Tensor | None = None, update_safety: bool = True) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Read-only state/metric extraction at a post-physics state."""
    data = world.robot.data
    n = world.env.num_envs
    root_pos = data.root_pos_w[:n]
    root_quat = data.root_quat_w[:n]
    root_lin_vel = data.root_lin_vel_w[:n]
    root_ang_vel = data.root_ang_vel_w[:n]
    joint_pos = data.joint_pos[:n]
    joint_vel = data.joint_vel[:n]
    body_pos = data.body_pos_w[:n]
    body_quat = data.body_quat_w[:n]
    body_lin_vel = data.body_lin_vel_w[:n]
    body_ang_vel = data.body_ang_vel_w[:n]
    root_pose = torch.cat((root_pos, root_quat), dim=1)
    root_velocity = torch.cat((root_lin_vel, root_ang_vel), dim=1)
    com, com_velocity, total_mass = com_from_body(data, masses, n)
    omega = torch.sqrt(torch.tensor(9.81, device=com.device, dtype=com.dtype) / com[:, 2].clamp_min(0.1))
    dcm = com[:, :2] + com_velocity[:, :2] / omega[:, None]
    com_root = quat_rotate_inverse(root_quat, com - root_pos)
    com_velocity_root = quat_rotate_inverse(root_quat, com_velocity)
    dcm_root = quat_rotate_inverse(root_quat, torch.cat((dcm, com[:, 2:3]), dim=1))[:, :2]

    force_history = world.sensor.data.net_forces_w_history[:, :, world.sf, :]
    force = force_history[:, -1]
    contact = force.norm(dim=-1) > 5.0
    foot_velocity = body_lin_vel[:, world.rf]
    tangential_speed = foot_velocity[..., :2].norm(dim=-1)
    slip_now = (contact & (tangential_speed > 0.55)).any(dim=1)
    slip_streak = torch.where(slip_now, streaks["slip"] + 1, torch.zeros_like(streaks["slip"]))
    dangerous_slip_now = slip_streak >= 5

    impact_now = force.abs().amax(dim=(1, 2)) > 3500.0
    joint_velocity_limits = data.joint_vel_limits[:n]
    if joint_velocity_limits.ndim == 3:
        joint_velocity_limits = joint_velocity_limits[..., 1].abs()
    effort_limits = data.joint_effort_limits[:n].abs().clamp_min(1.0e-6)
    applied_torque = getattr(data, "applied_torque", torch.zeros_like(joint_pos))[:n]
    computed_torque = getattr(data, "computed_torque", applied_torque)[:n]
    velocity_ratio = joint_vel.abs() / joint_velocity_limits.abs().clamp_min(1.0e-6)
    torque_ratio = applied_torque.abs() / effort_limits
    velocity_now = velocity_ratio.amax(dim=1) > 0.95
    torque_now = torque_ratio.amax(dim=1) > 0.95
    velocity_streak = torch.where(velocity_now, streaks["velocity"] + 1, torch.zeros_like(streaks["velocity"]))
    torque_streak = torch.where(torque_now, streaks["torque"] + 1, torch.zeros_like(streaks["torque"]))
    velocity_saturation_now = velocity_streak >= 5
    torque_saturation_now = torque_streak >= 5
    support_loss_now = ~contact.any(dim=1)
    support_streak = torch.where(support_loss_now, streaks["support"] + 1, torch.zeros_like(streaks["support"]))
    support_loss_now = support_streak >= 5

    finite_candidates = [obs, action, previous_action, root_pose, root_velocity, joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel, com, com_velocity, dcm, applied_torque, computed_torque]
    finite = torch.ones(n, dtype=torch.bool, device=obs.device)
    for value in finite_candidates:
        finite &= torch.isfinite(value.reshape(n, -1)).all(dim=1)
    nonfinite_now = ~finite
    if done is None:
        done = torch.zeros(n, dtype=torch.bool, device=obs.device)
    if timeout is None:
        timeout = torch.zeros_like(done)
    fall_now = done & ~timeout

    if update_safety:
        streaks["slip"] = slip_streak
        streaks["velocity"] = velocity_streak
        streaks["torque"] = torque_streak
        streaks["support"] = support_streak
        for name, value in (("fall", fall_now), ("dangerous_slip", dangerous_slip_now), ("impact", impact_now), ("velocity_saturation", velocity_saturation_now), ("torque_saturation", torque_saturation_now), ("support_loss", support_loss_now), ("nonfinite", nonfinite_now)):
            flags[name] |= value

    metrics = {
        "fall": fall_now,
        "dangerous_slip": dangerous_slip_now,
        "impact": impact_now,
        "velocity_saturation": velocity_saturation_now,
        "torque_saturation": torque_saturation_now,
        "support_loss": support_loss_now,
        "nonfinite": nonfinite_now,
        "contact": contact,
        "force": force,
        "velocity_ratio": velocity_ratio,
        "torque_ratio": torque_ratio,
        "support_count": contact.sum(dim=1),
    }

    state = {
        "obs_123": world.env.observation_manager.compute()["policy"],
        "obs_141": obs,
        "obs_143_compatible": torch.cat((obs, torch.zeros((n, 2), device=obs.device, dtype=obs.dtype)), dim=1),
        "root_pose": root_pose,
        "root_velocity": root_velocity,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "current_action": action,
        "previous_action": previous_action,
        "command": world.state.physical_command,
        "previous_command": world.state.previous_physical_command,
        "mode": world.state.target_mode,
        "previous_mode": world.state.previous_target_mode,
        "time_since_mode_change": world.state.time_since_mode_change_s,
        "ramp_progress": world.state.ramp_progress,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
        "left_right_foot_pose": torch.cat((body_pos[:, world.rf], body_quat[:, world.rf]), dim=-1),
        "foot_velocity": body_lin_vel[:, world.rf],
        "contact_force": force,
        "contact_force_history": force_history,
        "current_air_time": (getattr(world.sensor.data, "current_air_time", None)[:, world.sf] if getattr(world.sensor.data, "current_air_time", None) is not None else torch.zeros((n, len(world.sf)), device=obs.device)),
        "last_air_time": (getattr(world.sensor.data, "last_air_time", None)[:, world.sf] if getattr(world.sensor.data, "last_air_time", None) is not None else torch.zeros((n, len(world.sf)), device=obs.device)),
        "support_state": contact.to(torch.int8),
        "support_count": metrics["support_count"],
        "com_position_w": com,
        "com_position_root": com_root,
        "com_velocity_w": com_velocity,
        "com_velocity_root": com_velocity_root,
        "dcm": dcm,
        "dcm_root": dcm_root,
        "dcm_offset": dcm - com[:, :2],
        "body_jacobians": get_jacobians(world),
        "body_com_pos_w": data.body_com_pos_w[:n],
        "body_com_lin_vel_w": data.body_com_lin_vel_w[:n],
        "body_masses": masses,
        "computed_torque": computed_torque[:n],
        "applied_torque": applied_torque,
        "effort_limits": data.joint_effort_limits[:n],
        "joint_velocity_limits": joint_velocity_limits,
        "joint_position_limits": data.soft_joint_pos_limits[:n],
        "joint_velocity_ratio": velocity_ratio,
        "torque_ratio": torque_ratio,
        "joint_limit_margin": torch.minimum(joint_pos - data.soft_joint_pos_limits[:n, ..., 0], data.soft_joint_pos_limits[:n, ..., 1] - joint_pos),
        "joint_velocity_margin": 0.8 * joint_velocity_limits.abs() - joint_vel.abs(),
        "action_margin": 1.0 - action.abs(),
        "termination_done": done,
        "termination_timeout": timeout,
        "safety_fall": flags["fall"],
        "safety_dangerous_slip": flags["dangerous_slip"],
        "safety_impact": flags["impact"],
        "safety_velocity_saturation": flags["velocity_saturation"],
        "safety_torque_saturation": flags["torque_saturation"],
        "safety_support_loss": flags["support_loss"],
        "safety_nonfinite": flags["nonfinite"],
        "total_mass": total_mass,
    }
    return state, metrics


def clone_payload(state: dict[str, object], next_action: torch.Tensor, recipe_ids: list[int], source_step: int) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for name, value in state.items():
        if torch.is_tensor(value):
            payload[name] = tensor_cpu(value).numpy()
        else:
            payload[name] = numpy_value(value)
    payload["next_action"] = tensor_cpu(next_action).numpy()
    payload["recipe_id"] = np.asarray(recipe_ids, dtype=np.int64)
    payload["control_step"] = np.full((len(recipe_ids),), source_step, dtype=np.int64)
    return payload


def state_hash_input(state: dict[str, object], action: torch.Tensor, previous_action: torch.Tensor) -> dict[str, object]:
    excluded = {"body_jacobians", "body_com_pos_w", "body_com_lin_vel_w", "body_masses", "joint_position_limits", "joint_velocity_limits"}
    values = {name: value for name, value in state.items() if name not in excluded}
    values["action_for_hash"] = action
    values["previous_action_for_hash"] = previous_action
    return values


def per_environment(values: dict[str, object], index: int, n: int) -> dict[str, object]:
    result = {}
    for name, value in values.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n:
            result[name] = value[index]
        else:
            result[name] = value
    return result


def localize_hash_values(values: dict[str, object], origins: torch.Tensor) -> dict[str, object]:
    """Remove only the deterministic replicated-scene environment origin.

    The paired OFF/ON replicas intentionally occupy different IsaacLab
    environment indices.  Their world positions therefore differ by the
    known environment origin even when their physical state is identical.
    The durable bundle still stores world coordinates; this normalization is
    used only for the parity hash.
    """
    result = dict(values)
    for name in ("root_pose", "body_pos_w", "left_right_foot_pose", "com_position_w", "body_com_pos_w"):
        if name in result and torch.is_tensor(result[name]) and result[name].ndim >= 2:
            value = result[name].clone()
            if value.ndim == 2:
                value[..., :3] -= origins
            elif value.ndim == 3:
                value[..., :3] -= origins[:, None, :]
            else:
                value[..., :3] -= origins[:, None, None, :]
            result[name] = value
    if "dcm" in result and torch.is_tensor(result["dcm"]):
        value = result["dcm"].clone()
        value[..., :2] -= origins[:, :2]
        result["dcm"] = value
    return result


def evaluate_pass(world, actor, masses: torch.Tensor, recipe_ids: list[int], capture_enabled: bool | set[int]) -> tuple[dict, dict[str, np.ndarray] | None]:
    recipes_tensor = torch.as_tensor(recipe_ids, dtype=torch.long, device=world.device)
    obs = world.restore(recipes_tensor)
    n = len(recipe_ids)
    capture_indices = set(range(n)) if capture_enabled is True else (set(capture_enabled) if isinstance(capture_enabled, set) else set())
    flags = {name: torch.zeros(n, dtype=torch.bool, device=world.device) for name in ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nonfinite")}
    streaks = {name: torch.zeros(n, dtype=torch.long, device=world.device) for name in ("slip", "velocity", "torque", "support")}
    confirmation = torch.zeros(n, dtype=torch.long, device=world.device)
    confirmation_end = torch.full((n,), -1, dtype=torch.long, device=world.device)
    source_step_tensor = torch.full((n,), -1, dtype=torch.long, device=world.device)
    source_payload_by_env: dict[int, dict[str, np.ndarray]] = {}
    prephysics_hashes: list[list[str]] = [[] for _ in range(n)]
    postphysics_hashes: list[list[str]] = [[] for _ in range(n)]
    action_hashes: list[list[str]] = [[] for _ in range(n)]
    metrics_rows = []
    first_safety_flag_step = {name: torch.full((n,), -1, dtype=torch.long, device=world.device) for name in flags}
    source_state_hashes = [None for _ in range(n)]
    source_captured = torch.zeros(n, dtype=torch.bool, device=world.device)

    for step in range(MAX_STEPS):
        base = world.env.observation_manager.compute()["policy"]
        obs = d3.build_observation_141(base, world.state)
        action = action_actor(world, actor, obs)
        previous_action = world.env.action_manager.prev_action.clone()
        # This is a diagnostic hash only; both OFF and ON passes execute it.
        pre_state, _ = runtime_arrays(world, obs, action, previous_action, masses, streaks, flags, update_safety=False)
        origins = world.env.scene.env_origins[:n]
        pre_hash_values = localize_hash_values(state_hash_input(pre_state, action, previous_action), origins)
        _, _, done, extras = world.step(action, None)
        timeout = extras.get("time_outs", torch.zeros_like(done)).bool()[:n]
        post_obs = world.obs()
        post_state, metrics = runtime_arrays(world, post_obs, action, previous_action, masses, streaks, flags, done[:n].bool(), timeout, update_safety=True)
        for name, values in flags.items():
            first = (values & (first_safety_flag_step[name] < 0))
            first_safety_flag_step[name][first] = step + 1
        speed = world.robot.data.root_lin_vel_b[:n, :2].norm(dim=1)
        yaw = world.robot.data.root_ang_vel_b[:n, 2].abs()
        # RESET_TO_STAND acquisition is the frozen D24/D5 evaluator's
        # continuous speed/yaw window.  Safety is evaluated over the same
        # lifecycle and is a separate source-validity gate below.
        good = (speed <= 0.08) & (yaw <= 0.08)
        confirmation = torch.where(good, confirmation + 1, torch.zeros_like(confirmation))
        just_confirmed = (confirmation >= CONFIRMATION_STEPS) & (confirmation_end < 0)
        confirmation_end[just_confirmed] = step + 1
        ready_for_source = (confirmation_end >= 0) & ((step + 1) >= confirmation_end + ADDITIONAL_HOLD_STEPS) & ~source_captured

        for i in range(n):
            prephysics_hashes[i].append(hash_values(per_environment(pre_hash_values, i, n)))
            post_hash_values = localize_hash_values(state_hash_input(post_state, action, previous_action), origins)
            postphysics_hashes[i].append(hash_values(per_environment(post_hash_values, i, n)))
            action_hashes[i].append(hash_values({"action": action[i], "previous_action": previous_action[i]}))
        metrics_rows.append({
            "step": step + 1,
            "speed_mean": float(speed.mean()),
            "yaw_mean": float(yaw.mean()),
            "support_count_min": int(metrics["support_count"].min()),
            "fall": int(flags["fall"].sum()),
            "dangerous_slip": int(flags["dangerous_slip"].sum()),
            "impact": int(flags["impact"].sum()),
            "velocity_saturation": int(flags["velocity_saturation"].sum()),
            "torque_saturation": int(flags["torque_saturation"].sum()),
            "support_loss": int(flags["support_loss"].sum()),
            "nonfinite": int(flags["nonfinite"].sum()),
        })

        if ready_for_source.any():
            # Capture the endpoint after the additional 1.0 s hold, before the
            # next action is applied.  The one following step only supplies the
            # explicit next_action field and is included in the parity trace.
            source_step = int(step + 1)
            source_obs = post_obs.clone()
            source_action = action_actor(world, actor, source_obs)
            source_previous = world.env.action_manager.prev_action.clone()
            source_state, _ = runtime_arrays(world, source_obs, source_action, source_previous, masses, streaks, flags, update_safety=False)
            _, _, next_done, next_extras = world.step(source_action, None)
            next_timeout = next_extras.get("time_outs", torch.zeros_like(next_done)).bool()[:n]
            next_obs = world.obs()
            next_action = action_actor(world, actor, next_obs)
            # The extra step is safety-checked but the source fields remain the
            # exact pre-action endpoint captured above.
            runtime_arrays(world, next_obs, next_action, source_action, masses, streaks, flags, next_done[:n].bool(), next_timeout, update_safety=True)
            for i in ready_for_source.nonzero().flatten().tolist():
                source_step_tensor[i] = source_step
                source_hash_values = localize_hash_values(state_hash_input(source_state, source_action, source_previous), origins)
                source_state_hashes[i] = hash_values({**per_environment(source_hash_values, i, n), "next_action": next_action[i]})
                if i in capture_indices:
                    one = {name: (value[i:i + 1] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n else value) for name, value in source_state.items()}
                    one_action = next_action[i:i + 1]
                    one_payload = clone_payload(one, one_action, [recipe_ids[i]], source_step)
                    source_payload_by_env[i] = one_payload
            source_captured |= ready_for_source
            # The extra step advances all environments.  Rebuild observations on
            # the next loop; the source endpoint is already durably cloned.
            obs = next_obs

        if bool(source_captured.all()):
            break

    result = {
        "capture_enabled": bool(capture_indices),
        "capture_indices": sorted(capture_indices),
        "recipe_ids": recipe_ids,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "source_captured": [bool(x) for x in source_captured.detach().cpu().tolist()],
        "confirmation_end_step": confirmation_end.detach().cpu().tolist(),
        "source_control_step": source_step_tensor.detach().cpu().tolist(),
        "source_lifecycle_hash": source_state_hashes,
        "first_safety_flag_step": {name: values.detach().cpu().tolist() for name, values in first_safety_flag_step.items()},
        "prephysics_hashes": [hash_values({"trajectory": np.asarray(v, dtype="U64")}) for v in prephysics_hashes],
        "postphysics_hashes": [hash_values({"trajectory": np.asarray(v, dtype="U64")}) for v in postphysics_hashes],
        "action_trajectory_hashes": [hash_values({"trajectory": np.asarray(v, dtype="U64")}) for v in action_hashes],
        "metrics_tail": metrics_rows[-10:],
        "source_next_action_available": len(source_payload_by_env) == len(capture_indices) if capture_indices else bool(torch.all(source_captured)),
        "persistent_update": 0,
        "physics_start": 0,
    }
    if capture_indices and len(source_payload_by_env) == len(capture_indices):
        source_payload = {}
        for name in source_payload_by_env[min(capture_indices)]:
            source_payload[name] = np.concatenate([source_payload_by_env[i][name] for i in sorted(capture_indices)], axis=0)
    else:
        source_payload = None
    return result, source_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-mode", choices=("paired", "off", "on"), default="paired")
    add_launcher_args(parser)
    args, hydra = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]
    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    n_envs = 16 if args.capture_mode == "paired" else 8
    cfg.scene.num_envs = n_envs
    cfg.seed = SEED
    cfg.episode_length_s = 20.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = agent.device = args.device
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        world = d3.StandWorld(wrapped, d3.load_resets(), torch.zeros(680))
        actor = d3.initialize("P0_STAND_PARENT", world.device)[0].eval()
        masses = world.robot.root_physx_view.get_masses()
        if not torch.is_tensor(masses):
            masses = torch.as_tensor(masses, device=world.device, dtype=torch.float32)
        masses = as_batch(masses.to(world.device), n_envs)

        if args.capture_mode in ("off", "on"):
            result, payload = evaluate_pass(world, actor, masses, RECIPES, capture_enabled=(args.capture_mode == "on"))
            raw_dir = OUT / "raw_d26u_capture"
            raw_dir.mkdir(parents=True, exist_ok=True)
            dump(raw_dir / f"fresh_shold_capture_{args.capture_mode}.json", result)
            if args.capture_mode == "on" and payload is not None:
                np.savez_compressed(raw_dir / "fresh_shold_identity_complete_sources_on.npz", **payload)
                (raw_dir / "fresh_shold_identity_complete_sources_on.sha256").write_text(sha256_file(raw_dir / "fresh_shold_identity_complete_sources_on.npz") + "\n", encoding="ascii")
            print(json.dumps({"capture_mode": args.capture_mode, "source_captured": result["source_captured"], "source_control_step": result["source_control_step"], "payload": payload is not None}, indent=2), flush=True)
            wrapped.close()
            return

        paired_recipes = RECIPES + RECIPES
        paired, payload = evaluate_pass(world, actor, masses, paired_recipes, capture_enabled=set(range(len(RECIPES), 2 * len(RECIPES))))
        off = {"capture_enabled": False, "capture_indices": [], "recipe_ids": RECIPES, "source_captured": paired["source_captured"][:8], "confirmation_end_step": paired["confirmation_end_step"][:8], "source_control_step": paired["source_control_step"][:8], "source_lifecycle_hash": paired["source_lifecycle_hash"][:8], "prephysics_hashes": paired["prephysics_hashes"][:8], "postphysics_hashes": paired["postphysics_hashes"][:8], "action_trajectory_hashes": paired["action_trajectory_hashes"][:8], "metrics_tail": paired["metrics_tail"], "persistent_update": 0, "physics_start": 0}
        on = {"capture_enabled": True, "capture_indices": list(range(8, 16)), "recipe_ids": RECIPES, "source_captured": paired["source_captured"][8:], "confirmation_end_step": paired["confirmation_end_step"][8:], "source_control_step": paired["source_control_step"][8:], "source_lifecycle_hash": paired["source_lifecycle_hash"][8:], "prephysics_hashes": paired["prephysics_hashes"][8:], "postphysics_hashes": paired["postphysics_hashes"][8:], "action_trajectory_hashes": paired["action_trajectory_hashes"][8:], "metrics_tail": paired["metrics_tail"], "persistent_update": 0, "physics_start": 0}
        if payload is None:
            dump(OUT / "fresh_shold_capture_runtime_diagnostics.json", {"status": "FAIL", "capture_off": off, "capture_on": on, "paired": paired, "reason": "not all eight capture-ON source endpoints were captured before MAX_STEPS"})
            dump(OUT / "fresh_shold_capture_parity.json", {"status": "FAIL", "capture_mutation": "NOT_REACHED", "capture_off": off, "capture_on": on, "paired": paired, "fail_closed": True})
            dump(OUT / "stage_classification.json", {"classification": "EXP014_D26U_FRESH_SOURCE_CAPTURE_FAIL", "reason": "not all eight capture-ON source endpoints were captured before MAX_STEPS", "fail_closed": True})
            wrapped.close()
            return

        comparisons = []
        parity_keys = ("prephysics_hashes", "postphysics_hashes", "action_trajectory_hashes", "source_control_step", "source_lifecycle_hash")
        for i, recipe in enumerate(RECIPES):
            equal = {key: paired[key][i] == paired[key][i + len(RECIPES)] for key in parity_keys}
            comparisons.append({"recipe_id": recipe, "capture_off_environment_index": i, "capture_on_environment_index": i + len(RECIPES), "pass": all(equal.values()), **equal})
        parity_pass = all(row["pass"] for row in comparisons)
        if not parity_pass:
            dump(OUT / "fresh_shold_capture_runtime_diagnostics.json", {"status": "PARITY_FAIL", "paired": paired, "comparisons": comparisons, "capture_off": off, "capture_on": on})
            dump(OUT / "fresh_shold_capture_parity.json", {"status": "FAIL", "capture_mutation": "UNRESOLVED", "paired": paired, "comparisons": comparisons, "fail_closed": True})
            dump(OUT / "stage_classification.json", {"classification": "EXP014_D26U_FRESH_SOURCE_CAPTURE_FAIL", "reason": "capture OFF/ON paired identity parity failed", "fail_closed": True})
            wrapped.close()
            return

        OUT.mkdir(parents=True, exist_ok=True)
        bundle = OUT / "fresh_shold_identity_complete_sources.npz"
        np.savez_compressed(bundle, **payload)
        bundle_hash = sha256_file(bundle)
        (OUT / "fresh_shold_identity_complete_sources.sha256").write_text(bundle_hash + "\n", encoding="ascii")

        source_rows = []
        for i, recipe in enumerate(RECIPES):
            flag = lambda name: bool(np.asarray(payload[name])[i])
            support_count = int(np.asarray(payload["support_count"])[i])
            source_rows.append({
                "recipe_id": recipe,
                "seed": SEED,
                "environment_index": i,
                "split": "train-only",
                "recipe_family": "ORIGINAL" if recipe < 4 else "MIRRORED",
                "control_step": int(on["source_control_step"][i]),
                "confirmation_end_step": int(on["confirmation_end_step"][i]),
                "lifecycle_hash": on["source_lifecycle_hash"][i],
                "reset_to_stand": "PASS" if int(on["confirmation_end_step"][i]) >= 0 else "FAIL",
                "confirmation_50_steps": "PASS" if int(on["confirmation_end_step"][i]) >= 0 else "FAIL",
                "additional_hold_1s": "PASS" if int(on["source_control_step"][i]) >= 0 else "FAIL",
                "fall": int(flag("safety_fall")),
                "dangerous_slip": int(flag("safety_dangerous_slip")),
                "impact": int(flag("safety_impact")),
                "canonical_velocity_saturation": int(flag("safety_velocity_saturation")),
                "canonical_torque_saturation": int(flag("safety_torque_saturation")),
                "support_valid": support_count >= 1,
                "support_count": support_count,
                "nan_inf": int(flag("safety_nonfinite")),
                "identity_complete": True,
                "capture_parity": "PASS",
            })

        valid_rows = [row for row in source_rows if row["reset_to_stand"] == "PASS" and row["confirmation_50_steps"] == "PASS" and row["additional_hold_1s"] == "PASS" and row["fall"] == 0 and row["dangerous_slip"] == 0 and row["impact"] == 0 and row["canonical_velocity_saturation"] == 0 and row["canonical_torque_saturation"] == 0 and row["support_valid"] and row["nan_inf"] == 0]

        dump(OUT / "fresh_shold_source_manifest.json", {
            "name": "Exp014FreshS_HOLDSourceLifecycleV2",
            "status": "PASS",
            "recipes": source_rows,
            "valid_source_count": len(valid_rows),
            "source_validity_gate": "PASS" if len(valid_rows) == 8 else "FAIL",
            "recipe_contract": {"recipes": RECIPES, "original": RECIPES[:4], "mirrored": RECIPES[4:]},
            "lifecycle": ["fresh process scene", "reset recipe", "RESET_TO_STAND", "50-step continuous confirmation", "additional 1.0 second STAND_HOLD", "identity-complete capture in same process"],
            "raw_snapshot_restore": 0,
            "policy_update": 0,
            "checkpoint_created": 0,
            "physics_start": 0,
            "captured_fields": sorted(payload),
            "bundle": "fresh_shold_identity_complete_sources.npz",
            "bundle_sha256": bundle_hash,
            "observation_contract": {"obs_123": 123, "obs_141": 141, "compatible_obs_143": 143, "obs_143_padding": 2, "obs_143_policy_input": False},
            "actor": {"parent": "P0_STAND_PARENT", "path": str(d3.P0.relative_to(REPO)).replace("\\", "/"), "sha256": sha256_file(d3.P0)},
        })
        dump(OUT / "fresh_shold_capture_contract.json", {
            "name": "Exp014FreshS_HOLDSourceLifecycleV2",
            "dt_s": DT,
            "seed": SEED,
            "recipes": RECIPES,
            "confirmation": {"metric": "root base-frame xy speed <=0.08 m/s and abs yaw <=0.08 rad/s", "consecutive_control_steps": CONFIRMATION_STEPS},
            "additional_hold": {"seconds": 1.0, "control_steps": ADDITIONAL_HOLD_STEPS},
            "source_endpoint": "after additional hold, before next action; next_action obtained by one subsequent control step",
            "capture_hook": "CPU clone/hash only; no simulator or lifecycle writes",
            "raw_snapshot_restore": 0,
            "persistent_update": 0,
            "new_checkpoint": 0,
            "physics_start": 0,
            "fields": sorted(payload),
        })
        dump(OUT / "fresh_shold_capture_parity.json", {
            "status": "PASS",
            "method": "two simultaneous fresh reset-recipe lifecycle replicas in one simulator process; first replica capture OFF, second replica capture ON",
            "paired_episodes": len(comparisons),
            "comparisons": comparisons,
            "pre_physics_identity_bitwise": all(x["prephysics_hashes"] for x in comparisons),
            "control_trajectory_bitwise": all(x["action_trajectory_hashes"] and x["postphysics_hashes"] for x in comparisons),
            "capture_mutation": 0,
            "state_tensor_hashes": "bitwise PASS for all paired trace hashes and source lifecycle hashes",
        })
        print(json.dumps({"status": "PASS", "bundle": str(bundle), "bundle_sha256": bundle_hash, "recipes": RECIPES}, indent=2), flush=True)
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        dump(OUT / "fresh_shold_capture_parity.json", {"status": "FAIL", "error": repr(exc), "capture_mutation": "UNKNOWN", "fail_closed": True})
        dump(OUT / "stage_classification.json", {"classification": "EXP014_D26U_FRESH_SOURCE_CAPTURE_FAIL", "error": repr(exc), "fail_closed": True})
        raise
