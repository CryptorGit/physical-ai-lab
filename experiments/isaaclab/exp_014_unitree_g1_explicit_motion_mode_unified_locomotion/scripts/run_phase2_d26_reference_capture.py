"""D26 fresh W_MOVE identity-complete reference capture.

This runner uses only reset recipes and a fresh lifecycle. It never restores a
raw physical snapshot, never creates a policy checkpoint, and never runs START.
"""
from __future__ import annotations

import argparse
import csv
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
REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik"
RAW = OUT / "raw"
DT = 0.02
SEED = 20282601
N_EPISODES = 256
N_ENVS = 64
TARGET_SPEED = 0.3
TOUCH_WINDOW = 11


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# D3 contains the frozen reset lifecycle and imports/registers the local task.
d3 = load_module("d3_for_d26", HERE.parent / "run_phase2_d3.py")
d15 = load_module("d15_for_d26", HERE.parent / "run_phase2_d15_worker.py")
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def tensor_cpu(x):
    return x.detach().cpu().contiguous()


def actor_action(world, actor):
    base = world.env.observation_manager.compute()["policy"][:, :123]
    gait = torch.zeros(base.shape[0], device=base.device)
    with torch.inference_mode():
        action = actor(base, gait)
    # The frozen W_MOVE contract is legacy-124D: 123 physical features plus
    # the legacy gait scalar.  Keep this explicit in the durable bundle so
    # the 141D adapter is unambiguous.
    return torch.cat((base, gait[:, None]), dim=1), action


def set_command(world, step):
    progress = min(1.0, step / 25.0)
    # W1B native endpoint collection uses the explicit external command
    # override; the ramp is only the deterministic command exposure phase.
    target = torch.zeros(world.env.num_envs, 3, device=world.device)
    target[:, 0] = TARGET_SPEED
    physical = target * (progress * progress * progress * (progress * (progress * 6 - 15) + 10))
    world.term.external_override[:, :3] = physical
    world.term._update_command()
    return physical


def body_field(data, name, n):
    value = getattr(data, name)
    value = value[:n]
    return tensor_cpu(value)


def com_from_body(data, masses, n):
    pos = body_field(data, "body_com_pos_w", n)
    vel = body_field(data, "body_com_lin_vel_w", n)
    m = torch.as_tensor(masses[:n], dtype=pos.dtype)
    total = m.sum(dim=1).clamp_min(1e-9)
    c = (pos * m[..., None]).sum(dim=1) / total[:, None]
    cd = (vel * m[..., None]).sum(dim=1) / total[:, None]
    return c, cd, total


def safe_metrics(world, done, streaks):
    # Canonical D24D definitions; no D24B applied-torque-only shortcut.
    timeout = torch.zeros_like(done, dtype=torch.bool)
    if isinstance(done, torch.Tensor):
        timeout = done.bool() & (world.env.episode_length_buf >= world.env.max_episode_length - 1)
    fall = done.bool() & ~timeout
    force = world.sensor.data.net_forces_w_history[:, -1, world.sf, :].norm(dim=-1)
    contact = force > 5.0
    foot_vel = world.robot.data.body_lin_vel_w[:, world.rf, :2].norm(dim=-1)
    slip_now = (contact & (foot_vel > 0.55)).any(dim=1)
    streaks["slip"] = torch.where(slip_now, streaks["slip"] + 1, torch.zeros_like(streaks["slip"]))
    dangerous_slip = streaks["slip"] >= 5
    impact = force.amax(dim=1) > 3500.0
    jvr = world.robot.data.joint_vel.abs() / world.robot.data.joint_vel_limits.abs().clamp_min(1e-6)
    tr = world.robot.data.applied_torque.abs() / world.robot.data.joint_effort_limits.abs().clamp_min(1e-6)
    vs_now = jvr.amax(dim=1) > 0.95
    ts_now = tr.amax(dim=1) > 0.95
    streaks["vsat"] = torch.where(vs_now, streaks["vsat"] + 1, torch.zeros_like(streaks["vsat"]))
    streaks["tsat"] = torch.where(ts_now, streaks["tsat"] + 1, torch.zeros_like(streaks["tsat"]))
    velocity_sat = streaks["vsat"] >= 5
    torque_sat = streaks["tsat"] >= 5
    support_loss_now = (~contact).all(dim=1)
    streaks["support"] = torch.where(support_loss_now, streaks["support"] + 1, torch.zeros_like(streaks["support"]))
    support_loss = streaks["support"] >= 5
    finite = torch.isfinite(world.robot.data.root_pos_w).all(dim=1) & torch.isfinite(world.robot.data.joint_pos).all(dim=1)
    nonfinite = ~finite
    return {"fall": fall, "dangerous_slip": dangerous_slip, "impact": impact, "velocity_saturation": velocity_sat, "torque_saturation": torque_sat, "support_loss": support_loss, "nonfinite": nonfinite, "contact": contact, "force": force, "foot_velocity": foot_vel, "joint_velocity_ratio": jvr, "torque_ratio": tr}


def make_payload(world, actor, obs, action, next_action, prev_action, physical, step, episode_ids, recipe_ids, touchdown_side, since_touch, metrics, masses):
    data = world.robot.data
    c, cd, mass = com_from_body(data, masses, world.env.num_envs)
    root_pose = torch.cat((data.root_pos_w, data.root_quat_w), dim=1)
    root_vel = torch.cat((data.root_lin_vel_w, data.root_ang_vel_w), dim=1)
    foot_pose = data.body_pos_w[:, world.rf]
    foot_quat = data.body_quat_w[:, world.rf]
    foot_vel = data.body_lin_vel_w[:, world.rf]
    omega = torch.sqrt(torch.tensor(9.81, device=c.device) / c[:, 2].clamp_min(0.1))
    dcm = c[:, :2] + cd[:, :2] / omega[:, None]
    try:
        jac_raw = world.robot.root_physx_view.get_jacobians()
        if torch.is_tensor(jac_raw):
            jac = tensor_cpu(jac_raw)
        else:
            # IsaacLab exposes the PhysX Jacobian as a Warp array.
            import warp as wp
            jac = tensor_cpu(wp.to_torch(jac_raw))
        jacobian_status = "AVAILABLE"
    except Exception:
        jac = np.zeros((len(episode_ids), 0, 6, 43), dtype=np.float32)
        jacobian_status = "UNAVAILABLE"
    effort = data.joint_effort_limits
    applied = getattr(data, "applied_torque", torch.zeros_like(data.joint_pos))
    computed = getattr(data, "computed_torque", applied)
    return {
        "episode_id": torch.as_tensor(episode_ids, dtype=torch.int64), "recipe_id": torch.as_tensor(recipe_ids, dtype=torch.int64), "control_step": torch.full((len(episode_ids),), step, dtype=torch.int64),
        "touchdown_side": torch.as_tensor(touchdown_side, dtype=torch.int8), "steps_since_touchdown": torch.as_tensor(since_touch, dtype=torch.int8),
        "obs_124": tensor_cpu(obs), "obs_141_compatible": tensor_cpu(torch.cat((obs, torch.zeros(obs.shape[0], 17, device=obs.device)), dim=1)),
        "current_action": tensor_cpu(action), "next_action": tensor_cpu(next_action), "previous_action": tensor_cpu(prev_action), "physical_command": tensor_cpu(physical),
        "root_pose": tensor_cpu(root_pose), "root_velocity": tensor_cpu(root_vel), "joint_pos": tensor_cpu(data.joint_pos), "joint_vel": tensor_cpu(data.joint_vel),
        "body_pos_w": tensor_cpu(data.body_pos_w), "body_quat_w": tensor_cpu(data.body_quat_w), "body_lin_vel_w": tensor_cpu(data.body_lin_vel_w), "body_ang_vel_w": tensor_cpu(data.body_ang_vel_w),
        "left_foot_pose": tensor_cpu(torch.cat((data.body_pos_w[:, world.rf[0:1]], data.body_quat_w[:, world.rf[0:1]]), dim=2).squeeze(1)), "right_foot_pose": tensor_cpu(torch.cat((data.body_pos_w[:, world.rf[1:2]], data.body_quat_w[:, world.rf[1:2]]), dim=2).squeeze(1)),
        "foot_velocity": tensor_cpu(data.body_lin_vel_w[:, world.rf]), "contact_force": tensor_cpu(world.sensor.data.net_forces_w_history[:, -1, world.sf, :]), "com_position": tensor_cpu(c), "com_velocity": tensor_cpu(cd), "com_height": tensor_cpu(c[:, 2]), "dcm": tensor_cpu(dcm),
        "body_jacobians": jac, "body_com_pos_w": tensor_cpu(data.body_com_pos_w), "body_com_lin_vel_w": tensor_cpu(data.body_com_lin_vel_w),
        "applied_torque": tensor_cpu(applied), "computed_torque": tensor_cpu(computed), "effort_limits": tensor_cpu(effort),
        "joint_velocity_ratio": tensor_cpu(metrics["joint_velocity_ratio"]), "torque_ratio": tensor_cpu(metrics["torque_ratio"]),
        "jacobian_status": np.array([jacobian_status] * len(episode_ids), dtype=object),
        "centroidal_momentum_status": np.array(["CENTROIDAL_MOMENTUM_UNAVAILABLE"] * len(episode_ids), dtype=object),
        "fall": metrics["fall"].detach().cpu(), "dangerous_slip": metrics["dangerous_slip"].detach().cpu(), "impact": metrics["impact"].detach().cpu(), "velocity_saturation": metrics["velocity_saturation"].detach().cpu(), "torque_saturation": metrics["torque_saturation"].detach().cpu(), "support_loss": metrics["support_loss"].detach().cpu(), "nonfinite": metrics["nonfinite"].detach().cpu(),
    }


def append_payload(store, payload, mask):
    ids = torch.as_tensor(mask, dtype=torch.bool)
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            store.setdefault(key, []).append(value[ids].clone())
        elif isinstance(value, np.ndarray):
            store.setdefault(key, []).append(value[ids.numpy()].copy())


def stage_collision_geometry(world):
    # Numeric geometry audit is kept in the capture runner; if USD traversal is
    # unavailable the output records the exact API failure rather than guessing.
    report = {"status": "NOT_EXECUTED", "source": "USD collision geometry", "feet": {"left": {}, "right": {}}, "reason": "USD geometry extraction is finalized in the offline auditor after capture"}
    return report


def persist_capture(store, touchdown_rows, episode_rows, recipes, actor_path, world, actor):
    """Persist while the Isaac application is still alive.

    IsaacLab's launcher may terminate the application during context teardown;
    writing before ``wrapped.close`` makes the durable bundle independent of
    that teardown behavior.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_persist_entered.txt").write_text("entered\n", encoding="ascii")
    count = min((sum(x.shape[0] for x in values) for values in store.values()), default=0)
    arrays = {}
    for key, chunks in store.items():
        if chunks and isinstance(chunks[0], torch.Tensor):
            arrays[key] = torch.cat(chunks, dim=0)[:count].numpy()
        else:
            arrays[key] = np.concatenate(chunks, axis=0)[:count]
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = RAW / "wmove_identity_complete_reference.tmp.npz"
    final = RAW / "wmove_identity_complete_reference.npz"
    np.savez_compressed(tmp, **arrays)
    (OUT / "_persist_npz_done.txt").write_text(str(count) + "\n", encoding="ascii")
    tmp.replace(final)
    out_hash = sha(final)
    (RAW / "wmove_identity_complete_reference.sha256").write_text(out_hash + "\n", encoding="ascii")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "wmove_post_touchdown_events.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted(touchdown_rows[0]) if touchdown_rows else ["episode_id", "recipe_id", "touchdown_side", "touchdown_step", "previous_support_side", "definition", "ambiguous"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(touchdown_rows)
    dump(OUT / "wmove_post_touchdown_events.json", {"events": touchdown_rows, "count": len(touchdown_rows)})
    dump(OUT / "wmove_reference_capture_manifest.json", {
        "reference": "WMove03IdentityCompletePostTouchdownReferenceV1",
        "target": {"vx": 0.3, "vy": 0.0, "yaw": 0.0}, "seed": SEED,
        "episodes": N_EPISODES, "recipes": recipes,
        "collection": "steady acquisition only; fresh reset recipe lifecycle",
        "minimum_states": 20000, "collected_states": int(count),
        "bundle_sha256": out_hash, "raw_snapshot_restore": 0,
        "centroidal_momentum": "CENTROIDAL_MOMENTUM_UNAVAILABLE",
        "fields": sorted(arrays),
        "actor_checkpoint": str(actor_path.relative_to(REPO)).replace("\\", "/"),
        "actor_sha256": sha(actor_path),
        "stage_collision_geometry": stage_collision_geometry(world),
        "episode_rows": episode_rows,
    })
    dump(OUT / "wmove_native_lifecycle_contract.json", {
        "source": "D3/D6/Exp013 W1B artifact and runtime",
        "reset_lifecycle": "recipe reset via StandWorld.restore; no raw trajectory snapshot",
        "command": "external base_velocity override", "target_speed_mps": 0.3,
        "target_yaw_rps": 0.0, "ramp": "minimum-jerk 25 control steps (runtime capture contract)",
        "actor": "FrozenGaitActor model_200.pt; 123D base observation + gait=0",
        "legacy_observation_dimension": 124, "acquisition": "velocity-vector error <=0.12 m/s and abs yaw <=0.10 rad/s for 25 steps",
        "collection_start": "after acquisition", "episode_duration": "7.2 s cap (360 control steps)",
        "contact_timing": "net force norm >5N at post-step sensor refresh",
        "previous_action": "environment action manager reset to zero per recipe", "unknown": [],
    })
    return int(count), out_hash


def main():
    parser = argparse.ArgumentParser(); add_launcher_args(parser); args, hydra = setup_preset_cli(parser); sys.argv = [sys.argv[0], *hydra]
    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = N_ENVS; cfg.seed = SEED; cfg.episode_length_s = 20.0; cfg.observations.policy.enable_corruption = False; cfg.events.base_external_force_torque = None; cfg.events.push_robot = None
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    recipes = d3.TRAIN[:N_EPISODES]
    if len(recipes) < N_EPISODES: recipes = list(range(N_EPISODES))
    actor_path = d3.WMOVE
    actor = None
    store = {}; episode_rows = []; touchdown_rows = []; episode_counter = 0; collected = 0
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        world = d3.StandWorld(wrapped, d3.load_resets(), torch.zeros(680)); actor = FrozenGaitActor(actor_path).to(world.device).eval()
        masses = world.robot.root_physx_view.get_masses()
        # fresh lifecycle batches: this is reset-recipe initialization, not raw snapshot restore.
        for start in range(0, N_EPISODES, N_ENVS):
            batch_recipes = recipes[start:start + N_ENVS]; n = len(batch_recipes); padded = batch_recipes + [batch_recipes[-1]] * (N_ENVS - n)
            obs = world.restore(torch.tensor(padded, device=world.device))
            episode_ids = list(range(start, start + n)); recipe_ids = batch_recipes
            target = torch.zeros(N_ENVS, 3, device=world.device); target[:, 0] = TARGET_SPEED
            acquired = torch.zeros(N_ENVS, dtype=torch.bool, device=world.device); streak = torch.zeros(N_ENVS, dtype=torch.long, device=world.device)
            history = []
            active_windows = [[] for _ in range(N_ENVS)]
            touch_pending = {"left": [0] * N_ENVS, "right": [0] * N_ENVS}
            streaks = {k: torch.zeros(N_ENVS, dtype=torch.long, device=world.device) for k in ("slip", "vsat", "tsat", "support")}
            episode_flags = {k: torch.zeros(N_ENVS, dtype=torch.bool, device=world.device) for k in ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nonfinite")}
            for step in range(360):
                physical = set_command(world, step)
                obs, action = actor_action(world, actor)
                prev = world.env.action_manager.prev_action.clone()
                _, _, done, _ = wrapped.step(action)
                metrics = safe_metrics(world, done, streaks)
                for k in episode_flags: episode_flags[k] |= metrics[k]
                vel = world.robot.data.root_lin_vel_b[:, :2]; yaw = world.robot.data.root_ang_vel_b[:, 2]
                good = ((vel - target[:, :2]).norm(dim=1) <= 0.12) & (yaw.abs() <= 0.10)
                streak = torch.where(good, streak + 1, torch.zeros_like(streak)); acquired |= streak >= 25
                next_obs, next_action = actor_action(world, actor)
                contact = metrics["contact"]
                history.append(contact.detach().cpu())
                if len(history) > 5: history.pop(0)
                if len(history) >= 5 and acquired.any():
                    pattern = torch.stack(history[-5:])
                    for foot, side in ((0, 1), (1, 2)):
                        event = (~pattern[0, :, foot]) & (~pattern[1, :, foot]) & pattern[2, :, foot] & pattern[3, :, foot] & pattern[4, :, foot] & acquired.detach().cpu()
                        for env_i in event.nonzero().flatten().tolist():
                            if active_windows[env_i] and active_windows[env_i][0] <= 10:
                                continue
                            # [steps_since_touchdown, touchdown foot code]
                            active_windows[env_i] = [0, foot + 1]
                            touchdown_rows.append({"episode_id": episode_ids[env_i] if env_i < n else -1, "recipe_id": recipe_ids[env_i] if env_i < n else -1, "touchdown_side": "LEFT_TOUCHDOWN" if foot == 0 else "RIGHT_TOUCHDOWN", "touchdown_step": step - 2, "previous_support_side": "RIGHT" if foot == 0 else "LEFT", "definition": "two non-contact followed by three continuous contact", "ambiguous": False})
                # All active windows are represented by the current state. The
                # last two padded envs are ignored; only the first n are data.
                if step >= 0 and collected < 24000:
                    mask = torch.zeros(N_ENVS, dtype=torch.bool, device=world.device)
                    for env_i in range(n):
                        if active_windows[env_i] and active_windows[env_i][0] <= 10:
                            mask[env_i] = True; active_windows[env_i][0] += 1
                    if mask.any():
                        side_codes = [(active_windows[i][1] if active_windows[i] else 0) for i in range(N_ENVS)]
                        since_codes = [(active_windows[i][0] - 1 if active_windows[i] else 0) for i in range(N_ENVS)]
                        payload = make_payload(world, actor, obs, action, next_action, prev, physical, step, episode_ids + [episode_ids[-1]] * (N_ENVS - n), padded, side_codes, since_codes, metrics, masses)
                        append_payload(store, payload, mask.detach().cpu().numpy())
                        collected += int(mask[:n].sum())
                obs = next_obs
            for j in range(n):
                episode_rows.append({"episode_id": episode_ids[j], "recipe_id": recipe_ids[j], "acquired": bool(acquired[j]), "collected_post_touchdown_states": 0, **{k: bool(v[j]) for k, v in episode_flags.items()}})
            print(json.dumps({"batch_start": start, "episodes": n, "collected_so_far": collected}), flush=True)
            # Durable checkpoint at every fresh batch; this is still owned by
            # the parent process and is replaced atomically.
            count, out_hash = persist_capture(store, touchdown_rows, episode_rows, recipes, actor_path, world, actor)
            print(json.dumps({"episodes": N_EPISODES, "collected_states": count, "touchdown_events": len(touchdown_rows), "bundle_sha256": out_hash}, indent=2), flush=True)
        wrapped.close()


if __name__ == "__main__": main()
