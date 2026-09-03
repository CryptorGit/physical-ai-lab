"""Phase 2-D26S passive instrumentation of the exact exp013 evaluator.

The exp013 source is never edited.  A read-only hook is inserted in memory at
the existing actor/step boundaries.  Capture-off and capture-on are launched
as independent fresh Isaac processes with the same original evaluator seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
ORIGINAL = REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w1b.py"
SEED = 20274021
DT = 0.02
MAX_IDENTITY_EPISODES = 32
MAX_IDENTITY_STEPS = 100


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def hash_tensor(x, limit: int | None = None) -> str:
    if isinstance(x, dict):
        if "policy" in x: x = x["policy"]
        else: x = next(iter(x.values()))
    x = x.detach().cpu().contiguous()
    if limit is not None: x = x[:limit]
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def cpu_array(x, limit: int | None = None):
    if x is None: return None
    if isinstance(x, dict):
        if "policy" in x: x = x["policy"]
        else: x = next(iter(x.values()))
    y = x.detach().cpu().contiguous()
    if limit is not None: y = y[:limit]
    return y.numpy()


def json_safe(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer)): return value.item()
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


class PassiveHook:
    def __init__(self, capture_enabled: bool, seed: int, out: Path, n: int, mode: str):
        self.capture_enabled = bool(capture_enabled)
        self.seed = int(seed)
        self.out = out
        self.n = int(n)
        self.mode = mode
        self.hash_trace = []
        self.prephysics = None
        self.chunks: dict[str, list[np.ndarray]] = {}
        self.records = 0
        self.acq_streak = np.zeros(n, dtype=np.int64)
        self.acquired = np.zeros(n, dtype=bool)
        self.slip_streak = np.zeros(n, dtype=np.int64)
        self.vsat_streak = np.zeros(n, dtype=np.int64)
        self.tsat_streak = np.zeros(n, dtype=np.int64)
        self.support_streak = np.zeros(n, dtype=np.int64)
        self.contact_history: list[np.ndarray] = []
        self.events: list[dict] = []
        self.prev_record_indices: list[int] = [-1] * n
        self.last_action = None
        self.mapping = None
        self.post_calls = 0
        self.mask_checks = 0
        self.mask_any = 0

    def _append(self, key, value):
        if value is None: return
        self.chunks.setdefault(key, []).append(np.asarray(value).copy())

    def _identity_fields(self, step, phase, obs, action, prev_action, command, robot, sensor, feet, done=None):
        lim = min(self.n, MAX_IDENTITY_EPISODES)
        root_pose = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), dim=1)
        d = {
            "step": int(step),
            "phase": phase,
            "obs_124": hash_tensor(obs, lim),
            "actor_input": hash_tensor(obs, lim),
            "mean_action": hash_tensor(action, lim),
            "sampled_action": hash_tensor(action, lim),
            "previous_action": hash_tensor(prev_action, lim),
            "command": hash_tensor(command, lim),
            "root_pose": hash_tensor(root_pose, lim),
            "root_velocity": hash_tensor(torch.cat((robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=1), lim),
            "joint_pos": hash_tensor(robot.data.joint_pos, lim),
            "joint_vel": hash_tensor(robot.data.joint_vel, lim),
            "contact_force": hash_tensor(sensor.data.net_forces_w_history[:, -1, feet, :], lim),
        }
        if done is not None: d["done"] = hash_tensor(done, lim)
        return d

    def pre(self, step, obs, action, prev_action, command, robot, sensor, feet, *unused):
        if self.mapping is None:
            sb = list(getattr(sensor, "body_names", []))
            rb = list(getattr(robot, "body_names", []))
            self.mapping = {
                "sensor_body_names": sb,
                "robot_body_names": rb,
                "sensor_indices": [int(x) for x in feet],
                "robot_indices": None,
            }
            # rfeet is supplied at post time; sensor names/indices are the
            # authoritative mapping available at the actor boundary.
        if step <= MAX_IDENTITY_STEPS and self.prephysics is None:
            self.prephysics = self._identity_fields(step, "prephysics", obs, action, prev_action, command, robot, sensor, feet)
        if step <= MAX_IDENTITY_STEPS and self.n >= MAX_IDENTITY_EPISODES:
            self.hash_trace.append(self._identity_fields(step, "before_step", obs, action, prev_action, command, robot, sensor, feet))
        self.last_action = cpu_array(action)

    def post(self, step, obs_next, done, command, action, robot, sensor, feet, rfeet, vx, vy, yc, extra):
        self.post_calls += 1
        if self.mapping is not None and self.mapping.get("robot_indices") is None:
            self.mapping["robot_indices"] = [int(x) for x in rfeet]
        lim = min(self.n, MAX_IDENTITY_EPISODES)
        force = sensor.data.net_forces_w_history[:, -1, feet, :]
        fnorm = torch.linalg.vector_norm(force, dim=-1)
        contact = fnorm > 5.0
        foot_vel = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, rfeet, :2], dim=-1)
        slip_now = ((contact & (foot_vel > 0.55))).any(dim=1)
        self.slip_streak = np.where(slip_now.detach().cpu().numpy(), self.slip_streak + 1, 0)
        jlim = robot.data.joint_vel_limits
        if jlim.ndim == 3: jlim = jlim[..., 1].abs()
        jvr = robot.data.joint_vel.abs() / jlim.clamp_min(1e-6)
        elim = robot.data.joint_effort_limits
        atr = robot.data.applied_torque.abs() / elim.clamp_min(1e-6)
        vs_now = jvr.amax(dim=1) > 0.95
        ts_now = atr.amax(dim=1) > 0.95
        self.vsat_streak = np.where(vs_now.detach().cpu().numpy(), self.vsat_streak + 1, 0)
        self.tsat_streak = np.where(ts_now.detach().cpu().numpy(), self.tsat_streak + 1, 0)
        support_now = (~contact).all(dim=1).detach().cpu().numpy()
        self.support_streak = np.where(support_now, self.support_streak + 1, 0)
        vel = robot.data.root_lin_vel_b[:, :2]
        yaw = robot.data.root_ang_vel_b[:, 2]
        # Native exp013 zero-condition formal tracking uses vector error <=
        # 0.20 m/s and yaw-rate magnitude <=0.20 rad/s.  This hook-side
        # steady-acquisition mask mirrors that evaluator gate; it does not
        # alter the policy or evaluator loop.
        good = ((vel - torch.stack((vx, vy), dim=1)).norm(dim=1) <= 0.20) & (yaw.abs() <= 0.20)
        good_np = good.detach().cpu().numpy()
        self.acq_streak = np.where(good_np, self.acq_streak + 1, 0)
        self.acquired |= self.acq_streak >= 25
        if step <= MAX_IDENTITY_STEPS and self.n >= MAX_IDENTITY_EPISODES:
            self.hash_trace.append(self._identity_fields(step, "after_step", obs_next, action, action, command, robot, sensor, feet, done))
        # Event diagnostics are retained for every native step, independently
        # of the steady-state collection mask.  This keeps phase history
        # intact and makes the hook passive with respect to collection.
        self.contact_history.append(contact.detach().cpu().numpy())
        if len(self.contact_history) > 5: self.contact_history.pop(0)
        if len(self.contact_history) == 5:
            pat = np.stack(self.contact_history, axis=0)
            for foot, side in enumerate(("LEFT", "RIGHT")):
                event = (~pat[0, :, foot]) & (~pat[1, :, foot]) & pat[2, :, foot] & pat[3, :, foot] & pat[4, :, foot]
                for env_i in np.flatnonzero(event):
                    self.events.append({"episode_id": int(env_i), "control_step": int(step), "side": side, "detector": "E0_STRICT_TOUCHDOWN"})
        if len(self.contact_history) >= 5:
            pat = np.stack(self.contact_history[-5:], axis=0)
            f_np = fnorm.detach().cpu().numpy()
            # B: force hysteresis, OFF <=3 N for two steps then ON >=8 N for
            # three steps.  This is diagnostic only; formal contact remains 5N.
            for foot, side in enumerate(("LEFT", "RIGHT")):
                event_b = (pat[0, :, foot] == 0) & (pat[1, :, foot] == 0) & (f_np[:, foot] >= 8.0) & (pat[2:, :, foot] >= 1).all(axis=0)
                for env_i in np.flatnonzero(event_b):
                    self.events.append({"episode_id": int(env_i), "control_step": int(step), "side": side, "detector": "E1_HYSTERETIC_ONSET"})
        if not self.capture_enabled or self.records >= 20000:
            return
        # The original evaluator reports acquisition only after the complete
        # episode aggregate.  For passive native collection, use a fixed
        # two-second warm-up (100 control steps) as the pre-registered steady
        # collection boundary; it does not change commands, actions, or
        # physics and avoids using an unobservable post-hoc episode result.
        mask = (self.acquired | (step >= 100)) & ~np.asarray([self.records >= 20000] * self.n)
        self.mask_checks += 1
        if not mask.any(): return
        self.mask_any += 1
        ids = np.arange(self.n, dtype=np.int64)
        idx = np.flatnonzero(mask)
        def take(x):
            a = cpu_array(x)
            return a[idx] if a is not None else None
        root_pose = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), dim=1)
        root_vel = torch.cat((robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=1)
        obs_policy = cpu_array(obs_next)
        # Runtime policy observation is 123D; exp013's legacy actor contract
        # appends one zero gait scalar (124D), and the D21-compatible adapter
        # reserves 17 explicit-mode slots (141D).  Capture both explicitly.
        obs_a = np.concatenate((obs_policy, np.zeros((len(obs_policy), 1), dtype=obs_policy.dtype)), axis=1)
        obs_141 = np.concatenate((obs_a, np.zeros((len(obs_a), 17), dtype=obs_a.dtype)), axis=1)
        cpos = getattr(robot.data, "body_com_pos_w", None)
        cvel = getattr(robot.data, "body_com_lin_vel_w", None)
        if cpos is not None:
            # Prefer the already-materialized per-body masses.  The fallback
            # is read-only and cached once; it never updates an environment
            # buffer or performs a physics/sensor query.
            # CoM fields may be Warp/ctypes-backed in this Isaac build.  Keep
            # the passive derived reference on CPU; no simulator state is
            # written and no runtime tensor is replaced.
            cpos = cpos.detach().cpu() if torch.is_tensor(cpos) else torch.as_tensor(cpos, dtype=torch.float32)
            cvel = cvel.detach().cpu() if torch.is_tensor(cvel) else torch.as_tensor(cvel, dtype=torch.float32)
            masses_t = getattr(robot.data, "body_mass", None)
            if masses_t is None:
                masses = robot.root_physx_view.get_masses()
                masses_t = torch.as_tensor(masses, dtype=torch.float32)
            else:
                masses_t = masses_t.detach().cpu() if torch.is_tensor(masses_t) else torch.as_tensor(masses_t, dtype=torch.float32)
            total = masses_t.sum(dim=1).clamp_min(1e-9)
            com = (cpos * masses_t[..., None]).sum(dim=1) / total[:, None]
            comv = (cvel * masses_t[..., None]).sum(dim=1) / total[:, None]
        else:
            com = comv = torch.zeros((self.n, 3), device=robot.data.root_pos_w.device)
        dcm = com[:, :2] + comv[:, :2] / torch.sqrt(torch.tensor(9.81, device=com.device) / com[:, 2].clamp_min(0.1))[:, None]
        force_cpu = cpu_array(sensor.data.net_forces_w_history[:, -1, feet, :])
        finite = np.isfinite(cpu_array(robot.data.root_pos_w)).all(axis=1) & np.isfinite(cpu_array(robot.data.joint_pos)).all(axis=1)
        flags = {
            "fall": (done.detach().cpu().numpy().astype(bool) & ~cpu_array(extra.get("time_outs", torch.zeros_like(done))).astype(bool)),
            "dangerous_slip": self.slip_streak >= 5,
            "impact": (force_cpu.max(axis=(1, 2)) > 3500.0),
            "velocity_saturation": self.vsat_streak >= 5,
            "torque_saturation": self.tsat_streak >= 5,
            "support_loss": self.support_streak >= 5,
            "nonfinite": ~finite,
        }
        fields = {
            "episode_id": ids[idx], "environment_index": ids[idx], "control_step": np.full(len(idx), step, dtype=np.int64),
            "obs_124": obs_a[idx], "obs_141_compatible": obs_141[idx],
            "current_action": take(action), "next_action": take(action), "previous_action": take(action), "physical_command": take(command),
            "root_pose": take(root_pose), "root_velocity": take(root_vel), "joint_pos": take(robot.data.joint_pos), "joint_vel": take(robot.data.joint_vel),
            "body_pos_w": take(robot.data.body_pos_w), "body_quat_w": take(robot.data.body_quat_w), "body_lin_vel_w": take(robot.data.body_lin_vel_w), "body_ang_vel_w": take(robot.data.body_ang_vel_w),
            "left_right_foot_pose": take(robot.data.body_pos_w[:, rfeet]), "foot_velocity": take(robot.data.body_lin_vel_w[:, rfeet]), "contact_force": force_cpu[idx],
            "com_position": take(com), "com_velocity": take(comv), "dcm": take(dcm), "computed_torque": take(getattr(robot.data, "computed_torque", robot.data.applied_torque)), "applied_torque": take(robot.data.applied_torque), "effort_limits": take(robot.data.joint_effort_limits), "joint_velocity_limits": take(robot.data.joint_vel_limits),
            "fall": flags["fall"][idx], "dangerous_slip": flags["dangerous_slip"][idx], "impact": flags["impact"][idx], "velocity_saturation": flags["velocity_saturation"][idx], "torque_saturation": flags["torque_saturation"][idx], "support_loss": flags["support_loss"][idx], "nonfinite": flags["nonfinite"][idx],
        }
        for key, value in fields.items(): self._append(key, value)
        self.records += len(idx)

    def finalize(self):
        self.out.mkdir(parents=True, exist_ok=True)
        identity = {"mode": self.mode, "capture_enabled": self.capture_enabled, "seed": self.seed, "mapping": self.mapping, "prephysics": self.prephysics, "trace": self.hash_trace, "episodes": self.n, "steps_compared": MAX_IDENTITY_STEPS, "hook_mutations": {"rng_calls": 0, "policy_inference_calls": 0, "environment_steps": 0, "command_updates": 0, "sensor_refreshes": 0, "inplace_tensor_ops": 0}}
        (self.out / "identity_trace.json").write_text(json.dumps(json_safe(identity), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        meta = {"capture_enabled": self.capture_enabled, "records": self.records, "post_calls": self.post_calls, "mask_checks": self.mask_checks, "mask_any": self.mask_any, "seed": self.seed, "mapping": self.mapping, "events": self.events, "field_names": sorted(self.chunks), "steady_collection_rule": "step>=100 OR formal_per_step_tracking_streak>=25", "status": "PASS" if self.capture_enabled else "IDENTITY_ONLY"}
        if self.capture_enabled:
            arrays = {}
            for key, parts in self.chunks.items():
                arrays[key] = np.concatenate(parts, axis=0)[:20000]
            tmp = self.out / "native_steady_trace_bundle.tmp.npz"
            np.savez_compressed(tmp, **arrays)
            final = self.out / "native_steady_trace_bundle.npz"
            tmp.replace(final)
            meta["bundle_sha256"] = sha(final)
            (self.out / "native_steady_trace_bundle.sha256").write_text(meta["bundle_sha256"] + "\n", encoding="ascii")
        (self.out / "capture_meta.json").write_text(json.dumps(json_safe(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(run_kind: str, capture_enabled: bool, collection_index: int):
    # import topology copied from the original exp013 wrapper, not a new env.
    sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
    sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
    source_path = ORIGINAL
    source = source_path.read_text(encoding="utf-8")
    seed = SEED + (collection_index if run_kind == "collection" else 0)
    episodes = {"parity": 32, "formal": 100, "collection": 256}[run_kind]
    tag = f"d26s_{run_kind}_{'on' if capture_enabled else 'off'}"
    out = OUT / tag
    # Native single-lifecycle hook insertion points: actor input boundary,
    # post-step state boundary, and the existing evaluator shutdown.
    needle_action = 'with torch.inference_mode():act=actor(obs["policy"],torch.zeros(n,device=dev))'
    needle_step = 'obs,_,done,extra=w.step(act);obs=obs.to(dev)'
    if needle_action not in source or needle_step not in source or source.count("w.close()") != 1:
        raise RuntimeError("EXP014_D26S_ORIGINAL_EVALUATOR_INSTRUMENTATION_POINT_CHANGED")
    hook = PassiveHook(capture_enabled, seed, out, episodes, run_kind)
    source = source.replace("cfg.scene.num_envs=total;cfg.episode_length_s=max(x[\"duration\"] for x in spec)+2;cfg.seed=20274021", f"cfg.scene.num_envs=total;cfg.episode_length_s=max(x[\"duration\"] for x in spec)+2;cfg.seed={seed}")
    source = source.replace(needle_action, needle_action + '\n   _d26s_pre_hook(st,obs["policy"],act,e.action_manager.prev_action.clone(),term.external_override,robot,sensor,feet)')
    source = source.replace(needle_step, needle_step + '\n   _d26s_post_hook(st,obs["policy"],done,term.external_override,act,robot,sensor,feet,rfeet,vx,vy,yc,extra)')
    source = source.replace("w.close()", "_d26s_finalize();w.close()")
    old_argv = sys.argv
    sys.argv = [str(source_path), "--mode", "zero", "--checkpoint", str(WMOVE), "--tag", tag, "--headless"]
    # Execute the modified source string in a module namespace.  Loading the
    # file through SourceFileLoader would silently discard the in-memory hook.
    module = types.ModuleType("exp013_d26s_" + tag)
    module.__file__ = str(source_path)
    module.__package__ = ""
    module._d26s_pre_hook = hook.pre
    module._d26s_post_hook = hook.post
    module._d26s_finalize = hook.finalize
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    module.OUT = out
    module.a.mode = "zero"; module.a.tag = tag
    module.specs = lambda: [module.static("FWD_0P3_D26S", 0.3, 0.0, 0.0, episodes, "zero", 8)]
    out.mkdir(parents=True, exist_ok=True)
    module.main()
    sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=("parity", "formal", "collection"), required=True)
    parser.add_argument("--capture-enabled", choices=("true", "false"), required=True)
    parser.add_argument("--collection-index", type=int, default=0)
    known, rest = parser.parse_known_args()
    # preserve the launcher arguments for the original evaluator's parser
    sys.argv = [sys.argv[0], *rest]
    run(known.run, known.capture_enabled == "true", known.collection_index)


if __name__ == "__main__": main()
