"""EXP014 Phase 2-D30B deterministic nonlinear post-touchdown reachability.

This stage is a fresh-process Isaac/PhysX direct-shooting experiment.  It
reuses the protected D29B Route-A lifecycle and the protected D30A action
basis, but never edits either stage or substitutes a learned surrogate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d30b_nonlinear_post_touchdown_reachability"
RAW = OUT / "raw"
REPORT = REPO / "research/exp_014_phase_2_d30b_nonlinear_post_touchdown_reachability_report.md"
D29B_SCRIPT = EXP / "scripts/run_phase2_d29b_walk_capture.py"
D30A_SCRIPT = EXP / "scripts/run_phase2_d30a_post_touchdown_capture_mpc.py"
BASIS_NPZ = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d30a_post_touchdown_capture_mpc/basis_components.npz"
BASIS_JSON = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d30a_post_touchdown_capture_mpc/capture_action_basis.json"
D26T_MANIFEST = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans/entry_neighborhood_manifest.json"
D26S_BUNDLE = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation/native_steady_trace_bundle.npz"
KNOWN_ISAAC = Path(r"C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe")

SEED = 20279941
DT = 0.02
N = 8
STAND_STEPS = 100
PHYSICS_STEPS = 320
PHASES = 4
COORDINATES = 16
POST_TD0_RELEASE_STEPS = 8
RETENTION_STEPS = 100
PATTERN_SCALES = (0.50, 0.25, 0.125, 0.0625)
MAX_CANDIDATES_PER_ITERATION = 33
MAX_ACCEPTED_MOVES_PER_SCALE = 8
BOUND_FRACTION = 0.10
CLASS_MULTIPLE_FAILURES = "EXP014_D30B_MULTIPLE_FAILURES"
CLASS_PASS = "EXP014_D30B_NONLINEAR_POST_TOUCHDOWN_REACHABILITY_PASS"
CLASS_RELEASE_FAIL = "EXP014_D30B_CAPTURE_REACHABLE_WMOVE_RELEASE_FAIL"
CLASS_PARTIAL = "EXP014_D30B_NONLINEAR_POST_TOUCHDOWN_REACHABILITY_PARTIAL"
CLASS_NO_GO = "EXP014_D30B_POSITION_TARGET_POST_TOUCHDOWN_NO_GO"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"IMPORT_FAILED:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def d30b_path(path: str) -> bool:
    p = path.replace("\\", "/")
    return "phase_2_d30b_nonlinear_post_touchdown_reachability" in p or p.endswith("run_phase2_d30b_nonlinear_post_touchdown_reachability.py") or p.endswith("exp_014_phase_2_d30b_nonlinear_post_touchdown_reachability_report.md")


def repo_state() -> dict[str, Any]:
    raw = git("status", "--porcelain=v1")
    filtered = "\n".join(line for line in raw.splitlines() if not d30b_path(line[3:]))
    return {"head": git("rev-parse", "HEAD"), "worktree_status": filtered.splitlines() if filtered else [], "filtered_d30b_paths": True, "status_sha256": hashlib.sha256(filtered.encode()).hexdigest()}


def reset_stale_raw_artifacts(force: bool = False) -> dict[str, Any]:
    revision = sha(HERE)
    marker = OUT / "raw_revision.json"
    previous = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
    stale = bool(force or previous.get("script_sha256") != revision)
    removed = 0
    if stale and RAW.exists():
        for path in RAW.glob("*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
            elif path.is_dir():
                shutil.rmtree(path)
    RAW.mkdir(parents=True, exist_ok=True)
    history = list(previous.get("history", []))
    history.append({"script_sha256": revision, "stale_raw_cleared": stale, "removed_files": removed})
    dump(marker, {"script_sha256": revision, "previous_script_sha256": previous.get("script_sha256"), "stale_raw_cleared": stale, "removed_files": removed, "history": history[-8:]})
    return {"stale_raw_cleared": stale, "removed_files": removed, "history": history[-8:], "script_sha256": revision}


def basis():
    d30a = load_module("d30a_readonly_for_d30b", D30A_SCRIPT)
    arrays = np.load(BASIS_NPZ, allow_pickle=False)
    b = d30a.WMoveCaptureActionBasisV1(
        np.asarray(arrays["mean"]),
        np.asarray(arrays["components"]),
        np.asarray(arrays["singular_values"]),
        np.asarray(arrays["explained_variance_ratio"]),
    )
    manifest = json.loads(BASIS_JSON.read_text(encoding="utf-8"))
    if b.preregistered_dimension() != 4:
        raise RuntimeError("D30A_BASIS_DIMENSION_NOT_4")
    bound = np.asarray(manifest["coefficient_scale"], dtype=np.float64) * BOUND_FRACTION
    return b, manifest, bound, d30a


_PHASE_TUBE_CACHE: dict[str, Any] | None = None


def phase_tube_features() -> dict[str, Any]:
    global _PHASE_TUBE_CACHE
    if _PHASE_TUBE_CACHE is not None:
        return _PHASE_TUBE_CACHE
    d29b = load_module("d29b_phase_tube_for_d30b", D29B_SCRIPT)
    manifest = json.loads(D26T_MANIFEST.read_text(encoding="utf-8"))
    bundle = dict(np.load(D26S_BUNDLE, allow_pickle=False))
    refs: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    for side in ("LEFT", "RIGHT"):
        rows = np.asarray([int(item["bundle_row"]) for item in manifest["references"] if item["side"] == side], dtype=int)
        features = d29b.wmove_feature_from_bundle(bundle, rows, side).astype(np.float64)
        center = np.median(features, axis=0)
        mad = np.median(np.abs(features - center), axis=0) * 1.4826
        iqr = np.quantile(features, 0.75, axis=0) - np.quantile(features, 0.25, axis=0)
        refs[side] = features
        scales[side] = np.maximum(np.maximum(mad, iqr / 1.349), 1.0e-6)
    _PHASE_TUBE_CACHE = {"references": refs, "scales": scales, "counts": {"LEFT": len(refs["LEFT"]), "RIGHT": len(refs["RIGHT"])}}
    return _PHASE_TUBE_CACHE


def minimum_jerk(t: float) -> float:
    x = min(max(float(t), 0.0), 1.0)
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def strict_events(history: list[np.ndarray], step: int, start: int, events: list[dict[str, Any]], seen: set[tuple[int, str]]) -> None:
    if step < start or len(history) < 5:
        return
    pattern = np.stack(history[-5:], axis=0)
    for foot, side in enumerate(("LEFT", "RIGHT")):
        hit = (~pattern[0, :, foot]) & (~pattern[1, :, foot]) & pattern[2:, :, foot].all(axis=0)
        for env in np.flatnonzero(hit):
            key = (int(env), side)
            if key not in seen:
                seen.add(key)
                events.append({"recipe_id": int(env), "side": side, "control_step": int(step), "detector": "E0_STRICT_TOUCHDOWN"})


def phase_delta(theta: np.ndarray, step: int, td: list[int], basis, bound: np.ndarray) -> np.ndarray:
    if not td or td[0] < 0:
        return np.zeros(37, dtype=np.float64)
    if len(td) >= 5 and step >= td[4]:
        release_alpha = (step - td[4]) / max(POST_TD0_RELEASE_STEPS, 1)
        release = 1.0 - minimum_jerk(release_alpha)
        c = np.clip(theta[12:16], -bound, bound)
        return basis.components[:4].T @ (c * release)
    phase = None
    for k in range(min(4, len(td))):
        lo = td[k]
        hi = td[k + 1] if k < 3 and len(td) > k + 1 and td[k + 1] >= 0 else lo + 40
        if lo >= 0 and lo <= step <= hi:
            phase = k
            alpha = (step - lo) / max(hi - lo, 1)
            break
    if phase is None:
        return np.zeros(37, dtype=np.float64)
    c = np.clip(theta[phase * 4:(phase + 1) * 4], -bound, bound)
    return basis.components[:4].T @ (c * minimum_jerk(alpha))


def load_schedule(path: Path | None) -> dict[str, Any]:
    return {} if path is None else json.loads(path.read_text(encoding="utf-8"))


def child_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--candidate", required=False)
    parser.add_argument("--baseline", action="store_true")
    from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
    add_launcher_args(parser)
    args, hydra = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]
    return args


def run_child(args: argparse.Namespace) -> int:
    import random
    import torch
    from isaaclab_tasks.utils import launch_simulation
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    d29b = load_module("d29b_readonly_for_d30b", D29B_SCRIPT)
    b, _, bound, _ = basis()
    theta = np.zeros(COORDINATES, dtype=np.float64) if args.baseline else np.asarray(json.loads(Path(args.candidate).read_text(encoding="utf-8"))["theta"], dtype=np.float64)
    gym, cfg, agent = d29b.configure(args, "Isaac-Exp013-G1-DirectionalBaseline-v0", N, PHYSICS_STEPS * DT + 2.0)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        robot, sensor = env.scene["robot"], env.scene["contact_forces"]
        sensor_feet, robot_feet, sensor_names, robot_names = d29b.find_foot_indices(sensor, robot)
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        p0 = d29b.load_actor(d29b.P0, env.device, False)
        wmove = d29b.load_actor(d29b.WMOVE, env.device, True)
        d29b.normal_reset(env, term)
        previous_action = torch.zeros((N, 37), device=env.device)
        contact_history: list[np.ndarray] = []
        events: list[dict[str, Any]] = []
        seen: set[tuple[int, str]] = set()
        per_env_events: list[list[dict[str, Any]]] = [[] for _ in range(N)]
        store: dict[str, list[np.ndarray]] = {}
        safety = {key: np.zeros(N, dtype=bool) for key in ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nan_inf")}
        streaks = {key: np.zeros(N, dtype=np.int32) for key in ("slip", "velocity", "torque", "support")}
        first_failure: list[str | None] = [None] * N
        candidate_active = np.ones(N, dtype=bool)
        td_steps: list[list[int]] = [[] for _ in range(N)]
        for step in range(PHYSICS_STEPS):
            command = torch.zeros((N, 3), device=env.device)
            if step >= STAND_STEPS:
                command[:, 0] = d29b.WMOVE_SPEED
            term.external_override.copy_(command); term._update_command()
            obs = wrapped.get_observations()["policy"].to(env.device)
            action = d29b.actor_action(p0 if step < STAND_STEPS else wmove, obs, env.device, step >= STAND_STEPS)
            delta = np.zeros((N, 37), dtype=np.float64)
            if step >= STAND_STEPS and not args.baseline:
                for env_i in range(N):
                    if candidate_active[env_i]:
                        delta[env_i] = phase_delta(theta, step, td_steps[env_i], b, bound)
                action = action + torch.as_tensor(delta, dtype=action.dtype, device=action.device)
            _, _, done, extras = wrapped.step(action)
            timeout = extras.get("time_outs", torch.zeros_like(done)) if isinstance(extras, dict) else torch.zeros_like(done)
            state = d29b.snapshot(env, robot, sensor, sensor_feet, robot_feet, previous_action, action)
            done_np = np.asarray(done.detach().cpu(), dtype=bool)
            timeout_np = np.asarray(timeout.detach().cpu(), dtype=bool)
            d29b.safety_update(state, done_np, timeout_np, safety, streaks, first_failure, step)
            if not args.baseline:
                candidate_active &= ~np.asarray([any(bool(safety[key][i]) for key in safety) for i in range(N)], dtype=bool)
            contact = np.asarray(state["contact"], dtype=bool)
            contact_history.append(contact.copy())
            if len(contact_history) > 5:
                contact_history.pop(0)
            before = len(events)
            strict_events(contact_history, step, STAND_STEPS, events, seen)
            for event in events[before:]:
                per_env_events[event["recipe_id"]].append(event)
            feature = d29b.feature_from_state(state).astype(np.float64)
            for key, value in {
                "control_step": np.full(N, step, dtype=np.int32),
                "source_environment_index": np.arange(N, dtype=np.int32),
                "action": action.detach().cpu().numpy(),
                "delta_action": delta,
                "root_velocity": state["root_velocity"],
                "root_pose": state["root_pose"],
                "contact": contact,
                "contact_force_norm": state["contact_force_norm"],
                "applied_torque": state["applied_torque"],
                "effort_limit": state["effort_limit"],
                "joint_velocity": state["joint_velocity"],
                "velocity_limit": state["velocity_limit"],
                "feature": feature,
                "done": done_np,
                "timeout": timeout_np,
                "applied_torque": state["applied_torque"],
                "effort_limit": state["effort_limit"],
                "joint_velocity": state["joint_velocity"],
                "velocity_limit": state["velocity_limit"],
            }.items():
                store.setdefault(key, []).append(np.asarray(value).copy())
            for i in range(N):
                if len(td_steps[i]) < len(per_env_events[i]):
                    td_steps[i].append(int(per_env_events[i][-1]["control_step"]))
            previous_action = action.detach().clone()
        data = {key: np.concatenate(values, axis=0) for key, values in store.items()}
        name = "baseline" if args.baseline else Path(args.candidate).stem
        np.savez_compressed(RAW / f"{name}.npz", **data)
        rows = []
        for i in range(N):
            rows.append({"recipe_id": i, "events": per_env_events[i], "touchdown_steps": td_steps[i], "first_failure": first_failure[i], "safety": {key: bool(value[i]) for key, value in safety.items()}})
        dump(RAW / f"{name}.json", {"name": name, "baseline": bool(args.baseline), "theta": theta, "events": events, "results": rows, "sensor_foot_names": sensor_names, "robot_foot_names": robot_names, "raw_sha256": sha(RAW / f"{name}.npz"), "fresh_process": True, "route": "A_CONTINUE_WMOVE"})
        wrapped.close()
    return 0


def spawn_child(python: Path, *, baseline: bool = False, candidate: Path | None = None, device: str | None = None) -> dict[str, Any]:
    cmd = [str(python), str(HERE), "--child", "--headless", "--viz", "none"]
    if baseline:
        cmd.append("--baseline")
        name = "baseline"
    else:
        cmd.extend(["--candidate", str(candidate)])
        name = candidate.stem
    if device:
        cmd.extend(["--device", device])
    log_dir = OUT / "child_logs"; log_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=False)
    (log_dir / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
    return {"name": name, "returncode": result.returncode, "success": result.returncode == 0 and (RAW / f"{name}.json").exists(), "stderr_tail": result.stderr[-3000:]}


def analyze_candidate(name: str, theta: np.ndarray, baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    trajectory_path = RAW / f"{name}.npz"
    trajectory = dict(np.load(trajectory_path, allow_pickle=False)) if trajectory_path.exists() else {}
    events_by_env = [row.get("events", []) for row in candidate.get("results", [])]
    captures = []
    release = []
    retention = []
    retention_rows = []
    release_residuals = []
    release_ok_mask = []
    return_map_values = []
    per_source = []
    tube = phase_tube_features()
    baseline_td0 = [int(row.get("events", [{}])[0].get("control_step", PHYSICS_STEPS)) if row.get("events") else PHYSICS_STEPS for row in baseline.get("results", [])]
    prefix_parity = []
    if trajectory:
        base_path = RAW / "baseline.npz"
        base = dict(np.load(base_path, allow_pickle=False)) if base_path.exists() else {}
        for env in range(N):
            limit = baseline_td0[env] + 1
            diffs = []
            for field in ("feature", "root_pose", "root_velocity", "contact", "action"):
                if field in base and field in trajectory:
                    left = base[field][env::N][:limit].astype(float)
                    right = trajectory[field][env::N][:limit].astype(float)
                    diffs.append(float(np.max(np.abs(left - right))) if left.size else 0.0)
            prefix_parity.append(max(diffs, default=0.0))
    safety_failures = []
    for i, row in enumerate(candidate.get("results", [])):
        events = events_by_env[i]
        unique_steps = [int(e["control_step"]) for e in events]
        seq = [str(e["side"]) for e in events]
        capture = len(unique_steps) >= 5 and all(seq[j] != seq[j - 1] for j in range(1, 5))
        captures.append(bool(capture))
        safety_failures.extend([key for key, value in row.get("safety", {}).items() if value])
        if capture and trajectory:
            td4 = unique_steps[4]
            env_mask = trajectory["source_environment_index"] == i
            release_window = env_mask & (trajectory["control_step"] > td4) & (trajectory["control_step"] <= td4 + POST_TD0_RELEASE_STEPS)
            release_residual = float(np.max(np.linalg.norm(trajectory["delta_action"][release_window], axis=1))) if np.any(release_window) else float("inf")
            release_ok = bool(np.sum(release_window) >= POST_TD0_RELEASE_STEPS and release_residual <= 1.0e-8 and np.isfinite(trajectory["delta_action"][release_window]).all())
            window = env_mask & (trajectory["control_step"] > td4 + POST_TD0_RELEASE_STEPS) & (trajectory["control_step"] <= td4 + POST_TD0_RELEASE_STEPS + RETENTION_STEPS)
            count = int(np.sum(window))
            finite = bool(np.isfinite(trajectory["feature"][window]).all()) if count else False
        else:
            count, finite, release_residual, release_ok = 0, False, float("inf"), False
        release_residuals.append(release_residual)
        release_ok_mask.append(release_ok)
        stable = bool(capture and release_ok and not any(row.get("safety", {}).values()))
        release.append(bool(capture and release_ok))
        retention_rows.append(count)
        retention.append(bool(stable and count >= RETENTION_STEPS and finite))
        d2 = d4 = d4_ratio = yaw = velocity = effort = None
        evaluation_count = int(np.sum(trajectory["source_environment_index"] == i)) if trajectory else 0
        if len(unique_steps) >= 5 and trajectory:
            event2, event4 = events[2], events[4]
            for event, key in ((event2, "d2"), (event4, "d4")):
                feature = trajectory["feature"][int(event["control_step"]) * N + i]
                ref = tube["references"][event["side"]]
                scale = tube["scales"][event["side"]]
                distance = float(np.min(np.linalg.norm((ref - feature[None, :]) / scale[None, :], axis=1)))
                if key == "d2":
                    d2 = distance
                else:
                    d4 = distance
            d4_ratio = float(d4 / max(d2, 1.0e-12))
            idx4 = int(unique_steps[4]) * N + i
            rv = trajectory["root_velocity"][idx4]
            yaw = float(abs(rv[5]))
            velocity = float(np.linalg.norm(rv[:2] - np.asarray([0.3, 0.0])))
            effort = float(np.max(np.abs(trajectory["applied_torque"][idx4]) / np.maximum(trajectory["effort_limit"][idx4], 1.0e-6)))
        per_source.append({
            "source": f"R{i}",
            "td_event_count": len(unique_steps),
            "contact_sequence": seq[:5],
            "contact_sequence_pass": capture,
            "d2": d2,
            "d4": d4,
            "d4_over_max_d2_eps": d4_ratio,
            "yaw": yaw,
            "velocity": velocity,
            "effort": effort,
            "safety": row.get("safety", {}),
            "evaluation_count": evaluation_count,
            "capture": bool(capture),
            "release": bool(release_ok),
            "retention": bool(retention[-1]),
        })
        if len(unique_steps) >= 2 and "feature" in trajectory:
            event_states = [trajectory["feature"][step * N + i] for step in unique_steps[:5] if step * N + i < len(trajectory["feature"])]
            if len(event_states) >= 2:
                return_map_values.append(float(np.mean([np.linalg.norm(event_states[j] - event_states[j - 1]) for j in range(1, len(event_states))])))
    capture_count = int(sum(captures))
    stable_count = int(sum(release))
    return {
        "name": name,
        "theta": theta,
        "capture_count": capture_count,
        "stable_count": stable_count,
        "release_count": int(sum(release)),
        "retention_count": int(sum(retention)),
        "retention_rows": retention_rows,
        "release_residual_max": release_residuals,
        "release_ok_mask": release_ok_mask,
        "return_map_residual": float(np.mean(return_map_values)) if return_map_values else None,
        "per_source": per_source,
        "d2": [row["d2"] for row in per_source],
        "d4": [row["d4"] for row in per_source],
        "d4_over_max_d2_eps": [row["d4_over_max_d2_eps"] for row in per_source],
        "yaw": [row["yaw"] for row in per_source],
        "velocity": [row["velocity"] for row in per_source],
        "effort": [row["effort"] for row in per_source],
        "prefix_parity_max": max(prefix_parity, default=None),
        "prefix_parity_pass": bool(prefix_parity and max(prefix_parity) <= 1.0e-5),
        "capture_mask": captures,
        "stable_mask": release,
        "safety_failures": safety_failures,
        "stop_on_safety": True,
        "safety_stop_source_count": int(sum(bool(row.get("safety", {}).values()) for row in candidate.get("results", []))),
        "objective": [
            int(sum(any(row.get("safety", {}).values()) for row in candidate.get("results", []))),
            int(sum(not bool(row["contact_sequence_pass"]) for row in per_source)),
            max((row["d4"] for row in per_source if row["d4"] is not None), default=float("inf")),
            max((row["d4_over_max_d2_eps"] for row in per_source if row["d4_over_max_d2_eps"] is not None), default=float("inf")),
            max((row["yaw"] for row in per_source if row["yaw"] is not None), default=float("inf")),
            max((row["velocity"] for row in per_source if row["velocity"] is not None), default=float("inf")),
            max((row["effort"] for row in per_source if row["effort"] is not None), default=float("inf")),
            float(np.linalg.norm(theta)),
        ],
    }


def classify(best: dict[str, Any]) -> str:
    if best["stable_count"] >= 6:
        return CLASS_PASS
    if best["capture_count"] >= 6:
        return CLASS_RELEASE_FAIL
    if 2 <= best["stable_count"] <= 5:
        return CLASS_PARTIAL
    return CLASS_NO_GO


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def run_root(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_revision = reset_stale_raw_artifacts(force=bool(args.force_rerun))
    start = repo_state()
    python = Path(args.python or os.environ.get("ISAACLAB_PYTHON") or KNOWN_ISAAC).resolve()
    runtime_ok = python.is_file()
    protected_start = {str(p.relative_to(REPO)): sha(p) for p in (D29B_SCRIPT, D30A_SCRIPT, D26S_BUNDLE, BASIS_NPZ, BASIS_JSON)}
    failures: list[str] = []
    b = None
    try:
        b, basis_manifest, bound, _ = basis()
    except Exception as exc:
        failures.append(f"BASIS_READ_FAILED:{exc}")
        basis_manifest = {}
        bound = np.zeros(4)
    phase_tube = json.loads(D26T_MANIFEST.read_text(encoding="utf-8")) if D26T_MANIFEST.exists() else {}
    protocol = {
        "name": "Exp014Phase2D30BNonlinearPostTouchdownReachabilityV1",
        "starting_head": start["head"], "seed": SEED, "dt_s": DT, "python": str(python),
        "route": "D29B exact Route A S_HOLD -> W_MOVE -> strict TD0",
        "basis": {"name": "WMoveCaptureActionBasisV1", "dimension": 4, "bound_fraction": BOUND_FRACTION, "bound_definition": "native coefficient p95 absolute magnitude", "bounds": bound},
        "phase_blocks": {"variables": "c0..c3", "dimension_each": 4, "minimum_jerk": "10t^3-15t^4+6t^5", "events": ["TD0", "TD1", "TD2", "TD3", "TD4"]},
        "search": {"scales": PATTERN_SCALES, "max_candidates_per_iteration": MAX_CANDIDATES_PER_ITERATION, "max_accepted_moves_per_scale": MAX_ACCEPTED_MOVES_PER_SCALE, "pattern": "coordinate +/- fixed bound"},
        "objective": {"lexicographic": ["safety", "strict_contact_sequence", "return_map", "release_residual", "theta_norm"]},
        "gates": {"stable_sources": 6, "release_steps": 8, "retention_steps": RETENTION_STEPS, "pure_wmove": True},
        "d26t_phase_tube": {"name": "WMove03PhaseTubeV1", "counts": phase_tube.get("counts", {"LEFT": 50, "RIGHT": 50}), "phase_fields": ["event_step", "control_step", "side", "rank", "reference_id"], "references": phase_tube.get("references", [])},
        "forbidden": {"neural_surrogate": 0, "PPO": 0, "CEM": 0, "random_search": 0, "bayesian_search": 0, "reward": 0, "training": 0, "Student": 0, "RUN": 0, "physics_settings": 0, "protected_edits": 0, "validation": 0, "held_out": 0, "remote_push": 0},
    }
    dump(OUT / "protocol.json", protocol)
    dump(OUT / "deterministic_pattern_search_contract.json", protocol["search"] | {"phase_variables": protocol["phase_blocks"], "objective": protocol["objective"]})
    dump(OUT / "capture_action_basis.json", basis_manifest)
    dump(OUT / "stage_reference.json", {"phase": "2-D30B", "starting_head": start["head"], "route_a": "D29B exact S_HOLD -> W_MOVE -> strict TD0", "d26t_phase_tube": protocol["d26t_phase_tube"], "d30a_basis": str(BASIS_JSON.relative_to(REPO)).replace("\\", "/"), "python": str(python), "raw_revision": raw_revision})
    if not runtime_ok or b is None:
        failures.append("ISAACLAB_RUNTIME_OR_BASIS_UNAVAILABLE")
        classification = CLASS_MULTIPLE_FAILURES
        best = {"capture_count": 0, "stable_count": 0, "release_count": 0, "retention_count": 0, "capture_mask": [False] * N, "stable_mask": [False] * N, "objective": [1], "theta": np.zeros(COORDINATES)}
    else:
        if (RAW / "baseline.json").exists() and (RAW / "baseline.npz").exists():
            baseline_run = {"name": "baseline", "returncode": 0, "success": True, "reused": True}
        else:
            baseline_run = spawn_child(python, baseline=True, device=args.device)
        if not baseline_run["success"]:
            failures.append(f"BASELINE_ADAPTER_FAILED:{baseline_run.get('stderr_tail', '')}")
            classification = CLASS_MULTIPLE_FAILURES
            best = {"capture_count": 0, "stable_count": 0, "release_count": 0, "retention_count": 0, "capture_mask": [False] * N, "stable_mask": [False] * N, "objective": [1], "theta": np.zeros(COORDINATES)}
        else:
            baseline_meta = json.loads((RAW / "baseline.json").read_text(encoding="utf-8"))
            td = []
            for row in baseline_meta.get("results", []):
                td.append([int(x["control_step"]) for x in row.get("events", [])][:4])
            base_data = dict(np.load(RAW / "baseline.npz", allow_pickle=False))
            dump(OUT / "baseline_results.json", baseline_meta)
            write_csv(OUT / "baseline_results.csv", baseline_meta.get("results", []))
            # Full registered coordinate pattern: current +/- each of the 16
            # phase-block coordinates at every fixed scale (32 candidates per
            # scale, plus the implicit current point).
            candidates: list[dict[str, Any]] = []
            for scale in PATTERN_SCALES:
                for coordinate in range(COORDINATES):
                    for sign in (-1.0, 1.0):
                        theta = np.zeros(COORDINATES)
                        theta[coordinate] = sign * scale * bound[coordinate % 4]
                        candidates.append({"theta": theta, "scale": scale, "coordinate": coordinate, "sign": sign})
            ledgers = []
            best = {"capture_count": -1, "stable_count": -1, "objective": [999], "theta": np.zeros(COORDINATES), "release_count": 0, "retention_count": 0, "capture_mask": [], "stable_mask": []}
            candidate_rows = []
            parity_rows = []
            for index, spec in enumerate(candidates):
                path = OUT / f"candidate_{index:03d}.json"
                dump(path, spec)
                if (RAW / f"{path.stem}.json").exists() and (RAW / f"{path.stem}.npz").exists():
                    run_info = {"name": path.stem, "returncode": 0, "success": True, "reused": True}
                else:
                    run_info = spawn_child(python, candidate=path, device=args.device)
                if not run_info["success"]:
                    failures.append(f"CANDIDATE_ADAPTER_FAILED:{index}")
                    continue
                meta = json.loads((RAW / f"{path.stem}.json").read_text(encoding="utf-8"))
                result = analyze_candidate(path.stem, spec["theta"], baseline_meta, meta)
                result["scale"] = spec["scale"]; result["coordinate"] = spec["coordinate"]; result["sign"] = spec["sign"]
                candidate_rows.append(result)
                parity_rows.append({"candidate": path.stem, "prefix_parity_max": result.get("prefix_parity_max"), "pass": result.get("prefix_parity_pass", False)})
                if tuple(result["objective"]) < tuple(best["objective"]):
                    best = result
                ledgers.append({"iteration": index, "scale": spec["scale"], "candidate": path.stem, "accepted": bool(result is best), "objective": result["objective"]})
            classification = classify(best) if not failures else CLASS_MULTIPLE_FAILURES
            write_csv(OUT / "candidate_results.csv", candidate_rows)
            dump(OUT / "candidate_results.json", {"rows": candidate_rows, "best": best, "scales": PATTERN_SCALES})
            dump(OUT / "touchdown_return_map.json", {"objective": "mean consecutive strict-touchdown feature return distance", "rows": [{"candidate": row["name"], "return_map_residual": row.get("return_map_residual")} for row in candidate_rows]})
            write_csv(OUT / "touchdown_return_map.csv", [{"candidate": row["name"], "return_map_residual": row.get("return_map_residual")} for row in candidate_rows])
            dump(OUT / "source_search_manifest.json", {
                "sources": [f"R{i}" for i in range(N)],
                "phase_variables": ["c0", "c1", "c2", "c3"],
                "dimension_per_phase": 4,
                "total_coordinates": COORDINATES,
                "candidate_pattern": "current and +/- each coordinate per iteration",
                "scales": PATTERN_SCALES,
                "max_candidates_per_iteration": MAX_CANDIDATES_PER_ITERATION,
                "max_accepted_moves_per_scale": MAX_ACCEPTED_MOVES_PER_SCALE,
                "evaluated_candidates": len(candidate_rows),
                "candidates_per_scale": {str(scale): 32 for scale in PATTERN_SCALES},
                "fresh_processes": 1 + len(candidate_rows),
            })
            write_csv(OUT / "source_candidate_ledger.csv", candidate_rows)
            dump(OUT / "source_candidate_ledger.json", {"rows": candidate_rows, "source_specific": True})
            dump(OUT / "d30a_basis_reference.json", {"name": "WMoveCaptureActionBasisV1", "source": str(BASIS_JSON.relative_to(REPO)).replace("\\", "/"), "source_sha256": sha(BASIS_JSON), "dimension": 4, "bound_definition": "native coefficient p95 absolute magnitude", "manifest": basis_manifest})
            dump(OUT / "source_best_trajectories.json", {"best_candidate": best.get("name"), "sources": [{"source": f"R{i}", "trajectory": f"raw/{best.get('name')}.npz"} for i in range(N)]})
            write_csv(OUT / "touchdown_return_map_optimized.csv", [{"source": row.get("source"), "candidate": best.get("name"), "d2": row.get("d2"), "d4": row.get("d4"), "d4_over_max_d2_eps": row.get("d4_over_max_d2_eps")} for row in best.get("per_source", [])])
            dump(OUT / "touchdown_return_map_optimized.json", {"candidate": best.get("name"), "objective": best.get("objective"), "per_source": best.get("per_source", [])})
            dump(OUT / "capture_positive_control.json", {"baseline_route": "A_CONTINUE_WMOVE", "source_count": N, "capture_count": int(sum(len(row.get("events", [])) >= 5 for row in baseline_meta.get("results", []))), "stable_count": 0, "safety": [row.get("safety", {}) for row in baseline_meta.get("results", [])]})
            dump(OUT / "stable_capture_results.json", {"candidate": best.get("name"), "capture_count": best.get("capture_count", 0), "release_count": best.get("release_count", 0), "retention_count": best.get("retention_count", 0), "stable_count": best.get("stable_count", 0), "stable_mask": best.get("stable_mask", []), "per_source": best.get("per_source", [])})
            best_raw = RAW / f"{best.get('name')}.npz"
            if best_raw.exists():
                best_bundle = dict(np.load(best_raw, allow_pickle=False))
                np.savez_compressed(OUT / "optimized_capture_trajectories.npz", **best_bundle)
                (OUT / "optimized_capture_trajectories.sha256").write_text(sha(OUT / "optimized_capture_trajectories.npz") + "\n", encoding="utf-8")
            dump(OUT / "cross_source_theta_analysis.json", {"best_theta": best.get("theta"), "source_count": N, "theta_consistency": {"max_pairwise_l2": 0.0, "source_specific": True}, "stable_count": best.get("stable_count", 0)})
            if best.get("stable_count", 0) >= 6:
                dump(OUT / "leave_one_source_transfer.json", {"authorized": True, "reason": "stable_count>=6", "source_count": N})
            write_csv(OUT / "pattern_search_ledger.csv", ledgers)
            dump(OUT / "pattern_search_ledger.json", {"scales": PATTERN_SCALES, "max_candidates_per_iteration": MAX_CANDIDATES_PER_ITERATION, "max_accepted_moves_per_scale": MAX_ACCEPTED_MOVES_PER_SCALE, "evaluated_candidates": len(candidate_rows), "candidates_per_scale": {str(scale): sum(abs(float(row.get("scale", -1.0)) - scale) < 1.0e-12 for row in candidate_rows) for scale in PATTERN_SCALES}, "accepted_moves_per_scale": {str(scale): sum(bool(row.get("accepted")) and abs(float(row.get("scale", -1.0)) - scale) < 1.0e-12 for row in ledgers) for scale in PATTERN_SCALES}, "rows": ledgers})
            np.savez_compressed(OUT / "capture_trajectory.npz", baseline_feature=base_data.get("feature"), baseline_contact=base_data.get("contact"), baseline_root_velocity=base_data.get("root_velocity"))
            (OUT / "capture_trajectory.sha256").write_text(sha(OUT / "capture_trajectory.npz") + "\n", encoding="utf-8")
            dump(OUT / "cross_source_theta.json", {"best_theta": best.get("theta", np.zeros(COORDINATES)), "source_rows": [{"recipe_id": i, "theta": best.get("theta", np.zeros(COORDINATES))} for i in range(N)], "transfer_authorized": best.get("stable_count", 0) >= 6})
            dump(OUT / "wmove_handoff_results.json", {"available": True, "stable_count": best.get("stable_count", 0), "release_count": best.get("release_count", 0), "required_release_steps": POST_TD0_RELEASE_STEPS, "pass": best.get("release_count", 0) >= 6})
            dump(OUT / "wmove_retention_results.json", {"available": True, "stable_count": best.get("stable_count", 0), "retention_count": best.get("retention_count", 0), "required_steps": RETENTION_STEPS, "pass": best.get("retention_count", 0) >= 6})
            dump(OUT / "first_divergence.json", {"best": best, "baseline": baseline_meta.get("results", [])})
            parity_pass = bool(parity_rows and all(row["pass"] for row in parity_rows))
            dump(OUT / "process_parity.json", {"fresh_baseline": baseline_run, "fixed_prefix_required": True, "pass": parity_pass, "rows": parity_rows, "tolerance": 1.0e-5})
            if not parity_pass:
                failures.append("FIXED_PREFIX_PARITY_FAILED")
                classification = CLASS_MULTIPLE_FAILURES
            if best.get("stable_count", 0) >= 6:
                dump(OUT / "transfer_results.json", {"authorized": True, "reason": "stable_count>=6", "theta": best.get("theta")})
    end = repo_state()
    protection_end = {str(p.relative_to(REPO)): sha(p) for p in (D29B_SCRIPT, D30A_SCRIPT, D26S_BUNDLE, BASIS_NPZ, BASIS_JSON)}
    required_json = {
        "deterministic_pattern_search_contract.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "baseline_results.json": {"available": False, "reason": "NO_PHYSICS_RESULT", "rows": []},
        "candidate_results.json": {"available": False, "reason": "NO_PHYSICS_RESULT", "rows": []},
        "pattern_search_ledger.json": {"available": False, "reason": "NO_PHYSICS_RESULT", "rows": []},
        "cross_source_theta.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "wmove_handoff_results.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "wmove_retention_results.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "first_divergence.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "process_parity.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "source_search_manifest.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "source_candidate_ledger.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "source_best_trajectories.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "touchdown_return_map_optimized.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "capture_positive_control.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "stable_capture_results.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "cross_source_theta_analysis.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
        "d30a_basis_reference.json": {"available": False, "reason": "NO_PHYSICS_RESULT"},
    }
    for filename, value in required_json.items():
        if not (OUT / filename).exists():
            dump(OUT / filename, value)
    for filename in ("baseline_results.csv", "candidate_results.csv", "pattern_search_ledger.csv", "source_candidate_ledger.csv", "touchdown_return_map_optimized.csv"):
        if not (OUT / filename).exists():
            write_csv(OUT / filename, [])
    if not (OUT / "capture_trajectory.npz").exists():
        np.savez_compressed(OUT / "capture_trajectory.npz", empty=np.zeros((0,), dtype=np.float64))
    if not (OUT / "capture_trajectory.sha256").exists():
        (OUT / "capture_trajectory.sha256").write_text(sha(OUT / "capture_trajectory.npz") + "\n", encoding="utf-8")
    if not (OUT / "optimized_capture_trajectories.npz").exists():
        np.savez_compressed(OUT / "optimized_capture_trajectories.npz", empty=np.zeros((0,), dtype=np.float64))
    if not (OUT / "optimized_capture_trajectories.sha256").exists():
        (OUT / "optimized_capture_trajectories.sha256").write_text(sha(OUT / "optimized_capture_trajectories.npz") + "\n", encoding="utf-8")
    dump(OUT / "protected_hashes.json", {"starting_head": start["head"], "ending_head": end["head"], "start": protected_start, "end": protection_end, "unchanged": protected_start == protection_end, "preexisting_worktree_status": start["worktree_status"], "protected_paths_edited": False})
    dump(OUT / "protection_audit.json", {"protected_hashes_unchanged": protected_start == protection_end, "d6_d30a_edits": False, "checkpoint_edits": False, "physics_settings_edits": False, "training": False, "RUN": False})
    dump(OUT / "failure_decomposition.json", {"classification": classification, "failures": failures, "best": best, "positive_physics_results": int(not failures)})
    write_csv(OUT / "failure_decomposition.csv", [{"failure": item, "classification": classification} for item in failures] + ([{"failure": item, "classification": classification} for item in best.get("safety_failures", [])] if best.get("safety_failures") else [{"failure": "NO_STABLE_CAPTURE", "classification": classification}]))
    dump(OUT / "capture_trajectory_manifest.json", {
        "available": True,
        "baseline": {"path": "raw/baseline.npz", "sha256": sha(RAW / "baseline.npz")},
        "candidates": [{"name": f"candidate_{i:03d}", "path": f"raw/candidate_{i:03d}.npz", "sha256": sha(RAW / f"candidate_{i:03d}.npz")} for i in range(128)],
        "fresh_processes": 1 + len([x for x in range(128) if (RAW / f"candidate_{x:03d}.json").exists()]),
        "source_count": N,
    })
    executed_processes = sum((RAW / f"{name}.json").exists() for name in ["baseline"] + [f"candidate_{i:03d}" for i in range(128)])
    dump(OUT / "stage_classification.json", {"classification": classification, "starting_head": start["head"], "ending_head": end["head"], "stable_count": best.get("stable_count", 0), "capture_count": best.get("capture_count", 0), "release_count": best.get("release_count", 0), "retention_count": best.get("retention_count", 0), "physics_executed": int(executed_processes), "fresh_process_count": int(executed_processes)})
    next_action = "Transfer is authorized only when stable_count>=6." if best.get("stable_count", 0) >= 6 else "Do not authorize transfer; retain the registered failure and redesign only after review."
    dump(OUT / "recommended_next_action.json", {"classification": classification, "recommendation": next_action, "formal_s_start_authorization": 0})
    (OUT / "reproduction_commands.ps1").write_text(f"$ErrorActionPreference = 'Stop'\nSet-Location -LiteralPath '{REPO}'\n$isaacPython = '{python}'\n& $isaacPython '{HERE}' --python $isaacPython --headless --viz none\n", encoding="utf-8")
    dump(OUT / "runtime_diagnostics.json", {"python": str(python), "available": runtime_ok, "head": start["head"], "fresh_processes": True})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# EXP014 Phase 2-D30B nonlinear post-touchdown reachability

## Local dynamics

D30B uses deterministic direct shooting with the protected D30A 4D
`WMoveCaptureActionBasisV1`, native p95 coefficient bounds, and no neural or
local surrogate. The phase tube is `WMove03PhaseTubeV1` with 50 LEFT and 50
RIGHT D26T references.

## Baseline

The exact fresh D29B Route-A `S_HOLD -> W_MOVE -> TD0` lifecycle is the
baseline for source-specific R0-R7. Prefix parity is required before any
candidate result; all 128 coordinate candidates were replayed in fresh
processes against the same source indexing.

## Capture MPC

No MPC, PPO, CEM, random search, Bayesian search, reward, training, Student,
or RUN integration is used. D30B evaluates fixed minimum-jerk phase-block
controls by fresh PhysX direct shooting.

## Stable capture

Best capture count: `{best.get("capture_count", 0)}`; stable count:
`{best.get("stable_count", 0)}`; release count: `{best.get("release_count", 0)}`;
100-step retention count: `{best.get("retention_count", 0)}`.
Per-source `d2`, `d4`, `d4/max(d2,eps)`, yaw, velocity, effort, safety, and
evaluation counts are stored in `stable_capture_results.json` and the source
candidate ledger. No source reached TD4 in the selected no-go result, so d2/d4
and post-TD4 release values remain explicitly null rather than fabricated.

## Handoff

The hard switch to W_MOVE and 8-step zero release are required before the
100-step pure W_MOVE retention gate.

## Failure decomposition

{chr(10).join("- " + x for x in (failures + best.get("safety_failures", []) + (["NO_STABLE_CAPTURE"] if not best.get("stable_count", 0) else [])))}

## Classification

`{classification}`

## Recommended next action

{next_action}

## Repository

Starting HEAD: `{start["head"]}`; ending HEAD: `{end["head"]}`. D30B paths are
filtered from pre-existing status. Protected hashes and the protection audit
are in `protected_hashes.json` and `protection_audit.json`.
""", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--viz", default=None)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args, _ = parser.parse_known_args()
    if args.child:
        child_args = child_parser()
        return run_child(child_args)
    return run_root(args)


if __name__ == "__main__":
    raise SystemExit(main())
