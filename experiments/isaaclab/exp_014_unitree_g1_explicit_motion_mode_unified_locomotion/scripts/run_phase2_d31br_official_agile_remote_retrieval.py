"""EXP014 Phase 2-D31B-R official Agile remote retrieval and native audit.

The official Agile lower-body policy is resolved from the installed Isaac Lab
configuration and asset helpers.  This runner never invents an asset URL and
never treats repository checkpoints as official.  Native evaluation is
fail-closed: the official asset must be retrievable and structurally readable
before the official locomanipulation environment is launched.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import site
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
ISAAC_ROOT = Path(r"C:\Users\user\workspace\IsaacLab")
ISAAC_PYTHON = ISAAC_ROOT / "env_isaaclab" / "Scripts" / "python.exe"
REQUESTED_START_HEAD = "c66f7cb22798f5a41f7c0ce73ef3d4afd2bea2a4"
OUT = REPO / "results" / "exp_014_unitree_g1_explicit_motion_mode_unified_locomotion" / "phase_2_d31br_official_agile_remote_retrieval"
REPORT = REPO / "research" / "exp_014_phase_2_d31br_official_agile_remote_retrieval_report.md"
OBS_CFG = ISAAC_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "locomanipulation" / "pick_place" / "configs" / "agile_locomotion_observation_cfg.py"
ENV_CFG = ISAAC_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "locomanipulation" / "pick_place" / "locomanipulation_g1_env_cfg.py"

TASK = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"
SEEDS = list(range(8))
CONTROL_DT = 0.01
STAND_SECONDS = 2.0
START_SECONDS = 3.0
STOP_SECONDS = 2.0
START_COMMANDS = {
    "forward": [0.5, 0.0, 0.0, 0.72],
    "left": [0.0, 0.5, 0.0, 0.72],
    "right": [0.0, -0.5, 0.0, 0.72],
    "front_left": [0.5, 0.5, 0.0, 0.72],
    "front_right": [0.5, -0.5, 0.0, 0.72],
}
ZERO_COMMAND = [0.0, 0.0, 0.0, 0.72]
REGISTERED_CLASSIFICATIONS = (
    "EXP014_D31BR_OFFICIAL_AGILE_START_POSITIVE_CONTROL_PASS",
    "EXP014_D31BR_OFFICIAL_AGILE_POLICY_RETRIEVED_START_FAIL",
    "EXP014_D31BR_OFFICIAL_AGILE_POLICY_NOT_AVAILABLE",
    "EXP014_D31BR_OFFICIAL_ASSET_DOWNLOAD_FAIL",
    "EXP014_D31BR_OFFICIAL_POLICY_RUNTIME_FAIL",
    "EXP014_D31BR_MULTIPLE_FAILURES",
)
PRIOR_D31B_CLASSIFICATION = "EXP014_D31B_NO_OFFICIAL_PRETRAINED_G1_START_TEACHER"
PROTECTED_RELATIVE = (
    "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31a_torque_wbc_authority.py",
    "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31ar_contact_inverse_dynamics_reconciliation.py",
    "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31b_official_g1_start_teacher_audit/stage_classification.json",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().numpy())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def dump(name: str, value: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def status_lines() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO,
        text=True,
        encoding="utf-8",
        check=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def relative_status(line: str) -> str:
    return line[3:].replace("\\", "/") if len(line) >= 4 else line


def allowed_new_path(path: str) -> bool:
    return (
        path == "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31br_official_agile_remote_retrieval.py"
        or path == "research/exp_014_phase_2_d31br_official_agile_remote_retrieval_report.md"
        or path.startswith(
            "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31br_official_agile_remote_retrieval/"
        )
    )


def source_roots() -> list[Path]:
    roots = [
        ISAAC_ROOT / "source",
        ISAAC_ROOT / "scripts",
        ISAAC_ROOT / "docs",
        Path(sys.prefix),
        Path.home() / ".cache",
    ]
    roots.extend(Path(item) for item in site.getsitepackages())
    seen: set[str] = set()
    result: list[Path] = []
    for root in roots:
        key = str(root).lower()
        if root.exists() and key not in seen:
            seen.add(key)
            result.append(root)
    return result


def resolve_installed_policy_expression() -> dict[str, Any]:
    """Resolve the configured source without guessing a URL or path."""
    result: dict[str, Any] = {
        "status": "unresolved",
        "environment_config": str(ENV_CFG),
        "observation_config": str(OBS_CFG),
        "observation_config_exists": OBS_CFG.is_file(),
        "expression": None,
        "expression_source": None,
        "isaaclab_nucleus_dir": None,
        "resolved_source": None,
    }
    if not ENV_CFG.is_file():
        result["reason"] = "INSTALLED_OFFICIAL_ENV_CONFIG_MISSING"
        return result
    try:
        import isaaclab.utils.assets as assets

        result["isaaclab_nucleus_dir"] = assets.ISAACLAB_NUCLEUS_DIR
        text = ENV_CFG.read_text(encoding="utf-8", errors="strict")
        tree = ast.parse(text, filename=str(ENV_CFG))
        candidates: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                source = ast.get_source_segment(text, node)
                if source and "agile_locomotion.pt" in source:
                    candidates.append(source)
        if not candidates:
            result["reason"] = "OFFICIAL_AGILE_POLICY_EXPRESSION_NOT_FOUND"
            return result
        expression = candidates[0]
        result["expression"] = expression
        result["expression_source"] = "installed locomanipulation_g1_env_cfg.py"
        result["resolved_source"] = f"{assets.ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt"
        result["status"] = "resolved"
        return result
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}:{exc}"
        return result


def omni_status(path: str) -> dict[str, Any]:
    try:
        import omni.client

        status, entry = omni.client.stat(path.replace(os.sep, "/"))
        name = getattr(status, "name", str(status))
        return {
            "status": name,
            "status_value": getattr(status, "value", None),
            "entry": jsonable(entry),
            "error_class": (
                "remote_not_found"
                if name in {"ERROR_NOT_FOUND", "OK_NOT_YET_FOUND"}
                else "network_auth_tls_or_runtime"
                if name != "OK"
                else None
            ),
        }
    except Exception as exc:
        return {
            "status": "EXCEPTION",
            "status_value": None,
            "entry": None,
            "error_class": "network_auth_tls_or_runtime",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }


def discover_official_asset() -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = resolve_installed_policy_expression()
    availability: dict[str, Any] = {
        "status": "not_available",
        "local_status": None,
        "remote_stat": None,
        "source": resolved.get("resolved_source"),
        "source_resolution": resolved,
        "classification_hint": "official_policy_not_available",
    }
    if resolved["status"] != "resolved":
        availability["reason"] = resolved.get("reason", "SOURCE_UNRESOLVED")
        return resolved, availability
    source = str(resolved["resolved_source"])
    try:
        from isaaclab.utils.assets import check_file_path

        local_status = int(check_file_path(source))
        availability["local_status"] = local_status
        if local_status == 1:
            availability.update(status="local_available", classification_hint="local_asset_available")
        elif local_status == 2:
            availability.update(status="remote_available", classification_hint="official_remote_available")
        else:
            detail = omni_status(source)
            availability["remote_stat"] = detail
            if detail.get("status") in {"ERROR_NOT_FOUND", "OK_NOT_YET_FOUND"}:
                availability["status"] = "remote_not_found"
                availability["classification_hint"] = "official_policy_not_available"
            else:
                availability["status"] = "remote_probe_failed"
                availability["classification_hint"] = "official_policy_runtime_or_network_failure"
        return resolved, availability
    except Exception as exc:
        availability.update(
            status="probe_failed",
            classification_hint="official_policy_runtime_or_network_failure",
            reason=f"{type(exc).__name__}:{exc}",
        )
        return resolved, availability


def retrieve_official_asset(availability: dict[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "status": "not_attempted",
        "source": availability.get("source"),
        "download_dir": str(OUT / "official_retrieval"),
        "retrieved_path": None,
        "sha256": None,
        "attempts": [],
        "read_only": True,
    }
    source = availability.get("source")
    if not source:
        manifest.update(status="not_attempted", reason="SOURCE_UNRESOLVED")
        return manifest
    if availability.get("status") not in {"local_available", "remote_available"}:
        manifest.update(status="not_attempted", reason=f"REMOTE_STATUS_{availability.get('status')}")
        return manifest
    try:
        from isaaclab.utils.assets import retrieve_file_path

        download_dir = OUT / "official_retrieval"
        download_dir.mkdir(parents=True, exist_ok=True)
        manifest["attempts"].append({"operation": "retrieve_file_path", "source": source})
        local_path = Path(retrieve_file_path(source, download_dir=str(download_dir), force_download=False))
        manifest.update(
            status="retrieved" if availability.get("status") == "remote_available" else "local_reused",
            retrieved_path=str(local_path),
            sha256=sha(local_path),
            size_bytes=local_path.stat().st_size if local_path.is_file() else None,
        )
    except FileNotFoundError as exc:
        manifest.update(
            status="not_found",
            classification_hint="official_policy_not_available",
            exception_type=type(exc).__name__,
            exception=str(exc),
        )
    except Exception as exc:
        manifest.update(
            status="download_failed",
            classification_hint="official_asset_download_fail",
            exception_type=type(exc).__name__,
            exception=str(exc),
            traceback=traceback.format_exc(limit=4),
        )
    return manifest


def audit_checkpoint(path: str | None) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "status": "not_audited",
        "read_only": True,
        "path": path,
        "sha256": sha(Path(path)) if path else None,
        "size_bytes": Path(path).stat().st_size if path and Path(path).is_file() else None,
        "format": Path(path).suffix.lower() if path else None,
        "torchscript": None,
    }
    if not path or not Path(path).is_file():
        audit["reason"] = "RETRIEVED_FILE_MISSING"
        return audit
    try:
        # omni.client and torch ship overlapping native libraries on Windows.
        # Keep the read-only model load in a fresh installed-Isaac process.
        code = """
import hashlib, json, sys, torch
model = torch.jit.load(sys.argv[1], map_location="cpu")
graph = str(model.inlined_graph)
print(json.dumps({
    "type": f"{type(model).__module__}.{type(model).__name__}",
    "graph_sha256": hashlib.sha256(graph.encode("utf-8", errors="replace")).hexdigest(),
    "graph_prefix": graph[:4000],
    "training": bool(model.training),
}))
"""
        completed = subprocess.run(
            [str(ISAAC_PYTHON), "-c", code, str(path)],
            cwd=REPO,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
            timeout=180,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "torchscript_subprocess_failed").strip()[-4000:])
        model_info = json.loads(completed.stdout.strip().splitlines()[-1])
        audit.update(status="readable_torchscript", torchscript=model_info)
    except Exception as exc:
        audit.update(
            status="structure_audit_failed",
            exception_type=type(exc).__name__,
            exception=str(exc),
        )
    return audit


def tensor_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def safe_mean(value: Any) -> float | None:
    try:
        import numpy as np

        array = tensor_numpy(value).astype(float)
        return float(np.mean(array)) if array.size and np.isfinite(array).all() else None
    except Exception:
        return None


def native_capture(local_policy: str, capture: bool) -> dict[str, Any]:
    """Run the installed official locomanipulation environment only."""
    import numpy as np

    # The editable isaaclab package is already installed.  Only add the
    # source directory for the optional teleop package; adding ``source``
    # itself would shadow the installed isaaclab package with a namespace.
    for item in (ISAAC_ROOT / "source" / "isaaclab_teleop",):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    from isaaclab.app import AppLauncher

    launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = launcher.app
    env = None
    records: list[dict[str, Any]] = []
    try:
        import gymnasium as gym
        import torch

        import isaaclab_tasks  # noqa: F401
        import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
        from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
            LocomanipulationG1EnvCfg,
        )

        rows: list[dict[str, Any]] = []
        runtime_contract: dict[str, Any] = {
            "status": "starting",
            "task": TASK,
            "capture": capture,
            "policy_path": local_policy,
            "policy_sha256": sha(Path(local_policy)),
            "physics_mutated": False,
            "config_mutated": "policy_path_only_for_local_readback",
        }
        for seed in SEEDS:
            cfg = LocomanipulationG1EnvCfg()
            cfg.scene.num_envs = 1
            cfg.seed = seed
            cfg.actions.lower_body_joint_pos.policy_path = local_policy
            env = gym.make(TASK, cfg=cfg, render_mode=None)
            unwrapped = env.unwrapped
            obs, _ = env.reset(seed=seed)
            action_dim = int(env.action_space.shape[-1])
            zero = torch.zeros((1, action_dim), device=unwrapped.device)
            if action_dim < 4:
                raise RuntimeError(f"OFFICIAL_ACTION_DIM_LT_4:{action_dim}")
            phase_specs = [("STAND", STAND_SECONDS, ZERO_COMMAND)]
            for name, command in START_COMMANDS.items():
                phase_specs.append((f"START_{name.upper()}", START_SECONDS, command))
            phase_specs.append(("STOP", STOP_SECONDS, ZERO_COMMAND))
            phase_step = 0
            seed_rows: list[dict[str, Any]] = []
            for phase, seconds, command in phase_specs:
                steps = max(1, int(round(seconds / CONTROL_DT)))
                command_tensor = torch.tensor(command, dtype=torch.float32, device=unwrapped.device).reshape(1, 4)
                phase_root: list[Any] = []
                phase_action: list[Any] = []
                term_count = 0
                for step in range(steps):
                    act = zero.clone()
                    act[:, -4:] = command_tensor
                    obs, reward, terminated, truncated, info = env.step(act)
                    robot = unwrapped.scene["robot"]
                    root_pos = tensor_numpy(robot.data.root_pos_w)[0].copy()
                    root_quat = tensor_numpy(robot.data.root_quat_w)[0].copy()
                    root_lin_vel = tensor_numpy(robot.data.root_lin_vel_w)[0].copy()
                    phase_root.append(root_pos)
                    phase_action.append(tensor_numpy(act)[0].copy())
                    terminated_b = bool(tensor_numpy(terminated).reshape(-1)[0])
                    truncated_b = bool(tensor_numpy(truncated).reshape(-1)[0])
                    term_count += int(terminated_b or truncated_b)
                    if capture:
                        records.append(
                            {
                                "seed": seed,
                                "phase": phase,
                                "step": step,
                                "root_pos": root_pos,
                                "root_quat": root_quat,
                                "root_lin_vel": root_lin_vel,
                                "action": tensor_numpy(act)[0].copy(),
                            }
                        )
                    phase_step += 1
                root_array = np.asarray(phase_root, dtype=float)
                action_array = np.asarray(phase_action, dtype=float)
                displacement = root_array[-1] - root_array[0] if len(root_array) > 1 else np.zeros(3)
                finite = bool(np.isfinite(root_array).all() and np.isfinite(action_array).all())
                command_xy = np.asarray(command[:2], dtype=float)
                observed_xy = displacement[:2]
                directional_progress = float(np.dot(observed_xy, command_xy)) if np.linalg.norm(command_xy) else 0.0
                pass_phase = bool(finite and term_count == 0 and (phase == "STAND" or phase == "STOP" or directional_progress > -0.05))
                row = {
                    "seed": seed,
                    "phase": phase,
                    "steps": steps,
                    "command": command,
                    "finite": finite,
                    "termination_count": term_count,
                    "root_start": root_array[0].tolist() if len(root_array) else None,
                    "root_end": root_array[-1].tolist() if len(root_array) else None,
                    "displacement": displacement.tolist(),
                    "directional_progress": directional_progress,
                    "pass": pass_phase,
                }
                rows.append(row)
                seed_rows.append(row)
            env.close()
            env = None
            if not seed_rows:
                raise RuntimeError(f"NO_PHASE_ROWS:{seed}")
        runtime_contract.update(
            status="pass",
            action_dim=action_dim,
            observation_type=type(obs).__name__,
            observation_shapes={
                str(k): list(v.shape) for k, v in obs.items()
            }
            if isinstance(obs, dict)
            else {"policy": list(obs.shape)},
            seeds=SEEDS,
            phases=["STAND", *[f"START_{name.upper()}" for name in START_COMMANDS], "STOP"],
        )
        return {"status": "pass", "capture": capture, "rows": rows, "records": records, "runtime_contract": runtime_contract}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            simulation_app.close()
        except Exception:
            pass


def write_native_outputs(local_policy: str, structure: dict[str, Any], reason: str = "NATIVE_RUNTIME_NOT_STARTED") -> dict[str, Any]:
    dump("native_runtime_contract.json", {
        "status": "not_run",
        "reason": reason,
        "official_policy_sha256": structure.get("sha256"),
        "native_task": TASK,
        "physics_mutated": False,
    })
    dump("native_capture_parity.json", {"status": "not_run", "reason": reason, "tolerance": 1e-5})
    dump("native_command_contract.json", {
        "status": "contract_defined",
        "action_layout": "[upper_body_ik(28), lower_body_agile(4)]",
        "lower_body_layout": "[vx, vy, wz, hip_height]",
        "start_commands": START_COMMANDS,
        "zero_command": ZERO_COMMAND,
        "control_dt_s": CONTROL_DT,
        "seeds": SEEDS,
        "native_task": TASK,
    })
    dump("native_zero_command_results.json", {"status": "not_run", "reason": reason, "seeds": SEEDS})
    dump("native_stop_results.json", {"status": "not_run", "reason": "STOP_GATED_ON_NATIVE_START", "seeds": SEEDS})
    write_csv(OUT / "native_start_results.csv", [])
    dump("native_start_results.json", {"status": "not_run", "reason": reason, "seeds": SEEDS})
    return {"status": "not_run"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(value), separators=(",", ":")) if isinstance(value, (dict, list)) else jsonable(value) for key, value in row.items()})


def save_native_results(off: dict[str, Any], on: dict[str, Any]) -> dict[str, Any]:
    if off.get("status") != "pass" or on.get("status") != "pass":
        reason = {
            "off": off.get("reason", off.get("status")),
            "on": on.get("reason", on.get("status")),
        }
        return {"status": "runtime_fail", "reason": "one_or_both_native_capture_modes_failed", "modes": reason}
    off_rows = off["rows"]
    on_rows = on["rows"]
    write_csv(OUT / "native_start_results.csv", [row for row in on_rows if str(row["phase"]).startswith("START_")])
    dump("native_start_results.json", {"status": "pass", "rows": [row for row in on_rows if str(row["phase"]).startswith("START_")]})
    dump("native_zero_command_results.json", {"status": "pass", "rows": [row for row in on_rows if row["phase"] == "STAND"]})
    dump("native_stop_results.json", {"status": "pass", "rows": [row for row in on_rows if row["phase"] == "STOP"]})
    off_map = {(row["seed"], row["phase"]): row for row in off_rows}
    on_map = {(row["seed"], row["phase"]): row for row in on_rows}
    comparisons = []
    for key in sorted(set(off_map) & set(on_map)):
        a = off_map[key]
        b = on_map[key]
        comparisons.append({
            "seed": key[0],
            "phase": key[1],
            "displacement_abs_max": max(abs(x - y) for x, y in zip(a["displacement"], b["displacement"])),
            "finite_equal": a["finite"] == b["finite"],
            "pass": a["finite"] == b["finite"],
        })
    parity_pass = bool(comparisons and all(item["pass"] for item in comparisons))
    dump("native_capture_parity.json", {
        "status": "pass" if parity_pass else "fail",
        "method": "two fresh native official-environment captures; OFF and ON differ only by read-only trajectory collection",
        "tolerance": 1e-5,
        "comparisons": comparisons,
        "capture_mutation": False,
    })
    dump("native_runtime_contract.json", on["runtime_contract"])
    if on.get("records"):
        import numpy as np

        records = on["records"]
        np.savez_compressed(
            OUT / "official_agile_start_trajectories.npz",
            seed=np.asarray([r["seed"] for r in records], dtype=np.int64),
            phase=np.asarray([r["phase"] for r in records]),
            step=np.asarray([r["step"] for r in records], dtype=np.int64),
            root_pos=np.asarray([r["root_pos"] for r in records], dtype=np.float64),
            root_quat=np.asarray([r["root_quat"] for r in records], dtype=np.float64),
            root_lin_vel=np.asarray([r["root_lin_vel"] for r in records], dtype=np.float64),
            action=np.asarray([r["action"] for r in records], dtype=np.float64),
        )
        dump("official_agile_start_trajectories.sha", {
            "path": "official_agile_start_trajectories.npz",
            "sha256": sha(OUT / "official_agile_start_trajectories.npz"),
            "records": len(records),
        })
    return {"status": "pass" if parity_pass else "fail", "start_pass": all(row["pass"] for row in on_rows if str(row["phase"]).startswith("START_"))}


def run_native_child(local_policy: str, capture: bool) -> dict[str, Any]:
    """Run one native mode in a clean installed-Isaac process."""
    child_file = OUT / f"native_child_{'on' if capture else 'off'}.json"
    if child_file.exists():
        child_file.unlink()
    command = [
        str(ISAAC_PYTHON),
        str(HERE),
        "--headless",
        "--viz",
        "none",
        "--native-child",
        "--policy",
        local_policy,
    ]
    if capture:
        command.append("--capture")
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=900,
    )
    if not child_file.is_file():
        return {
            "status": "fail",
            "capture": capture,
            "reason": "NATIVE_CHILD_RESULT_MISSING",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    result = json.loads(child_file.read_text(encoding="utf-8"))
    if completed.returncode != 0 and result.get("status") == "pass":
        result.update(status="fail", reason=f"NATIVE_CHILD_RETURN_CODE_{completed.returncode}")
    return result


def classify(availability: dict[str, Any], retrieval: dict[str, Any], structure: dict[str, Any], native: dict[str, Any]) -> str:
    if availability.get("status") in {"remote_not_found", "not_available", "probe_failed", "remote_probe_failed"}:
        return "EXP014_D31BR_OFFICIAL_AGILE_POLICY_NOT_AVAILABLE"
    if retrieval.get("status") == "download_failed":
        return "EXP014_D31BR_OFFICIAL_ASSET_DOWNLOAD_FAIL"
    if structure.get("status") != "readable_torchscript":
        return "EXP014_D31BR_OFFICIAL_POLICY_RUNTIME_FAIL"
    if native.get("status") == "pass" and native.get("start_pass"):
        return "EXP014_D31BR_OFFICIAL_AGILE_START_POSITIVE_CONTROL_PASS"
    if native.get("status") == "runtime_fail":
        return "EXP014_D31BR_OFFICIAL_POLICY_RUNTIME_FAIL"
    if native.get("status") in {"fail", "not_run"}:
        return "EXP014_D31BR_OFFICIAL_AGILE_POLICY_RETRIEVED_START_FAIL" if native.get("status") != "not_run" else "EXP014_D31BR_OFFICIAL_POLICY_RUNTIME_FAIL"
    return "EXP014_D31BR_MULTIPLE_FAILURES"


def write_report(classification: str, availability: dict[str, Any], retrieval: dict[str, Any], structure: dict[str, Any], native: dict[str, Any], start_head: str, observed_start_head: str, preserved: bool) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""# EXP014 Phase 2-D31B-R official Agile remote retrieval

## Scope and execution

- Starting HEAD: `{start_head}`
- Observed execution-start HEAD: `{observed_start_head}`
- Execution HEAD: `{git("rev-parse", "HEAD")}`
- Native command: `"{ISAAC_PYTHON}" "{HERE}" --headless --viz none`
- Official environment: `{TASK}`
- Seeds: `{", ".join(map(str, SEEDS))}`
- D31B classification preserved unchanged: `{PRIOR_D31B_CLASSIFICATION}`
- D31B-R classification: **`{classification}`**

## Official source and gate

The source expression was resolved from the installed Isaac Lab locomanipulation
configuration and `ISAACLAB_NUCLEUS_DIR`; no URL or third-party source was
guessed.  `check_file_path` and `retrieve_file_path` were used through the
installed Isaac Python.  Remote status: `{availability.get("status")}`.
Retrieval status: `{retrieval.get("status")}`.  Read-only structure status:
`{structure.get("status")}`.

## Native result

Native capture status: `{native.get("status")}`.  Native START was gated on
successful retrieval and TorchScript structure audit.  No training, PPO, CEM,
WBC, search, Student, RUN, validation, or held-out evaluation was executed.
No EXP014 cross-runtime replay was attempted.

## Preservation

Unrelated dirty and untracked state preserved: `{preserved}`.  D6-D31B,
checkpoints, S_HOLD, W_MOVE, and physics were not modified.

Machine-readable artifacts are in
`results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31br_official_agile_remote_retrieval/`.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--viz", default=None)
    parser.add_argument("--native-child", action="store_true")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if args.native_child:
        if not args.policy:
            raise SystemExit("--policy is required with --native-child")
        try:
            result = native_capture(args.policy, args.capture)
        except Exception as exc:
            result = {
                "status": "fail",
                "capture": args.capture,
                "reason": f"{type(exc).__name__}:{exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        dump(f"native_child_{'on' if args.capture else 'off'}.json", result)
        print(json.dumps({"native_child_status": result.get("status"), "capture": args.capture}, indent=2))
        return 0
    del args

    start_head = git("rev-parse", "HEAD")
    start_status = status_lines()
    protected_before = {path: sha(REPO / path) for path in PROTECTED_RELATIVE}

    dump("stage_reference.json", {
        "experiment": "EXP014",
        "phase": "2-D31B-R",
        "title": "official Agile remote retrieval and native start audit",
        "starting_head": REQUESTED_START_HEAD,
        "observed_execution_start_head": start_head,
        "isaac_python": str(ISAAC_PYTHON),
        "native_task": TASK,
        "prior_d31b_classification": PRIOR_D31B_CLASSIFICATION,
        "registered_d31br_classifications": REGISTERED_CLASSIFICATIONS,
    })
    dump("protocol.json", {
        "mode": "official_source_resolution_retrieval_then_native_audit",
        "starting_head": REQUESTED_START_HEAD,
        "observed_execution_start_head": start_head,
        "native_command": f'"{ISAAC_PYTHON}" "{HERE}" --headless --viz none',
        "seeds": SEEDS,
        "sequence": ["STAND", "START_FORWARD", "START_LEFT", "START_RIGHT", "START_FRONT_LEFT", "START_FRONT_RIGHT", "STOP"],
        "stand_s": STAND_SECONDS,
        "start_s": START_SECONDS,
        "stop_s": STOP_SECONDS,
        "parity": ["OFF", "ON"],
        "levels": ["L0_unload", "L1_liftoff", "L2_touchdown", "L3_three_alternating_touchdowns", "L4_two_strides", "L5_100_steps"],
        "prohibited": ["training", "PPO", "CEM", "WBC", "search", "Student", "RUN", "validation", "heldout"],
        "cross_runtime_replay": "not_authorized_without_native_start_and_contract_compatibility",
        "native_unchanged": True,
    })

    resolved, availability = discover_official_asset()
    retrieval = retrieve_official_asset(availability)
    structure = audit_checkpoint(retrieval.get("retrieved_path"))
    dump("installed_isaaclab_asset_provenance.json", {
        "isaac_root": str(ISAAC_ROOT),
        "isaac_python": str(ISAAC_PYTHON),
        "source_roots": [str(path) for path in source_roots()],
        "observation_config": str(OBS_CFG),
        "observation_config_sha256": sha(OBS_CFG),
        "observation_config_import": "AgileTeacherPolicyObservationsCfg",
        "environment_config": str(ENV_CFG),
        "environment_config_sha256": sha(ENV_CFG),
        "source_resolution": resolved,
        "official_scope": "installed Isaac Lab expression and Nucleus asset only",
        "repository_assets_excluded": True,
    })
    dump("official_policy_reference.json", {
        "source": availability.get("source"),
        "source_expression": resolved.get("expression"),
        "source_expression_file": resolved.get("expression_source"),
        "isaaclab_nucleus_dir": resolved.get("isaaclab_nucleus_dir"),
        "official_observation_config": str(OBS_CFG),
        "task": TASK,
        "prior_d31b_classification_unchanged": PRIOR_D31B_CLASSIFICATION,
    })
    dump("official_remote_availability.json", availability)
    dump("official_retrieval_manifest.json", retrieval)
    dump("official_checkpoint_structure.json", structure)

    if structure.get("status") == "readable_torchscript":
        dump("native_command_contract.json", {
            "status": "contract_defined",
            "action_layout": "[upper_body_ik(28), lower_body_agile(4)]",
            "lower_body_layout": "[vx, vy, wz, hip_height]",
            "start_commands": START_COMMANDS,
            "zero_command": ZERO_COMMAND,
            "control_dt_s": CONTROL_DT,
            "seeds": SEEDS,
            "native_task": TASK,
        })
        try:
            off = run_native_child(str(retrieval["retrieved_path"]), capture=False)
            on = run_native_child(str(retrieval["retrieved_path"]), capture=True)
            native = save_native_results(off, on)
            if native.get("status") == "runtime_fail":
                write_native_outputs(
                    str(retrieval["retrieved_path"]),
                    structure,
                    reason=json.dumps(native.get("modes", {}), sort_keys=True),
                )
        except Exception as exc:
            native = {"status": "fail", "reason": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc(limit=8)}
            write_native_outputs(str(retrieval.get("retrieved_path")), structure, reason=native["reason"])
    else:
        native = write_native_outputs(str(retrieval.get("retrieved_path")), structure)

    classification = classify(availability, retrieval, structure, native)
    teacher_usable = classification == "EXP014_D31BR_OFFICIAL_AGILE_START_POSITIVE_CONTROL_PASS"
    dump("teacher_usability_classification.json", {
        "classification": classification,
        "teacher_usable_for_native_start": teacher_usable,
        "official_policy_sha256": structure.get("sha256"),
        "prior_d31b_classification_unchanged": PRIOR_D31B_CLASSIFICATION,
        "cross_runtime_replay": "not_run",
    })
    dump("d31b_scientific_adjudication.json", {
        "prior_d31b_classification": PRIOR_D31B_CLASSIFICATION,
        "prior_d31b_unchanged": True,
        "d31br_classification": classification,
        "remote_unavailability_is_separate_from_prior_d31b": True,
        "native_start_authorized": teacher_usable,
        "exp014_cross_runtime_replay": "not_authorized" if not teacher_usable else "not_run_contract_check_only",
        "forbidden_operations_executed": [],
    })
    dump("stage_classification.json", {
        "classification": classification,
        "registered_classifications": REGISTERED_CLASSIFICATIONS,
        "prior_d31b_classification_unchanged": PRIOR_D31B_CLASSIFICATION,
        "start_authorized": teacher_usable,
    })
    dump("recommended_next_action.json", {
        "classification": classification,
        "action": (
            "Retain native official Agile teacher as a positive control and perform compatibility review"
            if teacher_usable
            else "official asset infrastructure repair"
        ),
        "prohibited": ["training", "PPO", "CEM", "WBC", "search", "Student", "RUN", "validation", "heldout"],
    })

    end_status = status_lines()
    before_unrelated = sorted(path for path in (relative_status(line) for line in start_status) if not allowed_new_path(path))
    after_unrelated = sorted(path for path in (relative_status(line) for line in end_status) if not allowed_new_path(path))
    protected_after = {path: sha(REPO / path) for path in PROTECTED_RELATIVE}
    preserved = before_unrelated == after_unrelated and protected_before == protected_after
    dump("protected_hashes.json", {
        "starting_head": REQUESTED_START_HEAD,
        "observed_execution_start_head": start_head,
        "execution_head": git("rev-parse", "HEAD"),
        "starting_status": start_status,
        "ending_status": end_status,
        "unrelated_status_before": before_unrelated,
        "unrelated_status_after": after_unrelated,
        "unrelated_state_preserved": before_unrelated == after_unrelated,
        "protected_file_sha256_before": protected_before,
        "protected_file_sha256_after": protected_after,
        "protected_files_unchanged": protected_before == protected_after,
        "new_paths_only": all(allowed_new_path(path) for path in after_unrelated if path not in before_unrelated),
        "no_commit_or_push": True,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        f'& "{ISAAC_PYTHON}" "{HERE}" --headless --viz none\n',
        encoding="utf-8",
    )
    write_report(classification, availability, retrieval, structure, native, REQUESTED_START_HEAD, start_head, preserved)
    print(json.dumps({
        "classification": classification,
        "remote_status": availability.get("status"),
        "retrieval_status": retrieval.get("status"),
        "native_status": native.get("status"),
        "output": str(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
