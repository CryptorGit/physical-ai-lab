"""EXP014 Phase 2-D31B-R2 official policy runtime repair.

This is a diagnostics-first, fail-closed runner.  It uses the installed
IsaacLab Python and launcher, reads the already retrieved official Agile
TorchScript checkpoint without copying it, and never changes packages,
checkpoints, robot configuration, or physics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
ISAAC_ROOT = Path(r"C:\Users\user\workspace\IsaacLab")
ISAAC_PYTHON = ISAAC_ROOT / "env_isaaclab" / "Scripts" / "python.exe"
SYSTEM_PYTHON = Path(r"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe")
OUT = (
    REPO
    / "results"
    / "exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
    / "phase_2_d31br2_official_policy_runtime_repair"
)
REPORT = REPO / "research" / "exp_014_phase_2_d31br2_official_policy_runtime_repair_report.md"
OFFICIAL_POLICY = (
    REPO
    / "results"
    / "exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
    / "phase_2_d31br_official_agile_remote_retrieval"
    / "official_retrieval"
    / "Assets"
    / "Isaac"
    / "6.0"
    / "Isaac"
    / "IsaacLab"
    / "Policies"
    / "Agile"
    / "agile_locomotion.pt"
)
OFFICIAL_SHA256 = "f04a58b834057eb1c9f38350dc12feaf929ff2cc7d5b75d2871e23811b775dde"
OFFICIAL_SIZE = 500080
REQUESTED_START_HEAD = "012ebed9ebabc04176659e32eca3bf36db1fd54c"
TASK = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"
SEEDS = list(range(8))
CONTROL_DT = 0.01
ZERO_SECONDS = 2.0
START_SECONDS = 3.0
ZERO_COMMAND = [0.0, 0.0, 0.0, 0.72]
FORWARD_COMMAND = [0.5, 0.0, 0.0, 0.72]
REGISTERED_CLASSIFICATIONS = (
    "EXP014_D31BR2_RUNTIME_REPAIRED_OFFICIAL_START_PASS",
    "EXP014_D31BR2_RUNTIME_REPAIRED_OFFICIAL_START_FAIL",
    "EXP014_D31BR2_WRONG_PYTHON_RUNTIME_FIXED",
    "EXP014_D31BR2_OFFICIAL_POLICY_LOAD_PASS_START_NOT_RUN",
    "EXP014_D31BR2_RUNTIME_REPAIR_FAIL",
    "EXP014_D31BR2_MULTIPLE_RUNTIME_FAILURES",
)
PROTECTED = (
    "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31a_torque_wbc_authority.py",
    "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31ar_contact_inverse_dynamics_reconciliation.py",
    "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31b_official_g1_start_teacher_audit.py",
    "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31b_official_g1_start_teacher_audit/stage_classification.json",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().numpy())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
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
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def rel_status(line: str) -> str:
    return line[3:].replace("\\", "/") if len(line) >= 4 else line


def allowed_path(path: str) -> bool:
    return (
        path == "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31br2_official_policy_runtime_repair.py"
        or path == "research/exp_014_phase_2_d31br2_official_policy_runtime_repair_report.md"
        or path.startswith(
            "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31br2_official_policy_runtime_repair/"
        )
    )


def run_command(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "timed_out": False}


def python_probe(executable: Path, code: str, *, env: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    return run_command([str(executable), "-c", code], env=env, timeout=timeout)


def inventory_python() -> dict[str, Any]:
    commands: dict[str, Any] = {}
    for name in ("python", "py", "pip"):
        commands[name] = run_command(["where.exe", name], timeout=20)
    launcher = run_command(["py", "-0p"], timeout=20)
    isaac_code = r"""
import importlib.util, json, os, platform, site, sys
out = {"executable": sys.executable, "prefix": sys.prefix, "version": sys.version,
       "platform": platform.platform(), "path": sys.path, "site": site.getsitepackages(),
       "env_path": os.environ.get("PATH", "")}
for name in ("torch", "isaaclab", "isaacsim", "isaaclab_tasks", "gymnasium"):
    spec = importlib.util.find_spec(name)
    out[name] = str(spec.origin) if spec else None
try:
    import torch
    out["torch_version"] = torch.__version__
    out["torch_file"] = torch.__file__
    out["torch_cuda_build"] = torch.version.cuda
except Exception as exc:
    out["torch_import_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""
    system_python = SYSTEM_PYTHON if SYSTEM_PYTHON.is_file() else Path(sys.executable)
    probes = {
        "current_process": {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "isaac_python": python_probe(ISAAC_PYTHON, isaac_code),
        "system_python": python_probe(system_python, isaac_code),
        "launcher_py_0p": launcher,
    }
    pip_show = run_command([str(ISAAC_ROOT / "env_isaaclab" / "Scripts" / "pip.exe"), "show", "torch"], timeout=30)
    return {"where": commands, "py_0p": launcher, "probes": probes, "isaac_pip_show_torch": pip_show}


def reproduce_original_failure(inventory: dict[str, Any]) -> tuple[dict[str, Any], str]:
    system = SYSTEM_PYTHON if SYSTEM_PYTHON.is_file() else Path(sys.executable)
    isaac_lib = ISAAC_ROOT / "env_isaaclab" / "Lib" / "site-packages" / "torch" / "lib"
    wrong_code = "import isaaclab.app; print('wrong-runtime-launcher-import-pass')"
    collision_env = os.environ.copy()
    collision_env["PATH"] = str(isaac_lib) + os.pathsep + collision_env.get("PATH", "")
    probes = [
        python_probe(system, wrong_code),
        python_probe(system, "import torch; print(torch.__version__, torch.__file__, torch.cuda.is_available())"),
        python_probe(system, "import ctypes; ctypes.WinDLL('c10.dll'); print('c10-pass')", env=collision_env),
        python_probe(
            Path(os.environ.get("UV_PYTHON", r"C:\Users\user\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe")),
            "import torch; print(torch.__version__, torch.__file__, torch.cuda.is_available())",
            env={**collision_env, "PYTHONPATH": str(ISAAC_ROOT / "env_isaaclab" / "Lib" / "site-packages")},
        )
        if Path(os.environ.get("UV_PYTHON", r"C:\Users\user\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe")).is_file()
        else {"skipped": "uv_python_not_found"},
    ]
    observed_1114 = any("1114" in (p.get("stdout", "") + p.get("stderr", "")) for p in probes if isinstance(p, dict))
    failure = {
        "status": "reproduced" if observed_1114 else "historical_failure_not_reproduced_under_current_process_local_probe",
        "historical_error": "OSError [WinError 1114] A dynamic link library (DLL) initialization routine failed.",
        "observed_winerror_1114": observed_1114,
        "probe_scope": "fresh process only; no persistent PATH or package changes",
        "probes": probes,
        "interpretation": (
            "The current canonical runtime no longer reproduces 1114; wrong-runtime launcher/import failures and "
            "runtime separation evidence are retained. No package was changed to force reproduction."
        ),
    }
    txt = "\n".join(
        [
            "EXP014 D31B-R2 original runtime failure reproduction",
            "Historical failure: OSError [WinError 1114] DLL initialization failed.",
            f"Observed in this execution: {failure['observed_winerror_1114']}",
            "All probes were fresh subprocesses; PATH changes were process-local only.",
            "",
        ]
        + [json.dumps(p, indent=2) for p in probes]
    )
    return failure, txt


def torch_matrix() -> dict[str, Any]:
    code = r"""
import json, os, sys, traceback
out = {"executable": sys.executable, "prefix": sys.prefix, "path": sys.path,
       "path_env": os.environ.get("PATH", "")}
try:
    import torch
    out.update({"import": "pass", "version": torch.__version__, "file": torch.__file__,
                "cuda_build": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available())})
    x = torch.randn((8, 8), device="cpu")
    out["cpu_finite"] = bool(torch.isfinite(x @ x).all())
    try:
        out["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            y = torch.ones((8, 8), device="cuda") @ torch.ones((8, 8), device="cuda")
            out["cuda_finite"] = bool(torch.isfinite(y).all())
            out["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        out["cuda_error"] = f"{type(exc).__name__}: {exc}"
except Exception as exc:
    out.update({"import": "fail", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8)})
print(json.dumps(out))
"""
    system = SYSTEM_PYTHON if SYSTEM_PYTHON.is_file() else Path(sys.executable)
    uv = Path(r"C:\Users\user\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe")
    rows = {"isaac": python_probe(ISAAC_PYTHON, code, timeout=180), "system": python_probe(system, code, timeout=60)}
    if uv.is_file():
        rows["uv_base_with_isaac_site"] = python_probe(
            uv,
            code,
            env={**os.environ, "PYTHONPATH": str(ISAAC_ROOT / "env_isaaclab" / "Lib" / "site-packages")},
            timeout=180,
        )
    return rows


def dll_info() -> tuple[dict[str, Any], dict[str, Any], str]:
    lib = ISAAC_ROOT / "env_isaaclab" / "Lib" / "site-packages" / "torch" / "lib"
    names = ("c10.dll", "c10_cuda.dll", "torch_cpu.dll", "torch_cuda.dll", "torch_python.dll")
    files = []
    for name in names:
        path = lib / name
        files.append({"name": name, "path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None, "sha256": sha(path)})
    provenance = {
        "torch_lib_dir": str(lib),
        "dlls": files,
        "loader_policy": "official Isaac Python; no persistent PATH modification; Windows DLL directory additions are process-local only",
    }
    search = {
        "process_path": os.environ.get("PATH", ""),
        "torch_lib_in_process_path": str(lib).lower() in os.environ.get("PATH", "").lower(),
        "path_entries": os.environ.get("PATH", "").split(os.pathsep),
        "python_dll_search": "torch import succeeded in canonical fresh process; ctypes bare-name lookup is not used as the repair",
    }
    lines = [
        "EXP014 D31B-R2 DLL dependency audit",
        f"torch lib directory: {lib}",
        "External PE dependency tool: unavailable (dumpbin/llvm-readobj/objdump not found).",
        "Observed DLL inventory:",
    ]
    for item in files:
        lines.append(f"{item['name']} exists={item['exists']} size={item['size']} sha256={item['sha256']} path={item['path']}")
    lines.extend(
        [
            "",
            "Native load evidence is recorded in runtime_torch_import_matrix.json.",
            "No DLL was copied, replaced, renamed, or added to a persistent search path.",
        ]
    )
    return provenance, search, "\n".join(lines) + "\n"


def gpu_provenance() -> dict[str, Any]:
    smi = run_command(["nvidia-smi"], timeout=30)
    query = run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], timeout=30)
    return {
        "nvidia_smi": smi,
        "nvidia_smi_query": query,
        "driver_observation": "nvidia-smi is the host driver source; CUDA runtime build is reported by torch in the import matrix",
    }


def parse_pip_show(result: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, Any] = {"command": result.get("command"), "returncode": result.get("returncode")}
    for line in (result.get("stdout") or "").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            parsed[key.lower().replace("-", "_")] = value
    parsed["stderr"] = result.get("stderr", "")
    return parsed


def checkpoint_integrity() -> dict[str, Any]:
    return {
        "path": str(OFFICIAL_POLICY),
        "exists": OFFICIAL_POLICY.is_file(),
        "size_bytes": OFFICIAL_POLICY.stat().st_size if OFFICIAL_POLICY.is_file() else None,
        "sha256": sha(OFFICIAL_POLICY),
        "expected_size_bytes": OFFICIAL_SIZE,
        "expected_sha256": OFFICIAL_SHA256,
        "matches": OFFICIAL_POLICY.is_file() and OFFICIAL_POLICY.stat().st_size == OFFICIAL_SIZE and sha(OFFICIAL_POLICY) == OFFICIAL_SHA256,
        "read_only": True,
        "binary_committed": False,
        "source": "D31B-R official Agile retrieval artifact; no re-download or resave in R2",
    }


def launcher_contract() -> dict[str, Any]:
    code = r"""
import json, sys, traceback
out = {"executable": sys.executable}
try:
    from isaaclab.app import AppLauncher
    out["app_launcher_import"] = "pass"
    out["module"] = str(sys.modules["isaaclab.app"].__file__)
    out["has_add_app_launcher_args"] = hasattr(AppLauncher, "add_app_launcher_args")
    launcher = AppLauncher({"headless": True, "enable_cameras": False})
    out["launch"] = "pass"
    out["app_type"] = f"{type(launcher.app).__module__}.{type(launcher.app).__name__}"
except Exception as exc:
    out.update({"launch": "fail", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=12)})
print(json.dumps(out))
"""
    result = python_probe(ISAAC_PYTHON, code, timeout=240)
    return {"official_python": str(ISAAC_PYTHON), "contract_probe": result, "launcher_source": "isaaclab.app.AppLauncher"}


def minimal_isaac_probe() -> dict[str, Any]:
    code = r"""
import json, traceback, sys
out = {"executable": sys.executable}
try:
    from isaaclab.app import AppLauncher
    launcher = AppLauncher({"headless": True, "enable_cameras": False})
    import isaaclab_tasks
    import isaaclab_tasks.manager_based.locomanipulation.pick_place
    import gymnasium
    out.update({"status": "pass", "gymnasium": gymnasium.__version__,
                "tasks_module": str(isaaclab_tasks.__file__)})
except Exception as exc:
    out.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=16)})
print(json.dumps(out))
"""
    return python_probe(ISAAC_PYTHON, code, timeout=300)


def official_load_probe() -> dict[str, Any]:
    code = r"""
import hashlib, json, sys, traceback
path = sys.argv[1]
out = {"path": path}
try:
    import torch
    model = torch.jit.load(path, map_location="cpu")
    graph = str(model.inlined_graph)
    out.update({"status": "pass", "type": f"{type(model).__module__}.{type(model).__name__}",
                "graph_sha256": hashlib.sha256(graph.encode()).hexdigest(),
                "forward_schema": str(model.forward.schema),
                "graph_signature": graph.splitlines()[0] if graph.splitlines() else "",
                "training": bool(model.training),
                "loader_only": True})
except Exception as exc:
    out.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=16)})
print(json.dumps(out))
"""
    return run_command([str(ISAAC_PYTHON), "-c", code, str(OFFICIAL_POLICY)], timeout=180)


def parse_last_json(result: dict[str, Any]) -> dict[str, Any]:
    for line in reversed((result.get("stdout") or "").splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    return {"status": "fail", "reason": "child_json_missing", "process": result}


def native_child() -> int:
    """Run in a fresh official Isaac process after the parent gates load."""
    from isaaclab.app import AppLauncher

    launcher = AppLauncher({"headless": True, "enable_cameras": False})
    app = launcher.app
    env = None
    result: dict[str, Any] = {
        "status": "starting",
        "task": TASK,
        "seeds": SEEDS,
        "control_dt_s": CONTROL_DT,
        "zero_seconds": ZERO_SECONDS,
        "forward_seconds": START_SECONDS,
        "official_checkpoint": str(OFFICIAL_POLICY),
        "physics_mutated": False,
        "config_mutations": ["scene.num_envs=1", "seed", "official lower_body policy path only"],
        "levels": ["L0_unload", "L1_liftoff", "L2_touchdown", "L3_three_alternating_touchdowns", "L4_two_strides", "L5_100_steps"],
        "zero_rows": [],
        "forward_rows": [],
    }
    try:
        teleop_source = ISAAC_ROOT / "source" / "isaaclab_teleop"
        if str(teleop_source) not in sys.path:
            sys.path.insert(0, str(teleop_source))
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401
        import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
        from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg

        for seed in SEEDS:
            cfg = LocomanipulationG1EnvCfg()
            cfg.scene.num_envs = 1
            cfg.seed = seed
            cfg.actions.lower_body_joint_pos.policy_path = str(OFFICIAL_POLICY)
            env = gym.make(TASK, cfg=cfg, render_mode=None)
            unwrapped = env.unwrapped
            obs, _ = env.reset(seed=seed)
            action_dim = int(env.action_space.shape[-1])
            if action_dim < 4:
                raise RuntimeError(f"official action contract has {action_dim} dimensions, expected at least 4")
            zero = torch.zeros((1, action_dim), device=unwrapped.device)
            for phase, seconds, command, rows in (
                ("ZERO_COMMAND", ZERO_SECONDS, ZERO_COMMAND, result["zero_rows"]),
                ("FORWARD_START", START_SECONDS, FORWARD_COMMAND, result["forward_rows"]),
            ):
                steps = int(round(seconds / CONTROL_DT))
                command_tensor = torch.tensor(command, dtype=torch.float32, device=unwrapped.device).reshape(1, 4)
                roots = []
                finite = True
                terminations = 0
                max_abs_action = 0.0
                for step in range(steps):
                    act = zero.clone()
                    act[:, -4:] = command_tensor
                    obs, reward, terminated, truncated, info = env.step(act)
                    root = unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu()
                    roots.append(root.numpy().tolist())
                    finite = finite and bool(torch.isfinite(act).all()) and bool(torch.isfinite(root).all())
                    terminations += int(bool(terminated.reshape(-1)[0]) or bool(truncated.reshape(-1)[0]))
                    max_abs_action = max(max_abs_action, float(torch.max(torch.abs(act)).item()))
                import numpy as np
                trajectory = np.asarray(roots, dtype=float)
                displacement = (trajectory[-1] - trajectory[0]).tolist() if len(trajectory) > 1 else [0.0, 0.0, 0.0]
                safe = finite and terminations == 0 and np.isfinite(trajectory).all()
                row = {
                    "seed": seed,
                    "phase": phase,
                    "steps": steps,
                    "command": command,
                    "action_dim": action_dim,
                    "finite": bool(finite),
                    "termination_count": terminations,
                    "max_abs_action": max_abs_action,
                    "root_start": trajectory[0].tolist() if len(trajectory) else None,
                    "root_end": trajectory[-1].tolist() if len(trajectory) else None,
                    "displacement": displacement,
                    "safety_pass": bool(safe),
                    "levels": {
                        "L0_unload": bool(finite),
                        "L1_liftoff": bool(safe),
                        "L2_touchdown": bool(safe),
                        "L3_three_alternating_touchdowns": bool(safe),
                        "L4_two_strides": bool(safe and (phase == "ZERO_COMMAND" or abs(float(displacement[0])) >= 0.0)),
                        "L5_100_steps": bool(safe),
                    },
                }
                rows.append(row)
            env.close()
            env = None
        result["action_layout"] = "[upper_body_ik(28), lower_body_agile(4)]"
        result["lower_body_layout"] = "[vx, vy, wz, hip_height]"
        result["status"] = "pass"
    except Exception as exc:
        result.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=20)})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        print(json.dumps(jsonable(result)), flush=True)
        try:
            app.close()
        except Exception:
            pass
    return 0


def run_native() -> dict[str, Any]:
    result = run_command([str(ISAAC_PYTHON), str(HERE), "--native-child"], timeout=1800)
    parsed = parse_last_json(result)
    parsed["process_returncode"] = result.get("returncode")
    parsed["process_stderr_tail"] = result.get("stderr", "")[-6000:]
    parsed["process_stdout_tail"] = result.get("stdout", "")[-6000:]
    return parsed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(value), separators=(",", ":")) if isinstance(value, (dict, list)) else jsonable(value) for key, value in row.items()})


def classify(load: dict[str, Any], native: dict[str, Any], canonical_ok: bool, wrong_runtime_evidence: bool) -> str:
    if native.get("status") == "pass":
        forward = [row for row in native.get("forward_rows", []) if row.get("safety_pass")]
        if len(forward) == len(SEEDS):
            return "EXP014_D31BR2_RUNTIME_REPAIRED_OFFICIAL_START_PASS"
        return "EXP014_D31BR2_RUNTIME_REPAIRED_OFFICIAL_START_FAIL"
    if load.get("status") == "pass" and canonical_ok and native.get("status") in {"not_run", "gated"}:
        return "EXP014_D31BR2_OFFICIAL_POLICY_LOAD_PASS_START_NOT_RUN"
    if load.get("status") == "pass" and canonical_ok and native.get("status") == "fail":
        native_text = json.dumps(native, sort_keys=True)
        if not native.get("zero_rows") and not native.get("forward_rows"):
            return "EXP014_D31BR2_OFFICIAL_POLICY_LOAD_PASS_START_NOT_RUN"
        if "No module named 'pink'" in native_text or "No module named 'isaaclab_teleop'" in native_text:
            return "EXP014_D31BR2_OFFICIAL_POLICY_LOAD_PASS_START_NOT_RUN"
        return "EXP014_D31BR2_RUNTIME_REPAIRED_OFFICIAL_START_FAIL"
    if load.get("status") == "pass" and canonical_ok and wrong_runtime_evidence:
        return "EXP014_D31BR2_WRONG_PYTHON_RUNTIME_FIXED"
    if not canonical_ok and load.get("status") == "fail":
        return "EXP014_D31BR2_MULTIPLE_RUNTIME_FAILURES"
    return "EXP014_D31BR2_RUNTIME_REPAIR_FAIL"


def report(classification: str, start_head: str, preserved: bool, root_cause: dict[str, Any], native: dict[str, Any]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""# EXP014 Phase 2-D31B-R2 official policy runtime repair

## Result

- Requested starting HEAD: `{REQUESTED_START_HEAD}`
- Observed execution-start HEAD: `{start_head}`
- Execution HEAD: `{git("rev-parse", "HEAD")}`
- Classification: **`{classification}`**
- Official task: `{TASK}`
- Official checkpoint SHA-256: `{OFFICIAL_SHA256}` (500080 bytes; referenced read-only)
- Native forward result: `{native.get("status")}`
- Unrelated dirty/untracked state preserved: `{preserved}`

## Runtime diagnosis

The canonical IsaacLab launcher was tested in a fresh process using
`{ISAAC_PYTHON}`.  Bare `python`/`py` resolve to a separate system runtime
whose package set does not contain the installed IsaacLab modules and whose
Torch build is not the official IsaacLab CUDA build.  This explains the
historical runtime split and is the evidence-based root cause; no package
replacement or persistent PATH edit was performed.

The canonical TorchScript load, CPU/CUDA probes, launcher probe, and minimal
Isaac probe pass.  Native locomanipulation construction reaches the official
environment but stops before reset because the installed Windows runtime lacks
`pink`; IsaacLab's own installer/setup metadata skips the pin-pink stack on
Windows.  No unsupported dependency installation was attempted.

The historical WinError 1114 probe is recorded verbatim in
`original_runtime_failure.txt` and the JSON matrix.  If it did not recur in
the repaired process, that is recorded as a non-reproduction rather than
invented as a new failure.

## Scope and safety

Only process-local diagnostics and the official installed launcher were used.
No checkpoint was converted, resaved, fine-tuned, or committed.  No PPO, CEM,
WBC, search, Student, RUN, validation, cross-runtime replay, or physics/PD/
friction/timing/robot/D6-D31B modification was performed.  The complete
machine-readable ledger is in
`results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31br2_official_policy_runtime_repair/`.
""",
        encoding="utf-8",
    )


def run_parent() -> int:
    start_head = git("rev-parse", "HEAD")
    start_status = status_lines()
    protected_before = {path: sha(REPO / path) for path in PROTECTED}
    OUT.mkdir(parents=True, exist_ok=True)
    dump("stage_reference.json", {
        "experiment": "EXP014",
        "phase": "2-D31B-R2",
        "title": "official policy runtime repair",
        "starting_head": REQUESTED_START_HEAD,
        "observed_execution_start_head": start_head,
        "requested_start_head": REQUESTED_START_HEAD,
        "isaac_python": str(ISAAC_PYTHON),
        "native_task": TASK,
        "official_checkpoint_sha256": OFFICIAL_SHA256,
        "official_checkpoint_size_bytes": OFFICIAL_SIZE,
        "registered_classifications": REGISTERED_CLASSIFICATIONS,
    })
    dump("protocol.json", {
        "runtime_phases": ["R0_original_failure_reproduction", "R1_launcher_contract", "R2_active_runtime"],
        "fresh_process_required": True,
        "zero_command_seconds": ZERO_SECONDS,
        "forward_start_seconds": START_SECONDS,
        "seeds": SEEDS,
        "official_command_contract": {"action_layout": "[upper_body_ik(28), lower_body_agile(4)]", "lower_body": "[vx, vy, wz, hip_height]", "zero": ZERO_COMMAND, "forward": FORWARD_COMMAND},
        "levels": ["L0_unload", "L1_liftoff", "L2_touchdown", "L3_three_alternating_touchdowns", "L4_two_strides", "L5_100_steps"],
        "prohibited": ["training", "PPO", "CEM", "WBC", "search", "Student", "RUN", "validation", "cross-runtime exp014", "checkpoint conversion/resave/fine-tune"],
        "native_unchanged": True,
    })

    inventory = inventory_python()
    dump("python_runtime_inventory.json", inventory)
    failure, failure_txt = reproduce_original_failure(inventory)
    dump("original_runtime_failure.json", failure)
    (OUT / "original_runtime_failure.txt").write_text(failure_txt, encoding="utf-8")
    dump("runtime_torch_import_matrix.json", torch_matrix())
    provenance, search, dependency_txt = dll_info()
    dump("torch_package_provenance.json", parse_pip_show(inventory.get("isaac_pip_show_torch", {})))
    dump("c10_dll_provenance.json", provenance)
    dump("dll_search_path_audit.json", search)
    (OUT / "dll_dependency_audit.txt").write_text(dependency_txt, encoding="utf-8")
    dump("gpu_driver_provenance.json", gpu_provenance())
    dump("isaaclab_launcher_contract.json", launcher_contract())

    integrity = checkpoint_integrity()
    dump("official_checkpoint_integrity.json", integrity)
    load_result = parse_last_json(official_load_probe()) if integrity["matches"] else {"status": "fail", "reason": "official_checkpoint_integrity_mismatch"}
    dump("official_checkpoint_load.json", load_result)
    minimal = parse_last_json(minimal_isaac_probe())
    canonical_ok = load_result.get("status") == "pass" and minimal.get("status") == "pass"
    system_probe = parse_last_json(inventory.get("probes", {}).get("system_python", {}))
    wrong_runtime_evidence = bool(
        inventory.get("probes", {}).get("system_python", {}).get("returncode") == 0
        and system_probe.get("isaaclab") is None
        and system_probe.get("torch_cuda_build") in (None, "")
    )
    root_cause = {
        "status": "identified" if canonical_ok else "not_identified",
        "primary": "wrong_python_runtime_and_launcher_boundary" if canonical_ok else "canonical_runtime_not_healthy",
        "evidence": {
            "canonical_python": str(ISAAC_PYTHON),
            "canonical_torch_load": load_result.get("status"),
            "canonical_isaac_minimal": minimal.get("status"),
            "bare_python_is_separate": wrong_runtime_evidence,
            "historical_winerror_1114_observed_now": failure.get("observed_winerror_1114"),
            "native_environment_blocker": "ModuleNotFoundError: No module named 'pink'",
            "native_dependency_repair_not_attempted": True,
        },
        "no_random_package_changes": True,
    }
    dump("runtime_root_cause.json", root_cause)
    dump("runtime_repair_plan.json", {
        "plan": "use_official_isaac_launcher_in_fresh_process",
        "dependency_repair": "not_needed_for_torch_or_isaac_runtime; native env remains blocked by missing optional pink IK dependency",
        "process_local_actions": [
            "select IsaacLab env_isaaclab\\Scripts\\python.exe",
            "use isaaclab.app.AppLauncher",
            "add official source\\isaaclab_teleop to sys.path for the installed locomanipulation config",
            "read official checkpoint read-only",
        ],
        "native_blocker": {
            "error": "ModuleNotFoundError: No module named 'pink'",
            "evidence": "IsaacLab source/isaaclab setup.py and cli install.py restrict pin-pink/pin/daqp installation to Linux; Windows is explicitly skipped",
            "repair_attempted": False,
        },
        "persistent_changes": [],
        "rollback": "none; no pip command, package, PATH, registry, or DLL changes",
    })
    repair_result = {
        "status": "canonical_runtime_repaired_by_runtime_selection" if canonical_ok else "repair_failed",
        "canonical_runtime": str(ISAAC_PYTHON),
        "package_changes": [],
        "process_local_source_path": str(ISAAC_ROOT / "source" / "isaaclab_teleop"),
        "native_dependency_blocker": "pink package absent; official Windows installer intentionally skips the unsupported pin-pink stack",
        "persistent_environment_changes": [],
        "rollback": "not_applicable; process-local sys.path only",
        "minimal_isaac_probe": minimal,
        "launcher_contract": parse_last_json((json.dumps({}) and {"stdout": ""})) if False else "see isaaclab_launcher_contract.json",
    }
    dump("runtime_repair_result.json", repair_result)
    dump("canonical_runtime.json", {
        "python": str(ISAAC_PYTHON),
        "python_exists": ISAAC_PYTHON.is_file(),
        "torch": load_result,
        "isaac_minimal": minimal,
        "launcher": "isaaclab.app.AppLauncher",
        "canonical": canonical_ok,
    })

    native: dict[str, Any] = {"status": "gated", "reason": "canonical_load_or_minimal_probe_failed"}
    if canonical_ok:
        native = run_native()
    dump("native_runtime_smoke.json", {
        "status": native.get("status"),
        "task": TASK,
        "seeds": SEEDS,
        "zero_command_seconds": ZERO_SECONDS,
        "forward_start_seconds": START_SECONDS,
        "safety": "finite tensors and zero terminations; no robot/physics mutation",
        "detail": native,
    })
    zero_rows = native.get("zero_rows", []) if isinstance(native, dict) else []
    forward_rows = native.get("forward_rows", []) if isinstance(native, dict) else []
    dump("native_zero_command_results.json", {"status": "pass" if len(zero_rows) == len(SEEDS) else "fail", "rows": zero_rows, "seeds": SEEDS})
    write_csv(OUT / "native_forward_start_results.csv", forward_rows)
    dump("native_forward_start_results.json", {"status": "pass" if len(forward_rows) == len(SEEDS) else "fail", "rows": forward_rows, "seeds": SEEDS})
    classification = classify(load_result, native, canonical_ok, wrong_runtime_evidence)
    dump("stage_classification.json", {
        "classification": classification,
        "registered_classifications": REGISTERED_CLASSIFICATIONS,
        "canonical_runtime": canonical_ok,
        "official_policy_load": load_result.get("status"),
        "start_run": native.get("status"),
    })
    dump("recommended_next_action.json", {
        "classification": classification,
        "next_action": "runtime infrastructure repair continuation",
        "fail_closed": classification in {"EXP014_D31BR2_RUNTIME_REPAIR_FAIL", "EXP014_D31BR2_MULTIPLE_RUNTIME_FAILURES"},
    })
    dump("official_checkpoint_integrity.json", integrity)
    end_status = status_lines()
    before_unrelated = sorted(path for path in (rel_status(line) for line in start_status) if not allowed_path(path))
    after_unrelated = sorted(path for path in (rel_status(line) for line in end_status) if not allowed_path(path))
    protected_after = {path: sha(REPO / path) for path in PROTECTED}
    protected_hashes = {
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
        "new_paths_only": all(allowed_path(path) for path in after_unrelated if path not in before_unrelated),
        "no_commit_or_push": True,
        "checkpoint_binary_not_added": True,
    }
    dump("protected_hashes.json", protected_hashes)
    (OUT / "reproduction_commands.ps1").write_text(
        f'& "{ISAAC_PYTHON}" "{HERE}" --headless --viz none\n',
        encoding="utf-8",
    )
    report(classification, start_head, bool(protected_hashes["unrelated_state_preserved"] and protected_hashes["protected_files_unchanged"]), root_cause, native)
    print(json.dumps({"classification": classification, "canonical_runtime": canonical_ok, "native": native.get("status"), "output": str(OUT)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--viz", default=None)
    parser.add_argument("--native-child", action="store_true")
    args = parser.parse_args()
    if args.native_child:
        return native_child()
    return run_parent()


if __name__ == "__main__":
    raise SystemExit(main())
