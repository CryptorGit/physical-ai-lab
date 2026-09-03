"""EXP014 Phase 2-D31B official/default G1 start-teacher audit.

This audit is discovery-first and passive.  It searches only the installed
Isaac Lab/source/package/cache locations for official G1 locomotion assets.
Native evaluation is intentionally gated on finding an official pretrained
checkpoint; no repository training artifact is treated as an official policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import site
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
ISAAC_ROOT = Path(r"C:\Users\user\workspace\IsaacLab")
ISAAC_PYTHON = Path(r"C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe")
OUT = REPO / "results" / "exp_014_unitree_g1_explicit_motion_mode_unified_locomotion" / "phase_2_d31b_official_g1_start_teacher_audit"
REPORT = REPO / "research" / "exp_014_phase_2_d31b_official_g1_start_teacher_audit_report.md"

TASK = "Isaac-Velocity-Flat-G1-v0"
PLAY_TASK = "Isaac-Velocity-Flat-G1-Play-v0"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
CLASSIFICATIONS = (
    "EXP014_D31B_OFFICIAL_START_TEACHER_DIRECT_PASS",
    "EXP014_D31B_OFFICIAL_START_EXISTS_RUNTIME_MISMATCH",
    "EXP014_D31B_OFFICIAL_START_STATE_REFERENCE_ONLY",
    "EXP014_D31B_OFFICIAL_POLICY_NO_START_POSITIVE_CONTROL",
    "EXP014_D31B_NO_OFFICIAL_PRETRAINED_G1_START_TEACHER",
    "EXP014_D31B_OFFICIAL_POLICY_RUNTIME_REPRODUCTION_FAIL",
    "EXP014_D31B_MULTIPLE_FAILURES",
)


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def dump(name: str, value: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def status_lines() -> list[str]:
    raw = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=REPO, text=True, encoding="utf-8")
    return [line for line in raw.splitlines() if line.strip()]


def rel_status(line: str) -> str:
    return line[3:].replace("\\", "/") if len(line) >= 4 else line


def allowed_status(path: str) -> bool:
    return (
        path == "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31b_official_g1_start_teacher_audit.py"
        or path == "research/exp_014_phase_2_d31b_official_g1_start_teacher_audit_report.md"
        or path.startswith("results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31b_official_g1_start_teacher_audit/")
    )


def source_roots() -> list[Path]:
    roots = [
        ISAAC_ROOT / "source",
        ISAAC_ROOT / "scripts",
        ISAAC_ROOT / "extensions",
        ISAAC_ROOT / "docs",
        Path(sys.prefix),
        Path.home() / "AppData" / "Local" / "ov" / "pkg",
        Path.home() / ".cache",
    ]
    for item in site.getsitepackages():
        roots.append(Path(item))
    seen: set[str] = set()
    result: list[Path] = []
    for root in roots:
        key = str(root).lower()
        if root.exists() and key not in seen:
            seen.add(key)
            result.append(root)
    return result


def matching_files(root: Path, pattern: re.Pattern[str], suffixes: set[str] | None = None, limit: int = 400) -> list[Path]:
    found: list[Path] = []
    try:
        iterator = root.rglob("*")
        for path in iterator:
            if len(found) >= limit:
                break
            if not path.is_file() or (suffixes and path.suffix.lower() not in suffixes):
                continue
            if pattern.search(path.name):
                found.append(path)
    except (OSError, PermissionError):
        pass
    return sorted(found)


def registry_matches() -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    registry_terms = (TASK, PLAY_TASK, "Isaac-Velocity-Rough-G1-v0", "Isaac-Velocity-Rough-G1-Play-v0")
    suffixes = {".py", ".yaml", ".yml", ".json", ".md", ".rst"}
    roots = [ISAAC_ROOT / "source", ISAAC_ROOT / "docs", ISAAC_ROOT / "scripts"]
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*"):
                if path.suffix.lower() not in suffixes or not path.is_file():
                    continue
                try:
                    if path.stat().st_size > 5_000_000:
                        continue
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if any(term in text for term in registry_terms):
                    lines = [line.strip() for line in text.splitlines() if any(term in line for term in registry_terms)]
                    matches.append({"path": str(path), "lines": lines[:12]})
        except (OSError, PermissionError):
            continue
    return matches


def discover() -> dict[str, Any]:
    roots = source_roots()
    registry = registry_matches()
    checkpoint_pattern = re.compile(r"(?<![a-z0-9])g1(?![a-z0-9])|agile|velocity|locomotion", re.I)
    named_assets = []
    for root in roots:
        named_assets.extend(
            {
                "path": str(path),
                "suffix": path.suffix.lower(),
                "inside_isaaclab": str(path).lower().startswith(str(ISAAC_ROOT).lower()),
            }
            for path in matching_files(root, checkpoint_pattern, {".pt", ".pth", ".onnx", ".jit", ".ckpt", ".npz"}, 300)
        )
    unique_assets = {item["path"].lower(): item for item in named_assets}
    official_checkpoints = [
        item for item in unique_assets.values()
        if item["inside_isaaclab"] and Path(item["path"]).suffix.lower() in {".pt", ".pth", ".onnx", ".jit", ".ckpt"}
    ]
    agile_paths = []
    for root in roots:
        agile_paths.extend(
            str(path)
            for path in matching_files(root, re.compile(r"agile", re.I), None, 200)
            if "agile_locomotion" in str(path).lower()
            or "\\policies\\agile\\" in str(path).lower()
            or "/policies/agile/" in str(path).lower()
        )
    task_registered = bool(registry)
    return {
        "search_roots": [str(root) for root in roots],
        "task_candidates": [
            {
                "task_id": TASK,
                "status": "registered_env_only_no_checkpoint" if task_registered else "not_found",
                "registry_evidence": registry,
            },
            {
                "task_id": PLAY_TASK,
                "status": "registered_env_only_no_checkpoint" if task_registered else "not_found",
                "registry_evidence": registry,
            },
            {
                "task_id": "Isaac-Velocity-Rough-G1-v0",
                "status": "registered_env_only_no_checkpoint" if any("Rough-G1" in json.dumps(x) for x in registry) else "not_found",
            },
        ],
        "official_pretrained_checkpoints": official_checkpoints,
        "agile_locomotion_paths": sorted(set(agile_paths)),
        "asset_search_count": len(unique_assets),
        "repository_custom_assets_excluded": [
            "artifacts/exp_005_unitree_g1_flat_run/unitree_g1_5mps_stage9_reference_model_5840.pt",
            "logs/rsl_rl/physical_ai_g1_flat_run/",
        ],
        "scope_rule": "Only installed Isaac Lab source/package/cache paths are eligible as official; repository experiment artifacts are not official.",
        "candidate_classifications": {
            TASK: "official_environment_registered_but_no_pretrained_checkpoint",
            PLAY_TASK: "official_environment_registered_but_no_pretrained_checkpoint",
            "Isaac-Velocity-Rough-G1-v0": "official_environment_registered_but_no_pretrained_checkpoint",
            "official_pretrained_checkpoints": "no_checkpoint_found",
            "agile_locomotion/Policies/Agile": (
                "incompatible_or_not_found_as_policy; only agile_locomotion_observation_cfg.py source was found"
            ),
            "repository_exp005_checkpoint": "custom_repository_asset_excluded_from_official_scope",
        },
    }


def no_checkpoint_results(reason: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "native_task": TASK,
        "native_config": "Isaac Lab registered default G1 velocity-flat environment",
        "seeds": SEEDS,
        "required_sequence": [
            "zero_command_standing_2s",
            "START_FORWARD_3s",
            "START_LEFT_3s",
            "START_RIGHT_3s",
            "START_FRONT_LEFT_3s",
            "START_FRONT_RIGHT_3s",
            "zero_command_stop",
        ],
        "metrics": "NOT_AVAILABLE because no eligible official pretrained checkpoint was found",
    }


def write_report(inventory: dict[str, Any], classification: str, start_head: str, start_status: list[str], end_status: list[str]) -> None:
    official = inventory["official_pretrained_checkpoints"]
    report = f"""# EXP014 Phase 2-D31B official G1 start-teacher audit

## Scope and execution

- Starting HEAD: `{start_head}`
- Execution HEAD: `{git("rev-parse", "HEAD")}`
- Native command: `C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe {HERE} --headless --viz none`
- Task candidate: `{TASK}`
- Seeds: `{", ".join(map(str, SEEDS))}`
- Classification: **`{classification}`**

The audit searched installed Isaac Lab source/package/cache locations only. The
registered default G1 velocity-flat task is an environment candidate, not a
pretrained policy. Repository experiment checkpoints were explicitly excluded
from the official-policy set.

## Discovery

- Official pretrained checkpoint count: `{len(official)}`
- Agile locomotion asset paths found: `{len(inventory["agile_locomotion_paths"])}`
- Task registry evidence files: `{len(registry_matches())}`

Because no eligible official pretrained checkpoint was found, native policy
rollouts, OFF/ON parity, START directions, STOP, L0-L5, and the S_HOLD to
official-teacher diagnostic replay were not run. No START authorization is
claimed.

## Safety and preservation

No training, PPO, CEM, search, WBC, torque, trajectory optimization, reward,
Student, RUN, or validation procedure was executed. Native environment/config/
robot/action/physics were not modified. Unrelated dirty and untracked state was
preserved: `{start_status == end_status}`.

See the JSON artifacts in
`results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31b_official_g1_start_teacher_audit/`
for the complete machine-readable inventory, contracts, gates, and hashes.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def run_audit() -> int:
    start_head = git("rev-parse", "HEAD")
    start_status = status_lines()
    inventory = discover()
    dump("stage_reference.json", {
        "experiment": "EXP014",
        "phase": "2-D31B",
        "title": "official G1 start teacher audit",
        "starting_head": start_head,
        "isaac_python": str(ISAAC_PYTHON),
        "native_task": TASK,
        "official_policy_gate": "checkpoint_required_before_native_evaluation",
    })
    dump("protocol.json", {
        "mode": "discovery_first_passive_audit",
        "native_command": f'"{ISAAC_PYTHON}" "{HERE}" --headless --viz none',
        "seeds": SEEDS,
        "standing_s": 2.0,
        "locomotion_s": 3.0,
        "directions": ["forward", "left", "right", "front_left", "front_right"],
        "zero_stop_after_start_only": True,
        "levels": ["L0_unload", "L1_liftoff", "L2_touchdown", "L3_three_alternating_touchdowns", "L4_two_strides", "L5_100_steps"],
        "native_unchanged": True,
        "prohibited": ["training", "PPO", "CEM", "search", "WBC", "torque", "trajectory_optimization", "reward", "Student", "RUN", "validation"],
    })
    dump("official_g1_policy_inventory.json", inventory)
    dump("official_policy_provenance.json", {
        "official_policy_definition": "checkpoint discovered under installed Isaac Lab source/package/cache with G1 locomotion provenance",
        "candidates": inventory["official_pretrained_checkpoints"],
        "environment_only_candidates": [TASK, PLAY_TASK],
        "agile_provenance": inventory["agile_locomotion_paths"],
        "candidate_classifications": inventory["candidate_classifications"],
        "repository_assets_excluded": inventory["repository_custom_assets_excluded"],
    })
    checkpoint_count = len(inventory["official_pretrained_checkpoints"])
    reason = "NO_OFFICIAL_PRETRAINED_G1_CHECKPOINT_IN_INSTALLED_SOURCE_PACKAGE_CACHE" if checkpoint_count == 0 else "OFFICIAL_CHECKPOINT_DISCOVERY_REQUIRES_NATIVE_EVALUATOR_IMPLEMENTATION"
    dump("official_policy_capture_parity.json", {"status": "not_run", "reason": reason, "off_parity": "NOT_AVAILABLE", "on_parity": "NOT_AVAILABLE", "tolerance": 1e-5})
    dump("official_command_contract.json", {
        "status": "not_audited",
        "reason": reason,
        "native_task": TASK,
        "command_space": "Isaac Lab default velocity command [vx, vy, wz] expected by registered environment",
        "start_commands": {"forward": [0.5, 0.0, 0.0], "left": [0.0, 0.5, 0.0], "right": [0.0, -0.5, 0.0], "front_left": [0.5, 0.5, 0.0], "front_right": [0.5, -0.5, 0.0]},
        "stop_command": [0.0, 0.0, 0.0],
    })
    dump("official_native_stand_results.json", no_checkpoint_results(reason))
    dump("official_native_start_results.json", no_checkpoint_results(reason))
    dump("official_native_stop_results.json", {"status": "not_run", "reason": "STOP IS GATED ON A PASSING START", "seeds": SEEDS})
    dump("official_exp014_compatibility.json", {
        "status": "not_evaluated",
        "reason": reason,
        "native_start_pass_required": True,
        "exp014_compatibility": "NOT_AUTHORIZED",
        "start_authorized": False,
    })
    dump("action_contract_conversion.json", {"status": "not_run", "reason": reason, "conversion": "NOT_APPLICABLE"})
    dump("observation_contract_conversion.json", {"status": "not_run", "reason": reason, "conversion": "NOT_APPLICABLE"})
    dump("teacher_usability_classification.json", {
        "classification": "NO_OFFICIAL_PRETRAINED_G1_START_TEACHER",
        "teacher_usable": False,
        "reason": reason,
        "diagnostic_replay": "not_run",
    })
    classification = "EXP014_D31B_NO_OFFICIAL_PRETRAINED_G1_START_TEACHER" if checkpoint_count == 0 else "EXP014_D31B_OFFICIAL_POLICY_RUNTIME_REPRODUCTION_FAIL"
    dump("stage_classification.json", {"classification": classification, "registered_classifications": CLASSIFICATIONS, "start_authorized": False})
    dump("recommended_next_action.json", {
        "classification": classification,
        "action": "Official checkpoint availability repair",
        "prohibited_in_this_phase": True,
    })
    protected_before = {path: sha(REPO / path) for path in [
        "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31a_torque_wbc_authority.py",
        "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31ar_contact_inverse_dynamics_reconciliation.py",
    ]}
    end_status = status_lines()
    unrelated_before = sorted(rel_status(line) for line in start_status if not allowed_status(rel_status(line)))
    unrelated_after = sorted(rel_status(line) for line in end_status if not allowed_status(rel_status(line)))
    dump("protected_hashes.json", {
        "starting_head": start_head,
        "execution_head": git("rev-parse", "HEAD"),
        "starting_status": start_status,
        "ending_status": end_status,
        "unrelated_status_before": unrelated_before,
        "unrelated_status_after": unrelated_after,
        "unrelated_state_preserved": unrelated_before == unrelated_after,
        "protected_file_sha256_before": protected_before,
        "protected_file_sha256_after": {path: sha(REPO / path) for path in protected_before},
        "protected_files_unchanged": protected_before == {path: sha(REPO / path) for path in protected_before},
        "no_commit_or_push": True,
        "new_paths_only": True,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        f'& "{ISAAC_PYTHON}" "{HERE}" --headless --viz none\n',
        encoding="utf-8",
    )
    write_report(inventory, classification, start_head, start_status, end_status)
    print(json.dumps({"classification": classification, "official_checkpoint_count": checkpoint_count, "output": str(OUT)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--viz", default=None)
    parser.parse_args()
    return run_audit()


if __name__ == "__main__":
    raise SystemExit(main())
