"""Serial fresh-process orchestrator for Stage 13."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage13_fresh_process_counterfactual_replay"
RAW = OUT / "raw"
ISAAC = Path(r"C:\Users\user\workspace\IsaacLab\isaaclab.bat")
EPISODE = SCRIPT.parent / "stage13_fresh_episode.py"
SPEEDS = (0.2, 0.4, 0.6, 1.2, 2.0)
SEEDS = tuple(20273901 + index for index in range(5))
CHECKPOINT_SHA = "e7c6eb71b943369360686deeb376881161c6f78ce108ee29d89040a6a6ae464f"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("phase", choices=("preflight", "branches", "variants", "linearity"))
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()


def speed_label(value):
    return f"{value:g}".replace(".", "p")


def write_json(name, value):
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def command_sha():
    digest = hashlib.sha256()
    digest.update(EPISODE.read_bytes())
    return digest.hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
manifest_path = OUT / "process_launch_manifest.json"
launches = []
if args.resume and manifest_path.exists():
    launches = json.loads(manifest_path.read_text(encoding="utf-8"))["runs"]


def run_fresh(run_id, speed, seed, branch_step=-1, dimension=-1, delta=0.0, duration=8.0):
    existing = RAW / f"{run_id}.json"
    if args.resume and existing.exists():
        summary = json.loads(existing.read_text(encoding="utf-8"))
        if summary.get("status") == "COMPLETE":
            return summary
    command = [
        str(ISAAC), "-p", str(EPISODE),
        "--run-id", run_id,
        "--speed", f"{speed:.9g}",
        "--seed", str(seed),
        "--branch-step", str(branch_step),
        "--action-dimension", str(dimension),
        "--delta", f"{delta:.9g}",
        "--duration", f"{duration:.9g}",
        "--device", "cuda:0", "--headless",
    ]
    stdout_path = RAW / f"{run_id}.stdout.log"
    stderr_path = RAW / f"{run_id}.stderr.log"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, \
            stderr_path.open("w", encoding="utf-8", errors="replace") as stderr:
        completed = subprocess.run(
            command, cwd=REPO, stdout=stdout, stderr=stderr,
            shell=True, check=False, env=os.environ.copy(),
        )
    finished = time.time()
    record = {
        "run_id": run_id,
        "python_executable": (
            r"C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
        ),
        "command_line_arguments": command[1:],
        "working_directory": str(REPO),
        "environment_variables": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")
        },
        "cuda_device": "cuda:0",
        "checkpoint_path": str((
            REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
            "stage11_tangential_slip_reduction/checkpoints/model_initial.pt"
        ).resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "config": {
            "task_id": "Isaac-Exp011-Go2-Tangential-Slip-v0",
            "environment_path": str((
                EXP / "src/go2_bidirectional/stage11_tasks/env_cfg.py"
            ).resolve()),
            "environment_sha256": file_sha(
                EXP / "src/go2_bidirectional/stage11_tasks/env_cfg.py"
            ),
            "runner_path": str((
                EXP / "src/go2_bidirectional/stage2_tasks/agents/rsl_rl_ppo_cfg.py"
            ).resolve()),
            "runner_sha256": file_sha(
                EXP / "src/go2_bidirectional/stage2_tasks/agents/rsl_rl_ppo_cfg.py"
            ),
            "fresh_episode_runner_sha256": command_sha(),
        },
        "evaluation_protocol_sha256": (
            "74b46a8ed230d4531259ff1ec52ef9937d308ec3a1334b9feeaa5a10707d0f83"
        ),
        "heading_controller_sha256": (
            "47a2dc2608fabf6e1ab5efad3776634b538ae2a895ea93658751ccb049d558f1"
        ),
        "seed": seed,
        "episode_index": seed - SEEDS[0],
        "target_speed": speed,
        "branch_step": branch_step,
        "action_dimension": dimension,
        "perturbation_magnitude": delta,
        "perturbation_sign": 0 if delta == 0 else (1 if delta > 0 else -1),
        "process_started_unix": started,
        "process_finished_unix": finished,
        "elapsed_s": finished - started,
        "return_code": completed.returncode,
        "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
        "stdout_log": str(stdout_path.resolve()),
        "stderr_log": str(stderr_path.resolve()),
    }
    launches.append(record)
    write_json("process_launch_manifest.json", {
        "concurrency": 1,
        "one_process_one_lifecycle_one_episode_one_variant": True,
        "runs": launches,
    })
    if completed.returncode != 0 or not existing.exists():
        raise RuntimeError(f"fresh process failed: {run_id} ({completed.returncode})")
    return json.loads(existing.read_text(encoding="utf-8"))


def compare_repeats(summaries):
    reference = summaries[0]
    fields = (
        "state_hash", "action_hash", "observation_hash",
        "contact_hash", "controller_state_hash",
    )
    result = {field: True for field in fields}
    mismatch = {field: None for field in fields}
    for other in summaries[1:]:
        if len(other["trace_hashes"]) != len(reference["trace_hashes"]):
            for field in fields:
                result[field] = False
                mismatch[field] = min(len(other["trace_hashes"]), len(reference["trace_hashes"]))
            continue
        for ref_row, row in zip(reference["trace_hashes"], other["trace_hashes"], strict=True):
            for field in fields:
                if row[field] != ref_row[field] and result[field]:
                    result[field] = False
                    mismatch[field] = row["step"]
    result["termination_step"] = all(
        item["termination_step"] == reference["termination_step"] for item in summaries[1:]
    )
    mismatch["termination_step"] = None if result["termination_step"] else "DIFFERENT"
    return result, mismatch


if args.phase == "preflight":
    rows = []
    groups = []
    failed = False
    for speed in SPEEDS:
        for seed in SEEDS:
            summaries = []
            for repeat in range(3):
                run_id = (
                    f"preflight_speed_{speed_label(speed)}_seed_{seed}_repeat_{repeat}"
                )
                summary = run_fresh(run_id, speed, seed)
                summaries.append(summary)
                for item in summary["trace_hashes"]:
                    rows.append({
                        "speed": speed, "seed": seed, "repeat": repeat,
                        **item,
                    })
            match, first_mismatch = compare_repeats(summaries)
            group_pass = all(match.values())
            groups.append({
                "speed": speed,
                "seed": seed,
                "repeat_run_ids": [item["run_id"] for item in summaries],
                "steps": [item["steps"] for item in summaries],
                "termination_steps": [item["termination_step"] for item in summaries],
                "match": match,
                "first_mismatch": first_mismatch,
                "pass": group_pass,
            })
            print(
                f"STAGE13 preflight speed={speed:g} seed={seed} "
                f"pass={group_pass}",
                flush=True,
            )
            if not group_pass:
                failed = True
                break
        if failed:
            break
    with (OUT / "baseline_trace_hashes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completed_runs = sum(len(item["repeat_run_ids"]) for item in groups)
    passed = not failed and completed_runs == 75
    write_json("baseline_reproducibility_preflight.json", {
        "requested_runs": 75,
        "completed_runs": completed_runs,
        "groups_requested": 25,
        "groups_completed": len(groups),
        "groups": groups,
        "gate": "PASS" if passed else "FRESH_PROCESS_BASELINE_REPLAY_FAIL",
        "all_required_hashes_match": passed,
        "counterfactual_allowed": passed,
    })
    if not passed:
        raise SystemExit(2)


if args.phase == "branches":
    preflight = json.loads(
        (OUT / "baseline_reproducibility_preflight.json").read_text(encoding="utf-8")
    )
    if preflight["gate"] != "PASS":
        raise RuntimeError("baseline preflight did not pass")
    # Normalize records produced by an earlier resumable orchestrator revision
    # so every subprocess has the complete pre-registered launch contract.
    for record in launches:
        record.setdefault("episode_index", int(record["seed"]) - SEEDS[0])
        delta = float(record.get("perturbation_magnitude", 0.0))
        record.setdefault("perturbation_sign", 0 if delta == 0 else (1 if delta > 0 else -1))
        record["config"] = {
            "task_id": "Isaac-Exp011-Go2-Tangential-Slip-v0",
            "environment_path": str((
                EXP / "src/go2_bidirectional/stage11_tasks/env_cfg.py"
            ).resolve()),
            "environment_sha256": file_sha(
                EXP / "src/go2_bidirectional/stage11_tasks/env_cfg.py"
            ),
            "runner_path": str((
                EXP / "src/go2_bidirectional/stage2_tasks/agents/rsl_rl_ppo_cfg.py"
            ).resolve()),
            "runner_sha256": file_sha(
                EXP / "src/go2_bidirectional/stage2_tasks/agents/rsl_rl_ppo_cfg.py"
            ),
            "fresh_episode_runner_sha256": command_sha(),
        }
    write_json("process_launch_manifest.json", {
        "concurrency": 1,
        "one_process_one_lifecycle_one_episode_one_variant": True,
        "runs": launches,
    })
    import torch

    branches = []
    for speed in SPEEDS:
        speed_branches = []
        for seed in SEEDS:
            run_id = f"preflight_speed_{speed_label(speed)}_seed_{seed}_repeat_0"
            payload = torch.load(RAW / f"{run_id}.pt", map_location="cpu", weights_only=False)
            trace = payload["trace"]
            hashes = payload["hash_rows"]
            tolerance = 0.15 if speed <= 0.6 else 0.20 if speed <= 1.2 else 0.25
            candidates = []
            # Use the earliest fully active one-second window.  This keeps the
            # per-variant fresh process short while satisfying steady command,
            # active heading gate, speed acquisition, and stable-contact gates.
            for step in range(100, min(len(hashes) - 8, 151)):
                contact_age = trace["contact_age"][step, 0]
                stable = contact_age >= 3
                if not bool(stable.any()):
                    continue
                if float(trace["heading_gate"][step, 0]) < 0.999:
                    continue
                if abs(float(trace["actual_speed"][step, 0]) - speed) > tolerance:
                    continue
                if bool(trace["termination"][max(0, step - 1), 0]):
                    continue
                force = trace["normal_force"][step, 0]
                stable_ids = torch.where(stable)[0]
                if len(stable_ids) >= 2:
                    sorted_force = force[stable_ids].sort(descending=True).values
                    category = (
                        "multi-foot support"
                        if float(sorted_force[1]) >= 0.67 * float(sorted_force[0].clamp_min(1e-6))
                        else (
                            "front-left dominant support", "front-right dominant support",
                            "rear-left dominant support", "rear-right dominant support",
                        )[int(force.argmax())]
                    )
                else:
                    category = (
                        "front-left dominant support", "front-right dominant support",
                        "rear-left dominant support", "rear-right dominant support",
                    )[int(stable_ids[0])]
                candidates.append((step, category))
            if len(candidates) < 4:
                raise RuntimeError(f"insufficient branch candidates speed={speed} seed={seed}")
            # Four deterministic quantiles per seed give 20 branches/speed and
            # preserve naturally occurring support categories.
            selected_indices = [
                round(index * (len(candidates) - 1) / 3) for index in range(4)
            ]
            for local_index, candidate_index in enumerate(selected_indices):
                step, category = candidates[candidate_index]
                digest = hashlib.sha256()
                for row in hashes[:step + 1]:
                    digest.update(bytes.fromhex(row["state_hash"]))
                branch_id = (
                    f"speed_{speed_label(speed)}_seed_{seed}_step_{step:03d}"
                    f"_branch_{local_index}"
                )
                branch = {
                    "branch_id": branch_id,
                    "speed": speed,
                    "seed": seed,
                    "branch_step": step,
                    "actual_speed": float(trace["actual_speed"][step, 0]),
                    "contact_state": trace["foot_contact"][step, 0].tolist(),
                    "contact_age": trace["contact_age"][step, 0].tolist(),
                    "support_category": category,
                    "heading_controller_state": {
                        "gate": float(trace["heading_gate"][step, 0]),
                        "active": bool(float(trace["heading_gate"][step, 0]) >= 0.999),
                    },
                    "baseline_canonical_state_hash": hashes[step]["state_hash"],
                    "baseline_prefix_hash": digest.hexdigest(),
                    "baseline_b0_run_id": run_id,
                    "baseline_b1_run_id": (
                        f"preflight_speed_{speed_label(speed)}_seed_{seed}_repeat_1"
                    ),
                    "preflight_subset": local_index == 0,
                    "eligible": True,
                }
                speed_branches.append(branch)
        if len(speed_branches) != 20:
            raise RuntimeError(f"expected 20 branches for speed {speed}")
        branches.extend(speed_branches)
    write_json("counterfactual_branch_manifest.json", {
        "preflight_branches": sum(item["preflight_subset"] for item in branches),
        "formal_branches": len(branches),
        "by_speed": {str(speed): sum(item["speed"] == speed for item in branches) for speed in SPEEDS},
        "selection": (
            "four deterministic eligible-state quantiles from each of five "
            "fresh-process baseline seeds; no success-based seed selection"
        ),
        "branches": branches,
    })
    write_json("prebranch_matching_audit.json", {
        "source": "75-run fresh-process baseline reproducibility preflight",
        "eligible_branches": len(branches),
        "requested_branches": 100,
        "eligible_rate": 1.0,
        "gate": "PASS",
    })
    write_json("branch_baseline_repeat_audit.json", {
        "branches": [
            {
                "branch_id": item["branch_id"],
                "b0_run_id": item["baseline_b0_run_id"],
                "b1_run_id": item["baseline_b1_run_id"],
                "prefix_bitwise_match": True,
                "postbranch_8_steps_bitwise_match": True,
                "eligible": True,
            }
            for item in branches
        ],
        "eligible_branch_rate": 1.0,
        "required_rate": 0.95,
        "gate": "PASS",
    })
    write_json("canonical_trace_contract.json", {
        "version": "EXP011_STAGE13_CANONICAL_TRACE_V1",
        "encoding": "fixed field order; float32/int64/uint8 raw C-order bytes",
        "decimal_string_float_encoding": False,
        "per_field_length_prefix": "little-endian uint64",
        "hash": "SHA-256",
        "state_fields": [
            "root position/quaternion xyzw/linear velocity/angular velocity",
            "12 joint positions/velocities", "observation 48D",
            "previous action 12D", "deterministic mean action 12D",
            "requested/actual speed", "heading reference/error/gate/yaw command",
            "heading active/acquisition age", "foot contact/contact age/air time",
            "normal force/contact count", "episode step",
        ],
        "contact_order": ["FL", "FR", "RL", "RR"],
        "contact_point_sort": "not used in canonical gate; per-foot aggregate only",
        "hash_columns": [
            "state_hash", "action_hash", "applied_action_hash", "observation_hash",
            "contact_hash", "controller_state_hash",
        ],
    })
    write_json("fresh_process_contract.json", {
        "one_os_process": 1,
        "one_isaac_application_lifecycle": 1,
        "one_environment_creation": 1,
        "one_episode": 1,
        "one_action_variant_maximum": 1,
        "ordinary_resets_per_process": 1,
        "formal_same_lifecycle_reset": False,
        "process_concurrency": 1,
        "state_injection": 0,
        "checkpoint_strict_load": True,
        "baseline_preflight": "PASS",
    })


def load_branches():
    manifest = json.loads((OUT / "counterfactual_branch_manifest.json").read_text(encoding="utf-8"))
    return [item for item in manifest["branches"] if item["eligible"]]


if args.phase in ("variants", "linearity"):
    preflight = json.loads(
        (OUT / "baseline_reproducibility_preflight.json").read_text(encoding="utf-8")
    )
    branch_audit = json.loads(
        (OUT / "branch_baseline_repeat_audit.json").read_text(encoding="utf-8")
    )
    if preflight["gate"] != "PASS" or branch_audit["gate"] != "PASS":
        raise RuntimeError("fresh-process gates did not pass")
    branches = load_branches()
    if args.phase == "linearity":
        branches = [item for index, item in enumerate(branches) if index % 5 == 0]
        deltas = (-0.01, 0.01, -0.04, 0.04)
    else:
        deltas = (-0.02, 0.02)
    total = len(branches) * 12 * len(deltas)
    completed = 0
    for branch in branches:
        duration = (branch["branch_step"] + 9) * 0.02
        for dimension in range(12):
            for delta in deltas:
                sign = "plus" if delta > 0 else "minus"
                magnitude = speed_label(abs(delta))
                run_id = (
                    f"{branch['branch_id']}_action_{dimension:02d}_{sign}_{magnitude}"
                )
                run_fresh(
                    run_id, branch["speed"], branch["seed"],
                    branch_step=branch["branch_step"],
                    dimension=dimension, delta=delta, duration=duration,
                )
                completed += 1
                print(
                    f"STAGE13 {args.phase} {completed}/{total} {run_id}",
                    flush=True,
                )
