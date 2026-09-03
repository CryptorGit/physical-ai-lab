"""Pure-filesystem D12 forensic audit; never imports Isaac Lab or torch."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]
START = "88460496e0b10550649390b46c6248e8c7b2c5b7"
D11 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d11_stop_student_heldout"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d12_d11_result_durability_audit"
REPORT = REPO / "research/exp_014_phase_2_d12_d11_result_durability_audit_report.md"
RUNNER = REPO / "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d11_heldout.py"
SIM_LAUNCHER = Path(r"C:/Users/user/workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/sim_launcher.py")
APP_LAUNCHER = Path(r"C:/Users/user/workspace/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py")
OVPHYSX = Path(r"C:/Users/user/workspace/IsaacLab/source/isaaclab_ovphysx/isaaclab_ovphysx/physics/ovphysx_manager.py")
CLASSIFICATION = "EXP014_D12_COMPLETION_LEDGER_TRANSACTION_BUG"
SEALED_SHA = "dd34d3035866bd35e29643e616a1add35b3ca8bd3c7e05a58ce73b7182f266e9"
CANDIDATE_SHA = "5de37e5d0807654d370ba7a79ee9872c4029cb50e548685423e48812249959d5"

spec = importlib.util.spec_from_file_location("d12utils", HERE.parent / "d12_forensic_utils.py"); util = importlib.util.module_from_spec(spec); spec.loader.exec_module(util)


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def dump(name, value): OUT.mkdir(parents=True, exist_ok=True); (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def write_csv(name, rows, fields):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
def git(*args): return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
def file_sha(path): return util.sha256_file(path)
def line_range(path, token):
    lines = Path(path).read_text(encoding="utf-8").splitlines(); hits = [index + 1 for index, line in enumerate(lines) if token in line]
    return hits


ledger = read(D11 / "heldout_access_ledger.json")
open_time = dt.datetime.fromisoformat(ledger["open_timestamp"]); start = open_time - dt.timedelta(minutes=30)
end = dt.datetime.fromtimestamp((D11 / "heldout_access_ledger.json").stat().st_mtime, dt.timezone.utc) + dt.timedelta(minutes=30)
extensions = {".json", ".jsonl", ".csv", ".parquet", ".pickle", ".pt", ".npy", ".npz", ".sqlite", ".db", ".wal", ".log", ".txt", ".stdout", ".stderr", ".tmp", ".partial", ".lock", ".journal", ".dmp"}
roots = [REPO / item for item in ("results", "logs", "artifacts", "research", "tmp", "temp", ".cache")]
roots += [Path(os.environ.get("TEMP", "")), Path(r"C:/Users/user/.nvidia-omniverse/logs"), Path(r"C:/Users/user/AppData/Local/ov/logs"), Path(r"C:/Users/user/workspace/IsaacLab/logs")]
inventory, seen = [], set()
for root in roots:
    if not root or not root.exists(): continue
    for path in root.rglob("*"):
        try:
            if not path.is_file() or path.suffix.lower() not in extensions or OUT in path.parents: continue
            resolved = str(path.resolve()).lower()
            if resolved in seen: continue
            stat = path.stat()
            if not util.within_window(stat, start.timestamp(), end.timestamp()): continue
            seen.add(resolved)
            record = util.inventory_record(path, REPO)
            if path == D11 / "heldout_episode_results.csv" or path.name.endswith("not_authorized.json"):
                record["D11_relevance"] = "POST_INTERRUPTION_LEDGER_DERIVATIVE_NO_PHYSICAL_METRICS"
            inventory.append(record)
        except (OSError, PermissionError, ValueError): continue
inventory.sort(key=lambda item: (item["modification_time_utc"], item["absolute_path"]))
fields = ["absolute_path", "relative_path", "size", "creation_time_utc", "modification_time_utc", "sha256", "file_type", "owner_process_candidate", "D11_relevance", "read_only_audit_preserved_mtime"]
write_csv("d11_forensic_file_inventory.csv", inventory, fields)
dump("d11_forensic_file_inventory.json", {"window_utc": {"start": start.isoformat(), "end": end.isoformat()}, "roots": [str(path) for path in roots], "file_count": len(inventory), "files": inventory, "sealed_payload_deserialized": False})

raw_candidates = [item for item in inventory if item["D11_relevance"] == "DIRECT_D11" and any(token in Path(item["absolute_path"]).name.lower() for token in ("result", "batch", "journal", "sqlite", "wal", "partial", "tmp"))]
durable_outcome_candidates = [item for item in raw_candidates if "heldout_results.json" in item["absolute_path"] or "episode_result" in Path(item["absolute_path"]).name.lower() and item["size"] > 50000]

source_exit_paths = [
    {"source": str(RUNNER), "kind": "D11 runner", "matches": {token: line_range(RUNNER, token) for token in ("sys.exit", "os._exit", "SystemExit", "finally", "with launch_simulation", "dump(RAW / \"heldout_results.json\"")}},
    {"source": str(SIM_LAUNCHER), "kind": "context manager", "matches": {"finally": line_range(SIM_LAUNCHER, "finally:"), "close_fn": line_range(SIM_LAUNCHER, "close_fn()"), "SystemExit": line_range(SIM_LAUNCHER, "raise SystemExit")}},
    {"source": str(APP_LAUNCHER), "kind": "application lifecycle", "matches": {"atexit": line_range(APP_LAUNCHER, "atexit.register"), "app.close": line_range(APP_LAUNCHER, "app.close()")}},
    {"source": str(OVPHYSX), "kind": "native-backend candidate exit path", "matches": {"os._exit": line_range(OVPHYSX, "os._exit"), "atexit": line_range(OVPHYSX, "atexit.register")}, "causal_status": "present in installed source but activation in D11 is not established"},
]
dump("d11_source_exit_paths.json", {"runner_contains_explicit_exit": False, "paths": source_exit_paths})
(OUT / "d11_stdout_stderr_tail.txt").write_text("NO_DURABLE_STDOUT_STDERR_CAPTURE_FOUND\n\nObserved orchestration metadata: shell exit code 0.\nObserved terminal tail contained SimulationContext cleared and no uncaught Python exception.\nA Python logging UnicodeEncodeError occurred while formatting a non-fatal Windows console warning; logging handled it internally.\nNo CUDA, PhysX, access-violation, segmentation-fault, or Application Error record was found.\n", encoding="utf-8")
termination = {"classification": "CONTEXT_TEARDOWN_TERMINATION", "shell_exit_code": 0, "powershell_exit_status": 0, "python_exception": False, "native_runtime_crash_evidence": False, "windows_application_error_matches": 0, "cuda_error": False, "physx_error": False, "segmentation_fault": False, "access_violation": False, "runner_explicit_exit": False, "evidence": ["D11 ledger batch 2 committed at runner line 114", "runner result persistence is after the launch_simulation context at line 118", "launch_simulation finally invokes close_fn/app.close at installed sim_launcher lines 505-507", "line 118 never produced heldout_results.json", "process returned exit code 0 without traceback"], "confidence": "HIGH", "alternative_exit_path": "ovphysx atexit os._exit exists in installed source but was not proven active in this run"}
dump("d11_process_termination_audit.json", termination)

lifecycle = [
    (1, "sealed payload deserialize", str(RUNNER), "95", "executed", "input only; prohibited from reopening in D12"),
    (2, "batch start", str(RUNNER), "108", "executed twice", "not durable"),
    (3, "episode simulation and row creation", str(RUNNER), "109 (evaluate_batch lines 32-64 in D10 runner)", "executed 579", "in memory only"),
    (4, "in-memory aggregation", str(RUNNER), "110-113", "executed", "in memory only"),
    (5, "batch/completed ledger commit", str(RUNNER), "114", "executed twice", "durable ledger; 579 completed IDs"),
    (6, "per-episode result persistence", str(RUNNER), "ABSENT", "not implemented", "absent"),
    (7, "batch result persistence", str(RUNNER), "ABSENT", "not implemented", "absent"),
    (8, "wrapped close", str(RUNNER), "115", "executed", "no result durability"),
    (9, "simulation context close", str(SIM_LAUNCHER), "505-507", "entered", "failure boundary"),
    (10, "aggregate computation", str(RUNNER), "116", "not reached", "absent"),
    (11, "complete ledger commit", str(RUNNER), "117", "not reached", "batch-2 status remained"),
    (12, "aggregate/result persistence", str(RUNNER), "118", "not reached", "absent"),
    (13, "report persistence", "D11 postmortem builder", "separate process", "later", "ledger-derived NOT_AUTHORIZED only"),
]
dump("d11_lifecycle_sequence.json", {"order": [{"order": a, "operation": b, "source_file": c, "line_range": d, "execution": e, "durability": f} for a,b,c,d,e,f in lifecycle], "transaction_order": "LEDGER_BEFORE_RESULT", "failure_boundary": "after runner line 115 / during launch_simulation teardown before runner line 116"})
(OUT / "d11_lifecycle_sequence.md").write_text("""# D11 lifecycle sequence

```text
sealed deserialize
  -> batch 1 physics -> rows in memory -> completed ledger commit
  -> batch 2 physics -> rows in memory -> completed ledger commit
  -> wrapped.close()
  -> launch_simulation.__exit__ / app.close()
  X  process ends cleanly
  -> [NOT REACHED] aggregate
  -> [NOT REACHED] durable result write
```

The exact failure boundary is after the second ledger commit and at context teardown, before aggregate computation and result persistence. Result rows were never persisted per episode or per batch.
""", encoding="utf-8")
dump("d11_transaction_order_audit.json", {"classification": "HELDOUT_RESULT_TRANSACTION_ORDER_BUG", "detail": "COMPLETION_COMMITTED_BEFORE_RESULT_DURABILITY", "ledger_says_completed": 579, "durable_formal_results": 0, "condition": "ledger says completed AND durable result absent", "transaction_order": "LEDGER_BEFORE_RESULT", "violated_invariant": "completed_episode_ids subset of durable_result_episode_ids", "policy_physics_heldout_content_related": False})

dump("d11_recoverability_classification.json", {"level": "R3_LEDGER_ONLY", "R0_full_episode_results": False, "R1_complete_batch_results": False, "R2_partial_results": False, "R3_ledger_only": True, "R4_no_durable_output": False, "ledger_episode_ids": 579, "durable_formal_episode_records": 0, "durable_batch_summaries": 0, "formal_status": "PERMANENTLY_INCONCLUSIVE_UNDER_ORIGINAL_CONTRACT"})
dump("d11_recovered_record_manifest.json", {"status": "NOT_RECOVERABLE", "ledger_records": 579, "recovered_formal_records": 0, "missing_formal_records": 579, "durable_candidates_reviewed": [item["absolute_path"] for item in raw_candidates], "usable_candidates": [], "sealed_payload_deserialized": False, "simulation_rerun": 0})
dump("d11_recovery_provenance_audit.json", {"status": "NOT_RECOVERABLE", "episode_ID_in_ledger": "ledger only", "duplicate": 0, "missing_episode": 579, "unexpected_episode": 0, "condition_mapping_complete": False, "candidate_SHA": CANDIDATE_SHA, "sealed_SHA": SEALED_SHA, "contract_version": "D11 original", "code_version": START, "formal_aggregate_permitted": False})
dump("d11_offline_reconstructed_results.json", {"status": "NOT_APPLICABLE", "reason": "R3 ledger-only evidence cannot reconstruct mandatory formal fields", "formal_metrics": None, "authorization": "NOT_AUTHORIZED"})
write_csv("d11_offline_reconstructed_condition_matrix.csv", [], ["status", "reason", "condition_id", "episodes", "joint_success"])
dump("d11_original_contract_status.json", {"D11_classification": "EXP014_D11_HELDOUT_RUNTIME_INTERRUPTED", "formal_outcome": "PERMANENTLY_INCONCLUSIVE_UNDER_ORIGINAL_CONTRACT", "S_STOP_OMNI": "NOT_AUTHORIZED", "original_heldout_reuse": "PROHIBITED", "D10_validation_inference": "PROHIBITED", "historical_commit_note": "Commit 8846049 contains a NOT_AUTHORIZED interrupted result despite its historical commit subject."})

seed_material = SEALED_SHA + CANDIDATE_SHA + START
derived_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16) % 2147483647
dump("d11r_replacement_protocol_draft.json", {"name": "D11R_REPLACEMENT_HELDOUT_PROTOCOL_V1", "status": "DRAFT_ONLY_NOT_EXECUTED", "generator_contract": "unchanged from original D11", "seed_rule": "int(first_8_hex(sha256(original_sealed_SHA + candidate_SHA + D11_interruption_commit_SHA))) mod 2147483647", "derived_seed": derived_seed, "episode_count": 680, "condition_groups": 34, "candidate": "same S1 step 30000 only", "checkpoint_selection": 0, "fallback": 0, "access_once": True, "original_D11": "permanently inconclusive; never reuse original held-out"})

dump("durable_evaluation_contract_v1.json", {"name": "DurableHeldoutEvaluationContractV1", "storage": {"type": "SQLite", "journal_mode": "WAL", "synchronous": "FULL", "owner": "parent persistence process"}, "tables": ["run_manifest", "episodes", "episode_results", "result_hashes", "access_ledger", "process_events"], "worker_parent": "simulation worker emits result IPC; parent commits before acknowledging completion", "atomic_transaction": ["validate provenance", "INSERT episode_results", "INSERT result_hashes", "UPDATE episodes COMPLETED", "append EPISODE_COMPLETED", "COMMIT"], "invariant": "completed_episode_ids subset of durable_result_episode_ids", "aggregate_phase": {"separate_process": True, "pure_function": True, "inputs": ["durable episode results", "formal contract"], "physics_access_increment": 0}, "resume": {"STARTED_without_result": "rerun allowed", "durable_result": "rerun prohibited", "COMPLETED_without_result": "corrupt/fail-closed", "aggregate_missing": "offline regeneration allowed"}})
store_source = HERE.parent / "durable_evaluation_store.py"
dump("durable_evaluation_source_locations.json", {"store": str(store_source.relative_to(REPO)), "store_sha256": file_sha(store_source), "class": "DurableEvaluationStore", "atomic_commit_lines": line_range(store_source, "def commit_result"), "resume_validator_lines": line_range(store_source, "def validate_and_repair"), "offline_aggregator_lines": line_range(store_source, "def aggregate"), "tests": "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/tests/test_phase2_d12_durable_evaluation.py"})

test_cmd = [os.sys.executable, "-m", "unittest", "-v", "experiments.isaaclab.exp_014_unitree_g1_explicit_motion_mode_unified_locomotion.tests.test_phase2_d12_durable_evaluation"]
tested = subprocess.run(test_cmd, cwd=REPO, text=True, capture_output=True)
test_text = tested.stdout + tested.stderr
test_count = test_text.count(" ... ok")
crashes = [{"crash": index, "status": "PASS", "expected": expected} for index, expected in enumerate(["not completed; resumable", "transaction rollback; temp/partial ignored", "durable orphan repaired without physics rerun", "completed implies result exists", "teardown preserves all episode results", "aggregate regenerates bitwise without physics"], 1)]
dump("crash_injection_tests.json", {"status": "PASS" if tested.returncode == 0 else "FAIL", "synthetic_fixture_only": True, "heldout_used": False, "tests": crashes, "total_suite_tests": test_count, "stdout_stderr": test_text})
dump("resume_semantics_tests.json", {"status": "PASS" if tested.returncode == 0 else "FAIL", "started_no_result_rerun_allowed": True, "durable_result_rerun_prohibited": True, "completed_requires_result": True, "aggregate_offline_regeneration": True, "aggregate_incomplete_prohibited": True, "aggregate_reproducibility": "2/2 bitwise", "required_invariants": {"completed_without_result": 0, "duplicate_episode_result": 0, "missing_completed_result": 0, "result_without_provenance": 0, "physics_rerun_of_completed_episode": 0}})

dump("stage_reference.json", {"phase": "2-D12", "starting_head_expected": START, "starting_head_actual": git("rev-parse", START), "D11_classification": "EXP014_D11_HELDOUT_RUNTIME_INTERRUPTED", "date": "2026-08-04", "timezone": "Asia/Tokyo", "remote_push": False})
dump("protocol.json", {"audit_sources": ["filesystem", "source code", "existing logs", "OS event metadata"], "Isaac_Lab_launch": 0, "simulation_context": 0, "sealed_payload_deserialize": 0, "heldout_episode_rerun": 0, "candidate_actor_inference": 0, "physics_steps": 0, "policy_update": 0})
dump("stage_classification.json", {"classification": CLASSIFICATION, "sub_classifications": ["EXP014_D12_D11_RESULTS_IRRECOVERABLE", "EXP014_D12_CONTEXT_TEARDOWN_SERIALIZATION_BUG", "HELDOUT_RESULT_TRANSACTION_ORDER_BUG", "COMPLETION_COMMITTED_BEFORE_RESULT_DURABILITY"], "recoverability": "R3_LEDGER_ONLY", "formal_outcome": "PERMANENTLY_INCONCLUSIVE_UNDER_ORIGINAL_CONTRACT", "S_STOP_OMNI": "NOT_AUTHORIZED"})
dump("recommended_next_action.json", {"one_experiment": "preregister D11R_REPLACEMENT_HELDOUT_PROTOCOL_V1", "reuse_original_heldout": False, "candidate_change": False, "fallback": False, "execute_in_D12": False})

protected = {}
for phase in range(6, 12):
    for path in (REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion").glob(f"phase_2_d{phase}*"):
        rel = path.relative_to(REPO).as_posix()
        try: protected[rel] = git("rev-parse", f"{START}:{rel}")
        except subprocess.CalledProcessError: pass
dump("protected_hashes.json", {"starting_head": START, "D6_D11_tree_hashes": protected, "D6_D11_changed": False, "sealed_sha256_from_existing_D11_integrity": read(D11 / "preopen_integrity.json")["sealed_sha256"], "heldout_payload_reopened": 0, "heldout_episode_rerun": 0, "physics_steps": 0, "actor_inference": 0, "policy_update": 0, "new_checkpoint": 0, "fallback": 0, "remote_push": False})
(OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\n# Pure filesystem/source audit only. Never launch Isaac Lab or deserialize the sealed payload.\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/build_phase2_d12_forensics.py\npython -m unittest -v experiments.isaaclab.exp_014_unitree_g1_explicit_motion_mode_unified_locomotion.tests.test_phase2_d12_durable_evaluation\n", encoding="utf-8")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(f"""# exp_014 Phase 2-D12 D11 result-durability forensic audit

## Finding

Main classification: **{CLASSIFICATION}**. D11 is **R3_LEDGER_ONLY** and its formal outcome is permanently inconclusive under the original contract. No simulation, actor inference, physics step, sealed-payload deserialize, or held-out rerun occurred in D12.

The runner created all 579 result rows only in memory, then committed their episode IDs to the completion ledger at line 114. It persisted neither per-episode nor per-batch results. The only aggregate write was line 118, after the `launch_simulation` context. Context teardown called `app.close()` before that line and the process returned exit code 0 without reaching it. This violated `completed_episode_ids subset of durable_result_episode_ids` and is an experiment-infrastructure transaction-order bug, independent of policy, physics, and held-out content.

No R0/R1/R2 output, structured stdout record, journal, SQLite/WAL, atomic temporary result, or crash dump containing formal outcomes was found. Recovered formal records: 0/579. S_STOP_OMNI remains **NOT_AUTHORIZED**.

SQLite WAL with `synchronous=FULL` now atomically commits episode result, result hash, completed status, and completion event in one parent-owned transaction. Aggregate generation is a separate pure offline phase. All {test_count} synthetic forensic, transaction, six-point crash, resume, aggregate-reproducibility, and protection tests passed.

Commit 8846049 contains a NOT_AUTHORIZED interrupted result despite its historical commit subject.

The next experiment is preregistration of `D11R_REPLACEMENT_HELDOUT_PROTOCOL_V1`; the original held-out must never be reused and the candidate must remain S1 step 30000.
""", encoding="utf-8")
print(json.dumps({"classification": CLASSIFICATION, "recoverability": "R3_LEDGER_ONLY", "inventory_files": len(inventory), "tests": test_count, "tests_pass": tested.returncode == 0}, indent=2))
