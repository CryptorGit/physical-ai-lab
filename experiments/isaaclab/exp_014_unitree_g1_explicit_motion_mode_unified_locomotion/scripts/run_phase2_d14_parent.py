"""Parent-owned durable coordinator for Phase 2-D14."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
START = "ebd08a8ee9a301affcf8509e562b0546404b3cd6"
D7 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation"
D9 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d9_static_evaluator_correction"
D10 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d10_s1_stop_closed_loop"
D11 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d11_stop_student_heldout"
D13R = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d13r_seed_contract_correction"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d14_d11r_durable_heldout"
DB = OUT / "durable_evaluation.sqlite"
LEDGER = OUT / "replacement_access_ledger.json"
S1 = D7 / "raw/bc_checkpoints/s1_step_30000.pt"
SEALED = D13R / "d11r_sealed_payload.bin"
WORKER = HERE.parent / "run_phase2_d14_worker.py"
STORE = HERE.parent / "durable_evaluation_store.py"
ISAACLAB = Path(r"C:\Users\user\workspace\IsaacLab\isaaclab.bat")
RUN_ID = "exp014-d14-d11r-v1"
PROTOCOL = "D11R_REPLACEMENT_HELDOUT_PROTOCOL_V1"
EXPECTED_CANDIDATE_SHA = "5de37e5d0807654d370ba7a79ee9872c4029cb50e548685423e48812249959d5"
EXPECTED_TENSOR_SHA = "e1df768438830af2da2ea393afb187b7ceb735826975019b02dc03d80dca6f78"
EXPECTED_EPISODE_MANIFEST_SHA = "b37eccb1211a39f87f8fe7326c13b88ce80bc378c8b19f750adca55bfee69f1b"
EXPECTED_PAYLOAD_SHA = "c6ef724da6fcafb25eb5c7d6a7b0b1ade17deb5cd4051a7fa16172c9465b9cfa"


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def load_store():
    import importlib.util
    spec = importlib.util.spec_from_file_location("d14_store", STORE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def preopen() -> tuple[dict, list[dict]]:
    seal = json.loads((D13R / "d11r_seal_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((D13R / "d11r_episode_manifest.json").read_text(encoding="utf-8"))
    overlap = json.loads((D13R / "d11r_overlap_audit.json").read_text(encoding="utf-8"))
    authorization = json.loads((D13R / "exp014_d11r_evaluation_authorization.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(S1, map_location="cpu", weights_only=False)
    episodes = manifest["episodes"]
    condition_counts = {str(index): sum(int(item["condition_id"]) == index for item in episodes) for index in range(34)}
    protected_patterns = [
        "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6*",
        "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7*",
        "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d8*",
        "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d9*",
        "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d10*",
        "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d11*",
        "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d12*",
        "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d13_d11r_preregistration",
        "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d13r_seed_contract_correction",
    ]
    protected_diff = git("diff", "--name-only", START, "--", *protected_patterns)
    checks = {
        "candidate_byte_sha_match": file_sha(S1) == EXPECTED_CANDIDATE_SHA,
        "candidate_tensor_hash_match": tensor_hash(checkpoint["actor_state_dict"]) == EXPECTED_TENSOR_SHA,
        "architecture_match": checkpoint["architecture"] == [141, 512, 512, 256, 37],
        "observation_dimension_141": checkpoint["architecture"][0] == 141,
        "action_dimension_37": checkpoint["architecture"][-1] == 37,
        "episode_manifest_sha_match": hashlib.sha256(canonical(episodes)).hexdigest() == EXPECTED_EPISODE_MANIFEST_SHA,
        "sealed_payload_sha_match": file_sha(SEALED) == EXPECTED_PAYLOAD_SHA,
        "episode_count_680": len(episodes) == 680,
        "condition_count_34": len(condition_counts) == 34,
        "each_condition_count_20": all(value == 20 for value in condition_counts.values()),
        "replacement_access_count_zero": seal["access_count"] == 0,
        "replacement_status_sealed_unopened": seal["status"] == "SEALED_UNOPENED",
        "overlap_audit_pass": overlap["status"] == "PASS" and overlap["all_overlap"] == 0,
        "D14_authorized": authorization["status"] == "AUTHORIZED_ONE_TIME_ACCESS",
        "protected_committed_diff_zero": not protected_diff,
        "starting_head_match": git("rev-parse", "HEAD") == START,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "candidate_sha": file_sha(S1), "candidate_tensor_hash": tensor_hash(checkpoint["actor_state_dict"]),
        "architecture": checkpoint["architecture"], "episode_manifest_sha": hashlib.sha256(canonical(episodes)).hexdigest(),
        "sealed_payload_sha": file_sha(SEALED), "condition_counts": condition_counts,
        "access_count_before": seal["access_count"], "seal_status_before": seal["status"],
        "protected_diff": protected_diff.splitlines() if protected_diff else [], "payload_deserialized": False,
        "original_D11_reopened": 0,
    }
    return result, episodes


def database_state(store, expected_ids: list[str]) -> dict:
    rows = store.db.execute("SELECT episode_id,status,attempt_count FROM episodes WHERE run_id=? ORDER BY episode_id", (RUN_ID,)).fetchall()
    completed = [episode for episode, status, _ in rows if status == "COMPLETED"]
    started = [episode for episode, status, _ in rows if status == "STARTED"]
    pending = [episode for episode, status, _ in rows if status == "PENDING"]
    attempts = {episode: count for episode, _, count in rows}
    return {"expected_episode_ids": expected_ids, "durable_completed_ids": completed, "started_but_incomplete_ids": started,
            "pending_ids": pending, "attempt_count_per_episode": attempts, "completed": len(completed)}


def update_ledger(store, opened: str, expected_ids: list[str], before: int, status: str, resume_count: int) -> None:
    state = database_state(store, expected_ids)
    dump(LEDGER, {
        "open_timestamp": opened, "replacement_payload_sha": EXPECTED_PAYLOAD_SHA,
        "episode_manifest_sha": EXPECTED_EPISODE_MANIFEST_SHA, "candidate_sha": EXPECTED_CANDIDATE_SHA,
        **state, "access_count_before": before, "access_count_after": 1,
        "resume_count": resume_count, "status": status, "result_selection_access": False,
    })


def insert_process_event(store, event: str, detail: dict) -> None:
    with store.db:
        store.db.execute("INSERT INTO process_events(run_id,worker_id,event,detail_json) VALUES(?,?,?,?)",
                         (RUN_ID, "isaac-worker", event, canonical(detail)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preopen-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    preopen_result, episodes = preopen()
    dump(OUT / "preopen_integrity.json", preopen_result)
    if preopen_result["status"] != "PASS":
        raise RuntimeError("EXP014_D14_PREOPEN_INTEGRITY_FAIL")
    if args.preopen_only:
        print(json.dumps(preopen_result, indent=2)); return
    expected_ids = [item["replacement_episode_id"] for item in episodes]
    store_mod = load_store()
    is_resume = DB.exists()
    opened = datetime.now(timezone.utc).isoformat()
    resume_count = 0
    if is_resume:
        if not LEDGER.exists():
            raise RuntimeError("existing database without replacement access ledger")
        existing_ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
        if existing_ledger["access_count_after"] != 1 or existing_ledger["replacement_payload_sha"] != EXPECTED_PAYLOAD_SHA:
            raise RuntimeError("replacement access identity mismatch")
        opened = existing_ledger["open_timestamp"]; resume_count = int(existing_ledger.get("resume_count", 0)) + 1
        store = store_mod.DurableEvaluationStore(DB); store.validate_and_repair(RUN_ID)
    else:
        store = store_mod.DurableEvaluationStore(DB)
        store.create_run(RUN_ID, EXPECTED_CANDIDATE_SHA, EXPECTED_PAYLOAD_SHA, PROTOCOL,
                         [{"episode_id": item["replacement_episode_id"], "condition_id": int(item["condition_id"])} for item in episodes])
    state = database_state(store, expected_ids)
    incomplete = state["started_but_incomplete_ids"] + state["pending_ids"]
    update_ledger(store, opened, expected_ids, 0, "OPENED_RUNNING", resume_count)
    if not incomplete:
        update_ledger(store, opened, expected_ids, 0, "PHYSICS_COMPLETE", resume_count)
        store.close(); print(json.dumps({"status": "ALREADY_COMPLETE", "episodes": 680})); return
    env = os.environ.copy(); env["D14_EPISODE_IDS_JSON"] = json.dumps(incomplete, separators=(",", ":"))
    command = subprocess.list2cmdline([str(ISAACLAB), "-p", str(WORKER), "--headless", "--device", args.device])
    process = subprocess.Popen(["cmd.exe", "/d", "/s", "/c", command], cwd=REPO, env=env,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    assert process.stdin is not None and process.stdout is not None
    log_path = OUT / "simulation_worker.log"
    finished = None
    with log_path.open("a", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line); log.flush()
            if not line.startswith("D14_IPC:"):
                continue
            message = json.loads(line[len("D14_IPC:"):]); kind, value = message["kind"], message["value"]
            if kind == "START_REQUEST":
                for episode_id in value["episode_ids"]:
                    status = store.db.execute("SELECT status FROM episodes WHERE run_id=? AND episode_id=?", (RUN_ID, episode_id)).fetchone()[0]
                    if status == "STARTED":
                        # Resume is legal only because no durable result exists.
                        exists = store.db.execute("SELECT 1 FROM episode_results WHERE run_id=? AND episode_id=?", (RUN_ID, episode_id)).fetchone()
                        if exists: raise RuntimeError("durable result may not be rerun")
                    store.start_episode(RUN_ID, episode_id, worker_id="isaac-worker")
                update_ledger(store, opened, expected_ids, 0, "RUNNING", resume_count)
                process.stdin.write("D14_ACK\n"); process.stdin.flush()
            elif kind == "RESULT":
                episode_id = value["episode_id"]
                provenance = {"candidate_sha": EXPECTED_CANDIDATE_SHA, "sealed_sha": EXPECTED_PAYLOAD_SHA,
                              "contract_version": PROTOCOL, "code_version": START}
                store.commit_result(RUN_ID, episode_id, value, provenance)
                invariants = store.invariants(RUN_ID)
                if any(invariants.values()): raise RuntimeError(f"durable invariant violation: {invariants}")
                update_ledger(store, opened, expected_ids, 0, "RUNNING", resume_count)
                process.stdin.write("D14_ACK\n"); process.stdin.flush()
            elif kind == "WORKER_FINISHED":
                finished = value
                insert_process_event(store, "WORKER_FINISHED", value)
    return_code = process.wait()
    invariants = store.invariants(RUN_ID)
    final_state = database_state(store, expected_ids)
    complete = return_code == 0 and final_state["completed"] == 680 and not any(invariants.values()) and finished is not None
    update_ledger(store, opened, expected_ids, 0, "PHYSICS_COMPLETE" if complete else "RUNTIME_INTERRUPTED_RESUMABLE", resume_count)
    dump(OUT / "worker_completion.json", {"return_code": return_code, "worker_finished": finished,
         "durable_completed": final_state["completed"], "invariants": invariants,
         "status": "PASS" if complete else "INTERRUPTED_RESUMABLE"})
    store.close()
    if not complete:
        raise RuntimeError("EXP014_D14_RUNTIME_INTERRUPTED_RESUMABLE")
    print(json.dumps({"status": "PHYSICS_COMPLETE", "durable_completed": 680, "invariants": invariants}, indent=2))


if __name__ == "__main__":
    main()
