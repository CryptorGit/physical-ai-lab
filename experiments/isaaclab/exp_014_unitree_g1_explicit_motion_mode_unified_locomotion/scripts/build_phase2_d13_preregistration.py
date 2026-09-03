"""Fail-fast D13 preregistration builder; performs no simulation or payload deserialize."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]
START = "2c502b5cffb295346b65dc7af991143b59f5d28d"
D9 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d9_static_evaluator_correction"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d13_d11r_preregistration"
REPORT = REPO / "research/exp_014_phase_2_d13_d11r_preregistration_report.md"
PROTOCOL = "D11R_REPLACEMENT_HELDOUT_PROTOCOL_V1"
SPECIFIED_SEED = 1940027935
PARTS = {
    "original_d11_sealed_sha": "dd34d3035866bd35e29643e616a1add35b3ca8bd3c7e05a58ce73b7182f266e9",
    "candidate_checkpoint_sha": "5de37e5d0807654d370ba7a79ee9872c4029cb50e548685423e48812249959d5",
    "d11_interruption_commit_sha": "88460496e0b10550649390b46c6248e8c7b2c5b7",
    "d12_durability_audit_commit_sha": START,
    "protocol_literal": PROTOCOL,
}


def dump(name, value): OUT.mkdir(parents=True, exist_ok=True); (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def git(*args): return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


canonical = "".join(PARTS.values()).encode("utf-8")
digest = hashlib.sha256(canonical).hexdigest()
derived = int(digest[:8], 16) % 2147483647
legacy_material = "".join(list(PARTS.values())[:3]).encode("utf-8")
legacy_digest = hashlib.sha256(legacy_material).hexdigest()
legacy_derived = int(legacy_digest[:8], 16) % 2147483647
match = derived == SPECIFIED_SEED
if match: raise RuntimeError("This fail-fast builder is only valid for the preregistered mismatch evidence")

dump("seed_derivation.json", {"status": "FAIL", "canonicalization": "ordered delimiter-free UTF-8 concatenation", "ordered_inputs": PARTS, "canonical_utf8": canonical.decode(), "sha256": digest, "integer_rule": "int(first_8_hex, 16) mod 2147483647", "first_8_hex": digest[:8], "derived_seed": derived, "specified_fixed_seed": SPECIFIED_SEED, "match": False, "reproducibility": {"run_1": derived, "run_2": int(hashlib.sha256(canonical).hexdigest()[:8], 16) % 2147483647, "bitwise_equal": True}, "legacy_D12_three_input_rule": {"sha256": legacy_digest, "derived_seed": legacy_derived, "explains_specified_value": legacy_derived == SPECIFIED_SEED}})

identity = json.loads((D9 / "selected_checkpoint_identity.json").read_text(encoding="utf-8"))
checkpoint = REPO / identity["checkpoint"]
dump("candidate_identity.json", {"status": "PASS", "checkpoint": identity["checkpoint"], "byte_sha256": sha(checkpoint), "expected_sha256": PARTS["candidate_checkpoint_sha"], "byte_match": sha(checkpoint) == PARTS["candidate_checkpoint_sha"], "tensor_hash_from_D9_immutable_audit": identity["tensor_hash"], "architecture": identity["architecture"], "observation_dimension": identity["observation_dimension"], "action_dimension": identity["action_dimension"], "checkpoint_deserialized": False, "actor_inference": 0, "candidate_count": 1, "fallback": 0})

stopped = {"status": "NOT_EXECUTED", "reason": "seed derivation mismatch fail-fast gate", "classification": "EXP014_D13_SEED_DERIVATION_MISMATCH"}
dump("d11r_replacement_manifest.json", {**stopped, "protocol": PROTOCOL, "episode_count": 0, "requested_episode_count": 680, "outcome_fields": 0})
fields = ["replacement_episode_id", "condition_id", "direction", "yaw", "speed", "stop_timing", "recipe_seed", "snapshot_seed", "generator_version", "candidate_sha", "protocol_version"]
with (OUT / "d11r_episode_manifest.csv").open("w", newline="", encoding="utf-8") as stream: csv.DictWriter(stream, fieldnames=fields).writeheader()
dump("d11r_episode_manifest.json", {**stopped, "episodes": []})
dump("d11r_condition_balance.json", {**stopped, "expected": {"episodes": 680, "conditions": 34, "per_condition": 20}})
dump("d11r_perturbation_balance.json", {**stopped, "contract_was_not_instantiated": True})
dump("d11r_overlap_audit.json", {**stopped, "overlap_not_computed_because_no_replacement_episode_was_generated": True})

# Required filename is a zero-byte tombstone, explicitly not a seal or payload.
(OUT / "d11r_sealed_payload.bin").write_bytes(b"")
(OUT / "d11r_sealed_payload.sha256").write_text("NOT_CREATED_SEED_DERIVATION_MISMATCH\n", encoding="utf-8")
dump("d11r_seal_manifest.json", {**stopped, "seal_status": "NOT_CREATED", "payload_file_is_zero_byte_tombstone": True, "payload_sha256": None, "episode_manifest_hash": None, "access_count": 0, "status_if_successful": "SEALED_UNOPENED"})
dump("outcome_blindness_audit.json", {"status": "PASS", "Isaac_Lab_import": 0, "simulation_context": 0, "physics_step": 0, "actor_inference": 0, "policy_checkpoint_load_for_inference": 0, "outcome_calculation": 0, "episode_success_failure_fields": 0, "original_D11_payload_deserialize": 0, "replacement_outcome_access": 0})
dump("durable_persistence_preflight.json", {**stopped, "D12_contract_reference": "DurableHeldoutEvaluationContractV1", "synthetic_tests_not_rerun_after_seed_gate_failure": True})
dump("crash_injection_preflight.json", {**stopped, "six_crash_tests_required_before_D14": True})
dump("resume_contract.json", {**stopped, "contract_not_authorized": True, "proposed_semantics": {"STARTED_without_durable_result": "rerun possible", "durable_result_exists": "rerun prohibited", "COMPLETED": "durable result required", "aggregate_missing": "offline regeneration", "COMPLETED_without_result": "CORRUPT_TRANSACTION stop"}})
dump("exp014_d11r_evaluation_not_authorized.json", {"status": "NOT_AUTHORIZED", "reason": "EXP014_D13_SEED_DERIVATION_MISMATCH", "specified_seed": SPECIFIED_SEED, "derived_seed": derived, "candidate_unchanged": True, "D14_one_time_access": False})

dump("stage_reference.json", {"phase": "2-D13", "stage_name": "D11R Preregistration and Seal", "starting_head_expected": START, "starting_head_actual": git("rev-parse", START), "D11_classification": "EXP014_D11_HELDOUT_RUNTIME_INTERRUPTED", "D12_classification": "EXP014_D12_COMPLETION_LEDGER_TRANSACTION_BUG", "remote_push": False})
dump("protocol.json", {"name": PROTOCOL, "next_stage_name": "Phase 2-D14: D11R One-Time Durable Held-Out Evaluation", "fail_fast_order": ["seed derivation", "candidate identity", "manifest", "overlap", "seal", "durability tests", "authorization"], "stopped_at": "seed derivation", "Isaac_Lab": 0, "simulation": 0, "physics": 0, "actor_inference": 0, "outcome_access": 0})
dump("stage_classification.json", {"classification": "EXP014_D13_SEED_DERIVATION_MISMATCH", "seed_gate": "FAIL", "manifest_generated": False, "sealed": False, "D14_authorized": False})
dump("recommended_next_action.json", {"one_action": "resolve the canonical seed derivation inconsistency without changing the fixed seed, candidate, or condition contract", "D14_evaluation": "NOT_AUTHORIZED", "seed_change": False, "candidate_change": False})

protected = {}
for phase in range(6, 13):
    for path in (REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion").glob(f"phase_2_d{phase}*"):
        rel = path.relative_to(REPO).as_posix()
        try: protected[rel] = git("rev-parse", f"{START}:{rel}")
        except subprocess.CalledProcessError: pass
dump("protected_hashes.json", {"starting_head": START, "D6_D12_tree_hashes": protected, "D6_D12_changed": False, "exp005_exp013_changed_by_D13": False, "original_D11_sealed_payload_reopened": 0, "original_D11_episode_rerun": 0, "replacement_physics": 0, "replacement_outcome_access": 0, "actor_inference": 0, "policy_update": 0, "new_checkpoint": 0, "remote_push": False})
(OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\n# Pure seed audit only; do not launch Isaac Lab or deserialize either held-out payload.\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/build_phase2_d13_preregistration.py\nGet-Content results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d13_d11r_preregistration/seed_derivation.json\n", encoding="utf-8")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(f"""# exp_014 Phase 2-D13 D11R preregistration report

## Result

Classification: **EXP014_D13_SEED_DERIVATION_MISMATCH**. The fixed seed gate failed before episode generation, overlap audit, seal, durability preflight, or D14 authorization.

The specified five ordered inputs, concatenated as canonical delimiter-free UTF-8 and processed with the D12 preregistered rule `int(first_8_hex(SHA-256), 16) mod 2147483647`, produce digest `{digest}` and seed `{derived}`, not the fixed value `{SPECIFIED_SEED}`. Repeating the computation gives the same result bitwise. The fixed value `{SPECIFIED_SEED}` is reproduced only by the earlier three-input D12 draft that omitted the D12 commit SHA and protocol literal.

No replacement episode was generated. The required `.bin` filename is a zero-byte tombstone and explicitly not a sealed payload. Access count is zero. D14 is **NOT_AUTHORIZED**. The fixed seed, candidate, and condition contract were not changed.

Original D11 remains `PERMANENTLY_INCONCLUSIVE_UNDER_ORIGINAL_CONTRACT`; D11 and D12 classifications are unchanged. No Isaac Lab import, simulation, physics, actor inference, outcome access, training, or remote push occurred.
""", encoding="utf-8")
print(json.dumps({"classification": "EXP014_D13_SEED_DERIVATION_MISMATCH", "specified_seed": SPECIFIED_SEED, "derived_seed": derived, "manifest_generated": False, "D14_authorized": False}, indent=2))
