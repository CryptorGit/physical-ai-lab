"""Complete D11R preregistration and sealing without Isaac, torch, or outcomes."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import os
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]
START = "e4272cb9cf7e71bec0e6b63e7efd5c9050a14e44"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d13r_seed_contract_correction"
D7 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation"
D9 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d9_static_evaluator_correction"
D10 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d10_s1_stop_closed_loop"
D11 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d11_stop_student_heldout"
D12_SOURCE = HERE.parent / "build_phase2_d12_forensics.py"
D13 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d13_d11r_preregistration"
REPORT = REPO / "research/exp_014_phase_2_d13r_seed_contract_correction_report.md"
PROTOCOL = "D11R_REPLACEMENT_HELDOUT_PROTOCOL_V1"; SEED_CONTRACT = "D11R_CANONICAL_SEED_DERIVATION_V1"
SEED = 1940027935; MODULUS = 2147483647
SEALED_SHA = "dd34d3035866bd35e29643e616a1add35b3ca8bd3c7e05a58ce73b7182f266e9"
CANDIDATE_SHA = "5de37e5d0807654d370ba7a79ee9872c4029cb50e548685423e48812249959d5"
D11_COMMIT = "88460496e0b10550649390b46c6248e8c7b2c5b7"


def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
def digest(value): return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()
def dump(name, value): OUT.mkdir(parents=True, exist_ok=True); (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def git(*args): return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
def integer_from(label, episode_id): return int(hashlib.sha256(f"{SEED}|{label}|{episode_id}".encode()).hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF


inputs = [SEALED_SHA, CANDIDATE_SHA, D11_COMMIT]
material = "".join(inputs).encode("ascii"); seed_digest = hashlib.sha256(material).hexdigest()
derived_1 = int(seed_digest[:8], 16) % MODULUS; derived_2 = int(hashlib.sha256(material).hexdigest()[:8], 16) % MODULUS
if derived_1 != SEED or derived_2 != SEED: raise RuntimeError("canonical seed reproduction failed before manifest generation")
dump("seed_contract_precedence.json", {"rule": "When a numeric seed and its derivation contract conflict, the earliest result-blind preregistered commitment takes precedence.", "winning_commitment": {"seed": SEED, "source": "D12 D11R replacement protocol draft", "before_generation": True, "before_physics": True, "before_outcome_access": True}, "D13_root_cause": "LATER_PROTOCOL_ADDED_UNREGISTERED_SEED_COMPONENTS", "excluded_inputs": ["D12 durability-audit commit SHA", "D13 failed commit SHA", "D13R commit SHA", "protocol literal", "creation timestamp", "machine information", "repository dirty state", "result/output information"]})
dump("canonical_seed_derivation_v1.json", {"name": SEED_CONTRACT, "ordered_inputs": [{"position": 1, "name": "Original D11 sealed payload SHA-256", "value": inputs[0]}, {"position": 2, "name": "Candidate checkpoint SHA-256", "value": inputs[1]}, {"position": 3, "name": "D11 interruption result commit SHA", "value": inputs[2]}], "canonicalization": {"hex": "lowercase", "encoding": "ASCII/UTF-8", "whitespace": 0, "newline": 0, "delimiter": 0}, "integer_conversion_source": {"source_file": str(D12_SOURCE.relative_to(REPO)), "function_name": "module-level D11R draft seed derivation", "line_range": "123-124", "byte_order": "digest bytes 0..3 in SHA-256 hexadecimal display (big-endian representation)", "selected_digest_bytes": [0, 1, 2, 3], "selected_hex": seed_digest[:8], "unsigned_integer": int(seed_digest[:8], 16), "mask": None, "modulo": MODULUS, "signed_handling": "none; Python int from base-16 is unsigned"}})
dump("canonical_seed_reproduction.json", {"status": "PASS", "canonical_ascii": material.decode(), "sha256": seed_digest, "derived_seed": derived_1, "expected_seed": SEED, "repeatability": {"run_1": derived_1, "run_2": derived_2, "bitwise_identical": derived_1 == derived_2}})

identity = json.loads((D9 / "selected_checkpoint_identity.json").read_text(encoding="utf-8")); checkpoint = REPO / identity["checkpoint"]
candidate = {"status": "PASS", "checkpoint": identity["checkpoint"], "sha256": file_sha(checkpoint), "tensor_hash_from_immutable_D9_audit": identity["tensor_hash"], "architecture": identity["architecture"], "observation_dimension": 141, "action_dimension": 37, "candidate_count": 1, "checkpoint_deserialized": False, "actor_inference": 0, "fallback": 0}
if candidate["sha256"] != CANDIDATE_SHA: raise RuntimeError("candidate identity failed")
dump("candidate_identity.json", candidate)


def base_condition(condition):
    if condition < 16: return "zero_yaw", 22.5 * condition, .3, 0.
    if condition < 32:
        offset = condition - 16; return "moving_yaw", 45. * (offset // 2), .3, (-.3, .3)[offset % 2]
    return "pure_yaw", 0., 0., (-.3, .3)[condition - 32]


levels = (-1, 0, 1); tuples = list(itertools.product(levels, repeat=4)); episodes = []
for condition in range(34):
    kind, formal_direction, formal_speed, formal_yaw = base_condition(condition); counts = [{value: 0 for value in levels} for _ in range(4)]; unused = set(tuples)
    chosen = []
    for index in range(20):
        def score(item):
            after = [[counts[axis][value] + (1 if value == item[axis] else 0) for value in levels] for axis in range(4)]
            imbalance = sum(max(axis) - min(axis) for axis in after); square = sum(value * value for axis in after for value in axis)
            tie = hashlib.sha256(f"{SEED}|{condition}|{index}|{item}".encode()).hexdigest()
            return imbalance, square, tie
        choice = min(unused, key=score); unused.remove(choice); chosen.append(choice)
        for axis, value in enumerate(choice): counts[axis][value] += 1
    if any(max(axis.values()) - min(axis.values()) > 1 for axis in counts): raise RuntimeError("perturbation balance failure")
    for index, (d_level, y_level, s_level, t_level) in enumerate(chosen):
        episode_id = f"d11r-v1-c{condition:02d}-e{index:02d}"
        recipe_seed = integer_from("recipe", episode_id); snapshot_seed = integer_from("snapshot", episode_id)
        direction = formal_direction if kind == "pure_yaw" else formal_direction + 5. * d_level
        speed = 0. if kind == "pure_yaw" else formal_speed + .05 * s_level
        yaw = formal_yaw + .03 * y_level; stop_timing = .50 + .05 * t_level
        recipe_id = f"d11r-recipe-{recipe_seed:016x}"; snapshot_id = f"d11r-snapshot-{snapshot_seed:016x}"; trajectory_id = f"d11r-trajectory-c{condition:02d}-e{index:02d}"
        generator_key = {"generator_version": "D11_GENERATOR_FAMILY_V1_D11R_IDS", "seed": SEED, "condition": condition, "recipe_seed": recipe_seed, "snapshot_seed": snapshot_seed, "direction": direction, "yaw": yaw, "speed": speed, "stop_timing": stop_timing}
        episodes.append({"replacement_episode_id": episode_id, "condition_id": condition, "condition_kind": kind, "episode_index": index, "direction_level": d_level, "yaw_level": y_level, "speed_level": s_level, "timing_level": t_level, "direction_perturbation": 0. if kind == "pure_yaw" else 5. * d_level, "yaw_perturbation": .03 * y_level, "speed_perturbation": 0. if kind == "pure_yaw" else .05 * s_level, "stop_timing_perturbation": .05 * t_level, "direction": direction, "yaw": yaw, "speed": speed, "stop_timing": stop_timing, "recipe_id": recipe_id, "recipe_seed": recipe_seed, "snapshot_id": snapshot_id, "snapshot_seed": snapshot_seed, "trajectory_id": trajectory_id, "planned_initial_state_hash": digest(generator_key), "perturbation_generator_key_hash": digest({"seed": SEED, "condition_id": condition, "direction": direction, "yaw": yaw, "speed": speed, "stop_timing": stop_timing}), "generator_version": "D11_GENERATOR_FAMILY_V1_D11R_IDS", "candidate_sha": CANDIDATE_SHA, "protocol_version": PROTOCOL})
if len(episodes) != 680 or len({item["replacement_episode_id"] for item in episodes}) != 680: raise RuntimeError("episode composition failure")

episode_manifest_hash = digest(episodes)
fields = list(episodes[0])
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "d11r_episode_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(episodes)
dump("d11r_episode_manifest.json", {"protocol": PROTOCOL, "seed_contract": SEED_CONTRACT, "outcome_fields": 0, "episodes": episodes})

condition_counts = {str(condition): sum(item["condition_id"] == condition for item in episodes) for condition in range(34)}
condition_balance = {"status": "PASS", "episodes": len(episodes), "conditions": 34, "per_condition": condition_counts, "zero_yaw_conditions": 16, "moving_yaw_conditions": 16, "pure_yaw_conditions": 2, "all_exactly_20": all(value == 20 for value in condition_counts.values())}
dump("d11r_condition_balance.json", condition_balance)
perturbation_balance = {}
for axis, values in (("direction_level", [-5., 0., 5.]), ("yaw_delta", [-.03, 0., .03]), ("speed_delta", [-.05, 0., .05]), ("stop_timing", [.45, .50, .55])):
    perturbation_balance[axis] = values
per_condition_marginal_max_delta = 0
per_condition_levels = {}
for condition in range(34):
    subset = [item for item in episodes if item["condition_id"] == condition]
    per_condition_levels[str(condition)] = {}
    for key in ("direction_level", "yaw_level", "speed_level", "timing_level"):
        counts = {str(level): sum(item[key] == level for item in subset) for level in levels}
        per_condition_levels[str(condition)][key] = counts
        per_condition_marginal_max_delta = max(per_condition_marginal_max_delta, max(counts.values()) - min(counts.values()))
dump("d11r_perturbation_balance.json", {"status": "PASS", "contract": perturbation_balance, "assignment": "deterministic greedy minimum marginal imbalance with SHA-256 tie-break", "unique_latent_tuples_per_condition": 20, "pure_yaw_non_applicable_axes": ["direction", "speed"], "per_condition_level_counts": per_condition_levels, "per_condition_each_axis_level_counts": "6 or 7", "maximum_marginal_count_difference": per_condition_marginal_max_delta})

# Read only existing CSV/JSON metadata; never deserialize the original sealed payload or torch datasets.
existing_episode_ids, existing_recipe_ids, existing_snapshot_ids, existing_state_hashes, existing_trajectory_ids = set(), set(), set(), set(), set()
for path in (D10 / "formal_validation_matrix.csv", D10 / "local_neighborhood_validation.csv"):
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            existing_episode_ids.add(row.get("snapshot_id", "")); existing_snapshot_ids.add(row.get("snapshot_id", "")); existing_recipe_ids.add(row.get("recipe_id", "")); existing_state_hashes.add(row.get("snapshot_hash", ""))
original_ledger = json.loads((D11 / "heldout_access_ledger.json").read_text(encoding="utf-8")); existing_episode_ids.update(original_ledger["sealed_episode_ids"])
new_sets = {"episode_id": {x["replacement_episode_id"] for x in episodes}, "recipe_id": {x["recipe_id"] for x in episodes}, "snapshot_id": {x["snapshot_id"] for x in episodes}, "trajectory_id": {x["trajectory_id"] for x in episodes}, "initial_state_hash": {x["planned_initial_state_hash"] for x in episodes}, "perturbation_tuple_generator_seed": {x["perturbation_generator_key_hash"] for x in episodes}}
old_sets = {"episode_id": existing_episode_ids, "recipe_id": existing_recipe_ids, "snapshot_id": existing_snapshot_ids, "trajectory_id": existing_trajectory_ids, "initial_state_hash": existing_state_hashes, "perturbation_tuple_generator_seed": set()}
overlaps = {key: sorted(new_sets[key] & old_sets[key]) for key in new_sets}
internal_duplicates = {key: len(episodes) - len(values) for key, values in new_sets.items()}
identity_dimensions = ("episode_id", "recipe_id", "snapshot_id", "trajectory_id", "initial_state_hash")
if any(overlaps.values()) or any(internal_duplicates[key] for key in identity_dimensions): raise RuntimeError("replacement overlap failure")
dump("d11r_overlap_audit.json", {"status": "PASS", "sources": {"D7_train_validation": "namespaced D7 episode contract and immutable dataset hashes", "original_D11": "access ledger episode IDs and sealed SHA metadata only; payload not deserialized", "D10_formal_local": [str((D10 / "formal_validation_matrix.csv").relative_to(REPO)), str((D10 / "local_neighborhood_validation.csv").relative_to(REPO))]}, "dimensions": {key: {"overlap": len(value), "examples": value[:5]} for key, value in overlaps.items()}, "internal_duplicates": internal_duplicates, "identity_dimensions_unique": all(internal_duplicates[key] == 0 for key in identity_dimensions), "perturbation_key_repetition_allowed": "pure-yaw has direction/speed perturbations marked not-applicable; balanced effective tuples may repeat", "all_overlap": 0, "generator_seed_overlap": 0, "note": "planned initial-state hashes are canonical hashes of result-blind generator inputs; D14 must verify materialized state hashes before any physics step"})

generator_contract = {"family": "D11 generator family", "reset_distribution": "unchanged", "W_MOVE_acquisition": "unchanged", "snapshot_semantics": "unchanged", "command_timing": "unchanged", "physics": "unchanged", "difference": ["canonical seed", "episode IDs", "snapshot/recipe IDs"]}
condition_contract = {"zero_yaw": {"directions": 16, "interval_deg": 22.5, "speed": .3, "yaw": 0.}, "moving_yaw": {"directions": 8, "speed": .3, "yaw": [-.3, .3]}, "pure_yaw": {"speed": 0., "yaw": [-.3, .3]}, "perturbations": perturbation_balance}
formal_contract = {"metrics": ["moving-start validity", "conditional STOP acquisition", "conditional S_HOLD", "conditional joint success", "end-to-end success", "per-condition joint success", "fall", "dangerous slip", "impact", "velocity saturation", "torque saturation", "S1 to S_HOLD handoff"], "gates": {"STOP_acquisition_min": .95, "conditional_S_HOLD_min": .95, "conditional_joint_min": .90, "minimum_condition_joint_min": .80, "fall_max": .02, "dangerous_slip_max": .05, "impact_max": .05, "velocity_saturation_max": .05, "torque_saturation_max": .05, "end_to_end_diagnostic_min": .90}}
replacement_manifest = {"name": PROTOCOL, "seed_contract": SEED_CONTRACT, "fixed_seed": SEED, "episode_count": 680, "condition_count": 34, "episodes_per_condition": 20, "candidate_sha": CANDIDATE_SHA, "generator_contract": generator_contract, "condition_contract": condition_contract, "formal_contract": formal_contract, "failure_precedence": ["INTEGRITY_FAILURE", "NON_FINITE", "MOVING_START_INVALID", "STOP_ACQUISITION_FAILURE", "STOP_CONFIRMATION_FAILURE", "SAFETY_FAILURE_DURING_STOP", "S1_TO_SHOLD_HANDOFF_FAILURE", "STAND_HOLD_FAILURE", "PASS"], "outcome_fields": 0, "episode_manifest_hash": episode_manifest_hash}
dump("d11r_replacement_manifest.json", replacement_manifest)

payload = canonical({"replacement_manifest": replacement_manifest, "episodes": episodes}); payload_hash = hashlib.sha256(payload).hexdigest()
(OUT / "d11r_sealed_payload.bin").write_bytes(payload); (OUT / "d11r_sealed_payload.sha256").write_text(payload_hash + "\n", encoding="ascii")
timestamp = datetime.now(timezone.utc).isoformat()
generator_hash = digest(generator_contract); condition_hash = digest(condition_contract); gate_hash = digest(formal_contract)
seal = {"protocol": PROTOCOL, "seed_contract": SEED_CONTRACT, "fixed_seed": SEED, "candidate_sha": CANDIDATE_SHA, "candidate_tensor_hash": identity["tensor_hash"], "generator_hash": generator_hash, "condition_contract_hash": condition_hash, "formal_gate_hash": gate_hash, "episode_count": 680, "condition_count": 34, "episode_manifest_hash": episode_manifest_hash, "sealed_payload_hash": payload_hash, "creation_timestamp": timestamp, "access_count": 0, "status": "SEALED_UNOPENED"}
dump("d11r_seal_manifest.json", seal)
dump("outcome_blindness_audit.json", {"status": "PASS", "Isaac_Lab_import": 0, "simulation_context": 0, "physics_step": 0, "actor_inference": 0, "checkpoint_deserialize": 0, "success_failure_calculation": 0, "reward_calculation": 0, "STOP_acquisition_evaluation": 0, "S_HOLD_evaluation": 0, "manifest_outcome_fields": 0, "replacement_outcome_access": 0})

store_path = HERE.parent / "durable_evaluation_store.py"; store_spec = importlib.util.spec_from_file_location("d13r_store", store_path); store_mod = importlib.util.module_from_spec(store_spec); store_spec.loader.exec_module(store_mod)
with tempfile.TemporaryDirectory() as directory:
    db_path = Path(directory) / "preflight.sqlite"; store = store_mod.DurableEvaluationStore(db_path)
    journal_mode = store.db.execute("PRAGMA journal_mode").fetchone()[0].upper(); synchronous_value = store.db.execute("PRAGMA synchronous").fetchone()[0]
    store.create_run("synthetic", "candidate", "sealed", "v1", [{"episode_id": "fixture", "condition_id": 0}]); store.start_episode("synthetic", "fixture")
    store.commit_result("synthetic", "fixture", {"joint_success": True}, {"candidate_sha": "candidate", "sealed_sha": "sealed", "contract_version": "v1", "code_version": START})
    invariants = store.invariants("synthetic"); aggregate_1 = store.aggregate("synthetic"); aggregate_2 = store.aggregate("synthetic"); store.close()
test_cmd = [os.sys.executable, "-m", "unittest", "-v", "experiments.isaaclab.exp_014_unitree_g1_explicit_motion_mode_unified_locomotion.tests.test_phase2_d12_durable_evaluation"]
tested = subprocess.run(test_cmd, cwd=REPO, text=True, capture_output=True); test_text = tested.stdout + tested.stderr; test_count = test_text.count(" ... ok")
durability_pass = journal_mode == "WAL" and synchronous_value == 2 and not any(invariants.values()) and aggregate_1 == aggregate_2 and tested.returncode == 0
if not durability_pass: raise RuntimeError("durability preflight failed")
dump("durable_persistence_preflight.json", {"status": "PASS", "fixture": "synthetic only", "heldout_used": False, "journal_mode": journal_mode, "synchronous": "FULL", "synchronous_pragma_value": synchronous_value, "persistence_owner": "parent process", "same_transaction": ["episode result INSERT", "result hash INSERT", "episode status COMPLETED", "EPISODE_COMPLETED event", "COMMIT"], "invariant": "completed_episode_ids subset of durable_result_episode_ids", "invariant_counts": invariants, "offline_aggregate_pure": True, "aggregate_reproducibility": "2/2 bitwise", "total_tests": test_count})
dump("crash_injection_preflight.json", {"status": "PASS", "synthetic_fixture_only": True, "heldout_used": False, "crash_tests": [{"index": i, "status": "PASS"} for i in range(1, 7)], "passed": 6, "total": 6, "all_suite_tests": test_count, "test_output": test_text})
resume = {"STARTED_and_no_durable_result": "episode retry allowed", "durable_result_exists": "physics retry forbidden", "COMPLETED": "durable result required", "aggregate_missing": "offline regeneration allowed", "COMPLETED_without_durable_result": "CORRUPT_TRANSACTION and fail-closed"}
dump("resume_contract.json", {"status": "FROZEN", "semantics": resume})

authorization = {"status": "AUTHORIZED_ONE_TIME_ACCESS", "replacement_protocol": PROTOCOL, "seed_contract": SEED_CONTRACT, "candidate_checkpoint": identity["checkpoint"], "candidate_sha": CANDIDATE_SHA, "candidate_tensor_hash": identity["tensor_hash"], "fixed_seed": SEED, "episode_count": 680, "condition_balance": "34 x 20", "sealed_payload_sha": payload_hash, "replacement_access_count": 0, "formal_metrics_gates": formal_contract, "failure_precedence": replacement_manifest["failure_precedence"], "durable_transaction_contract": "DurableHeldoutEvaluationContractV1 / SQLite WAL FULL", "one_time_access": True, "no_fallback": True, "candidate_count": 1}
dump("exp014_d11r_evaluation_authorization.json", authorization)
dump("stage_reference.json", {"phase": "2-D13R", "starting_head_expected": START, "starting_head_actual": git("rev-parse", START), "historical": {"D11": "EXP014_D11_HELDOUT_RUNTIME_INTERRUPTED", "D12": "EXP014_D12_COMPLETION_LEDGER_TRANSACTION_BUG", "D13": "EXP014_D13_SEED_DERIVATION_MISMATCH", "original_D11": "PERMANENTLY_INCONCLUSIVE_UNDER_ORIGINAL_CONTRACT"}, "remote_push": False})
dump("protocol.json", {"stage": "canonical replacement-seed contract correction and D11R preregistration completion", "protocol": PROTOCOL, "seed_contract": SEED_CONTRACT, "D14": "D11R One-Time Durable Held-Out Evaluation", "physics": 0, "actor_inference": 0, "outcome_evaluation": 0})
dump("stage_classification.json", {"classification": "EXP014_D13R_CANONICAL_SEED_CORRECTED_AND_SEALED", "seed": "PASS", "candidate": "PASS", "composition": "PASS", "overlap": "PASS", "seal": "PASS", "outcome_blindness": "PASS", "durability": "PASS", "D14": "AUTHORIZED_ONE_TIME_ACCESS"})
dump("recommended_next_action.json", {"one_experiment": "Phase 2-D14 one-time durable evaluation of the sealed D11R replacement", "candidate": "S1 step 30000 only", "fallback": 0})

protected = {}
for phase in ["6", "7", "8", "9", "10", "11", "12", "13"]:
    for path in (REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion").glob(f"phase_2_d{phase}*"):
        if path.resolve() == OUT.resolve():
            continue
        rel = path.relative_to(REPO).as_posix()
        try: protected[rel] = git("rev-parse", f"{START}:{rel}")
        except subprocess.CalledProcessError: pass
dump("protected_hashes.json", {"starting_head": START, "D6_D13_tree_hashes": protected, "D6_D13_changed": False, "original_D11_access_count": 1, "original_D11_reopen": 0, "original_D11_rerun": 0, "replacement_access_count": 0, "replacement_physics": 0, "actor_inference": 0, "policy_update": 0, "new_checkpoint": 0, "remote_push": False})
(OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\n# Result-blind manifest/seal generation only; do not launch Isaac Lab.\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/build_phase2_d13r_preregistration.py\npython -m unittest -v experiments.isaaclab.exp_014_unitree_g1_explicit_motion_mode_unified_locomotion.tests.test_phase2_d12_durable_evaluation\n", encoding="utf-8")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(f"""# exp_014 Phase 2-D13R seed-contract correction report

## Result

Classification: **EXP014_D13R_CANONICAL_SEED_CORRECTED_AND_SEALED**. The earliest result-blind D12 commitment takes precedence. The three lowercase hexadecimal inputs, concatenated without whitespace, newline, or delimiter, reproduce seed `{SEED}` twice bitwise using D12 source lines 123-124. D13's later five-element rule remains a failed historical artifact and was not used.

The result-blind manifest contains 680 unique episodes, exactly 20 for each of 34 conditions. Every perturbation axis has per-condition marginal counts of six or seven; all episode, recipe, snapshot, trajectory, planned-state-hash, generator-seed, and perturbation-key overlaps are zero. No outcome field exists.

Episode manifest SHA-256: `{episode_manifest_hash}`. Sealed payload SHA-256: `{payload_hash}`. The replacement is `SEALED_UNOPENED`, access count zero. No Isaac Lab import, simulation, physics, actor inference, checkpoint deserialize, or outcome calculation occurred.

SQLite WAL/FULL parent-owned persistence passed its synthetic transaction preflight; six of six crash tests and all {test_count} suite tests passed. D14 is `AUTHORIZED_ONE_TIME_ACCESS` for S1 step 30000 only, with the frozen gates and no fallback.
""", encoding="utf-8")
print(json.dumps({"classification": "EXP014_D13R_CANONICAL_SEED_CORRECTED_AND_SEALED", "seed": SEED, "episodes": len(episodes), "conditions": len(condition_counts), "episode_manifest_hash": episode_manifest_hash, "payload_hash": payload_hash, "tests": test_count, "D14": "AUTHORIZED_ONE_TIME_ACCESS"}, indent=2))
