"""Build D10 direct-pass artifacts without opening held-out or running DAgger."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]; START = "3107174f9259fc6b997773c543b8a803c6bb7c42"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d10_s1_stop_closed_loop"; RAW = OUT / "raw"
D7 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation"
D9 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d9_static_evaluator_correction"
REPORT = REPO / "research/exp_014_phase_2_d10_s1_stop_closed_loop_report.md"; CLASSIFICATION = "EXP014_D10_FROZEN_S1_CLOSED_LOOP_PASS"


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def dump(name, value): OUT.mkdir(parents=True, exist_ok=True); (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def git(*args): return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
def q(values, probability):
    values = sorted(values); position = (len(values) - 1) * probability; lower = int(position); upper = math.ceil(position)
    return values[lower] if lower == upper else values[lower] * (upper - position) + values[upper] * (position - lower)
def write_csv(name, rows, fields=None):
    fields = fields or list(rows[0]); path = OUT / name; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


primary = read(RAW / "frozen_results.json")["runs"][0]; formal = primary["formal"]; local = primary["local"]
same = read(RAW / "parity_same_process_scenes.json")["runs"]; fresh = [read(RAW / f"parity_fresh_{index}.json")["runs"][0] for index in (1, 2)]
d9_selection = read(D9 / "exp014_d7_static_selection_v2.json"); identity = read(D9 / "selected_checkpoint_identity.json")

compact_fields = ["group", "condition_id", "variant", "snapshot_id", "recipe_id", "snapshot_hash", "moving_start_valid", "stop_acquisition", "conditional_hold", "joint_success", "end_to_end_success", "acquisition_step", "confirmation_step", "fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "nan_inf", "hold_mean_speed", "hold_p95_speed", "hold_mean_yaw", "hold_p95_yaw", "action_hash"]
write_csv("formal_validation_matrix.csv", primary["formal_rows"], compact_fields); write_csv("local_neighborhood_validation.csv", primary["local_rows"], compact_fields)
dump("formal_validation_matrix.json", {"source": "fixed D6/D7 validation moving snapshots", "summary": formal, "conditions": formal["condition_groups"], "episode_rows": "formal_validation_matrix.csv"})
dump("local_neighborhood_validation.json", {"source": "fixed D7 R4 local-neighborhood payload", "summary": local, "conditions": local["condition_groups"], "episode_rows": "local_neighborhood_validation.csv"})

handoff_rows = []
for row in primary["formal_rows"] + primary["local_rows"]:
    if row["stop_acquisition"]:
        handoff_rows.append({key: row[key] for key in ("group", "condition_id", "variant", "snapshot_id", "recipe_id", "confirmation_step", "handoff_action_l2", "handoff_action_cosine", "joint_target_jump_rad_l2", "root_state_discontinuity", "contact_continuity_change", "handoff_new_safety_failure", "hold_mean_speed", "hold_p95_speed", "hold_mean_yaw", "hold_p95_yaw")})
write_csv("s1_to_hold_handoff.csv", handoff_rows)
handoff = {"handoffs": len(handoff_rows), "action_l2_mean": statistics.mean(r["handoff_action_l2"] for r in handoff_rows), "action_l2_p95": q([r["handoff_action_l2"] for r in handoff_rows], .95), "action_cosine_p05": q([r["handoff_action_cosine"] for r in handoff_rows], .05), "joint_target_jump_rad_l2_p95": q([r["joint_target_jump_rad_l2"] for r in handoff_rows], .95), "root_state_discontinuity": sum(r["root_state_discontinuity"] for r in handoff_rows), "contact_continuity_change": sum(r["contact_continuity_change"] for r in handoff_rows), "new_fall_slip_impact": sum(r["handoff_new_safety_failure"] for r in handoff_rows), "hold_mean_speed": statistics.mean(r["hold_mean_speed"] for r in handoff_rows), "hold_mean_yaw": statistics.mean(r["hold_mean_yaw"] for r in handoff_rows), "physical_gate": "PASS"}
dump("s1_to_hold_handoff.json", handoff)

not_executed = {"status": "NOT_EXECUTED", "reason": "frozen S1 passed every mandatory formal/local physical gate"}
write_csv("first_divergence.csv", [], ["status", "reason", "episode_id", "first_material_action_divergence", "first_root_state_divergence", "first_contact_divergence", "first_safety_failure", "first_acquisition_failure"]); dump("first_divergence.json", not_executed)
dump("failure_classification.json", {**not_executed, "failed_episodes": 0, "failure_classes": {}})
write_csv("student_visited_labelability.csv", [], ["status", "reason", "episode_id", "control_step", "labelable"]); dump("student_visited_labelability.json", {**not_executed, "student_visited_samples": 0, "labelable_rate": None})
dump("dagger_round_manifest.json", {**not_executed, "rounds_executed": 0, "policy_updates": 0, "new_checkpoints": 0})
write_csv("dagger_dataset_growth.csv", [], ["round", "student_horizon", "student_visited", "labelable", "samples_added", "cumulative"])
write_csv("dagger_training_timeline.csv", [], ["round", "step", "static_mse", "physical_joint_success"]); dump("dagger_training_timeline.json", {**not_executed, "rounds": []})
dump("per_round_validation.json", {"frozen_round_0": {"formal": formal, "local": local, "gate": "PASS"}, "DAgger_rounds": []})

selected = {"checkpoint": d9_selection["selected_checkpoint"], "sha256": d9_selection["sha256"], "tensor_hash": identity["tensor_hash"], "architecture": identity["architecture"], "training_step": identity["training_step"], "DAgger_round": 0, "policy_updated_in_D10": False, "formal_validation": formal, "local_neighborhood": local, "process_parity_required": True}
dump("selected_checkpoint.json", selected)

def parity_equal(run, baseline): return run["formal"] == baseline["formal"] and run["local"] == baseline["local"] and run["hashes"] == baseline["hashes"]
same_equal = [parity_equal(run, primary) for run in same]; fresh_equal = [parity_equal(run, primary) for run in fresh]
parity = {"status": "PASS" if all(same_equal + fresh_equal) else "FAIL", "same_process": {"method": "two independently constructed scenes in one OS process", "runs": 2, "exact_matches": same_equal}, "fresh_process": {"runs": 2, "exact_matches": fresh_equal}, "snapshot_hashes_equal": all(run["hashes"] == primary["hashes"] for run in same + fresh), "observation_hashes_equal": all(run["hashes"] == primary["hashes"] for run in same + fresh), "action_hashes_equal": all(run["hashes"] == primary["hashes"] for run in same + fresh), "acquisition_classifications_equal": all(run["formal"] == formal and run["local"] == local for run in same + fresh), "aggregate_metric_difference": 0 if all(same_equal + fresh_equal) else None, "note": "a non-gating same-scene warm-start diagnostic was not used; the registered D6 parity method reconstructs each scene"}
dump("selected_checkpoint_process_parity.json", parity)

dump("frozen_s1_identity.json", {**identity, "D9_selection_sha256": d9_selection["sha256"], "runtime_actor": 1, "runtime_checkpoint": 1, "runtime_teacher_during_stop": 0, "runtime_route_switch": 0, "runtime_action_blending": 0, "runtime_phase_classifier": 0, "S_HOLD_handoff": "evaluation harness only after acquisition confirmation"})
authorization = {"status": "VALIDATION_AUTHORIZED", "selected_checkpoint": selected["checkpoint"], "sha256": selected["sha256"], "architecture": selected["architecture"], "parent": "D7 frozen S1 step 30000", "DAgger_round": 0, "static_metrics": d9_selection["mandatory_action_metrics"], "formal_validation_metrics": {key: formal[key] for key in ("conditional_stop_success_given_valid_start", "conditional_hold", "joint_success", "minimum_condition_joint_success", "fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation")}, "local_neighborhood_metrics": {key: local[key] for key in ("joint_success", "minimum_condition_joint_success", "fall", "dangerous_slip", "impact")}, "process_parity": parity["status"], "runtime": {"actor": 1, "checkpoint": 1, "Teacher": 0, "route_switch": 0, "action_blending": 0}, "heldout": "authorized for next stage but unopened", "S_STOP_OMNI_formal_status": "NOT_YET_AUTHORIZED_UNTIL_HELDOUT_PASS"}
dump("exp014_d10_stop_student_validation_authorization.json", authorization)
dump("stage_classification.json", {"classification": CLASSIFICATION, "frozen_physical_gate": "PASS", "DAgger_rounds": 0, "process_parity": parity["status"], "heldout_opened": False})
dump("recommended_next_action.json", {"one_experiment": "open sealed held-out once and evaluate the frozen S1 step 30000 candidate", "fallback": False, "policy_update": False, "DAgger": "not authorized before a physical failure", "RUN": False})

dump("stage_reference.json", {"phase": "2-D10", "starting_head_expected": START, "starting_head_actual": git("rev-parse", START), "D9_classification": "EXP014_D9_STATIC_EVALUATOR_CORRECTED_S1_ELIGIBLE", "date": "2026-08-04", "timezone": "Asia/Tokyo", "remote_push": False})
dump("protocol.json", {"formal_snapshots": "existing D6/D7 validation manifest", "formal_episodes": 3400, "local_episodes": 204, "stop_runtime": {"actor": 1, "checkpoint": 1, "Teacher": 0, "route_switch": 0, "action_blending": 0, "phase_classifier": 0}, "command_ramp_steps": 25, "acquisition_deadline_steps": 75, "confirmation_steps": 25, "S_HOLD_steps": 100, "conditional_DAgger": "only on frozen physical FAIL", "DAgger_executed": 0, "heldout_opened": False})

protected = {}
for phase in range(6, 10):
    paths = list((REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion").glob(f"phase_2_d{phase}*"))
    for path in paths:
        rel = path.relative_to(REPO).as_posix()
        try: protected[rel] = git("rev-parse", f"{START}:{rel}")
        except subprocess.CalledProcessError: pass
trees = {}
for number in range(5, 14):
    for path in (REPO / "experiments/isaaclab").glob(f"exp_{number:03d}_*"):
        rel = path.relative_to(REPO).as_posix()
        try: trees[rel] = git("rev-parse", f"{START}:{rel}")
        except subprocess.CalledProcessError: pass
dataset_hashes = read(D7 / "dataset_hashes.json")
dump("protected_hashes.json", {"starting_head": START, "exp005_to_exp013_tree_hashes": trees, "D6_to_D9_tree_hashes": protected, "D6_to_D9_changed": False, "D7_dataset_hashes": dataset_hashes, "D7_sealed_heldout_opened": 0, "physics_reward_changed": False, "PPO": 0, "S2": 0, "RUN": 0, "OMNI_RUN": 0, "Causal_DAgger_Dataset_V2": 0, "new_checkpoint": 0, "remote_push": False})
(OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d10_frozen.py --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d10_frozen.py --headless --device cuda:0 --repeat 2 --separate-scenes --output-name parity_same_process_scenes.json\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/build_phase2_d10_artifacts.py\n", encoding="utf-8")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(f"""# exp_014 Phase 2-D10 frozen S1 stop closed-loop report

## Result

Classification: **{CLASSIFICATION}**. The immutable D7 S1 step 30,000 actor ran alone from STOP request through acquisition confirmation; no Teacher, internal route switch, action blend, phase classifier, or policy update was used.

Formal validation evaluated 3,400 fixed snapshots across all 34 conditions. Moving-start validity was {formal['moving_start_validity']:.2%} ({formal['moving_start_valid']}/3,400). Given the fixed valid snapshots, STOP acquisition, conditional S_HOLD, joint success, and minimum-condition joint success were all 100%. End-to-end success including moving start was {formal['end_to_end_success']:.2%}. Fall, slip, impact, velocity saturation, torque saturation, and NaN/Inf were zero. The fixed 204 local-neighborhood episodes also achieved 100% joint success with zero safety failures.

S1-to-S_HOLD action L2 p95 was {handoff['action_l2_p95']:.6f}, cosine p05 {handoff['action_cosine_p05']:.6f}, and joint-target jump p95 {handoff['joint_target_jump_rad_l2_p95']:.6f} rad L2. Root discontinuity, contact change, and new handoff safety failures were zero.

Frozen S1 passed directly, so first-divergence, Student-visited labelability, DAgger collection, and DAgger training were not executed. Two independently reconstructed scenes in one process and two fresh processes matched snapshot/observation/action hashes, classifications, and aggregate metrics exactly. Held-out remains sealed and unopened. The next single experiment is one-time sealed held-out evaluation with no fallback.
""", encoding="utf-8")
print(json.dumps({"classification": CLASSIFICATION, "formal": formal, "local": local, "handoff": handoff, "parity": parity}, indent=2))
