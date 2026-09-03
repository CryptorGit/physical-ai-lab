"""Fail-closed adjudication before the sole persistent W1B-R2 run."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name, value):
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


checks = {
    "parent_strict_restore": read("w1b_r2_parent_identity_audit.json")["status"],
    "optimizer_restore": read("w1b_r2_optimizer_resume_audit.json")["status"],
    "sampler_boundary": read("pending_queue_boundary_tests.json")["status"],
    "even_path_parity": read("even_path_bitwise_parity.json")["status"],
    "odd_path_determinism": read("odd_path_determinism.json")["status"],
    "distribution": read("pending_queue_distribution_audit.json")["status"],
    "serialization": read("pending_queue_serialization_audit.json")["status"],
    "evaluation_parity": read("evaluation_parity_revalidation.json")["status"],
    "evaluation_isolation": read(
        "evaluation_process_isolation_revalidation.json"
    )["status"],
    "first_update": read("first_update_stability.json")["status"],
    "training_prefix_parity": read("training_prefix_parity.json")["status"],
    "first_odd_transition": read("first_odd_reset_transition_audit.json")["status"],
}
failed = {key: value for key, value in checks.items() if value != "PASS"}
gate = read("gate.json")
gate.update(checks)
gate["repair_gates"] = "PASS" if not failed else "FAIL"
gate["failed_repair_gates"] = failed
gate["continue_persistent_training"] = not failed
write("gate.json", gate)
print(json.dumps(checks, sort_keys=True))
if failed:
    raise SystemExit("W1B-R2 repair gate failed: " + json.dumps(failed))
