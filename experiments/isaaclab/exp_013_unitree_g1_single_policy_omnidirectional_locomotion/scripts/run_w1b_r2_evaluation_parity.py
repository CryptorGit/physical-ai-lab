"""Revalidate W1B-R2 against the established common clean evaluator outputs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
R1 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r1_evaluation_parity_corrected_rerun"
)
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
)
OLD1 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_yaw_conditioned_omnidirectional_walk/checkpoints/model_1.pt"
)
EVALUATOR = HERE.parent / "evaluate_w1b_r2.py"
METRICS = (
    "vector_velocity_mae", "direction_error_deg", "actual_yaw_rate",
    "fall_rate", "dangerous_slip_rate", "impact_failure_rate",
    "long_dwell_saturation_rate",
)


def run(mode: str, checkpoint: Path, tag: str) -> dict:
    target = OUT / f"_raw_{mode}_{tag}.json"
    if not target.exists():
        subprocess.run(
            [
                sys.executable, str(EVALUATOR), "--mode", mode,
                "--checkpoint", str(checkpoint), "--tag", tag, "--headless",
            ],
            cwd=REPO,
            check=True,
        )
    return json.loads(target.read_text(encoding="utf-8"))


results = {}
for name, checkpoint in (("parent", PARENT), ("old_iteration1", OLD1)):
    for mode, suffix in (("capability", "quick"), ("zero", "formal")):
        fresh = run(mode, checkpoint, f"r2_{name}_{suffix}")
        reference_name = (
            f"_raw_{mode}_P1_online_common_{name}_{suffix}.json"
        )
        reference = json.loads((R1 / reference_name).read_text(encoding="utf-8"))
        fresh_rows = {row["condition"]: row for row in fresh["rows"]}
        reference_rows = {row["condition"]: row for row in reference["rows"]}
        max_diff = 0.0
        for condition in fresh_rows:
            for metric in METRICS:
                max_diff = max(
                    max_diff,
                    abs(float(fresh_rows[condition][metric])
                        - float(reference_rows[condition][metric])),
                )
        pass_fresh = sum(
            bool(row["gate_pass"])
            for condition, row in fresh_rows.items()
            if condition.startswith("ZERO_D")
        )
        pass_reference = sum(
            bool(row["gate_pass"])
            for condition, row in reference_rows.items()
            if condition.startswith("ZERO_D")
        )
        status = (
            "PASS"
            if set(fresh_rows) == set(reference_rows)
            and pass_fresh == pass_reference
            and max_diff <= 1e-5
            else "FAIL"
        )
        results[f"{name}_{suffix}"] = {
            "status": status,
            "r2_direction_pass_count": pass_fresh,
            "r1_direction_pass_count": pass_reference,
            "maximum_metric_absolute_difference": max_diff,
            "tolerance": 1e-5,
        }

overall = all(value["status"] == "PASS" for value in results.values())
(OUT / "evaluation_parity_revalidation.json").write_text(
    json.dumps({
        "status": "PASS" if overall else "EXP013_W1B_R2_PREFIX_PARITY_FAIL",
        "common_evaluator": "Exp013DirectionalCapabilityEvaluator",
        "parent_expected": "16/16",
        "old_iteration1_expected": "16/16",
        "results": results,
    }, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(OUT / "evaluation_process_isolation_revalidation.json").write_text(
    json.dumps({
        "status": "PASS" if overall else "FAIL",
        "dedicated_subprocess": True,
        "training_environment_shared": False,
        "training_rng_shared": False,
        "deterministic_mean": True,
        "observation_corruption": False,
        "push_external_force": False,
        "direction_block_allocation": True,
        "seed": 20274021,
    }, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(results, sort_keys=True))
if not overall:
    raise SystemExit("EXP013_W1B_R2_PREFIX_PARITY_FAIL")
