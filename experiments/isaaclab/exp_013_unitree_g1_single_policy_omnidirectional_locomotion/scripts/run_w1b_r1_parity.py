"""Run and adjudicate the W1B-R1 clean evaluator parity preflight."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r1_evaluation_parity_corrected_rerun"
PARENT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
OLD1 = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk/checkpoints/model_1.pt"
EVALUATOR = HERE.parent / "evaluate_w1b_r1.py"
METRICS = (
    "vector_velocity_mae", "direction_error_deg", "actual_yaw_rate",
    "fall_rate", "dangerous_slip_rate", "impact_failure_rate",
    "long_dwell_saturation_rate",
)


def run(mode: str, checkpoint: Path, tag: str) -> dict:
    target = OUT / f"_raw_{mode}_{tag}.json"
    if not target.exists():
        cmd = [
            sys.executable, str(EVALUATOR), "--mode", mode,
            "--checkpoint", str(checkpoint), "--tag", tag, "--headless",
        ]
        subprocess.run(cmd, cwd=REPO, check=True)
    return json.loads(target.read_text(encoding="utf-8"))


def row_map(payload: dict) -> dict:
    return {row["condition"]: row for row in payload["rows"]}


records = []
summary = {}
for ck_name, checkpoint in (("parent", PARENT), ("old_iteration1", OLD1)):
    for mode, suffix in (("capability", "quick"), ("zero", "formal")):
        payloads = {}
        for path in ("P1_online_common", "P2_fresh_baseline", "P3_fresh_common"):
            tag = f"{path}_{ck_name}_{suffix}"
            payloads[path] = run(mode, checkpoint, tag)
        maps = {key: row_map(value) for key, value in payloads.items()}
        reference = maps["P1_online_common"]
        max_diff = 0.0
        complete = True
        failure_match = True
        pass_counts = {}
        for path, rows in maps.items():
            complete &= set(rows) == set(reference)
            pass_counts[path] = sum(
                bool(row["gate_pass"]) for name, row in rows.items()
                if name.startswith("ZERO_D")
            )
            for name in sorted(set(reference) & set(rows)):
                for metric in METRICS:
                    max_diff = max(max_diff, abs(float(reference[name][metric]) - float(rows[name][metric])))
                failure_match &= (
                    reference[name]["success_rate"] == rows[name]["success_rate"]
                    and reference[name]["gate_pass"] == rows[name]["gate_pass"]
                )
                records.append({
                    "checkpoint": ck_name, "evaluation": suffix, "path": path,
                    "condition": name, "success_rate": rows[name]["success_rate"],
                    "gate_pass": rows[name]["gate_pass"],
                    **{metric: rows[name][metric] for metric in METRICS},
                })
        pass_exact = len(set(pass_counts.values())) == 1
        status = "PASS" if complete and failure_match and pass_exact and max_diff <= 1e-5 else "FAIL"
        summary[f"{ck_name}_{suffix}"] = {
            "status": status,
            "direction_pass_counts": pass_counts,
            "condition_sets_identical": complete,
            "failure_reason_counts_identical": failure_match,
            "maximum_mean_metric_absolute_difference": max_diff,
            "tolerance": 1e-5,
        }

fields = list(records[0])
with (OUT / "evaluation_parity_preflight.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fields)
    writer.writeheader()
    writer.writerows(records)
overall = all(item["status"] == "PASS" for item in summary.values())
(OUT / "evaluation_parity_preflight.json").write_text(
    json.dumps({
        "status": "PASS" if overall else "EXP013_W1B_R1_EVALUATOR_PARITY_FAIL",
        "common_evaluator": "Exp013DirectionalCapabilityEvaluator",
        "paths": {
            "P1": "corrected online guard evaluator in a standalone process",
            "P2": "fresh DirectionalBaseline evaluator via the common source",
            "P3": "second fresh standalone common-evaluator process",
        },
        "results": summary,
    }, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
isolation = json.loads((OUT / "evaluation_process_isolation_audit.json").read_text(encoding="utf-8"))
isolation.update({
    "parity_preflight": "PASS" if overall else "FAIL",
    "training_environment_shared": False,
    "evaluation_subprocess": True,
    "dedicated_seed": 20274021,
})
(OUT / "evaluation_process_isolation_audit.json").write_text(
    json.dumps(isolation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if not overall:
    raise SystemExit("EXP013_W1B_R1_EVALUATOR_PARITY_FAIL")
