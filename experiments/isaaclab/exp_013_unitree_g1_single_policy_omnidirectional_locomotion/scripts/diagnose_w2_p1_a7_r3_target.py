"""Run preregistered R3-B target/mirror diagnosis and emit its authorization decision."""
from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
R2 = BASE / "phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2"
OUT = BASE / "phase_w2_p1_a7_r3_start_retention_recovery"
RAW = OUT / "raw/target_diagnosis"
EVALUATOR = HERE.parent / "evaluate_w2_p1_a7_r3.py"
ISAAC = Path.home() / "workspace/IsaacLab/isaaclab.bat"
POLICY = R2 / "checkpoints/model_075.pt"
CONDITIONS = [
    ("T0", 315.0, 0.3), ("M0", 45.0, -0.3),
    ("C315_ZERO", 315.0, 0.0), ("C315_NEG", 315.0, -0.3),
    ("C45_ZERO", 45.0, 0.0), ("C45_POS", 45.0, 0.3),
    ("REAR_NEG", 180.0, -0.3), ("REAR_POS", 180.0, 0.3),
    ("FRONT_NEG", 0.0, -0.3), ("FRONT_POS", 0.0, 0.3),
]


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows: rows = [{"status": "NO_ROWS"}]
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    summaries, attribution_rows, gait_rows, traces = [], [], [], {}
    for name, direction, yaw in CONDITIONS:
        output = RAW / f"{name}.csv"; trace = RAW / f"{name}_trace.csv"
        if not output.with_suffix(".json").exists():
            command = [str(ISAAC), "-p", str(EVALUATOR), "--policy", str(POLICY), "--batch", "4", "--split", "validation", "--direction", str(direction), "--speed", "0.3", "--yaw", str(yaw), "--episodes", "300", "--group", name, "--output", str(output), "--diagnostic-output", str(trace), "--headless", "--device", "cuda:0"]
            with output.with_suffix(".log").open("w", encoding="utf-8") as log: subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8")); row = payload["row"]; row["condition_name"] = name; summaries.append(row)
        for episode in payload["episode_attribution"]: attribution_rows.append({"condition_name": name, "direction": direction, "yaw": yaw, **episode})
        diagnostic = json.loads(trace.with_suffix(".json").read_text(encoding="utf-8"))
        for phase in diagnostic["phase_rows"]: gait_rows.append({"condition_name": name, "direction": direction, "yaw": yaw, "dominant_oscillation_frequency_hz": diagnostic["dominant_oscillation_frequency_hz"], "stride_frequency_hz": diagnostic["stride_frequency_hz"], "phase_locking_strength": diagnostic["phase_locking_strength"], **phase})
        traces[name] = list(csv.DictReader(trace.open(encoding="utf-8")))
        print(json.dumps({"condition": name, "status": "COMPLETE"}), flush=True)

    write_csv("target_pair_component_diagnosis.csv", summaries)
    write_json("target_pair_component_diagnosis.json", {"split": "validation", "episodes_per_condition": 300, "rows": summaries})
    write_csv("target_pair_failure_attribution.csv", attribution_rows)
    counts = {}
    for row in attribution_rows: counts.setdefault(row["condition_name"], {}).setdefault(row["category"], 0); counts[row["condition_name"]][row["category"]] += 1
    write_json("target_pair_failure_attribution.json", {"rows": attribution_rows, "counts": counts})
    write_csv("target_pair_gait_phase_analysis.csv", gait_rows)
    write_json("target_pair_gait_phase_analysis.json", {"rows": gait_rows})

    action_rows = []
    target_trace, mirror_trace = traces["T0"], traces["M0"]
    mirror_contract = json.loads((BASE / "phase_w1b_d1_yaw_translation_interference_diagnosis/robot_mirror_contract.json").read_text(encoding="utf-8"))
    mirror_indices = mirror_contract["mirror_indices"]
    mirror_signs = mirror_contract["mirror_signs"]
    groups = {"legs": range(0, 18), "waist": range(18, 21), "torso_arms": range(21, 33), "hands": range(33, 37)}
    for step, (target_row, mirror_row) in enumerate(zip(target_trace, mirror_trace)):
        target_action = [float(target_row[f"action_{joint:02d}"]) for joint in range(37)]
        raw_mirror_action = [float(mirror_row[f"action_{joint:02d}"]) for joint in range(37)]
        mirror_action = [raw_mirror_action[index] * sign for index, sign in zip(mirror_indices, mirror_signs)]
        difference = [left - right for left, right in zip(target_action, mirror_action)]
        target_norm = math.sqrt(sum(value * value for value in target_action)); mirror_norm = math.sqrt(sum(value * value for value in mirror_action))
        cosine = sum(left * right for left, right in zip(target_action, mirror_action)) / max(target_norm * mirror_norm, 1e-12)
        total = sum(value * value for value in difference) or 1.0
        action_rows.append({"control_step": step, "whole_body_action_l2": math.sqrt(total), "action_cosine": cosine, **{f"{name}_contribution": sum(difference[index] ** 2 for index in indices) / total for name, indices in groups.items()}})
    write_csv("target_mirror_action_audit.csv", action_rows)
    write_json("target_mirror_action_audit.json", {"comparison": "T0 action versus symmetry-transformed M0 action", "mirror_contract": "phase_w1b_d1 robot_mirror_contract.json", "rows": action_rows, "mean": {key: sum(float(row[key]) for row in action_rows) / len(action_rows) for key in action_rows[0] if key != "control_step"}})

    rescue = json.loads((OUT / "existing_checkpoint_rescue_timeline.json").read_text(encoding="utf-8"))["rows"]
    trajectory = [row for row in rescue if int(row["update"]) in (20, 45, 75, 100, 120, 150) and ((row["direction"] == 315.0 and row["yaw"] == 0.3) or (row["direction"] == 45.0 and row["yaw"] == -0.3))]
    write_csv("target_checkpoint_trajectory.csv", trajectory)
    write_json("target_checkpoint_trajectory.json", {"rows": trajectory})

    target = next(row for row in summaries if row["condition_name"] == "T0")
    mirror = next(row for row in summaries if row["condition_name"] == "M0")
    rear = [row for row in summaries if row["condition_name"] in ("REAR_NEG", "REAR_POS")]
    target_categories = counts["T0"]
    primary = "YAW_RATE_OSCILLATION" if target_categories.get("YAW_RATE_OSCILLATION", 0) >= max(target_categories.get("MULTIPLE_COMPONENTS", 0), target_categories.get("TRANSLATION_VECTOR_LIMIT", 0), target_categories.get("DIRECTION_LIMIT", 0), target_categories.get("GAIT_LIMIT", 0)) else "INCONCLUSIVE"
    authorized = (
        target["endpoint_success"] >= 0.95 and target["fall_rate"] <= 0.02 and target["acquisition_0p10"] >= 0.95
        and primary in ("YAW_RATE_OSCILLATION", "SUSTAINED_WINDOW_ONLY")
        and target["translation_sustained_0p20"] >= 0.95 and target["direction_sustained_0p20"] >= 0.95 and target["gait_safety_sustained_0p20"] >= 0.95
        and mirror["endpoint_success"] >= 0.95 and mirror["acquisition_0p20"] >= 0.85
        and all(row["endpoint_success"] >= 0.95 and row["acquisition_0p20"] >= 0.90 for row in rear)
    )
    write_json("target_pair_component_diagnosis.json", {"split": "validation", "episodes_per_condition": 300, "primary_limiter": primary, "rows": summaries})
    write_json("r3_training_authorization.json", {"status": "AUTHORIZED" if authorized else "DENIED", "existing_checkpoint_rescue": "FAIL", "primary_limiter": primary, "target": target, "mirror": mirror, "rear": rear})


if __name__ == "__main__":
    main()
