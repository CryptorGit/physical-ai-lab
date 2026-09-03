"""Build the W1A4 capability timeline and select only among 16/16 low-speed checkpoints."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation"
LABELS = ("initial", "1", "5", "10", "20", "30", "40", "50", "60")


def read(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_write(name: str, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    return hashlib.sha256(b"".join(value.cpu().numpy().tobytes() for key, value in sorted(state.items()))).hexdigest()


timeline: list[dict] = []
ranks: list[dict] = []
manifest: list[dict] = []
for label in LABELS:
    iteration = 0 if label == "initial" else int(label)
    raw = read(f"_raw_formal_capability_{label}.json")
    rows = raw["rows"]
    timeline.extend({"checkpoint_iteration": iteration, **row} for row in rows)
    low = [row for row in rows if row["condition"].startswith("S0.30_")]
    fast = [row for row in rows if row["condition"].startswith("S0.60_")]
    by_name = {row["condition"]: row for row in rows}
    by_direction_low = {row["direction_deg"]: row for row in low}
    mirror = [
        abs(by_direction_low[d]["vector_velocity_mae"] - by_direction_low[360.0 - d]["vector_velocity_mae"])
        for d in (22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)
    ]
    row225 = by_direction_low[225.0]
    row247 = by_direction_low[247.5]
    ranks.append(
        {
            "iteration": iteration,
            "pass_0p3": sum(row["gate_pass"] for row in low),
            "success_225_0p3": row225["success_rate"],
            "success_247p5_0p3": row247["success_rate"],
            "rear_left_direction_error_deg": (row225["direction_error_deg"] + row247["direction_error_deg"]) / 2,
            "pass_0p6": sum(row["gate_pass"] for row in fast),
            "forward_0p6_success": by_name["FWD_0P6"]["success_rate"],
            "forward_1p2_success": by_name["FWD_1P2"]["success_rate"],
            "front_left_1p0_success": by_name["FL_1P0"]["success_rate"],
            "front_right_1p0_success": by_name["FR_1P0"]["success_rate"],
            "fall_rate": sum(row["fall_rate"] for row in rows) / len(rows),
            "dangerous_slip_rate": sum(row["dangerous_slip_rate"] for row in rows) / len(rows),
            "vector_velocity_mae": sum(row["vector_velocity_mae"] for row in rows) / len(rows),
            "mirror_mae_difference": sum(mirror) / len(mirror),
            "impact_failure_rate": sum(row["impact_failure_rate"] for row in rows) / len(rows),
        }
    )
    checkpoint = OUT / "checkpoints" / f"model_{label}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    buffer = io.BytesIO()
    torch.save(payload["optimizer_state_dict"], buffer)
    manifest.append(
        {
            "iteration": iteration,
            "path": str(checkpoint),
            "sha256": file_sha(checkpoint),
            "actor_hash": state_hash(payload["actor_state_dict"]),
            "critic_hash": state_hash(payload["critic_state_dict"]),
            "optimizer_hash": hashlib.sha256(buffer.getvalue()).hexdigest(),
            "learning_rate": payload.get("infos", {}).get("learning_rate"),
            "beta": payload.get("infos", {}).get("beta"),
            "low_speed_holdout_kl": payload.get("infos", {}).get("low_speed_holdout_kl"),
            "rollout_kl": payload.get("infos", {}).get("rollout_kl"),
            "clip_fraction": payload.get("infos", {}).get("clip_fraction"),
        }
    )

eligible = [row for row in ranks if row["pass_0p3"] == 16]
if not eligible:
    raise RuntimeError("No checkpoint satisfies mandatory 0.3 m/s 16/16 selection constraint")
eligible.sort(
    key=lambda row: (
        -row["success_225_0p3"],
        -row["success_247p5_0p3"],
        row["rear_left_direction_error_deg"],
        -row["pass_0p6"],
        -min(row["forward_0p6_success"], row["forward_1p2_success"]),
        row["fall_rate"],
        row["dangerous_slip_rate"],
        row["vector_velocity_mae"],
        row["mirror_mae_difference"],
    )
)
selected = eligible[0]
selected_label = "initial" if selected["iteration"] == 0 else str(selected["iteration"])
selected_path = OUT / "checkpoints" / f"model_{selected_label}.pt"
csv_write("capability_timeline.csv", timeline)
write("checkpoint_manifest.json", {"entries": manifest})
write(
    "selected_checkpoint.json",
    {
        **selected,
        "path": str(selected_path),
        "sha256": file_sha(selected_path),
        "selection_rule": "0.3m/s 16/16 mandatory, then rear-left margins before 0.6m/s count",
        "latest_auto_selected": False,
        "eligible_ranked_candidates": eligible,
        "ineligible_candidates": [row for row in ranks if row["pass_0p3"] < 16],
    },
)
print(json.dumps({"selected_iteration": selected["iteration"], "sha256": file_sha(selected_path), **selected}, indent=2))
