"""Freeze the fresh iteration-80 0.6 m/s failed-sector set before W1A4."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation"


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


raw = json.loads((OUT / "_raw_formal_parent80_0p6.json").read_text(encoding="utf-8"))
rows = sorted(raw["rows"], key=lambda row: row["direction_deg"])
failed = [float(row["direction_deg"]) for row in rows if not row["gate_pass"]]
write(
    "iteration80_failed_0p6_sector_manifest.json",
    {
        "checkpoint_sha256": raw["checkpoint_sha256"],
        "episodes_per_direction": 30,
        "fixed_before_training": True,
        "count": len(failed),
        "failed_directions_deg": failed,
        "conditions": rows,
        "source": "_raw_formal_parent80_0p6.json",
    },
)
curriculum = json.loads((OUT / "resolved_w1a4_curriculum.json").read_text(encoding="utf-8"))
curriculum["groups"]["C"]["failed_directions_deg"] = failed
curriculum["groups"]["C"]["source"] = "fresh 30-episode iteration-80 audit"
write("resolved_w1a4_curriculum.json", curriculum)
print(f"Frozen {len(failed)} failed sectors: {failed}")
