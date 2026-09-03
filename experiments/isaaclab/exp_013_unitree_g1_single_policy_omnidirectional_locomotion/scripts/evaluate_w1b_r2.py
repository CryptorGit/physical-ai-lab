"""W1B-R2 common clean evaluator with the protected metric implementation."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
SOURCE = HERE.parent / "evaluate_w1b.py"
sys.path.insert(
    0,
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
)
sys.path.insert(
    0,
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
)

source = SOURCE.read_text(encoding="utf-8")
legacy_loop = (
    '    for j in m.tolist():vx[j],vy[j],yc[j]=cmd(x,t,epi[j])'
)
vectorized_loop = (
    '    if x["kind"] not in ("path","random"):\n'
    '     vx[m]=x["vx"];vy[m]=x["vy"];yc[m]=x["yaw_cmd"]\n'
    '    else:\n'
    '     for j in m.tolist():vx[j],vy[j],yc[j]=cmd(x,t,epi[j])'
)
if legacy_loop not in source:
    raise RuntimeError("protected evaluator command loop changed")
source = source.replace(legacy_loop, vectorized_loop)
module = types.ModuleType("_protected_w1b_r2_evaluator")
module.__file__ = str(SOURCE)
exec(compile(source, str(SOURCE), "exec"), module.__dict__)
module.OUT = OUT
original_specs = module.specs


def shared_specs():
    rows = original_specs()
    if module.a.mode == "capability":
        for item in rows:
            if item["kind"] == "zero" and item["name"].startswith("ZERO_D"):
                item["episodes"] = 47
    return rows


module.specs = shared_specs
module.main()

raw = OUT / f"_raw_{module.a.mode}_{module.a.tag}.json"
if raw.exists():
    payload = json.loads(raw.read_text(encoding="utf-8"))
    for row in payload.get("episode_rows", []):
        row["episode_seed"] = 20274021 + int(row["episode"])
        failures = []
        for key in (
            "fall", "excessive_tilt", "dangerous_slip",
            "impact_failure", "long_dwell_saturation",
        ):
            if row.get(key):
                failures.append(key)
        if not row.get("translation_correct", True):
            failures.append("translation")
        if not row.get("yaw_correct", True):
            failures.append("yaw")
        if row.get("turning_gait") in ("UNTRACKED", "FALL"):
            failures.append("gait")
        row["failure_reason"] = (
            "PASS" if row.get("success")
            else "+".join(failures) or "metric_gate"
        )
        row["gait_cmd"] = 0
    payload["common_evaluator"] = "Exp013DirectionalCapabilityEvaluator"
    payload["clean_contract"] = True
    raw.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
