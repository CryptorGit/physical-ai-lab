"""Common clean capability evaluator for W1B-R1.

This wrapper intentionally reuses the protected W1B fresh evaluator as the sole
metric/success implementation.  Online guards and formal evaluation call this
same executable in isolated processes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r1_evaluation_parity_corrected_rerun"
SOURCE = HERE.parent / "evaluate_w1b.py"
sys.path.insert(
    0,
    str(
        REPO
        / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"
    ),
)
sys.path.insert(
    0,
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
)

spec = importlib.util.spec_from_file_location("_protected_w1b_fresh_evaluator", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
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

# Enrich the shared episode schema without changing any success calculation.
raw = OUT / f"_raw_{module.a.mode}_{module.a.tag}.json"
if raw.exists():
    payload = json.loads(raw.read_text(encoding="utf-8"))
    for row in payload.get("episode_rows", []):
        row["episode_seed"] = 20274021 + int(row["episode"])
        failures = []
        for key in ("fall", "excessive_tilt", "dangerous_slip", "impact_failure",
                    "long_dwell_saturation"):
            if row.get(key):
                failures.append(key)
        if not row.get("translation_correct", True):
            failures.append("translation")
        if not row.get("yaw_correct", True):
            failures.append("yaw")
        if row.get("turning_gait") in ("UNTRACKED", "FALL"):
            failures.append("gait")
        row["failure_reason"] = "PASS" if row.get("success") else "+".join(failures) or "metric_gate"
        row["gait_cmd"] = 0
    payload["common_evaluator"] = "Exp013DirectionalCapabilityEvaluator"
    payload["clean_contract"] = True
    raw.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
