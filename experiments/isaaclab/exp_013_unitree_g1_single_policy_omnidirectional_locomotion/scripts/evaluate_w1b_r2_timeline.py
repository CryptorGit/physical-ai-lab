"""Evaluate the capability and moving-turn timeline in one clean process."""
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
module = types.ModuleType("_w1b_r2_timeline")
module.__file__ = str(SOURCE)
exec(compile(source, str(SOURCE), "exec"), module.__dict__)
module.OUT = OUT
original_specs = module.specs


def timeline_specs():
    saved = module.a.mode
    module.a.mode = "capability"
    capability = original_specs()
    module.a.mode = "moving"
    moving = original_specs()
    module.a.mode = saved
    for item in capability + moving:
        item["episodes"] = 20
    return capability + moving


module.specs = timeline_specs
module.main()

raw = OUT / f"_raw_capability_{module.a.tag}.json"
if raw.exists():
    payload = json.loads(raw.read_text(encoding="utf-8"))
    payload["common_evaluator"] = "Exp013DirectionalCapabilityEvaluator"
    payload["clean_contract"] = True
    payload["timeline_union"] = ["capability", "moving"]
    raw.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
