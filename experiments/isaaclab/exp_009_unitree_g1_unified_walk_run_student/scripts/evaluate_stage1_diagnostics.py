"""Closed-loop Stage 1 diagnostic evaluator using the frozen Stage-0 physics path.

This wrapper intentionally imports (rather than edits) the Stage-0 evaluator,
redirects every output to the Stage-1 directory, and substitutes only the
diagnostic network constructor. It performs no PPO or reward optimization.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
STAGE1 = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis"
SOURCE = EXP / "scripts/evaluate_unified_student.py"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.stage1_models import DiagnosticSingleHead

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--diagnostic-checkpoint", required=True)
parser.add_argument("--scope", choices=["walk", "run", "walk_to_run", "all"], default="all")
known, passthrough = parser.parse_known_args()
checkpoint = Path(known.diagnostic_checkpoint).resolve()
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
if payload.get("multihead"):
    raise RuntimeError("oracle multi-head requires the dedicated offline upper-bound audit")

input_dim = int(payload["input_dim"])
if input_dim != 123:
    raise RuntimeError("closed-loop wrapper supports canonical 123D models; context upper bounds are offline diagnostics")
hidden = list(payload["hidden"])
repacked = STAGE1 / "eval_checkpoints" / checkpoint.name
repacked.parent.mkdir(parents=True, exist_ok=True)
torch.save({"student": payload["model"], "epoch": 10, "diagnostic_source": str(checkpoint)}, repacked)

# The imported module owns Isaac Lab CLI parsing. Feed it one checkpoint plus
# the caller's launcher arguments, then redirect output before calling main.
sys.argv = [str(SOURCE), "--checkpoint", str(repacked), *passthrough]
spec = importlib.util.spec_from_file_location("exp009_stage0_eval_runtime", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.OUT = STAGE1
stage0_cfg_path = EXP / "configs/stage0_multiteacher_distillation.yaml"
runtime_cfg = yaml.safe_load(stage0_cfg_path.read_text(encoding="utf-8"))
scope = known.scope
if scope == "walk":
    runtime_cfg["evaluation"].update({"run_speeds_mps": [], "forward_targets_mps": [], "intermediate_speeds_mps": [], "reverse_sources_mps": []})
elif scope == "run":
    runtime_cfg["evaluation"].update({"walk_speeds_mps": [], "forward_targets_mps": [], "intermediate_speeds_mps": [], "reverse_sources_mps": []})
elif scope == "walk_to_run":
    runtime_cfg["evaluation"].update({"walk_speeds_mps": [], "run_speeds_mps": [], "intermediate_speeds_mps": [], "reverse_sources_mps": []})
runtime_cfg_path = STAGE1 / "runtime_eval_config.yaml"
runtime_cfg_path.write_text(yaml.safe_dump(runtime_cfg, sort_keys=False), encoding="utf-8")
module.CFG_PATH = runtime_cfg_path


class RuntimeDiagnostic123(DiagnosticSingleHead):
    def __init__(self):
        super().__init__(123, hidden)


module.UnifiedWalkRunStudent123 = RuntimeDiagnostic123
module.main()
