"""Oracle-regime closed-loop upper bound for the diagnostic multi-head model."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis/multihead_runtime"
SOURCE = EXP / "scripts/evaluate_unified_student.py"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.stage1_models import DiagnosticMultiHead

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--scope", choices=["walk", "run", "walk_to_run"], required=True)
known, passthrough = parser.parse_known_args()
head = {"walk": 0, "run": 1, "walk_to_run": 2}[known.scope]
source = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis/checkpoints/diagnostic_multihead.pt"
payload = torch.load(source, map_location="cpu", weights_only=False)


class FixedOracleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DiagnosticMultiHead()
        self.register_buffer("oracle_head", torch.tensor(head, dtype=torch.long))

    def forward(self, observation):
        regime = self.oracle_head.expand(len(observation))
        return self.model(observation, regime)


policy = FixedOracleHead()
policy.model.load_state_dict(payload["model"])
OUT.mkdir(parents=True, exist_ok=True)
checkpoint = OUT / f"oracle_{known.scope}.pt"
torch.save({"student": policy.state_dict(), "epoch": 10, "oracle_scope": known.scope}, checkpoint)
cfg = yaml.safe_load((EXP / "configs/stage0_multiteacher_distillation.yaml").read_text(encoding="utf-8"))
if known.scope == "walk":
    cfg["evaluation"].update({"run_speeds_mps": [], "forward_targets_mps": [], "intermediate_speeds_mps": [], "reverse_sources_mps": []})
elif known.scope == "run":
    cfg["evaluation"].update({"walk_speeds_mps": [], "forward_targets_mps": [], "intermediate_speeds_mps": [], "reverse_sources_mps": []})
else:
    cfg["evaluation"].update({"walk_speeds_mps": [], "run_speeds_mps": [], "intermediate_speeds_mps": [], "reverse_sources_mps": []})
cfg_path = OUT / f"runtime_{known.scope}.yaml"
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
sys.argv = [str(SOURCE), "--checkpoint", str(checkpoint), "--append", *passthrough]
spec = importlib.util.spec_from_file_location("exp009_stage0_multihead_runtime", SOURCE)
module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
module.OUT = OUT; module.CFG_PATH = cfg_path; module.UnifiedWalkRunStudent123 = FixedOracleHead
module.main()

