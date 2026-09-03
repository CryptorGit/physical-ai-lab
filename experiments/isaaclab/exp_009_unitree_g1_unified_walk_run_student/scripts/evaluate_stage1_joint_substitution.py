"""Isaac closed-loop joint-group substitution audit on WALK occupancy."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
STAGE0 = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis/joint_substitution_runtime"
SOURCE = EXP / "scripts/evaluate_unified_student.py"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.student_actor import UnifiedWalkRunStudent123

groups = {
    "hip_pitch": [0, 1],
    "knee": [11, 12],
    "ankle_pitch": [15, 16],
    "ankle_roll": [19, 20],
}
student_state = torch.load(STAGE0 / "checkpoints/epoch_10.pt", map_location="cpu", weights_only=False)["student"]
teacher_state = torch.load(STAGE0 / "checkpoints/initial.pt", map_location="cpu", weights_only=False)["student"]


class SubstitutionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = UnifiedWalkRunStudent123()
        self.teacher = UnifiedWalkRunStudent123()
        self.register_buffer("mask", torch.zeros(37))
        self.register_buffer("teacher_base", torch.tensor(0.0))

    def forward(self, observation):
        student, teacher = self.student(observation), self.teacher(observation)
        base = torch.where(self.teacher_base.bool(), teacher, student)
        replacement = torch.where(self.teacher_base.bool(), student, teacher)
        return torch.where(self.mask.bool(), replacement, base)


OUT.mkdir(parents=True, exist_ok=True)
checkpoints = []
variants = [("student_baseline", [], False), ("teacher_baseline", [], True)]
for name, indices in groups.items():
    variants += [(f"student_plus_teacher_{name}", indices, False), (f"teacher_plus_student_{name}", indices, True)]
for name, indices, teacher_base in variants:
    policy = SubstitutionPolicy()
    policy.student.load_state_dict(student_state)
    policy.teacher.load_state_dict(teacher_state)
    policy.mask[indices] = 1
    policy.teacher_base.fill_(float(teacher_base))
    path = OUT / f"{name}.pt"
    torch.save({"student": policy.state_dict(), "epoch": 10, "variant": name}, path)
    checkpoints.append(path)

runtime_cfg = yaml.safe_load((EXP / "configs/stage0_multiteacher_distillation.yaml").read_text(encoding="utf-8"))
runtime_cfg["evaluation"].update({"run_speeds_mps": [], "forward_targets_mps": [], "intermediate_speeds_mps": [], "reverse_sources_mps": []})
cfg_path = OUT / "runtime_eval_config.yaml"
cfg_path.write_text(yaml.safe_dump(runtime_cfg, sort_keys=False), encoding="utf-8")
sys.argv = [str(SOURCE), *sum((["--checkpoint", str(path)] for path in checkpoints), []), "--headless"]
spec = importlib.util.spec_from_file_location("exp009_stage0_joint_runtime", SOURCE)
module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
module.OUT = OUT
module.CFG_PATH = cfg_path
module.UnifiedWalkRunStudent123 = SubstitutionPolicy
module.main()

