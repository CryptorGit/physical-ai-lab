"""Run the frozen Stage-0 Isaac evaluator against Stage-2 WALK-only checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation/walk_only_runtime"
BASE = EXP / "scripts/evaluate_unified_student.py"
BASE_CFG = EXP / "configs/stage0_multiteacher_distillation.yaml"

OUT.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
cfg["experiment"]["name"] = "exp_009_stage2_walk_only_dynamics_sensitive_evaluation"
cfg["evaluation"].update({
    "evaluation_seed": 20270321,
    "physical_envs": 512,
    "episodes_per_supported_condition": 50,
    "walk_speeds_mps": [0.6, 0.8, 1.0, 1.2],
    "run_speeds_mps": [],
    "forward_targets_mps": [],
    "intermediate_speeds_mps": [],
    "reverse_sources_mps": [],
})
cfg["runtime"].update({
    "ppo_training": False, "reward_optimization": False, "teacher_updates": False,
    "production_capability_update": False, "runtime_controller_switch": False,
})
runtime_cfg = OUT / "runtime_eval_config.yaml"
runtime_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

source = BASE.read_text(encoding="utf-8")
old_out = 'OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"'
old_cfg = 'CFG_PATH = EXP / "configs/stage0_multiteacher_distillation.yaml"'
if old_out not in source or old_cfg not in source:
    raise RuntimeError("frozen evaluator source contract changed")
source = source.replace(old_out, f'OUT = Path(r"{OUT}")', 1)
source = source.replace(old_cfg, f'CFG_PATH = Path(r"{runtime_cfg}")', 1)
full_suite = '''        for speed in [0.6, 0.8, 1.0, 1.2]:
            run_condition("walk", "walk", speed, speed, "walk", 50)
        for speed in [2.4, 2.6, 2.8]:
            run_condition("run", "run", speed, speed, "run", 50)
        for target in [2.6, 2.8]:
            run_condition("walk_to_run", "walk", 1.2, target, "wtr", 50)
        if not args.collect_dagger:
            for source in [2.6, 2.8]:
                run_condition("run_to_walk", "run", source, 1.2, "reverse", 50)
            for target in [1.4, 1.6, 1.8, 2.0, 2.2]:
                run_condition("intermediate", "walk", 1.2, target, "intermediate", 20)
'''
walk_suite = '''        for speed in [0.6, 0.8, 1.0, 1.2]:
            run_condition("walk", "walk", speed, speed, "walk", 50)
'''
if full_suite not in source:
    raise RuntimeError("frozen evaluator condition block changed")
source = source.replace(full_suite, walk_suite, 1)
namespace = {"__file__": str(BASE), "__name__": "__main__"}
exec(compile(source, str(BASE), "exec"), namespace)
