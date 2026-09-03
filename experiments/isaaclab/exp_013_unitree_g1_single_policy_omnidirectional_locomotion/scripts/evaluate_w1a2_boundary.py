"""Run W1A2 boundary conditions through the frozen W1A evaluator."""
import json, sys
from pathlib import Path
HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
PARENT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk/checkpoints/model_120.pt"
sys.argv=["evaluate_w1a.py","--suite","formal","--checkpoint",str(PARENT),"--tag","w1a2_boundary","--headless"]
sys.path.insert(0,str(HERE.parent))
import evaluate_w1a as base
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion"
manifest=json.loads((OUT/"w1a_failed_0p6_sector_manifest.json").read_text(encoding="utf-8"))
def boundary_conditions():
 return [base.static(f"S{s:.2f}_D{d:05.1f}",s,d,20) for d in [r["angle"] for r in manifest["sectors"]] for s in (.30,.40,.45,.50,.55,.60)]
base.OUT=OUT; base.conditions=boundary_conditions
base.main()
