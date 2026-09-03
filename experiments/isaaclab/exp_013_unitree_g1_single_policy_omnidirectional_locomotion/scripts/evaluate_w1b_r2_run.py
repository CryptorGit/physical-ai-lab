"""RUN-retention diagnostic using the protected W1A evaluator."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
SOURCE = HERE.parent / "evaluate_w1a.py"
sys.path.insert(
    0,
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
)
sys.path.insert(
    0,
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
)

spec = importlib.util.spec_from_file_location("_protected_w1b_r2_run_evaluator", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.OUT = OUT
module.main()
