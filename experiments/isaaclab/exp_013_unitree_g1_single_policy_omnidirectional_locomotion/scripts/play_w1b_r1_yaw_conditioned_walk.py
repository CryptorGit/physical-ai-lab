"""W1BR1YawConditionedWalk playback using the actor alone in clean capability mode."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
for path in (
    REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src",
    REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src",
):
    sys.path.insert(0, str(path))
source = HERE.parent / "play_w1b_yaw_conditioned_walk.py"
spec = importlib.util.spec_from_file_location("_protected_w1b_playback", source)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print("MODE W1BR1YawConditionedWalk | CLEAN CAPABILITY MODE | external action correction: OFF")
module.main()
