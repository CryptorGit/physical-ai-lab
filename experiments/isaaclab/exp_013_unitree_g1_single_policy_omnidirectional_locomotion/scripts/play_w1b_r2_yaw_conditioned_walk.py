"""W1BR2YawConditionedWalk playback using one actor without action correction."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
for path in (
    REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src",
    REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src",
):
    sys.path.insert(0, str(path))
out = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
selected = json.loads((out / "selected_checkpoint.json").read_text(encoding="utf-8"))
source = HERE.parent / "play_w1b_yaw_conditioned_walk.py"
spec = importlib.util.spec_from_file_location("_protected_w1b_r2_playback", source)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(
    "MODE W1BR2YawConditionedWalk | gait=0 | external action correction: OFF\n"
    f"CHECKPOINT SHA {selected['sha256']}\n"
    "PENDING MIRROR QUEUE LENGTH / PAIR BALANCE / RESET EVENT COUNT are "
    "read-only sampler diagnostics; runtime joint actions come only from the actor."
)
module.main()
