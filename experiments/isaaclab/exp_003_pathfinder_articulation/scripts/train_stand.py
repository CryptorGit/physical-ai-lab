"""Launch the local Pathfinder task through Isaac Lab's RL-Games trainer."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parent.parent
SRC_ROOT = EXP_ROOT / "src"
ISAAC_LAB_ROOT = Path.home() / "workspace" / "IsaacLab"
TRAIN_SCRIPT = ISAAC_LAB_ROOT / "scripts" / "reinforcement_learning" / "rl_games" / "train.py"

if not TRAIN_SCRIPT.is_file():
    raise FileNotFoundError(f"Isaac Lab trainer not found: {TRAIN_SCRIPT}")

sys.path.insert(0, str(SRC_ROOT))

# Register only the Gym task here. The heavy environment module is imported later,
# after SimulationApp has started.
import pathfinder_stand.tasks  # noqa: F401,E402

runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
