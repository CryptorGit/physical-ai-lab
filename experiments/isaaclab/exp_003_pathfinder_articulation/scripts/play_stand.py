"""Play a trained Pathfinder standing policy with RL-Games."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parent.parent
SRC_ROOT = EXP_ROOT / "src"
ISAAC_LAB_ROOT = Path.home() / "workspace" / "IsaacLab"
PLAY_SCRIPT = ISAAC_LAB_ROOT / "scripts" / "reinforcement_learning" / "rl_games" / "play.py"

if not PLAY_SCRIPT.is_file():
    raise FileNotFoundError(f"Isaac Lab play script not found: {PLAY_SCRIPT}")

sys.path.insert(0, str(SRC_ROOT))

import pathfinder_stand.tasks  # noqa: F401,E402

runpy.run_path(str(PLAY_SCRIPT), run_name="__main__")
