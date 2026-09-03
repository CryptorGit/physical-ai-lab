from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Gymへの独自タスク登録を実行する
import peg_observation_ablation  # noqa: F401


ISAACLAB_ROOT = Path.home() / "workspace" / "IsaacLab"
TRAIN_SCRIPT = (
    ISAACLAB_ROOT
    / "scripts"
    / "reinforcement_learning"
    / "rl_games"
    / "train.py"
)

if not TRAIN_SCRIPT.exists():
    raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

sys.argv[0] = str(TRAIN_SCRIPT)
runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
