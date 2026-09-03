"""Run the no-state-copy counterfactual mode of the Stage 0 replay."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

target = Path(__file__).with_name("build_observability_dataset.py")
sys.argv = [str(target), "--counterfactual-only", *sys.argv[1:]]
runpy.run_path(str(target), run_name="__main__")
