"""Run the read-only W1B-R1 evaluation suite on the best available checkpoint."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r1_evaluation_parity_corrected_rerun"
CHECKPOINT = OUT / "checkpoints/model_10.pt"
EVALUATOR = HERE.parent / "evaluate_w1b_r1.py"

for mode in ("zero", "pure", "moving", "independence", "envelope", "path", "random"):
    target = OUT / f"_raw_{mode}_selected.json"
    if target.exists():
        continue
    subprocess.run([
        sys.executable, str(EVALUATOR), "--mode", mode,
        "--checkpoint", str(CHECKPOINT), "--tag", "selected", "--headless",
    ], cwd=REPO, check=True)
