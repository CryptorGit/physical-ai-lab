"""Evaluate one disposable Stage-2H actor through the validated Stage-2G evaluator."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2h_short_horizon_completion_replay_preflight"

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--branch", required=True)
pre.add_argument("--shadow-iteration", type=int, required=True)
pre.add_argument("--diagnostic-seed", type=int, required=True)
known, remaining = pre.parse_known_args()
branch_dir = OUT / "raw" / known.branch
prefix = f"eval_{known.shadow_iteration}_"
shutil.copyfile(
    branch_dir / f"shadow_{known.shadow_iteration}.pt",
    branch_dir / f"{prefix}shadow_{known.branch}.pt",
)

sys.path.insert(0, str(SCRIPT.parent))
sys.argv = [
    sys.argv[0],
    "--seed-root",
    str(known.diagnostic_seed),
    "--conditions",
    known.branch,
    "--output-prefix",
    prefix,
    *remaining,
]
import evaluate_stage2g_shadows as base  # noqa: E402

base.OUT = branch_dir
base.RAW = branch_dir
base.args.conditions = [known.branch]
base.args.output_prefix = prefix

if __name__ == "__main__":
    base.main()
