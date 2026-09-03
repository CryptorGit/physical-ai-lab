"""Fresh-process Stage-2H collector using the frozen Stage-2G collection contract."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2h_short_horizon_completion_replay_preflight"

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--branch", required=True)
pre.add_argument("--shadow-iteration", type=int, required=True)
pre.add_argument("--checkpoint", required=True)
pre.add_argument("--diagnostic-seed", type=int, required=True)
known, remaining = pre.parse_known_args()

# The validated Stage-2G collector owns Isaac launcher parsing and telemetry semantics.
sys.path.insert(0, str(SCRIPT.parent))
sys.argv = [
    sys.argv[0],
    "--batch-index",
    str(known.shadow_iteration),
    "--seed-root",
    str(known.diagnostic_seed - known.shadow_iteration),
    *remaining,
]
import collect_stage2g_on_policy as base  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


branch_raw = OUT / "raw" / known.branch
branch_raw.mkdir(parents=True, exist_ok=True)
checkpoint = Path(known.checkpoint).resolve()
base.OUT = branch_raw
base.RAW = branch_raw
base.CHECKPOINT = checkpoint
base.EXPECTED_SHA = sha256(checkpoint)

if __name__ == "__main__":
    base.main()
