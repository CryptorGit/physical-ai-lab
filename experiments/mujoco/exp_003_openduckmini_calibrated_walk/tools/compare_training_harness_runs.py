#!/usr/bin/env python3
"""Compares two exact-resume checkpoints and emits leaf-level CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from training.checkpointing import load_checkpoint, tree_leaf_digest  # noqa: E402
from training.comparison import compare_trees  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    reference, reference_model, reference_manifest = load_checkpoint(args.reference)
    comparison, comparison_model, comparison_manifest = load_checkpoint(
        args.comparison
    )
    rows = compare_trees(reference, comparison)
    model_rows = compare_trees(reference_model, comparison_model)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "bit_exact": all(row["bit_exact"] for row in rows + model_rows),
        "first_divergence": next(
            (row for row in rows + model_rows if not row["bit_exact"]), None
        ),
        "state_leaf_count": len(rows),
        "model_leaf_count": len(model_rows),
        "reference_state_hash": tree_leaf_digest(reference),
        "comparison_state_hash": tree_leaf_digest(comparison),
        "reference_payload_hash": reference_manifest["payload_sha256"],
        "comparison_payload_hash": comparison_manifest["payload_sha256"],
    }
    args.json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

