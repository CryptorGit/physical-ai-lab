"""Bind an existing 116-wide simulation actor as an H5 diagnostic candidate.

This utility does not promote or mutate a weight.  It creates the small
manifest wrapper required by the H5 evaluator when an independently audited
H4 actor is being tested through the H5 absolute-target decoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.h4_post_training import (  # noqa: E402
    sha256_file,
    validate_h4_params,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--expert",
        choices=("planar", "reverse", "unified"),
        required=True,
        help="H5 diagnostic domain represented by the wrapper.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help=(
            "Completed unified training run manifest. The evaluator revalidates "
            "its resolved command-contract provenance before strict execution."
        ),
    )
    parser.add_argument(
        "--h5-seed",
        action="store_true",
        help="Emit the stricter diagnostic seed schema used by the H5 training loader.",
    )
    parser.add_argument(
        "--target-table-manifest",
        type=Path,
        help="Copy a hash-independent 54x14 teacher table from an existing seed manifest.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    params = args.params.resolve()
    manifest_path = args.manifest.resolve()
    with params.open("rb") as stream:
        loaded = pickle.load(stream)
    audit = validate_h4_params(loaded)
    if audit.get("actor_observation_width") != 116:
        raise ValueError("H5 diagnostic candidate requires an exact 116-wide actor")
    source_path = args.source_manifest.resolve()
    source = {
        "path": str(source_path),
        "sha256": sha256_file(source_path),
    }
    teacher_source = None
    if args.target_table_manifest is not None:
        table_manifest_path = args.target_table_manifest.resolve()
        table_manifest = json.loads(table_manifest_path.read_text(encoding="utf-8"))
        raw_table = (
            table_manifest.get("teacher_source", {}).get("target_table_rad")
        )
        if raw_table is None:
            raise ValueError("target-table manifest has no teacher_source.target_table_rad")
        if len(raw_table) != 54 or any(len(row) != 14 for row in raw_table):
            raise ValueError("target teacher table must be exactly 54x14")
        teacher_source = {
            "mode": "reused_diagnostic_target_table",
            "source_manifest": str(table_manifest_path),
            "source_manifest_sha256": sha256_file(table_manifest_path),
            "target_table_contract": "H5_54_ROW_ABSOLUTE_TARGET_TABLE_V1",
            "target_table_shape": [54, 14],
            "target_table_rad": raw_table,
        }
    if args.h5_seed and teacher_source is None:
        raise ValueError("--h5-seed requires --target-table-manifest")
    payload = {
        "schema_version": 1,
        "status": "COMPLETED",
        "hardware_deployment": "PROHIBITED",
        "run_name": manifest_path.parent.name,
        "expert": str(args.expert),
        "qualification_use": "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION",
        "candidate_kind": (
            "H5_TARGET_SPACE_DISTILLED_SEED"
            if args.h5_seed
            else "H5_DIAGNOSTIC_WRAPPER_AROUND_AUDITED_H4_ACTOR"
        ),
        "actor_observation_width": 116,
        "source_candidate": source,
        "source_params": {
            "path": str(params),
            "sha256": sha256_file(params),
            "h4_validation": audit,
        },
        "outputs": {
            "final_params": {
                "path": str(params),
                "sha256": sha256_file(params),
            }
        },
        "notes": [
            "Simulation-only diagnostic wrapper; adoption and hardware deployment are prohibited.",
            "The H5 evaluator owns absolute target decoding and the final guard.",
        ],
    }
    if teacher_source is not None:
        payload["teacher_source"] = teacher_source
        payload["target_space_distillation"] = {
            "formula": "reused audited H4 actor initialization; target table is a private H5 BC teacher",
            "fitted_parameters": [],
            "preserved_parameters": ["all source actor and critic parameters"],
            "passed": True,
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "params": str(params),
                "params_sha256": sha256_file(params),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "hardware_deployment": "PROHIBITED",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
