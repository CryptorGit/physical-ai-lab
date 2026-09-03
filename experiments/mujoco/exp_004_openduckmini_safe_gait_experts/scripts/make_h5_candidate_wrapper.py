"""Create a reproducible diagnostic manifest for one H5 candidate domain.

The underlying parameter file is not copied or modified.  The wrapper binds
the exact parameter SHA and source distillation manifest to the domain name
expected by the strict evaluator.  It is simulation-only and never authorizes
hardware deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.h4_post_training import sha256_file  # noqa: E402


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", choices=("planar", "reverse"), required=True)
    parser.add_argument("--params", type=_path, required=True)
    parser.add_argument("--source-manifest", type=_path, required=True)
    parser.add_argument("--output-manifest", type=_path, required=True)
    parser.add_argument("--run-name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    params = args.params.resolve()
    source_manifest = args.source_manifest.resolve()
    output_manifest = args.output_manifest.resolve()
    if not params.is_file() or not source_manifest.is_file():
        raise FileNotFoundError("candidate params and source manifest must exist")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    params_sha = sha256_file(params)
    source_sha = sha256_file(source_manifest)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETED",
        "hardware_deployment": "PROHIBITED",
        "run_name": str(args.run_name),
        "expert": str(args.expert),
        "qualification_use": "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION",
        "candidate_kind": "H5_SEMANTIC_RESET_CANDIDATE_WRAPPER",
        "actor_observation_width": 116,
        "source_distillation_manifest": {
            "path": str(source_manifest),
            "sha256": source_sha,
            "candidate_kind": source.get("candidate_kind"),
            "teacher_source": source.get("teacher_source"),
            "target_space_distillation": source.get("target_space_distillation"),
        },
        "source_params": {
            "path": str(params),
            "sha256": params_sha,
        },
        "outputs": {
            "final_params": {
                "path": str(params),
                "sha256": params_sha,
            }
        },
        "notes": [
            "Diagnostic H5 candidate wrapper; adoption and hardware deployment are prohibited.",
            "The H5 evaluator owns absolute target decoding and the final guard.",
            "The same parameter file may be bound to planar and reverse domains only when unified mode verifies identical SHA256.",
        ],
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_manifest": str(output_manifest),
                "manifest_sha256": sha256_file(output_manifest),
                "params_sha256": params_sha,
                "expert": args.expert,
                "hardware_deployment": "PROHIBITED",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
