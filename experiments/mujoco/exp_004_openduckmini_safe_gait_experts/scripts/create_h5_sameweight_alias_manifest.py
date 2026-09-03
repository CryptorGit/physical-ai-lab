"""Create an auditable diagnostic manifest alias for the same H5 weight.

The H5 evaluator keeps domain labels in its manifest contract.  This helper
does not copy or alter parameters: it binds the exact same parameter file and
SHA to a second diagnostic domain so a one-weight screening run can exercise
every route without silently introducing a second actor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_alias_manifest(
    *, source_manifest: Path, params: Path, domain: str
) -> dict[str, object]:
    """Build a wrapper that binds to, rather than impersonates, its source run.

    The strict H5 evaluator resolves the command contract from the immutable
    unified training manifest.  Consequently a domain wrapper must carry one
    ``source_candidate`` record to that manifest; copying and relabelling the
    training manifest would destroy that provenance edge.
    """

    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETED":
        raise ValueError("source manifest must be COMPLETED")
    if payload.get("expert") != "unified":
        raise ValueError("source manifest must describe one unified policy")
    if payload.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("source manifest must prohibit hardware deployment")
    if (
        payload.get("qualification_use")
        != "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION"
    ):
        raise ValueError("source manifest must be a diagnostic-only run")

    source_output_record = payload.get("outputs", {}).get("final_params", {})
    source_manifest_param_path = str(source_output_record.get("path", ""))
    source_manifest_param_sha = source_output_record.get("sha256")
    if not source_manifest_param_path or not isinstance(source_manifest_param_sha, str):
        raise ValueError("source manifest is missing outputs.final_params binding")
    params_sha = sha256_file(params)
    if source_manifest_param_sha != params_sha:
        raise ValueError("provided params do not match source manifest binding")

    # Keep this deliberately small.  It is a candidate wrapper, not a second
    # training run, and all training/command provenance is resolved through the
    # single hash-bound source_candidate record below.
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "expert": domain,
        "activity": "H5_DIAGNOSTIC_SINGLE_WEIGHT_ALIAS",
        "candidate_kind": "H5_SINGLE_WEIGHT_ALIAS_UNQUALIFIED",
        "qualification_use": "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION",
        "hardware_deployment": "PROHIBITED",
        "notes": (
            "Diagnostic-only alias of one unified H5 actor. No parameters are "
            "copied, modified, or trained by this artifact."
        ),
        "outputs": {
            "final_params": {
                # Preserve the source manifest's runtime-native path (normally
                # /mnt/c on WSL); the evaluator binds this exact location.
                "path": source_manifest_param_path,
                "sha256": params_sha,
            }
        },
        "source_candidate": {
            "path": str(source_manifest),
            "sha256": sha256_file(source_manifest),
        },
        "single_weight_alias": {
            "parameter_path_as_recorded": source_manifest_param_path,
            "same_parameter_path": True,
            "same_parameter_sha256": params_sha,
            "one_weight_screening_only": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--domain", choices=("planar", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = args.source_manifest.resolve()
    params = args.params.resolve()
    output = args.output.resolve()
    if not source_manifest.is_file() or not params.is_file():
        raise FileNotFoundError("source manifest and parameter file are required")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = build_alias_manifest(
        source_manifest=source_manifest, params=params, domain=args.domain
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
