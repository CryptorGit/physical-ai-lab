"""Build or verify a self-contained exp_004 SafeGaitRouter package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
GENERATED = EXP_ROOT / "artifacts" / "generated_playground"
GENERATED_ROBOT = GENERATED / "playground" / "open_duck_mini_v2"
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from package_manifest import (  # noqa: E402
    BASE_V22_SHA256,
    build_router_package,
    load_and_validate_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a simulation-only OpenDuckMini SafeGaitRouter "
            "distribution package."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--formal-evidence",
        type=Path,
        required=True,
        help=(
            "Allowlisted adopted 20x30 release evidence. Stage-A/H2 component "
            "or screening artifacts are not valid for this argument."
        ),
    )
    build.add_argument(
        "--package-id", default="openduckmini-safe-gait-router-v1"
    )
    build.add_argument(
        "--base-v22-onnx",
        type=Path,
        default=(
            WORKSPACE
            / ".openduck_runtime_source_review"
            / "calibrated_hybrid_policy_v22.onnx"
        ),
    )
    build.add_argument(
        "--reverse-onnx",
        type=Path,
        help=(
            "Optional exp_004 reverse residual ONNX for audit only. It is "
            "packaged disabled at residual_scale=0.0 and is never executed."
        ),
    )
    build.add_argument(
        "--reverse-export-report",
        type=Path,
        help=(
            "Optional export_expert_onnx.py JSON. If omitted, an adjacent "
            "<model>.onnx.json is used when present."
        ),
    )
    build.add_argument(
        "--scene",
        type=Path,
        default=(
            GENERATED_ROBOT
            / "xmls"
            / "scene_flat_terrain_backlash_hardware_safe_calibrated.xml"
        ),
    )
    build.add_argument(
        "--reference",
        type=Path,
        default=(
            GENERATED_ROBOT
            / "data"
            / "polynomial_coefficients_calibrated.pkl"
        ),
    )

    verify = subparsers.add_parser("verify")
    verify.add_argument("package_root", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "verify":
        manifest = load_and_validate_package(args.package_root)
        print(
            json.dumps(
                {
                    "package": str(args.package_root.resolve()),
                    "package_id": manifest["package_id"],
                    "hardware_deployment": manifest["safety"][
                        "hardware_deployment"
                    ],
                    "reachable_model_ids": manifest["safety"][
                        "reachable_model_ids"
                    ],
                    "verified": True,
                },
                indent=2,
            )
        )
        return

    data_dir = GENERATED_ROBOT / "data"
    artifacts_dir = EXP_ROOT / "artifacts"
    manifest_path = build_router_package(
        args.output,
        package_id=args.package_id,
        base_v22_onnx=args.base_v22_onnx,
        reverse_onnx=args.reverse_onnx,
        reverse_export_report=args.reverse_export_report,
        scene=args.scene,
        reference=args.reference,
        reverse_profile=(
            artifacts_dir
            / "optimized_reverse_margin050_slew200_h1_phase7_rate105_candidate_v1.json"
        ),
        reverse_turn_left_profile=(
            artifacts_dir
            / "reverse_turn_candidates_v1"
            / "optimized_reverse_turn_left_margin050_slew200_candidate_v1.json"
        ),
        reverse_turn_right_profile=(
            artifacts_dir
            / "reverse_turn_candidates_v1"
            / "optimized_reverse_turn_right_margin050_slew200_candidate_v1.json"
        ),
        router_source=EXP_ROOT / "router.py",
        target_safety_source=EXP_ROOT / "target_safety.py",
        contract_source=EXP_ROOT / "contract.json",
        formal_release_evidence=args.formal_evidence,
        expected_base_v22_sha256=BASE_V22_SHA256,
    )
    manifest = load_and_validate_package(manifest_path.parent)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "package_id": manifest["package_id"],
                "hardware_deployment": manifest["safety"][
                    "hardware_deployment"
                ],
                "reachable_model_ids": manifest["safety"][
                    "reachable_model_ids"
                ],
                "base_v22_sha256": BASE_V22_SHA256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
