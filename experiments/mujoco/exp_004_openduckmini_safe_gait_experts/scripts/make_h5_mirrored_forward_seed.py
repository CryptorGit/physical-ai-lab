"""Create a diagnostic H5 seed manifest from a mirrored audited-forward table.

The generated manifest is intentionally not an adoption artifact.  It binds an
existing H5 actor checkpoint to a 54-row absolute-target teacher table derived
from the calibrated V22 forward actor, then applies the same target-space
decoder inversion used by the exploration harness.  This makes the next PPO
run reproducible without copying or editing a trusted checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
for root in (EXP_ROOT, EXP_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from safe_gait_experts.h5_target_contract import h5_decode_absolute_targets  # noqa: E402
from scripts import evaluate_h5_routed_transitions as h5  # noqa: E402
from scripts import explore_h5_target_program as explore  # noqa: E402


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=_path, required=True)
    parser.add_argument("--planar-params", type=_path, required=True)
    parser.add_argument("--planar-manifest", type=_path, required=True)
    parser.add_argument("--reverse-params", type=_path, required=True)
    parser.add_argument("--reverse-manifest", type=_path, required=True)
    parser.add_argument("--seed-params", type=_path, required=True)
    parser.add_argument("--seed-manifest-source", type=_path, required=True)
    parser.add_argument("--search-evidence", type=_path, required=True)
    parser.add_argument("--output-manifest", type=_path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase-rate", type=float, required=True)
    parser.add_argument("--phase-offset", type=float, required=True)
    parser.add_argument("--target-scale", type=float, required=True)
    parser.add_argument("--target-transform", default="flip_sagittal")
    parser.add_argument("--target-leg-gains", type=float, nargs=10, required=True)
    return parser


def _policy_args(path: Path) -> list[str]:
    return [f"{role}={path}" for role in h5.REQUIRED_POLICY_ROLES]


def _build_table(args: argparse.Namespace) -> np.ndarray:
    params = {
        "planar": args.planar_params,
        "reverse": args.reverse_params,
    }
    manifests = {
        "planar": args.planar_manifest,
        "reverse": args.reverse_manifest,
    }
    setup = explore._args(params, manifests)
    setup.policy = _policy_args(args.policy)
    simulator, _bank, _metadata = h5._build_simulator(setup)
    table: list[np.ndarray] = []
    # H5 target tables are evaluated at 2*phase.  Row q therefore represents
    # the calibrated forward actor at source phase q/2 over the 27-frame H5
    # phase domain.
    for row_index in range(54):
        phase = float(row_index) / 2.0
        observation = np.zeros(116, dtype=np.float32)
        angle = phase / float(simulator.evaluator.phase_steps) * 2.0 * np.pi
        observation[99:101] = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float32)
        observation[101:104] = np.asarray((-0.05, 0.0, 0.0), dtype=np.float32)
        target = explore._legacy_forward_base_target(
            simulator,
            observation,
            phase_offset=float(args.phase_offset),
            phase_rate_scale=float(args.phase_rate),
        )
        target = explore._transform_forward_target(target, args.target_transform)
        target = explore._apply_target_leg_gains(target, args.target_leg_gains)
        initial = np.asarray(
            [float(explore.SAFE_INIT_POS[name]) for name in explore.ACTUATOR_JOINT_ORDER],
            dtype=np.float64,
        )
        target = initial + float(args.target_scale) * (target - initial)
        target[5:9] = 0.0
        action = explore._inverse_target(target)
        decoded = np.asarray(
            h5_decode_absolute_targets(action, domain="reverse"),
            dtype=np.float64,
        )
        table.append(decoded)
    result = np.asarray(table, dtype=np.float32)
    if result.shape != (54, 14) or not np.all(np.isfinite(result)):
        raise RuntimeError("generated target table is not finite 54x14")
    if not np.array_equal(result[:, 5:9], np.zeros((54, 4), dtype=np.float32)):
        raise RuntimeError("generated target table head channels are not zero")
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for path in (
        args.policy,
        args.planar_params,
        args.planar_manifest,
        args.reverse_params,
        args.reverse_manifest,
        args.seed_params,
        args.seed_manifest_source,
        args.search_evidence,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    table = _build_table(args)
    output = args.output_manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    seed_sha = h5.sha256_file(args.seed_params)
    source_manifest_sha = h5.sha256_file(args.seed_manifest_source)
    search_sha = h5.sha256_file(args.search_evidence)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETED",
        "hardware_deployment": "PROHIBITED",
        "run_name": str(args.run_name),
        "expert": "unified",
        "qualification_use": "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION",
        "candidate_kind": "H5_TARGET_SPACE_DISTILLED_SEED",
        "actor_observation_width": 116,
        "source_candidate": {
            "path": str(args.seed_manifest_source),
            "sha256": source_manifest_sha,
        },
        "source_params": {
            "path": str(args.seed_params),
            "sha256": seed_sha,
            "h4_validation": {
                "actor_observation_width": 116,
                "critic_observation_width": 227,
                "action_width": 14,
                "structure_validated": True,
            },
        },
        "outputs": {
            "final_params": {
                "path": str(args.seed_params),
                "sha256": seed_sha,
            }
        },
        "notes": [
            "Simulation-only diagnostic wrapper; adoption and hardware deployment are prohibited.",
            "The H5 evaluator owns absolute target decoding and the final guard.",
            "This table is a mirrored-forward hypothesis, not a qualification result.",
        ],
        "teacher_source": {
            "mode": "mirrored_v22_forward_target",
            "source_policy_path": str(args.policy),
            "source_policy_sha256": h5.sha256_file(args.policy),
            "source_search_evidence": str(args.search_evidence),
            "source_search_evidence_sha256": search_sha,
            "target_table_contract": "H5_54_ROW_ABSOLUTE_TARGET_TABLE_V1",
            "target_table_shape": [54, 14],
            "target_table_rad": table.tolist(),
            "phase_rate": float(args.phase_rate),
            "phase_offset": float(args.phase_offset),
            "target_scale": float(args.target_scale),
            "target_transform": str(args.target_transform),
            "target_leg_gains": [float(value) for value in args.target_leg_gains],
        },
        "candidate_domains": {
            "planar": {
                "params_path": str(args.planar_params),
                "params_sha256": h5.sha256_file(args.planar_params),
                "manifest_path": str(args.planar_manifest),
                "manifest_sha256": h5.sha256_file(args.planar_manifest),
            },
            "reverse": {
                "params_path": str(args.reverse_params),
                "params_sha256": h5.sha256_file(args.reverse_params),
                "manifest_path": str(args.reverse_manifest),
                "manifest_sha256": h5.sha256_file(args.reverse_manifest),
            },
        },
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "sha256": h5.sha256_file(output),
        "target_table_shape": list(table.shape),
        "target_table_range_rad": [float(np.min(table)), float(np.max(table))],
        "hardware_deployment": "PROHIBITED",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
