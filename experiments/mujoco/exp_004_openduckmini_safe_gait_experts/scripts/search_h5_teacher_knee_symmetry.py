"""Diagnose bilateral knee symmetry in an H5 reverse target table.

This is a simulation-only search.  It leaves the actor, decoder, guard,
thresholds, and deployment package untouched; only the reverse teacher table's
left-knee row is replaced by a bounded sign-reflected right-knee deviation.
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

from safe_gait_experts.contract import ACTUATOR_JOINT_ORDER, SAFE_INIT_POS  # noqa: E402
from scripts import evaluate_h5_routed_transitions as h5  # noqa: E402
from scripts import explore_h5_target_program as explore  # noqa: E402


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", type=_path, required=True)
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    parser.add_argument("--phase-rate", type=float, default=1.0)
    parser.add_argument("--phase-offset", type=float, default=0.0)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    return parser


def _load_table(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = np.asarray(payload["teacher_source"]["target_table_rad"], dtype=np.float64)
    if table.shape != (54, 14) or not np.all(np.isfinite(table)):
        raise ValueError("teacher manifest must contain a finite 54x14 table")
    return table


def _table_for_scale(source: np.ndarray, scale: float) -> np.ndarray:
    if not np.isfinite(float(scale)) or not 0.0 <= float(scale) <= 1.5:
        raise ValueError("knee symmetry scale must be in [0, 1.5]")
    table = np.asarray(source, dtype=np.float64).copy()
    left_initial = float(SAFE_INIT_POS["left_knee"])
    right_initial = float(SAFE_INIT_POS["right_knee"])
    table[:, 3] = left_initial - float(scale) * (table[:, 12] - right_initial)
    table[:, 5:9] = 0.0
    if not np.all(np.isfinite(table)):
        raise ValueError("transformed teacher table is not finite")
    return table


def _summary(item: dict[str, Any]) -> dict[str, Any]:
    segment = item["segment"]
    metrics = segment.get("metrics", {})
    quality = segment.get("gait_quality_metrics", {})
    return {
        "knee_symmetry_scale": item["knee_symmetry_scale"],
        "quality_passed": bool(item.get("quality_passed", False)),
        "fell": bool(segment.get("fell", False)),
        "projected_primary_velocity": metrics.get("projected_primary_velocity"),
        "steady_linear_tracking_ratio": quality.get("steady_linear_tracking_ratio"),
        "stance_slip_rms_mps": quality.get("stance_slip_rms_mps"),
        "stance_slip_p95_mps": quality.get("stance_slip_p95_mps"),
        "single_support_rate": quality.get("single_support_rate"),
        "contact_velocity_coverage": quality.get("contact_velocity_coverage"),
        "total_normal_force_p99_fraction_body_weight": quality.get(
            "total_normal_force_p99_fraction_body_weight"
        ),
        "left_step_count": quality.get("left_step_count"),
        "right_step_count": quality.get("right_step_count"),
        "quality_failures": list(item.get("quality_failures", [])),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = _load_table(args.teacher_manifest)
    params = dict(explore.DEFAULT_H5_PARAMS)
    manifests = dict(explore.DEFAULT_H5_MANIFESTS)
    setup = explore._args(params, manifests)
    simulator, bank, metadata = h5._build_simulator(setup)
    rows: list[dict[str, Any]] = []
    for scale in args.scales:
        table = _table_for_scale(source, float(scale))
        result = explore._run_on_simulator(
            simulator,
            bank,
            metadata,
            route="reverse",
            seconds=float(args.seconds),
            phase_offset=float(args.phase_offset),
            phase_rate_scale=float(args.phase_rate),
            target_scale=float(args.target_scale),
            mode="teacher",
            action_scale=1.0,
            action_smoothing=1.0,
            teacher_table=table,
            actor_residual_scale=0.0,
            target_smoothing=1.0,
        )
        result["knee_symmetry_scale"] = float(scale)
        rows.append(result)
    rows.sort(
        key=lambda item: (
            bool(item.get("quality_passed", False)),
            abs(float(item["segment"]["metrics"].get("projected_primary_velocity", 0.0)) - 0.05),
        )
    )
    payload = {
        "schema_version": 1,
        "status": "COMPLETED",
        "hardware_deployment": "PROHIBITED",
        "evaluator": "openduckmini-exp004-h5-teacher-knee-symmetry-search",
        "source_teacher_manifest": str(args.teacher_manifest),
        "source_teacher_manifest_sha256": h5.sha256_file(args.teacher_manifest),
        "configuration": {
            "scales": [float(value) for value in args.scales],
            "phase_rate": float(args.phase_rate),
            "phase_offset": float(args.phase_offset),
            "target_scale": float(args.target_scale),
            "seconds": float(args.seconds),
            "joint_order": list(ACTUATOR_JOINT_ORDER),
        },
        "best_summary": _summary(rows[0]),
        "summaries": [_summary(row) for row in rows],
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "sha256": h5.sha256_file(args.output),
        "best_summary": payload["best_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
