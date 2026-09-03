"""Freeze the best screened H4 reverse trajectory as a training-only teacher.

Selection is deterministic from the immutable pure bank and exact-home
failure3 screen.  The resulting JSON is not an adopted runtime profile and is
not release evidence.  It includes the complete explicit target table plus a
source-compatible ``_get_optimized_backward_reference(phase)`` adapter
contract for the isolated H4 training environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import build_h4_slew_feasible_reverse_teacher_bank as builder  # noqa: E402


DEFAULT_BANK = (
    EXP_ROOT / "artifacts" / "h4_reverse_slew_feasible_teacher_bank_v1.json"
)
DEFAULT_SCREEN = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_slew_feasible_teacher_exact_home_failure3_v1.json"
)
DEFAULT_OUTPUT = (
    EXP_ROOT / "artifacts" / "h4_reverse_slew_feasible_teacher_selected_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimized_backward_reference_adapter(
    selected_payload: Mapping[str, Any], phase: float | np.ndarray
) -> np.ndarray:
    """Pure NumPy contract for the training environment's teacher lookup."""

    teacher = selected_payload["teacher"]
    table = np.asarray(teacher["target_table_rad"], dtype=np.float64)
    if table.shape != (
        builder.RESAMPLED_PHASE_STEPS,
        len(builder.ACTUATOR_JOINT_ORDER),
    ):
        raise ValueError("selected teacher must contain a 54x14 target table")
    query = np.asarray(phase, dtype=np.float64)
    if not np.all(np.isfinite(query)):
        raise ValueError("phase must be finite")
    period = int(teacher["construction"]["phase_steps"])
    wrapped = np.mod(query, period)
    frame_index = np.floor(wrapped).astype(np.int64)
    next_index = (frame_index + 1) % period
    fraction = wrapped - np.floor(wrapped)
    return (
        (1.0 - fraction[..., None]) * table[frame_index]
        + fraction[..., None] * table[next_index]
    )


def select_payload(
    bank_payload: Mapping[str, Any],
    screen_payload: Mapping[str, Any],
    *,
    bank_path: Path = DEFAULT_BANK,
    screen_path: Path = DEFAULT_SCREEN,
) -> dict[str, Any]:
    bank_validation = builder.validate_bank(bank_payload)
    if not bank_validation["passed"]:
        raise ValueError(f"teacher bank failed validation: {bank_validation}")
    ranking = screen_payload.get("ranking_candidate_ids")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("screen must contain a non-empty candidate ranking")
    selected_id = str(ranking[0])
    candidates = bank_payload["candidates"]
    matches = [candidate for candidate in candidates if candidate["candidate_id"] == selected_id]
    if len(matches) != 1:
        raise ValueError("screen winner must identify exactly one bank candidate")
    selected = matches[0]
    validation = builder.validate_candidate(selected)
    if not validation["passed"]:
        raise ValueError(f"screen winner failed pure validation: {validation}")
    screen_matches = [
        record
        for record in screen_payload.get("candidates", [])
        if record.get("candidate_id") == selected_id
    ]
    if len(screen_matches) != 1:
        raise ValueError("screen winner must have exactly one result record")
    record = screen_matches[0]
    construction = selected["construction"]
    phase_steps = int(construction["phase_steps"])
    phase_advance = float(construction["phase_advance_bins_per_control"])
    cadence = float(construction["cadence_hz"])
    output = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_h4_reverse_selected_training_teacher",
        "status": "TRAINING_PRIOR_ONLY_FAILED_EXACT_HOME_H4_QUALIFICATION",
        "hardware_deployment": "PROHIBITED",
        "simulation_adoption_allowed": False,
        "release_allowed": False,
        "source": {
            "teacher_bank_path": str(bank_path.resolve().relative_to(EXP_ROOT)),
            "teacher_bank_sha256": _sha256(bank_path.resolve()),
            "exact_home_failure3_screen_path": str(
                screen_path.resolve().relative_to(EXP_ROOT)
            ),
            "exact_home_failure3_screen_sha256": _sha256(screen_path.resolve()),
            "selection_rule": "rank_1_force_contact_screen_training_prior_only",
        },
        "selection": {
            "candidate_id": selected_id,
            "candidate_name": selected["name"],
            "screen_rank": 1,
            "full_h4_pass_count": int(record["full_h4_pass_count"]),
            "failure3_episode_count": len(record["runs"]),
            "full_h4_failure3_passed": bool(record["full_h4_failure3_passed"]),
            "central_gait_quality_pass_count": int(
                record["summary"]["central_gait_quality"]["pass_count"]
            ),
            "fall_count": int(record["summary"]["fall_count"]),
            "minimum_speed_ratio": float(record["summary"]["minimum_speed_ratio"]),
            "maximum_force_contact_slip_rms_mps": float(
                record["summary"]["central_gait_quality"]["stance_slip"][
                    "rms_mps"
                ]["maximum"]
            ),
            "maximum_force_contact_slip_p95_mps": float(
                record["summary"]["central_gait_quality"]["stance_slip"][
                    "p95_mps"
                ]["maximum"]
            ),
            "maximum_per_stance_cumulative_slip_m": float(
                record["summary"]["central_gait_quality"]["stance_slip"][
                    "maximum_per_stance_cumulative_slip_m"
                ]["maximum"]
            ),
            "reason": (
                "best bounded screen result by force-contact slip and cadence; "
                "insufficient propulsion/support, therefore usable only as a "
                "slew-feasible initialization prior"
            ),
        },
        "adapter_contract": {
            "schema_version": 1,
            "method": "_get_optimized_backward_reference(phase)",
            "phase_unit": "table_bin",
            "phase_steps": phase_steps,
            "phase_wrap": "phase modulo 54",
            "interpolation": "periodic_linear_floor_next",
            "phase_advance_bins_per_control": phase_advance,
            "control_period_s": builder.CONTROL_FIRST_STARTUP_DT_S,
            "cadence_hz": cadence,
            "cycle_seconds": 1.0 / cadence,
            "entry_phase_preincrement_bins": float(
                construction["phase_entry_preincrement_bins"]
            ),
            "first_reference_phase_after_increment_bins": float(
                (
                    construction["phase_entry_preincrement_bins"]
                    + phase_advance
                )
                % phase_steps
            ),
            "output_joint_order": list(builder.ACTUATOR_JOINT_ORDER),
            "leg_joint_order": list(builder.LEG_JOINT_ORDER),
            "leg_actuator_indices": list(builder.LEG_ACTUATOR_INDICES),
            "head_joint_order": [
                builder.ACTUATOR_JOINT_ORDER[index]
                for index in builder.HEAD_ACTUATOR_INDICES
            ],
            "head_actuator_indices": list(builder.HEAD_ACTUATOR_INDICES),
            "head_target_rad": 0.0,
            "table_field": "teacher.target_table_rad",
            "jax_adapter": (
                "wrapped=mod(phase,54); i=floor(wrapped); j=(i+1)%54; "
                "f=wrapped-floor(wrapped); return (1-f)*table[i]+f*table[j]"
            ),
            "phase_update": (
                "phase=(phase+phase_advance_bins_per_control)%phase_steps"
            ),
            "expected_output_shape": [14],
        },
        "teacher": selected,
        "decision": {
            "training_use": "ALLOWED_AS_INITIALIZATION_PRIOR_ONLY",
            "direct_runtime_use": "PROHIBITED",
            "adoption": False,
            "expand_evaluation": False,
            "hardware": "PROHIBITED",
        },
    }
    # Exercise the serialized adapter contract at knots, midpoints, and wrap.
    probes = np.asarray((-0.25, 0.0, 0.5, 14.0, 53.75, 54.0, 108.125))
    values = optimized_backward_reference_adapter(output, probes)
    if values.shape != (len(probes), len(builder.ACTUATOR_JOINT_ORDER)):
        raise RuntimeError("selected adapter produced an unexpected shape")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("selected adapter produced non-finite targets")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate",
        type=Path,
        help="validate adapter parity for an existing selected teacher",
    )
    return parser.parse_args(argv)


def validate_selected(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate_validation = builder.validate_candidate(payload.get("teacher", {}))
    adapter = payload.get("adapter_contract", {})
    teacher = payload.get("teacher", {})
    construction = teacher.get("construction", {})
    checks = {
        "teacher_pure_validation": bool(candidate_validation["passed"]),
        "adapter_phase_steps": adapter.get("phase_steps")
        == construction.get("phase_steps")
        == builder.RESAMPLED_PHASE_STEPS,
        "adapter_phase_advance": adapter.get("phase_advance_bins_per_control")
        == construction.get("phase_advance_bins_per_control"),
        "adapter_joint_order": adapter.get("output_joint_order")
        == list(builder.ACTUATOR_JOINT_ORDER),
        "adapter_head_zero": adapter.get("head_target_rad") == 0.0,
        "adoption_false": payload.get("decision", {}).get("adoption") is False,
        "hardware_prohibited": payload.get("hardware_deployment") == "PROHIBITED",
    }
    try:
        probes = np.linspace(-2.0, 110.0, 257)
        actual = optimized_backward_reference_adapter(payload, probes)
        expected = builder.periodic_interpolate(
            np.asarray(teacher["target_table_rad"], dtype=np.float64), probes
        )
        parity_error = float(np.max(np.abs(actual - expected)))
    except (KeyError, TypeError, ValueError):
        parity_error = float("inf")
    checks["periodic_interpolation_parity"] = parity_error <= 1.0e-12
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": sorted(name for name, passed in checks.items() if not passed),
        "maximum_interpolation_parity_error_rad": parity_error,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate is not None:
        payload = json.loads(args.validate.resolve().read_text(encoding="utf-8"))
        validation = validate_selected(payload)
        print(json.dumps(validation, indent=2, sort_keys=True))
        if not validation["passed"]:
            raise SystemExit(1)
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selected teacher: {output}")
    bank_path = args.bank.resolve()
    screen_path = args.screen.resolve()
    payload = select_payload(
        json.loads(bank_path.read_text(encoding="utf-8")),
        json.loads(screen_path.read_text(encoding="utf-8")),
        bank_path=bank_path,
        screen_path=screen_path,
    )
    validation = validate_selected(payload)
    if not validation["passed"]:
        raise RuntimeError(f"selected teacher failed validation: {validation}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "candidate_id": payload["selection"]["candidate_id"],
                "candidate_name": payload["selection"]["candidate_name"],
                "adapter_validation": validation,
                "adoption": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
