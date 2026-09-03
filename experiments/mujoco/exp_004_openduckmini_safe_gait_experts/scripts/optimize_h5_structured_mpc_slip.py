"""Finite-difference local optimization of the structured reverse teacher.

The independent and structured CEM experiments established that a correlated
24-dimensional proposal can produce safe alternating support, but its stance
slip is too high.  This script performs one deterministic local trajectory
optimization from that candidate.  It does not enlarge CEM population,
change the frozen thresholds, or use a legacy target as a label.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
if str(EXP_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(EXP_ROOT / "scripts"))

from scripts import build_h5_mpc_reverse_teacher as base  # noqa: E402
from scripts import test_h5_structured_mpc_reverse_teacher as structured  # noqa: E402
from scripts import evaluate_h5_routed_transitions as h5  # noqa: E402


DEFAULT_START = (
    EXP_ROOT
    / "artifacts"
    / "h5_structured_mpc_reverse_teacher_prescreen2_20260811.json"
)
DEFAULT_OUTPUT = EXP_ROOT / "artifacts" / "h5_structured_mpc_slip_optimized_20260811"
MAX_COEFFICIENT = 2.5
MAXITER = 100


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _objective(
    coefficients_flat: np.ndarray,
    simulator: Any,
    snapshot_data: Any,
    snapshot_targets: np.ndarray,
    proposal_basis: dict[str, np.ndarray],
    history: list[dict[str, Any]],
) -> float:
    coefficients = np.asarray(coefficients_flat, dtype=np.float64).reshape(8, 3)
    clipped = np.clip(coefficients, -MAX_COEFFICIENT, MAX_COEFFICIENT)
    cost, details = base._rollout_cost(
        simulator,
        snapshot_data,
        snapshot_targets,
        clipped,
        horizon_ticks=structured.HORIZON_TICKS,
        block_ticks=structured.BLOCK_TICKS,
        proposal_basis=proposal_basis,
    )
    if not np.isfinite(cost):
        value = 1.0e9
    else:
        speed = float(details.get("projected_speed_mean_mps", 0.0))
        support = float(details.get("single_support_rate", 0.0))
        slip_rms = float(details.get("slip_rms_mps", 1.0))
        slip_p95 = float(details.get("slip_p95_mps", 1.0))
        force_p99 = float(details.get("force_p99_body_weight", 99.0))
        margin = float(details.get("minimum_joint_margin_rad", -1.0))
        touchdown = int(details.get("touchdown_count", 0))
        valid = bool(details.get("valid", False))
        pre = (
            structured._prescreen(details)
            if valid and "contacts" in details
            else None
        )
        # Fixed lexicographic-style penalties: safety/command/support first,
        # then the measured slip objective.  The numeric constants are only
        # ordering barriers and are recorded in the artifact.
        violation = 0.0
        violation += max(0.0, 0.0375 - speed) ** 2 * 1.0e4
        violation += max(0.0, speed - 0.0625) ** 2 * 1.0e4
        violation += max(0.0, 0.25 - support) ** 2 * 1.0e4
        violation += max(0.0, 0.005 - margin) ** 2 * 1.0e5
        violation += max(0.0, force_p99 - 3.0) ** 2 * 1.0e4
        violation += max(0.0, 2 - touchdown) ** 2 * 1.0e3
        if not valid:
            violation += 1.0e6
        value = (
            violation
            + (slip_rms / 0.015) ** 2
            + (slip_p95 / 0.030) ** 2
            + 0.05 * float(cost)
        )
        if pre is not None and bool(pre["passed"]):
            value -= 10.0
    history.append(
        {
            "evaluation": len(history),
            "objective": float(value),
            "rollout_cost": float(cost),
            "valid": bool(details.get("valid", False)),
            "projected_speed_mean_mps": float(details.get("projected_speed_mean_mps", 0.0)),
            "single_support_rate": float(details.get("single_support_rate", 0.0)),
            "slip_rms_mps": float(details.get("slip_rms_mps", 1.0)),
            "slip_p95_mps": float(details.get("slip_p95_mps", 1.0)),
            "force_p99_body_weight": float(details.get("force_p99_body_weight", 99.0)),
            "minimum_joint_margin_rad": float(details.get("minimum_joint_margin_rad", -1.0)),
            "touchdown_count": int(details.get("touchdown_count", 0)),
        }
    )
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, default=DEFAULT_START)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start_path = args.start.expanduser().resolve()
    if not start_path.is_file():
        raise FileNotFoundError(start_path)
    if args.maxiter <= 0:
        raise ValueError("maxiter must be positive")
    output_prefix = args.output_prefix.expanduser().resolve()
    json_path = output_prefix.with_suffix(".json")
    npz_path = output_prefix.with_suffix(".npz")
    if json_path.exists() or npz_path.exists():
        raise FileExistsError(f"refusing to overwrite {json_path} or {npz_path}")

    start_payload = json.loads(start_path.read_text(encoding="utf-8"))
    initial = np.asarray(
        start_payload["optimization"]["best_coefficients"], dtype=np.float64
    )
    if initial.shape != (8, 3) or not np.all(np.isfinite(initial)):
        raise ValueError("structured start coefficients must be finite 8x3")
    simulator, bank, build_metadata = h5._build_simulator(
        base._build_args(
            base.DEFAULT_PARAMS,
            base.DEFAULT_PLANAR_MANIFEST,
            base.DEFAULT_REVERSE_MANIFEST,
            base.DEFAULT_GENERATED_ROOT,
        )
    )
    (
        snapshot_data,
        snapshot_targets,
        snapshot_history,
        snapshot_phase,
        basis_provenance,
        proposal_basis,
    ) = structured._build_trace_basis(
        simulator,
        seed=structured.SEED,
        start_tick=structured.START_TICK,
        horizon_ticks=structured.HORIZON_TICKS,
    )
    del snapshot_history

    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("scipy is required for the finite-difference experiment") from exc

    history: list[dict[str, Any]] = []
    result = minimize(
        _objective,
        initial.reshape(-1),
        args=(simulator, snapshot_data, snapshot_targets, proposal_basis, history),
        method="Nelder-Mead",
        options={
            "maxiter": int(args.maxiter),
            "xatol": 1.0e-3,
            "fatol": 1.0e-3,
            "adaptive": True,
            "disp": False,
        },
    )
    optimized = np.clip(np.asarray(result.x, dtype=np.float64).reshape(8, 3), -MAX_COEFFICIENT, MAX_COEFFICIENT)
    best_cost, best_details = base._rollout_cost(
        simulator,
        snapshot_data,
        snapshot_targets,
        optimized,
        horizon_ticks=structured.HORIZON_TICKS,
        block_ticks=structured.BLOCK_TICKS,
        collect_trace=True,
        proposal_basis=proposal_basis,
    )
    screen = structured._prescreen(best_details)
    if bank.legacy_fallback_count != 0:
        raise RuntimeError("trajectory optimizer unexpectedly used a legacy H5 fallback")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        optimized_coefficients=optimized.astype(np.float32),
        proposal_trace_delta=np.asarray(proposal_basis["trace_delta"], dtype=np.float32),
        proposal_sagittal_delta=np.asarray(proposal_basis["sagittal_delta"], dtype=np.float32),
        proposal_roll_transfer=np.asarray(proposal_basis["roll_transfer"], dtype=np.float32),
        local_velocities=np.asarray(best_details["local_velocities"], dtype=np.float32),
        normal_forces=np.asarray(best_details["normal_forces"], dtype=np.float32),
        tangential_speeds=np.asarray(best_details["tangential_speeds"], dtype=np.float32),
        contacts=np.asarray(best_details["contacts"], dtype=np.uint8),
    )
    payload = {
        "schema_version": 1,
        "evaluator_id": "openduckmini-exp004-h5-structured-mpc-slip-optimizer-v1",
        "evaluation_mode": "DIAGNOSTIC_STRUCTURED_MPC_FINITE_DIFFERENCE_NOT_QUALIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_deployment": "PROHIBITED",
        "configuration": {
            "optimizer": "scipy.optimize.Nelder-Mead",
            "maxiter": int(args.maxiter),
            "xatol": 1.0e-3,
            "fatol": 1.0e-3,
            "parameterization": "8 blocks x [alpha_trace,beta_hip_roll,gamma_sagittal]",
            "snapshot_start_tick": structured.START_TICK,
            "snapshot_phase_index": float(snapshot_phase),
            "horizon_ticks": structured.HORIZON_TICKS,
            "block_ticks": structured.BLOCK_TICKS,
            "constraints": {
                "speed_band_mps": [0.0375, 0.0625],
                "minimum_single_support_rate": 0.25,
                "minimum_qpos_margin_rad": 0.005,
                "maximum_force_p99_body_weight": 3.0,
                "slip_rms_mps": 0.015,
                "slip_p95_mps": 0.030,
            },
        },
        "provenance": {
            "start_artifact": str(start_path),
            "start_artifact_sha256": h5.sha256_file(start_path),
            "teacher_script": str(Path(__file__).resolve()),
            "teacher_script_sha256": h5.sha256_file(Path(__file__).resolve()),
            "base_mpc_script": str(Path(base.__file__).resolve()),
            "base_mpc_script_sha256": h5.sha256_file(Path(base.__file__).resolve()),
            "structured_script": str(Path(structured.__file__).resolve()),
            "structured_script_sha256": h5.sha256_file(Path(structured.__file__).resolve()),
            "h5_build_metadata": build_metadata,
            "proposal_basis": basis_provenance,
            "policy_bank": bank.manifest(),
        },
        "optimization": {
            "scipy_success": bool(result.success),
            "scipy_status": int(result.status),
            "scipy_message": str(result.message),
            "function_evaluations": int(result.nfev),
            "iterations_reported": int(result.nit),
            "initial_coefficients": initial.tolist(),
            "optimized_coefficients": optimized.tolist(),
            "best_cost": float(best_cost),
            "best_details": _json_safe(best_details),
            "history_tail": _json_safe(history[-20:]),
        },
        "prescreen": _json_safe(screen),
        "outputs": {
            "npz": {"path": str(npz_path), "sha256": h5.sha256_file(npz_path)},
            "json": {"path": str(json_path)},
        },
        "adoption_allowed": False,
        "release_allowed": False,
        "next_step": (
            "ALLOW_STRICT_MULTI_SEED_ONLY_IF_PRESCREEN_PASSED"
            if screen["passed"]
            else "REJECT_STRUCTURED_MPC_TRAJECTORY_TEACHER"
        ),
    }
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "npz": str(npz_path),
                "prescreen_passed": screen["passed"],
                "scipy_success": bool(result.success),
                "function_evaluations": int(result.nfev),
                "next_step": payload["next_step"],
                "hardware_deployment": "PROHIBITED",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
