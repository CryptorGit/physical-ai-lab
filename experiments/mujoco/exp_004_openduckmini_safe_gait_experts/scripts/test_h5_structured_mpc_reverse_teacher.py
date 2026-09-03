"""Run the single structured-CEM reverse-teacher pre-screen.

This is the next PDCA experiment after the independent 80-dimensional CEM
failed to produce correlated load-transfer and touchdown events.  It is a
one-shot, simulation-only test: a legacy V22 target trace is regenerated only
as a proposal basis, never as a teacher label.  The candidate is optimized
from the exact MuJoCo state at control tick 60 for a 40-tick horizon.

The script deliberately refuses to launch a six-second or multi-seed teacher
run.  That is allowed only when this structured candidate passes every
pre-screen gate, after which the main PDCA loop can promote the same
parameterization to the frozen strict screen.
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

from safe_gait_experts.contract import ACTUATOR_JOINT_ORDER  # noqa: E402
from safe_gait_experts.h4_training_alignment import (  # noqa: E402
    OBSERVATION_POLICY_COMMAND_SLICE,
)
from safe_gait_experts.h5_command_contract import h5_unified_policy_command  # noqa: E402
from safe_gait_experts.h5_target_contract import h5_decode_absolute_targets  # noqa: E402
from scripts import evaluate_h4_routed_transitions as h4  # noqa: E402
from scripts import evaluate_h5_routed_transitions as h5  # noqa: E402
from scripts import evaluate_routed_transitions as routed  # noqa: E402
from scripts import build_h5_mpc_reverse_teacher as base  # noqa: E402


SEED = 20260810
START_TICK = 60
HORIZON_TICKS = 40
BLOCK_TICKS = 5
POPULATION = 64
ELITES = 8
ITERATIONS = 3
STRUCTURED_STD = 0.35
BASIS_PHASE_RATE = 0.75
BASIS_PHASE_OFFSET = 13.5
BASIS_TARGET_SCALE = 1.0
BASIS_COMMAND = np.asarray((0.05, 0.0, 0.0), dtype=np.float64)
DEFAULT_BASIS_EVIDENCE = (
    EXP_ROOT
    / "artifacts"
    / "h5_reverse_legacy_forward_unified_rate075_offset135_scale1_6s_20260811.json"
)
DEFAULT_OUTPUT_PREFIX = (
    EXP_ROOT / "artifacts" / "h5_structured_mpc_reverse_teacher_prescreen_20260811"
)


def _observation(
    simulator: Any,
    data: Any,
    default: np.ndarray,
    previous_targets: np.ndarray,
    action_history: list[np.ndarray],
    phase_index: float,
) -> np.ndarray:
    h4._H4_RUN_CONTEXT["physical_command"] = base.PHYSICAL_COMMAND.copy()
    command7 = np.asarray((*base.POLICY_COMMAND, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    observation = np.asarray(
        simulator.evaluator._observation(
            data,
            command7,
            default,
            previous_targets,
            action_history,
            phase_index / float(simulator.evaluator.phase_steps) * 2.0 * np.pi,
        ),
        dtype=np.float32,
    )
    if observation.shape != (116,):
        raise RuntimeError(f"structured teacher observation width drifted: {observation.shape}")
    return observation


def _legacy_basis_target(simulator: Any, observation: np.ndarray, phase_index: float) -> np.ndarray:
    """Regenerate the historical rate=.75/offset=13.5 target basis.

    The source artifact records this as ``legacy_forward_target`` with no
    sagittal transform.  Its target is only a proposal coordinate system;
    none of these targets are emitted as training labels by this experiment.
    """

    policy_observation = np.asarray(observation[:101], dtype=np.float32).copy()
    warped_angle = (
        (BASIS_PHASE_RATE * float(phase_index) + BASIS_PHASE_OFFSET)
        / float(simulator.evaluator.phase_steps)
        * 2.0
        * np.pi
    )
    policy_observation[99:101] = np.asarray(
        (np.cos(warped_angle), np.sin(warped_angle)), dtype=np.float32
    )
    policy_command = np.asarray(
        h5_unified_policy_command(BASIS_COMMAND), dtype=np.float32
    )
    policy_observation[OBSERVATION_POLICY_COMMAND_SLICE] = np.asarray(
        (*policy_command, 0.0, 0.0, 0.0, 0.0), dtype=np.float32
    )
    action = simulator.bank.base_bank.infer("forward", policy_observation)
    target = np.asarray(
        h5_decode_absolute_targets(action, domain="planar"), dtype=np.float64
    )
    target[base.HEAD_INDICES] = 0.0
    return target


def _build_trace_basis(
    simulator: Any,
    *,
    seed: int,
    start_tick: int,
    horizon_ticks: int,
) -> tuple[Any, np.ndarray, list[np.ndarray], float, dict[str, Any], np.ndarray]:
    """Replay the proposal-only legacy trace and capture the tick-60 state."""

    data, reset_audit = simulator._initial_data(seed, 0.0, 0.0)
    default = np.asarray(simulator.model.keyframe("home").ctrl, dtype=np.float64).copy()
    previous_targets = np.asarray(
        data.qpos[simulator.evaluator.actuator_qpos_addr], dtype=np.float64
    ).copy()
    guard = routed.FinalTargetSafetyGuard(
        base.SAFE_JOINT_LIMITS,
        previous_targets,
        margin_rad=base.TARGET_MARGIN_RAD,
        max_slew_rate_rad_s=2.0,
    )
    previous_targets = guard.previous_targets
    action_history = [np.zeros(14, dtype=np.float32) for _ in range(3)]
    phase_index = 7.0
    trace_deltas: list[np.ndarray] = []
    snapshot_data: Any | None = None
    snapshot_targets: np.ndarray | None = None
    snapshot_history: list[np.ndarray] | None = None
    snapshot_phase: float | None = None
    total_ticks = start_tick + horizon_ticks
    for tick in range(total_ticks):
        if tick == start_tick:
            snapshot_data = base._copy_data(simulator, data)
            snapshot_targets = previous_targets.copy()
            snapshot_history = [item.copy() for item in action_history]
            snapshot_phase = float(phase_index)
        observation = _observation(
            simulator,
            data,
            default,
            previous_targets,
            action_history,
            phase_index,
        )
        desired = _legacy_basis_target(simulator, observation, phase_index)
        applied = guard.step(desired, base.CONTROL_DT_S)
        trace_deltas.append(
            np.asarray(applied[base.LEG_INDICES] - previous_targets[base.LEG_INDICES])
        )
        data.ctrl[:] = applied
        simulator.mujoco.mj_step(
            simulator.model,
            data,
            nstep=int(simulator.runtime.DECIMATION),
        )
        action_label = base._inverse_decoder(applied)
        action_history = [
            action_label.astype(np.float32),
            action_history[0].copy(),
            action_history[1].copy(),
        ]
        previous_targets = applied.copy()
        phase_index = (phase_index + 0.81) % float(simulator.evaluator.phase_steps)
    if (
        snapshot_data is None
        or snapshot_targets is None
        or snapshot_history is None
        or snapshot_phase is None
    ):
        raise RuntimeError("structured teacher snapshot was not captured")
    trace = np.asarray(trace_deltas[start_tick : start_tick + horizon_ticks], dtype=np.float64)
    if trace.shape != (horizon_ticks, 10):
        raise RuntimeError(f"legacy proposal trace shape drifted: {trace.shape}")
    leg_names = [
        name
        for name in ACTUATOR_JOINT_ORDER
        if name not in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
    ]
    sagittal_mask = np.asarray(
        [
            name
            in {
                "left_hip_pitch",
                "left_knee",
                "left_ankle",
                "right_hip_pitch",
                "right_knee",
                "right_ankle",
            }
            for name in leg_names
        ],
        dtype=bool,
    )
    sagittal_trace = np.where(sagittal_mask[None, :], trace, 0.0)
    roll_transfer = np.zeros(10, dtype=np.float64)
    roll_transfer[leg_names.index("left_hip_roll")] = 0.02
    roll_transfer[leg_names.index("right_hip_roll")] = -0.02
    basis = {
        "trace_delta": trace,
        "sagittal_delta": sagittal_trace,
        "roll_transfer": roll_transfer,
    }
    provenance = {
        "basis_kind": "legacy_forward_target_proposal_only",
        "basis_artifact": str(DEFAULT_BASIS_EVIDENCE.resolve()),
        "basis_artifact_sha256": h5.sha256_file(DEFAULT_BASIS_EVIDENCE),
        "phase_rate": BASIS_PHASE_RATE,
        "phase_offset": BASIS_PHASE_OFFSET,
        "target_scale": BASIS_TARGET_SCALE,
        "target_transform": "none",
        "teacher_labels_from_basis": False,
        "reset_qpos_audit": reset_audit,
        "snapshot_start_tick": start_tick,
        "snapshot_phase_index": snapshot_phase,
    }
    return snapshot_data, snapshot_targets, snapshot_history, snapshot_phase, provenance, basis


def _touchdown_summary(contacts: np.ndarray) -> dict[str, Any]:
    contact_array = np.asarray(contacts, dtype=bool)
    if contact_array.ndim != 2 or contact_array.shape[1] != 2:
        raise ValueError("contacts must be N x 2")
    single_left = contact_array[:, 0] & ~contact_array[:, 1]
    single_right = contact_array[:, 1] & ~contact_array[:, 0]
    touchdown_feet: list[int] = []
    previous: np.ndarray | None = None
    for row in contact_array:
        if previous is not None:
            for foot in (0, 1):
                if not bool(previous[foot]) and bool(row[foot]):
                    touchdown_feet.append(foot)
        previous = row
    alternation = (
        float(
            np.mean(
                np.asarray(touchdown_feet[1:])
                != np.asarray(touchdown_feet[:-1])
            )
        )
        if len(touchdown_feet) >= 2
        else 0.0
    )
    return {
        "single_support_rate": float(np.mean(np.logical_xor(contact_array[:, 0], contact_array[:, 1]))),
        "single_left_samples": int(np.count_nonzero(single_left)),
        "single_right_samples": int(np.count_nonzero(single_right)),
        "flight_rate": float(np.mean(contact_array.sum(axis=1) == 0)),
        "touchdown_feet": touchdown_feet,
        "touchdown_count": len(touchdown_feet),
        "alternation_fraction": alternation,
    }


def _prescreen(details: dict[str, Any]) -> dict[str, Any]:
    contacts = np.asarray(details.get("contacts"), dtype=bool)
    touchdown = _touchdown_summary(contacts)
    checks = {
        "valid_40_ticks": bool(details.get("valid") and details.get("completed_ticks") == HORIZON_TICKS),
        "minimum_qpos_margin_rad_ge_005": bool(float(details.get("minimum_joint_margin_rad", -np.inf)) >= 0.005),
        "projected_reverse_speed_band": bool(0.0375 <= float(details.get("projected_speed_mean_mps", np.nan)) <= 0.0625),
        "slip_rms_le_015": bool(float(details.get("slip_rms_mps", np.inf)) <= 0.015),
        "slip_p95_le_030": bool(float(details.get("slip_p95_mps", np.inf)) <= 0.030),
        "force_p99_le_3": bool(float(details.get("force_p99_body_weight", np.inf)) <= 3.0),
        "flight_zero": bool(touchdown["flight_rate"] == 0.0),
        "single_support_ge_025": bool(touchdown["single_support_rate"] >= 0.25),
        "both_single_support_legs": bool(
            touchdown["single_left_samples"] > 0 and touchdown["single_right_samples"] > 0
        ),
        "touchdown_alternation_one": bool(touchdown["alternation_fraction"] == 1.0),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "touchdown": touchdown,
        "candidate_details": details,
        "adoption_allowed": False,
        "release_allowed": False,
        "hardware_deployment": "PROHIBITED",
    }


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--start-tick", type=int, default=START_TICK)
    parser.add_argument("--basis-evidence", type=Path, default=DEFAULT_BASIS_EVIDENCE)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_tick < 0:
        raise ValueError("start tick must be non-negative")
    basis_evidence = args.basis_evidence.expanduser().resolve()
    if not basis_evidence.is_file():
        raise FileNotFoundError(f"basis evidence is missing: {basis_evidence}")
    output_prefix = args.output_prefix.expanduser().resolve()
    json_path = output_prefix.with_suffix(".json")
    npz_path = output_prefix.with_suffix(".npz")
    if json_path.exists() or npz_path.exists():
        raise FileExistsError(f"refusing to overwrite {json_path} or {npz_path}")

    build_args = base._build_args(
        base.DEFAULT_PARAMS,
        base.DEFAULT_PLANAR_MANIFEST,
        base.DEFAULT_REVERSE_MANIFEST,
        base.DEFAULT_GENERATED_ROOT,
    )
    simulator, bank, build_metadata = h5._build_simulator(build_args)
    (
        snapshot_data,
        snapshot_targets,
        _snapshot_history,
        snapshot_phase,
        basis_provenance,
        proposal_basis,
    ) = _build_trace_basis(
        simulator,
        seed=int(args.seed),
        start_tick=int(args.start_tick),
        horizon_ticks=HORIZON_TICKS,
    )
    prior_coefficients = np.tile(
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        (HORIZON_TICKS, 1),
    )
    rng = np.random.default_rng(int(args.seed) + 910_000)
    best_coefficients, plan_info = base._cem_plan(
        simulator,
        snapshot_data,
        snapshot_targets,
        prior_coefficients,
        rng,
        horizon_ticks=HORIZON_TICKS,
        block_ticks=BLOCK_TICKS,
        population=POPULATION,
        elites=ELITES,
        iterations=ITERATIONS,
        warm_std=STRUCTURED_STD,
        proposal_basis=proposal_basis,
    )
    best_cost, best_details = base._rollout_cost(
        simulator,
        snapshot_data,
        snapshot_targets,
        best_coefficients,
        horizon_ticks=HORIZON_TICKS,
        block_ticks=BLOCK_TICKS,
        collect_trace=True,
        proposal_basis=proposal_basis,
    )
    screen = _prescreen(best_details)
    if bank.legacy_fallback_count != 0:
        raise RuntimeError("structured MPC unexpectedly used a legacy H5 fallback")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        best_coefficients=np.asarray(best_coefficients, dtype=np.float32),
        proposal_trace_delta=np.asarray(proposal_basis["trace_delta"], dtype=np.float32),
        proposal_sagittal_delta=np.asarray(proposal_basis["sagittal_delta"], dtype=np.float32),
        proposal_roll_transfer=np.asarray(proposal_basis["roll_transfer"], dtype=np.float32),
        projected_velocity=np.asarray(best_details["local_velocities"], dtype=np.float32),
        normal_forces=np.asarray(best_details["normal_forces"], dtype=np.float32),
        tangential_speeds=np.asarray(best_details["tangential_speeds"], dtype=np.float32),
        contacts=np.asarray(best_details["contacts"], dtype=np.uint8),
    )
    payload = {
        "schema_version": 1,
        "evaluator_id": "openduckmini-exp004-h5-structured-mpc-reverse-prescreen-v1",
        "evaluation_mode": "DIAGNOSTIC_STRUCTURED_MPC_ONE_SHOT_NOT_QUALIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_deployment": "PROHIBITED",
        "configuration": {
            "seed": int(args.seed),
            "snapshot_start_tick": int(args.start_tick),
            "snapshot_phase_index": float(snapshot_phase),
            "horizon_ticks": HORIZON_TICKS,
            "block_ticks": BLOCK_TICKS,
            "population": POPULATION,
            "elites": ELITES,
            "iterations": ITERATIONS,
            "structured_parameterization": "8 blocks x [alpha_trace,beta_hip_roll,gamma_sagittal]",
            "target_guard": {
                "margin_rad": base.TARGET_MARGIN_RAD,
                "max_delta_per_tick_rad": base.TARGET_DELTA_RAD,
                "owner": "target_safety.FinalTargetSafetyGuard",
            },
            "physical_command": base.PHYSICAL_COMMAND.tolist(),
            "policy_command": base.POLICY_COMMAND.tolist(),
        },
        "provenance": {
            "teacher_script": str(Path(__file__).resolve()),
            "teacher_script_sha256": h5.sha256_file(Path(__file__).resolve()),
            "base_mpc_script": str(Path(base.__file__).resolve()),
            "base_mpc_script_sha256": h5.sha256_file(Path(base.__file__).resolve()),
            "h5_build_metadata": build_metadata,
            "proposal_basis": basis_provenance,
            "basis_evidence_sha256": h5.sha256_file(basis_evidence),
            "policy_bank": bank.manifest(),
        },
        "optimization": {
            "best_cost": float(best_cost),
            "best_coefficients": np.asarray(best_coefficients, dtype=np.float64).tolist(),
            "cem_iterations": _json_safe(plan_info["iterations"]),
            "best_details": _json_safe(best_details),
        },
        "prescreen": _json_safe(screen),
        "outputs": {
            "npz": {"path": str(npz_path), "sha256": h5.sha256_file(npz_path)},
            "json": {"path": str(json_path)},
        },
        "adoption_allowed": False,
        "release_allowed": False,
        "next_step": (
            "ALLOW_FULL_STRICT_CEM_ONLY_IF_PRESCREEN_PASSED"
            if screen["passed"]
            else "REJECT_STRUCTURED_MPC_TEACHER_CONFIGURATION"
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
                "next_step": payload["next_step"],
                "hardware_deployment": "PROHIBITED",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
