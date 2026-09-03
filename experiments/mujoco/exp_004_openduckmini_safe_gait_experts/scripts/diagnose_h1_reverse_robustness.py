"""Isolated H1 reverse-robustness sweep using the central runtime semantics.

This diagnostic imports the routed evaluator without modifying its contract,
package, or release state.  Every MuJoCo substep is audited by the central
``PhysicsSubstepAudit`` implementation.  Outputs are simulation-only evidence;
hardware deployment is always prohibited.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import evaluate_routed_transitions as central  # noqa: E402
from safe_gait_experts.routed_evaluation import segment_acceptance  # noqa: E402


BASE_PROFILE = EXP_ROOT / "artifacts" / "optimized_reverse_margin050_slew200_candidate_v3.json"
LEFT_PROFILE = (
    EXP_ROOT
    / "artifacts"
    / "reverse_turn_candidates_v1"
    / "optimized_reverse_turn_left_margin050_slew200_candidate_v1.json"
)
RIGHT_PROFILE = (
    EXP_ROOT
    / "artifacts"
    / "reverse_turn_candidates_v1"
    / "optimized_reverse_turn_right_margin050_slew200_candidate_v1.json"
)
BASE_POLICY = WORKSPACE / ".openduck_runtime_source_review" / "calibrated_hybrid_policy_v22.onnx"
DEFAULT_OUTPUT = EXP_ROOT / "artifacts" / "h1_reverse_robustness_diagnostic.json"
REVERSE_CASE_INDEX = 2
FORMAL_MASTER_SEED = 20_260_808
FORMAL_REVERSE_SIMULATION_SEEDS = tuple(
    FORMAL_MASTER_SEED + episode_index * 1000 + REVERSE_CASE_INDEX
    for episode_index in range(20)
)
FORMAL_TRANSITION_SIMULATION_SEEDS = tuple(
    FORMAL_MASTER_SEED + 2_000_000 + episode_index for episode_index in range(20)
)


@dataclass(frozen=True)
class Candidate:
    left_knee_extra_upper_margin_rad: float
    reverse_entry_phase_index: float
    phase_rate_factor: float
    left_knee_amplitude_factor: float
    left_knee_bias_delta_rad: float

    @property
    def candidate_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated finite numbers") from exc
    if not parsed or not np.all(np.isfinite(parsed)):
        raise argparse.ArgumentTypeError("expected comma-separated finite numbers")
    return parsed


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--initial-joint-noise-scale", type=float, default=1.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.10)
    parser.add_argument(
        "--simulation-seeds",
        type=_csv_ints,
        default=(20_273_810,),
        help="Exact simulator reset seeds, not episode labels.",
    )
    parser.add_argument(
        "--formal-reverse-20-seeds",
        action="store_true",
        help="Use the exact 20 primitive-reverse simulator seeds from seed 20260808.",
    )
    parser.add_argument(
        "--formal-transition-20-seeds",
        action="store_true",
        help="Use the exact 20 transition reset seeds from seed 20260808.",
    )
    parser.add_argument("--caps", type=_csv_floats, default=(0.0125,))
    parser.add_argument("--phase-entries", type=_csv_floats, default=(6.0,))
    parser.add_argument("--phase-rate-factors", type=_csv_floats, default=(1.0,))
    parser.add_argument("--left-knee-amplitude-factors", type=_csv_floats, default=(1.0,))
    parser.add_argument("--left-knee-bias-deltas", type=_csv_floats, default=(0.0,))
    parser.add_argument(
        "--schedule",
        choices=(
            "reverse",
            "reverse_to_stand",
            "reverse_to_forward",
            "formal_reverse_prefix",
            "formal_reverse_left_prefix",
        ),
        default="reverse",
    )
    parser.add_argument("--exit-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.formal_reverse_20_seeds and args.formal_transition_20_seeds:
        parser.error("formal reverse and transition seed sets are mutually exclusive")
    if args.seconds <= 0.0 or args.exit_seconds <= 0.0:
        parser.error("durations must be positive")
    if args.warmup_seconds < 0.0 or args.warmup_seconds >= min(args.seconds, args.exit_seconds):
        parser.error("warmup must be non-negative and shorter than every segment")
    if args.initial_joint_noise_scale < 0.0 or args.initial_base_speed < 0.0:
        parser.error("initial perturbations must be non-negative")
    if any(not 0.0 <= value <= central.MAX_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD for value in args.caps):
        parser.error("caps must remain within the central diagnostic bound")
    if any(not 0.0 <= value < 20.0 for value in args.phase_entries):
        parser.error("phase entries must remain in [0, 20)")
    if any(value <= 0.0 for value in args.phase_rate_factors):
        parser.error("phase-rate factors must be positive")
    if any(value <= 0.0 for value in args.left_knee_amplitude_factors):
        parser.error("left-knee amplitude factors must be positive")
    return args


def candidate_grid(args: argparse.Namespace) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(cap, phase, rate, amplitude, bias)
        for cap in args.caps
        for phase in args.phase_entries
        for rate in args.phase_rate_factors
        for amplitude in args.left_knee_amplitude_factors
        for bias in args.left_knee_bias_deltas
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_profile(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _schedule(args: argparse.Namespace) -> tuple[tuple[Any, ...], ...]:
    if args.schedule in ("formal_reverse_prefix", "formal_reverse_left_prefix"):
        formal = central.transition_schedule(args.seconds, args.exit_seconds)
        stop = 5 if args.schedule == "formal_reverse_prefix" else 7
        return tuple(formal[:stop])
    reverse = next(case for case in central.PRIMITIVE_CASES if case.name == "reverse")
    entries: list[tuple[Any, ...]] = [
        (
            reverse.name,
            reverse.command,
            args.seconds,
            reverse.policy_observation_command,
            reverse.expected_expert,
            reverse.expected_policy_role,
        )
    ]
    if args.schedule == "reverse_to_stand":
        entries.append(("stand_after_reverse", (0.0, 0.0, 0.0), args.exit_seconds))
    elif args.schedule == "reverse_to_forward":
        forward = next(case for case in central.PRIMITIVE_CASES if case.name == "forward")
        entries.append(
            (
                "forward_after_reverse",
                forward.command,
                args.exit_seconds,
                forward.policy_observation_command,
                forward.expected_expert,
                forward.expected_policy_role,
            )
        )
    return tuple(entries)


def _apply_candidate(evaluator: Any, baseline: dict[str, Any], candidate: Candidate) -> None:
    parameters = baseline["parameters"]
    scales = np.asarray(parameters["joint_amplitude_scales"], dtype=np.float64)
    biases = np.asarray(parameters["joint_bias_offsets"], dtype=np.float64)
    scales[3] *= candidate.left_knee_amplitude_factor
    biases[3] += candidate.left_knee_bias_delta_rad
    evaluator.gait_scales = scales
    evaluator.gait_biases = biases
    evaluator.backward_phase_rate = float(parameters["phase_rate"]) * candidate.phase_rate_factor


def advance_routed_phase_candidate(
    phase_index: float,
    *,
    phase_steps: float,
    phase_delta: float,
    current_expert: str,
    previous_expert: str | None,
    effective_command: Sequence[float],
    previous_backward_feedforward_active: bool,
    diagnostic_entry_phase_indices: Mapping[str, float] | None = None,
    phase_entry_status: str = "DIAGNOSTIC_H1_REVERSE_ROBUSTNESS",
    diagnostic_only: bool = True,
    control_step: int | None = None,
    global_control_tick: int | None = None,
) -> tuple[float, bool, dict[str, Any] | None]:
    """Central phase semantics with only the frozen-value gate relaxed.

    The release helper rejects anything other than 6/4/4 before executing its
    otherwise generic algorithm.  This isolated equivalent retains every
    validation, activation, increment, and audit field while allowing the
    straight entry phase to be measured as a diagnostic candidate.
    """

    current = float(phase_index)
    count = float(phase_steps)
    delta = float(phase_delta)
    command = np.asarray(effective_command, dtype=np.float64)
    if (
        not np.isfinite(current)
        or not np.isfinite(count)
        or count <= 0.0
        or not np.isfinite(delta)
        or delta <= 0.0
    ):
        raise ValueError("phase index, phase steps, and phase delta must be finite")
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("effective command must be a finite vx/vy/yaw triplet")
    if not isinstance(phase_entry_status, str) or not phase_entry_status:
        raise ValueError("phase-entry status must be a non-empty string")
    if not isinstance(diagnostic_only, (bool, np.bool_)):
        raise ValueError("phase-entry diagnostic_only must be boolean")
    mapping = None
    if diagnostic_entry_phase_indices is not None:
        if set(diagnostic_entry_phase_indices) != set(central.BACKWARD_FAMILY_EXPERTS):
            raise ValueError("candidate phase mapping must cover the backward family exactly")
        mapping = {name: float(value) for name, value in diagnostic_entry_phase_indices.items()}
        if not np.all(np.isfinite(list(mapping.values()))) or any(
            not 0.0 <= value < count for value in mapping.values()
        ):
            raise ValueError("candidate phase mapping must remain finite and inside the period")
    backward_feedforward_active = bool(command[0] < -0.02)
    event: dict[str, Any] | None = None
    if (
        mapping is not None
        and backward_feedforward_active
        and not bool(previous_backward_feedforward_active)
        and current_expert in mapping
    ):
        before_reset = current
        current = mapping[current_expert]
        first_feedforward_phase = (current + delta) % count
        event = {
            "control_step": None if control_step is None else int(control_step),
            "global_control_tick": None if global_control_tick is None else int(global_control_tick),
            "previous_expert": previous_expert,
            "current_expert": current_expert,
            "effective_command": command.tolist(),
            "activation_predicate": "effective_vx_lt_negative_0p02_false_to_true",
            "previous_backward_feedforward_active": False,
            "current_backward_feedforward_active": True,
            "global_phase_index_before_reset": before_reset,
            "reset_preincrement_phase_index": current,
            "profile_phase_rate": delta,
            "first_feedforward_phase_index": first_feedforward_phase,
            "phase_steps": count,
            "status": phase_entry_status,
            "formal_candidate": False,
            "adopted_simulation_only": not bool(diagnostic_only),
            "diagnostic_only": bool(diagnostic_only),
        }
    advanced = (current + delta) % count
    if event is not None and advanced != event["first_feedforward_phase_index"]:
        raise RuntimeError("phase-entry reset produced inconsistent phase")
    return advanced, backward_feedforward_active, event


def _make_simulator(
    evaluator: Any,
    bank: Any,
    mujoco: Any,
    runtime: Any,
    candidate: Candidate,
) -> Any:
    # Construct through the exact Stage-A path, then override only the isolated
    # candidate knobs.  Target generation, one final guard, slew, MuJoCo
    # decimation, and every-substep audits remain central implementations.
    simulator = central.RoutedSimulator(
        evaluator,
        bank,
        mujoco,
        runtime,
        leg_target_margin_rad=central.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
        target_slew_rate_rad_s=central.RUNTIME_TARGET_SLEW_RATE_RAD_S,
        diagnostic_noncontract_safety=False,
        left_knee_extra_upper_margin_rad=central.BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
        formal_candidate_default=True,
    )
    simulator.left_knee_extra_upper_margin_rad = candidate.left_knee_extra_upper_margin_rad
    simulator.left_knee_profile_upper_target_rad = (
        float(central.SAFE_JOINT_LIMITS["left_knee"][1])
        - central.RUNTIME_TARGET_SAFETY_MARGIN_RAD
        - candidate.left_knee_extra_upper_margin_rad
    )
    simulator.reverse_entry_phase_indices = {
        "reverse": candidate.reverse_entry_phase_index,
        "reverse_turn_left": 4.0,
        "reverse_turn_right": 4.0,
    }
    simulator.phase_entry_status = "DIAGNOSTIC_H1_REVERSE_ROBUSTNESS"
    simulator.phase_entry_diagnostic_only = True
    return simulator


def _hard_checks(segment: dict[str, Any]) -> dict[str, bool]:
    safety = segment["safety_audit"]
    substeps = segment["physics_substep_audit"]
    return {
        "completed": bool(segment["completed"]),
        "no_fall": not bool(segment["fell"]),
        "all_expected_substeps_audited": (
            int(substeps["sample_count"]) == int(segment["expected_physics_substeps"])
            == int(segment["completed_physics_substeps"])
        ),
        "control_qpos_zero": int(safety["qpos_limit_violations"]) == 0,
        "substep_qpos_zero": int(substeps["qpos_limit_violations"]) == 0,
        "control_nonfinite_zero": int(safety["nonfinite_sample_count"]) == 0,
        "substep_nonfinite_zero": int(substeps["nonfinite_state_samples"]) == 0,
        "target_limit_zero": int(safety["applied_target_limit_violations"]) == 0,
        "target_margin_zero": int(safety["unauthorized_applied_target_margin_violations"]) == 0,
        "target_slew_zero": int(safety["target_slew_violations"]) == 0,
        "contact_count_exact": (
            int(substeps["contact_sample_count"]) == int(substeps["sample_count"])
            and bool(substeps["contact_sample_count_matches_sample_count"])
        ),
    }


def _summarize_candidate(candidate: Candidate, runs: list[dict[str, Any]]) -> dict[str, Any]:
    segments = [segment for run in runs for segment in run["segments"]]
    hard = [_hard_checks(segment) for segment in segments]
    acceptances = [segment_acceptance(segment) for segment in segments]
    reverse_segments = [
        segment for segment in segments if segment.get("expected_expert") == "reverse"
    ]
    total_expected = sum(int(segment["expected_physics_substeps"]) for segment in segments)
    total_audited = sum(int(segment["physics_substep_audit"]["sample_count"]) for segment in segments)
    progress_ratios = [
        float(segment["metrics"]["projected_primary_velocity"])
        / float(segment["metrics"]["commanded_linear_speed"])
        for segment in reverse_segments
    ]
    return {
        "candidate_id": candidate.candidate_id,
        "parameters": asdict(candidate),
        "left_knee_profile_upper_target_rad": (
            float(central.SAFE_JOINT_LIMITS["left_knee"][1])
            - central.RUNTIME_TARGET_SAFETY_MARGIN_RAD
            - candidate.left_knee_extra_upper_margin_rad
        ),
        "run_count": len(runs),
        "segment_count": len(segments),
        "fall_count": sum(bool(segment["fell"]) for segment in segments),
        "qpos_violation_samples": sum(
            int(segment["physics_substep_audit"]["qpos_limit_violations"])
            for segment in segments
        ),
        "all_hard_checks_passed": all(all(check.values()) for check in hard),
        "central_segment_acceptance_count": sum(item["passed"] for item in acceptances),
        "central_segment_count": len(acceptances),
        "expected_physics_substeps": total_expected,
        "audited_physics_substeps": total_audited,
        "all_expected_physics_substeps_audited": total_expected == total_audited,
        "minimum_upright": min(
            float(segment["physics_substep_audit"]["minimum_upright"])
            for segment in segments
        ),
        "minimum_height_m": min(
            float(segment["physics_substep_audit"]["minimum_height_m"])
            for segment in segments
        ),
        "worst_signed_linear_progress_fraction": min(progress_ratios),
        "worst_absolute_orthogonal_velocity_mps": max(
            float(segment["metrics"]["absolute_orthogonal_velocity"])
            for segment in reverse_segments
        ),
        "worst_uncommanded_yaw_rate_radps": max(
            float(segment["metrics"]["uncommanded_yaw_rate"])
            for segment in reverse_segments
        ),
        "minimum_single_support_rate": min(
            float(segment["physics_substep_audit"]["single_support_rate"])
            for segment in reverse_segments
        ),
        "maximum_flight_rate": max(
            float(segment["physics_substep_audit"]["flight_rate"])
            for segment in reverse_segments
        ),
        "hard_checks": hard,
        "central_segment_acceptances": acceptances,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact: {output}")
    candidates = candidate_grid(args)
    # Process-local diagnostic substitution; the central source file and its
    # frozen contract remain untouched.
    central.advance_routed_phase = advance_routed_phase_candidate
    seeds = (
        FORMAL_REVERSE_SIMULATION_SEEDS
        if args.formal_reverse_20_seeds
        else FORMAL_TRANSITION_SIMULATION_SEEDS
        if args.formal_transition_20_seeds
        else args.simulation_seeds
    )
    baseline = _read_profile(BASE_PROFILE)
    asset_paths = central.generated_asset_paths(central.DEFAULT_GENERATED_ROOT.resolve())
    mujoco, onnxruntime, runtime, runtime_provenance = central._load_runtime(include_provenance=True)
    policy_paths = {role: BASE_POLICY.resolve() for role in central.REQUIRED_POLICY_ROLES}
    bank = central.RoutedPolicyBank(policy_paths, onnxruntime)
    evaluator = runtime.OfficialPolicyEvaluator(
        asset_paths["scene"], BASE_POLICY.resolve(), asset_paths["reference"]
    )
    evaluator.backward_residual_scale = 0.0
    evaluator.load_backward_profile(BASE_PROFILE)
    evaluator.load_backward_turn_profile(1, LEFT_PROFILE)
    evaluator.load_backward_turn_profile(-1, RIGHT_PROFILE)
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0
    schedule = _schedule(args)
    candidate_records = []
    for candidate in candidates:
        _apply_candidate(evaluator, baseline, candidate)
        simulator = _make_simulator(evaluator, bank, mujoco, runtime, candidate)
        runs = [
            simulator.run_schedule(
                schedule,
                seed=seed,
                joint_noise_scale=args.initial_joint_noise_scale,
                initial_base_speed=args.initial_base_speed,
                warmup_seconds=args.warmup_seconds,
            )
            for seed in seeds
        ]
        candidate_records.append(
            {
                "summary": _summarize_candidate(candidate, runs),
                "runs": runs,
            }
        )
    ranked = sorted(
        (record["summary"] for record in candidate_records),
        key=lambda summary: (
            not summary["all_hard_checks_passed"],
            summary["fall_count"],
            summary["qpos_violation_samples"],
            -summary["central_segment_acceptance_count"],
            -summary["minimum_upright"],
            -summary["worst_signed_linear_progress_fraction"],
        ),
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_h1_reverse_robustness_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY_NOT_ADOPTED",
        "hardware_deployment": "PROHIBITED",
        "central_semantics": {
            "control_order": "observe_route_policy_profile_recovery_final_guard_ctrl_all_physics_substeps_audit",
            "control_dt_seconds": runtime.CONTROL_DT,
            "sim_dt_seconds": runtime.SIM_DT,
            "decimation": runtime.DECIMATION,
            "target_margin_rad": central.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
            "target_slew_rate_rad_per_s": central.RUNTIME_TARGET_SLEW_RATE_RAD_S,
            "backward_exit_recovery_enabled": True,
            "backward_exit_recovery_hold_ticks": central.BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
            "backward_exit_recovery_cap_rad": central.BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
            "all_mujoco_substeps_audited": True,
        },
        "configuration": {
            "simulation_seeds": list(seeds),
            "schedule": args.schedule,
            "seconds": args.seconds,
            "exit_seconds": args.exit_seconds,
            "warmup_seconds": args.warmup_seconds,
            "initial_joint_noise_scale": args.initial_joint_noise_scale,
            "initial_base_speed": args.initial_base_speed,
            "candidate_count": len(candidates),
        },
        "dependencies": {
            "central_evaluator": {"path": str(Path(central.__file__).resolve()), "sha256": _sha256(Path(central.__file__).resolve())},
            "base_profile": {"path": str(BASE_PROFILE.resolve()), "sha256": _sha256(BASE_PROFILE)},
            "left_profile": {"path": str(LEFT_PROFILE.resolve()), "sha256": _sha256(LEFT_PROFILE)},
            "right_profile": {"path": str(RIGHT_PROFILE.resolve()), "sha256": _sha256(RIGHT_PROFILE)},
            "policy": {"path": str(BASE_POLICY.resolve()), "sha256": _sha256(BASE_POLICY)},
            "runtime": runtime_provenance,
            "onnx_providers": bank.session_providers,
        },
        "ranking": ranked,
        "candidates": candidate_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "ranking": ranked}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
