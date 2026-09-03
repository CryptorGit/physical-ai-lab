"""Trace and diagnose deterministic H3 exact-home startup transients.

The central evaluator, router, contracts, and package are never edited.  This
module wraps their process-local call sites to capture control-tick traces and
to test explicitly diagnostic, route-aware observation/phase bootstraps.  It
never injects physical base velocity; every physical target still passes
through the central final safety guard and every MuJoCo substep audit.
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


BASE_POLICY = (
    WORKSPACE / ".openduck_runtime_source_review" / "calibrated_hybrid_policy_v22.onnx"
)
DEFAULT_OUTPUT = EXP_ROOT / "artifacts" / "h3_exact_home_bootstrap_trace.json"
SUPPORTED_CASES = ("forward", "yaw_left")
H3_PREFIX_SCHEDULE: tuple[
    tuple[str, tuple[float, float, float], float, None, str, str], ...
] = (
    ("intro_stand", (0.0, 0.0, 0.0), 1.5, None, "stand", "stand"),
    ("forward", (0.05, 0.0, 0.0), 4.0, None, "forward", "forward"),
    ("stand_after_forward", (0.0, 0.0, 0.0), 5.0, None, "stand", "stand"),
    ("reverse", (-0.05, 0.0, 0.0), 4.5, None, "reverse", "reverse"),
    ("stand_after_reverse", (0.0, 0.0, 0.0), 5.0, None, "stand", "stand"),
    (
        "lateral_left",
        (0.0, 0.06, 0.0),
        3.5,
        None,
        "lateral_left",
        "lateral_left",
    ),
    (
        "stand_after_lateral_left",
        (0.0, 0.0, 0.0),
        5.0,
        None,
        "stand",
        "stand",
    ),
    (
        "lateral_right",
        (0.0, -0.06, 0.0),
        3.5,
        None,
        "lateral_right",
        "lateral_right",
    ),
    (
        "stand_after_lateral_right",
        (0.0, 0.0, 0.0),
        5.0,
        None,
        "stand",
        "stand",
    ),
    ("yaw_left", (0.0, 0.0, 0.30), 3.5, None, "yaw_left", "yaw_left"),
)
OBSERVATION_BLOCKS: Mapping[str, tuple[int, int]] = {
    "gyro": (0, 3),
    "accelerometer": (3, 6),
    "command": (6, 13),
    "joint_position": (13, 27),
    "joint_velocity": (27, 41),
    "action_history_0": (41, 55),
    "action_history_1": (55, 69),
    "action_history_2": (69, 83),
    "motor_targets": (83, 97),
    "contacts": (97, 99),
    "phase": (99, 101),
}


@dataclass(frozen=True)
class BootstrapCandidate:
    observation_mode: str = "baseline"
    route_entry_preincrement_phase: float | None = None
    activation_threshold_factor: float = 1.0

    @property
    def candidate_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class TraceContext:
    case_name: str
    candidate: BootstrapCandidate
    current_decision: Any | None = None
    current_requested_command: list[float] | None = None
    rows: list[dict[str, Any]] | None = None
    route_phase_reset_count: int = 0
    previous_activation_active: bool = False
    phase_reset_events: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.rows = []
        self.phase_reset_events = []


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _phase_values(value: str) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for item in _csv_strings(value):
        if item.lower() in ("none", "baseline"):
            parsed.append(None)
        else:
            try:
                number = float(item)
            except ValueError as exc:
                raise argparse.ArgumentTypeError("phase values must be finite or 'none'") from exc
            if not np.isfinite(number) or not 0.0 <= number < 20.0:
                raise argparse.ArgumentTypeError("phase values must be in [0, 20)")
            parsed.append(number)
    return tuple(parsed)


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in _csv_strings(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated finite numbers") from exc
    if not np.all(np.isfinite(parsed)):
        raise argparse.ArgumentTypeError("expected comma-separated finite numbers")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--schedule-mode",
        choices=("independent", "continuous", "h3_prefix"),
        default="h3_prefix",
    )
    parser.add_argument(
        "--policy-command-mode",
        choices=("h3_physical", "formal_override"),
        default="h3_physical",
    )
    parser.add_argument("--stand-seconds", type=float, default=4.0)
    parser.add_argument("--forward-motion-seconds", type=float, default=4.0)
    parser.add_argument("--yaw-left-motion-seconds", type=float, default=3.5)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--cases", type=_csv_strings, default=SUPPORTED_CASES)
    parser.add_argument(
        "--observation-modes",
        type=_csv_strings,
        default=("baseline",),
        help="baseline, route_blend_scaled, or route_delayed_full",
    )
    parser.add_argument(
        "--route-entry-phases",
        type=_phase_values,
        default=(None,),
        help="Comma-separated pre-increment phase indices or 'none'.",
    )
    parser.add_argument(
        "--activation-threshold-factors",
        type=_csv_floats,
        default=(1.0,),
        help="Factors on linear 0.02 or yaw 0.10 effective-command thresholds.",
    )
    parser.add_argument(
        "--candidate-exact-only",
        action="store_true",
        help="Run perturbed comparison only for the all-baseline candidate.",
    )
    parser.add_argument(
        "--trace-retention",
        choices=("full", "target_window"),
        default="target_window",
    )
    args = parser.parse_args(argv)
    durations = (
        (args.seconds,)
        if args.schedule_mode == "independent"
        else (args.stand_seconds, args.forward_motion_seconds, args.yaw_left_motion_seconds)
    )
    if any(value <= 0.0 for value in durations):
        parser.error("all active durations must be positive")
    if args.warmup_seconds < 0.0 or args.warmup_seconds >= min(durations):
        parser.error("warmup must be non-negative and shorter than every segment")
    if any(case not in SUPPORTED_CASES for case in args.cases):
        parser.error(f"--cases must be drawn from {SUPPORTED_CASES}")
    modes = {"baseline", "route_blend_scaled", "route_delayed_full"}
    if any(mode not in modes for mode in args.observation_modes):
        parser.error(f"observation modes must be drawn from {sorted(modes)}")
    if "baseline" not in args.observation_modes or None not in args.route_entry_phases:
        parser.error("candidate grid must retain baseline observation mode and phase 'none'")
    if any(value <= 0.0 for value in args.activation_threshold_factors):
        parser.error("activation threshold factors must be positive")
    return args


def candidate_grid(args: argparse.Namespace) -> tuple[BootstrapCandidate, ...]:
    return tuple(
        BootstrapCandidate(mode, phase, threshold)
        for mode in args.observation_modes
        for phase in args.route_entry_phases
        for threshold in args.activation_threshold_factors
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_family_activation(
    case_name: str,
    effective_command: Sequence[float],
    threshold_factor: float,
) -> tuple[bool, str, float, float, str]:
    """Return the route-scoped effective-command activation predicate."""

    command = np.asarray(effective_command, dtype=np.float64)
    factor = float(threshold_factor)
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("effective command must be a finite vx/vy/yaw triplet")
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("activation threshold factor must be finite and positive")
    if case_name == "forward":
        axis = "effective_vx"
        value = float(command[0])
        threshold = 0.02 * factor
        predicate = "effective_vx_gt_positive_threshold_false_to_true"
    elif case_name == "yaw_left":
        axis = "effective_yaw"
        value = float(command[2])
        threshold = 0.10 * factor
        predicate = "effective_yaw_gt_positive_threshold_false_to_true"
    else:
        raise ValueError(f"unsupported route family: {case_name}")
    return value > threshold, axis, threshold, value, predicate


def _motion_start_index(trace: Sequence[Mapping[str, Any]]) -> int:
    return next(
        (
            index
            for index, row in enumerate(trace)
            if bool(row.get("is_target_motion"))
        ),
        0,
    )


def _trace_summary(
    trace: list[dict[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    target_segment: Mapping[str, Any],
    *,
    expected_segment_count: int,
    phase_reset_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    motion_start_index = _motion_start_index(trace)
    startup = trace[motion_start_index : motion_start_index + 100]
    contacts = [tuple(bool(v) for v in row["contacts_after_step"]) for row in startup]
    contact_transitions = sum(a != b for a, b in zip(contacts, contacts[1:]))
    single_support_ticks = [
        index for index, contact in enumerate(contacts) if contact[0] != contact[1]
    ]
    expected = sum(int(item["expected_physics_substeps"]) for item in segments)
    completed = sum(int(item["completed_physics_substeps"]) for item in segments)
    audited = sum(int(item["physics_substep_audit"]["sample_count"]) for item in segments)
    segment_acceptances = {
        str(item["name"]): segment_acceptance(item) for item in segments
    }
    return {
        "control_tick_count": len(trace),
        "motion_start_trace_index": motion_start_index,
        "motion_start_control_tick": trace[motion_start_index]["control_tick"],
        "startup_tick_count": len(startup),
        "route_phase_reset_count": len(phase_reset_events),
        "route_phase_reset_events": list(phase_reset_events),
        "first_nonstand_tick": next(
            (row["control_tick"] for row in trace if row["expert"] != "stand"), None
        ),
        "first_single_support_motion_relative_tick": (
            single_support_ticks[0] if single_support_ticks else None
        ),
        "first_single_support_control_tick": (
            startup[single_support_ticks[0]]["control_tick"]
            if single_support_ticks
            else None
        ),
        "startup_single_support_tick_count": len(single_support_ticks),
        "startup_contact_transition_count": contact_transitions,
        "startup_minimum_upright": min(float(row["upright_after_step"]) for row in startup),
        "startup_maximum_raw_action_abs": max(
            max(abs(float(v)) for v in row["raw_action"]) for row in startup
        ),
        "startup_maximum_applied_action_abs": max(
            max(abs(float(v)) for v in row["applied_action"]) for row in startup
        ),
        "startup_maximum_applied_target_delta_rad": max(
            max(abs(float(v)) for v in row["applied_target_delta_rad"]) for row in startup
        ),
        "completed": bool(target_segment["completed"]),
        "fell": bool(target_segment["fell"]),
        "central_acceptance": segment_acceptance(target_segment),
        "segment_acceptances": segment_acceptances,
        "requested_segment_count": expected_segment_count,
        "completed_segment_count": len(segments),
        "expected_physics_substeps": expected,
        "audited_physics_substeps": audited,
        "all_physics_substeps_audited": audited == completed,
        "all_expected_substeps_completed": bool(
            len(segments) == expected_segment_count and expected == audited
        ),
        "qpos_violation_samples": sum(
            int(item["physics_substep_audit"]["qpos_limit_violations"])
            for item in segments
        ),
        "minimum_upright": min(
            float(item["physics_substep_audit"]["minimum_upright"])
            for item in segments
        ),
        "minimum_height_m": min(
            float(item["physics_substep_audit"]["minimum_height_m"])
            for item in segments
        ),
        "motion_metrics": target_segment["metrics"],
    }


def compare_traces(
    exact: Sequence[Mapping[str, Any]], perturbed: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    exact_start = _motion_start_index(exact)
    perturbed_start = _motion_start_index(perturbed)
    count = min(len(exact) - exact_start, len(perturbed) - perturbed_start, 100)
    block_norms: dict[str, list[float]] = {name: [] for name in OBSERVATION_BLOCKS}
    action_norms = []
    target_norms = []
    first_contact_divergence = None
    for index in range(count):
        left_row = exact[exact_start + index]
        right_row = perturbed[perturbed_start + index]
        left = np.asarray(left_row["observation"], dtype=np.float64)
        right = np.asarray(right_row["observation"], dtype=np.float64)
        for name, (start, stop) in OBSERVATION_BLOCKS.items():
            block_norms[name].append(float(np.linalg.norm(left[start:stop] - right[start:stop])))
        action_norms.append(
            float(
                np.linalg.norm(
                    np.asarray(left_row["applied_action"], dtype=np.float64)
                    - np.asarray(right_row["applied_action"], dtype=np.float64)
                )
            )
        )
        target_norms.append(
            float(
                np.linalg.norm(
                    np.asarray(left_row["applied_targets_rad"], dtype=np.float64)
                    - np.asarray(right_row["applied_targets_rad"], dtype=np.float64)
                )
            )
        )
        if (
            first_contact_divergence is None
            and left_row["contacts_after_step"] != right_row["contacts_after_step"]
        ):
            first_contact_divergence = index
    return {
        "compared_control_ticks": count,
        "exact_motion_start_control_tick": exact[exact_start]["control_tick"],
        "perturbed_motion_start_control_tick": perturbed[perturbed_start]["control_tick"],
        "first_contact_divergence_motion_relative_tick": first_contact_divergence,
        "observation_block_l2": {
            name: {
                "mean": float(np.mean(values)),
                "maximum": float(np.max(values)),
                "first_tick": values[0],
            }
            for name, values in block_norms.items()
        },
        "applied_action_l2": {
            "mean": float(np.mean(action_norms)),
            "maximum": float(np.max(action_norms)),
            "first_tick": action_norms[0],
        },
        "applied_target_l2_rad": {
            "mean": float(np.mean(target_norms)),
            "maximum": float(np.max(target_norms)),
            "first_tick": target_norms[0],
        },
    }


def retain_target_trace_window(run: dict[str, Any]) -> None:
    """Bound artifact size while retaining pre-entry and startup evidence."""

    trace = run["trace"]
    start = _motion_start_index(trace)
    retained_start = max(0, start - 20)
    retained_stop = min(len(trace), start + 120)
    run["trace_retention"] = {
        "mode": "target_window",
        "original_control_tick_count": len(trace),
        "retained_start_trace_index": retained_start,
        "retained_stop_trace_index_exclusive": retained_stop,
        "retained_control_tick_count": retained_stop - retained_start,
    }
    run["trace"] = trace[retained_start:retained_stop]


def run_traced(
    simulator: Any,
    bank: Any,
    evaluator: Any,
    case: Any,
    candidate: BootstrapCandidate,
    *,
    seed: int,
    schedule_mode: str,
    policy_command_mode: str,
    seconds: float,
    stand_seconds: float,
    motion_seconds: float,
    warmup_seconds: float,
    joint_noise_scale: float,
    initial_base_speed: float,
) -> dict[str, Any]:
    context = TraceContext(case.name, candidate)
    original_router = central.SafeGaitRouter
    original_resolve = central.resolve_policy_observation_command
    original_advance = central.advance_routed_phase
    original_step = central.apply_guarded_control_then_step_physics
    original_observation = evaluator._observation
    original_infer_route = bank.infer_route
    original_policy_target = simulator._policy_target

    class TracedRouter(original_router):
        def route(self, command: Sequence[float], dt: float) -> Any:
            decision = super().route(command, dt)
            context.current_decision = decision
            context.current_requested_command = np.asarray(
                command, dtype=np.float64
            ).tolist()
            return decision

    def traced_resolve(
        routed_expert: str,
        effective_command: Sequence[float],
        *,
        backward_residual_scale: float,
        override: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, float, bool]:
        command, yaw_offset, source = original_resolve(
            routed_expert,
            effective_command,
            backward_residual_scale=backward_residual_scale,
            override=override,
        )
        decision = context.current_decision
        if candidate.observation_mode == "baseline":
            return command, yaw_offset, source
        if decision is None:
            raise RuntimeError("route decision missing before observation command")
        target_role = case.expected_expert
        if decision.expert != target_role:
            scale = 0.0
        elif candidate.observation_mode == "route_blend_scaled":
            scale = float(decision.blend_alpha)
        elif candidate.observation_mode == "route_delayed_full":
            scale = float(decision.blend_alpha >= 1.0)
        else:
            raise RuntimeError("unsupported observation mode")
        return np.asarray(command, dtype=np.float64) * scale, yaw_offset * scale, True

    def traced_advance(phase_index: float, **kwargs: Any) -> tuple[float, bool, Any]:
        (
            activation_active,
            activation_axis,
            activation_threshold,
            activation_value,
            activation_predicate,
        ) = route_family_activation(
            case.name,
            kwargs["effective_command"],
            candidate.activation_threshold_factor,
        )
        target_requested = bool(
            context.current_requested_command is not None
            and np.array_equal(
                np.asarray(context.current_requested_command, dtype=np.float64),
                np.asarray(case.command, dtype=np.float64),
            )
        )
        activation_active = bool(
            activation_active
            and target_requested
            and kwargs["current_expert"] == case.expected_expert
        )
        should_reset = bool(
            candidate.route_entry_preincrement_phase is not None
            and context.route_phase_reset_count == 0
            and activation_active
            and not context.previous_activation_active
        )
        event: dict[str, Any] | None = None
        if should_reset:
            before_reset = float(phase_index)
            phase_index = float(candidate.route_entry_preincrement_phase)
            context.route_phase_reset_count += 1
            event = {
                "control_step": (
                    None
                    if kwargs.get("control_step") is None
                    else int(kwargs["control_step"])
                ),
                "global_control_tick": (
                    None
                    if kwargs.get("global_control_tick") is None
                    else int(kwargs["global_control_tick"])
                ),
                "previous_expert": kwargs.get("previous_expert"),
                "current_expert": kwargs["current_expert"],
                "effective_command": np.asarray(
                    kwargs["effective_command"], dtype=np.float64
                ).tolist(),
                "activation_axis": activation_axis,
                "activation_value": activation_value,
                "activation_threshold": activation_threshold,
                "activation_threshold_factor": candidate.activation_threshold_factor,
                "activation_predicate": activation_predicate,
                "previous_activation_active": False,
                "current_activation_active": True,
                "global_phase_index_before_reset": before_reset,
                "reset_preincrement_phase_index": phase_index,
                "phase_delta": float(kwargs["phase_delta"]),
                "first_used_phase_index": (
                    phase_index + float(kwargs["phase_delta"])
                )
                % float(kwargs["phase_steps"]),
                "phase_steps": float(kwargs["phase_steps"]),
                "status": "DIAGNOSTIC_ONLY_NOT_ADOPTED",
                "physical_base_velocity_injected": False,
            }
        result = original_advance(phase_index, **kwargs)
        if event is not None:
            event["advanced_phase_index"] = float(result[0])
            event["first_used_phase_matches_advance"] = bool(
                float(result[0]) == event["first_used_phase_index"]
            )
            context.phase_reset_events.append(event)
        context.previous_activation_active = activation_active
        context._last_phase_reset = should_reset  # type: ignore[attr-defined]
        return result

    def traced_observation(
        data: Any,
        command: np.ndarray,
        default_actuator: np.ndarray,
        motor_targets: np.ndarray,
        action_history: list[np.ndarray],
        phase: float,
    ) -> np.ndarray:
        observation = original_observation(
            data, command, default_actuator, motor_targets, action_history, phase
        )
        decision = context.current_decision
        if decision is None:
            raise RuntimeError("route decision missing before observation")
        row = {
            "control_tick": len(context.rows),
            "time_before_step_seconds": float(data.time),
            "expert": decision.expert,
            "blend_from_expert": decision.blend_from_expert,
            "blend_to_expert": decision.blend_to_expert,
            "blend_alpha": float(decision.blend_alpha),
            "switched": bool(decision.switched),
            "requested_command": list(context.current_requested_command or ()),
            "is_target_motion": bool(
                context.current_requested_command is not None
                and np.array_equal(
                    np.asarray(context.current_requested_command, dtype=np.float64),
                    np.asarray(case.command, dtype=np.float64),
                )
            ),
            "effective_command": list(decision.effective_command),
            "policy_observation_command": np.asarray(command[:3], dtype=float).tolist(),
            "phase_radians": float(phase),
            "phase_index": float(phase / (2.0 * np.pi) * evaluator.phase_steps),
            "route_phase_reset": bool(getattr(context, "_last_phase_reset", False)),
            "observation": observation.astype(float).tolist(),
            "contacts_before_step": evaluator._feet_contacts(data).astype(bool).tolist(),
            "joint_qpos_before_step_rad": np.asarray(
                data.qpos[evaluator.actuator_qpos_addr], dtype=float
            ).tolist(),
            "joint_qvel_before_step_rad_s": np.asarray(
                data.qvel[evaluator.actuator_qvel_addr], dtype=float
            ).tolist(),
            "motor_targets_before_step_rad": np.asarray(motor_targets, dtype=float).tolist(),
        }
        context.rows.append(row)
        return observation

    def traced_infer_route(decision: Any, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw, applied = original_infer_route(decision, observation)
        context.rows[-1]["raw_action"] = np.asarray(raw, dtype=float).tolist()
        context.rows[-1]["applied_action"] = np.asarray(applied, dtype=float).tolist()
        return raw, applied

    def traced_policy_target(
        applied_action: np.ndarray,
        effective_command: np.ndarray,
        phase_index: float,
        default: np.ndarray,
    ) -> np.ndarray:
        target = original_policy_target(applied_action, effective_command, phase_index, default)
        context.rows[-1]["candidate_targets_rad"] = np.asarray(target, dtype=float).tolist()
        return target

    def traced_step(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        result = original_step(*args, **kwargs)
        previous, applied, _, completed_substeps, terminated = result
        data = kwargs["data"]
        position = np.asarray(data.xpos[evaluator.trunk_body_id], dtype=float)
        rotation = np.asarray(data.xmat[evaluator.trunk_body_id], dtype=float).reshape(3, 3)
        row = context.rows[-1]
        row.update(
            {
                "completed_physics_substeps": int(completed_substeps),
                "substep_terminated": bool(terminated),
                "applied_targets_rad": np.asarray(applied, dtype=float).tolist(),
                "applied_target_delta_rad": (
                    np.asarray(applied, dtype=float) - np.asarray(previous, dtype=float)
                ).tolist(),
                "contacts_after_step": evaluator._feet_contacts(data).astype(bool).tolist(),
                "trunk_position_after_step_m": position.tolist(),
                "upright_after_step": float(rotation[2, 2]),
                "height_after_step_m": float(position[2]),
                "gyro_after_step_rad_s": np.asarray(
                    evaluator._sensor(data, "gyro"), dtype=float
                ).tolist(),
            }
        )
        return result

    central.SafeGaitRouter = TracedRouter
    central.resolve_policy_observation_command = traced_resolve
    central.advance_routed_phase = traced_advance
    central.apply_guarded_control_then_step_physics = traced_step
    evaluator._observation = traced_observation
    bank.infer_route = traced_infer_route
    simulator._policy_target = traced_policy_target
    policy_override = (
        case.policy_observation_command
        if policy_command_mode == "formal_override"
        else None
    )
    schedule = (
        (
            (
                case.name,
                case.command,
                seconds,
                policy_override,
                case.expected_expert,
                case.expected_policy_role,
            ),
        )
        if schedule_mode == "independent"
        else (
            (
                f"stand_before_{case.name}",
                (0.0, 0.0, 0.0),
                stand_seconds,
                None,
                "stand",
                "stand",
            ),
            (
                case.name,
                case.command,
                motion_seconds,
                policy_override,
                case.expected_expert,
                case.expected_policy_role,
            ),
        )
    )
    if schedule_mode == "h3_prefix":
        target_index = next(
            index
            for index, item in enumerate(H3_PREFIX_SCHEDULE)
            if item[0] == case.name
        )
        schedule = H3_PREFIX_SCHEDULE[: target_index + 1]
        if policy_command_mode == "formal_override":
            schedule = tuple(
                (
                    item[0],
                    item[1],
                    item[2],
                    (
                        case.policy_observation_command
                        if item[0] == case.name
                        else item[3]
                    ),
                    item[4],
                    item[5],
                )
                for item in schedule
            )
    try:
        result = simulator.run_schedule(
            schedule,
            seed=seed,
            joint_noise_scale=joint_noise_scale,
            initial_base_speed=initial_base_speed,
            warmup_seconds=warmup_seconds,
        )
    finally:
        central.SafeGaitRouter = original_router
        central.resolve_policy_observation_command = original_resolve
        central.advance_routed_phase = original_advance
        central.apply_guarded_control_then_step_physics = original_step
        evaluator._observation = original_observation
        bank.infer_route = original_infer_route
        simulator._policy_target = original_policy_target
    segments = result["segments"]
    segment = next(
        (item for item in segments if item["name"] == case.name), segments[-1]
    )
    return {
        "reset_seed": seed,
        "joint_noise_scale": joint_noise_scale,
        "initial_base_speed": initial_base_speed,
        "physical_base_velocity_injected": initial_base_speed > 0.0,
        "reset_qpos_audit": result["reset_qpos_audit"],
        "control_first_startup_audit": result["control_first_startup_audit"],
        "target_segment_present": any(
            item["name"] == case.name for item in segments
        ),
        "summary": _trace_summary(
            context.rows,
            segments,
            segment,
            expected_segment_count=len(schedule),
            phase_reset_events=context.phase_reset_events,
        ),
        "segments": segments,
        "segment": segment,
        "trace": context.rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact: {output}")
    candidates = candidate_grid(args)
    baseline_candidate = next(
        candidate
        for candidate in candidates
        if candidate.observation_mode == "baseline"
        and candidate.route_entry_preincrement_phase is None
    )
    mujoco, onnxruntime, runtime, provenance = central._load_runtime(include_provenance=True)
    asset_paths = central.generated_asset_paths(central.DEFAULT_GENERATED_ROOT.resolve())
    policy_paths = {role: BASE_POLICY.resolve() for role in central.REQUIRED_POLICY_ROLES}
    bank = central.RoutedPolicyBank(policy_paths, onnxruntime)
    evaluator = runtime.OfficialPolicyEvaluator(
        asset_paths["scene"], BASE_POLICY.resolve(), asset_paths["reference"]
    )
    evaluator.backward_residual_scale = 0.0
    evaluator.load_backward_profile(central.DEFAULT_BACKWARD_PROFILE)
    evaluator.load_backward_turn_profile(1, central.DEFAULT_BACKWARD_LEFT_PROFILE)
    evaluator.load_backward_turn_profile(-1, central.DEFAULT_BACKWARD_RIGHT_PROFILE)
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0
    simulator = central.RoutedSimulator(
        evaluator,
        bank,
        mujoco,
        runtime,
        leg_target_margin_rad=central.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
        target_slew_rate_rad_s=central.RUNTIME_TARGET_SLEW_RATE_RAD_S,
        diagnostic_noncontract_safety=False,
        left_knee_extra_upper_margin_rad=0.0125,
        formal_candidate_default=True,
    )
    records: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    for case_name in args.cases:
        case = next(item for item in central.PRIMITIVE_CASES if item.name == case_name)
        seed = 20_260_808 + list(central.PRIMITIVE_CASES).index(case)
        case_records = []
        for candidate in candidates:
            motion_seconds = (
                args.forward_motion_seconds
                if case_name == "forward"
                else args.yaw_left_motion_seconds
            )
            exact = run_traced(
                simulator,
                bank,
                evaluator,
                case,
                candidate,
                seed=seed,
                schedule_mode=args.schedule_mode,
                policy_command_mode=args.policy_command_mode,
                seconds=args.seconds,
                stand_seconds=args.stand_seconds,
                motion_seconds=motion_seconds,
                warmup_seconds=args.warmup_seconds,
                joint_noise_scale=0.0,
                initial_base_speed=0.0,
            )
            record = {
                "case": case_name,
                "candidate": asdict(candidate),
                "candidate_id": candidate.candidate_id,
                "exact_home": exact,
            }
            is_baseline = candidate == baseline_candidate
            if is_baseline or not args.candidate_exact_only:
                perturbed = run_traced(
                    simulator,
                    bank,
                    evaluator,
                    case,
                    candidate,
                    seed=seed,
                    schedule_mode=args.schedule_mode,
                    policy_command_mode=args.policy_command_mode,
                    seconds=args.seconds,
                    stand_seconds=args.stand_seconds,
                    motion_seconds=motion_seconds,
                    warmup_seconds=args.warmup_seconds,
                    joint_noise_scale=1.0,
                    initial_base_speed=0.10,
                )
                record["perturbed_formal"] = perturbed
                record["exact_vs_perturbed"] = compare_traces(
                    exact["trace"], perturbed["trace"]
                )
            if args.trace_retention == "target_window":
                retain_target_trace_window(exact)
                if "perturbed_formal" in record:
                    retain_target_trace_window(record["perturbed_formal"])
            case_records.append(record)
        records.extend(case_records)
        baseline = next(
            record
            for record in case_records
            if record["candidate"]["observation_mode"] == "baseline"
            and record["candidate"]["route_entry_preincrement_phase"] is None
            and record["candidate_id"] == baseline_candidate.candidate_id
        )
        comparisons[case_name] = baseline["exact_vs_perturbed"]
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_h3_exact_home_bootstrap_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY_NOT_ADOPTED",
        "hardware_deployment": "PROHIBITED",
        "physical_base_velocity_injection_for_exact_home_candidates": False,
        "configuration": {
            "cases": list(args.cases),
            "schedule_mode": args.schedule_mode,
            "policy_command_mode": args.policy_command_mode,
            "seconds": args.seconds,
            "stand_seconds": args.stand_seconds,
            "forward_motion_seconds": args.forward_motion_seconds,
            "yaw_left_motion_seconds": args.yaw_left_motion_seconds,
            "warmup_seconds": args.warmup_seconds,
            "activation_threshold_bases": {
                "forward_effective_vx": 0.02,
                "yaw_left_effective_yaw": 0.10,
            },
            "candidate_count": len(candidates),
            "candidate_exact_only": args.candidate_exact_only,
            "trace_retention": args.trace_retention,
        },
        "central_semantics": {
            "target_margin_rad": central.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
            "target_slew_rate_rad_per_s": central.RUNTIME_TARGET_SLEW_RATE_RAD_S,
            "control_dt_seconds": runtime.CONTROL_DT,
            "sim_dt_seconds": runtime.SIM_DT,
            "decimation": runtime.DECIMATION,
            "all_physics_substeps_audited": True,
        },
        "dependencies": {
            "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
            "central_evaluator_sha256": _sha256(Path(central.__file__).resolve()),
            "policy_sha256": _sha256(BASE_POLICY.resolve()),
            "runtime": provenance,
            "onnx_providers": bank.session_providers,
        },
        "baseline_exact_vs_perturbed": comparisons,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    concise = [
        {
            "case": record["case"],
            "candidate_id": record["candidate_id"],
            "candidate": record["candidate"],
            "exact_summary": record["exact_home"]["summary"],
        }
        for record in records
    ]
    print(json.dumps({"output": str(output), "records": concise}, indent=2))


if __name__ == "__main__":
    main()
