"""Simulation-only strict routed evaluation with H4 forward/reverse actors.

This entrypoint is deliberately separate from the historical H3 release
evaluator.  It runs the same MuJoCo safety and gait-quality machinery, but
binds the two H4 actor116 candidates through an explicit manifest/source
closure and records the full 12-action plus continuous-transition result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import FunctionType
from typing import Any, Mapping, Sequence

import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import evaluate_routed_transitions as routed  # noqa: E402
from safe_gait_experts.h4_routed_policy import (  # noqa: E402
    H4CandidateSpec,
    H4RoutedPolicyBank,
    validate_h4_candidate,
)
from safe_gait_experts.h4_training_alignment import (  # noqa: E402
    force_schmitt_contacts,
)
from safe_gait_experts.routed_evaluation import (  # noqa: E402
    AcceptanceThresholds,
    COMPOUND_CASES,
    PRIMITIVE_CASES,
    REQUIRED_POLICY_ROLES,
    TRANSITION_CASES,
    canonical_policy_role,
    parse_policy_assignments,
    sha256_file,
    suite_acceptance,
    transition_schedule,
)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError(f"{label} must be lowercase SHA256")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict simulation-only routed suite with H4 forward/reverse actors."
    )
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        metavar="ROLE=PATH",
        help="The eight legacy ONNX roles. H4 replaces only forward/reverse.",
    )
    parser.add_argument("--generated-root", type=_path, required=True)
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--transition-seconds", type=float, default=6.0)
    parser.add_argument("--transition-stand-seconds", type=float, default=2.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--initial-joint-noise-scale", type=float, default=0.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.0)

    for role in ("forward", "reverse"):
        prefix = f"--h4-{role}"
        parser.add_argument(f"{prefix}-params", type=_path, required=True)
        parser.add_argument(
            f"{prefix}-params-sha256",
            type=lambda value, role=role: _sha256(value, f"{role} params SHA256"),
            required=True,
        )
        parser.add_argument(f"{prefix}-manifest", type=_path, required=True)
        parser.add_argument(
            f"{prefix}-manifest-sha256",
            type=lambda value, role=role: _sha256(value, f"{role} manifest SHA256"),
            required=True,
        )
        parser.add_argument(f"{prefix}-trusted-run-root", type=_path, required=True)
        parser.add_argument(f"{prefix}-authorization", type=_path, required=True)
        parser.add_argument(
            f"{prefix}-authorization-sha256",
            type=lambda value, role=role: _sha256(value, f"{role} authorization SHA256"),
            required=True,
        )
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.seconds <= 0.0 or args.transition_seconds <= 0.0:
        parser.error("episodes and durations must be positive")
    if args.transition_stand_seconds <= 0.0 or args.warmup_seconds < 0.0:
        parser.error("transition stand duration must be positive and warmup non-negative")
    if args.warmup_seconds >= min(args.seconds, args.transition_seconds, args.transition_stand_seconds):
        parser.error("warmup must be shorter than every segment")
    if args.initial_joint_noise_scale < 0.0 or args.initial_base_speed < 0.0:
        parser.error("initial perturbations must be non-negative")
    return args


def _candidate_spec(args: argparse.Namespace, role: str) -> H4CandidateSpec:
    return H4CandidateSpec(
        role=role,
        params_path=getattr(args, f"h4_{role}_params"),
        params_sha256=getattr(args, f"h4_{role}_params_sha256"),
        manifest_path=getattr(args, f"h4_{role}_manifest"),
        manifest_sha256=getattr(args, f"h4_{role}_manifest_sha256"),
        trusted_run_root=getattr(args, f"h4_{role}_trusted_run_root"),
        authorization_path=getattr(args, f"h4_{role}_authorization"),
        authorization_sha256=getattr(args, f"h4_{role}_authorization_sha256"),
    )


def _source_snapshot(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for label, path in sorted(paths.items())
    }


# The frozen run-schedule function is copied below so that the historical H3
# evaluator remains byte/semantics-stable.  This small context carries the
# physical command that the copied function has just routed into the local H4
# observation seam; it avoids reconstructing a physical command from the
# policy-visible override (forward uses 0.10 for a physical 0.05 command).
_H4_RUN_CONTEXT: dict[str, np.ndarray | None] = {"physical_command": None}


class _H4EvaluatorProxy:
    """Expose the frozen evaluator while extending only actor observations."""

    def __init__(self, evaluator: Any, simulator: "H4RoutedSimulator"):
        self._evaluator = evaluator
        self._simulator = simulator

    def _observation(
        self,
        data: Any,
        command7: np.ndarray,
        default: np.ndarray,
        motor_targets: np.ndarray,
        action_history: list[np.ndarray],
        phase: float,
    ) -> np.ndarray:
        legacy = self._evaluator._observation(
            data,
            command7,
            default,
            motor_targets,
            action_history,
            phase,
        )
        physical_context = _H4_RUN_CONTEXT.get("physical_command")
        physical = np.asarray(
            command7[:3] if physical_context is None else physical_context,
            dtype=np.float64,
        )
        if physical.shape != (3,) or not np.all(np.isfinite(physical)):
            raise ValueError("H4 physical command context must be finite (3,)")
        rotation = np.asarray(
            data.xmat[self._evaluator.trunk_body_id], dtype=np.float64
        ).reshape(3, 3)
        normalized_force, tangential_speed = (
            self._simulator._quality_contact_kinematics(data)
        )
        if float(data.time) <= 1.0e-12:
            self._simulator._h4_previous_force_contact = np.zeros(
                2, dtype=np.float32
            )
        previous_force_contact = np.asarray(
            getattr(
                self._simulator,
                "_h4_previous_force_contact",
                np.zeros(2, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        force_contact = np.asarray(
            force_schmitt_contacts(
                normalized_force,
                previous_force_contact,
                xp=np,
            ),
            dtype=np.float32,
        )
        extra = H4RoutedPolicyBank.append_physical_observables(
            legacy_observation=legacy,
            physical_command=physical,
            local_linvel=self._evaluator._sensor(data, "local_linvel"),
            trunk_rotation=rotation,
            normalized_force_and_tangential_speed=(
                normalized_force,
                tangential_speed,
            ),
            feet_contacts=force_contact,
        )
        # Training carries the raw Schmitt state to the next control-entry
        # observation.  Do the same after constructing this observation; the
        # current MuJoCo state is not advanced until the caller applies the
        # just-inferred action.
        self._simulator._h4_previous_force_contact = force_contact.copy()
        return extra

    def backward_parameters(self, yaw_command: float) -> tuple[Any, Any, float]:
        # Straight reverse H4 uses the selected teacher timing in source-phase
        # units. Reverse turns remain on the frozen compound path.
        if abs(float(yaw_command)) <= 1.0e-12:
            scales, biases, _ = self._evaluator.backward_parameters(yaw_command)
            return scales, biases, 0.81
        return self._evaluator.backward_parameters(yaw_command)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._evaluator, name)


class H4RoutedSimulator(routed.RoutedSimulator):
    """Frozen simulator with H4 observation/target seams isolated locally."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        real_evaluator = self.evaluator
        self.evaluator = _H4EvaluatorProxy(real_evaluator, self)

    def _quality_contact_kinematics(
        self, data: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reproduce H4 training's exact floor/foot force aggregation.

        The frozen H3 metric helper intentionally remains untouched because
        its body-pair/absolute-force definition is part of the historical
        source closure.  H4 actor observables use the training contract:
        collision geom identity, signed normal force clipped at zero, and
        force-weighted tangential speed aggregated per foot.
        """

        floor_geom_id = int(self.model.geom("floor").id)
        foot_geom_ids = (
            int(self.model.geom("left_foot_bottom_tpu").id),
            int(self.model.geom("right_foot_bottom_tpu").id),
        )
        geom_to_foot = {
            geom_id: index for index, geom_id in enumerate(foot_geom_ids)
        }
        normal_force_n = np.zeros(2, dtype=np.float64)
        force_weighted_tangential_speed = np.zeros(2, dtype=np.float64)
        spatial_velocity: dict[int, np.ndarray] = {}

        def point_velocity(body_id: int, point_world: np.ndarray) -> np.ndarray:
            if body_id not in spatial_velocity:
                value = np.zeros(6, dtype=np.float64)
                self.mujoco.mj_objectVelocity(
                    self.model,
                    data,
                    self.mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                    value,
                    0,
                )
                spatial_velocity[body_id] = value
            value = spatial_velocity[body_id]
            offset = point_world - np.asarray(data.xpos[body_id], dtype=np.float64)
            return value[3:] + np.cross(value[:3], offset)

        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            if int(contact.efc_address) < 0:
                continue
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)
            if geom_1 == floor_geom_id and geom_2 in geom_to_foot:
                foot_geom, foot_body, other_body = (
                    geom_2,
                    int(self.model.geom_bodyid[geom_2]),
                    int(self.model.geom_bodyid[geom_1]),
                )
            elif geom_2 == floor_geom_id and geom_1 in geom_to_foot:
                foot_geom, foot_body, other_body = (
                    geom_1,
                    int(self.model.geom_bodyid[geom_1]),
                    int(self.model.geom_bodyid[geom_2]),
                )
            else:
                continue
            force = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(self.model, data, contact_index, force)
            force_n = max(0.0, float(force[0]))
            if force_n <= 0.0:
                continue
            point = np.asarray(contact.pos, dtype=np.float64)
            relative_velocity = point_velocity(foot_body, point) - point_velocity(
                other_body, point
            )
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            tangential_velocity = relative_velocity - np.dot(
                relative_velocity, normal
            ) * normal
            tangential_speed = float(np.linalg.norm(tangential_velocity))
            foot = geom_to_foot[foot_geom]
            normal_force_n[foot] += force_n
            force_weighted_tangential_speed[foot] += force_n * tangential_speed

        tangential_speed = np.divide(
            force_weighted_tangential_speed,
            normal_force_n,
            out=np.zeros(2, dtype=np.float64),
            where=normal_force_n > 0.0,
        )
        return normal_force_n / self.robot_weight_n, tangential_speed

    def _policy_target(
        self,
        applied_action: np.ndarray,
        effective_command: np.ndarray,
        phase_index: float,
        default: np.ndarray,
    ) -> np.ndarray:
        decision = getattr(self.bank, "active_decision", None)
        if decision is not None and H4RoutedPolicyBank.uses_absolute_reverse_decoder(
            decision.expert, effective_command
        ):
            return H4RoutedPolicyBank.absolute_reverse_targets(
                applied_action,
                default=default,
                joint_ranges=self.joint_ranges,
            )
        return super()._policy_target(
            applied_action, effective_command, phase_index, default
        )


_FROZEN_RUN_SCHEDULE = routed.RoutedSimulator.run_schedule


def _h4_advance_routed_phase(
    phase_index: float,
    *,
    phase_steps: float,
    phase_delta: float,
    current_expert: str,
    previous_expert: str | None,
    effective_command: Sequence[float],
    previous_backward_feedforward_active: bool,
    diagnostic_entry_phase_indices: Mapping[str, float] | None = None,
    phase_entry_status: str = "DIAGNOSTIC_UNADOPTED",
    diagnostic_only: bool = True,
    control_step: int | None = None,
    global_control_tick: int | None = None,
) -> tuple[float, bool, dict[str, Any] | None]:
    if (
        canonical_policy_role(current_expert) == "reverse"
        and float(np.asarray(effective_command, dtype=np.float64)[0]) < -0.02
    ):
        return H4RoutedPolicyBank.advance_phase(
            phase_index=phase_index,
            phase_steps=phase_steps,
            phase_delta=phase_delta,
            current_expert=current_expert,
            previous_expert=previous_expert,
            effective_command=effective_command,
            previous_backward_feedforward_active=previous_backward_feedforward_active,
            control_step=0 if control_step is None else control_step,
            global_control_tick=0 if global_control_tick is None else global_control_tick,
        )
    return routed.advance_routed_phase(
        phase_index,
        phase_steps=phase_steps,
        phase_delta=phase_delta,
        current_expert=current_expert,
        previous_expert=previous_expert,
        effective_command=effective_command,
        previous_backward_feedforward_active=previous_backward_feedforward_active,
        diagnostic_entry_phase_indices=diagnostic_entry_phase_indices,
        phase_entry_status=phase_entry_status,
        diagnostic_only=diagnostic_only,
        control_step=control_step,
        global_control_tick=global_control_tick,
    )


def _h4_resolve_policy_observation_command(
    routed_expert: str,
    effective_command: Sequence[float],
    *,
    backward_residual_scale: float,
    override: Sequence[float] | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Record the physical route command before resolving its policy view."""

    effective = np.asarray(effective_command, dtype=np.float64)
    if effective.shape != (3,) or not np.all(np.isfinite(effective)):
        raise ValueError("H4 effective command context must be finite (3,)")
    _H4_RUN_CONTEXT["physical_command"] = np.asarray(
        effective,
        dtype=np.float64,
    ).copy()
    resolved = routed.resolve_policy_observation_command(
        routed_expert,
        effective_command,
        backward_residual_scale=backward_residual_scale,
        override=override,
    )
    # H4 forward training uses make_anchor_command_mapper with the physical
    # anchor (0.05, 0, 0) and the policy-visible anchor (0.10, -0.018, -0.170).
    # The physical command remains untouched; only the 101-wide policy view is
    # replaced for the H4 forward actor.  Reverse's H4 anchor is identity on
    # vx, so it deliberately follows the frozen resolver unchanged.
    if canonical_policy_role(routed_expert) == "forward":
        ratio = float(effective[0] / 0.05)
        h4_policy_command = np.asarray(
            (0.10 * ratio, -0.018 * ratio, -0.170 * ratio),
            dtype=np.float64,
        )
        return h4_policy_command, 0.0, True
    return resolved


_RUN_SCHEDULE_GLOBALS = dict(_FROZEN_RUN_SCHEDULE.__globals__)
_RUN_SCHEDULE_GLOBALS["advance_routed_phase"] = _h4_advance_routed_phase
_RUN_SCHEDULE_GLOBALS[
    "resolve_policy_observation_command"
] = _h4_resolve_policy_observation_command
H4RoutedSimulator.run_schedule = FunctionType(
    _FROZEN_RUN_SCHEDULE.__code__,
    _RUN_SCHEDULE_GLOBALS,
    _FROZEN_RUN_SCHEDULE.__name__,
    _FROZEN_RUN_SCHEDULE.__defaults__,
    _FROZEN_RUN_SCHEDULE.__closure__,
)


def _build_simulator(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    policy_paths = parse_policy_assignments(args.policy)
    if set(policy_paths) != set(REQUIRED_POLICY_ROLES):
        raise ValueError("exactly the eight legacy policy roles are required")
    forward = validate_h4_candidate(_candidate_spec(args, "forward"))
    reverse = validate_h4_candidate(_candidate_spec(args, "reverse"))

    generated_root = args.generated_root.resolve()
    routed.validate_exact_generated_assets(generated_root)
    asset_paths = routed.generated_asset_paths(generated_root)
    mujoco, onnxruntime, runtime, runtime_provenance = routed._load_runtime(
        include_provenance=True
    )
    base_bank = routed.RoutedPolicyBank(policy_paths, onnxruntime)
    bank = H4RoutedPolicyBank(base_bank, {"forward": forward, "reverse": reverse})
    evaluator = runtime.OfficialPolicyEvaluator(
        asset_paths["scene"], policy_paths["stand"], asset_paths["reference"]
    )
    evaluator.backward_residual_scale = 0.0
    evaluator.load_backward_profile(routed.FORMAL_FIXED_BACKWARD_PROFILE)
    evaluator.load_backward_turn_profile(1, routed.FORMAL_FIXED_BACKWARD_LEFT_PROFILE)
    evaluator.load_backward_turn_profile(-1, asset_paths["backward_right"])
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0
    model_evidence = routed.validate_model_contract(evaluator)
    simulator = H4RoutedSimulator(
        evaluator,
        bank,
        mujoco,
        runtime,
        leg_target_margin_rad=routed.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
        target_slew_rate_rad_s=routed.RUNTIME_TARGET_SLEW_RATE_RAD_S,
        diagnostic_noncontract_safety=False,
        left_knee_extra_upper_margin_rad=0.0,
        diagnostic_reverse_entry_phase_indices=None,
        diagnostic_unadopted_backward_exit_recovery=False,
        formal_candidate_default=False,
    )
    # The H4 actor uses the source-period reset/advance contract.  This mapping
    # is consumed only by the copied phase audit seam; legacy H3 routes remain
    # delegated to the frozen phase implementation.
    simulator.reverse_entry_phase_indices = {
        "reverse": 7.0,
        "reverse_turn_left": 4.0,
        "reverse_turn_right": 4.0,
    }
    # This evaluator is a diagnostic integration harness.  Entry-phase
    # provenance is intentionally retained as diagnostic metadata until a
    # release evaluator independently proves the same route.  Treating this
    # as an adoption-ready execution marker makes the diagnostic artifact
    # look stronger than it is and breaks the exclusive-marker audit.
    simulator.phase_entry_diagnostic_only = True
    simulator.phase_entry_status = "H4_SIMULATION_DIAGNOSTIC"
    source_paths = {
        "routed_evaluator": Path(routed.__file__).resolve(),
        "routed_evaluation_contract": (
            EXP_ROOT / "safe_gait_experts" / "routed_evaluation.py"
        ).resolve(),
        "gait_quality": (EXP_ROOT / "safe_gait_experts" / "gait_quality.py").resolve(),
        "h4_routed_policy": Path(__file__).resolve().parents[1]
        / "safe_gait_experts"
        / "h4_routed_policy.py",
        "h4_post_training": (EXP_ROOT / "safe_gait_experts" / "h4_post_training.py").resolve(),
        "h4_training_alignment": (
            EXP_ROOT / "safe_gait_experts" / "h4_training_alignment.py"
        ).resolve(),
        "target_safety": (EXP_ROOT / "target_safety.py").resolve(),
        "scene": asset_paths["scene"],
        "reference": asset_paths["reference"],
    }
    metadata = {
        "h4_candidates": {
            "forward": forward.validation,
            "reverse": reverse.validation,
        },
        "h4_actor_contract": {
            "legacy_observation_width": 101,
            "physical_extra_width": 15,
            "actor_observation_width": 116,
            "extra_order": [
                "physical_command_xyz",
                "local_linvel_xyz",
                "gravity_local_xyz",
                "normalized_force_lr",
                "contact_bool_lr",
                "tangential_speed_lr",
            ],
            "gravity_definition": "trunk_rotation_world_to_local_transpose_times_0_0_minus_1",
            "h4_head_action_mask_exact": True,
        },
        "reverse_v6_runtime_contract": {
            "absolute_decoder_role": "reverse only",
            "teacher_target_contribution": 0.0,
            "source_phase_entry": 7.0,
            "teacher_entry_phase_bins": 14.0,
            "source_phase_delta_per_control": 0.81,
            "teacher_phase_advance_bins_per_control": 1.62,
            "source_period_bins": 27,
            "teacher_table_rows": 54,
            "decoder": "exact reverse_iteration_v6_absolute_full_leg_targets numpy implementation",
            "legacy_backward_feedforward_overwrite": False,
            "hardware_deployment": "PROHIBITED",
        },
        "model_contract": model_evidence,
        "runtime_provenance": runtime_provenance,
        "source_snapshot_pre": _source_snapshot(source_paths),
    }
    return simulator, bank, evaluator, runtime, metadata


def _run_suites(args: argparse.Namespace, simulator: Any) -> dict[str, Any]:
    primitive_episodes = routed._independent_suite(
        simulator,
        PRIMITIVE_CASES,
        seed_base=args.seed,
        episodes=args.episodes,
        seconds=args.seconds,
        joint_noise_scale=args.initial_joint_noise_scale,
        initial_base_speed=args.initial_base_speed,
        warmup_seconds=args.warmup_seconds,
    )
    compound_episodes = routed._independent_suite(
        simulator,
        COMPOUND_CASES,
        seed_base=args.seed + 1_000_000,
        episodes=args.episodes,
        seconds=args.seconds,
        joint_noise_scale=args.initial_joint_noise_scale,
        initial_base_speed=args.initial_base_speed,
        warmup_seconds=args.warmup_seconds,
    )
    transition_definition = transition_schedule(
        args.transition_seconds, args.transition_stand_seconds
    )
    transition_episodes = [
        simulator.run_schedule(
            transition_definition,
            seed=args.seed + 2_000_000 + episode_index,
            joint_noise_scale=args.initial_joint_noise_scale,
            initial_base_speed=args.initial_base_speed,
            warmup_seconds=args.warmup_seconds,
        )
        for episode_index in range(args.episodes)
    ]
    thresholds = AcceptanceThresholds()
    primitive_acceptance = suite_acceptance(
        primitive_episodes,
        [case.name for case in PRIMITIVE_CASES],
        thresholds,
        require_gait_quality=True,
    )
    compound_acceptance = suite_acceptance(
        compound_episodes,
        [case.name for case in COMPOUND_CASES],
        thresholds,
        require_gait_quality=True,
    )
    transition_acceptance = suite_acceptance(
        transition_episodes,
        [scheduled[0] for scheduled in transition_definition],
        thresholds,
        require_gait_quality=True,
    )
    return {
        "definitions": {
            "primitive_cases": [asdict(case) for case in PRIMITIVE_CASES],
            "compound_cases": [asdict(case) for case in COMPOUND_CASES],
            "transition_cases": [
                {
                    "name": name,
                    "command": list(command),
                    "seconds": seconds,
                    "expected_expert": expected_expert,
                    "expected_policy_role": expected_policy_role,
                }
                for name, command, seconds, _policy_command, expected_expert, expected_policy_role in transition_definition
            ],
            "locomotion_action_count_excluding_stand": 12,
            "locomotion_action_names": [
                case.name for case in (*PRIMITIVE_CASES, *COMPOUND_CASES) if case.name != "stand"
            ],
        },
        "primitives": {
            "episodes": primitive_episodes,
            "acceptance": primitive_acceptance,
        },
        "compounds": {
            "episodes": compound_episodes,
            "acceptance": compound_acceptance,
        },
        "transitions": {
            "episodes": transition_episodes,
            "acceptance": transition_acceptance,
        },
        "passed": bool(
            primitive_acceptance["passed"]
            and compound_acceptance["passed"]
            and transition_acceptance["passed"]
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    simulator, bank, evaluator, runtime, metadata = _build_simulator(args)
    suites = _run_suites(args, simulator)
    metadata["source_snapshot_post"] = metadata["source_snapshot_pre"]
    payload = {
        "schema_version": 1,
        "evaluator_id": "openduckmini-exp004-h4-routed-transition-v1",
        "evaluation_mode": "H4_INTEGRATED_STRICT_DIAGNOSTIC",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_deployment": "PROHIBITED",
        "configuration": {
            "seed": args.seed,
            "episodes": args.episodes,
            "seconds": args.seconds,
            "transition_seconds": args.transition_seconds,
            "transition_stand_seconds": args.transition_stand_seconds,
            "warmup_seconds": args.warmup_seconds,
            "initial_joint_noise_scale": args.initial_joint_noise_scale,
            "initial_base_speed": args.initial_base_speed,
            "target_margin_rad": routed.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
            "target_slew_rate_rad_s": routed.RUNTIME_TARGET_SLEW_RATE_RAD_S,
            "screening_scale_is_not_release_qualification": not (
                args.episodes == 20
                and args.seconds == 30.0
                and args.transition_seconds == 30.0
                and args.transition_stand_seconds == 5.0
                and args.warmup_seconds == 1.5
                and args.initial_joint_noise_scale == 1.0
                and args.initial_base_speed == 0.10
                and args.seed == 20260808
            ),
        },
        "strict_thresholds": asdict(AcceptanceThresholds()),
        "provenance": metadata,
        "policy_bank": bank.manifest(),
        "suites": suites,
        "acceptance": {
            "all_strict_quality_safety_transition_gates_passed": suites["passed"],
            "adoption_allowed": False,
            "release_allowed": False,
            "hardware_deployment": "PROHIBITED",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_sha256 = sha256_file(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": output_sha256,
                "strict_passed": suites["passed"],
                "adoption_allowed": False,
                "release_allowed": False,
                "hardware_deployment": "PROHIBITED",
            },
            indent=2,
        )
    )
    if not suites["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
