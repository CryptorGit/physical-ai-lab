"""Pilot-search a reverse profile under the final margin/slew composition.

This is a simulation-only diagnostic.  Every candidate is evaluated on the
exact generated scene with three or more fixed reset/push seeds, a 0.050 rad
leg-target margin, a 2.0 rad/s target slew, zero residual, and exact-zero head
targets.  Falls or any physical SAFE qpos/target violation receive a hard
penalty.  The selected candidate is then replayed through exp_004's routed
evaluator at both ends of the requested reverse-command interval.

Hardware deployment remains PROHIBITED regardless of the result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_root in (EXP_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from safe_gait_experts.contract import (  # noqa: E402
    ACTUATOR_JOINT_ORDER,
    CONTRACT,
    HEAD_JOINTS,
    SAFE_JOINT_LIMITS,
)
from safe_gait_experts.routed_evaluation import (  # noqa: E402
    SafetyAudit,
    blend_and_mask_actions,
    build_target_envelope,
    canonical_policy_role,
    generated_asset_paths,
    segment_acceptance,
    sha256_file,
    validate_exact_generated_assets,
)
from safe_gait_experts.safe_randomization import (  # noqa: E402
    actuator_name_to_index,
    build_qpos_noise_scale,
    clip_reset_qpos_to_physical_safe_limits,
)
from evaluate_routed_transitions import (  # noqa: E402
    DiagnosticTargetSafetyGuard,
    RoutedSimulator,
    _load_runtime,
    apply_control_first_startup,
    validate_model_contract,
)


TARGET_MARGIN_RAD = 0.050
TARGET_SLEW_RAD_S = 2.0
DEFAULT_SEEDS = (20260808, 20260809, 20260810)
DEFAULT_GENERATED_ROOT = EXP_ROOT / "artifacts" / "generated_playground"
DEFAULT_POLICY = (
    WORKSPACE
    / ".openduck_runtime_source_review"
    / "calibrated_hybrid_policy_v22.onnx"
)
DEFAULT_INITIAL_PROFILE = EXP_ROOT / "artifacts" / "optimized_reverse_exact_safe_v1.json"
DEFAULT_OUTPUT = (
    EXP_ROOT / "artifacts" / "optimized_reverse_margin050_slew200_pilot.json"
)


@dataclass(frozen=True)
class ProfileParameters:
    amplitude_scales: np.ndarray
    bias_offsets: np.ndarray
    phase_rate: float

    def as_json(self) -> dict[str, Any]:
        return {
            "joint_amplitude_scales": self.amplitude_scales.tolist(),
            "joint_bias_offsets": self.bias_offsets.tolist(),
            "phase_rate": float(self.phase_rate),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a reverse feedforward profile with margin=0.050 rad, "
            "slew=2.0 rad/s, residual=0, and multi-seed hard safety."
        )
    )
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--initial-profile", type=Path, default=DEFAULT_INITIAL_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--validation-seconds", type=float, default=6.0)
    parser.add_argument("--warmup-seconds", type=float, default=0.8)
    parser.add_argument("--maxiter", type=int, default=3)
    parser.add_argument("--popsize", type=int, default=2)
    parser.add_argument("--search-seed", type=int, default=20260808)
    parser.add_argument(
        "--perturb-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="At least three fixed reset/push seeds used for every candidate.",
    )
    parser.add_argument("--initial-joint-noise-scale", type=float, default=1.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.10)
    parser.add_argument(
        "--reset-qpos-inward-margin-rad",
        type=float,
        default=0.005,
        help=(
            "Additional inward sampling buffer for perturbed reset qpos. "
            "The hard physical-SAFE qpos audit still covers every physics step."
        ),
    )
    parser.add_argument("--target-vx-min", type=float, default=-0.050)
    parser.add_argument("--target-vx-max", type=float, default=-0.040)
    parser.add_argument(
        "--minimum-reverse-vx",
        type=float,
        default=-0.015,
        help="Pilot gate requires every routed seed/command vx at or below this value.",
    )
    parser.add_argument("--minimum-scale-factor", type=float, default=0.25)
    parser.add_argument("--maximum-scale-factor", type=float, default=2.00)
    parser.add_argument("--maximum-absolute-scale", type=float, default=8.0)
    parser.add_argument("--minimum-phase-rate", type=float, default=0.70)
    parser.add_argument("--maximum-phase-rate", type=float, default=4.50)
    parser.add_argument(
        "--max-bias",
        type=float,
        default=0.0,
        help="Optional symmetric per-leg-joint bias bound in rad.",
    )
    parser.add_argument(
        "--bias-search-radius",
        type=float,
        help="Optional local bias-search radius around the initial profile.",
    )
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument(
        "--routed-finalists",
        type=int,
        default=5,
        help="Replay this many fast-search finalists through the routed evaluator.",
    )
    parser.add_argument(
        "--require-gait-quality",
        action="store_true",
        help=(
            "Require full strict gait-quality acceptance during routed "
            "finalist validation, not only safety and signed progress."
        ),
    )
    args = parser.parse_args(argv)

    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.seconds <= 0.0 or args.validation_seconds <= 0.0:
        parser.error("--seconds and --validation-seconds must be positive")
    if args.warmup_seconds < 0.0 or args.warmup_seconds >= min(
        args.seconds, args.validation_seconds
    ):
        parser.error("--warmup-seconds must be non-negative and shorter than runs")
    if args.maxiter < 1 or args.popsize < 1:
        parser.error("--maxiter and --popsize must be positive")
    if len(args.perturb_seeds) < 3 or len(set(args.perturb_seeds)) != len(
        args.perturb_seeds
    ):
        parser.error("--perturb-seeds requires at least three distinct values")
    if (
        args.initial_joint_noise_scale < 0.0
        or args.initial_base_speed < 0.0
        or args.reset_qpos_inward_margin_rad < 0.0
    ):
        parser.error("initial perturbations must be non-negative")
    try:
        build_target_envelope(
            leg_margin_rad=args.reset_qpos_inward_margin_rad
        )
    except ValueError as error:
        parser.error(f"invalid reset qpos inward margin: {error}")
    if not args.target_vx_min < args.target_vx_max < 0.0:
        parser.error("target vx interval must satisfy MIN < MAX < 0")
    if not args.target_vx_min <= args.minimum_reverse_vx < 0.0:
        parser.error("--minimum-reverse-vx must lie in [target-vx-min, 0)")
    if not 0.0 <= args.minimum_scale_factor < args.maximum_scale_factor:
        parser.error("scale factors must satisfy 0 <= minimum < maximum")
    if args.maximum_absolute_scale <= 0.0:
        parser.error("--maximum-absolute-scale must be positive")
    if not 0.0 < args.minimum_phase_rate < args.maximum_phase_rate:
        parser.error("phase bounds must be finite, positive, and ordered")
    if (
        args.max_bias < 0.0
        or args.top_candidates <= 0
        or args.routed_finalists <= 0
    ):
        parser.error(
            "--max-bias must be non-negative; candidate counts must be positive"
        )
    if args.bias_search_radius is not None and args.bias_search_radius <= 0.0:
        parser.error("--bias-search-radius must be positive")
    return args


def load_profile(path: Path) -> ProfileParameters:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    scales = np.asarray(parameters["joint_amplitude_scales"], dtype=np.float64)
    biases = np.asarray(
        parameters.get("joint_bias_offsets", [0.0] * 10), dtype=np.float64
    )
    phase_rate = float(parameters["phase_rate"])
    if scales.shape != (10,) or biases.shape != (10,):
        raise ValueError("profile must contain ten amplitude scales and ten biases")
    if not np.all(np.isfinite(scales)) or not np.all(np.isfinite(biases)):
        raise ValueError("profile parameters must be finite")
    if np.any(scales < 0.0) or not np.isfinite(phase_rate) or phase_rate <= 0.0:
        raise ValueError("profile scales must be non-negative and phase positive")
    return ProfileParameters(scales, biases, phase_rate)


def candidate_to_parameters(candidate: Sequence[float], max_bias: float) -> ProfileParameters:
    values = np.asarray(candidate, dtype=np.float64)
    expected = 21 if max_bias > 0.0 else 11
    if values.shape != (expected,) or not np.all(np.isfinite(values)):
        raise ValueError(f"candidate must contain {expected} finite values")
    scales = values[:10].copy()
    if max_bias > 0.0:
        biases = values[10:20].copy()
    else:
        biases = np.zeros(10, dtype=np.float64)
    return ProfileParameters(scales, biases, float(values[-1]))


def parameters_to_candidate(parameters: ProfileParameters, max_bias: float) -> np.ndarray:
    values = list(parameters.amplitude_scales)
    if max_bias > 0.0:
        values.extend(parameters.bias_offsets)
    values.append(parameters.phase_rate)
    return np.asarray(values, dtype=np.float64)


def candidate_bounds(
    initial: ProfileParameters, args: argparse.Namespace
) -> list[tuple[float, float]]:
    bounds = []
    for scale in initial.amplitude_scales:
        lower = max(0.0, float(scale) * args.minimum_scale_factor)
        upper = min(
            args.maximum_absolute_scale,
            max(float(scale) * args.maximum_scale_factor, lower + 0.25),
        )
        bounds.append((lower, upper))
    if args.max_bias > 0.0:
        if args.bias_search_radius is None:
            bounds.extend([(-args.max_bias, args.max_bias)] * 10)
        else:
            for bias in initial.bias_offsets:
                lower = max(-args.max_bias, float(bias) - args.bias_search_radius)
                upper = min(args.max_bias, float(bias) + args.bias_search_radius)
                if lower >= upper:
                    raise ValueError("local bias bound collapsed at max-bias limit")
                bounds.append((lower, upper))
    bounds.append((args.minimum_phase_rate, args.maximum_phase_rate))
    return bounds


def initial_population(
    bounds: Sequence[tuple[float, float]],
    initial: np.ndarray,
    popsize: int,
    seed: int,
) -> np.ndarray:
    """Create a reproducible population containing useful baseline variants."""

    dimensions = len(bounds)
    population_size = max(5, popsize * dimensions)
    lower = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
    upper = np.asarray([bound[1] for bound in bounds], dtype=np.float64)
    rng = np.random.default_rng(seed)
    population = rng.uniform(lower, upper, size=(population_size, dimensions))
    population[0] = np.clip(initial, lower, upper)

    row = 1
    for amplitude_factor in (0.55, 0.75, 1.25, 1.50):
        if row >= population_size:
            break
        variant = initial.copy()
        variant[:10] *= amplitude_factor
        population[row] = np.clip(variant, lower, upper)
        row += 1
    for phase_factor in (0.60, 0.80, 1.20, 1.40):
        if row >= population_size:
            break
        variant = initial.copy()
        variant[-1] *= phase_factor
        population[row] = np.clip(variant, lower, upper)
        row += 1
    return population


class SharedV22PolicyBank:
    """Use the evaluator's one CPU ONNX session for every frozen-v22 role."""

    def __init__(self, evaluator: Any):
        self.evaluator = evaluator
        self.inference_counts: Counter[str] = Counter()

    def infer(self, role: str, observation: np.ndarray) -> np.ndarray:
        canonical = canonical_policy_role(role)
        result = self.evaluator.session.run(
            None, {"obs": np.asarray(observation, dtype=np.float32)[None, :]}
        )[0][0]
        action = np.asarray(result, dtype=np.float64)
        if action.shape != (14,) or not np.all(np.isfinite(action)):
            raise ValueError("v22 policy returned an invalid action")
        self.inference_counts[canonical] += 1
        return action

    def infer_route(
        self, decision: Any, observation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        from_action = self.infer(decision.blend_from_expert, observation)
        if decision.blend_to_expert == decision.blend_from_expert:
            to_action = from_action
        else:
            to_action = self.infer(decision.blend_to_expert, observation)
        return blend_and_mask_actions(from_action, to_action, decision.blend_alpha)


class PerJointCapRoutedSimulator(RoutedSimulator):
    """Central control-first evaluator with a stricter left-knee target cap."""

    def __init__(
        self,
        *args: Any,
        left_knee_extra_upper_margin_rad: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            *args,
            left_knee_extra_upper_margin_rad=(
                left_knee_extra_upper_margin_rad
            ),
            **kwargs,
        )


class MarginAwareReverseEvaluator:
    """Fast residual-zero search rollouts plus exact routed validation."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.asset_evidence = validate_exact_generated_assets(
            args.generated_root.resolve()
        )
        self.asset_paths = generated_asset_paths(args.generated_root.resolve())
        self.mujoco, self.onnxruntime, self.runtime = _load_runtime()
        self.evaluator = self.runtime.OfficialPolicyEvaluator(
            self.asset_paths["scene"],
            args.policy.resolve(),
            self.asset_paths["reference"],
        )
        self.evaluator.backward_residual_scale = 0.0
        self.model_evidence = validate_model_contract(self.evaluator)
        self.model = self.evaluator.model
        self.joint_names = tuple(
            self.model.actuator(index).name for index in range(self.model.nu)
        )
        self.name_to_index = actuator_name_to_index(self.model)
        noise = CONTRACT["qpos_noise_scale_rad"]
        self.qpos_noise_scale = build_qpos_noise_scale(
            self.name_to_index,
            hip_scale=float(noise["hip"]),
            knee_scale=float(noise["knee"]),
            ankle_scale=float(noise["ankle"]),
        )
        self.target_lower, self.target_upper = build_target_envelope(
            self.joint_names, leg_margin_rad=TARGET_MARGIN_RAD
        )
        self.physical_lower, self.physical_upper = build_target_envelope(
            self.joint_names, leg_margin_rad=0.0
        )
        self.reset_lower, self.reset_upper = build_target_envelope(
            self.joint_names,
            leg_margin_rad=args.reset_qpos_inward_margin_rad,
        )
        self.home = self.model.keyframe("home")
        self.default = np.asarray(self.home.ctrl, dtype=np.float64).copy()
        self.joint_ranges = np.asarray(
            self.model.jnt_range[self.evaluator.actuator_joint_ids],
            dtype=np.float64,
        )
        self.zero_action = np.zeros(14, dtype=np.float64)

    def _initial_data(self, seed: int) -> Any:
        rng = np.random.default_rng(seed)
        data = self.mujoco.MjData(self.model)
        data.qpos[:] = self.home.qpos
        noise = rng.uniform(-1.0, 1.0, size=14) * (
            self.args.initial_joint_noise_scale * self.qpos_noise_scale
        )
        noisy = data.qpos[self.evaluator.actuator_qpos_addr] + noise
        reset = clip_reset_qpos_to_physical_safe_limits(
            noisy,
            self.joint_names,
            noise_applied=self.args.initial_joint_noise_scale > 0.0,
            reset_noise_margin_rad=self.args.reset_qpos_inward_margin_rad,
        )
        data.qpos[self.evaluator.actuator_qpos_addr] = reset
        data.ctrl[:] = reset
        if self.args.initial_base_speed > 0.0:
            angle = rng.uniform(-np.pi, np.pi)
            magnitude = rng.uniform(0.0, self.args.initial_base_speed)
            data.qvel[:2] = magnitude * np.asarray(
                [np.cos(angle), np.sin(angle)], dtype=np.float64
            )
        self.mujoco.mj_forward(self.model, data)
        return data

    def run_fast_seed(
        self,
        parameters: ProfileParameters,
        seed: int,
        seconds: float,
        *,
        left_knee_extra_upper_margin_rad: float = 0.0,
    ) -> dict[str, Any]:
        data = self._initial_data(seed)
        initial_targets = np.asarray(
            data.qpos[self.evaluator.actuator_qpos_addr], dtype=np.float64
        )
        target_upper = self.target_upper.copy()
        knee_index = self.joint_names.index("left_knee")
        target_upper[knee_index] -= float(left_knee_extra_upper_margin_rad)
        if target_upper[knee_index] < self.target_lower[knee_index]:
            raise ValueError("left-knee dynamic cap collapsed the target envelope")
        guard = DiagnosticTargetSafetyGuard(
            initial_targets,
            self.target_lower,
            target_upper,
            self.physical_lower,
            self.physical_upper,
            slew_rate_rad_per_s=TARGET_SLEW_RAD_S,
        )
        audit = SafetyAudit(
            self.joint_names,
            leg_target_margin_rad=TARGET_MARGIN_RAD,
            target_slew_limit_rad_per_s=TARGET_SLEW_RAD_S,
        )
        previous_position = data.xpos[self.evaluator.trunk_body_id].copy()
        phase_index = 0.0
        control_first_startup_audit: dict[str, Any] | None = None
        velocities: list[np.ndarray] = []
        angular_velocities: list[np.ndarray] = []
        contacts: list[np.ndarray] = []
        heights: list[float] = []
        uprights: list[float] = []
        target_steps = int(round(seconds / self.runtime.CONTROL_DT))
        warmup_steps = int(round(self.args.warmup_seconds / self.runtime.CONTROL_DT))
        fell = False
        completed_steps = 0
        physics_substep_qpos_limit_violations = 0
        maximum_physics_substep_qpos_excess_rad = 0.0
        leg_indices = np.asarray(
            [
                index
                for index, name in enumerate(self.joint_names)
                if name not in HEAD_JOINTS
            ],
            dtype=np.int64,
        )

        for control_step in range(target_steps):
            phase_index = (
                phase_index + parameters.phase_rate
            ) % self.evaluator.phase_steps
            preclip = self.evaluator._backward_feedforward(
                phase_index,
                self.default,
                self.joint_ranges,
                self.zero_action,
                gait_scales=parameters.amplitude_scales,
                gait_biases=parameters.bias_offsets,
                leg_residual_factor=0.0,
                head_residual_factor=0.0,
            )
            preclip[5:9] = 0.0
            desired = guard.desired_targets(preclip)
            previous_targets = guard.previous_targets
            if control_step == 0:
                applied, tick_startup_audit = apply_control_first_startup(
                    guard,
                    data.ctrl,
                    preclip,
                    self.joint_names,
                    control_dt=self.runtime.CONTROL_DT,
                    leg_target_margin_rad=TARGET_MARGIN_RAD,
                    target_slew_rate_rad_s=TARGET_SLEW_RAD_S,
                    physics_steps_before_control=0,
                )
            else:
                guard_steps_before = int(guard.steps_since_reset)
                applied = guard.step(preclip, self.runtime.CONTROL_DT)
                if int(guard.steps_since_reset) - guard_steps_before != 1:
                    raise RuntimeError("target guard must run exactly once per tick")
                data.ctrl[:] = applied
                tick_startup_audit = None
            if tick_startup_audit is not None:
                control_first_startup_audit = tick_startup_audit
            for _ in range(int(self.runtime.DECIMATION)):
                self.mujoco.mj_step(self.model, data)
                substep_qpos = np.asarray(
                    data.qpos[self.evaluator.actuator_qpos_addr], dtype=np.float64
                )
                substep_excess = np.maximum(
                    np.maximum(
                        self.physical_lower[leg_indices]
                        - substep_qpos[leg_indices],
                        substep_qpos[leg_indices]
                        - self.physical_upper[leg_indices],
                    ),
                    0.0,
                )
                physics_substep_qpos_limit_violations += int(
                    np.count_nonzero(substep_excess > 1e-9)
                )
                maximum_physics_substep_qpos_excess_rad = max(
                    maximum_physics_substep_qpos_excess_rad,
                    float(np.max(substep_excess)),
                )

            position = data.xpos[self.evaluator.trunk_body_id].copy()
            rotation = data.xmat[self.evaluator.trunk_body_id].reshape(3, 3)
            local_velocity = rotation.T @ (
                (position - previous_position) / self.runtime.CONTROL_DT
            )
            previous_position = position
            angular_velocity = rotation.T @ data.qvel[3:6]
            joint_qpos = np.asarray(
                data.qpos[self.evaluator.actuator_qpos_addr], dtype=np.float64
            )
            audit.update(
                raw_policy_action=self.zero_action,
                applied_action=self.zero_action,
                preclip_targets=preclip,
                margin_clipped_targets=desired,
                applied_targets=applied,
                previous_applied_targets=previous_targets,
                joint_qpos=joint_qpos,
                control_dt=self.runtime.CONTROL_DT,
            )
            completed_steps += 1
            height = float(position[2])
            upright = float(rotation[2, 2])
            heights.append(height)
            uprights.append(upright)
            if control_step >= warmup_steps:
                velocities.append(local_velocity)
                angular_velocities.append(angular_velocity)
                contacts.append(self.evaluator._feet_contacts(data))
            if (
                height < 0.12
                or upright < 0.65
                or not np.all(np.isfinite(data.qpos))
                or not np.all(np.isfinite(data.qvel))
            ):
                fell = True
                break

        if not velocities:
            velocities = [np.zeros(3, dtype=np.float64)]
            angular_velocities = [np.zeros(3, dtype=np.float64)]
            contacts = [self.evaluator._feet_contacts(data)]
        velocity_array = np.asarray(velocities, dtype=np.float64)
        angular_array = np.asarray(angular_velocities, dtype=np.float64)
        contact_array = np.asarray(contacts, dtype=np.float64)
        safety = audit.to_dict()
        safety_passed, hard_failures = hard_safety_status(safety, fell)
        if physics_substep_qpos_limit_violations:
            hard_failures["physics_substep_qpos_limit_violations"] = (
                physics_substep_qpos_limit_violations
            )
            safety_passed = False
        if control_first_startup_audit is None:
            raise RuntimeError("fast rollout executed no control-first tick")
        return {
            "seed": int(seed),
            "requested_seconds": float(seconds),
            "completed_seconds": completed_steps * self.runtime.CONTROL_DT,
            "fell": fell,
            "mean_local_velocity_xyz": velocity_array.mean(axis=0).tolist(),
            "mean_local_angular_velocity_xyz": angular_array.mean(axis=0).tolist(),
            "minimum_height_m": min(heights) if heights else 0.0,
            "minimum_upright": min(uprights) if uprights else 0.0,
            "left_contact_rate": float(contact_array[:, 0].mean()),
            "right_contact_rate": float(contact_array[:, 1].mean()),
            "single_support_rate": float(
                np.logical_xor(contact_array[:, 0], contact_array[:, 1]).mean()
            ),
            "flight_rate": float((contact_array.sum(axis=1) == 0).mean()),
            "safety_passed": safety_passed,
            "hard_safety_failures": hard_failures,
            "safety_audit": safety,
            "control_first_startup_audit": control_first_startup_audit,
            "physics_substep_qpos_limit_violations": (
                physics_substep_qpos_limit_violations
            ),
            "maximum_physics_substep_qpos_excess_rad": (
                maximum_physics_substep_qpos_excess_rad
            ),
            "left_knee_extra_upper_margin_rad": float(
                left_knee_extra_upper_margin_rad
            ),
        }

    def evaluate_candidate(
        self,
        parameters: ProfileParameters,
        seeds: Sequence[int],
        seconds: float,
        target_vx_min: float,
        target_vx_max: float,
        *,
        left_knee_extra_upper_margin_rad: float = 0.0,
    ) -> dict[str, Any]:
        episodes = [
            self.run_fast_seed(
                parameters,
                seed,
                seconds,
                left_knee_extra_upper_margin_rad=(
                    left_knee_extra_upper_margin_rad
                ),
            )
            for seed in seeds
        ]
        target_center = 0.5 * (target_vx_min + target_vx_max)
        episode_costs = []
        for episode in episodes:
            vx, vy, _ = episode["mean_local_velocity_xyz"]
            wx, wy, wz = episode["mean_local_angular_velocity_xyz"]
            if vx < target_vx_min:
                band_error = target_vx_min - vx
            elif vx > target_vx_max:
                band_error = vx - target_vx_max
            else:
                band_error = 0.0
            motion_cost = (
                4000.0 * band_error**2
                + 800.0 * (vx - target_center) ** 2
                + 700.0 * vy**2
                + 500.0 * wz**2
                + 30.0 * (wx**2 + wy**2)
                + 20.0 * max(0.0, 0.98 - episode["minimum_upright"])
                + 3.0 * max(0.0, 0.20 - episode["single_support_rate"]) ** 2
            )
            hard_cost = 0.0
            if not episode["safety_passed"]:
                violation_count = sum(
                    int(value) if isinstance(value, (int, np.integer)) else 1
                    for value in episode["hard_safety_failures"].values()
                )
                hard_cost = 1_000_000.0 + 1_000.0 * violation_count
            episode["motion_cost"] = float(motion_cost)
            episode["hard_safety_cost"] = float(hard_cost)
            episode_costs.append(float(motion_cost + hard_cost))
        vx_values = np.asarray(
            [episode["mean_local_velocity_xyz"][0] for episode in episodes]
        )
        aggregate_cost = (
            float(np.mean(episode_costs))
            + 0.50 * float(np.max(episode_costs))
            + 800.0 * float(np.var(vx_values))
        )
        return {
            "cost": aggregate_cost,
            "all_hard_safety_passed": all(
                episode["safety_passed"] for episode in episodes
            ),
            "mean_vx": float(np.mean(vx_values)),
            "worst_seed_vx": float(np.max(vx_values)),
            "vx_standard_deviation": float(np.std(vx_values)),
            "left_knee_extra_upper_margin_rad": float(
                left_knee_extra_upper_margin_rad
            ),
            "episodes": episodes,
        }

    def routed_validation(
        self,
        parameters: ProfileParameters,
        seeds: Sequence[int],
        seconds: float,
        commands_vx: Sequence[float],
        *,
        left_knee_extra_upper_margin_rad: float = 0.0,
    ) -> dict[str, Any]:
        self.evaluator.backward_gait_scales = parameters.amplitude_scales.copy()
        self.evaluator.backward_gait_biases = parameters.bias_offsets.copy()
        self.evaluator.backward_phase_rate = float(parameters.phase_rate)
        self.evaluator.backward_residual_scale = 0.0
        bank = SharedV22PolicyBank(self.evaluator)
        simulator = PerJointCapRoutedSimulator(
            self.evaluator,
            bank,
            self.mujoco,
            self.runtime,
            leg_target_margin_rad=TARGET_MARGIN_RAD,
            target_slew_rate_rad_s=TARGET_SLEW_RAD_S,
            diagnostic_noncontract_safety=True,
            left_knee_extra_upper_margin_rad=(
                left_knee_extra_upper_margin_rad
            ),
        )
        episodes = []
        for seed in seeds:
            for command_vx in commands_vx:
                run = simulator.run_schedule(
                    (
                        (
                            "reverse",
                            (float(command_vx), 0.0, 0.0),
                            seconds,
                            None,
                            "reverse",
                            "reverse",
                        ),
                    ),
                    seed=int(seed),
                    joint_noise_scale=self.args.initial_joint_noise_scale,
                    initial_base_speed=self.args.initial_base_speed,
                    warmup_seconds=self.args.warmup_seconds,
                )
                segment = run["segments"][0]
                acceptance = segment_acceptance(
                    segment,
                    require_gait_quality=bool(self.args.require_gait_quality),
                )
                audit = segment["safety_audit"]
                hard_passed, hard_failures = hard_safety_status(
                    audit, bool(segment["fell"])
                )
                substep_audit = segment["physics_substep_audit"]
                expected_substeps = int(segment["completed_physics_substeps"])
                if int(substep_audit["sample_count"]) != expected_substeps:
                    hard_failures["unaudited_physics_substeps"] = (
                        expected_substeps - int(substep_audit["sample_count"])
                    )
                for field in (
                    "qpos_limit_violations",
                    "nonfinite_state_samples",
                    "height_fall_samples",
                    "upright_fall_samples",
                ):
                    value = int(substep_audit[field])
                    if value:
                        hard_failures[f"physics_substep_{field}"] = value
                hard_passed = not hard_failures
                episodes.append(
                    {
                        "seed": int(seed),
                        "command": [float(command_vx), 0.0, 0.0],
                        "segment": segment,
                        "standard_segment_acceptance": acceptance,
                        "hard_safety_passed": hard_passed,
                        "hard_safety_failures": hard_failures,
                        "reset_qpos_audit": run["reset_qpos_audit"],
                        "control_first_startup_audit": run[
                            "control_first_startup_audit"
                        ],
                    }
                )
        vx_values = np.asarray(
            [episode["segment"]["metrics"]["mean_local_velocity_xyz"][0] for episode in episodes]
        )
        vy_values = np.asarray(
            [episode["segment"]["metrics"]["mean_local_velocity_xyz"][1] for episode in episodes]
        )
        yaw_values = np.asarray(
            [episode["segment"]["metrics"]["mean_local_yaw_rate"] for episode in episodes]
        )
        motion_retained = bool(
            np.max(vx_values) <= self.args.minimum_reverse_vx
        )
        return {
            "commands_vx": [float(value) for value in commands_vx],
            "fixed_perturb_seeds": [int(seed) for seed in seeds],
            "episode_count": len(episodes),
            "all_hard_safety_passed": all(
                episode["hard_safety_passed"] for episode in episodes
            ),
            "all_standard_segment_checks_passed": all(
                episode["standard_segment_acceptance"]["passed"]
                for episode in episodes
            ),
            "reverse_motion_retained_threshold_vx": self.args.minimum_reverse_vx,
            "reverse_motion_retained": motion_retained,
            "mean_vx": float(np.mean(vx_values)),
            "worst_seed_command_vx": float(np.max(vx_values)),
            "mean_vy": float(np.mean(vy_values)),
            "maximum_absolute_mean_vy": float(np.max(np.abs(vy_values))),
            "mean_yaw_rate": float(np.mean(yaw_values)),
            "maximum_absolute_mean_yaw_rate": float(np.max(np.abs(yaw_values))),
            "episodes": episodes,
            "policy_inference_counts": dict(sorted(bank.inference_counts.items())),
            "control_first_initial_target": True,
            "home_only_startup_precharge_used": False,
            "left_knee_extra_upper_margin_rad": float(
                left_knee_extra_upper_margin_rad
            ),
        }


def hard_safety_status(
    audit: Mapping[str, Any], fell: bool
) -> tuple[bool, dict[str, Any]]:
    """Return hard failures for physical SAFE targets/qpos, slew, head, fall."""

    failures: dict[str, Any] = {}
    integer_fields = (
        "nonfinite_sample_count",
        "preclip_target_limit_violations",
        "applied_target_limit_violations",
        "desired_target_margin_violations",
        "unauthorized_applied_target_margin_violations",
        "target_slew_violations",
        "qpos_limit_violations",
    )
    if fell:
        failures["fell"] = True
    for field in integer_fields:
        value = int(audit[field])
        if value != 0:
            failures[field] = value
    for field in ("applied_head_action_peak", "head_target_peak_rad"):
        value = float(audit[field])
        if value != 0.0:
            failures[field] = value
    return not failures, failures


def summarize_candidate(
    parameters: ProfileParameters, evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    return {"parameters": parameters.as_json(), "evaluation": dict(evaluation)}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for label, path in (
        ("policy", args.policy),
        ("initial profile", args.initial_profile),
    ):
        if not path.resolve().is_file():
            raise FileNotFoundError(f"missing {label}: {path.resolve()}")

    from scipy.optimize import differential_evolution

    initial_parameters = load_profile(args.initial_profile)
    bounds = candidate_bounds(initial_parameters, args)
    initial_candidate = parameters_to_candidate(initial_parameters, args.max_bias)
    population = initial_population(
        bounds, initial_candidate, args.popsize, args.search_seed
    )
    evaluator = MarginAwareReverseEvaluator(args)
    cache: dict[tuple[float, ...], tuple[ProfileParameters, dict[str, Any]]] = {}

    def evaluate_vector(candidate: Sequence[float]) -> tuple[ProfileParameters, dict[str, Any]]:
        key = tuple(np.round(np.asarray(candidate, dtype=np.float64), 9))
        if key not in cache:
            parameters = candidate_to_parameters(candidate, args.max_bias)
            evaluation = evaluator.evaluate_candidate(
                parameters,
                args.perturb_seeds,
                args.seconds,
                args.target_vx_min,
                args.target_vx_max,
            )
            cache[key] = (parameters, evaluation)
        return cache[key]

    baseline_evaluation = evaluator.evaluate_candidate(
        initial_parameters,
        args.perturb_seeds,
        args.seconds,
        args.target_vx_min,
        args.target_vx_max,
    )
    print(
        "baseline "
        f"cost={baseline_evaluation['cost']:.6f} "
        f"mean_vx={baseline_evaluation['mean_vx']:.6f} "
        f"safe={baseline_evaluation['all_hard_safety_passed']}",
        flush=True,
    )

    generation = 0

    def objective(candidate: np.ndarray) -> float:
        return float(evaluate_vector(candidate)[1]["cost"])

    def callback(intermediate_result: Any) -> None:
        nonlocal generation
        generation += 1
        _, evaluation = evaluate_vector(intermediate_result.x)
        print(
            f"generation={generation} cost={evaluation['cost']:.6f} "
            f"mean_vx={evaluation['mean_vx']:.6f} "
            f"worst_vx={evaluation['worst_seed_vx']:.6f} "
            f"safe={evaluation['all_hard_safety_passed']}",
            flush=True,
        )

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.search_seed,
        workers=1,
        updating="immediate",
        polish=False,
        callback=callback,
        tol=1e-4,
        x0=initial_candidate,
        init=population,
    )
    ranked = sorted(cache.values(), key=lambda item: item[1]["cost"])
    top_candidates = [
        summarize_candidate(parameters, evaluation)
        for parameters, evaluation in ranked[: args.top_candidates]
    ]

    commands = (args.target_vx_max, args.target_vx_min)
    baseline_routed = evaluator.routed_validation(
        initial_parameters,
        args.perturb_seeds,
        args.validation_seconds,
        commands,
    )
    routed_finalists = []
    selected_parameters: ProfileParameters | None = None
    selected_evaluation: dict[str, Any] | None = None
    selected_routed: dict[str, Any] | None = None
    for rank, (parameters, evaluation) in enumerate(
        ranked[: args.routed_finalists], start=1
    ):
        routed = evaluator.routed_validation(
            parameters,
            args.perturb_seeds,
            args.validation_seconds,
            commands,
        )
        finalist_passed = bool(
            routed["all_hard_safety_passed"]
            and routed["reverse_motion_retained"]
        )
        routed_finalists.append(
            {
                "fast_search_rank": rank,
                "passed_pilot_gate": finalist_passed,
                "parameters": parameters.as_json(),
                "fast_multi_seed_evaluation": evaluation,
                "routed_validation": routed,
            }
        )
        if finalist_passed and selected_parameters is None:
            selected_parameters = parameters
            selected_evaluation = evaluation
            selected_routed = routed
    if selected_parameters is None:
        selected_parameters, selected_evaluation = ranked[0]
        selected_routed = routed_finalists[0]["routed_validation"]
    assert selected_evaluation is not None and selected_routed is not None
    selected_passed = bool(
        selected_routed["all_hard_safety_passed"]
        and selected_routed["reverse_motion_retained"]
    )

    output = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_margin_aware_reverse_profile_pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PILOT_CANDIDATE" if selected_passed else "PILOT_NOT_ACCEPTED",
        "hardware_deployment": "PROHIBITED",
        "simulation_only": True,
        "optimizer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "method": "scipy_differential_evolution",
            "success": bool(result.success),
            "message": str(result.message),
            "evaluations": int(result.nfev),
            "generations_completed": int(result.nit),
            "search_seed": int(args.search_seed),
            "population_size": int(len(population)),
            "maxiter": int(args.maxiter),
            "popsize_multiplier": int(args.popsize),
            "routed_finalist_count": min(
                int(args.routed_finalists), len(ranked)
            ),
            "bounds": [list(bound) for bound in bounds],
            "max_bias_rad": float(args.max_bias),
            "bias_search_radius_rad": args.bias_search_radius,
        },
        "fixed_constraints": {
            "leg_target_margin_rad": TARGET_MARGIN_RAD,
            "target_slew_rate_rad_per_s": TARGET_SLEW_RAD_S,
            "backward_residual_scale": 0.0,
            "head_target_indices": [5, 6, 7, 8],
            "head_target_value": 0.0,
            "target_vx_interval": [args.target_vx_min, args.target_vx_max],
            "minimum_reverse_vx_gate": args.minimum_reverse_vx,
            "hard_penalty_conditions": [
                "fall",
                "nonfinite_state",
                "preclip_target_outside_physical_SAFE",
                "applied_target_outside_physical_SAFE",
                "desired_target_outside_0.050_rad_margin",
                "unauthorized_startup_margin_transition",
                "target_slew_above_2.0_rad_per_s",
                "joint_qpos_outside_physical_SAFE",
                "nonzero_applied_head_action_or_target",
            ],
        },
        "perturbation": {
            "fixed_seeds_per_candidate": [int(seed) for seed in args.perturb_seeds],
            "seed_count_per_candidate": len(args.perturb_seeds),
            "initial_joint_noise_scale": args.initial_joint_noise_scale,
            "initial_base_speed_max_mps": args.initial_base_speed,
            "reset_qpos_inward_margin_rad": args.reset_qpos_inward_margin_rad,
        },
        "timing": {
            "search_seconds_per_seed": args.seconds,
            "validation_seconds_per_seed_command": args.validation_seconds,
            "warmup_seconds": args.warmup_seconds,
            "control_dt": evaluator.runtime.CONTROL_DT,
        },
        "provenance": {
            "generated_assets": evaluator.asset_evidence,
            "model_contract": evaluator.model_evidence,
            "base_policy": {
                "path": str(args.policy.resolve()),
                "sha256": sha256_file(args.policy.resolve()),
            },
            "initial_profile": {
                "path": str(args.initial_profile.resolve()),
                "sha256": sha256_file(args.initial_profile.resolve()),
            },
        },
        "baseline": {
            "parameters": initial_parameters.as_json(),
            "fast_multi_seed_evaluation": baseline_evaluation,
            "routed_validation": baseline_routed,
        },
        "selected": {
            "passed_pilot_gate": selected_passed,
            "parameters": selected_parameters.as_json(),
            "fast_multi_seed_evaluation": selected_evaluation,
            "routed_validation": selected_routed,
        },
        "top_search_candidates": top_candidates,
        "routed_finalists": routed_finalists,
        "adoption": {
            "status": "NOT_ADOPTED_DIAGNOSTIC_ONLY",
            "reason": (
                "A margin/slew diagnostic cannot alter the frozen exp_004 "
                "contract or authorize hardware deployment."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": output["status"],
                "evaluations": result.nfev,
                "baseline_routed_mean_vx": baseline_routed["mean_vx"],
                "selected_routed_mean_vx": selected_routed["mean_vx"],
                "selected_worst_vx": selected_routed["worst_seed_command_vx"],
                "selected_all_hard_safety_passed": selected_routed[
                    "all_hard_safety_passed"
                ],
                "selected_reverse_motion_retained": selected_routed[
                    "reverse_motion_retained"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
