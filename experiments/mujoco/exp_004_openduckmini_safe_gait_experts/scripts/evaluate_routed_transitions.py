"""Formal CPU evaluation for exp_004's routed OpenDuckMini policy bank.

This entrypoint always uses the generated exact hardware-safe scene and keeps
the hardware gate PROHIBITED.  MuJoCo and ONNX Runtime are imported lazily so
the suite/metrics contract can be tested without touching a running GPU job.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from router import (  # noqa: E402
    ATOMIC_EXPERTS,
    PROHIBITED_EXPERTS,
    REVERSE_TURN_ENDPOINTS,
    SafeGaitRouter,
)
from safe_gait_experts.contract import (  # noqa: E402
    ACTUATOR_JOINT_ORDER,
    CONTRACT,
    HEAD_JOINTS,
    LEG_TARGET_MARGIN_RAD,
    RESET_NOISE_MARGIN_RAD,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
    TARGET_SLEW_LIMIT_RAD_PER_S,
)
from safe_gait_experts.gait_quality import (  # noqa: E402
    GaitQualityAccumulator,
    GaitQualitySubstep,
    gait_quality_acceptance,
)
from safe_gait_experts.routed_evaluation import (  # noqa: E402
    AcceptanceThresholds,
    BACKWARD_FAMILY_EXPERTS,
    COMPOUND_CASES,
    CURRENT_FORMAL_REVERSE_ENDPOINT_MPS,
    DIAGNOSTIC_REVERSE_V3_PROFILE_PATH,
    DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_PATH,
    DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_SHA256,
    DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_SOURCE_REVERSE_ENDPOINT_MPS,
    EVALUATOR_ID,
    DIAGNOSTIC_REVERSE_PHASE_ENTRY_EVIDENCE_PATH,
    DIAGNOSTIC_REVERSE_PHASE_ENTRY_SOURCE_REVERSE_ENDPOINT_MPS,
    FORMAL_CANDIDATE_MASTER_SEED,
    FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
    FORMAL_CANDIDATE_PROFILE_PATHS,
    FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES,
    FORMAL_ADOPTION_EVIDENCE_PATH,
    FORMAL_ADOPTION_EVIDENCE_SHA256,
    FORMAL_CANDIDATE_SELECTION_EVIDENCE_PATH,
    FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256,
    FORMAL_CANDIDATE_STATUS,
    FORMAL_H3_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS,
    FROZEN_DIAGNOSTIC_REVERSE_ENTRY_PHASE_INDICES,
    FROZEN_RUNTIME_BINARY_SHA256,
    FROZEN_RUNTIME_VERSIONS,
    H1_REJECTED_COUPLED_CAP_EVIDENCE_PATH,
    H1_STRAIGHT_20X30_EVIDENCE_PATH,
    H1_TRANSITION_PREFIX_20SEED_EVIDENCE_PATH,
    HISTORICAL_FAILED_FORMAL_CANDIDATE_EVIDENCE_PATH,
    H2_COMPONENT_SELECTION_EVIDENCE_PATH,
    H2_5X15_SELECTION_EVIDENCE_PATH,
    H2_5X15_SELECTION_EVIDENCE_SHA256,
    H2_SUPERSEDED_ADOPTION_EVIDENCE_PATH,
    H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256,
    H3_FAST_EXIT_SAFETY_EVIDENCE_PATH,
    H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256,
    MAX_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
    POLICY_COMMAND_DIAGNOSTIC_CASES,
    PhysicsSubstepAudit,
    PRIMITIVE_CASES,
    REQUIRED_POLICY_ROLES,
    REJECTED_POLICY_COMMAND_DIAGNOSTIC_CASES,
    REVERSE_V1_ADOPTION_STATUS,
    REVERSE_V1_MEASURED_FORWARD_VELOCITY_MPS,
    SCHEMA_VERSION,
    SafetyAudit,
    TRANSITION_CASES,
    advance_routed_phase,
    audit_control_first_startup,
    audit_reset_qpos,
    blend_and_mask_actions,
    build_target_envelope,
    canonical_policy_role,
    command_case_validation_gate,
    compute_motion_metrics,
    capture_runtime_source_dependency_closure,
    derive_reverse_profile_adoption,
    generated_asset_paths,
    hardware_gate,
    parse_policy_assignments,
    resolve_policy_observation_command,
    sha256_file,
    suite_acceptance,
    summarize_backward_exit_recovery_steps,
    transition_schedule,
    validate_adopted_reverse_profiles,
    validate_diagnostic_unadopted_reverse_profile,
    validate_diagnostic_unadopted_reverse_turn_profile,
    validate_diagnostic_backward_exit_recovery_evidence,
    validate_diagnostic_backward_exit_recovery_execution_bundle,
    validate_diagnostic_reverse_entry_phase_indices,
    validate_diagnostic_reverse_phase_entry_evidence,
    validate_exact_generated_assets,
    validate_formal_candidate_execution_bundle,
    validate_formal_candidate_reverse_profiles,
    validate_formal_candidate_selection_evidence,
    validate_formal_adoption_evidence,
    validate_superseded_h2_adoption_evidence,
    validate_h3_fast_exit_safety_evidence,
    validate_frozen_runtime_source_dependencies,
    validate_policy_provenance,
    validate_runtime_versions,
)
from safe_gait_experts.safe_randomization import (  # noqa: E402
    actuator_name_to_index,
    build_qpos_noise_scale,
    clip_reset_qpos_to_physical_safe_limits,
)
from target_safety import (  # noqa: E402
    BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT,
    BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
    BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
    BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
    BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
    BackwardExitRecovery,
    FinalTargetSafetyGuard,
    RUNTIME_TARGET_SAFETY_MARGIN_RAD,
    RUNTIME_TARGET_SLEW_RATE_RAD_S,
    apply_final_target_safety,
    backward_exit_recovery_contract,
)


DEFAULT_GENERATED_ROOT = EXP_ROOT / "artifacts" / "generated_playground"
DEFAULT_OUTPUT = EXP_ROOT / "artifacts" / "routed_transition_acceptance.json"
FORMAL_FIXED_BACKWARD_PROFILE = (
    EXP_ROOT / "artifacts" / "optimized_reverse_exact_safe_v1.json"
)
FORMAL_FIXED_BACKWARD_LEFT_PROFILE = (
    EXP_ROOT / "artifacts" / "optimized_reverse_left_exact_safe_v1.json"
)
DEFAULT_BACKWARD_PROFILE = FORMAL_CANDIDATE_PROFILE_PATHS["straight"]
DEFAULT_BACKWARD_LEFT_PROFILE = FORMAL_CANDIDATE_PROFILE_PATHS["left"]
DEFAULT_BACKWARD_RIGHT_PROFILE = FORMAL_CANDIDATE_PROFILE_PATHS["right"]
RELEASE_QUALIFICATION_CONFIGURATION = {
    "episodes": 20,
    "seconds": 30.0,
    "transition_seconds": 30.0,
    "transition_stand_seconds": 5.0,
    "warmup_seconds": 1.5,
    "initial_joint_noise_scale": 1.0,
    "initial_base_speed": 0.10,
    "recommended_master_seed": FORMAL_CANDIDATE_MASTER_SEED,
}
LEGACY_EVALUATOR_PATH = (
    WORKSPACE
    / "experiments"
    / "mujoco"
    / "exp_003_openduckmini_calibrated_walk"
    / "evaluate_official_policy.py"
)
OWN_RUNTIME_SOURCE_PATHS = {
    "exp004_routed_evaluator": Path(__file__).resolve(),
    "exp004_router": (EXP_ROOT / "router.py").resolve(),
    "exp004_target_safety": (EXP_ROOT / "target_safety.py").resolve(),
    "safe_gait_experts_package_init": (
        EXP_ROOT / "safe_gait_experts" / "__init__.py"
    ).resolve(),
    "safe_gait_experts_contract": (
        EXP_ROOT / "safe_gait_experts" / "contract.py"
    ).resolve(),
    "safe_gait_experts_contract_json": (
        EXP_ROOT / "contract.json"
    ).resolve(),
    "safe_gait_experts_gait_quality": (
        EXP_ROOT / "safe_gait_experts" / "gait_quality.py"
    ).resolve(),
    "safe_gait_experts_reward": (
        EXP_ROOT / "safe_gait_experts" / "reward.py"
    ).resolve(),
    "safe_gait_experts_routed_evaluation": (
        EXP_ROOT / "safe_gait_experts" / "routed_evaluation.py"
    ).resolve(),
    "safe_gait_experts_safe_randomization": (
        EXP_ROOT / "safe_gait_experts" / "safe_randomization.py"
    ).resolve(),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 7 primitives, compound anchors, and continuous command "
            "transitions using exp_004's exact-safe scene."
        )
    )
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        metavar="ROLE=PATH",
        help=(
            "ONNX policy for one routed role. Repeat for stand, forward, reverse, "
            "lateral_left/right, yaw_left/right, and compound."
        ),
    )
    parser.add_argument(
        "--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT
    )
    parser.add_argument(
        "--diagnostic-unadopted-policy",
        action="store_true",
        help=(
            "Permit non-base-v22 ONNX hashes only as explicit diagnostic inputs. "
            "This mode is never eligible for adoption."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Duration of every independently reset primitive/compound case.",
    )
    parser.add_argument("--transition-seconds", type=float, default=6.0)
    parser.add_argument("--transition-stand-seconds", type=float, default=3.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument(
        "--initial-joint-noise-scale",
        type=float,
        default=1.0,
        help="Multiplier for the contract's per-joint qpos noise; head remains zero.",
    )
    parser.add_argument("--initial-base-speed", type=float, default=0.10)
    parser.add_argument(
        "--leg-target-margin-rad",
        type=float,
        default=RUNTIME_TARGET_SAFETY_MARGIN_RAD,
        help=(
            "Inward margin applied to every leg actuator target; formal adoption "
            f"is frozen at {RUNTIME_TARGET_SAFETY_MARGIN_RAD:.3f} rad. "
            "Head remains exactly zero."
        ),
    )
    parser.add_argument(
        "--target-slew-rate-rad-s",
        type=float,
        default=RUNTIME_TARGET_SLEW_RATE_RAD_S,
        help=(
            "Maximum leg target slew; formal adoption is frozen at "
            f"{RUNTIME_TARGET_SLEW_RATE_RAD_S:.2f} rad/s."
        ),
    )
    parser.add_argument(
        "--diagnostic-noncontract-safety",
        action="store_true",
        help=(
            "Allow positive non-frozen target margin/slew for simulation-only "
            "diagnosis. This always fails adoption and never promotes hardware."
        ),
    )
    parser.add_argument(
        "--policy-command-diagnostic-suite",
        action="store_true",
        help=(
            "Run only the fixed 1-episode x 5-second low-physical/high-policy "
            "command comparison suite. This mode is never adoptable."
        ),
    )
    parser.add_argument(
        "--backward-residual-scale",
        type=float,
        default=0.0,
        help="Formal adoption requires 0.0; positive values remain non-adoptable diagnostics.",
    )
    parser.add_argument(
        "--backward-profile", type=Path, default=DEFAULT_BACKWARD_PROFILE
    )
    parser.add_argument(
        "--diagnostic-unadopted-reverse-profile",
        type=Path,
        metavar="PATH",
        help=(
            "Execute one schema-v1 reverse candidate instead of the frozen "
            "straight profile. Its hash and finite schema are recorded, but "
            "adoption is forced false."
        ),
    )
    parser.add_argument(
        "--diagnostic-unadopted-reverse-left-profile",
        type=Path,
        metavar="PATH",
        help=(
            "Execute one schema-v1 atomic reverse-left candidate. Requires "
            "--diagnostic-unadopted-reverse-profile with its exact v3 base."
        ),
    )
    parser.add_argument(
        "--diagnostic-unadopted-reverse-right-profile",
        type=Path,
        metavar="PATH",
        help=(
            "Execute one schema-v1 atomic reverse-right candidate. Requires "
            "--diagnostic-unadopted-reverse-profile with its exact v3 base."
        ),
    )
    parser.add_argument(
        "--diagnostic-unadopted-reverse-entry-phase-index",
        type=float,
        metavar="6.0",
        help=(
            "Diagnostic-only straight-reverse pre-increment phase entry. Must be "
            "specified with both turn values and remain exactly 6.0."
        ),
    )
    parser.add_argument(
        "--diagnostic-unadopted-reverse-left-entry-phase-index",
        type=float,
        metavar="4.0",
        help=(
            "Diagnostic-only reverse-left pre-increment phase entry. Must be "
            "specified with the full unadopted three-profile bank and remain 4.0."
        ),
    )
    parser.add_argument(
        "--diagnostic-unadopted-reverse-right-entry-phase-index",
        type=float,
        metavar="4.0",
        help=(
            "Diagnostic-only reverse-right pre-increment phase entry. Must be "
            "specified with the full unadopted three-profile bank and remain 4.0."
        ),
    )
    parser.add_argument(
        "--diagnostic-unadopted-backward-exit-recovery",
        action="store_true",
        help=(
            "Enable the hash-pinned, diagnostic-only 13-tick left-knee cap "
            "after backward feedforward exits. Requires the complete "
            "unadopted reverse profile bank, phase indices 6/4/4, formal "
            "target safety, base-v22 policies, and zero backward residual. "
            "This mode always fails adoption and simulation acceptance."
        ),
    )
    parser.add_argument(
        "--backward-left-profile", type=Path, default=DEFAULT_BACKWARD_LEFT_PROFILE
    )
    parser.add_argument(
        "--backward-right-profile",
        type=Path,
        default=DEFAULT_BACKWARD_RIGHT_PROFILE,
        help="Exact hash-pinned Stage-A reverse-right candidate profile.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.seconds <= 0.0 or args.transition_seconds <= 0.0:
        parser.error("--seconds and --transition-seconds must be positive")
    if args.transition_stand_seconds <= 0.0 or args.warmup_seconds < 0.0:
        parser.error("stand duration must be positive and warmup non-negative")
    if args.initial_joint_noise_scale < 0.0 or args.initial_base_speed < 0.0:
        parser.error("initial perturbations must be non-negative")
    if (
        not np.isfinite(args.leg_target_margin_rad)
        or args.leg_target_margin_rad <= 0.0
    ):
        parser.error("--leg-target-margin-rad must be finite and positive")
    try:
        build_target_envelope(leg_margin_rad=args.leg_target_margin_rad)
    except ValueError as error:
        parser.error(f"invalid --leg-target-margin-rad: {error}")
    if (
        not np.isfinite(args.target_slew_rate_rad_s)
        or args.target_slew_rate_rad_s <= 0.0
    ):
        parser.error("--target-slew-rate-rad-s must be finite and positive")
    if (
        not args.diagnostic_noncontract_safety
        and args.leg_target_margin_rad != RUNTIME_TARGET_SAFETY_MARGIN_RAD
    ):
        parser.error(
            "--leg-target-margin-rad must remain exactly "
            f"{RUNTIME_TARGET_SAFETY_MARGIN_RAD:.3f} for exp_004"
        )
    if (
        not args.diagnostic_noncontract_safety
        and args.target_slew_rate_rad_s != RUNTIME_TARGET_SLEW_RATE_RAD_S
    ):
        parser.error(
            "--target-slew-rate-rad-s must remain exactly "
            f"{RUNTIME_TARGET_SLEW_RATE_RAD_S:.2f} for exp_004"
        )
    if not 0.0 <= args.backward_residual_scale <= 0.25:
        parser.error("--backward-residual-scale must be in [0, 0.25]")
    if (
        args.diagnostic_unadopted_reverse_profile is not None
        and args.backward_profile.resolve() != DEFAULT_BACKWARD_PROFILE.resolve()
    ):
        parser.error(
            "--diagnostic-unadopted-reverse-profile cannot be combined with "
            "a non-default --backward-profile"
        )
    if (
        args.diagnostic_unadopted_reverse_left_profile is not None
        or args.diagnostic_unadopted_reverse_right_profile is not None
    ) and args.diagnostic_unadopted_reverse_profile is None:
        parser.error(
            "diagnostic reverse-turn candidates require "
            "--diagnostic-unadopted-reverse-profile with the exact v3 base"
        )
    phase_values = (
        args.diagnostic_unadopted_reverse_entry_phase_index,
        args.diagnostic_unadopted_reverse_left_entry_phase_index,
        args.diagnostic_unadopted_reverse_right_entry_phase_index,
    )
    if any(value is not None for value in phase_values):
        if not all(value is not None for value in phase_values):
            parser.error(
                "diagnostic reverse entry phase indices must be supplied together"
            )
        if any(
            profile is None
            for profile in (
                args.diagnostic_unadopted_reverse_profile,
                args.diagnostic_unadopted_reverse_left_profile,
                args.diagnostic_unadopted_reverse_right_profile,
            )
        ):
            parser.error(
                "diagnostic reverse entry phase indices require the full "
                "unadopted straight/left/right profile bank"
            )
        try:
            args.diagnostic_reverse_entry_phase_indices = (
                validate_diagnostic_reverse_entry_phase_indices(
                    {
                        "reverse": phase_values[0],
                        "reverse_turn_left": phase_values[1],
                        "reverse_turn_right": phase_values[2],
                    }
                )
            )
        except ValueError as error:
            parser.error(str(error))
    else:
        args.diagnostic_reverse_entry_phase_indices = None
    if args.diagnostic_unadopted_backward_exit_recovery:
        if any(
            profile is None
            for profile in (
                args.diagnostic_unadopted_reverse_profile,
                args.diagnostic_unadopted_reverse_left_profile,
                args.diagnostic_unadopted_reverse_right_profile,
            )
        ) or args.diagnostic_reverse_entry_phase_indices is None:
            parser.error(
                "diagnostic backward-exit recovery requires the complete "
                "unadopted straight/left/right profile bank and phase 6/4/4"
            )
        if (
            args.diagnostic_noncontract_safety
            or args.leg_target_margin_rad != RUNTIME_TARGET_SAFETY_MARGIN_RAD
            or args.target_slew_rate_rad_s != RUNTIME_TARGET_SLEW_RATE_RAD_S
        ):
            parser.error(
                "diagnostic backward-exit recovery requires formal target "
                "margin 0.050 and slew 2.0"
            )
        if args.backward_residual_scale != 0.0:
            parser.error(
                "diagnostic backward-exit recovery requires zero backward residual"
            )
        if args.diagnostic_unadopted_policy:
            parser.error(
                "diagnostic backward-exit recovery requires formal base-v22 policies"
            )
        if args.policy_command_diagnostic_suite:
            parser.error(
                "diagnostic backward-exit recovery requires the full routed suites"
            )
    if args.warmup_seconds >= min(
        args.seconds, args.transition_seconds, args.transition_stand_seconds
    ):
        parser.error("--warmup-seconds must be shorter than every segment")
    if args.policy_command_diagnostic_suite and args.warmup_seconds >= 5.0:
        parser.error("--warmup-seconds must be shorter than the 5-second diagnostic")
    args.formal_candidate_default = bool(
        not args.diagnostic_noncontract_safety
        and not args.diagnostic_unadopted_policy
        and not args.policy_command_diagnostic_suite
        and args.backward_residual_scale == 0.0
        and args.diagnostic_unadopted_reverse_profile is None
        and args.diagnostic_unadopted_reverse_left_profile is None
        and args.diagnostic_unadopted_reverse_right_profile is None
        and args.diagnostic_reverse_entry_phase_indices is None
        and not args.diagnostic_unadopted_backward_exit_recovery
    )
    return args


def classify_evaluation_scale(args: argparse.Namespace) -> dict[str, Any]:
    """Separate screening runs from the frozen 20x30 release qualification."""

    expected = RELEASE_QUALIFICATION_CONFIGURATION
    actual = {
        "episodes": int(args.episodes),
        "seconds": float(args.seconds),
        "transition_seconds": float(args.transition_seconds),
        "transition_stand_seconds": float(args.transition_stand_seconds),
        "warmup_seconds": float(args.warmup_seconds),
        "initial_joint_noise_scale": float(args.initial_joint_noise_scale),
        "initial_base_speed": float(args.initial_base_speed),
        "master_seed": int(args.seed),
    }
    scale_keys = (
        "episodes",
        "seconds",
        "transition_seconds",
        "transition_stand_seconds",
        "warmup_seconds",
        "initial_joint_noise_scale",
        "initial_base_speed",
    )
    master_seed_matches = bool(
        actual["master_seed"] == expected["recommended_master_seed"]
    )
    scale_matches = bool(
        all(actual[key] == expected[key] for key in scale_keys)
        and master_seed_matches
    )
    diagnostic_mode = bool(
        args.diagnostic_noncontract_safety
        or args.policy_command_diagnostic_suite
        or args.diagnostic_unadopted_policy
        or args.diagnostic_unadopted_reverse_profile is not None
        or args.diagnostic_unadopted_reverse_left_profile is not None
        or args.diagnostic_unadopted_reverse_right_profile is not None
        or args.diagnostic_reverse_entry_phase_indices is not None
        or args.diagnostic_unadopted_backward_exit_recovery
        or args.backward_residual_scale != 0.0
    )
    eligible = bool(scale_matches and not diagnostic_mode)
    return {
        "status": "RELEASE_QUALIFICATION" if eligible else "SCREENING_CANDIDATE",
        "release_qualification_eligible": eligible,
        "scale_matches_frozen_contract": scale_matches,
        "diagnostic_mode_disabled": not diagnostic_mode,
        "master_seed_matches_recommendation": master_seed_matches,
        "master_seed_is_hard_gate": True,
        "expected": dict(expected),
        "actual": actual,
        "screening_cannot_promote_release_or_adoption": not eligible,
    }


def _runtime_binary_dependency_paths(
    mujoco: Any, onnxruntime: Any
) -> dict[str, Path]:
    """Resolve the exact native runtime files used by the formal WSL process."""

    numpy_core = importlib.import_module("numpy._core._multiarray_umath")
    ort_pybind = importlib.import_module(
        "onnxruntime.capi.onnxruntime_pybind11_state"
    )
    mujoco_root = Path(mujoco.__file__).resolve().parent
    ort_root = Path(onnxruntime.__file__).resolve().parent
    paths = {
        "python_executable": Path(sys.executable).resolve(),
        "libmujoco": (
            mujoco_root / f"libmujoco.so.{mujoco.__version__}"
        ).resolve(),
        "libonnxruntime": (
            ort_root / "capi" / f"libonnxruntime.so.{onnxruntime.__version__}"
        ).resolve(),
        "onnxruntime_pybind": Path(ort_pybind.__file__).resolve(),
        "numpy_core": Path(numpy_core.__file__).resolve(),
    }
    prefix = Path(sys.prefix).resolve()
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing formal runtime binary {label}: {path}")
        if label != "python_executable":
            try:
                path.relative_to(prefix)
            except ValueError as error:
                raise ValueError(
                    f"formal runtime binary {label} must be under sys.prefix {prefix}"
                ) from error
    return paths


def _load_runtime(
    *, include_provenance: bool = False
) -> tuple[Any, Any, Any] | tuple[Any, Any, Any, dict[str, Any]]:
    """Hard-gate source/version provenance before importing the CPU evaluator."""

    external_preimport = validate_frozen_runtime_source_dependencies()
    own_preimport = capture_runtime_source_dependency_closure(
        OWN_RUNTIME_SOURCE_PATHS
    )
    import mujoco
    import onnxruntime

    runtime_versions = validate_runtime_versions(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "onnxruntime": onnxruntime.__version__,
        }
    )
    runtime_binary_paths = _runtime_binary_dependency_paths(mujoco, onnxruntime)
    runtime_binaries_preimport = capture_runtime_source_dependency_closure(
        runtime_binary_paths,
        expected_sha256=FROZEN_RUNTIME_BINARY_SHA256,
    )
    onnxruntime_build_info = str(onnxruntime.get_build_info())
    if "45de2a8b06" not in onnxruntime_build_info:
        raise ValueError("ONNX Runtime build commit must remain 45de2a8b06")
    if not LEGACY_EVALUATOR_PATH.is_file():
        raise FileNotFoundError(f"missing reusable evaluator: {LEGACY_EVALUATOR_PATH}")
    spec = importlib.util.spec_from_file_location(
        "exp004_reused_official_policy_evaluator", LEGACY_EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evaluator from {LEGACY_EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    provenance = {
        "pre_import": {
            "external_hard_allowlisted_source_closure": external_preimport,
            "exp004_source_and_contract_snapshot": own_preimport,
            "hard_allowlisted_runtime_binary_closure": runtime_binaries_preimport,
        },
        "runtime_environment": {
            **runtime_versions,
            "platform": {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python_implementation": platform.python_implementation(),
            },
            "onnxruntime_available_providers": list(
                onnxruntime.get_available_providers()
            ),
            "onnxruntime_build_info": onnxruntime_build_info,
            "onnxruntime_build_commit_verified": "45de2a8b06",
            "sys_executable_as_invoked": sys.executable,
            "sys_executable_resolved": str(Path(sys.executable).resolve()),
            "sys_prefix_resolved": str(Path(sys.prefix).resolve()),
        },
    }
    if include_provenance:
        return mujoco, onnxruntime, module, provenance
    return mujoco, onnxruntime, module


class RoutedPolicyBank:
    """Validated CPU ONNX sessions for the eight routable policy roles."""

    def __init__(self, policies: Mapping[str, Path], onnxruntime: Any):
        if set(policies) != set(REQUIRED_POLICY_ROLES):
            raise ValueError("policy bank must contain exactly the eight required roles")
        self.paths = {role: path.resolve() for role, path in policies.items()}
        self.sessions: dict[str, Any] = {}
        self.session_providers: dict[str, list[str]] = {}
        self.inference_counts: Counter[str] = Counter()
        for role in REQUIRED_POLICY_ROLES:
            path = self.paths[role]
            if not path.is_file():
                raise FileNotFoundError(f"missing {role} ONNX policy: {path}")
            session = onnxruntime.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            actual_providers = list(session.get_providers())
            if actual_providers != ["CPUExecutionProvider"]:
                raise ValueError(
                    f"{role} policy session must use only CPUExecutionProvider; "
                    f"got {actual_providers}"
                )
            model_input = session.get_inputs()[0]
            model_output = session.get_outputs()[0]
            if model_input.name != "obs" or model_input.shape != [1, 101]:
                raise ValueError(
                    f"{role} policy has unexpected input: "
                    f"{model_input.name} {model_input.shape}"
                )
            if model_output.shape != [1, 14]:
                raise ValueError(
                    f"{role} policy has unexpected output: {model_output.shape}"
                )
            self.sessions[role] = session
            self.session_providers[role] = actual_providers

    def infer(self, role: str, observation: np.ndarray) -> np.ndarray:
        canonical = canonical_policy_role(role)
        action = self.sessions[canonical].run(
            None, {"obs": np.asarray(observation, dtype=np.float32)[None, :]}
        )[0][0]
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (14,) or not np.all(np.isfinite(action)):
            raise ValueError(f"{canonical} policy produced an invalid action")
        self.inference_counts[canonical] += 1
        return action

    def infer_route(self, decision: Any, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from_action = self.infer(decision.blend_from_expert, observation)
        if decision.blend_to_expert == decision.blend_from_expert:
            to_action = from_action
        else:
            to_action = self.infer(decision.blend_to_expert, observation)
        return blend_and_mask_actions(from_action, to_action, decision.blend_alpha)

    def manifest(self) -> dict[str, Any]:
        return {
            role: {
                "path": str(self.paths[role]),
                "sha256": sha256_file(self.paths[role]),
                "execution_providers": self.session_providers[role],
                "cpu_only_provider_verified": (
                    self.session_providers[role] == ["CPUExecutionProvider"]
                ),
            }
            for role in REQUIRED_POLICY_ROLES
        }


class DiagnosticTargetSafetyGuard:
    """Parameterised evaluator-only guard for explicitly non-adoptable sweeps."""

    def __init__(
        self,
        initial_targets: Sequence[float],
        target_lower: Sequence[float],
        target_upper: Sequence[float],
        physical_lower: Sequence[float],
        physical_upper: Sequence[float],
        *,
        slew_rate_rad_per_s: float,
    ) -> None:
        arrays = [
            np.asarray(value, dtype=np.float64)
            for value in (
                initial_targets,
                target_lower,
                target_upper,
                physical_lower,
                physical_upper,
            )
        ]
        if any(value.shape != (14,) for value in arrays) or not all(
            np.all(np.isfinite(value)) for value in arrays
        ):
            raise ValueError("diagnostic target guard requires finite 14-axis vectors")
        initial, self.target_lower, self.target_upper, lower, upper = arrays
        if np.any(self.target_lower > self.target_upper) or np.any(lower > upper):
            raise ValueError("diagnostic target guard received invalid bounds")
        self.physical_lower = lower
        self.physical_upper = upper
        self.slew_rate_rad_per_s = float(slew_rate_rad_per_s)
        if not np.isfinite(self.slew_rate_rad_per_s) or self.slew_rate_rad_per_s <= 0.0:
            raise ValueError("diagnostic target slew must be finite and positive")
        self._previous_targets = np.clip(initial, lower, upper)
        self._steps_since_reset = 0

    @property
    def previous_targets(self) -> np.ndarray:
        return self._previous_targets.copy()

    @property
    def steps_since_reset(self) -> int:
        return self._steps_since_reset

    def desired_targets(self, targets: Sequence[float]) -> np.ndarray:
        values = np.asarray(targets, dtype=np.float64)
        if values.shape != (14,) or not np.all(np.isfinite(values)):
            raise ValueError("diagnostic desired targets must be one finite 14-axis vector")
        return np.clip(values, self.target_lower, self.target_upper)

    def control_first_startup(
        self, first_command_policy_targets: Sequence[float], *, dt: float
    ) -> np.ndarray:
        if self._steps_since_reset != 0:
            raise RuntimeError("control-first startup must be the first guard call")
        return self.step(first_command_policy_targets, dt)

    def step(self, targets: Sequence[float], dt: float) -> np.ndarray:
        dt_value = float(dt)
        if not np.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError("diagnostic target guard dt must be finite and positive")
        desired = self.desired_targets(targets)
        maximum_delta = self.slew_rate_rad_per_s * dt_value
        applied = np.clip(
            desired,
            self._previous_targets - maximum_delta,
            self._previous_targets + maximum_delta,
        )
        applied = np.clip(applied, self.physical_lower, self.physical_upper)
        self._previous_targets = applied
        self._steps_since_reset += 1
        return applied.copy()


def apply_control_first_startup(
    target_guard: Any,
    data_ctrl: Any,
    desired_targets: Sequence[float],
    joint_names: Sequence[str],
    *,
    control_dt: float,
    leg_target_margin_rad: float,
    target_slew_rate_rad_s: float,
    physics_steps_before_control: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Commit the first routed policy control with exactly one guard call."""

    reset_targets = target_guard.previous_targets
    guard_calls_before = int(target_guard.steps_since_reset)
    applied_targets = target_guard.control_first_startup(
        desired_targets,
        dt=control_dt,
    )
    guard_calls_for_first_tick = (
        int(target_guard.steps_since_reset) - guard_calls_before
    )
    data_ctrl[:] = applied_targets
    startup_audit = audit_control_first_startup(
        reset_targets,
        desired_targets,
        applied_targets,
        joint_names,
        control_dt=control_dt,
        leg_target_margin_rad=leg_target_margin_rad,
        target_slew_limit_rad_per_s=target_slew_rate_rad_s,
        physics_steps_before_control=physics_steps_before_control,
        guard_calls_before_control=guard_calls_before,
        guard_calls_for_first_tick=guard_calls_for_first_tick,
    )
    if not startup_audit["passed"]:
        raise RuntimeError(f"unsafe control-first startup: {startup_audit}")
    return np.asarray(applied_targets, dtype=np.float64).copy(), startup_audit


def apply_guarded_control_then_step_physics(
    target_guard: Any,
    desired_targets: Sequence[float],
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    decimation: int,
    control_dt: float,
    joint_names: Sequence[str],
    leg_target_margin_rad: float,
    target_slew_rate_rad_s: float,
    first_control_tick: bool,
    physics_steps_before_control: int,
    physics_substep_callback: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None, int, bool]:
    """Apply one guarded control, then and only then advance physics."""

    decimation_steps = int(decimation)
    if decimation_steps != decimation or decimation_steps <= 0:
        raise ValueError("decimation must be a positive integer")
    previous_targets = target_guard.previous_targets
    if first_control_tick:
        applied_targets, startup_audit = apply_control_first_startup(
            target_guard,
            data.ctrl,
            desired_targets,
            joint_names,
            control_dt=control_dt,
            leg_target_margin_rad=leg_target_margin_rad,
            target_slew_rate_rad_s=target_slew_rate_rad_s,
            physics_steps_before_control=physics_steps_before_control,
        )
    else:
        steps_before = int(target_guard.steps_since_reset)
        applied_targets = target_guard.step(desired_targets, control_dt)
        if int(target_guard.steps_since_reset) - steps_before != 1:
            raise RuntimeError("target guard must be called exactly once per tick")
        data.ctrl[:] = applied_targets
        startup_audit = None
    completed_substeps = 0
    terminated = False
    for _ in range(decimation_steps):
        mujoco.mj_step(model, data)
        completed_substeps += 1
        if physics_substep_callback is not None and physics_substep_callback():
            terminated = True
            break
    return (
        previous_targets,
        applied_targets,
        startup_audit,
        completed_substeps,
        terminated,
    )


def synchronize_telemetry_data(
    mujoco: Any, model: Any, source_data: Any, telemetry_data: Any
) -> Any:
    """Build a coherent read-only snapshot at the newly integrated state."""

    mujoco.mj_copyData(telemetry_data, model, source_data)
    mujoco.mj_forward(model, telemetry_data)
    return telemetry_data


def validate_model_contract(evaluator: Any) -> dict[str, Any]:
    """Prove the loaded MuJoCo model is the 14-axis SAFE_INIT/SAFE_LIMIT model."""

    model = evaluator.model
    if int(model.nu) != 14:
        raise ValueError(f"exact-safe model must have 14 actuators, got {model.nu}")
    joint_names = tuple(model.actuator(index).name for index in range(model.nu))
    if joint_names != tuple(ACTUATOR_JOINT_ORDER):
        raise ValueError(
            "exact-safe actuator order mismatch: "
            f"expected {ACTUATOR_JOINT_ORDER}, got {joint_names}"
        )
    home = model.keyframe("home")
    home_target = np.asarray(home.ctrl, dtype=np.float64)
    expected_home = np.asarray([SAFE_INIT_POS[name] for name in joint_names])
    home_error = np.abs(home_target - expected_home)
    if not np.allclose(home_target, expected_home, atol=1e-9, rtol=0.0):
        raise ValueError(f"scene home does not match SAFE_INIT; max error={home_error.max()}")

    ranges = np.asarray(model.jnt_range[evaluator.actuator_joint_ids], dtype=np.float64)
    leg_range_errors: dict[str, list[float]] = {}
    for index, name in enumerate(joint_names):
        if name in HEAD_JOINTS:
            if home_target[index] != 0.0:
                raise ValueError(f"head home target must be zero: {name}")
            continue
        expected = np.asarray(SAFE_JOINT_LIMITS[name], dtype=np.float64)
        error = np.abs(ranges[index] - expected)
        leg_range_errors[name] = error.tolist()
        if not np.allclose(ranges[index], expected, atol=1e-9, rtol=0.0):
            raise ValueError(f"scene joint range does not match SAFE limits: {name}")
    return {
        "actuator_joint_order": list(joint_names),
        "home_matches_safe_init": True,
        "maximum_home_error_rad": float(np.max(home_error)),
        "leg_ranges_match_safe_limits": True,
        "leg_range_absolute_errors_rad": leg_range_errors,
        "head_home_targets_zero": True,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
    }


class RoutedSimulator:
    """One deterministic CPU MuJoCo simulator with routed ONNX inference."""

    def __init__(
        self,
        evaluator: Any,
        bank: RoutedPolicyBank,
        mujoco: Any,
        runtime_module: Any,
        *,
        leg_target_margin_rad: float,
        target_slew_rate_rad_s: float,
        diagnostic_noncontract_safety: bool,
        left_knee_extra_upper_margin_rad: float = 0.0,
        diagnostic_reverse_entry_phase_indices: Mapping[str, float] | None = None,
        diagnostic_unadopted_backward_exit_recovery: bool = False,
        formal_candidate_default: bool = False,
    ):
        self.evaluator = evaluator
        self.bank = bank
        self.mujoco = mujoco
        self.runtime = runtime_module
        self.model = evaluator.model
        self.telemetry_data = self.mujoco.MjData(self.model)
        gravity_magnitude = float(np.linalg.norm(self.model.opt.gravity))
        self.robot_weight_n = float(np.sum(self.model.body_mass)) * gravity_magnitude
        if not np.isfinite(self.robot_weight_n) or self.robot_weight_n <= 0.0:
            raise ValueError("model must have positive finite body weight")
        self.left_foot_site_id = int(self.model.site("left_foot").id)
        self.right_foot_site_id = int(self.model.site("right_foot").id)
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

        self.joint_ranges = np.asarray(
            self.model.jnt_range[evaluator.actuator_joint_ids], dtype=np.float64
        )
        self.safe_lower = self.joint_ranges[:, 0].copy()
        self.safe_upper = self.joint_ranges[:, 1].copy()
        for index, name in enumerate(self.joint_names):
            if name in HEAD_JOINTS:
                self.safe_lower[index] = 0.0
                self.safe_upper[index] = 0.0
            else:
                self.safe_lower[index], self.safe_upper[index] = SAFE_JOINT_LIMITS[name]
        self.leg_target_margin_rad = float(leg_target_margin_rad)
        self.target_slew_rate_rad_s = float(target_slew_rate_rad_s)
        self.diagnostic_noncontract_safety = bool(diagnostic_noncontract_safety)
        if not isinstance(formal_candidate_default, (bool, np.bool_)):
            raise ValueError("formal-candidate default flag must be boolean")
        self.formal_candidate_default = bool(formal_candidate_default)
        if not isinstance(
            diagnostic_unadopted_backward_exit_recovery, (bool, np.bool_)
        ):
            raise ValueError(
                "diagnostic backward-exit recovery flag must be boolean"
            )
        if self.formal_candidate_default and (
            diagnostic_reverse_entry_phase_indices is not None
            or diagnostic_unadopted_backward_exit_recovery
        ):
            raise ValueError(
                "formal-candidate default cannot be combined with diagnostic "
                "phase/recovery history flags"
            )
        self.diagnostic_unadopted_backward_exit_recovery = bool(
            diagnostic_unadopted_backward_exit_recovery
        )
        self.diagnostic_reverse_entry_phase_indices = (
            validate_diagnostic_reverse_entry_phase_indices(
                diagnostic_reverse_entry_phase_indices
            )
        )
        self.reverse_entry_phase_indices = (
            dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
            if self.formal_candidate_default
            else self.diagnostic_reverse_entry_phase_indices
        )
        self.backward_exit_recovery_enabled = bool(
            BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT
            if self.formal_candidate_default
            else self.diagnostic_unadopted_backward_exit_recovery
        )
        self.phase_entry_status = (
            FORMAL_CANDIDATE_STATUS
            if self.formal_candidate_default
            else "DIAGNOSTIC_UNADOPTED"
        )
        self.phase_entry_diagnostic_only = not self.formal_candidate_default
        if self.reverse_entry_phase_indices is not None and any(
            phase_index >= float(self.evaluator.phase_steps)
            for phase_index in self.reverse_entry_phase_indices.values()
        ):
            raise ValueError("reverse entry phase index must be below phase_steps")
        self.target_lower, self.target_upper = build_target_envelope(
            self.joint_names, leg_margin_rad=self.leg_target_margin_rad
        )
        extra_margin = float(left_knee_extra_upper_margin_rad)
        if (
            not np.isfinite(extra_margin)
            or not 0.0 <= extra_margin <= MAX_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        ):
            raise ValueError(
                "left-knee extra upper margin must be finite and in "
                f"[0, {MAX_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD}]"
            )
        self.left_knee_extra_upper_margin_rad = extra_margin
        self.left_knee_index = self.joint_names.index("left_knee")
        self.left_knee_profile_upper_target_rad = (
            float(SAFE_JOINT_LIMITS["left_knee"][1])
            - self.leg_target_margin_rad
            - self.left_knee_extra_upper_margin_rad
        )
        if (
            self.left_knee_profile_upper_target_rad
            < self.target_lower[self.left_knee_index]
        ):
            raise ValueError("left-knee profile cap collapsed the target envelope")
        if self.backward_exit_recovery_enabled and (
            self.diagnostic_noncontract_safety
            or self.leg_target_margin_rad != RUNTIME_TARGET_SAFETY_MARGIN_RAD
            or self.target_slew_rate_rad_s != RUNTIME_TARGET_SLEW_RATE_RAD_S
            or self.left_knee_extra_upper_margin_rad
            != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            or self.reverse_entry_phase_indices
            != dict(
                FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
                if self.formal_candidate_default
                else FROZEN_DIAGNOSTIC_REVERSE_ENTRY_PHASE_INDICES
            )
        ):
            raise ValueError(
                "backward-exit recovery requires the frozen "
                "margin/slew/profile-cap/route-phase bundle"
            )

    def _quality_contact_kinematics(
        self, data: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-foot normalized normal force and tangential contact speed."""

        body_to_foot = {
            int(self.evaluator.left_foot_body_id): 0,
            int(self.evaluator.right_foot_body_id): 1,
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
            body_1 = int(self.model.geom_bodyid[contact.geom1])
            body_2 = int(self.model.geom_bodyid[contact.geom2])
            if body_1 == int(self.evaluator.floor_body_id) and body_2 in body_to_foot:
                foot_body, other_body = body_2, body_1
            elif body_2 == int(self.evaluator.floor_body_id) and body_1 in body_to_foot:
                foot_body, other_body = body_1, body_2
            else:
                continue
            force = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(self.model, data, contact_index, force)
            force_n = max(0.0, abs(float(force[0])))
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
            foot = body_to_foot[foot_body]
            normal_force_n[foot] += force_n
            force_weighted_tangential_speed[foot] += force_n * tangential_speed
        tangential_speed = np.divide(
            force_weighted_tangential_speed,
            normal_force_n,
            out=np.zeros(2, dtype=np.float64),
            where=normal_force_n > 0.0,
        )
        return normal_force_n / self.robot_weight_n, tangential_speed

    def _initial_data(
        self, seed: int, joint_noise_scale: float, initial_base_speed: float
    ) -> tuple[Any, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        data = self.mujoco.MjData(self.model)
        home = self.model.keyframe("home")
        data.qpos[:] = home.qpos
        noise = rng.uniform(-1.0, 1.0, size=14) * (
            joint_noise_scale * self.qpos_noise_scale
        )
        noisy_qpos = data.qpos[self.evaluator.actuator_qpos_addr] + noise
        reset_qpos = clip_reset_qpos_to_physical_safe_limits(
            noisy_qpos,
            self.joint_names,
            noise_applied=joint_noise_scale > 0.0,
            reset_noise_margin_rad=RESET_NOISE_MARGIN_RAD,
        )
        data.qpos[self.evaluator.actuator_qpos_addr] = reset_qpos
        # Preserve exact SAFE_INIT at zero noise (or the physically SAFE noisy
        # reset) as the currently applied target.  The stateful guard performs
        # the visible 2.0 rad/s transition into the inward target envelope.
        data.ctrl[:] = reset_qpos
        if initial_base_speed > 0.0:
            angle = rng.uniform(-np.pi, np.pi)
            magnitude = rng.uniform(0.0, initial_base_speed)
            data.qvel[:2] = magnitude * np.asarray((np.cos(angle), np.sin(angle)))
        self.mujoco.mj_forward(self.model, data)
        reset_audit = audit_reset_qpos(
            np.asarray(
                data.qpos[self.evaluator.actuator_qpos_addr], dtype=np.float64
            ),
            self.joint_names,
            noise_applied=joint_noise_scale > 0.0,
            reset_noise_margin_rad=RESET_NOISE_MARGIN_RAD,
        )
        if not reset_audit["passed"]:
            raise RuntimeError(f"unsafe reset qpos: {reset_audit}")
        return data, reset_audit

    def _policy_target(
        self,
        applied_action: np.ndarray,
        effective_command: np.ndarray,
        phase_index: float,
        default: np.ndarray,
    ) -> np.ndarray:
        bounded = np.clip(applied_action, -1.0, 1.0)
        positive_span = 0.9 * (self.joint_ranges[:, 1] - default)
        negative_span = 0.9 * (default - self.joint_ranges[:, 0])
        directional_span = np.where(bounded >= 0.0, positive_span, negative_span)
        base_span = np.minimum(self.runtime.ACTION_SCALE, directional_span)
        magnitude = np.abs(bounded)
        target_magnitude = (
            base_span * magnitude + (directional_span - base_span) * magnitude**5
        )
        targets = default + np.sign(bounded) * target_magnitude
        if effective_command[0] < -0.02:
            gait_scales, gait_biases, _ = self.evaluator.backward_parameters(
                float(effective_command[2])
            )
            turn_blend = float(np.clip(abs(effective_command[2]) / 0.20, 0.0, 1.0))
            targets = self.evaluator._backward_feedforward(
                phase_index,
                default,
                self.joint_ranges,
                bounded,
                gait_scales=gait_scales,
                gait_biases=gait_biases,
                leg_residual_factor=max(0.50, 1.0 - 0.50 * turn_blend),
                head_residual_factor=0.0,
            )
            # Profile-specific tightening is composed before the uniform
            # stateful final guard.  The guard remains the single slew/clamp
            # authority and old profiles reproduce the exact cap with extra=0.
            targets[self.left_knee_index] = min(
                targets[self.left_knee_index],
                self.left_knee_profile_upper_target_rad,
            )
        targets[5:9] = 0.0
        return targets

    def run_schedule(
        self,
        schedule: Sequence[
            tuple[str, tuple[float, float, float], float]
            | tuple[
                str,
                tuple[float, float, float],
                float,
                tuple[float, float, float] | None,
            ]
            | tuple[
                str,
                tuple[float, float, float],
                float,
                tuple[float, float, float] | None,
                str,
                str,
            ]
        ],
        *,
        seed: int,
        joint_noise_scale: float,
        initial_base_speed: float,
        warmup_seconds: float,
    ) -> dict[str, Any]:
        data, reset_qpos_audit = self._initial_data(
            seed, joint_noise_scale, initial_base_speed
        )
        telemetry_data = synchronize_telemetry_data(
            self.mujoco, self.model, data, self.telemetry_data
        )
        home = self.model.keyframe("home")
        default = np.asarray(home.ctrl, dtype=np.float64).copy()
        initial_targets = np.asarray(
            data.qpos[self.evaluator.actuator_qpos_addr], dtype=np.float64
        ).copy()
        if self.diagnostic_noncontract_safety:
            target_guard = DiagnosticTargetSafetyGuard(
                initial_targets,
                self.target_lower,
                self.target_upper,
                self.safe_lower,
                self.safe_upper,
                slew_rate_rad_per_s=self.target_slew_rate_rad_s,
            )
        else:
            target_guard = FinalTargetSafetyGuard(
                SAFE_JOINT_LIMITS,
                initial_targets,
                margin_rad=self.leg_target_margin_rad,
                max_slew_rate_rad_s=self.target_slew_rate_rad_s,
            )
        backward_exit_recovery = BackwardExitRecovery(
            SAFE_JOINT_LIMITS,
            enabled=self.backward_exit_recovery_enabled,
        )
        # No home-only precharge is allowed.  The first routed policy/profile
        # target is the first and only guard call before the first physics tick.
        motor_targets = target_guard.previous_targets
        control_first_startup_audit: dict[str, Any] | None = None
        control_ticks_completed = 0
        physics_substeps_completed = 0
        action_history = [np.zeros(14, dtype=np.float32) for _ in range(3)]
        previous_position = telemetry_data.xpos[self.evaluator.trunk_body_id].copy()
        phase_index = 0.0
        router = SafeGaitRouter()
        previous_routed_expert: str | None = None
        previous_backward_feedforward_active = False
        gait_contact_continuity: dict[str, object] | None = None
        fell = False
        segments: list[dict[str, Any]] = []

        for segment_index, scheduled_case in enumerate(schedule):
            if len(scheduled_case) == 3:
                name, requested_command, duration = scheduled_case
                policy_observation_override = None
                expected_expert = None
                expected_policy_role = None
            elif len(scheduled_case) == 4:
                (
                    name,
                    requested_command,
                    duration,
                    policy_observation_override,
                ) = scheduled_case
                expected_expert = None
                expected_policy_role = None
            elif len(scheduled_case) == 6:
                (
                    name,
                    requested_command,
                    duration,
                    policy_observation_override,
                    expected_expert,
                    expected_policy_role,
                ) = scheduled_case
            else:
                raise ValueError("scheduled cases must contain 3, 4, or 6 fields")
            requested = np.asarray(requested_command, dtype=np.float64)
            policy_override = (
                None
                if policy_observation_override is None
                else np.asarray(policy_observation_override, dtype=np.float64)
            )
            if policy_override is not None and (
                policy_override.shape != (3,) or not np.all(np.isfinite(policy_override))
            ):
                raise ValueError("policy observation override must be a finite triplet")
            if (expected_expert is None) != (expected_policy_role is None):
                raise ValueError("expected expert and policy role must be paired")
            if expected_expert is not None and (
                canonical_policy_role(expected_expert) != expected_policy_role
                or expected_expert in PROHIBITED_EXPERTS
            ):
                raise ValueError("invalid expected routing contract")
            segment_start = telemetry_data.xpos[self.evaluator.trunk_body_id].copy()
            segment_start_time = float(telemetry_data.time)
            audit = SafetyAudit(
                self.joint_names,
                leg_target_margin_rad=self.leg_target_margin_rad,
                target_slew_limit_rad_per_s=self.target_slew_rate_rad_s,
            )
            velocities: list[np.ndarray] = []
            yaw_rates: list[float] = []
            effective_commands: list[np.ndarray] = []
            policy_observation_commands: list[np.ndarray] = []
            heights: list[float] = []
            uprights: list[float] = []
            contacts: list[np.ndarray] = []
            routed_experts: Counter[str] = Counter()
            policy_roles: Counter[str] = Counter()
            steady_routed_experts: Counter[str] = Counter()
            steady_policy_roles: Counter[str] = Counter()
            steady_state_steps = 0
            steady_prohibited_expert_steps = 0
            atomic_endpoint_mismatch_steps = 0
            switch_count = 0
            command_clip_events = 0
            prohibited_expert_steps = 0
            policy_yaw_offsets: Counter[str] = Counter()
            policy_yaw_offset_sum = 0.0
            target_steps = int(round(duration / self.runtime.CONTROL_DT))
            warmup_steps = int(round(warmup_seconds / self.runtime.CONTROL_DT))
            completed_steps = 0
            completed_segment_physics_substeps = 0
            substep_audit = PhysicsSubstepAudit(self.joint_names)
            gait_quality = GaitQualityAccumulator(joint_names=self.joint_names)
            if gait_contact_continuity is not None:
                gait_quality.restore_contact_continuity_state(
                    gait_contact_continuity
                )
            quality_effective_command = requested.copy()
            reverse_entry_phase_events: list[dict[str, Any]] = []
            backward_exit_recovery_step_audits: list[dict[str, Any]] = []
            backward_feedforward_entry_count = 0
            within_backward_family_active_switch_count = 0

            def update_gait_quality(
                sample_data: Any, time_s: float, effective_command: np.ndarray
            ) -> None:
                normal_force_fraction, contact_tangential_speed = (
                    self._quality_contact_kinematics(sample_data)
                )
                # H5's substep-alignment preflight may capture the exact
                # measurement payload consumed below.  The hook is absent for
                # every legacy/formal route, and receives copies only after
                # MuJoCo state coherence is established; it cannot affect the
                # action, target, guard, or physics trajectory.
                capture_substep_measurement = getattr(
                    self, "_capture_substep_measurement", None
                )
                if capture_substep_measurement is not None:
                    capture_substep_measurement(
                        run_seed=seed,
                        segment_index=segment_index,
                        segment_name=name,
                        time_s=float(time_s),
                        normal_force_fraction=np.asarray(
                            normal_force_fraction, dtype=np.float64
                        ).copy(),
                        tangential_speed_mps=np.asarray(
                            contact_tangential_speed, dtype=np.float64
                        ).copy(),
                    )
                trunk_rotation = np.asarray(
                    sample_data.xmat[self.evaluator.trunk_body_id],
                    dtype=np.float64,
                ).reshape(3, 3)
                trunk_yaw = float(
                    np.arctan2(trunk_rotation[1, 0], trunk_rotation[0, 0])
                )
                gait_quality.update(
                    GaitQualitySubstep(
                        time_s=time_s,
                        requested_command=requested,
                        effective_command=effective_command,
                        local_velocity_xyz_mps=self.evaluator._sensor(
                            sample_data, "local_linvel"
                        ),
                        local_yaw_rate_radps=float(
                            self.evaluator._sensor(sample_data, "gyro")[2]
                        ),
                        trunk_position_world_m=sample_data.xpos[
                            self.evaluator.trunk_body_id
                        ],
                        feet_contacts=self.evaluator._feet_contacts(sample_data),
                        foot_contact_points_world_m=np.asarray(
                            [
                                sample_data.site_xpos[self.left_foot_site_id],
                                sample_data.site_xpos[self.right_foot_site_id],
                            ],
                            dtype=np.float64,
                        ),
                        leg_joint_positions_rad=np.asarray(
                            sample_data.qpos[self.evaluator.actuator_qpos_addr],
                            dtype=np.float64,
                        ),
                        feet_normal_force_fraction_body_weight=normal_force_fraction,
                        foot_contact_tangential_speeds_mps=(
                            contact_tangential_speed
                        ),
                        trunk_yaw_world_rad=trunk_yaw,
                        trunk_pose_measurement_source=(
                            "mujoco_shadow_xpos_xmat_after_mj_forward"
                        ),
                    )
                )

            update_gait_quality(
                telemetry_data,
                0.0,
                np.asarray(router.ramped_command, dtype=np.float64),
            )

            def audit_physics_substep() -> bool:
                coherent_data = synchronize_telemetry_data(
                    self.mujoco, self.model, data, telemetry_data
                )
                substep_position = coherent_data.xpos[self.evaluator.trunk_body_id]
                substep_rotation = coherent_data.xmat[
                    self.evaluator.trunk_body_id
                ].reshape(
                    3, 3
                )
                substep_contacts = self.evaluator._feet_contacts(coherent_data)
                substep_joint_qpos = np.asarray(
                    coherent_data.qpos[self.evaluator.actuator_qpos_addr],
                    dtype=np.float64,
                )
                substep_audit.update(
                    joint_qpos=substep_joint_qpos,
                    full_qpos=np.asarray(coherent_data.qpos, dtype=np.float64),
                    full_qvel=np.asarray(coherent_data.qvel, dtype=np.float64),
                    height_m=float(substep_position[2]),
                    upright=float(substep_rotation[2, 2]),
                    feet_contacts=substep_contacts,
                )
                update_gait_quality(
                    coherent_data,
                    float(coherent_data.time) - segment_start_time,
                    quality_effective_command,
                )
                return substep_audit.termination_required

            for control_step in range(target_steps):
                decision = router.route(requested, self.runtime.CONTROL_DT)
                effective = np.asarray(decision.effective_command, dtype=np.float64)
                quality_effective_command = effective.copy()
                routed_experts[decision.expert] += 1
                policy_roles[canonical_policy_role(decision.expert)] += 1
                switch_count += int(decision.switched)
                command_clip_events += int(decision.command_was_clipped)
                prohibited_expert_steps += int(
                    decision.expert in PROHIBITED_EXPERTS
                    or decision.blend_from_expert in PROHIBITED_EXPERTS
                    or decision.blend_to_expert in PROHIBITED_EXPERTS
                )
                if control_step >= warmup_steps:
                    selected_role = canonical_policy_role(decision.expert)
                    steady_state_steps += 1
                    steady_routed_experts[decision.expert] += 1
                    steady_policy_roles[selected_role] += 1
                    steady_prohibited_expert_steps += int(
                        decision.expert in PROHIBITED_EXPERTS
                        or decision.blend_from_expert in PROHIBITED_EXPERTS
                        or decision.blend_to_expert in PROHIBITED_EXPERTS
                    )
                    if expected_expert in ATOMIC_EXPERTS:
                        expected_endpoint = np.asarray(
                            REVERSE_TURN_ENDPOINTS[expected_expert],
                            dtype=np.float64,
                        )
                        atomic_endpoint_mismatch_steps += int(
                            decision.expert != expected_expert
                            or not np.array_equal(effective, expected_endpoint)
                        )

                if effective[0] < -0.02:
                    _, _, phase_delta = self.evaluator.backward_parameters(
                        float(effective[2])
                    )
                else:
                    phase_delta = 1.0
                current_backward_feedforward_active = bool(effective[0] < -0.02)
                backward_feedforward_entry_count += int(
                    current_backward_feedforward_active
                    and not previous_backward_feedforward_active
                    and decision.expert in BACKWARD_FAMILY_EXPERTS
                )
                within_backward_family_active_switch_count += int(
                    current_backward_feedforward_active
                    and previous_backward_feedforward_active
                    and decision.expert in BACKWARD_FAMILY_EXPERTS
                    and previous_routed_expert in BACKWARD_FAMILY_EXPERTS
                    and decision.expert != previous_routed_expert
                )
                (
                    phase_index,
                    current_backward_feedforward_active,
                    phase_entry_event,
                ) = advance_routed_phase(
                    phase_index,
                    phase_steps=self.evaluator.phase_steps,
                    phase_delta=phase_delta,
                    current_expert=decision.expert,
                    previous_expert=previous_routed_expert,
                    effective_command=effective,
                    previous_backward_feedforward_active=(
                        previous_backward_feedforward_active
                    ),
                    diagnostic_entry_phase_indices=(
                        self.reverse_entry_phase_indices
                    ),
                    phase_entry_status=self.phase_entry_status,
                    diagnostic_only=self.phase_entry_diagnostic_only,
                    control_step=control_step,
                    global_control_tick=control_ticks_completed,
                )
                if phase_entry_event is not None:
                    phase_entry_event["segment"] = name
                    reverse_entry_phase_events.append(phase_entry_event)
                previous_backward_feedforward_active = (
                    current_backward_feedforward_active
                )
                previous_routed_expert = decision.expert
                phase = phase_index / self.evaluator.phase_steps * 2.0 * np.pi
                policy_command, yaw_offset, _ = resolve_policy_observation_command(
                    decision.expert,
                    effective,
                    backward_residual_scale=self.evaluator.backward_residual_scale,
                    override=policy_override,
                )
                policy_yaw_offsets[f"{yaw_offset:+.2f}"] += 1
                policy_yaw_offset_sum += yaw_offset
                command7 = np.asarray(
                    [*policy_command, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
                )
                observation = self.evaluator._observation(
                    data,
                    command7,
                    default,
                    motor_targets,
                    action_history,
                    phase,
                )
                # H5 phase-use diagnosis captures the exact actor input before
                # inference.  The optional hook is append-only and receives a
                # copy; it cannot alter routing, action generation, or physics.
                capture_policy_observation = getattr(
                    self, "_capture_policy_observation", None
                )
                if capture_policy_observation is not None:
                    capture_policy_observation(
                        run_seed=seed,
                        segment_index=segment_index,
                        segment_name=name,
                        control_step=control_step,
                        observation=np.asarray(observation, dtype=np.float32).copy(),
                    )
                raw_action, applied_action = self.bank.infer_route(decision, observation)
                action_history = [
                    applied_action.astype(np.float32),
                    action_history[0].copy(),
                    action_history[1].copy(),
                ]
                candidate_targets = self._policy_target(
                    applied_action,
                    effective,
                    phase_index,
                    default,
                )
                candidate_targets, backward_exit_recovery_step_audit = (
                    backward_exit_recovery.compose(
                        candidate_targets,
                        backward_feedforward_active=(
                            current_backward_feedforward_active
                        ),
                    )
                )
                backward_exit_recovery_step_audits.append(
                    backward_exit_recovery_step_audit
                )
                if self.diagnostic_noncontract_safety:
                    desired_targets = target_guard.desired_targets(candidate_targets)
                else:
                    desired_targets = apply_final_target_safety(
                        candidate_targets,
                        SAFE_JOINT_LIMITS,
                        margin_rad=self.leg_target_margin_rad,
                    )
                (
                    previous_targets,
                    applied_targets,
                    tick_startup_audit,
                    completed_tick_substeps,
                    substep_terminated,
                ) = apply_guarded_control_then_step_physics(
                    target_guard,
                    candidate_targets,
                    mujoco=self.mujoco,
                    model=self.model,
                    data=data,
                    decimation=self.runtime.DECIMATION,
                    control_dt=self.runtime.CONTROL_DT,
                    joint_names=self.joint_names,
                    leg_target_margin_rad=self.leg_target_margin_rad,
                    target_slew_rate_rad_s=self.target_slew_rate_rad_s,
                    first_control_tick=control_ticks_completed == 0,
                    physics_steps_before_control=physics_substeps_completed,
                    physics_substep_callback=audit_physics_substep,
                )
                if tick_startup_audit is not None:
                    tick_startup_audit.update(
                        {
                            "desired_target_source": (
                                "first routed policy/profile composition"
                            ),
                            "requested_physical_command": requested.tolist(),
                            "effective_physical_command": effective.tolist(),
                            "policy_observation_command": policy_command.tolist(),
                            "routed_expert": decision.expert,
                            "canonical_policy_role": canonical_policy_role(
                                decision.expert
                            ),
                        }
                    )
                    control_first_startup_audit = tick_startup_audit
                control_ticks_completed += 1
                physics_substeps_completed += completed_tick_substeps
                completed_segment_physics_substeps += completed_tick_substeps
                motor_targets = applied_targets.copy()

                # H5's paired command-contract screen installs this optional
                # hook to hash the exact post-control state and the complete
                # actor/target pipeline.  The frozen H3 path has no hook and
                # therefore retains its previous execution and evidence shape.
                capture_control_trace = getattr(self, "_capture_control_trace", None)
                if capture_control_trace is not None:
                    capture_control_trace(
                        segment_index=segment_index,
                        segment_name=name,
                        control_step=control_step,
                        global_control_tick=control_ticks_completed - 1,
                        requested_command=requested,
                        effective_command=effective,
                        policy_command=policy_command,
                        raw_action=raw_action,
                        candidate_targets=candidate_targets,
                        desired_targets=desired_targets,
                        applied_targets=applied_targets,
                        qpos=np.asarray(data.qpos, dtype=np.float64),
                        qvel=np.asarray(data.qvel, dtype=np.float64),
                    )

                position = telemetry_data.xpos[self.evaluator.trunk_body_id].copy()
                rotation = telemetry_data.xmat[self.evaluator.trunk_body_id].reshape(3, 3)
                elapsed_tick_seconds = completed_tick_substeps * self.runtime.SIM_DT
                world_velocity = (position - previous_position) / elapsed_tick_seconds
                local_velocity = rotation.T @ world_velocity
                previous_position = position
                # Match the training reward's body/site-local gyro yaw channel.
                local_yaw_rate = float(
                    self.evaluator._sensor(telemetry_data, "gyro")[2]
                )
                upright = float(rotation[2, 2])
                height = float(position[2])
                contact = self.evaluator._feet_contacts(telemetry_data)
                joint_qpos = np.asarray(
                    telemetry_data.qpos[self.evaluator.actuator_qpos_addr],
                    dtype=np.float64,
                )
                audit.update(
                    raw_policy_action=raw_action,
                    applied_action=applied_action,
                    preclip_targets=candidate_targets,
                    margin_clipped_targets=desired_targets,
                    applied_targets=applied_targets,
                    previous_applied_targets=previous_targets,
                    joint_qpos=joint_qpos,
                    control_dt=self.runtime.CONTROL_DT,
                )
                completed_steps += 1

                if control_step >= warmup_steps:
                    velocities.append(local_velocity)
                    yaw_rates.append(local_yaw_rate)
                    effective_commands.append(effective)
                    policy_observation_commands.append(policy_command)
                    contacts.append(contact)
                heights.append(height)
                uprights.append(upright)

                if (
                    substep_terminated
                    or height < 0.12
                    or upright < 0.65
                    or not np.all(np.isfinite(data.qpos))
                    or not np.all(np.isfinite(data.qvel))
                ):
                    fell = True
                    break

            # A fall before warmup is still serializable and fails acceptance.
            if not velocities:
                rotation = telemetry_data.xmat[self.evaluator.trunk_body_id].reshape(
                    3, 3
                )
                velocities = [np.zeros(3, dtype=np.float64)]
                yaw_rates = [0.0]
                effective_commands = [np.asarray(router.ramped_command)]
                policy_observation_commands = [
                    np.asarray(
                        router.ramped_command
                        if policy_override is None
                        else policy_override,
                        dtype=np.float64,
                    )
                ]
                contacts = [self.evaluator._feet_contacts(telemetry_data)]
            contact_array = np.asarray(contacts, dtype=np.float64)
            substep_audit_payload = substep_audit.to_dict()
            substep_single_support_rate = float(
                substep_audit_payload["single_support_rate"]
            )
            substep_flight_rate = float(substep_audit_payload["flight_rate"])
            endpoint_single_support_rate = float(
                np.logical_xor(contact_array[:, 0], contact_array[:, 1]).mean()
            )
            endpoint_flight_rate = float(
                (contact_array.sum(axis=1) == 0).mean()
            )
            final_position = telemetry_data.xpos[self.evaluator.trunk_body_id].copy()
            metrics = compute_motion_metrics(
                requested,
                velocities,
                yaw_rates,
                displacement_xyz=final_position - segment_start,
                minimum_height_m=min(heights) if heights else float(final_position[2]),
                minimum_upright=min(uprights) if uprights else 0.0,
                mean_effective_command=np.mean(effective_commands, axis=0),
                mean_policy_observation_command=np.mean(
                    policy_observation_commands, axis=0
                ),
                single_support_rate=substep_single_support_rate,
                flight_rate=substep_flight_rate,
                contact_sample_count=int(
                    substep_audit_payload["contact_sample_count"]
                ),
                contact_rate_sample_source="physics_substeps_after_mj_step",
                diagnostic_control_endpoint_single_support_rate=(
                    endpoint_single_support_rate
                ),
                diagnostic_control_endpoint_flight_rate=endpoint_flight_rate,
                diagnostic_control_endpoint_contact_sample_count=len(contact_array),
            )
            try:
                finalized_gait_quality = gait_quality.finalize()
            except ValueError as error:
                if str(error) not in {
                    "at least two physics substeps are required",
                    "at least two steady-state substeps are required",
                }:
                    raise
                gait_quality_metrics = {
                    "measurement_complete": False,
                    "sample_count": gait_quality.sample_count,
                    "error": str(error),
                }
                gait_quality_result = {
                    "passed": False,
                    "checks": {"measurement_complete": False},
                    "applicable": {"measurement_complete": True},
                    "failures": ["measurement_complete"],
                }
            else:
                gait_quality_metrics = {
                    "measurement_complete": True,
                    **finalized_gait_quality.as_dict(),
                }
                gait_quality_result = gait_quality_acceptance(
                    finalized_gait_quality
                ).as_dict()
            gait_contact_continuity = gait_quality.export_contact_continuity_state()
            phase_mapping = self.reverse_entry_phase_indices
            phase_event_values_valid = all(
                np.isfinite(
                    [
                        event["global_phase_index_before_reset"],
                        event["reset_preincrement_phase_index"],
                        event["profile_phase_rate"],
                        event["first_feedforward_phase_index"],
                        event["phase_steps"],
                    ]
                ).all()
                and event["previous_backward_feedforward_active"] is False
                and event["current_backward_feedforward_active"] is True
                and event["current_expert"] in BACKWARD_FAMILY_EXPERTS
                and event["reset_preincrement_phase_index"]
                == phase_mapping[event["current_expert"]]
                and event["first_feedforward_phase_index"]
                == (
                    event["reset_preincrement_phase_index"]
                    + event["profile_phase_rate"]
                )
                % event["phase_steps"]
                for event in reverse_entry_phase_events
            )
            reverse_entry_phase_audit = {
                "enabled": phase_mapping is not None,
                "status": self.phase_entry_status,
                "formal_candidate": False,
                "adopted_simulation_only": self.formal_candidate_default,
                "diagnostic_only": self.phase_entry_diagnostic_only,
                "formal_and_default_path_reset_disabled": phase_mapping is None,
                "activation": "effective vx < -0.02 false-to-true",
                "mapping_is_preincrement_phase_index": True,
                "mapping": None if phase_mapping is None else dict(phase_mapping),
                "apply_once_per_entry": True,
                "no_reset_while_backward_feedforward_remains_active": True,
                "backward_family": sorted(BACKWARD_FAMILY_EXPERTS),
                "backward_feedforward_entry_count": (
                    backward_feedforward_entry_count
                ),
                "within_backward_family_active_switch_count": (
                    within_backward_family_active_switch_count
                ),
                "event_count": len(reverse_entry_phase_events),
                "events": reverse_entry_phase_events,
                "event_values_valid": phase_event_values_valid,
                "passed": bool(
                    phase_event_values_valid
                    and (
                        len(reverse_entry_phase_events) == 0
                        if phase_mapping is None
                        else len(reverse_entry_phase_events)
                        == backward_feedforward_entry_count
                    )
                ),
            }
            backward_exit_recovery_segment_audit = (
                summarize_backward_exit_recovery_steps(
                    backward_exit_recovery_step_audits,
                    enabled=(
                        self.backward_exit_recovery_enabled
                    ),
                    expected_sample_count=completed_steps,
                    diagnostic_only=self.phase_entry_diagnostic_only,
                )
            )
            segments.append(
                {
                    "name": name,
                    "command": requested.tolist(),
                    "physical_command": requested.tolist(),
                    "policy_observation_command_override": (
                        None if policy_override is None else policy_override.tolist()
                    ),
                    "expected_expert": expected_expert,
                    "expected_policy_role": expected_policy_role,
                    "requested_seconds": float(duration),
                    "completed_seconds": (
                        completed_segment_physics_substeps * self.runtime.SIM_DT
                    ),
                    "completed_physics_substeps": (
                        completed_segment_physics_substeps
                    ),
                    "expected_physics_substeps": (
                        target_steps * int(self.runtime.DECIMATION)
                    ),
                    "physics_timestep_s": float(self.runtime.SIM_DT),
                    "completed": completed_steps == target_steps and not fell,
                    "fell": fell,
                    "warmup_seconds": warmup_seconds,
                    "warmup_fallback_used": completed_steps <= warmup_steps,
                    "metrics": metrics,
                    "safety_audit": audit.to_dict(),
                    "physics_substep_audit": substep_audit_payload,
                    "gait_quality_metrics": gait_quality_metrics,
                    "gait_quality_acceptance": gait_quality_result,
                    "backward_exit_recovery_audit": (
                        backward_exit_recovery_segment_audit
                    ),
                    "routing": {
                        "routed_expert_steps": dict(sorted(routed_experts.items())),
                        "canonical_policy_role_steps": dict(sorted(policy_roles.items())),
                        "steady_state_steps": steady_state_steps,
                        "steady_state_routed_expert_steps": dict(
                            sorted(steady_routed_experts.items())
                        ),
                        "steady_state_policy_role_steps": dict(
                            sorted(steady_policy_roles.items())
                        ),
                        "steady_state_prohibited_expert_steps": (
                            steady_prohibited_expert_steps
                        ),
                        "atomic_endpoint_required": expected_expert in ATOMIC_EXPERTS,
                        "atomic_endpoint_mismatch_steps": (
                            atomic_endpoint_mismatch_steps
                        ),
                        "switch_count": switch_count,
                        "command_clip_events": command_clip_events,
                        "prohibited_expert_steps": prohibited_expert_steps,
                        "policy_yaw_observation_offset_steps": dict(
                            sorted(policy_yaw_offsets.items())
                        ),
                        "mean_policy_yaw_observation_offset": (
                            policy_yaw_offset_sum / completed_steps
                            if completed_steps
                            else 0.0
                        ),
                        "mean_policy_observation_command": np.mean(
                            policy_observation_commands, axis=0
                        ).tolist(),
                        "reverse_entry_phase": reverse_entry_phase_audit,
                    },
                }
            )
            if fell:
                break

        if control_first_startup_audit is None:
            raise RuntimeError("schedule executed no control-first policy tick")
        backward_exit_recovery_audit = backward_exit_recovery.audit()
        backward_exit_recovery_audit.update(
            {
                "status": self.phase_entry_status,
                "formal_candidate_only": False,
                "adopted_simulation_only": self.formal_candidate_default,
                "diagnostic_unadopted_only": self.phase_entry_diagnostic_only,
                "final_guard_call_count": control_ticks_completed,
                "reset_clear_on_schedule_start": True,
                "composition_before_final_guard": True,
                "final_guard_calls_per_control_tick": 1,
            }
        )
        return {
            "seed": int(seed),
            "reset_qpos_audit": reset_qpos_audit,
            "control_first_startup_audit": control_first_startup_audit,
            "backward_exit_recovery_audit": backward_exit_recovery_audit,
            "fell": fell,
            "completed_segment_count": len(segments),
            "requested_segment_count": len(schedule),
            "segments": segments,
        }


def _independent_suite(
    simulator: RoutedSimulator,
    cases: Sequence[Any],
    *,
    seed_base: int,
    episodes: int,
    seconds: float,
    joint_noise_scale: float,
    initial_base_speed: float,
    warmup_seconds: float,
) -> list[dict[str, Any]]:
    results = []
    for episode_index in range(episodes):
        segments = []
        case_seeds: dict[str, int] = {}
        reset_qpos_audits: dict[str, dict[str, Any]] = {}
        control_first_startup_audits: dict[str, dict[str, Any]] = {}
        backward_exit_recovery_audits: dict[str, dict[str, Any]] = {}
        episode_fell = False
        for case_index, case in enumerate(cases):
            case_seed = seed_base + episode_index * 1000 + case_index
            case_result = simulator.run_schedule(
                (
                    (
                        case.name,
                        case.command,
                        seconds,
                        case.policy_observation_command,
                        case.expected_expert,
                        case.expected_policy_role,
                    ),
                ),
                seed=case_seed,
                joint_noise_scale=joint_noise_scale,
                initial_base_speed=initial_base_speed,
                warmup_seconds=warmup_seconds,
            )
            segment = case_result["segments"][0]
            segment["simulation_seed"] = case_seed
            segments.append(segment)
            case_seeds[case.name] = case_seed
            reset_qpos_audits[case.name] = case_result["reset_qpos_audit"]
            control_first_startup_audits[case.name] = case_result[
                "control_first_startup_audit"
            ]
            backward_exit_recovery_audits[case.name] = case_result[
                "backward_exit_recovery_audit"
            ]
            episode_fell |= bool(case_result["fell"])
        results.append(
            {
                "seed": seed_base + episode_index,
                "case_seeds": case_seeds,
                "reset_qpos_audits": reset_qpos_audits,
                "control_first_startup_audits": control_first_startup_audits,
                "backward_exit_recovery_audits": (
                    backward_exit_recovery_audits
                ),
                "fell": episode_fell,
                "completed_segment_count": len(segments),
                "requested_segment_count": len(cases),
                "segments": segments,
            }
        )
    return results


def _runtime_data_dependency_paths(
    *,
    policy_paths: Mapping[str, Path],
    generated_root: Path,
    asset_paths: Mapping[str, Path],
    asset_evidence: Mapping[str, Any],
    selected_reverse_profiles: Mapping[str, Path],
    include_phase_entry_evidence: bool,
    include_backward_exit_recovery_evidence: bool,
) -> dict[str, Path]:
    """Build the complete immutable-at-runtime model/data input closure."""

    dependencies: dict[str, Path] = {}

    def add(label: str, path: Path) -> None:
        dependencies[label] = path.resolve()

    for role, path in sorted(policy_paths.items()):
        add(f"policy_{role}", path)
    for label, path in sorted(asset_paths.items()):
        add(f"generated_{label}", path)
    closure_entries = asset_evidence["dependency_closure"]["entries"]
    for index, relative_path in enumerate(sorted(closure_entries)):
        add(f"generated_transitive_{index:03d}", generated_root / relative_path)
    for label, path in sorted(selected_reverse_profiles.items()):
        add(f"selected_reverse_profile_{label}", path)
    add(
        "formal_candidate_selection_evidence",
        FORMAL_CANDIDATE_SELECTION_EVIDENCE_PATH,
    )
    add(
        "superseded_h2_candidate_selection_evidence",
        H2_5X15_SELECTION_EVIDENCE_PATH,
    )
    add(
        "formal_adoption_evidence",
        FORMAL_ADOPTION_EVIDENCE_PATH,
    )
    add(
        "superseded_h2_adoption_evidence",
        H2_SUPERSEDED_ADOPTION_EVIDENCE_PATH,
    )
    add(
        "h3_fast_exit_safety_component_evidence",
        H3_FAST_EXIT_SAFETY_EVIDENCE_PATH,
    )
    add(
        "formal_candidate_h2_component_evidence",
        H2_COMPONENT_SELECTION_EVIDENCE_PATH,
    )
    add("formal_candidate_straight_20x30_component", H1_STRAIGHT_20X30_EVIDENCE_PATH)
    add(
        "formal_candidate_transition_prefix_component",
        H1_TRANSITION_PREFIX_20SEED_EVIDENCE_PATH,
    )
    add(
        "formal_candidate_rejected_coupled_cap_component",
        H1_REJECTED_COUPLED_CAP_EVIDENCE_PATH,
    )
    add(
        "formal_candidate_historical_failed_20x30",
        HISTORICAL_FAILED_FORMAL_CANDIDATE_EVIDENCE_PATH,
    )
    add("formal_candidate_turn_provenance_base_v3", DIAGNOSTIC_REVERSE_V3_PROFILE_PATH)
    if include_phase_entry_evidence:
        add(
            "diagnostic_reverse_phase_entry_evidence",
            DIAGNOSTIC_REVERSE_PHASE_ENTRY_EVIDENCE_PATH,
        )
    if include_backward_exit_recovery_evidence:
        add(
            "diagnostic_backward_exit_recovery_evidence",
            DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_PATH,
        )
    return dependencies


def _require_runtime_closure_unchanged(
    label: str, before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    """Fail closed if any snapshotted path or byte changes during evaluation."""

    if (
        before.get("dependency_count") != after.get("dependency_count")
        or before.get("root_sha256") != after.get("root_sha256")
        or before.get("entries") != after.get("entries")
    ):
        raise RuntimeError(f"{label} changed during evaluation")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scale_qualification = classify_evaluation_scale(args)
    policy_paths = parse_policy_assignments(args.policy)
    policy_provenance = validate_policy_provenance(
        policy_paths,
        diagnostic_unadopted=args.diagnostic_unadopted_policy,
    )
    asset_evidence = validate_exact_generated_assets(args.generated_root)
    asset_paths = generated_asset_paths(args.generated_root.resolve())
    candidate_selection_evidence = validate_formal_candidate_selection_evidence()
    formal_adoption_evidence = validate_formal_adoption_evidence()
    superseded_h2_adoption_evidence = (
        validate_superseded_h2_adoption_evidence()
    )
    h3_fast_exit_safety_component_evidence = (
        validate_h3_fast_exit_safety_evidence()
    )
    formal_reverse_profile_evidence = validate_adopted_reverse_profiles(
        FORMAL_FIXED_BACKWARD_PROFILE,
        FORMAL_FIXED_BACKWARD_LEFT_PROFILE,
        asset_paths["backward_right"],
    )
    if args.formal_candidate_default:
        selected_backward_profile = args.backward_profile.resolve()
        selected_backward_left_profile = args.backward_left_profile.resolve()
        selected_backward_right_profile = args.backward_right_profile.resolve()
        executed_reverse_profile_evidence = (
            validate_formal_candidate_reverse_profiles(
                selected_backward_profile,
                selected_backward_left_profile,
                selected_backward_right_profile,
            )
        )
    else:
        selected_backward_profile = FORMAL_FIXED_BACKWARD_PROFILE.resolve()
        selected_backward_left_profile = FORMAL_FIXED_BACKWARD_LEFT_PROFILE.resolve()
        selected_backward_right_profile = asset_paths["backward_right"].resolve()
        executed_reverse_profile_evidence = dict(formal_reverse_profile_evidence)
    diagnostic_reverse_profile_evidence = None
    diagnostic_reverse_turn_profile_evidence: dict[str, Any] = {}
    if args.diagnostic_unadopted_reverse_profile is not None:
        selected_backward_profile = (
            args.diagnostic_unadopted_reverse_profile.resolve()
        )
        diagnostic_reverse_profile_evidence = (
            validate_diagnostic_unadopted_reverse_profile(
                selected_backward_profile
            )
        )
        executed_reverse_profile_evidence["straight"] = (
            diagnostic_reverse_profile_evidence
        )
    if args.diagnostic_unadopted_reverse_left_profile is not None:
        selected_backward_left_profile = (
            args.diagnostic_unadopted_reverse_left_profile.resolve()
        )
        diagnostic_reverse_turn_profile_evidence["left"] = (
            validate_diagnostic_unadopted_reverse_turn_profile(
                selected_backward_left_profile,
                direction="left",
                straight_base_evidence=executed_reverse_profile_evidence["straight"],
            )
        )
        executed_reverse_profile_evidence["left"] = (
            diagnostic_reverse_turn_profile_evidence["left"]
        )
    if args.diagnostic_unadopted_reverse_right_profile is not None:
        selected_backward_right_profile = (
            args.diagnostic_unadopted_reverse_right_profile.resolve()
        )
        diagnostic_reverse_turn_profile_evidence["right"] = (
            validate_diagnostic_unadopted_reverse_turn_profile(
                selected_backward_right_profile,
                direction="right",
                straight_base_evidence=executed_reverse_profile_evidence["straight"],
            )
        )
        executed_reverse_profile_evidence["right"] = (
            diagnostic_reverse_turn_profile_evidence["right"]
        )
    reverse_profile_evidence = {
        "formal_fixed_profiles": formal_reverse_profile_evidence,
        "formal_candidate_default_profiles": (
            executed_reverse_profile_evidence
            if args.formal_candidate_default
            else None
        ),
        "executed_profiles": executed_reverse_profile_evidence,
        "diagnostic_unadopted_straight": diagnostic_reverse_profile_evidence,
        "diagnostic_unadopted_turns": diagnostic_reverse_turn_profile_evidence,
    }
    diagnostic_phase_entry_evidence = (
        None
        if args.diagnostic_reverse_entry_phase_indices is None
        else validate_diagnostic_reverse_phase_entry_evidence()
    )
    diagnostic_backward_exit_recovery_evidence = (
        validate_diagnostic_backward_exit_recovery_evidence()
        if args.diagnostic_unadopted_backward_exit_recovery
        else None
    )
    diagnostic_backward_exit_recovery_execution_bundle = (
        None
        if diagnostic_backward_exit_recovery_evidence is None
        else validate_diagnostic_backward_exit_recovery_execution_bundle(
            diagnostic_backward_exit_recovery_evidence,
            executed_reverse_profile_evidence,
            policy_provenance,
        )
    )
    execution_phase_mapping = (
        dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
        if args.formal_candidate_default
        else args.diagnostic_reverse_entry_phase_indices
    )
    execution_backward_exit_recovery_enabled = bool(
        BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT
        if args.formal_candidate_default
        else args.diagnostic_unadopted_backward_exit_recovery
    )
    formal_candidate_execution_bundle = (
        validate_formal_candidate_execution_bundle(
            candidate_selection_evidence,
            formal_adoption_evidence,
            executed_reverse_profile_evidence,
            policy_provenance,
            execution_phase_mapping,
            backward_exit_recovery_enabled=(
                execution_backward_exit_recovery_enabled
            ),
            safety_component_evidence=(
                h3_fast_exit_safety_component_evidence
            ),
        )
        if args.formal_candidate_default
        else None
    )
    reverse_profile_adoption = derive_reverse_profile_adoption(
        executed_reverse_profile_evidence,
        evaluation_evidence={
            label: formal_adoption_evidence
            for label in ("straight", "left", "right")
        },
        profile_hash_allowlists=(
            FORMAL_H3_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS
        ),
    )
    mujoco, onnxruntime, runtime, runtime_dependency_provenance = _load_runtime(
        include_provenance=True
    )
    runtime_data_paths = _runtime_data_dependency_paths(
        policy_paths=policy_paths,
        generated_root=args.generated_root.resolve(),
        asset_paths=asset_paths,
        asset_evidence=asset_evidence,
        selected_reverse_profiles={
            "straight": selected_backward_profile,
            "left": selected_backward_left_profile,
            "right": selected_backward_right_profile,
        },
        include_phase_entry_evidence=(
            diagnostic_phase_entry_evidence is not None
        ),
        include_backward_exit_recovery_evidence=(
            diagnostic_backward_exit_recovery_evidence is not None
        ),
    )
    runtime_data_pre_evaluation = capture_runtime_source_dependency_closure(
        runtime_data_paths
    )

    bank = RoutedPolicyBank(policy_paths, onnxruntime)
    evaluator = runtime.OfficialPolicyEvaluator(
        asset_paths["scene"], policy_paths["stand"], asset_paths["reference"]
    )
    evaluator.backward_residual_scale = args.backward_residual_scale
    evaluator.load_backward_profile(selected_backward_profile)
    evaluator.load_backward_turn_profile(1, selected_backward_left_profile)
    evaluator.load_backward_turn_profile(-1, selected_backward_right_profile)
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0
    model_evidence = validate_model_contract(evaluator)
    simulator = RoutedSimulator(
        evaluator,
        bank,
        mujoco,
        runtime,
        leg_target_margin_rad=args.leg_target_margin_rad,
        target_slew_rate_rad_s=args.target_slew_rate_rad_s,
        diagnostic_noncontract_safety=args.diagnostic_noncontract_safety,
        left_knee_extra_upper_margin_rad=(
            executed_reverse_profile_evidence["straight"]["composition"][
                "left_knee_extra_upper_margin_rad"
            ]
        ),
        diagnostic_reverse_entry_phase_indices=(
            args.diagnostic_reverse_entry_phase_indices
        ),
        diagnostic_unadopted_backward_exit_recovery=(
            args.diagnostic_unadopted_backward_exit_recovery
        ),
        formal_candidate_default=args.formal_candidate_default,
    )

    thresholds = AcceptanceThresholds()
    if args.policy_command_diagnostic_suite:
        diagnostic_episodes = _independent_suite(
            simulator,
            POLICY_COMMAND_DIAGNOSTIC_CASES,
            seed_base=args.seed + 3_000_000,
            episodes=1,
            seconds=5.0,
            joint_noise_scale=0.0,
            initial_base_speed=0.0,
            warmup_seconds=args.warmup_seconds,
        )
        diagnostic_acceptance = suite_acceptance(
            diagnostic_episodes,
            [case.name for case in POLICY_COMMAND_DIAGNOSTIC_CASES],
            thresholds,
            require_gait_quality=True,
        )
        suites_payload = {
            "policy_command_diagnostics": {
                "non_adoptable": True,
                "episodes_fixed": 1,
                "seconds_fixed": 5.0,
                "initial_joint_noise_scale_fixed": 0.0,
                "initial_base_speed_fixed": 0.0,
                "definition": [
                    asdict(case) for case in POLICY_COMMAND_DIAGNOSTIC_CASES
                ],
                "known_rejected_cases": {
                    name: {
                        "status": "REJECTED",
                        "reason": "prior 1x5s diagnostic fell",
                    }
                    for name in sorted(REJECTED_POLICY_COMMAND_DIAGNOSTIC_CASES)
                },
                "episodes": diagnostic_episodes,
                "diagnostic_acceptance": diagnostic_acceptance,
            }
        }
        formal_suites_passed = False
    else:
        primitive_episodes = _independent_suite(
            simulator,
            PRIMITIVE_CASES,
            seed_base=args.seed,
            episodes=args.episodes,
            seconds=args.seconds,
            joint_noise_scale=args.initial_joint_noise_scale,
            initial_base_speed=args.initial_base_speed,
            warmup_seconds=args.warmup_seconds,
        )
        compound_episodes = _independent_suite(
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
        suites_payload = {
            "primitives": {
                "definition": [asdict(case) for case in PRIMITIVE_CASES],
                "episodes": primitive_episodes,
                "acceptance": primitive_acceptance,
            },
            "compounds": {
                "definition": [asdict(case) for case in COMPOUND_CASES],
                "episodes": compound_episodes,
                "acceptance": compound_acceptance,
            },
            "transitions": {
                "definition": [
                    {
                        "name": name,
                        "command": list(command),
                        "physical_command": list(command),
                        "policy_observation_command": (
                            None if policy_command is None else list(policy_command)
                        ),
                        "validation_status": case.validation_status,
                        "expected_expert": expected_expert,
                        "expected_policy_role": expected_policy_role,
                        "seconds": seconds,
                    }
                    for case, (
                        name,
                        command,
                        seconds,
                        policy_command,
                        expected_expert,
                        expected_policy_role,
                    ) in zip(
                        TRANSITION_CASES, transition_definition, strict=True
                    )
                ],
                "episodes": transition_episodes,
                "acceptance": transition_acceptance,
            },
        }
        formal_suites_passed = bool(
            primitive_acceptance["passed"]
            and compound_acceptance["passed"]
            and transition_acceptance["passed"]
        )
    external_post_evaluation = validate_frozen_runtime_source_dependencies()
    own_post_evaluation = capture_runtime_source_dependency_closure(
        OWN_RUNTIME_SOURCE_PATHS
    )
    runtime_binaries_post_evaluation = capture_runtime_source_dependency_closure(
        _runtime_binary_dependency_paths(mujoco, onnxruntime),
        expected_sha256=FROZEN_RUNTIME_BINARY_SHA256,
    )
    runtime_data_post_evaluation = capture_runtime_source_dependency_closure(
        runtime_data_paths
    )
    _require_runtime_closure_unchanged(
        "external runtime source closure",
        runtime_dependency_provenance["pre_import"][
            "external_hard_allowlisted_source_closure"
        ],
        external_post_evaluation,
    )
    _require_runtime_closure_unchanged(
        "exp004 source/contract closure",
        runtime_dependency_provenance["pre_import"][
            "exp004_source_and_contract_snapshot"
        ],
        own_post_evaluation,
    )
    _require_runtime_closure_unchanged(
        "hard-allowlisted runtime binary closure",
        runtime_dependency_provenance["pre_import"][
            "hard_allowlisted_runtime_binary_closure"
        ],
        runtime_binaries_post_evaluation,
    )
    _require_runtime_closure_unchanged(
        "runtime model/data dependency closure",
        runtime_data_pre_evaluation,
        runtime_data_post_evaluation,
    )
    all_sessions_cpu_only = all(
        providers == ["CPUExecutionProvider"]
        for providers in bank.session_providers.values()
    )
    if not all_sessions_cpu_only:
        raise RuntimeError("all ONNX sessions must remain CPU-only")
    runtime_dependency_provenance.update(
        {
            "post_evaluation": {
                "external_hard_allowlisted_source_closure": (
                    external_post_evaluation
                ),
                "exp004_source_and_contract_snapshot": own_post_evaluation,
                "hard_allowlisted_runtime_binary_closure": (
                    runtime_binaries_post_evaluation
                ),
                "runtime_model_and_data_closure": (
                    runtime_data_post_evaluation
                ),
            },
            "runtime_model_and_data_pre_evaluation": (
                runtime_data_pre_evaluation
            ),
            "onnx_session_execution_providers": {
                role: providers
                for role, providers in sorted(bank.session_providers.items())
            },
            "all_onnx_sessions_cpu_only_verified": all_sessions_cpu_only,
            "pre_post_source_and_data_hashes_unchanged": True,
            "verified": True,
        }
    )

    target_margin_matches_contract = bool(
        args.leg_target_margin_rad
        == LEG_TARGET_MARGIN_RAD
        == RUNTIME_TARGET_SAFETY_MARGIN_RAD
    )
    target_slew_matches_contract = bool(
        args.target_slew_rate_rad_s
        == TARGET_SLEW_LIMIT_RAD_PER_S
        == RUNTIME_TARGET_SLEW_RATE_RAD_S
    )
    formal_case_validation = command_case_validation_gate(
        (*PRIMITIVE_CASES, *COMPOUND_CASES, *TRANSITION_CASES)
    )
    diagnostic_reverse_bank = bool(
        args.diagnostic_unadopted_reverse_profile is not None
        or args.diagnostic_unadopted_reverse_left_profile is not None
        or args.diagnostic_unadopted_reverse_right_profile is not None
    )
    reverse_profile_adoption_passed = bool(reverse_profile_adoption["passed"])
    formal_runtime_adopted = bool(
        args.formal_candidate_default
        and FORMAL_CANDIDATE_STATUS == "ADOPTED_SIMULATION_ONLY"
    )
    formal_candidate_pending = bool(
        args.formal_candidate_default and not formal_runtime_adopted
    )
    adoption_contract_passed = bool(
        not args.diagnostic_noncontract_safety
        and not args.policy_command_diagnostic_suite
        and not args.diagnostic_unadopted_policy
        and not diagnostic_reverse_bank
        and args.diagnostic_reverse_entry_phase_indices is None
        and not args.diagnostic_unadopted_backward_exit_recovery
        and formal_runtime_adopted
        and args.backward_residual_scale == 0.0
        and target_margin_matches_contract
        and target_slew_matches_contract
        and policy_provenance["adoption_eligible"]
        and scale_qualification["release_qualification_eligible"]
        and formal_case_validation["passed"]
        and reverse_profile_adoption_passed
    )
    simulation_passed = bool(
        adoption_contract_passed and formal_suites_passed
    )
    diagnostic_modes = []
    if args.policy_command_diagnostic_suite:
        diagnostic_modes.append("POLICY_OBSERVATION_COMMAND")
    if args.diagnostic_noncontract_safety:
        diagnostic_modes.append("NONCONTRACT_SAFETY")
    if args.diagnostic_unadopted_policy:
        diagnostic_modes.append("UNADOPTED_POLICY")
    if diagnostic_reverse_bank:
        diagnostic_modes.append("UNADOPTED_REVERSE_PROFILE_BANK")
    if args.diagnostic_reverse_entry_phase_indices is not None:
        diagnostic_modes.append("UNADOPTED_REVERSE_PHASE_ENTRY")
    if args.diagnostic_unadopted_backward_exit_recovery:
        diagnostic_modes.append("UNADOPTED_BACKWARD_EXIT_RECOVERY")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "evaluation_mode": (
            "DIAGNOSTIC_" + "+".join(diagnostic_modes)
            if diagnostic_modes
            else (
                "RELEASE_QUALIFICATION"
                if formal_runtime_adopted
                and scale_qualification["release_qualification_eligible"]
                else FORMAL_CANDIDATE_STATUS
            )
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_qualification": scale_qualification,
        "configuration": {
            "seed": args.seed,
            "episodes": args.episodes,
            "seconds": args.seconds,
            "transition_seconds": args.transition_seconds,
            "transition_stand_seconds": args.transition_stand_seconds,
            "warmup_seconds": args.warmup_seconds,
            "initial_joint_noise_scale": args.initial_joint_noise_scale,
            "reset_noise_margin_rad": RESET_NOISE_MARGIN_RAD,
            "initial_base_speed": args.initial_base_speed,
            "leg_target_margin_rad": args.leg_target_margin_rad,
            "target_slew_rate_rad_per_s": args.target_slew_rate_rad_s,
            "target_slew_per_control_tick_rad": (
                args.target_slew_rate_rad_s * runtime.CONTROL_DT
            ),
            "diagnostic_noncontract_safety": args.diagnostic_noncontract_safety,
            "policy_command_diagnostic_suite": (
                args.policy_command_diagnostic_suite
            ),
            "diagnostic_unadopted_policy": args.diagnostic_unadopted_policy,
            "diagnostic_unadopted_reverse_profile": (
                None
                if args.diagnostic_unadopted_reverse_profile is None
                else str(selected_backward_profile)
            ),
            "diagnostic_unadopted_reverse_left_profile": (
                None
                if args.diagnostic_unadopted_reverse_left_profile is None
                else str(selected_backward_left_profile)
            ),
            "diagnostic_unadopted_reverse_right_profile": (
                None
                if args.diagnostic_unadopted_reverse_right_profile is None
                else str(selected_backward_right_profile)
            ),
            "diagnostic_unadopted_reverse_entry_phase_indices": (
                None
                if args.diagnostic_reverse_entry_phase_indices is None
                else dict(args.diagnostic_reverse_entry_phase_indices)
            ),
            "diagnostic_unadopted_backward_exit_recovery": (
                args.diagnostic_unadopted_backward_exit_recovery
            ),
            "formal_candidate_default": args.formal_candidate_default,
            "formal_adopted_default": formal_runtime_adopted,
            "formal_candidate_status": (
                FORMAL_CANDIDATE_STATUS
                if formal_candidate_pending
                else None
            ),
            "formal_adopted_status": (
                FORMAL_CANDIDATE_STATUS if formal_runtime_adopted else None
            ),
            "executed_reverse_profile_paths": {
                "straight": str(selected_backward_profile),
                "left": str(selected_backward_left_profile),
                "right": str(selected_backward_right_profile),
            },
            "executed_reverse_entry_phase_indices": (
                None
                if execution_phase_mapping is None
                else dict(execution_phase_mapping)
            ),
            "backward_exit_recovery_enabled": (
                execution_backward_exit_recovery_enabled
            ),
            "backward_residual_scale": args.backward_residual_scale,
            "adoption_requires_zero_backward_residual": True,
            "left_knee_extra_upper_margin_rad": (
                simulator.left_knee_extra_upper_margin_rad
            ),
            "left_knee_profile_upper_target_rad": (
                simulator.left_knee_profile_upper_target_rad
            ),
            "control_dt": runtime.CONTROL_DT,
            "control_first_startup_required": True,
            "physics_steps_allowed_before_startup_control": 0,
            "home_only_startup_precharge_used": False,
            "guard_calls_per_control_tick": 1,
            "backward_exit_recovery_composition_before_final_guard": True,
            "sim_dt": runtime.SIM_DT,
            "router": "SafeGaitRouter default contract",
        },
        "exact_hardware_safe_assets": asset_evidence,
        "runtime_dependency_provenance": runtime_dependency_provenance,
        "formal_candidate_selection_evidence": candidate_selection_evidence,
        "h3_fast_exit_safety_component_evidence": (
            h3_fast_exit_safety_component_evidence
        ),
        "formal_adoption_evidence": formal_adoption_evidence,
        "superseded_h2_adoption_evidence": (
            superseded_h2_adoption_evidence
        ),
        "formal_candidate_execution_bundle": formal_candidate_execution_bundle,
        "reverse_profile_evidence": reverse_profile_evidence,
        "formal_reverse_phase_entry_contract": {
            "enabled": args.formal_candidate_default,
            "status": FORMAL_CANDIDATE_STATUS,
            "formal_candidate_only": False,
            "diagnostic_only": False,
            "enabled_by_default": True,
            "adopted": True,
            "adoption_eligible": True,
            "simulation_acceptance_eligible": True,
            "safety_component_only": False,
            "safety_component_evidence_is_safety_only": True,
            "fast_exit_safety_passed": True,
            "combined_5x15_required": False,
            "combined_5x15_passed": True,
            "requires_formal_20x30_requalification": False,
            "current_endpoint_requalified": True,
            "activation": "effective vx < -0.02 false-to-true",
            "activation_is_not_router_decision_switched": True,
            "preincrement_phase_indices": dict(
                FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
            ),
            "apply_exactly_once_per_entry": True,
            "no_reset_within_continuously_active_backward_family": True,
            "backward_family": sorted(BACKWARD_FAMILY_EXPERTS),
            "candidate_selection_evidence": candidate_selection_evidence,
            "safety_component_evidence": (
                h3_fast_exit_safety_component_evidence
            ),
            "safety_component_evidence_path": str(
                H3_FAST_EXIT_SAFETY_EVIDENCE_PATH
            ),
            "safety_component_evidence_sha256": (
                H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
            ),
            "adoption_evidence": formal_adoption_evidence,
            "adoption_evidence_path": str(FORMAL_ADOPTION_EVIDENCE_PATH),
            "adoption_evidence_sha256": FORMAL_ADOPTION_EVIDENCE_SHA256,
            "superseded_h2_adoption_evidence": (
                superseded_h2_adoption_evidence
            ),
            "execution_bundle_binding": formal_candidate_execution_bundle,
            "current_formal_reverse_endpoint_mps": (
                CURRENT_FORMAL_REVERSE_ENDPOINT_MPS
            ),
            "hardware_deployment": "PROHIBITED",
        },
        "formal_backward_exit_recovery_contract": {
            "enabled": args.formal_candidate_default,
            "status": FORMAL_CANDIDATE_STATUS,
            "enabled_by_default": BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT,
            "formal_candidate_only": False,
            "diagnostic_unadopted_only": False,
            "adopted": True,
            "adoption_eligible": True,
            "simulation_acceptance_eligible": True,
            "safety_component_only": False,
            "safety_component_evidence_is_safety_only": True,
            "fast_exit_safety_passed": True,
            "combined_5x15_required": False,
            "combined_5x15_passed": True,
            "requires_formal_20x30_requalification": False,
            "current_endpoint_requalified": True,
            "runtime_contract": backward_exit_recovery_contract(),
            "selection_evidence": candidate_selection_evidence,
            "execution_bundle_binding": formal_candidate_execution_bundle,
            "selection_evidence_path": str(
                FORMAL_CANDIDATE_SELECTION_EVIDENCE_PATH
            ),
            "selection_evidence_sha256": (
                FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256
            ),
            "safety_component_evidence": (
                h3_fast_exit_safety_component_evidence
            ),
            "safety_component_evidence_path": str(
                H3_FAST_EXIT_SAFETY_EVIDENCE_PATH
            ),
            "safety_component_evidence_sha256": (
                H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
            ),
            "adoption_evidence": formal_adoption_evidence,
            "adoption_evidence_path": str(FORMAL_ADOPTION_EVIDENCE_PATH),
            "adoption_evidence_sha256": FORMAL_ADOPTION_EVIDENCE_SHA256,
            "superseded_h2_adoption_evidence": (
                superseded_h2_adoption_evidence
            ),
            "current_formal_reverse_endpoint_mps": (
                CURRENT_FORMAL_REVERSE_ENDPOINT_MPS
            ),
            "hardware_deployment": "PROHIBITED",
        },
        "diagnostic_reverse_phase_entry_contract": {
            "enabled": args.diagnostic_reverse_entry_phase_indices is not None,
            "status": "DIAGNOSTIC_UNADOPTED",
            "diagnostic_only": True,
            "adoption_eligible": False,
            "simulation_acceptance_eligible": False,
            "formal_and_default_path_behavior": "NO_PHASE_RESET",
            "activation": "effective vx < -0.02 false-to-true",
            "activation_is_not_router_decision_switched": True,
            "preincrement_phase_indices": (
                None
                if args.diagnostic_reverse_entry_phase_indices is None
                else dict(args.diagnostic_reverse_entry_phase_indices)
            ),
            "frozen_diagnostic_mapping": dict(
                FROZEN_DIAGNOSTIC_REVERSE_ENTRY_PHASE_INDICES
            ),
            "apply_exactly_once_per_entry": True,
            "no_reset_within_continuously_active_backward_family": True,
            "backward_family": sorted(BACKWARD_FAMILY_EXPERTS),
            "straight_transition_evidence": diagnostic_phase_entry_evidence,
            "component_evidence_source_reverse_endpoint_mps": (
                DIAGNOSTIC_REVERSE_PHASE_ENTRY_SOURCE_REVERSE_ENDPOINT_MPS
            ),
            "current_formal_reverse_endpoint_mps": (
                CURRENT_FORMAL_REVERSE_ENDPOINT_MPS
            ),
            "component_evidence_endpoint_matches_current": False,
            "current_endpoint_status": (
                "CURRENT_ENDPOINT_REQUALIFICATION_REQUIRED"
            ),
            "usable_as_current_endpoint_evidence": False,
            "profile_hashes_unchanged_by_phase_mapping": True,
            "hardware_deployment": "PROHIBITED",
        },
        "diagnostic_backward_exit_recovery_contract": {
            "enabled": args.diagnostic_unadopted_backward_exit_recovery,
            "status": "DIAGNOSTIC_UNADOPTED",
            "enabled_by_default": False,
            "formal_candidate_default_is_separate": True,
            "diagnostic_unadopted_only": True,
            "adoption_eligible": False,
            "simulation_acceptance_eligible": False,
            "runtime_contract": backward_exit_recovery_contract(),
            "selection_evidence": (
                diagnostic_backward_exit_recovery_evidence
            ),
            "execution_bundle_binding": (
                diagnostic_backward_exit_recovery_execution_bundle
            ),
            "selection_evidence_path": str(
                DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_PATH
            ),
            "selection_evidence_sha256": (
                DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_SHA256
            ),
            "component_evidence_source_reverse_endpoint_mps": (
                DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_SOURCE_REVERSE_ENDPOINT_MPS
            ),
            "current_formal_reverse_endpoint_mps": (
                CURRENT_FORMAL_REVERSE_ENDPOINT_MPS
            ),
            "component_evidence_endpoint_matches_current": False,
            "current_endpoint_status": (
                "CURRENT_ENDPOINT_REQUALIFICATION_REQUIRED"
            ),
            "requires_current_endpoint_requalification": True,
            "usable_as_current_endpoint_evidence": False,
            "hardware_deployment": "PROHIBITED",
        },
        "reverse_profile_adoption": {
            **reverse_profile_adoption,
            "legacy_straight_status": REVERSE_V1_ADOPTION_STATUS,
            "rejected_release_id": "optimized_reverse_exact_safe_v1",
            "measured_forward_velocity_mps": (
                REVERSE_V1_MEASURED_FORWARD_VELOCITY_MPS
            ),
            "physical_reverse_command_mps": -0.050,
            "reason": "straight reverse v1 did not produce commanded reverse motion",
        },
        "policy_observation_yaw_offset_contract": {
            "pure_yaw_right": -0.30,
            "forward_or_compound_negative_yaw": -0.15,
            "positive_yaw": 0.0,
            "feedforward_reverse_at_zero_residual": 0.0,
            "requested_and_physical_commands_unchanged": True,
            "command_case_policy_observation_override_is_final": True,
            "route_yaw_offset_bypassed_when_override_present": True,
        },
        "command_mapping_contract": {
            "status": FORMAL_CANDIDATE_STATUS,
            "reverse_routes_status": FORMAL_CANDIDATE_STATUS,
            "validation_status_gate": formal_case_validation,
            "physical_command_drives_router_and_metrics": True,
            "policy_observation_command_drives_policy_only": True,
            "primitives": [asdict(case) for case in PRIMITIVE_CASES],
            "compounds": [asdict(case) for case in COMPOUND_CASES],
        },
        "target_safety_contract": {
            "leg_margin_rad": RUNTIME_TARGET_SAFETY_MARGIN_RAD,
            "target_slew_rate_rad_per_s": RUNTIME_TARGET_SLEW_RATE_RAD_S,
            "target_slew_per_control_tick_rad": (
                RUNTIME_TARGET_SLEW_RATE_RAD_S * runtime.CONTROL_DT
            ),
            "guard": (
                "scripts.evaluate_routed_transitions.DiagnosticTargetSafetyGuard"
                if args.diagnostic_noncontract_safety
                else "target_safety.FinalTargetSafetyGuard"
            ),
            "configured_leg_margin_rad": args.leg_target_margin_rad,
            "configured_target_slew_rate_rad_per_s": args.target_slew_rate_rad_s,
            "diagnostic_values_are_non_adoptable": args.diagnostic_noncontract_safety,
            "stage": (
                "policy_profile_then_optional_backward_exit_recovery_then_"
                "desired_margin_clip_then_slew_then_physical_safe_clamp"
            ),
            "backward_exit_recovery": {
                "enabled": execution_backward_exit_recovery_enabled,
                "status": FORMAL_CANDIDATE_STATUS,
                "formal_candidate_only": False,
                "adopted_simulation_only": args.formal_candidate_default,
                "diagnostic_unadopted_only": (
                    not args.formal_candidate_default
                ),
                "composed_before_final_guard": True,
                "final_guard_calls_per_control_tick": 1,
                "hold_control_ticks": BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
                "hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
                "left_knee_upper_target_rad": (
                    BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
                ),
                "safety_component_only": False,
                "safety_component_evidence_is_safety_only": (
                    args.formal_candidate_default
                ),
                "safety_component_evidence_sha256": (
                    H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
                ),
                "adoption_eligible": args.formal_candidate_default,
                "simulation_acceptance_eligible": args.formal_candidate_default,
                "adoption_evidence_sha256": FORMAL_ADOPTION_EVIDENCE_SHA256,
                "superseded_h2_adoption_evidence_sha256": (
                    H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256
                ),
            },
            "control_first_startup": {
                "required": True,
                "desired_target": "first routed policy/profile target",
                "home_only_precharge": "PROHIBITED",
                "ordering": (
                    "observe -> route -> policy/profile -> optional diagnostic "
                    "or candidate backward-exit recovery -> guard exactly once -> data.ctrl -> "
                    "physics -> post-step metrics/audit"
                ),
                "control_dt_seconds": runtime.CONTROL_DT,
                "physics_steps_allowed_before_control": 0,
                "guard_calls_for_first_tick": 1,
                "guard_calls_per_normal_tick": 1,
                "audit_required_for_acceptance": True,
            },
            "reset_qpos_contract": {
                "noise_margin_rad": RESET_NOISE_MARGIN_RAD,
                "zero_noise": "exact SAFE_INIT",
                "positive_noise": "physical SAFE limits inward by 0.005 rad",
                "head_qpos_rad": 0.0,
                "initial_qpos_violation_samples_required": 0,
            },
            "applied_leg_target_envelope_rad": {
                name: [
                    float(simulator.target_lower[index]),
                    float(simulator.target_upper[index]),
                ]
                for index, name in enumerate(simulator.joint_names)
                if name not in HEAD_JOINTS
            },
            "head_target_rad": 0.0,
            "measured_qpos_limit": "original SAFE_JOINT_LIMITS",
            "maximum_qpos_violation_samples_for_acceptance": 0,
        },
        "adoption_contract": {
            "formal_candidate_pending": formal_candidate_pending,
            "formal_candidate_status": (
                FORMAL_CANDIDATE_STATUS if formal_candidate_pending else None
            ),
            "formal_adopted_default": formal_runtime_adopted,
            "formal_adopted_status": (
                FORMAL_CANDIDATE_STATUS if formal_runtime_adopted else None
            ),
            "candidate_selection_evidence_is_not_adoption_evidence": True,
            "safety_component_evidence_is_not_adoption_evidence": True,
            "safety_component_evidence_sha256": (
                H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
            ),
            "superseded_h2_adoption_evidence_sha256": (
                H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256
            ),
            "superseded_h2_selection_evidence_sha256": (
                H2_5X15_SELECTION_EVIDENCE_SHA256
            ),
            "adoption_evidence_sha256": FORMAL_ADOPTION_EVIDENCE_SHA256,
            "formal_adoption_evidence_sha256_allowlist": [
                FORMAL_ADOPTION_EVIDENCE_SHA256
            ],
            "formal_release_evidence_sha256_allowlist": [],
            "combined_5x15_required": False,
            "combined_5x15_passed": True,
            "formal_20x30_required": False,
            "formal_20x30_adoption_passed": True,
            "new_exact_20x30_release_run_required": True,
            "formal_phase_entry_adopted": formal_runtime_adopted,
            "formal_backward_exit_recovery_adopted": formal_runtime_adopted,
            "backward_residual_is_zero": args.backward_residual_scale == 0.0,
            "diagnostic_noncontract_safety_disabled": (
                not args.diagnostic_noncontract_safety
            ),
            "policy_command_diagnostic_suite_disabled": (
                not args.policy_command_diagnostic_suite
            ),
            "diagnostic_unadopted_policy_disabled": (
                not args.diagnostic_unadopted_policy
            ),
            "diagnostic_unadopted_reverse_profile_disabled": (
                args.diagnostic_unadopted_reverse_profile is None
            ),
            "diagnostic_unadopted_reverse_left_profile_disabled": (
                args.diagnostic_unadopted_reverse_left_profile is None
            ),
            "diagnostic_unadopted_reverse_right_profile_disabled": (
                args.diagnostic_unadopted_reverse_right_profile is None
            ),
            "diagnostic_reverse_phase_entry_disabled": (
                args.diagnostic_reverse_entry_phase_indices is None
            ),
            "diagnostic_unadopted_backward_exit_recovery_disabled": (
                not args.diagnostic_unadopted_backward_exit_recovery
            ),
            "all_policy_roles_are_formal_base_v22": policy_provenance[
                "adoption_eligible"
            ],
            "target_margin_matches_contract": target_margin_matches_contract,
            "target_slew_matches_contract": target_slew_matches_contract,
            "release_qualification_scale_passed": scale_qualification[
                "release_qualification_eligible"
            ],
            "all_command_validation_statuses_adoptable": formal_case_validation[
                "passed"
            ],
            "screening_run_cannot_promote": not scale_qualification[
                "release_qualification_eligible"
            ],
            "reverse_profile_adopted": reverse_profile_adoption_passed,
            "passed": adoption_contract_passed,
        },
        "model_contract": model_evidence,
        "policy_provenance": policy_provenance,
        "policies": bank.manifest(),
        "policy_inference_counts": dict(sorted(bank.inference_counts.items())),
        "suites": suites_payload,
        "simulation_suite_acceptance_passed": formal_suites_passed,
        "simulation_acceptance_passed": simulation_passed,
        "hardware_gate": hardware_gate(simulation_passed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    raise SystemExit(0 if simulation_passed else 1)


if __name__ == "__main__":
    main()
