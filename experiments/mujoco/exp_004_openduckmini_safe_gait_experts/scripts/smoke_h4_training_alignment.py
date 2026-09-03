"""Run a one-to-five-step H4-aligned MJX environment smoke.

This command never starts PPO and never writes a checkpoint or artifact.  It
exists to prove, on the frozen WSL training stack, that reset, command
separation, target guarding, contact-point quality observations, and reward
wiring all execute together before any GPU training is authorized.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.h4_training_alignment import (  # noqa: E402
    H4QualityRewardScales,
    LEGACY_PRIVILEGED_IMITATION_INDEX_SLICE,
    LEGACY_PRIVILEGED_IMITATION_PHASE_SLICE,
    LEGACY_PRIVILEGED_REFERENCE_SLICE,
    LEG_ACTION_INDICES,
    MAX_TARGET_DELTA_PER_TICK_RAD,
    make_anchor_command_mapper,
    make_h4_aligned_environment_class,
    make_h4_forward_v2_physical_sampler,
    make_h4_reverse_v2_physical_sampler,
)


LEGACY_TRAINER_PATH = EXP_ROOT / "scripts" / "train_expert.py"
H4_RUNNER_PATH = EXP_ROOT / "scripts" / "train_h4_aligned_expert.py"
DEFAULT_SELECTED_REVERSE_TEACHER = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_slew_feasible_teacher_selected_v1.json"
)
PINNED_SELECTED_REVERSE_TEACHER_SHA256 = (
    "7a24a7c9096a1c4a9dc72ac85ec01c5e0a41acf8214d80cc7e2cf4ccc50ae237"
)


def _load_legacy_trainer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "exp004_h4_legacy_trainer", LEGACY_TRAINER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy trainer: {LEGACY_TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_h4_runner_backend_contract() -> Any:
    spec = importlib.util.spec_from_file_location(
        "exp004_h4_smoke_backend_contract", H4_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load H4 runner: {H4_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _triplet(text: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if len(values) != 3 or not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("expected one finite vx,vy,yaw triplet")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke the H4-aligned training environment without PPO."
    )
    parser.add_argument("--expert", choices=("forward", "reverse"), default="reverse")
    parser.add_argument("--forward-iteration-v2", action="store_true")
    parser.add_argument("--reverse-iteration-v2", action="store_true")
    parser.add_argument(
        "--forward-iteration-v3-touchdown-balance", action="store_true"
    )
    parser.add_argument(
        "--forward-iteration-v4-contact-event-validity-persistence",
        action="store_true",
    )
    parser.add_argument(
        "--forward-v5-contact-pulse-abort-scale-only",
        action="store_true",
    )
    parser.add_argument(
        "--forward-iteration-v6-contact-abort-island-only",
        action="store_true",
    )
    parser.add_argument(
        "--reverse-iteration-v3-no-target-imitation", action="store_true"
    )
    parser.add_argument(
        "--reverse-iteration-v4-residual-transfer-gain-024",
        action="store_true",
    )
    parser.add_argument(
        "--reverse-iteration-v5-no-contact-imitation",
        action="store_true",
    )
    parser.add_argument(
        "--reverse-iteration-v6-absolute-full-leg-targets",
        action="store_true",
    )
    parser.add_argument("--physical-command", type=_triplet)
    parser.add_argument("--policy-command", type=_triplet)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--reset-noise-multiplier", type=float, default=0.0)
    parser.add_argument(
        "--selected-reverse-teacher",
        type=Path,
        default=DEFAULT_SELECTED_REVERSE_TEACHER,
    )
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--jit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="JIT the environment step exactly as training will (default: true).",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/user/openduck_training_20260729"),
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=EXP_ROOT / "artifacts" / "generated_playground",
    )
    return parser


def _validate_smoke_cli(args: argparse.Namespace) -> None:
    modes = (
        args.forward_iteration_v2,
        args.reverse_iteration_v2,
        args.forward_iteration_v3_touchdown_balance,
        args.reverse_iteration_v3_no_target_imitation,
        args.forward_iteration_v4_contact_event_validity_persistence,
        args.reverse_iteration_v4_residual_transfer_gain_024,
        args.forward_v5_contact_pulse_abort_scale_only,
        args.reverse_iteration_v5_no_contact_imitation,
        args.forward_iteration_v6_contact_abort_island_only,
        args.reverse_iteration_v6_absolute_full_leg_targets,
    )
    if sum(bool(value) for value in modes) > 1:
        raise ValueError("H4 iteration smoke modes are mutually exclusive")
    for enabled, expert, seed, flag in (
        (args.forward_iteration_v2, "forward", 20260809, "--forward-iteration-v2"),
        (args.reverse_iteration_v2, "reverse", 20260810, "--reverse-iteration-v2"),
        (
            args.forward_iteration_v3_touchdown_balance,
            "forward",
            20260809,
            "--forward-iteration-v3-touchdown-balance",
        ),
        (
            args.reverse_iteration_v3_no_target_imitation,
            "reverse",
            20260810,
            "--reverse-iteration-v3-no-target-imitation",
        ),
        (
            args.forward_iteration_v4_contact_event_validity_persistence,
            "forward",
            20260809,
            "--forward-iteration-v4-contact-event-validity-persistence",
        ),
        (
            args.reverse_iteration_v4_residual_transfer_gain_024,
            "reverse",
            20260810,
            "--reverse-iteration-v4-residual-transfer-gain-024",
        ),
        (
            args.forward_v5_contact_pulse_abort_scale_only,
            "forward",
            20260809,
            "--forward-v5-contact-pulse-abort-scale-only",
        ),
        (
            args.reverse_iteration_v5_no_contact_imitation,
            "reverse",
            20260810,
            "--reverse-iteration-v5-no-contact-imitation",
        ),
        (
            args.forward_iteration_v6_contact_abort_island_only,
            "forward",
            20260809,
            "--forward-iteration-v6-contact-abort-island-only",
        ),
        (
            args.reverse_iteration_v6_absolute_full_leg_targets,
            "reverse",
            20260810,
            "--reverse-iteration-v6-absolute-full-leg-targets",
        ),
    ):
        if not enabled:
            continue
        if args.expert != expert:
            raise ValueError(f"{flag} smoke is valid only for {expert}")
        if args.seed != seed:
            raise ValueError(f"{flag} smoke requires seed {seed}")
    iteration_mode = any(modes)
    exact_iteration_after_v2 = bool(
        args.forward_iteration_v3_touchdown_balance
        or args.reverse_iteration_v3_no_target_imitation
        or args.forward_iteration_v4_contact_event_validity_persistence
        or args.reverse_iteration_v4_residual_transfer_gain_024
        or args.forward_v5_contact_pulse_abort_scale_only
        or args.reverse_iteration_v5_no_contact_imitation
        or args.forward_iteration_v6_contact_abort_island_only
        or args.reverse_iteration_v6_absolute_full_leg_targets
    )
    if iteration_mode and (
        args.physical_command is not None or args.policy_command is not None
    ):
        raise ValueError(
            "iteration smoke forbids command overrides and uses its exact "
            "curriculum and policy anchor"
        )
    if exact_iteration_after_v2 and not np.isclose(
        args.reset_noise_multiplier, 1.0, rtol=0.0, atol=0.0
    ):
        raise ValueError(
            "iteration-v3/v4/v5/v6 smoke requires reset-noise multiplier 1.0"
        )
    if not 1 <= args.steps <= 5:
        raise ValueError("--steps must be in [1, 5]; this is not a training runner")


def resolve_smoke_contract(
    args: argparse.Namespace, backend_contract: Any
) -> dict[str, Any]:
    """Load the exact iteration authorization/spec without importing PPO."""

    _validate_smoke_cli(args)
    iteration_v6 = bool(
        args.forward_iteration_v6_contact_abort_island_only
        or args.reverse_iteration_v6_absolute_full_leg_targets
    )
    iteration_v6_core_source = (
        backend_contract.require_iteration_v6_core_source()
        if iteration_v6
        else None
    )
    if args.forward_iteration_v2:
        minimum_spec = backend_contract.load_forward_minimum_spec()
        authorization = (
            backend_contract.load_forward_iteration_v2_authorization()
        )
        anchors = backend_contract.resolve_anchor_config(
            "forward", forward_iteration_v2=True
        )
        reward_scales = backend_contract.forward_iteration_v2_reward_scales()
        reverse_composition = None
        legacy_reward_config_overrides = None
        mode = "forward_iteration_v2"
        preflight_contract_id = "H4_FORWARD_ITERATION_V2_NO_PPO_PREFLIGHT_FROM_V22"
    elif args.forward_iteration_v3_touchdown_balance:
        minimum_spec = backend_contract.load_forward_minimum_spec()
        authorization = (
            backend_contract.load_forward_iteration_v3_touchdown_balance_authorization()
        )
        anchors = backend_contract.resolve_anchor_config(
            "forward", forward_iteration_v3_touchdown_balance=True
        )
        reward_scales = (
            backend_contract.forward_iteration_v3_touchdown_balance_reward_scales()
        )
        reverse_composition = None
        legacy_reward_config_overrides = None
        mode = "forward_iteration_v3_touchdown_balance"
        preflight_contract_id = (
            "H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_NO_PPO_PREFLIGHT_FROM_V22"
        )
    elif args.forward_iteration_v4_contact_event_validity_persistence:
        minimum_spec = backend_contract.load_forward_minimum_spec()
        authorization = (
            backend_contract.load_forward_iteration_v4_contact_event_validity_persistence_authorization()
        )
        anchors = backend_contract.resolve_anchor_config(
            "forward",
            forward_iteration_v4_contact_event_validity_persistence=True,
        )
        reward_scales = backend_contract.forward_iteration_v2_reward_scales()
        reverse_composition = None
        legacy_reward_config_overrides = None
        mode = "forward_iteration_v4_contact_event_validity_persistence"
        preflight_contract_id = (
            "H4_FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_"
            "NO_PPO_PREFLIGHT_FROM_V22"
        )
    elif args.forward_v5_contact_pulse_abort_scale_only:
        minimum_spec = backend_contract.load_forward_minimum_spec()
        authorization = (
            backend_contract.load_forward_iteration_v5_contact_pulse_abort_scale_only_authorization()
        )
        anchors = backend_contract.resolve_anchor_config(
            "forward", forward_v5_contact_pulse_abort_scale_only=True
        )
        reward_scales = (
            backend_contract.forward_iteration_v5_contact_pulse_abort_scale_only_reward_scales()
        )
        reverse_composition = None
        legacy_reward_config_overrides = None
        mode = "forward_v5_contact_pulse_abort_scale_only"
        preflight_contract_id = (
            backend_contract.FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_NO_PPO_CONTRACT_ID
        )
    elif args.forward_iteration_v6_contact_abort_island_only:
        minimum_spec = backend_contract.load_forward_minimum_spec()
        authorization = (
            backend_contract.load_forward_iteration_v6_contact_abort_island_only_authorization()
        )
        anchors = backend_contract.resolve_anchor_config(
            "forward", forward_iteration_v6_contact_abort_island_only=True
        )
        reward_scales = (
            backend_contract.forward_iteration_v6_contact_abort_island_only_reward_scales()
        )
        reverse_composition = None
        legacy_reward_config_overrides = None
        mode = "forward_iteration_v6_contact_abort_island_only"
        preflight_contract_id = (
            backend_contract.FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_NO_PPO_CONTRACT_ID
        )
    elif args.reverse_iteration_v2:
        minimum_spec = backend_contract.load_reverse_minimum_spec()
        reverse_composition = (
            backend_contract.load_reverse_composition_authorization()
        )
        authorization = (
            backend_contract.load_reverse_iteration_v2_authorization()
        )
        anchors = backend_contract.resolve_anchor_config(
            "reverse", reverse_iteration_v2=True
        )
        reward_scales = backend_contract.reverse_iteration_v2_reward_scales()
        legacy_reward_config_overrides = dict(
            backend_contract.REVERSE_ITERATION_V2_LEGACY_REWARD_CONFIG
        )
        mode = "reverse_iteration_v2"
        preflight_contract_id = "H4_REVERSE_ITERATION_V2_NO_PPO_PREFLIGHT_FROM_V22"
    elif args.reverse_iteration_v3_no_target_imitation:
        minimum_spec = backend_contract.load_reverse_minimum_spec()
        reverse_composition = (
            backend_contract.load_reverse_composition_authorization()
        )
        authorization = (
            backend_contract.load_reverse_iteration_v3_no_target_imitation_authorization()
        )
        anchors = backend_contract.resolve_anchor_config(
            "reverse", reverse_iteration_v3_no_target_imitation=True
        )
        reward_scales = backend_contract.reverse_iteration_v2_reward_scales()
        legacy_reward_config_overrides = dict(
            backend_contract.REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG
        )
        mode = "reverse_iteration_v3_no_target_imitation"
        preflight_contract_id = (
            "H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_NO_PPO_PREFLIGHT_FROM_V22"
        )
    elif args.reverse_iteration_v4_residual_transfer_gain_024:
        minimum_spec = backend_contract.load_reverse_minimum_spec()
        reverse_composition = (
            backend_contract.load_reverse_composition_authorization()
        )
        authorization = (
            backend_contract.load_reverse_iteration_v4_residual_transfer_gain_024_authorization()
        )
        anchors = backend_contract.resolve_anchor_config(
            "reverse", reverse_iteration_v4_residual_transfer_gain_024=True
        )
        reward_scales = backend_contract.reverse_iteration_v2_reward_scales()
        legacy_reward_config_overrides = dict(
            backend_contract.REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG
        )
        mode = "reverse_iteration_v4_residual_transfer_gain_024"
        preflight_contract_id = (
            "H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_"
            "NO_PPO_PREFLIGHT_FROM_V22"
        )
    elif args.reverse_iteration_v5_no_contact_imitation:
        minimum_spec = backend_contract.load_reverse_minimum_spec()
        reverse_composition = (
            backend_contract.load_reverse_composition_authorization()
        )
        authorization = (
            backend_contract.load_reverse_iteration_v5_no_contact_imitation_authorization()
        )
        anchors = backend_contract.resolve_anchor_config(
            "reverse", reverse_iteration_v5_no_contact_imitation=True
        )
        reward_scales = backend_contract.reverse_iteration_v2_reward_scales()
        legacy_reward_config_overrides = dict(
            backend_contract.REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_LEGACY_REWARD_CONFIG
        )
        mode = "reverse_iteration_v5_no_contact_imitation"
        preflight_contract_id = (
            backend_contract.REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_NO_PPO_CONTRACT_ID
        )
    elif args.reverse_iteration_v6_absolute_full_leg_targets:
        minimum_spec = backend_contract.load_reverse_minimum_spec()
        reverse_composition = (
            backend_contract.load_reverse_composition_authorization()
        )
        authorization = (
            backend_contract.load_reverse_iteration_v6_absolute_full_leg_targets_authorization()
        )
        anchors = backend_contract.resolve_anchor_config(
            "reverse", reverse_iteration_v6_absolute_full_leg_targets=True
        )
        reward_scales = backend_contract.reverse_iteration_v2_reward_scales()
        legacy_reward_config_overrides = dict(
            backend_contract.REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_LEGACY_REWARD_CONFIG
        )
        mode = "reverse_iteration_v6_absolute_full_leg_targets"
        preflight_contract_id = (
            backend_contract.REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_NO_PPO_CONTRACT_ID
        )
    else:
        return {
            "mode": "iteration_v1",
            "preflight_contract_id": f"H4_{args.expert.upper()}_ITERATION_V1_NO_PPO_SMOKE",
            "authorized_250k_contract_id": None,
            "minimum_spec": None,
            "authorization": None,
            "reverse_composition": None,
            "anchors": None,
            "reward_scales": None,
            "legacy_reward_config_overrides": None,
            "forward_v4_substep_contact": False,
            "forward_iteration_v6_contact_abort_island_only": False,
            "reverse_iteration_v6_absolute_full_leg_targets": False,
            "iteration_v6_core_source": None,
            "backward_residual_scale": 0.12,
        }
    if not all(authorization["semantic_audit"].values()):
        raise ValueError("iteration smoke authorization semantic audit failed")
    return {
        "mode": mode,
        "preflight_contract_id": preflight_contract_id,
        "authorized_250k_contract_id": authorization["contract_id"],
        "minimum_spec": minimum_spec,
        "authorization": authorization,
        "reverse_composition": reverse_composition,
        "anchors": anchors,
        "reward_scales": reward_scales,
        "legacy_reward_config_overrides": legacy_reward_config_overrides,
        "forward_v4_substep_contact": bool(
            args.forward_iteration_v4_contact_event_validity_persistence
            or args.forward_v5_contact_pulse_abort_scale_only
            or args.forward_iteration_v6_contact_abort_island_only
        ),
        "forward_iteration_v6_contact_abort_island_only": bool(
            args.forward_iteration_v6_contact_abort_island_only
        ),
        "reverse_iteration_v6_absolute_full_leg_targets": bool(
            args.reverse_iteration_v6_absolute_full_leg_targets
        ),
        "iteration_v6_core_source": iteration_v6_core_source,
        "backward_residual_scale": (
            backend_contract.REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN
            if args.reverse_iteration_v4_residual_transfer_gain_024
            else backend_contract.REVERSE_ITERATION_V6_RESIDUAL_AUTHORITY_SCALE
            if args.reverse_iteration_v6_absolute_full_leg_targets
            else 0.12
        ),
    }


def require_forward_v4_no_ppo_single_authority(
    traces: Sequence[Mapping[str, Any]], backend_contract: Any
) -> dict[str, Any]:
    """Close every single-authority invariant on each no-PPO control step."""

    return backend_contract.require_forward_v4_single_authority_samples(
        [item["forward_v4_single_authority"] for item in traces],
        label="forward_v4_no_ppo_smoke",
    )


def _resolved_commands(args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[float, ...]]:
    defaults = {
        "reverse": ((-0.05, 0.0, 0.0), (-0.05, 0.0, 0.0)),
        "forward": ((0.05, 0.0, 0.0), (0.10, -0.018, -0.170)),
    }
    physical_default, policy_default = defaults[args.expert]
    physical = args.physical_command or physical_default
    policy = args.policy_command or policy_default
    if np.linalg.norm(physical) <= 0.0:
        raise ValueError("smoke physical command must be moving")
    return tuple(physical), tuple(policy)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    _validate_smoke_cli(args)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    backend_contract = _load_h4_runner_backend_contract()
    smoke_contract = resolve_smoke_contract(args, backend_contract)
    backend_selector = backend_contract.resolve_jax_backend_selector(args.platform)
    os.environ["JAX_PLATFORMS"] = backend_selector
    xla_autotune_policy = backend_contract.configure_xla_autotune_policy(
        args.platform
    )
    physical, policy = _resolved_commands(args)
    trainer = _load_legacy_trainer()
    paths = trainer.generated_paths(args.generated_root.resolve())
    trainer._validate_generated_manifest(paths)
    stack = trainer._load_training_stack(args.source_root.resolve())
    backend_resolution = backend_contract.validate_resolved_jax_backend(
        stack["jax"],
        requested_platform=args.platform,
        selector=backend_selector,
    )
    debug_callback_preflight = backend_contract.run_jax_debug_callback_preflight(
        stack["jax"], stack["jp"]
    )
    constants = stack["constants"]
    scene_type = type(constants.FLAT_TERRAIN_BACKLASH_CALIBRATED_XML)
    constants.ROOT_PATH = scene_type(paths["package"].as_posix())
    constants.FLAT_TERRAIN_BACKLASH_CALIBRATED_XML = scene_type(
        paths["scene"].as_posix()
    )

    class TeacherArgs:
        backward_gait = None
        backward_left_gait = None
        backward_right_gait = None

    teacher_gaits = trainer.resolve_teacher_gaits(TeacherArgs(), paths)
    selected: dict[str, Any] | None = None
    if args.expert == "reverse":
        selected_path = args.selected_reverse_teacher.resolve()
        selected_sha = trainer.sha256_file(selected_path)
        if selected_sha != PINNED_SELECTED_REVERSE_TEACHER_SHA256:
            raise ValueError("selected reverse teacher hash drifted")
        selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
        adapter = selected_payload["adapter_contract"]
        teacher = selected_payload["teacher"]
        if (
            selected_payload["decision"]["training_use"]
            != "ALLOWED_AS_INITIALIZATION_PRIOR_ONLY"
            or teacher["validation"]["passed"] is not True
        ):
            raise ValueError("selected reverse teacher is not a validated training component")
        selected = {
            "path": selected_path,
            "sha256": selected_sha,
            "candidate_id": teacher["candidate_id"],
            "table": np.asarray(teacher["target_table_rad"], dtype=np.float64),
            "cadence_hz": float(adapter["cadence_hz"]),
            "phase_advance_bins": float(adapter["phase_advance_bins_per_control"]),
            "entry_phase_bins": float(adapter["entry_phase_preincrement_bins"]),
            "first_phase_bins": float(adapter["first_reference_phase_after_increment_bins"]),
        }
    LegacyEnvironment = trainer._make_environment_class(
        stack=stack,
        expert=args.expert,
        paths=paths,
        teacher_gaits=teacher_gaits,
        backward_residual_scale=smoke_contract["backward_residual_scale"],
    )
    jp = stack["jp"]

    def fixed_physical_sampler(_rng):
        return jp.asarray((*physical, 0.0, 0.0, 0.0, 0.0))

    mapper = make_anchor_command_mapper(physical, policy, xp=jp)
    if (
        args.forward_iteration_v2
        or args.forward_iteration_v3_touchdown_balance
        or args.forward_iteration_v4_contact_event_validity_persistence
        or args.forward_v5_contact_pulse_abort_scale_only
        or args.forward_iteration_v6_contact_abort_island_only
    ):
        physical_sampler = make_h4_forward_v2_physical_sampler(
            stack["jax"], jp
        )
    elif (
        args.reverse_iteration_v2
        or args.reverse_iteration_v3_no_target_imitation
        or args.reverse_iteration_v4_residual_transfer_gain_024
        or args.reverse_iteration_v5_no_contact_imitation
        or args.reverse_iteration_v6_absolute_full_leg_targets
    ):
        physical_sampler = make_h4_reverse_v2_physical_sampler(
            stack["jax"], jp
        )
    else:
        physical_sampler = fixed_physical_sampler
    reward_scales = smoke_contract["reward_scales"]
    if reward_scales is None:
        reward_scales = (
            H4QualityRewardScales(
                target_lag=0.0,
                left_target_lag=-0.125,
                right_target_lag=-0.125,
                reverse_speed_boundary=-1.0,
                reverse_cross_drift=-2.0,
                reverse_uncommanded_yaw_rate=-1.0,
                reverse_heading_drift=-1.0,
                reverse_phase_force_slip=-1.0,
                reverse_contact_priority_reversal_lag=-0.75,
            )
            if args.expert == "reverse"
            else H4QualityRewardScales()
        )
    Environment = make_h4_aligned_environment_class(
        legacy_environment_class=LegacyEnvironment,
        stack=stack,
        physical_command_sampler=physical_sampler,
        policy_observation_mapper=mapper,
        reward_scales=reward_scales,
        reset_noise_multiplier=args.reset_noise_multiplier,
        reverse_teacher_cycle_hz=(selected["cadence_hz"] if selected else 1.75),
        reverse_teacher_target_table=(selected["table"] if selected else None),
        reverse_teacher_phase_advance_bins=(
            selected["phase_advance_bins"] if selected else None
        ),
        reverse_teacher_entry_phase_bins=(
            selected["entry_phase_bins"] if selected else 0.0
        ),
        include_h4_actor_observables=True,
        forward_v4_substep_contact=smoke_contract["forward_v4_substep_contact"],
        forward_iteration_v6_contact_abort_island_only=smoke_contract[
            "forward_iteration_v6_contact_abort_island_only"
        ],
        reverse_iteration_v6_absolute_full_leg_targets=smoke_contract[
            "reverse_iteration_v6_absolute_full_leg_targets"
        ],
        legacy_reward_config_overrides=smoke_contract[
            "legacy_reward_config_overrides"
        ],
    )
    env = Environment()
    legacy_reward_config_audit = None
    if (
        args.reverse_iteration_v2
        or args.reverse_iteration_v3_no_target_imitation
        or args.reverse_iteration_v4_residual_transfer_gain_024
        or args.reverse_iteration_v5_no_contact_imitation
        or args.reverse_iteration_v6_absolute_full_leg_targets
    ):
        expected_legacy = {
            **smoke_contract["legacy_reward_config_overrides"],
            "backward_residual_scale": smoke_contract[
                "backward_residual_scale"
            ],
        }
        actual_legacy = {
            "target_imitation": float(
                env._config.reward_config.scales.target_imitation
            ),
            "contact_imitation": float(
                env._config.reward_config.scales.contact_imitation
            ),
            "tracking_sigma": float(env._config.reward_config.tracking_sigma),
            "backward_residual_scale": float(env._backward_residual_scale),
        }
        exact = all(
            np.isclose(
                actual_legacy[name], value, rtol=0.0, atol=0.0
            )
            for name, value in expected_legacy.items()
        )
        if not exact:
            raise RuntimeError(
                "reverse iteration no-PPO legacy reward contract drifted: "
                f"{actual_legacy} != {expected_legacy}"
            )
        legacy_reward_config_audit = {
            "expected": expected_legacy,
            "actual": actual_legacy,
            "exact": True,
        }
    state = env.reset(stack["jax"].random.PRNGKey(args.seed))
    forward_v4_source_semantic_preflight = None
    if (
        args.forward_iteration_v4_contact_event_validity_persistence
        or args.forward_v5_contact_pulse_abort_scale_only
        or args.forward_iteration_v6_contact_abort_island_only
    ):
        forward_v4_source_semantic_preflight = (
            backend_contract.run_forward_v4_source_semantic_preflight(
                stack["jax"],
                stack["jp"],
                env,
                state,
                source_physics_step=stack["joystick"].mjx_env.step,
                mjx_step=stack["joystick"].mjx_env.mjx.step,
                source_root=args.source_root.resolve(),
                joystick_module=stack["joystick"],
                mjx_env_module=stack["joystick"].mjx_env,
                seed=args.seed,
                reset_noise_multiplier=args.reset_noise_multiplier,
            )
        )
    sampled_physical_command = np.asarray(
        state.info["h4_physical_command"], dtype=np.float64
    )[:3]
    mapped_policy_command = np.asarray(
        mapper(
            jp.asarray(
                (*sampled_physical_command.tolist(), 0.0, 0.0, 0.0, 0.0)
            )
        ),
        dtype=np.float64,
    )[:3]
    reset_device_audit = backend_contract.audit_jax_tree_placement(
        stack["jax"],
        state,
        expected_platform=backend_contract.JAX_RESOLVED_BACKENDS[args.platform],
        label="smoke_reset_state",
    )
    reset_arrays_finite = all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in stack["jax"].tree_util.tree_leaves(state)
        if np.issubdtype(np.asarray(leaf).dtype, np.number)
    )
    reset_targets = np.asarray(state.data.ctrl, dtype=np.float64)
    traces: list[dict[str, Any]] = []
    previous_targets = reset_targets.copy()
    step_fn = stack["jax"].jit(env.step) if args.jit else env.step
    for step_index in range(args.steps):
        state = step_fn(state, jp.zeros(14))
        state.reward.block_until_ready()
        applied = np.asarray(state.data.ctrl, dtype=np.float64)
        delta = np.abs(applied - previous_targets)
        traces.append(
            {
                "step_index": step_index,
                "applied_targets": applied.tolist(),
                "maximum_leg_target_delta_rad": float(
                    np.max(delta[np.asarray(LEG_ACTION_INDICES)])
                ),
                "physical_command": np.asarray(
                    state.info["h4_physical_command"]
                ).tolist(),
                "policy_observation_command": np.asarray(
                    state.info["h4_policy_observation_command"]
                ).tolist(),
                "normalized_normal_force": np.asarray(
                    state.info["h4_normalized_force"]
                ).tolist(),
                "force_contact": np.asarray(
                    state.info["h4_force_contact"]
                ).astype(bool).tolist(),
                "per_foot_contact_point_slip_m_s": np.asarray(
                    state.info["h4_per_foot_slip_m_s"]
                ).tolist(),
                "force_contact_slip_rms_m_s": float(
                    state.info["h4_slip_rms_m_s"]
                ),
                "pre_guard_max_delta_rad": float(
                    state.info["h4_pre_guard_max_delta_rad"]
                ),
                "slew_feasibility_loss": float(
                    state.info["h4_slew_feasibility_loss"]
                ),
                "target_lag_loss": float(state.info["h4_target_lag_loss"]),
                "left_target_lag_loss": float(
                    state.info["h4_left_target_lag_loss"]
                ),
                "right_target_lag_loss": float(
                    state.info["h4_right_target_lag_loss"]
                ),
                "upstream_max_delta_rad": float(
                    state.info["h4_upstream_max_delta_rad"]
                ),
                "reverse_teacher_precomposer_active": bool(
                    state.info["h4_reverse_teacher_precomposer_active"]
                ),
                "slip_tail_loss": float(state.info["h4_slip_tail_loss"]),
                "stance_slip_budget_loss": float(
                    state.info["h4_stance_slip_budget_loss"]
                ),
                "single_support_band_loss": float(
                    state.info["h4_single_support_band_loss"]
                ),
                "touchdown_count_balance_loss": float(
                    state.info["h4_touchdown_count_balance_loss"]
                ),
                "flight": float(state.info["h4_flight"]),
                "phase17_left_force_slip_loss": float(
                    state.info["h4_phase17_left_force_slip_loss"]
                ),
                "phase17_left_knee_excess_loss": float(
                    state.info["h4_phase17_left_knee_excess_loss"]
                ),
                "phase17_opposite_leg_lag_loss": float(
                    state.info["h4_phase17_opposite_leg_lag_loss"]
                ),
                "forward_cross_drift_loss": float(
                    state.info["h4_forward_cross_drift_loss"]
                ),
                "forward_yaw_rate_loss": float(
                    state.info["h4_forward_yaw_rate_loss"]
                ),
                "forward_heading_drift_loss": float(
                    state.info["h4_forward_heading_drift_loss"]
                ),
                "reverse_speed_boundary_loss": float(
                    state.info["h4_reverse_speed_boundary_loss"]
                ),
                "reverse_cross_drift_loss": float(
                    state.info["h4_reverse_cross_drift_loss"]
                ),
                "reverse_yaw_rate_loss": float(
                    state.info["h4_reverse_yaw_rate_loss"]
                ),
                "reverse_heading_drift_loss": float(
                    state.info["h4_reverse_heading_drift_loss"]
                ),
                "reverse_phase_force_slip_loss": float(
                    state.info["h4_reverse_phase_force_slip_loss"]
                ),
                "reverse_contact_priority_reversal_lag_loss": float(
                    state.info[
                        "h4_reverse_contact_priority_reversal_lag_loss"
                    ]
                ),
            }
        )
        if (
            args.forward_iteration_v4_contact_event_validity_persistence
            or args.forward_v5_contact_pulse_abort_scale_only
            or args.forward_iteration_v6_contact_abort_island_only
        ):
            authority_sample = (
                backend_contract.forward_v4_single_authority_sample_from_info(
                    state.info
                )
            )
            traces[-1]["forward_v4_single_authority"] = {
                name: (
                    bool(np.asarray(value))
                    if name
                    in {
                        "dynamic6_exact",
                        "dynamic6_field_count_exact",
                        "saved_dynamic6_field_count_exact",
                        "saved_dynamic6_all_finite",
                        "telemetry_force_shape_valid",
                        "telemetry_force_all_finite",
                        "authority_violation",
                    }
                    else float(np.asarray(value))
                    if name == "dynamic6_max_abs_error"
                    else int(np.asarray(value))
                )
                for name, value in authority_sample.items()
            }
        if (
            args.forward_iteration_v6_contact_abort_island_only
            or args.reverse_iteration_v6_absolute_full_leg_targets
        ):
            v6_sample = backend_contract._iteration_v6_sample_from_info(
                state.info, expert=args.expert
            )
            boolean_names = (
                {"routing_exact", "routing_violation"}
                if args.expert == "forward"
                else {
                    "decoder_exact",
                    "leg_count_exact",
                    "head_zero_exact",
                    "teacher_target_contribution_zero_exact",
                    "decoder_all_finite",
                    "decoder_violation",
                    "precomposer_call_count_exact",
                    "final_guard_call_count_exact",
                }
            )
            traces[-1]["iteration_v6_runtime"] = {
                name: (
                    bool(np.asarray(value))
                    if name in boolean_names
                    else float(np.asarray(value))
                )
                for name, value in v6_sample.items()
            }
            if args.reverse_iteration_v6_absolute_full_leg_targets:
                vector_sample = (
                    backend_contract.reverse_iteration_v6_decoder_vector_sample_from_info(
                        state.info
                    )
                )
                traces[-1]["reverse_iteration_v6_decoder_vectors"] = {
                    name: np.asarray(value).tolist()
                    for name, value in vector_sample.items()
                }
        previous_targets = applied

    forward_v4_single_authority_runtime = None
    if (
        args.forward_iteration_v4_contact_event_validity_persistence
        or args.forward_v5_contact_pulse_abort_scale_only
        or args.forward_iteration_v6_contact_abort_island_only
    ):
        forward_v4_single_authority_runtime = (
            require_forward_v4_no_ppo_single_authority(
                traces, backend_contract
            )
        )
    iteration_v6_runtime = None
    reverse_iteration_v6_decoder_vector_runtime = None
    if (
        args.forward_iteration_v6_contact_abort_island_only
        or args.reverse_iteration_v6_absolute_full_leg_targets
    ):
        iteration_v6_runtime = backend_contract.require_iteration_v6_runtime_samples(
            [item["iteration_v6_runtime"] for item in traces],
            expert=args.expert,
            label="iteration_v6_no_ppo_smoke",
        )
        if args.reverse_iteration_v6_absolute_full_leg_targets:
            reverse_iteration_v6_decoder_vector_runtime = (
                backend_contract.require_reverse_iteration_v6_decoder_vector_samples(
                    [
                        item["reverse_iteration_v6_decoder_vectors"]
                        for item in traces
                    ],
                    label="reverse_iteration_v6_no_ppo_smoke",
                )
            )

    final_targets = np.asarray(state.data.ctrl, dtype=np.float64)
    final_device_audit = backend_contract.audit_jax_tree_placement(
        stack["jax"],
        state,
        expected_platform=backend_contract.JAX_RESOLVED_BACKENDS[args.platform],
        label="smoke_final_state",
    )
    final_arrays_finite = all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in stack["jax"].tree_util.tree_leaves(state)
        if np.issubdtype(np.asarray(leaf).dtype, np.number)
    )
    state_observation = np.asarray(state.obs["state"], dtype=np.float64)
    privileged_observation = np.asarray(
        state.obs["privileged_state"], dtype=np.float64
    )
    visible_targets = state_observation[83:97]
    maximum_delta = max(
        item["maximum_leg_target_delta_rad"] for item in traces
    )
    invariants = {
        "ppo_training_not_started": True,
        "backend_resolution_passed": backend_resolution["passed"],
        "xla_autotune_policy_passed": xla_autotune_policy["passed"],
        "debug_callback_preflight_passed": debug_callback_preflight["passed"],
        "reset_and_final_state_on_requested_backend": bool(
            reset_device_audit["passed"] and final_device_audit["passed"]
        ),
        "reset_and_final_state_arrays_finite": bool(
            reset_arrays_finite and final_arrays_finite
        ),
        "action_delay_exactly_zero": bool(
            env._config.noise_config.action_min_delay == 0
            and env._config.noise_config.action_max_delay == 1
        ),
        "guard_called_once_per_control_step": int(
            state.info["h4_guard_steps"]
        )
        == args.steps,
        "maximum_leg_target_delta_within_0p04_rad": maximum_delta
        <= MAX_TARGET_DELTA_PER_TICK_RAD + 1.0e-6,
        "reset_head_qpos_and_target_zero": bool(
            np.array_equal(reset_targets[5:9], np.zeros(4))
        ),
        "final_head_target_zero": bool(
            np.array_equal(final_targets[5:9], np.zeros(4))
        ),
        "observation_target_matches_applied_target": bool(
            np.allclose(visible_targets, final_targets, atol=1.0e-6, rtol=0.0)
        ),
        "physical_reward_command_preserved": bool(
            np.allclose(
                np.asarray(state.info["command"])[:3],
                sampled_physical_command,
                atol=1.0e-7,
            )
        ),
        "policy_command_separately_visible": bool(
            np.allclose(
                np.asarray(state.info["h4_policy_observation_command"])[:3],
                mapped_policy_command,
                atol=1.0e-7,
            )
        ),
        "iteration_v2_authorization_and_sources_bound": bool(
            not (args.forward_iteration_v2 or args.reverse_iteration_v2)
            or (
                all(
                    smoke_contract["authorization"][
                        "semantic_audit"
                    ].values()
                )
                and len(
                    smoke_contract["authorization"][
                        "bound_causal_inputs"
                    ]
                )
                == 3
            )
        ),
        "iteration_v2_uses_exact_curriculum_sampler": bool(
            not (args.forward_iteration_v2 or args.reverse_iteration_v2)
            or physical_sampler is not fixed_physical_sampler
        ),
        "reverse_iteration_v2_legacy_reward_config_exact": bool(
            not args.reverse_iteration_v2
            or (
                legacy_reward_config_audit is not None
                and legacy_reward_config_audit["exact"]
            )
        ),
        "actor_observation_has_h4_contact_slip_and_state_channels": bool(
            state_observation.shape == (116,)
        ),
        "critic_imitation_tail_matches_returned_info": bool(
            privileged_observation.shape == (227,)
            and np.allclose(
                privileged_observation[
                    LEGACY_PRIVILEGED_REFERENCE_SLICE.start
                    + 15 : LEGACY_PRIVILEGED_REFERENCE_SLICE.stop
                    + 15
                ],
                np.asarray(state.info["current_reference_motion"]),
                atol=1.0e-7,
                rtol=0.0,
            )
            and np.allclose(
                privileged_observation[
                    LEGACY_PRIVILEGED_IMITATION_INDEX_SLICE.start + 15
                ],
                np.asarray(state.info["imitation_i"]),
                atol=1.0e-7,
                rtol=0.0,
            )
            and np.allclose(
                privileged_observation[
                    LEGACY_PRIVILEGED_IMITATION_PHASE_SLICE.start
                    + 15 : LEGACY_PRIVILEGED_IMITATION_PHASE_SLICE.stop
                    + 15
                ],
                np.asarray(state.info["imitation_phase"]),
                atol=1.0e-7,
                rtol=0.0,
            )
        ),
        "all_trace_scalars_finite": all(
            np.isfinite(value)
            for item in traces
            for value in item.values()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        ),
        "quality_terms_are_live": any(
            item["force_contact_slip_rms_m_s"] > 0.0
            or item["single_support_band_loss"] > 0.0
            or item["forward_cross_drift_loss"] > 0.0
            for item in traces
        ),
    }
    if args.expert == "reverse":
        invariants.update(
            {
                "selected_reverse_teacher_hash_pinned": bool(
                    selected is not None
                    and selected["sha256"]
                    == PINNED_SELECTED_REVERSE_TEACHER_SHA256
                ),
                "selected_teacher_adapter_active": bool(
                    all(
                        item["reverse_teacher_precomposer_active"]
                        for item in traces
                    )
                ),
                "training_visible_teacher_jump_within_0p04_rad": bool(
                    traces[0]["pre_guard_max_delta_rad"]
                    <= MAX_TARGET_DELTA_PER_TICK_RAD + 1.0e-6
                ),
            }
        )
    iteration_v3 = bool(
        args.forward_iteration_v3_touchdown_balance
        or args.reverse_iteration_v3_no_target_imitation
    )
    if iteration_v3:
        authorization_bound = bool(
            smoke_contract["authorization"] is not None
            and all(smoke_contract["authorization"]["semantic_audit"].values())
            and len(smoke_contract["authorization"]["bound_causal_inputs"]) == 3
        )
        invariants.update(
            {
                "iteration_v3_authorization_and_sources_bound": authorization_bound,
                "iteration_v3_uses_exact_curriculum_sampler": (
                    physical_sampler is not fixed_physical_sampler
                ),
                "reverse_iteration_v3_legacy_reward_config_exact": bool(
                    not args.reverse_iteration_v3_no_target_imitation
                    or (
                        legacy_reward_config_audit is not None
                        and legacy_reward_config_audit["exact"]
                    )
                ),
            }
        )
    iteration_v4 = bool(
        args.forward_iteration_v4_contact_event_validity_persistence
        or args.reverse_iteration_v4_residual_transfer_gain_024
    )
    if iteration_v4:
        authorization_bound = bool(
            smoke_contract["authorization"] is not None
            and all(smoke_contract["authorization"]["semantic_audit"].values())
            and len(smoke_contract["authorization"]["bound_causal_inputs"]) == 4
            and len(smoke_contract["authorization"]["bound_causal_sources"]) == 5
        )
        invariants.update(
            {
                "iteration_v4_authorization_and_sources_bound": authorization_bound,
                "iteration_v4_uses_exact_curriculum_sampler": (
                    physical_sampler is not fixed_physical_sampler
                ),
                "forward_iteration_v4_core_opt_in_exact": bool(
                    not args.forward_iteration_v4_contact_event_validity_persistence
                    or env.h4_forward_v4_substep_contact is True
                ),
                "forward_iteration_v4_source_semantic_preflight_exact": bool(
                    not args.forward_iteration_v4_contact_event_validity_persistence
                    or (
                        forward_v4_source_semantic_preflight is not None
                        and forward_v4_source_semantic_preflight[
                            "dynamic6_exact"
                        ]
                        is True
                        and forward_v4_source_semantic_preflight[
                            "dynamic6_max_abs_error"
                        ]
                        == 0.0
                        and forward_v4_source_semantic_preflight[
                            "dynamic6_field_count"
                        ]
                        == 6
                        and forward_v4_source_semantic_preflight[
                            "derived_diagnostics"
                        ]["exclusion_is_semantic_not_tolerance"]
                        is True
                        and forward_v4_source_semantic_preflight[
                            "derived_diagnostics"
                        ]["numeric_tolerance_used"]
                        is False
                        and forward_v4_source_semantic_preflight["passed"] is True
                    )
                ),
                "forward_iteration_v4_single_authority_runtime_exact": bool(
                    not args.forward_iteration_v4_contact_event_validity_persistence
                    or (
                        forward_v4_single_authority_runtime is not None
                        and forward_v4_single_authority_runtime[
                            "dynamic6_exact"
                        ]
                        is True
                        and forward_v4_single_authority_runtime[
                            "dynamic6_max_abs_error"
                        ]
                        == 0.0
                        and forward_v4_single_authority_runtime[
                            "dynamic6_field_count"
                        ]
                        == 6
                        and forward_v4_single_authority_runtime[
                            "dynamic6_field_count_exact"
                        ]
                        is True
                        and forward_v4_single_authority_runtime[
                            "saved_dynamic6_substep_count"
                        ]
                        == 10
                        and forward_v4_single_authority_runtime[
                            "saved_dynamic6_field_count"
                        ]
                        == 6
                        and forward_v4_single_authority_runtime[
                            "saved_dynamic6_field_count_exact"
                        ]
                        is True
                        and forward_v4_single_authority_runtime[
                            "saved_dynamic6_all_finite"
                        ]
                        is True
                        and forward_v4_single_authority_runtime[
                            "telemetry_force_shape"
                        ]
                        == [2]
                        and forward_v4_single_authority_runtime[
                            "telemetry_force_shape_valid"
                        ]
                        is True
                        and forward_v4_single_authority_runtime[
                            "telemetry_force_all_finite"
                        ]
                        is True
                        and forward_v4_single_authority_runtime[
                            "authority_violation_count"
                        ]
                        == 0
                        and forward_v4_single_authority_runtime[
                            "assertion_token_sum"
                        ]
                        == 0.0
                        and forward_v4_single_authority_runtime["passed"] is True
                    )
                ),
                "reverse_iteration_v4_legacy_reward_and_residual_exact": bool(
                    not args.reverse_iteration_v4_residual_transfer_gain_024
                    or (
                        legacy_reward_config_audit is not None
                        and legacy_reward_config_audit["exact"]
                        and np.isclose(
                            legacy_reward_config_audit["actual"][
                                "backward_residual_scale"
                            ],
                            0.24,
                            rtol=0.0,
                            atol=0.0,
                        )
                    )
                ),
            }
        )
    iteration_v5 = bool(
        args.forward_v5_contact_pulse_abort_scale_only
        or args.reverse_iteration_v5_no_contact_imitation
    )
    if iteration_v5:
        authorization = smoke_contract["authorization"]
        authorization_bound = bool(
            authorization is not None
            and all(authorization["semantic_audit"].values())
            and len(authorization["bound_causal_inputs"])
            == (7 if args.forward_v5_contact_pulse_abort_scale_only else 13)
            and len(authorization["bound_historical_v4_sources"]) == 5
        )
        reward_contract = authorization["payload"]["reward_contract"]
        invariants.update(
            {
                "iteration_v5_authorization_and_sources_bound": authorization_bound,
                "iteration_v5_uses_exact_curriculum_sampler": (
                    physical_sampler is not fixed_physical_sampler
                ),
                "forward_v5_contact_pulse_single_delta_exact": bool(
                    not args.forward_v5_contact_pulse_abort_scale_only
                    or (
                        reward_contract["only_scale_delta"]
                        == {
                            "name": "h4_contact_pulse_40ms",
                            "iteration_v4_scale": -1.0,
                            "iteration_v5_scale": -2.0,
                        }
                        and reward_scales.as_reward_scale_dict()[
                            "h4_contact_pulse_40ms"
                        ]
                        == -2.0
                        and env.h4_forward_v4_substep_contact is True
                        and forward_v4_source_semantic_preflight is not None
                        and forward_v4_source_semantic_preflight["passed"] is True
                        and forward_v4_single_authority_runtime is not None
                        and forward_v4_single_authority_runtime["passed"] is True
                    )
                ),
                "reverse_v5_contact_imitation_single_delta_exact": bool(
                    not args.reverse_iteration_v5_no_contact_imitation
                    or (
                        authorization["payload"]["legacy_reward_config"][
                            "only_scale_delta"
                        ]
                        == {
                            "name": "contact_imitation",
                            "iteration_v3_scale": 15.0,
                            "iteration_v5_scale": 0.0,
                        }
                        and legacy_reward_config_audit is not None
                        and legacy_reward_config_audit["exact"]
                        and np.isclose(
                            legacy_reward_config_audit["actual"][
                                "backward_residual_scale"
                            ],
                            0.12,
                            rtol=0.0,
                            atol=0.0,
                        )
                        and authorization["payload"]["causal_inputs"][
                            "rejected_v4_diagnostic"
                        ]["promotion_allowed"]
                        is False
                    )
                ),
                "forward_reverse_v5_families_uncoupled": bool(
                    (args.forward_v5_contact_pulse_abort_scale_only)
                    is not (args.reverse_iteration_v5_no_contact_imitation)
                ),
            }
        )
    iteration_v6 = bool(
        args.forward_iteration_v6_contact_abort_island_only
        or args.reverse_iteration_v6_absolute_full_leg_targets
    )
    if iteration_v6:
        authorization = smoke_contract["authorization"]
        authorization_bound = bool(
            authorization is not None
            and all(authorization["semantic_audit"].values())
            and len(authorization["bound_causal_inputs"])
            == (6 if args.forward_iteration_v6_contact_abort_island_only else 8)
            and len(authorization["bound_historical_v5_sources"]) == 5
        )
        expected_core_contract = (
            backend_contract.FORWARD_ITERATION_V6_CONTACT_ABORT_CONTRACT_ID
            if args.expert == "forward"
            else backend_contract.REVERSE_ITERATION_V6_ABSOLUTE_TARGET_CONTRACT_ID
        )
        actual_core_contract = (
            env.h4_forward_iteration_v6_contract_id
            if args.expert == "forward"
            else env.h4_reverse_iteration_v6_contract_id
        )
        invariants.update(
            {
                "iteration_v6_authorization_and_sources_bound": authorization_bound,
                "iteration_v6_core_source_bound": bool(
                    smoke_contract["iteration_v6_core_source"]
                    == {
                        "path": str(backend_contract.ALIGNMENT_MODULE_PATH.resolve()),
                        "sha256": (
                            backend_contract.PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256
                        ),
                    }
                ),
                "iteration_v6_uses_exact_curriculum_sampler": (
                    physical_sampler is not fixed_physical_sampler
                ),
                "iteration_v6_core_family_exact": bool(
                    actual_core_contract == expected_core_contract
                    and env.h4_forward_iteration_v6_contact_abort_island_only
                    is args.forward_iteration_v6_contact_abort_island_only
                    and env.h4_reverse_iteration_v6_absolute_full_leg_targets
                    is args.reverse_iteration_v6_absolute_full_leg_targets
                ),
                "iteration_v6_runtime_exact": bool(
                    iteration_v6_runtime is not None
                    and iteration_v6_runtime["expert"] == args.expert
                    and iteration_v6_runtime["observed_step_count"] == args.steps
                    and iteration_v6_runtime["compiled_invariant_assertion_passed"]
                    is True
                    and iteration_v6_runtime["passed"] is True
                ),
                "forward_v6_island_only_routing_exact": bool(
                    not args.forward_iteration_v6_contact_abort_island_only
                    or (
                        env.h4_forward_v4_substep_contact is True
                        and reward_scales.as_reward_scale_dict()[
                            "h4_contact_pulse_40ms"
                        ]
                        == -1.0
                        and authorization["payload"]["reward_routing_contract"][
                            "qualifying_loss"
                        ]
                        == "aborted_contact_island_loss"
                        and authorization["payload"]["reward_routing_contract"][
                            "off_gap_reward_contribution"
                        ]
                        == 0.0
                        and forward_v4_source_semantic_preflight is not None
                        and forward_v4_source_semantic_preflight["passed"] is True
                        and forward_v4_single_authority_runtime is not None
                        and forward_v4_single_authority_runtime["passed"] is True
                        and args.forward_v5_contact_pulse_abort_scale_only is False
                    )
                ),
                "reverse_v6_absolute_teacher_timing_only_exact": bool(
                    not args.reverse_iteration_v6_absolute_full_leg_targets
                    or (
                        legacy_reward_config_audit is not None
                        and legacy_reward_config_audit["exact"]
                        and legacy_reward_config_audit["actual"]
                        == {
                            "target_imitation": 0.0,
                            "contact_imitation": 0.0,
                            "tracking_sigma": 0.01,
                            "backward_residual_scale": 0.0,
                        }
                        and authorization["payload"][
                            "action_parameterization_contract"
                        ]["decoder"]
                        == "FROZEN_V22_CALIBRATED_ABSOLUTE_FULL_LEG"
                        and authorization["payload"][
                            "action_parameterization_contract"
                        ]["directional_span_fraction"]
                        == 0.9
                        and authorization["payload"]["teacher_timing_contract"][
                            "role"
                        ]
                        == "PHASE_TIMING_PRIOR_ONLY"
                        and args.reverse_iteration_v5_no_contact_imitation is False
                    )
                ),
                "reverse_v6_decoder_vector_telemetry_exact": bool(
                    not args.reverse_iteration_v6_absolute_full_leg_targets
                    or (
                        reverse_iteration_v6_decoder_vector_runtime is not None
                        and reverse_iteration_v6_decoder_vector_runtime["passed"]
                        is True
                        and reverse_iteration_v6_decoder_vector_runtime[
                            "observed_step_count"
                        ]
                        == args.steps
                    )
                ),
                "forward_reverse_v6_families_uncoupled": bool(
                    args.forward_iteration_v6_contact_abort_island_only
                    is not args.reverse_iteration_v6_absolute_full_leg_targets
                ),
            }
        )
    result = {
        "schema_version": 1,
        "status": "PASS" if all(invariants.values()) else "FAIL",
        "hardware_deployment": "PROHIBITED",
        "purpose": "H4 alignment smoke only; no PPO and no artifact writes",
        "preflight_contract_id": smoke_contract["preflight_contract_id"],
        "authorized_250k_contract_id": smoke_contract[
            "authorized_250k_contract_id"
        ],
        "qualification_use": "NO_PPO_PREFLIGHT_ONLY",
        "configuration": {
            "expert": args.expert,
            "execution_mode": smoke_contract["mode"],
            "preflight_contract_id": smoke_contract[
                "preflight_contract_id"
            ],
            "authorized_250k_contract_id": smoke_contract[
                "authorized_250k_contract_id"
            ],
            "qualification_use": "NO_PPO_PREFLIGHT_ONLY",
            "forward_iteration_v2": bool(args.forward_iteration_v2),
            "reverse_iteration_v2": bool(args.reverse_iteration_v2),
            "forward_iteration_v3_touchdown_balance": bool(
                args.forward_iteration_v3_touchdown_balance
            ),
            "reverse_iteration_v3_no_target_imitation": bool(
                args.reverse_iteration_v3_no_target_imitation
            ),
            "forward_iteration_v4_contact_event_validity_persistence": bool(
                args.forward_iteration_v4_contact_event_validity_persistence
            ),
            "reverse_iteration_v4_residual_transfer_gain_024": bool(
                args.reverse_iteration_v4_residual_transfer_gain_024
            ),
            "forward_v5_contact_pulse_abort_scale_only": bool(
                args.forward_v5_contact_pulse_abort_scale_only
            ),
            "reverse_iteration_v5_no_contact_imitation": bool(
                args.reverse_iteration_v5_no_contact_imitation
            ),
            "forward_iteration_v6_contact_abort_island_only": bool(
                args.forward_iteration_v6_contact_abort_island_only
            ),
            "reverse_iteration_v6_absolute_full_leg_targets": bool(
                args.reverse_iteration_v6_absolute_full_leg_targets
            ),
            "forward_v4_substep_contact": bool(
                smoke_contract["forward_v4_substep_contact"]
            ),
            "backward_residual_scale": smoke_contract[
                "backward_residual_scale"
            ],
            "platform": args.platform,
            "backend_resolution": backend_resolution,
            "xla_autotune_policy": xla_autotune_policy,
            "debug_callback_preflight": debug_callback_preflight,
            "state_device_audits": {
                "reset": reset_device_audit,
                "final": final_device_audit,
            },
            "physical_command_anchor": list(physical),
            "policy_observation_anchor": list(policy),
            "sampled_physical_command": sampled_physical_command.tolist(),
            "sampled_policy_observation_command": (
                mapped_policy_command.tolist()
            ),
            "steps": args.steps,
            "jitted_step": bool(args.jit),
            "reset_noise_multiplier": args.reset_noise_multiplier,
            "reverse_teacher_cycle_hz": (
                selected["cadence_hz"] if selected else 1.75
            ),
            "actual_reverse_phase_advance_bins_per_control": float(
                env._backward_phase_rate
                * env._h4_reverse_teacher_phase_scale
            ),
            "actor_observation_width": int(state_observation.shape[0]),
            "quality_reward_scales": reward_scales.as_reward_scale_dict(),
            "curriculum_contract": smoke_contract["anchors"],
            "legacy_reward_config_audit": legacy_reward_config_audit,
            "force_contact_source": "MJX decoded constraint force",
            "slip_source": "normal-force-weighted contact-point relative tangential velocity",
            "force_normalization": "current randomized robot body-mass sum times gravity norm",
            "selected_reverse_teacher": (
                {
                    "candidate_id": selected["candidate_id"],
                    "entry_phase_bins": selected["entry_phase_bins"],
                    "first_phase_bins": selected["first_phase_bins"],
                    "phase_advance_bins_per_control": selected[
                        "phase_advance_bins"
                    ],
                    "composition_status": (
                        "TRAINING_COMPOSITION_COMPONENT_NOT_ADOPTED"
                    ),
                }
                if selected
                else None
            ),
        },
        "inputs": {
            "legacy_trainer": {
                "path": str(LEGACY_TRAINER_PATH),
                "sha256": trainer.sha256_file(LEGACY_TRAINER_PATH),
            },
            "h4_runner_backend_contract": {
                "path": str(H4_RUNNER_PATH),
                "sha256": trainer.sha256_file(H4_RUNNER_PATH),
            },
            "alignment_module": {
                "path": str(
                    EXP_ROOT
                    / "safe_gait_experts"
                    / "h4_training_alignment.py"
                ),
                "sha256": trainer.sha256_file(
                    EXP_ROOT
                    / "safe_gait_experts"
                    / "h4_training_alignment.py"
                ),
            },
            "iteration_v2_authorization": (
                {
                    "path": str(smoke_contract["authorization"]["path"]),
                    "sha256": smoke_contract["authorization"]["sha256"],
                    "contract_id": smoke_contract["authorization"][
                        "contract_id"
                    ],
                    "semantic_audit": smoke_contract["authorization"][
                        "semantic_audit"
                    ],
                    "bound_causal_inputs": smoke_contract[
                        "authorization"
                    ]["bound_causal_inputs"],
                    "strict_gate_contract": smoke_contract[
                        "authorization"
                    ]["payload"]["strict_gate_contract"],
                    "manifest_binding": smoke_contract["authorization"][
                        "payload"
                    ]["manifest_binding"],
                }
                if smoke_contract["authorization"]
                and smoke_contract["mode"].endswith("iteration_v2")
                else None
            ),
            "minimum_spec": (
                {
                    "path": str(smoke_contract["minimum_spec"]["path"]),
                    "sha256": smoke_contract["minimum_spec"]["sha256"],
                }
                if smoke_contract["minimum_spec"]
                else None
            ),
            "reverse_composition_authorization": (
                {
                    "path": str(
                        smoke_contract["reverse_composition"]["path"]
                    ),
                    "sha256": smoke_contract["reverse_composition"][
                        "sha256"
                    ],
                }
                if smoke_contract["reverse_composition"]
                else None
            ),
            "scene": {
                "path": str(paths["scene"]),
                "sha256": trainer.sha256_file(paths["scene"]),
            },
            "selected_reverse_teacher": (
                {"path": str(selected["path"]), "sha256": selected["sha256"]}
                if selected
                else None
            ),
        },
        "reset_targets": reset_targets.tolist(),
        "trace": traces,
        "invariants": invariants,
    }
    if iteration_v3:
        result["configuration"].update(
            {
                "forward_iteration_v3_touchdown_balance": bool(
                    args.forward_iteration_v3_touchdown_balance
                ),
                "reverse_iteration_v3_no_target_imitation": bool(
                    args.reverse_iteration_v3_no_target_imitation
                ),
            }
        )
        result["inputs"]["iteration_v3_authorization"] = {
            "path": str(smoke_contract["authorization"]["path"]),
            "sha256": smoke_contract["authorization"]["sha256"],
            "contract_id": smoke_contract["authorization"]["contract_id"],
            "semantic_audit": smoke_contract["authorization"]["semantic_audit"],
            "bound_causal_inputs": smoke_contract["authorization"][
                "bound_causal_inputs"
            ],
            "strict_gate_contract": smoke_contract["authorization"]["payload"][
                "strict_gate_contract"
            ],
            "manifest_binding": smoke_contract["authorization"]["payload"][
                "manifest_binding"
            ],
        }
    if iteration_v4:
        result["inputs"]["iteration_v4_authorization"] = {
            "path": str(smoke_contract["authorization"]["path"]),
            "sha256": smoke_contract["authorization"]["sha256"],
            "contract_id": smoke_contract["authorization"]["contract_id"],
            "semantic_audit": smoke_contract["authorization"]["semantic_audit"],
            "bound_causal_inputs": smoke_contract["authorization"][
                "bound_causal_inputs"
            ],
            "bound_causal_sources": smoke_contract["authorization"][
                "bound_causal_sources"
            ],
            "strict_gate_contract": smoke_contract["authorization"]["payload"][
                "strict_gate_contract"
            ],
            "manifest_binding": smoke_contract["authorization"]["payload"][
                "manifest_binding"
            ],
        }
    if iteration_v5:
        result["inputs"]["iteration_v5_authorization"] = {
            "path": str(smoke_contract["authorization"]["path"]),
            "sha256": smoke_contract["authorization"]["sha256"],
            "contract_id": smoke_contract["authorization"]["contract_id"],
            "semantic_audit": smoke_contract["authorization"]["semantic_audit"],
            "bound_causal_inputs": smoke_contract["authorization"][
                "bound_causal_inputs"
            ],
            "bound_historical_v4_sources": smoke_contract["authorization"][
                "bound_historical_v4_sources"
            ],
            "strict_gate_contract": smoke_contract["authorization"]["payload"][
                "strict_gate_contract"
            ],
            "manifest_binding": smoke_contract["authorization"]["payload"][
                "manifest_binding"
            ],
        }
    if iteration_v6:
        result["iteration_v6_core_source"] = dict(
            smoke_contract["iteration_v6_core_source"]
        )
        result["configuration"]["iteration_v6_core_source"] = dict(
            smoke_contract["iteration_v6_core_source"]
        )
        result["inputs"]["iteration_v6_core_source"] = dict(
            smoke_contract["iteration_v6_core_source"]
        )
        result["inputs"]["iteration_v6_authorization"] = {
            "path": str(smoke_contract["authorization"]["path"]),
            "sha256": smoke_contract["authorization"]["sha256"],
            "contract_id": smoke_contract["authorization"]["contract_id"],
            "semantic_audit": smoke_contract["authorization"]["semantic_audit"],
            "bound_causal_inputs": smoke_contract["authorization"][
                "bound_causal_inputs"
            ],
            "bound_historical_v5_sources": smoke_contract["authorization"][
                "bound_historical_v5_sources"
            ],
            "strict_gate_contract": smoke_contract["authorization"]["payload"][
                "strict_gate_contract"
            ],
            "manifest_binding": smoke_contract["authorization"]["payload"][
                "manifest_binding"
            ],
        }
        if args.forward_iteration_v6_contact_abort_island_only:
            result["configuration"][
                "forward_iteration_v6_reward_routing_runtime"
            ] = iteration_v6_runtime
            result["configuration"]["reward_routing_contract"] = dict(
                smoke_contract["authorization"]["payload"][
                    "reward_routing_contract"
                ]
            )
            result["forward_iteration_v6_reward_routing_runtime"] = (
                iteration_v6_runtime
            )
            result["reward_routing_contract"] = dict(
                smoke_contract["authorization"]["payload"][
                    "reward_routing_contract"
                ]
            )
        else:
            result["configuration"][
                "reverse_iteration_v6_decoder_runtime"
            ] = iteration_v6_runtime
            result["configuration"][
                "reverse_iteration_v6_decoder_vector_runtime"
            ] = reverse_iteration_v6_decoder_vector_runtime
            result["configuration"]["action_parameterization_contract"] = dict(
                smoke_contract["authorization"]["payload"][
                    "action_parameterization_contract"
                ]
            )
            result["configuration"]["teacher_timing_contract"] = dict(
                smoke_contract["authorization"]["payload"][
                    "teacher_timing_contract"
                ]
            )
            result["reverse_iteration_v6_decoder_runtime"] = iteration_v6_runtime
            result["reverse_iteration_v6_decoder_vector_runtime"] = (
                reverse_iteration_v6_decoder_vector_runtime
            )
            result["action_parameterization_contract"] = dict(
                smoke_contract["authorization"]["payload"][
                    "action_parameterization_contract"
                ]
            )
            result["teacher_timing_contract"] = dict(
                smoke_contract["authorization"]["payload"][
                    "teacher_timing_contract"
                ]
            )
    if (
        args.forward_iteration_v4_contact_event_validity_persistence
        or args.forward_v5_contact_pulse_abort_scale_only
        or args.forward_iteration_v6_contact_abort_island_only
    ):
        result["forward_v4_source_semantic_preflight"] = (
            forward_v4_source_semantic_preflight
        )
        result["forward_v4_single_authority_runtime"] = (
            forward_v4_single_authority_runtime
        )
        result["configuration"]["forward_v4_source_semantic_preflight"] = (
            forward_v4_source_semantic_preflight
        )
        result["configuration"]["forward_v4_single_authority_runtime"] = (
            forward_v4_single_authority_runtime
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_smoke(args)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
