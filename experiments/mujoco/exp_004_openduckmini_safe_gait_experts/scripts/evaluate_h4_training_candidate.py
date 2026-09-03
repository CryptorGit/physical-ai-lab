"""Evaluate one trusted H4 actor116 checkpoint on the fixed strict seed set.

The evaluator is simulation-only and CPU-only.  It reconstructs the same H4
aligned environment used for training, calls Brax deterministic inference on
the complete 116-wide actor observation, and captures all 3,000 MJX physics
substeps plus the t=0 sample for current full-P0 gait-quality recomputation.

This command does not adopt, package, release, or deploy a policy.  A promotion
evidence file is emitted only when a separate full-P0 baseline artifact uses
the identical expert, seeds, duration, and central semantics hashes and the
candidate passes all three fixed seeds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.contract import (  # noqa: E402
    ACTUATOR_JOINT_ORDER,
    SAFE_INIT_POS,
)
from safe_gait_experts.gait_quality import (  # noqa: E402
    GaitQualityAccumulator,
    GaitQualitySubstep,
    gait_quality_acceptance,
)
from safe_gait_experts.h4_post_training import (  # noqa: E402
    H4_ACTION_WIDTH,
    H4_ACTOR_OBSERVATION_WIDTH,
    H4_CONTROL_DT_S,
    H4_CRITIC_OBSERVATION_WIDTH,
    H4_GAIT_SAMPLE_SOURCE,
    H4_PHYSICS_DT_S,
    H4_REVERSE_ACTION_DELAY_MAX_EXCLUSIVE,
    H4_REVERSE_ACTION_DELAY_MIN,
    H4_REVERSE_COMPOSITION_TRACE_SEMANTICS,
    H4_REVERSE_RESIDUAL_SCALE,
    H4_REVERSE_SOURCE_PERIOD_BINS,
    H4_REVERSE_TEACHER_ENTRY_PHASE_BINS,
    H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS,
    H4_REVERSE_TEACHER_TABLE_ROWS,
    H4_STRICT_COMMANDS,
    H4_STRICT_CONTROL_TICKS,
    H4_STRICT_DURATION_S,
    H4_STRICT_GAIT_SAMPLES,
    H4_STRICT_PHYSICS_SUBSTEPS,
    H4_STRICT_SEEDS,
    PINNED_V22_PARENT_TREE_SHA256,
    PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256,
    PINNED_SELECTED_REVERSE_TEACHER_SHA256,
    STRICT_ARTIFACT_KIND,
    audit_v22_to_h4_transplant,
    build_integrated_promotion_evidence,
    compare_policy_outputs,
    current_source_hashes,
    infer_h4_action_numpy,
    h4_params_numeric_sha256,
    json_native,
    legacy_metrics_from_gait_quality,
    load_json_strict,
    load_trusted_h4_params,
    mask_h4_head_action,
    reconstruct_h4_training_source_paths,
    rederive_central_safety_audit_from_control_trace,
    rederive_h4_control_contract,
    rederive_h4_safety_acceptance,
    sha256_file,
    sha256_tree,
    validate_h4_strict_artifact,
    validate_h4_training_source_closure,
    validate_trusted_h4_bundle,
    write_new_json,
)
from safe_gait_experts.h4_training_alignment import (  # noqa: E402
    H4QualityRewardScales,
    make_anchor_command_mapper,
    make_h4_aligned_environment_class,
    require_checkpoint_observation_compatibility,
)
from safe_gait_experts.routed_evaluation import (  # noqa: E402
    PhysicsSubstepAudit,
    SafetyAudit,
)


LEGACY_TRAINER_PATH = EXP_ROOT / "scripts" / "train_expert.py"
H4_ALIGNMENT_PATH = EXP_ROOT / "safe_gait_experts" / "h4_training_alignment.py"
H4_RUNNER_PATH = EXP_ROOT / "scripts" / "train_h4_aligned_expert.py"
H4_POST_TRAINING_PATH = EXP_ROOT / "safe_gait_experts" / "h4_post_training.py"
REVERSE_COMPOSITION_VALIDATOR_PATH = (
    EXP_ROOT / "scripts" / "validate_h4_reverse_training_composition.py"
)
TRUSTED_H4_RUN_ROOT = EXP_ROOT / "artifacts" / "h4_training_runs"
CENTRAL_PATHS = {
    "evaluator": EXP_ROOT / "scripts" / "evaluate_routed_transitions.py",
    "gait_quality": EXP_ROOT / "safe_gait_experts" / "gait_quality.py",
    "routed_evaluation": EXP_ROOT / "safe_gait_experts" / "routed_evaluation.py",
}
DEFAULT_SOURCE_ROOT = Path("/home/user/openduck_training_20260729")
DEFAULT_GENERATED_ROOT = EXP_ROOT / "artifacts" / "generated_playground"
DEFAULT_V22_PARENT_CHECKPOINT = Path(
    "/home/user/openduck_training_runs/"
    "calibrated_hybrid_yaw_cost_v22_300m/2026_07_29_154427_10485760"
)
_COMPILED_ROLLOUT_CACHE: dict[tuple[int, int], tuple[Any, Any]] = {}


def _load_legacy_trainer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "exp004_h4_post_training_legacy_trainer", LEGACY_TRAINER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy trainer: {LEGACY_TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_and_close_training_sources(
    *, bundle: Any, source_root: Path, generated_root: Path
) -> tuple[Any, Any, Any, dict[str, Path]]:
    """Load import definitions, reconstruct source closure, then close bundle.

    This function deliberately does not accept or open the params pickle.
    """

    trainer = _load_legacy_trainer()
    generated = trainer.generated_paths(generated_root.resolve())
    trainer._validate_generated_manifest(generated)
    stack = trainer._load_training_stack(source_root.resolve())

    class TeacherArgs:
        backward_gait = None
        backward_left_gait = None
        backward_right_gait = None

    legacy_teacher_gaits = trainer.resolve_teacher_gaits(TeacherArgs(), generated)
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    expected_paths = reconstruct_h4_training_source_paths(
        bundle=bundle,
        experiment_root=EXP_ROOT,
        legacy_trainer_path=LEGACY_TRAINER_PATH,
        alignment_path=H4_ALIGNMENT_PATH,
        runner_path=H4_RUNNER_PATH,
        reverse_composition_validator_path=REVERSE_COMPOSITION_VALIDATOR_PATH,
        stack=stack,
        ppo_checkpoint_path=Path(ppo_checkpoint.__file__).resolve(),
        generated_paths=generated,
        legacy_teacher_gaits=legacy_teacher_gaits,
    )
    return (
        trainer,
        stack,
        validate_h4_training_source_closure(bundle, expected_paths),
        expected_paths,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-P0 fixed-seed CPU evaluation for a trusted H4 actor116."
    )
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--params-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trusted-run-root",
        type=Path,
        default=TRUSTED_H4_RUN_ROOT,
        help="Exact local runner root containing <expert>/<run_name>.",
    )
    parser.add_argument("--promotion-evidence-output", type=Path)
    parser.add_argument(
        "--allow-wiring-diagnostic",
        action="store_true",
        help="Evaluate an exact 40-interaction WIRING_PASS; never promotable.",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument(
        "--v22-parent-checkpoint",
        type=Path,
        default=DEFAULT_V22_PARENT_CHECKPOINT,
        help="Pinned official v22 tree used for the integrated zero-update baseline.",
    )
    parser.add_argument(
        "--platform",
        choices=("cpu",),
        default="cpu",
        help="Formal H4 evaluation is CPU-only.",
    )
    return parser


def _resolve_process_start_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Freeze every CLI path before legacy imports can change the cwd."""

    for name in (
        "params",
        "manifest",
        "output",
        "trusted_run_root",
        "promotion_evidence_output",
        "source_root",
        "generated_root",
        "v22_parent_checkpoint",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).resolve())
    return args


def _require_file_binding(record: Mapping[str, Any], expected_sha: str) -> Path:
    path = Path(record.get("path", "")).resolve()
    if not path.is_file() or record.get("sha256") != expected_sha:
        raise ValueError("composition component config binding is incomplete")
    if sha256_file(path) != expected_sha:
        raise ValueError("composition component file SHA256 mismatch")
    return path


def _load_reverse_teacher(bundle: Any) -> dict[str, Any]:
    selected_record = bundle.config["selected_reverse_teacher"]
    selected_path = _require_file_binding(
        selected_record, PINNED_SELECTED_REVERSE_TEACHER_SHA256
    )
    authorization_record = bundle.config["reverse_composition_authorization"]
    authorization_path = _require_file_binding(
        authorization_record, PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
    )
    selected = load_json_strict(selected_path)
    authorization = load_json_strict(authorization_path)
    teacher = selected.get("teacher", {})
    adapter = selected.get("adapter_contract", {})
    composition = authorization.get("composition_contract", {})
    permission = authorization.get("authorization", {})
    checks = {
        "selected_teacher_validated": teacher.get("validation", {}).get("passed")
        is True,
        "authorization_status": authorization.get("status")
        == "SIMULATION_TRAINING_COMPOSITION_AUTHORIZED_NOT_ADOPTED",
        "candidate_evaluation_authorized": permission.get(
            "candidate_strict_evaluation"
        )
        is True,
        "candidate_adoption_prohibited": permission.get("candidate_adoption")
        is False,
        "release_prohibited": permission.get("package_release") is False,
        "hardware_prohibited": permission.get("hardware") is False,
        "persistent_composition": composition.get("training_use")
        == "PERSISTENT_DETERMINISTIC_BASELINE_PLUS_TRAINABLE_RESIDUAL",
        "standalone_teacher_prohibited": composition.get(
            "standalone_teacher_direct_runtime_use"
        )
        == "PROHIBITED",
        "phase_steps": composition.get("teacher_phase_steps") == 54,
        "cadence": composition.get("teacher_cadence_hz") == 1.5,
        "phase_advance": composition.get(
            "teacher_phase_advance_bins_per_control"
        )
        == 1.62,
        "entry_phase": composition.get("teacher_entry_phase_preincrement_bins")
        == 14.0,
        "residual_scale": composition.get("maximum_residual_scale") == 0.12,
    }
    if not all(checks.values()):
        raise ValueError(f"reverse evaluation composition contract failed: {checks}")
    table = np.asarray(teacher.get("target_table_rad"), dtype=np.float64)
    if table.shape != (54, H4_ACTION_WIDTH) or not np.all(np.isfinite(table)):
        raise ValueError("selected reverse teacher table must be finite 54x14")
    if not np.array_equal(table[:, 5:9], np.zeros((54, 4))):
        raise ValueError("selected reverse teacher head targets must be exact zero")
    return {
        "selected_path": selected_path,
        "authorization_path": authorization_path,
        "table": table,
        "cadence_hz": float(adapter["cadence_hz"]),
        "phase_advance_bins": float(adapter["phase_advance_bins_per_control"]),
        "entry_phase_bins": float(adapter["entry_phase_preincrement_bins"]),
        "checks": checks,
    }


def _reward_scales_from_config(config: Mapping[str, Any]) -> H4QualityRewardScales:
    values = config.get("reward_scales")
    if not isinstance(values, Mapping) or not values:
        raise ValueError("H4 resolved config reward scales are missing")
    kwargs = {
        str(name)[3:]: float(value)
        for name, value in values.items()
        if isinstance(name, str) and name.startswith("h4_")
    }
    scales = H4QualityRewardScales(**kwargs)
    if scales.as_reward_scale_dict() != dict(values):
        raise ValueError("H4 reward scale reconstruction drifted")
    return scales


def _build_brax_policy(*, stack: Mapping[str, Any], bundle: Any, params: Any) -> Any:
    from brax.training.acme import running_statistics

    network = stack["ppo_networks"].make_ppo_networks(
        {
            "state": (H4_ACTOR_OBSERVATION_WIDTH,),
            "privileged_state": (H4_CRITIC_OBSERVATION_WIDTH,),
        },
        H4_ACTION_WIDTH,
        preprocess_observations_fn=running_statistics.normalize,
        **dict(bundle.config["network_factory"]),
    )
    return stack["ppo_networks"].make_inference_fn(network)(
        params, deterministic=True
    )


def _load_official_v22_baseline(
    *, checkpoint_path: Path, stack: Mapping[str, Any]
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load the pinned v22 tree and perform zero optimizer updates."""

    resolved = Path(checkpoint_path).resolve()
    tree_pre = sha256_tree(resolved)
    if tree_pre != PINNED_V22_PARENT_TREE_SHA256:
        raise ValueError(
            "official v22 baseline checkpoint tree SHA256 drifted: "
            f"{tree_pre}"
        )
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    source_params = ppo_checkpoint.load(str(resolved))
    transplanted, alignment_audit = require_checkpoint_observation_compatibility(
        source_params,
        actor_observation_width=H4_ACTOR_OBSERVATION_WIDTH,
        allow_explicit_v22_transplant=True,
        xp=stack["jp"],
    )
    independent_audit = audit_v22_to_h4_transplant(source_params, transplanted)
    if (
        alignment_audit.get("passed") is not True
        or alignment_audit.get("transplant_applied") is not True
    ):
        raise RuntimeError("official v22 alignment transplant audit failed")
    tree_post = sha256_tree(resolved)
    if tree_post != tree_pre:
        raise RuntimeError("official v22 checkpoint tree changed during baseline load")
    source = {
        "kind": "OFFICIAL_FROZEN_V22_BRAX_CHECKPOINT",
        "path": str(resolved),
        "sha256_tree_pre": tree_pre,
        "sha256_tree_post": tree_post,
        "unchanged": True,
    }
    transplant_audit = {
        **independent_audit,
        "alignment_contract_audit": dict(alignment_audit),
    }
    return transplanted, source, transplant_audit


def _yaw_from_wxyz(quaternion: Any, *, xp: Any) -> Any:
    w, x, y, z = quaternion
    return xp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _make_environment_and_policy(
    *,
    bundle: Any,
    params: Any,
    source_root: Path,
    generated_root: Path,
) -> tuple[Any, Any, Any, Any, dict[str, Any], dict[str, Path]]:
    trainer = _load_legacy_trainer()
    paths = trainer.generated_paths(generated_root.resolve())
    trainer._validate_generated_manifest(paths)
    stack = trainer._load_training_stack(source_root.resolve())
    jax = stack["jax"]
    jp = stack["jp"]
    if jax.default_backend() != "cpu" or any(
        device.platform != "cpu" for device in jax.devices()
    ):
        raise RuntimeError("formal H4 evaluation requires CPU-only JAX devices")

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
    composition = _load_reverse_teacher(bundle) if bundle.expert == "reverse" else None
    anchors = bundle.config.get("anchor_config", {})
    physical = tuple(float(value) for value in anchors.get("physical_primary", ()))
    policy_anchor = tuple(
        float(value) for value in anchors.get("policy_observation_anchor", ())
    )
    if physical != H4_STRICT_COMMANDS[bundle.expert] or len(policy_anchor) != 3:
        raise ValueError("H4 strict physical/policy anchor config drifted")

    LegacyEnvironment = trainer._make_environment_class(
        stack=stack,
        expert=bundle.expert,
        paths=paths,
        teacher_gaits=teacher_gaits,
        backward_residual_scale=float(bundle.config["backward_residual_scale"]),
    )

    def fixed_physical_sampler(_rng: Any) -> Any:
        return jp.asarray((*physical, 0.0, 0.0, 0.0, 0.0))

    mapper = make_anchor_command_mapper(physical, policy_anchor, xp=jp)
    Environment = make_h4_aligned_environment_class(
        legacy_environment_class=LegacyEnvironment,
        stack=stack,
        physical_command_sampler=fixed_physical_sampler,
        policy_observation_mapper=mapper,
        reward_scales=_reward_scales_from_config(bundle.config),
        reset_noise_multiplier=0.0,
        reverse_teacher_cycle_hz=(
            composition["cadence_hz"] if composition is not None else 1.75
        ),
        reverse_teacher_target_table=(
            composition["table"] if composition is not None else None
        ),
        reverse_teacher_phase_advance_bins=(
            composition["phase_advance_bins"] if composition is not None else None
        ),
        reverse_teacher_entry_phase_bins=(
            composition["entry_phase_bins"] if composition is not None else 0.0
        ),
        include_h4_actor_observables=True,
    )
    env = Environment()
    probe_state = env.reset(jax.random.PRNGKey(H4_STRICT_SEEDS[bundle.expert][0]))
    if (
        probe_state.obs["state"].shape != (H4_ACTOR_OBSERVATION_WIDTH,)
        or probe_state.obs["privileged_state"].shape
        != (H4_CRITIC_OBSERVATION_WIDTH,)
        or env.n_substeps != 10
        or abs(float(env.dt) - H4_CONTROL_DT_S) > 1.0e-12
        or abs(float(env._sim_dt) - H4_PHYSICS_DT_S) > 1.0e-12
    ):
        raise RuntimeError("H4 aligned environment width/timestep contract drifted")
    if composition is not None:
        reverse_runtime_checks = {
            "runtime_teacher_rows": (
                env._h4_reverse_teacher_table.shape
                == (H4_REVERSE_TEACHER_TABLE_ROWS, H4_ACTION_WIDTH)
            ),
            "runtime_source_period": (
                int(env.PRM.nb_steps_in_period)
                == H4_REVERSE_SOURCE_PERIOD_BINS
            ),
            "runtime_residual_scale": (
                float(env._backward_residual_scale)
                == H4_REVERSE_RESIDUAL_SCALE
            ),
            "runtime_action_delay_exact_zero": (
                int(env._config.noise_config.action_min_delay)
                == H4_REVERSE_ACTION_DELAY_MIN
                and int(env._config.noise_config.action_max_delay)
                == H4_REVERSE_ACTION_DELAY_MAX_EXCLUSIVE
            ),
            "runtime_phase_entry": (
                composition["entry_phase_bins"]
                == H4_REVERSE_TEACHER_ENTRY_PHASE_BINS
            ),
            "runtime_phase_advance": (
                composition["phase_advance_bins"]
                == H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS
            ),
        }
        if not all(reverse_runtime_checks.values()):
            raise RuntimeError(
                "H4 reverse runtime composition drifted: "
                f"{reverse_runtime_checks}"
            )
        composition["checks"] = {
            **composition["checks"],
            **reverse_runtime_checks,
        }

    policy = _build_brax_policy(stack=stack, bundle=bundle, params=params)
    source_paths = {
        "source_joystick": Path(stack["joystick"].__file__).resolve(),
        "source_constants": Path(stack["constants"].__file__).resolve(),
        "generated_manifest": paths["manifest"],
        "generated_scene": paths["scene"],
        "generated_reference": paths["reference"],
    }
    for key in ("backward", "backward_left", "backward_right"):
        source_paths[f"legacy_{key}_teacher"] = teacher_gaits[key]
    if composition is not None:
        source_paths["selected_reverse_teacher"] = composition["selected_path"]
        source_paths["reverse_composition_authorization"] = composition[
            "authorization_path"
        ]
    return env, policy, stack, trainer, composition or {}, source_paths


def _snapshot_function(env: Any, stack: Mapping[str, Any]) -> Any:
    jp = stack["jp"]
    joint_addresses = jp.asarray(env.get_actuator_joints_qpos_addr())
    feet_geom_ids = jp.asarray(env._feet_geom_id)
    torso_id = int(env._torso_body_id)

    def snapshot(data: Any, previous_contact: Any) -> tuple[Any, dict[str, Any]]:
        quality = env._h4_contact_observables(data, previous_contact)
        floating = env.get_floating_base_qpos(data.qpos)
        return quality.contact, {
            "joint_qpos": data.qpos[joint_addresses],
            "full_qpos": data.qpos,
            "full_qvel": data.qvel,
            "height": data.xpos[torso_id, 2],
            "upright": data.xmat[torso_id, 2, 2],
            "trunk_position": data.xpos[torso_id],
            "trunk_yaw": _yaw_from_wxyz(floating[3:7], xp=jp),
            "local_velocity": env.get_local_linvel(data),
            "local_yaw_rate": env.get_gyro(data)[2],
            "contacts": quality.contact,
            "normal_force": quality.normalized_force,
            "tangential_speed": quality.tangential_speed_m_s,
            "foot_points": data.geom_xpos[feet_geom_ids],
        }

    return snapshot


def _as_host_trace(jax: Any, trace: Mapping[str, Any]) -> dict[str, np.ndarray]:
    host = jax.device_get(trace)
    result = {name: np.asarray(value) for name, value in host.items()}
    lengths = {value.shape[0] for value in result.values()}
    if lengths != {10}:
        raise RuntimeError(f"one H4 control must expose exactly 10 substeps: {lengths}")
    return result


def _compiled_rollout_for(
    env: Any, policy: Any, stack: Mapping[str, Any]
) -> tuple[Any, Any]:
    """Return one cached 300x10 rollout executable per environment/policy."""

    key = (id(env), id(policy))
    cached = _COMPILED_ROLLOUT_CACHE.get(key)
    if cached is not None:
        return cached
    jax = stack["jax"]
    jp = stack["jp"]
    joystick = stack["joystick"]
    mjx = joystick.mjx
    snapshot = _snapshot_function(env, stack)
    original_physics_step = joystick.mjx_env.step
    joint_addresses = jp.asarray(env.get_actuator_joints_qpos_addr())
    reverse_composition = env._h4_reverse_teacher_table is not None

    def control_step(carry: tuple[Any, Any], _control_index: Any) -> tuple[Any, Any]:
        current_state, inference_key = carry
        actor_observation = current_state.obs["state"]
        previous_targets = current_state.data.ctrl
        guard_before = current_state.info["h4_guard_steps"]
        previous_contact = current_state.info["h4_previous_force_contact"]
        inference_key, action_key = jax.random.split(inference_key)
        raw_action, _extras = policy(current_state.obs, action_key)
        applied_action = raw_action.at[5:9].set(0.0)
        reverse_trace: dict[str, Any] = {}
        if reverse_composition:
            source_phase_before = current_state.info["imitation_i"]
            source_phase_after = jp.mod(
                source_phase_before + env._backward_phase_rate,
                env.PRM.nb_steps_in_period,
            )
            table_phase = jp.mod(
                source_phase_after * env._h4_reverse_teacher_phase_scale,
                env._h4_reverse_teacher_table.shape[0],
            )
            teacher_target = env._h4_selected_teacher_actuator_target(
                table_phase
            )
            action_history = (
                jp.roll(current_state.info["action_history"], env._actuators)
                .at[: env._actuators]
                .set(applied_action)
            )
            _next_rng, _push1_rng, _push2_rng, action_delay_rng = (
                jax.random.split(current_state.info["rng"], 4)
            )
            action_delay_index = jax.random.randint(
                action_delay_rng,
                (1,),
                minval=env._config.noise_config.action_min_delay,
                maxval=env._config.noise_config.action_max_delay,
            )[0]
            delayed_action = action_history.reshape(
                (-1, env._actuators)
            )[action_delay_index]
            reverse_trace = {
                "reverse_teacher_source_phase_before": source_phase_before,
                "reverse_teacher_table_phase": table_phase,
                "reverse_teacher_table_targets": teacher_target,
                "reverse_action_delay_index": action_delay_index,
                "reverse_delayed_applied_action": delayed_action,
                "reverse_precomposer_active": (
                    current_state.info["command"][0] < -0.02
                ),
            }
        trace_box: list[Mapping[str, Any]] = []

        def traced_physics_step(
            model: Any, data: Any, applied_targets: Any, n_substeps: int
        ) -> Any:
            if int(n_substeps) != 10:
                raise RuntimeError("H4 traced physics requires exactly 10 substeps")

            def single_step(
                physics_carry: tuple[Any, Any], _unused: Any
            ) -> tuple[Any, Any]:
                current_data, contact_carry = physics_carry
                current_data = current_data.replace(ctrl=applied_targets)
                current_data = mjx.step(model, current_data)
                next_contact, row = snapshot(current_data, contact_carry)
                return (current_data, next_contact), row

            (final_data, _final_contact), physics_trace = jax.lax.scan(
                single_step,
                (data, previous_contact),
                (),
                int(n_substeps),
            )
            trace_box.append(physics_trace)
            return final_data

        joystick.mjx_env.step = traced_physics_step
        next_state = env.step(current_state, raw_action)
        joystick.mjx_env.step = original_physics_step
        if len(trace_box) != 1:
            raise RuntimeError("H4 environment must expose one physics call per tick")
        control_trace = {
            "actor_observation": actor_observation,
            "raw_action": raw_action,
            "applied_action": applied_action,
            "preclip_targets": next_state.info["h4_pre_guard_raw_targets"],
            "margin_clipped_targets": next_state.info["h4_guard_desired_targets"],
            "applied_targets": next_state.data.ctrl,
            "previous_targets": previous_targets,
            "joint_qpos": next_state.data.qpos[joint_addresses],
            "guard_calls": next_state.info["h4_guard_steps"] - guard_before,
            "done": next_state.done,
            **reverse_trace,
        }
        if reverse_composition:
            control_trace["reverse_upstream_margin_targets"] = (
                next_state.info["h4_upstream_margin_targets"]
            )
        return (next_state, inference_key), {
            "physics": trace_box[0],
            "control": control_trace,
        }

    def complete_rollout(initial_state: Any, inference_key: Any) -> tuple[Any, Any]:
        return jax.lax.scan(
            control_step,
            (initial_state, inference_key),
            jp.arange(H4_STRICT_CONTROL_TICKS),
        )

    result = (snapshot, jax.jit(complete_rollout))
    _COMPILED_ROLLOUT_CACHE[key] = result
    return result


def _gait_sample(
    row: Mapping[str, Any],
    *,
    time_s: float,
    command: Sequence[float],
) -> GaitQualitySubstep:
    return GaitQualitySubstep(
        time_s=float(time_s),
        requested_command=command,
        effective_command=command,
        local_velocity_xyz_mps=np.asarray(row["local_velocity"], dtype=np.float64),
        local_yaw_rate_radps=float(row["local_yaw_rate"]),
        trunk_position_world_m=np.asarray(row["trunk_position"], dtype=np.float64),
        feet_contacts=np.asarray(row["contacts"], dtype=bool),
        foot_contact_points_world_m=np.asarray(row["foot_points"], dtype=np.float64),
        leg_joint_positions_rad=np.asarray(row["joint_qpos"], dtype=np.float64),
        feet_normal_force_fraction_body_weight=np.asarray(
            row["normal_force"], dtype=np.float64
        ),
        foot_contact_tangential_speeds_mps=np.asarray(
            row["tangential_speed"], dtype=np.float64
        ),
        trunk_yaw_world_rad=float(row["trunk_yaw"]),
        trunk_pose_measurement_source=H4_GAIT_SAMPLE_SOURCE,
    )


def _run_episode(
    *,
    env: Any,
    policy: Any,
    params: Any,
    stack: Mapping[str, Any],
    seed: int,
    expert: str,
) -> dict[str, Any]:
    jax = stack["jax"]
    jp = stack["jp"]
    joystick = stack["joystick"]
    mjx = joystick.mjx
    command = H4_STRICT_COMMANDS[expert]
    state = env.reset(jax.random.PRNGKey(seed))
    state.reward.block_until_ready()
    observation = np.asarray(state.obs["state"], dtype=np.float32)
    if observation.shape != (H4_ACTOR_OBSERVATION_WIDTH,) or not np.all(
        np.isfinite(observation)
    ):
        raise RuntimeError("H4 reset actor observation is invalid")
    initial_targets_source = np.asarray(state.data.ctrl)
    initial_targets = initial_targets_source.astype(np.float64)
    safe_init = np.asarray(
        [SAFE_INIT_POS[name] for name in ACTUATOR_JOINT_ORDER], dtype=np.float64
    )
    expected_source = safe_init.astype(initial_targets_source.dtype)
    reset_error = np.abs(
        initial_targets_source.astype(np.float64)
        - expected_source.astype(np.float64)
    )
    reset_audit = {
        "comparison_semantics": "SOURCE_DTYPE_FLOAT32_EXACT",
        "exact_safe_init": bool(np.array_equal(initial_targets_source, expected_source)),
        "maximum_safe_init_error_rad": float(np.max(reset_error)),
        "head_qpos_peak_rad": float(
            np.max(np.abs(initial_targets_source[5:9]))
        ),
    }
    control_audit = SafetyAudit(ACTUATOR_JOINT_ORDER)
    physics_audit = PhysicsSubstepAudit(ACTUATOR_JOINT_ORDER)
    gait = GaitQualityAccumulator(joint_names=ACTUATOR_JOINT_ORDER)
    snapshot, compiled_rollout = _compiled_rollout_for(env, policy, stack)
    _initial_force_contact, initial_snapshot = snapshot(
        state.data, jp.zeros(2, dtype=bool)
    )
    initial_host = {
        name: np.asarray(value) for name, value in jax.device_get(initial_snapshot).items()
    }
    gait.update(_gait_sample(initial_host, time_s=0.0, command=command))

    # Execute the cached complete 300-control rollout.  Keeping the Python
    # loop outside JIT would recompile the nested substep scan on every tick.
    original_physics_step = joystick.mjx_env.step
    try:
        (state, _final_key), device_trace = compiled_rollout(
            state, jax.random.PRNGKey(seed ^ 0x4844)
        )
        state.reward.block_until_ready()
    finally:
        joystick.mjx_env.step = original_physics_step
    host_trace = jax.device_get(device_trace)
    physics_trace = {
        name: np.asarray(value).reshape(
            (H4_STRICT_PHYSICS_SUBSTEPS,) + np.asarray(value).shape[2:]
        )
        for name, value in host_trace["physics"].items()
    }
    control_trace = {
        name: np.asarray(value) for name, value in host_trace["control"].items()
    }
    if any(value.shape[0] != H4_STRICT_PHYSICS_SUBSTEPS for value in physics_trace.values()):
        raise RuntimeError("H4 rollout did not expose exactly 3,000 physics rows")
    if any(value.shape[0] != H4_STRICT_CONTROL_TICKS for value in control_trace.values()):
        raise RuntimeError("H4 rollout did not expose exactly 300 control rows")

    completed_substeps = 0
    fell = False
    for substep_index in range(H4_STRICT_PHYSICS_SUBSTEPS):
        row = {name: value[substep_index] for name, value in physics_trace.items()}
        completed_substeps += 1
        physics_audit.update(
            joint_qpos=row["joint_qpos"],
            full_qpos=row["full_qpos"],
            full_qvel=row["full_qvel"],
            height_m=float(row["height"]),
            upright=float(row["upright"]),
            feet_contacts=row["contacts"],
        )
        gait.update(
            _gait_sample(
                row,
                time_s=completed_substeps * H4_PHYSICS_DT_S,
                command=command,
            )
        )
        fell = bool(fell or physics_audit.termination_required)

    for control_index in range(H4_STRICT_CONTROL_TICKS):
        control_audit.update(
            raw_policy_action=control_trace["raw_action"][control_index],
            applied_action=control_trace["applied_action"][control_index],
            preclip_targets=control_trace["preclip_targets"][control_index],
            margin_clipped_targets=control_trace["margin_clipped_targets"][
                control_index
            ],
            applied_targets=control_trace["applied_targets"][control_index],
            previous_applied_targets=control_trace["previous_targets"][control_index],
            joint_qpos=control_trace["joint_qpos"][control_index],
            control_dt=H4_CONTROL_DT_S,
        )
    actor_observations = control_trace["actor_observation"]
    raw_actions = control_trace["raw_action"]
    applied_actions = control_trace["applied_action"]
    guard_calls = control_trace["guard_calls"]
    nonfinite_observation_count = int(
        np.count_nonzero(~np.all(np.isfinite(actor_observations), axis=1))
    )
    nonfinite_action_count = int(
        np.count_nonzero(~np.all(np.isfinite(raw_actions), axis=1))
    )
    if (
        actor_observations.shape != (H4_STRICT_CONTROL_TICKS, H4_ACTOR_OBSERVATION_WIDTH)
        or raw_actions.shape != (H4_STRICT_CONTROL_TICKS, H4_ACTION_WIDTH)
        or nonfinite_observation_count
        or nonfinite_action_count
    ):
        raise RuntimeError("H4 rollout actor observations/actions are invalid")
    parity = compare_policy_outputs(
        infer_h4_action_numpy(params, actor_observations[0]), raw_actions[0]
    )
    if not parity["passed"]:
        raise RuntimeError(f"NumPy/Brax actor parity failed: {parity}")
    numpy_brax_parity = [parity]
    post_mask_nonzero_head_count = int(
        np.count_nonzero(applied_actions[:, 5:9])
    )
    raw_action_peak = float(np.max(np.abs(raw_actions)))
    guard_violations = int(np.count_nonzero(guard_calls != 1))
    maximum_guard_calls = int(np.max(guard_calls))
    fell = bool(fell or np.any(control_trace["done"]))

    if completed_substeps != H4_STRICT_PHYSICS_SUBSTEPS:
        raise RuntimeError("H4 strict episode did not complete exactly 3,000 substeps")
    gait_metrics = gait.finalize()
    # The central serializer deliberately carries one non-dataclass sentinel;
    # rederive_gait_quality_acceptance rejects even an otherwise complete field
    # set unless this exact measurement-complete assertion is present.
    gait_payload = {**gait_metrics.as_dict(), "measurement_complete": True}
    gait_result = gait_quality_acceptance(gait_metrics).as_dict()
    safety_payload = control_audit.to_dict()
    physics_payload = physics_audit.to_dict()
    guard_payload = {
        "control_tick_count": H4_STRICT_CONTROL_TICKS,
        "total_guard_calls": int(np.sum(guard_calls)),
        "guard_call_violation_count": guard_violations,
        "maximum_guard_calls_per_tick": maximum_guard_calls,
    }
    inference_payload = {
        "input_width": H4_ACTOR_OBSERVATION_WIDTH,
        "output_width": H4_ACTION_WIDTH,
        "inference_count": H4_STRICT_CONTROL_TICKS,
        "nonfinite_observation_count": nonfinite_observation_count,
        "nonfinite_action_count": nonfinite_action_count,
        "post_mask_nonzero_head_count": post_mask_nonzero_head_count,
        "maximum_raw_action_magnitude": raw_action_peak,
        "first_tick_numpy_brax_parity": numpy_brax_parity,
    }
    serialized_control_trace = {
        "source_dtype": str(raw_actions.dtype),
        "initial_applied_targets": initial_targets_source,
        "raw_action": control_trace["raw_action"],
        "applied_action": control_trace["applied_action"],
        "preclip_targets": control_trace["preclip_targets"],
        "margin_clipped_targets": control_trace["margin_clipped_targets"],
        "applied_targets": control_trace["applied_targets"],
        "previous_targets": control_trace["previous_targets"],
        "joint_qpos": control_trace["joint_qpos"],
    }
    reverse_composition_contract: dict[str, Any] | None = None
    if expert == "reverse":
        for name in (
            "reverse_teacher_source_phase_before",
            "reverse_teacher_table_phase",
            "reverse_teacher_table_targets",
            "reverse_action_delay_index",
            "reverse_delayed_applied_action",
            "reverse_upstream_margin_targets",
            "reverse_precomposer_active",
        ):
            serialized_control_trace[name] = control_trace[name]
        reverse_composition_contract = {
            "schema_version": 1,
            "semantics": H4_REVERSE_COMPOSITION_TRACE_SEMANTICS,
            "selected_reverse_teacher_sha256": (
                PINNED_SELECTED_REVERSE_TEACHER_SHA256
            ),
            "reverse_composition_authorization_sha256": (
                PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
            ),
            "teacher_table_rows": int(
                env._h4_reverse_teacher_table.shape[0]
            ),
            "teacher_entry_phase_preincrement_bins": float(
                env._h4_reverse_teacher_entry_source_phase
                * env._h4_reverse_teacher_phase_scale
            ),
            "teacher_phase_advance_bins_per_control": float(
                env._backward_phase_rate
                * env._h4_reverse_teacher_phase_scale
            ),
            "source_period_bins": int(env.PRM.nb_steps_in_period),
            "residual_scale": float(env._backward_residual_scale),
            "action_delay_min": int(
                env._config.noise_config.action_min_delay
            ),
            "action_delay_max_exclusive": int(
                env._config.noise_config.action_max_delay
            ),
            "step_entry_physical_command_x_mps": float(command[0]),
        }
    episode: dict[str, Any] = {
        "seed": seed,
        "segment_id": f"h4_{expert}_seed{seed}_6s",
        "expert": expert,
        "physical_command_mps_radps": list(command),
        "source_segment_kind": "H4_STRICT_6S",
        "completed": completed_substeps == H4_STRICT_PHYSICS_SUBSTEPS,
        "fell": bool(fell),
        "duration_s": completed_substeps * H4_PHYSICS_DT_S,
        "physics_timestep_s": H4_PHYSICS_DT_S,
        "completed_control_ticks": H4_STRICT_CONTROL_TICKS,
        "completed_physics_substeps": completed_substeps,
        "reset_audit": reset_audit,
        "control_trace": serialized_control_trace,
        "safety_audit": safety_payload,
        "physics_substep_audit": physics_payload,
        "guard_call_audit": guard_payload,
        "policy_inference_audit": inference_payload,
        "gait_quality_metrics": gait_payload,
        "gait_quality_acceptance": gait_result,
        "metrics": legacy_metrics_from_gait_quality(gait_payload),
    }
    if reverse_composition_contract is not None:
        episode["reverse_composition_contract"] = reverse_composition_contract
    episode["h4_control_contract"] = rederive_h4_control_contract(episode)
    if rederive_central_safety_audit_from_control_trace(episode) != safety_payload:
        raise RuntimeError("central SafetyAudit control-trace rederivation drifted")
    episode["h4_safety_acceptance"] = rederive_h4_safety_acceptance(episode)
    episode["safety"] = {
        "fall_count": int(fell),
        "qpos_violation_samples": int(
            safety_payload["qpos_limit_violations"]
            + physics_payload["qpos_limit_violations"]
        ),
        "target_violation_samples": int(
            safety_payload["applied_target_limit_violations"]
            + safety_payload["desired_target_margin_violations"]
        ),
        "slew_violation_samples": int(safety_payload["target_slew_violations"]),
        "guard_call_violation_samples": guard_violations,
        "nonfinite_samples": int(
            safety_payload["nonfinite_sample_count"]
            + physics_payload["nonfinite_state_samples"]
        ),
    }
    episode["strict_passed"] = bool(
        episode["h4_safety_acceptance"]["passed"] and gait_result["passed"]
    )
    return episode


def run_evaluation(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any, dict[str, str], dict[str, str]]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["JAX_PLATFORMS"] = "cpu"
    bundle = validate_trusted_h4_bundle(
        params_path=args.params,
        manifest_path=args.manifest,
        expected_params_sha256=args.params_sha256,
        expected_manifest_sha256=args.manifest_sha256,
        trusted_run_root=args.trusted_run_root,
        allow_wiring_diagnostic=args.allow_wiring_diagnostic,
    )
    if bundle.status == "WIRING_PASS" and args.promotion_evidence_output is not None:
        raise ValueError("WIRING_PASS evaluation can never produce promotion evidence")
    central_pre = current_source_hashes(CENTRAL_PATHS, root=EXP_ROOT)
    source_paths = {
        "h4_evaluator": Path(__file__).resolve(),
        "h4_post_training": H4_POST_TRAINING_PATH,
        "h4_alignment": H4_ALIGNMENT_PATH,
        "h4_runner": H4_RUNNER_PATH,
        "legacy_trainer": LEGACY_TRAINER_PATH,
        "candidate_params": bundle.params_path,
        "candidate_manifest": bundle.manifest_path,
        "candidate_config": bundle.config_path,
        "candidate_result": Path(
            bundle.manifest["outputs"]["result"]["path"]
        ).resolve(),
        "candidate_training_curve": Path(
            bundle.manifest["outputs"]["training_curve"]["path"]
        ).resolve(),
        **CENTRAL_PATHS,
    }
    # Import the frozen source stack before unpickling so Brax/Flax checkpoint
    # classes are available.  Exact label/path/SHA source closure is completed
    # before the params pickle is opened.
    _trainer, _stack, bundle, training_sources = _prepare_and_close_training_sources(
        bundle=bundle,
        source_root=args.source_root,
        generated_root=args.generated_root,
    )
    present_paths = {Path(path).resolve() for path in source_paths.values()}
    for label, path in training_sources.items():
        resolved = Path(path).resolve()
        if resolved not in present_paths:
            source_paths[f"training_{label}"] = resolved
            present_paths.add(resolved)
    source_pre = current_source_hashes(source_paths, root=EXP_ROOT)
    params, restore_audit = load_trusted_h4_params(bundle)
    env, policy, stack, _trainer, composition, environment_sources = (
        _make_environment_and_policy(
            bundle=bundle,
            params=params,
            source_root=args.source_root,
            generated_root=args.generated_root,
        )
    )
    present_paths = {Path(path).resolve() for path in source_paths.values()}
    for label, path in environment_sources.items():
        resolved = Path(path).resolve()
        if resolved not in present_paths:
            source_paths[f"environment_{label}"] = resolved
            present_paths.add(resolved)
    source_pre = current_source_hashes(source_paths, root=EXP_ROOT)
    jax = stack["jax"]

    # Independent deterministic parity probes bind actor116 normalization and
    # every MLP layer before the expensive simulation episodes begin.
    probe_rng = np.random.default_rng(20_260_809)
    probe_observations = np.stack(
        (
            np.zeros(H4_ACTOR_OBSERVATION_WIDTH, dtype=np.float32),
            np.linspace(-1.0, 1.0, H4_ACTOR_OBSERVATION_WIDTH, dtype=np.float32),
            probe_rng.normal(size=H4_ACTOR_OBSERVATION_WIDTH).astype(np.float32),
        )
    )
    brax_probe_actions = []
    for index, observation in enumerate(probe_observations):
        actor_observation = {
            "state": stack["jp"].asarray(observation),
            "privileged_state": stack["jp"].zeros(H4_CRITIC_OBSERVATION_WIDTH),
        }
        action, _ = policy(actor_observation, jax.random.PRNGKey(index))
        brax_probe_actions.append(np.asarray(action, dtype=np.float32))
    policy_parity = compare_policy_outputs(
        infer_h4_action_numpy(params, probe_observations),
        np.stack(brax_probe_actions),
    )
    if not policy_parity["passed"]:
        raise RuntimeError(f"actor116 NumPy/Brax probe parity failed: {policy_parity}")

    official_baseline: dict[str, Any] | None = None
    if bundle.status == "COMPLETED":
        baseline_params, baseline_source, transplant_audit = (
            _load_official_v22_baseline(
                checkpoint_path=args.v22_parent_checkpoint,
                stack=stack,
            )
        )
        baseline_policy = _build_brax_policy(
            stack=stack, bundle=bundle, params=baseline_params
        )
        baseline_brax_actions = []
        for index, observation in enumerate(probe_observations):
            action, _ = baseline_policy(
                {
                    "state": stack["jp"].asarray(observation),
                    "privileged_state": stack["jp"].zeros(
                        H4_CRITIC_OBSERVATION_WIDTH
                    ),
                },
                jax.random.PRNGKey(index),
            )
            baseline_brax_actions.append(np.asarray(action, dtype=np.float32))
        baseline_parity = compare_policy_outputs(
            infer_h4_action_numpy(baseline_params, probe_observations),
            np.stack(baseline_brax_actions),
        )
        if not baseline_parity["passed"]:
            raise RuntimeError(
                "official v22 actor116 NumPy/Brax probe parity failed: "
                f"{baseline_parity}"
            )
        baseline_episodes = []
        for seed in H4_STRICT_SEEDS[bundle.expert]:
            print(
                f"H4 official-v22 baseline: starting seed {seed}",
                file=sys.stderr,
                flush=True,
            )
            baseline_episodes.append(
                _run_episode(
                env=env,
                policy=baseline_policy,
                params=baseline_params,
                stack=stack,
                seed=seed,
                expert=bundle.expert,
                )
            )
        baseline_passing = [
            episode["seed"]
            for episode in baseline_episodes
            if episode["strict_passed"]
        ]
        official_baseline = {
            "source_checkpoint": baseline_source,
            "transplant_audit": transplant_audit,
            "transplanted_params_numeric_sha256": h4_params_numeric_sha256(
                baseline_params
            ),
            "evaluation_process": (
                "SAME_PROCESS_ENVIRONMENT_CONTRACT_AND_FIXED_SEEDS_AS_CANDIDATE"
            ),
            "optimizer_updates": 0,
            "policy_inference": "BRAX_DETERMINISTIC_NORMAL_TANH_ACTOR116",
            "actor116_numpy_brax_probe_parity": baseline_parity,
            "episodes": baseline_episodes,
            "summary": {
                "passing_seed_count": len(baseline_passing),
                "passing_seeds": baseline_passing,
                "all_three_strict_pass": len(baseline_passing) == 3,
            },
        }

    episodes = []
    for seed in H4_STRICT_SEEDS[bundle.expert]:
        print(
            f"H4 candidate: starting seed {seed}",
            file=sys.stderr,
            flush=True,
        )
        episodes.append(
            _run_episode(
                env=env,
                policy=policy,
                params=params,
                stack=stack,
                seed=seed,
                expert=bundle.expert,
            )
        )
    if official_baseline is not None:
        baseline_tree_post = sha256_tree(args.v22_parent_checkpoint)
        if (
            baseline_tree_post
            != official_baseline["source_checkpoint"]["sha256_tree_pre"]
        ):
            raise RuntimeError(
                "official v22 checkpoint tree changed during strict evaluation"
            )
        official_baseline["source_checkpoint"][
            "sha256_tree_post"
        ] = baseline_tree_post
    source_post = current_source_hashes(source_paths, root=EXP_ROOT)
    central_post = current_source_hashes(CENTRAL_PATHS, root=EXP_ROOT)
    if source_pre != source_post or central_pre != central_post:
        raise RuntimeError("H4 evaluator/source semantics changed during evaluation")
    passing_seeds = [episode["seed"] for episode in episodes if episode["strict_passed"]]
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": STRICT_ARTIFACT_KIND,
        "hardware_deployment": "PROHIBITED",
        "adoption_allowed": False,
        "release_allowed": False,
        "standalone_direct_runtime_allowed": False,
        "execution_provider": "CPU",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": bundle.candidate_record(),
        "evaluation_contract": {
            "fixed_seeds": list(H4_STRICT_SEEDS[bundle.expert]),
            "physical_command_mps_radps": list(H4_STRICT_COMMANDS[bundle.expert]),
            "duration_s": H4_STRICT_DURATION_S,
            "control_timestep_s": H4_CONTROL_DT_S,
            "physics_timestep_s": H4_PHYSICS_DT_S,
            "control_tick_count": H4_STRICT_CONTROL_TICKS,
            "physics_substep_count": H4_STRICT_PHYSICS_SUBSTEPS,
            "gait_sample_count": H4_STRICT_GAIT_SAMPLES,
            "gait_quality_semantics": (
                "FULL_CURRENT_P0_RECOMPUTED_FROM_N_PLUS_ONE_SUBSTEP_TRACE"
            ),
            "reset": "EXACT_SAFE_INIT_NO_RESET_NOISE",
            "policy_inference": "BRAX_DETERMINISTIC_NORMAL_TANH_ACTOR116",
            "reverse_composition": (
                "PINNED_PERSISTENT_TEACHER_PLUS_TRAINABLE_RESIDUAL"
                if bundle.expert == "reverse"
                else None
            ),
        },
        "central_hashes": central_post,
        "episodes": episodes,
        "official_v22_baseline": official_baseline,
        "summary": {
            "passing_seed_count": len(passing_seeds),
            "passing_seeds": passing_seeds,
            "all_three_strict_pass": len(passing_seeds) == 3,
        },
        "runtime_provenance": {
            "execution_provider": "CPU",
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [
                {"description": str(device), "platform": device.platform}
                for device in jax.devices()
            ],
            "training_jax_devices": bundle.manifest.get("jax_devices"),
            "training_provenance": dict(bundle.training_provenance),
            "candidate_manifest_sha256": bundle.manifest_sha256,
            "candidate_final_params_sha256": bundle.params_sha256,
            "candidate_resolved_config_sha256": bundle.config_sha256,
            "source_and_teacher_hashes": dict(bundle.source_hashes),
            "central_hashes": central_post,
            "evaluation_source_hashes_pre": source_pre,
            "evaluation_source_hashes_post": source_post,
            "pre_post_source_hashes_unchanged": True,
            "params_restore_audit": restore_audit,
            "actor116_numpy_brax_probe_parity": policy_parity,
            "reverse_composition_checks": composition.get("checks"),
        },
    }
    artifact = json_native(artifact)
    audit = validate_h4_strict_artifact(
        artifact,
        bundle=bundle,
        current_central_hashes=central_post,
        current_evaluation_hashes=source_post,
        require_all_three_pass=False,
    )
    artifact["summary"]["recomputed_validation_passed"] = bool(
        audit["passing_seed_count"] == len(passing_seeds)
    )
    return artifact, bundle, central_post, source_post


def main() -> None:
    args = _resolve_process_start_paths(build_parser().parse_args())
    artifact, bundle, central_hashes, evaluation_hashes = run_evaluation(args)
    output_path = args.output
    artifact_sha = write_new_json(output_path, artifact)
    result: dict[str, Any] = {
        "strict_artifact": {"path": str(output_path), "sha256": artifact_sha},
        "passing_seed_count": artifact["summary"]["passing_seed_count"],
        "all_three_strict_pass": artifact["summary"]["all_three_strict_pass"],
        "hardware_deployment": "PROHIBITED",
        "adoption_allowed": False,
    }
    if args.promotion_evidence_output is not None:
        evidence = build_integrated_promotion_evidence(
            strict_artifact_path=output_path,
            bundle=bundle,
            current_central_hashes=central_hashes,
            current_evaluation_hashes=evaluation_hashes,
        )
        evidence_path = args.promotion_evidence_output
        evidence_sha = write_new_json(evidence_path, evidence)
        result["promotion_evidence"] = {
            "path": str(evidence_path),
            "sha256": evidence_sha,
        }
    print(json.dumps(result, indent=2, allow_nan=False))
    if not artifact["summary"]["all_three_strict_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
