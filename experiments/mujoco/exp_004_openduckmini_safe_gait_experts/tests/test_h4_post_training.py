from __future__ import annotations

import csv
import copy
from dataclasses import dataclass, replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pytest
import safe_gait_experts.h4_post_training as h4pt

from safe_gait_experts.contract import (
    ACTUATOR_JOINT_ORDER,
    HEAD_JOINTS,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
)

from safe_gait_experts.h4_post_training import (
    H4_ACTION_WIDTH,
    H4_ACTOR_OBSERVATION_WIDTH,
    H4_CRITIC_OBSERVATION_WIDTH,
    H4_STRICT_GAIT_SAMPLES,
    H4_STRICT_PHYSICS_SUBSTEPS,
    H4_STRICT_COMMANDS,
    H4_STRICT_SEEDS,
    PINNED_FORWARD_ITERATION_V2_AUTHORIZATION_SHA256,
    PINNED_FORWARD_MINIMUM_SPEC_CANONICAL_SHA256,
    PINNED_FORWARD_MINIMUM_SPEC_SHA256,
    PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256,
    PINNED_REVERSE_ITERATION_V2_AUTHORIZATION_SHA256,
    PINNED_REVERSE_MINIMUM_SPEC_SHA256,
    PINNED_SELECTED_REVERSE_TEACHER_SHA256,
    STRICT_ARTIFACT_KIND,
    build_promotion_evidence,
    canonical_json_sha256,
    compare_policy_outputs,
    infer_h4_action_numpy,
    json_native,
    legacy_metrics_from_gait_quality,
    load_json_strict,
    load_trusted_h4_params,
    mask_h4_head_action,
    rederive_central_safety_audit_from_control_trace,
    rederive_h4_control_contract,
    rederive_h4_safety_acceptance,
    sha256_file,
    validate_h4_params,
    validate_h4_strict_artifact,
    validate_h4_strict_episode,
    validate_h4_training_source_closure,
    validate_trusted_h4_bundle,
    write_new_json,
)


EXP_ROOT = Path(__file__).resolve().parents[1]


def _load_runner_v4_authorization(expert: str) -> dict[str, Any]:
    """Use the immutable v4 payload as the semantic-audit fixture source.

    The active v4 loader intentionally rejects the now-changed current source
    tree after v5.  Historical payload semantics remain independently valid.
    """

    module_name = "exp004_post_training_runner_contract"
    runner = sys.modules.get(module_name)
    if runner is None:
        runner_path = EXP_ROOT / "scripts" / "train_h4_aligned_expert.py"
        module_spec = importlib.util.spec_from_file_location(module_name, runner_path)
        assert module_spec is not None and module_spec.loader is not None
        runner = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = runner
        module_spec.loader.exec_module(runner)
    spec = runner._iteration_v4_spec(expert)
    payload = runner._load_json_strict(spec["auth_path"])
    return {
        "payload": payload,
        "semantic_audit": runner.validate_iteration_v4_authorization_payload(
            payload, expert=expert
        ),
    }


def _load_runner_v5_authorization(expert: str) -> dict[str, Any]:
    module_name = "exp004_post_training_runner_contract"
    runner = sys.modules.get(module_name)
    if runner is None:
        runner_path = EXP_ROOT / "scripts" / "train_h4_aligned_expert.py"
        module_spec = importlib.util.spec_from_file_location(module_name, runner_path)
        assert module_spec is not None and module_spec.loader is not None
        runner = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = runner
        module_spec.loader.exec_module(runner)
    return runner.load_iteration_v5_authorization(expert=expert)


@dataclass(frozen=True)
class _Count:
    hi: Any
    lo: Any


@dataclass(frozen=True)
class _Normalizer:
    mean: Any
    std: Any
    summed_variance: Any
    count: Any


def _network(shapes: tuple[tuple[int, int], ...]) -> dict[str, Any]:
    return {
        "params": {
            f"hidden_{index}": {
                "kernel": np.zeros(shape, dtype=np.float32),
                "bias": np.zeros(shape[1], dtype=np.float32),
            }
            for index, shape in enumerate(shapes)
        }
    }


def _params() -> list[Any]:
    normalizer = _Normalizer(
        mean={
            "state": np.zeros(H4_ACTOR_OBSERVATION_WIDTH, np.float32),
            "privileged_state": np.zeros(H4_CRITIC_OBSERVATION_WIDTH, np.float32),
        },
        std={
            "state": np.ones(H4_ACTOR_OBSERVATION_WIDTH, np.float32),
            "privileged_state": np.ones(H4_CRITIC_OBSERVATION_WIDTH, np.float32),
        },
        summed_variance={
            "state": np.ones(H4_ACTOR_OBSERVATION_WIDTH, np.float32),
            "privileged_state": np.ones(H4_CRITIC_OBSERVATION_WIDTH, np.float32),
        },
        count=_Count(np.asarray(0, np.uint32), np.asarray(1, np.uint32)),
    )
    actor = _network(((116, 512), (512, 256), (256, 128), (128, 28)))
    actor["params"]["hidden_3"]["bias"][:14] = np.float32(0.5)
    critic = _network(((227, 512), (512, 256), (256, 128), (128, 1)))
    return [normalizer, actor, critic]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _bundle_files(
    tmp_path: Path,
    *,
    expert: str = "forward",
    forward_iteration_v2: bool = False,
    reverse_iteration_v2: bool = False,
    forward_iteration_v3: bool = False,
    reverse_iteration_v3: bool = False,
    forward_iteration_v4: bool = False,
    reverse_iteration_v4: bool = False,
    wiring_only: bool = False,
) -> dict[str, Any]:
    if sum(
        bool(value)
        for value in (
            forward_iteration_v2,
            reverse_iteration_v2,
            forward_iteration_v3,
            reverse_iteration_v3,
            forward_iteration_v4,
            reverse_iteration_v4,
        )
    ) > 1:
        raise ValueError("test fixture iteration modes are mutually exclusive")
    if wiring_only and not (forward_iteration_v4 or reverse_iteration_v4):
        raise ValueError("test fixture wiring mode is iteration-v4 only")
    run_name = f"h4_{expert}_test"
    trusted_run_root = tmp_path / "h4_training_runs"
    run_dir = trusted_run_root / expert / run_name
    run_dir.mkdir(parents=True)
    params = run_dir / "final_params.pkl"
    params.write_bytes(b"manifest-bound-test-pickle")
    params_sha = sha256_file(params)
    activity = "PPO_WIRING_TRAINING" if wiring_only else "PPO_PILOT_TRAINING"
    shape = (
        {
            "num_timesteps": 40,
            "num_envs": 2,
            "unroll_length": 20,
            "batch_size": 1,
            "num_minibatches": 2,
            "num_updates_per_batch": 1,
            "num_evals": 1,
        }
        if wiring_only
        else {
            "num_timesteps": 250_000,
            "num_envs": 1250,
            "unroll_length": 20,
            "batch_size": 125,
            "num_minibatches": 20,
            "num_updates_per_batch": 4,
            "num_evals": 2,
        }
    )
    backend = {
        "requested_cli_platform": "gpu",
        "jax_platform_selector": "cuda,cpu",
        "expected_resolved_backend": "gpu",
        "resolved_default_backend": "gpu",
        "resolved_device_platforms": ["gpu"],
        "resolved_devices": ["cuda:0"],
        "local_cpu_callback_devices": ["TFRT_CPU_0"],
        "local_cpu_callback_available": True,
        "passed": True,
    }
    xla = {
        "requested_cli_platform": "gpu",
        "xla_flags_before": None,
        "xla_flags_effective": "--xla_gpu_autotune_level=4",
        "policy": "CORRECTNESS_CHECKED_LEVEL4_DISQUALIFY_MISMATCH",
        "configured_before_training_stack_import": True,
        "correctness_check_enabled": True,
        "mismatching_autotune_candidates_disqualified": True,
        "cpu_mode_did_not_set_xla_flags": True,
        "passed": True,
    }
    callback = {
        "input": 2.0,
        "callback_observed": 2.0,
        "result": 3.0,
        "local_cpu_callback_executed": True,
        "passed": True,
    }

    def device_audit(label: str, leaves: int) -> dict[str, Any]:
        return {
            "label": label,
            "jax_array_leaf_count": leaves,
            "platforms": ["gpu"],
            "devices": ["cuda:0"],
            "expected_platform": "gpu",
            "passed": True,
        }

    pre_device_audits = {
        "probe_state": device_audit("pre_training_probe_state", 232),
        "restore_params": device_audit("pre_training_restore_params", 10),
    }
    checkpoint = {
        "source_actor_width": 101,
        "target_actor_width": 116,
        "source_critic_width": 212,
        "target_critic_width": 227,
        "inserted_feature_count": 15,
        "insert_offset": 101,
        "actor_new_15_rows_exact_zero": True,
        "critic_new_15_rows_exact_zero": True,
        "all_restore_leaves_finite": True,
        "passed": True,
    }
    post_checkpoint = {
        "source_actor_width": 116,
        "target_actor_width": 116,
        "critic_width": 227,
        "normalizer_state_width": 116,
        "normalizer_privileged_width": 227,
        "all_restore_leaves_finite": True,
        "restore_structure_validated": True,
        "transplant_applied": False,
        "passed": True,
    }
    config: dict[str, Any] = {
        "schema_version": 1,
        "hardware_deployment": "PROHIBITED",
        "expert": expert,
        "activity": activity,
        "wiring_only": wiring_only,
        "platform": "gpu",
        "backend_resolution": backend,
        "xla_autotune_policy": xla,
        "debug_callback_preflight": callback,
        "pre_training_device_audits": pre_device_audits,
        "shape": shape,
        "interactions_per_training_step": 40 if wiring_only else 50_000,
        "expected_training_steps": 1 if wiring_only else 5,
        "expected_optimizer_updates": 2 if wiring_only else 400,
        "output_dir": str(run_dir.resolve()),
        "actor_observation_width": 116,
        "observation_mode": "h4_116_transplant",
        "network_factory": {
            "policy_hidden_layer_sizes": [512, 256, 128],
            "policy_obs_key": "state",
        },
        "checkpoint_compatibility": checkpoint,
        "promotion_evidence": None,
        "promotion_protocol": {
            "candidate_stage_interactions": 250_000,
            "candidate_training_steps_of_50000_interactions": 5,
            "fixed_failure3_seeds": list(H4_STRICT_SEEDS[expert]),
            "promoted_stage_interactions": 1_000_000,
        },
        "ppo": {**shape, "normalize_observations": True},
    }
    legacy_trainer = EXP_ROOT / "scripts" / "train_expert.py"

    def source_record(path: Path) -> dict[str, str]:
        resolved = path.resolve()
        return {"path": str(resolved), "sha256": sha256_file(resolved)}

    source_hashes = {"legacy_trainer": source_record(legacy_trainer)}
    if expert == "forward":
        forward_spec = (
            EXP_ROOT
            / "artifacts"
            / "h4_forward_retraining_minimum_spec_from_slip_causality_v1.json"
        )
        source_hashes["forward_minimum_spec"] = source_record(forward_spec)
        assert (
            source_hashes["forward_minimum_spec"]["sha256"]
            == PINNED_FORWARD_MINIMUM_SPEC_SHA256
        )
        config["forward_minimum_spec"] = {
            "path": str(forward_spec.resolve()),
            "sha256": PINNED_FORWARD_MINIMUM_SPEC_SHA256,
            "canonical_sha256": PINNED_FORWARD_MINIMUM_SPEC_CANONICAL_SHA256,
        }
        if forward_iteration_v2:
            authorization = (
                EXP_ROOT / "artifacts" / "h4_forward_iteration_v2_authorization.json"
            ).resolve()
            failed_run = (
                EXP_ROOT
                / "artifacts"
                / "h4_training_runs"
                / "forward"
                / "h4_forward_250k_seed20260809_v1"
            ).resolve()
            failed_manifest = failed_run / "run_manifest.json"
            failed_params = failed_run / "final_params.pkl"
            strict_evaluation = failed_run / "h4_integrated_strict_3x6s_v2.json"
            assert sha256_file(authorization) == (
                PINNED_FORWARD_ITERATION_V2_AUTHORIZATION_SHA256
            )
            bound_causal_inputs = {
                "failed_candidate_manifest": source_record(failed_manifest),
                "failed_candidate_params": source_record(failed_params),
                "integrated_strict_evaluation": source_record(strict_evaluation),
            }
            source_hashes.update(
                {
                    "forward_iteration_v2_authorization": source_record(
                        authorization
                    ),
                    "forward_iteration_v2_failed_candidate_manifest": (
                        bound_causal_inputs["failed_candidate_manifest"]
                    ),
                    "forward_iteration_v2_failed_candidate_params": (
                        bound_causal_inputs["failed_candidate_params"]
                    ),
                    "forward_iteration_v2_integrated_strict_evaluation": (
                        bound_causal_inputs["integrated_strict_evaluation"]
                    ),
                }
            )
            config.update(
                {
                    "forward_iteration_v2": True,
                    "reverse_iteration_v2": False,
                    "authorized_iteration_v2_250k_contract_id": (
                        h4pt.H4_FORWARD_ITERATION_V2_CONTRACT_ID
                    ),
                    "forward_iteration_v2_authorization": {
                        "path": str(authorization),
                        "sha256": (
                            PINNED_FORWARD_ITERATION_V2_AUTHORIZATION_SHA256
                        ),
                        "contract_id": h4pt.H4_FORWARD_ITERATION_V2_CONTRACT_ID,
                        "status": "AUTHORIZED_SIMULATION_250K_ONLY",
                        "semantic_audit": {
                            key: True
                            for key in (
                                "schema",
                                "kind",
                                "status",
                                "hardware_prohibited",
                                "authorization_exact",
                                "contract_id",
                                "required_flag",
                                "training_exact",
                                "curriculum_exact",
                                "reward_scales_exact",
                                "reward_deltas_exact",
                                "force_band_exact",
                                "force_tail_exact",
                                "contact_pulse_exact",
                                "strict_gate_unchanged",
                                "central_hashes_exact",
                                "manifest_binding_exact",
                            )
                        },
                        "bound_causal_inputs": bound_causal_inputs,
                        "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                        "adoption_release_hardware": "PROHIBITED",
                    },
                }
            )
    else:
        selected = (
            EXP_ROOT / "artifacts" / "h4_reverse_slew_feasible_teacher_selected_v1.json"
        )
        authorization = (
            EXP_ROOT
            / "artifacts"
            / "h4_reverse_training_composition_authorization_v1.json"
        )
        reverse_spec = (
            EXP_ROOT
            / "artifacts"
            / "h4_reverse_retraining_minimum_spec_from_slip_causality_v1.json"
        )
        source_hashes.update(
            {
                "selected_reverse_teacher": source_record(selected),
                "reverse_composition_authorization": source_record(authorization),
                "reverse_minimum_spec": source_record(reverse_spec),
            }
        )
        config["selected_reverse_teacher"] = {
            "path": str(selected.resolve()),
            "sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256
        }
        config["reverse_composition_authorization"] = {
            "path": str(authorization.resolve()),
            "sha256": PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256,
            "status": "SIMULATION_TRAINING_COMPOSITION_AUTHORIZED_NOT_ADOPTED",
            "standalone_direct_runtime_allowed": False,
            "adoption_allowed": False,
            "release_allowed": False,
            "hardware_allowed": False,
        }
        config["reverse_minimum_spec"] = {
            "path": str(reverse_spec.resolve()),
            "sha256": PINNED_REVERSE_MINIMUM_SPEC_SHA256,
        }
        config["backward_residual_scale"] = 0.12
        if reverse_iteration_v2:
            iteration_authorization = (
                EXP_ROOT / "artifacts" / "h4_reverse_iteration_v2_authorization.json"
            ).resolve()
            failed_run = (
                EXP_ROOT
                / "artifacts"
                / "h4_training_runs"
                / "reverse"
                / "h4_reverse_250k_seed20260810_level4_v1"
            ).resolve()
            failed_manifest = failed_run / "run_manifest.json"
            failed_params = failed_run / "final_params.pkl"
            strict_evaluation = failed_run / "h4_integrated_strict_3x6s_v1.json"
            assert sha256_file(iteration_authorization) == (
                PINNED_REVERSE_ITERATION_V2_AUTHORIZATION_SHA256
            )
            bound_causal_inputs = {
                "failed_candidate_manifest": source_record(failed_manifest),
                "failed_candidate_params": source_record(failed_params),
                "integrated_strict_evaluation": source_record(strict_evaluation),
            }
            source_hashes.update(
                {
                    "reverse_iteration_v2_authorization": source_record(
                        iteration_authorization
                    ),
                    "reverse_iteration_v2_failed_candidate_manifest": (
                        bound_causal_inputs["failed_candidate_manifest"]
                    ),
                    "reverse_iteration_v2_failed_candidate_params": (
                        bound_causal_inputs["failed_candidate_params"]
                    ),
                    "reverse_iteration_v2_integrated_strict_evaluation": (
                        bound_causal_inputs["integrated_strict_evaluation"]
                    ),
                }
            )
            expected_legacy_reward = {
                "target_imitation": -20.0,
                "contact_imitation": 15.0,
                "tracking_sigma": 0.01,
                "backward_residual_scale": 0.12,
            }
            config.update(
                {
                    "forward_iteration_v2": False,
                    "reverse_iteration_v2": True,
                    "authorized_iteration_v2_250k_contract_id": (
                        h4pt.H4_REVERSE_ITERATION_V2_CONTRACT_ID
                    ),
                    "reverse_iteration_v2_authorization": {
                        "path": str(iteration_authorization),
                        "sha256": (
                            PINNED_REVERSE_ITERATION_V2_AUTHORIZATION_SHA256
                        ),
                        "contract_id": h4pt.H4_REVERSE_ITERATION_V2_CONTRACT_ID,
                        "status": "AUTHORIZED_SIMULATION_250K_ONLY",
                        "semantic_audit": {
                            key: True
                            for key in (
                                "schema",
                                "kind",
                                "status",
                                "hardware_prohibited",
                                "authorization_exact",
                                "contract_id",
                                "required_flag",
                                "training_exact",
                                "teacher_guard_exact",
                                "legacy_reward_exact",
                                "tracking_sigma_truthful",
                                "legacy_schema3_causal_truth",
                                "curriculum_exact",
                                "reward_scales_exact",
                                "new_force_pulse_disabled",
                                "strict_gate_unchanged",
                                "central_hashes_exact",
                                "manifest_binding_exact",
                            )
                        },
                        "bound_causal_inputs": bound_causal_inputs,
                        "legacy_reward_config_audit": {
                            "expected": expected_legacy_reward,
                            "per_environment": {
                                "train": expected_legacy_reward,
                                "eval": expected_legacy_reward,
                            },
                            "passed": True,
                        },
                        "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
                        "adoption_release_hardware": "PROHIBITED",
                    },
                }
            )
    v3_spec: dict[str, Any] | None = None
    v3_auth_config: dict[str, Any] | None = None
    if forward_iteration_v3 or reverse_iteration_v3:
        assert (expert == "forward") is forward_iteration_v3
        v3_spec = h4pt._iteration_v3_spec(expert)
        authorization_path = (
            EXP_ROOT / "artifacts" / v3_spec["auth_filename"]
        ).resolve()
        authorization_payload = load_json_strict(authorization_path)
        causal = authorization_payload["causal_input"]
        failed_run = (
            EXP_ROOT
            / causal["failed_candidate_root_relative_path"]
            / causal["failed_candidate_run_name"]
        ).resolve()
        evidence_path = (EXP_ROOT / causal["integrated_strict_evaluation"]["path"]).resolve()
        bound_causal_inputs = {
            "failed_candidate_manifest": source_record(
                failed_run / "run_manifest.json"
            ),
            "failed_candidate_params": source_record(failed_run / "final_params.pkl"),
            "integrated_strict_evaluation": source_record(evidence_path),
        }
        source_hashes.update(
            {
                v3_spec["auth_label"]: source_record(authorization_path),
                f"{v3_spec['prefix']}failed_candidate_manifest": (
                    bound_causal_inputs["failed_candidate_manifest"]
                ),
                f"{v3_spec['prefix']}failed_candidate_params": (
                    bound_causal_inputs["failed_candidate_params"]
                ),
                f"{v3_spec['prefix']}integrated_strict_evaluation": (
                    bound_causal_inputs["integrated_strict_evaluation"]
                ),
            }
        )
        curriculum = authorization_payload["curriculum"]
        training_contract = authorization_payload["training_contract"]
        v3_auth_config = {
            "path": str(authorization_path),
            "sha256": v3_spec["auth_sha"],
            "contract_id": v3_spec["contract"],
            "status": "AUTHORIZED_SIMULATION_250K_ONLY",
            "semantic_audit": {
                key: True for key in v3_spec["semantic_keys"]
            },
            "bound_causal_inputs": bound_causal_inputs,
            "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
            "adoption_release_hardware": "PROHIBITED",
        }
        if expert == "reverse":
            expected_legacy = {
                **authorization_payload["legacy_reward_config"]["iteration_v3_exact"],
                "backward_residual_scale": 0.12,
            }
            v3_auth_config["legacy_reward_config_audit"] = {
                "expected": expected_legacy,
                "per_environment": {
                    "train": expected_legacy,
                    "eval": expected_legacy,
                },
                "passed": True,
            }
        optimizer = {
            key: training_contract[key]
            for key in (
                "learning_rate",
                "entropy_cost",
                "clipping_epsilon",
                "discounting",
                "max_grad_norm",
            )
        }
        config.update(
            {
                "training_contract_id": v3_spec["contract"],
                "authorized_iteration_v2_250k_contract_id": None,
                "authorized_iteration_v3_250k_contract_id": v3_spec["contract"],
                "qualification_use": "AUTHORIZED_250K_PILOT",
                "forward_iteration_v2": False,
                "reverse_iteration_v2": False,
                v3_spec["flag"]: True,
                v3_spec["auth_key"]: v3_auth_config,
                "initialization_source": "V22_BRAX_CHECKPOINT",
                "trusted_h4_parent": None,
                "pinned_v22_parent_tree_sha256": h4pt.PINNED_V22_PARENT_TREE_SHA256,
                "seed": training_contract["seed"],
                "anchor_config": {
                    "physical_primary": curriculum["physical_primary_mps_radps"],
                    "policy_observation_anchor": curriculum[
                        "policy_observation_anchor"
                    ],
                    "stand_probability": curriculum["stand_probability"],
                    "exact_primary_probability": curriculum[
                        "exact_primary_probability"
                    ],
                    "local_probability": curriculum["local_probability"],
                    "local_vx_m_s": curriculum["local_vx_m_s"],
                    "transition_probability": curriculum["transition_probability"],
                    "transition_vx_m_s": curriculum[
                        "transition_vx_uniform_m_s"
                    ],
                },
                "reward_scales": authorization_payload["reward_contract"][
                    "exact_scales"
                ],
                "reset_noise_multiplier": 1.0,
                "backward_residual_scale": 0.12,
                **optimizer,
                "ppo": {
                    **config["ppo"],
                    **optimizer,
                },
            }
        )

    v4_spec: dict[str, Any] | None = None
    v4_auth_config: dict[str, Any] | None = None
    if forward_iteration_v4 or reverse_iteration_v4:
        assert (expert == "forward") is forward_iteration_v4
        v4_spec = h4pt._iteration_v4_spec(expert)
        authorization_path = (
            EXP_ROOT / "artifacts" / v4_spec["auth_filename"]
        ).resolve()
        authorization_payload = load_json_strict(authorization_path)
        authorization_sha = sha256_file(authorization_path)
        causal = authorization_payload["causal_input"]
        failed_run = (
            EXP_ROOT
            / causal["failed_candidate_root_relative_path"]
            / causal["failed_candidate_run_name"]
        ).resolve()
        evidence_path = (
            EXP_ROOT / causal["integrated_strict_evaluation"]["path"]
        ).resolve()
        previous_path = (
            EXP_ROOT / causal["previous_iteration_authorization"]["path"]
        ).resolve()
        bound_causal_inputs = {
            "previous_iteration_authorization": source_record(previous_path),
            "failed_candidate_params": source_record(failed_run / "final_params.pkl"),
            "failed_candidate_manifest": source_record(failed_run / "run_manifest.json"),
            "integrated_strict_evaluation": source_record(evidence_path),
        }
        bound_causal_sources = {
            label: source_record(EXP_ROOT / record["path"])
            for label, record in authorization_payload[
                "causal_source_closure"
            ].items()
        }
        # Real v4 manifests contain these ordinary runner sources plus the
        # authorization-owned aliases.  Exercise the exact permitted aliases
        # while retaining rejection of every unrelated duplicate path.
        source_hashes["h4_alignment"] = source_record(
            EXP_ROOT / "safe_gait_experts" / "h4_training_alignment.py"
        )
        source_hashes["h4_runner"] = source_record(
            EXP_ROOT / "scripts" / "train_h4_aligned_expert.py"
        )
        source_hashes[f"{v4_spec['prefix']}authorization"] = source_record(
            authorization_path
        )
        for label, record in bound_causal_inputs.items():
            source_hashes[f"{v4_spec['prefix']}{label}"] = record
        for label, record in bound_causal_sources.items():
            source_hashes[f"{v4_spec['prefix']}source_{label}"] = record
        v4_auth_config = {
            "path": str(authorization_path),
            "sha256": authorization_sha,
            "contract_id": v4_spec["contract"],
            "status": "AUTHORIZED_SIMULATION_250K_ONLY",
            "semantic_audit": {
                key: True for key in v4_spec["semantic_keys"]
            },
            "bound_causal_inputs": bound_causal_inputs,
            "bound_causal_sources": bound_causal_sources,
            "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
            "adoption_release_hardware": "PROHIBITED",
        }
        curriculum = authorization_payload["curriculum"]
        training_contract = authorization_payload["training_contract"]
        if expert == "reverse":
            expected_legacy = {
                **authorization_payload["legacy_reward_config"]["iteration_v4_exact"],
                "backward_residual_scale": 0.24,
            }
            v4_auth_config["legacy_reward_config_audit"] = {
                "expected": expected_legacy,
                "per_environment": {
                    "train": expected_legacy,
                    "eval": expected_legacy,
                },
                "passed": True,
            }
        optimizer = {
            key: training_contract[key]
            for key in (
                "learning_rate",
                "entropy_cost",
                "clipping_epsilon",
                "discounting",
                "max_grad_norm",
            )
        }
        source_semantic_preflight = {
            "timing": "ONCE_BEFORE_PPO_COLLECTION",
            "reference_source": "OFFICIAL_MJX_ENV_STEP_WRAPPER_NSUBSTEPS_10",
            "candidate_source": "SINGLE_INSTRUMENTED_TEN_SUBSTEP_SCAN_ENDPOINT",
            "source_provenance": {
                "source_root": "/home/user/openduck_training_20260729",
                "joystick": {
                    "resolved_path": (
                        "/home/user/openduck_training_20260729/"
                        "playground/open_duck_mini_v2/joystick.py"
                    ),
                    "relative_path": "playground/open_duck_mini_v2/joystick.py",
                    "sha256": (
                        "95890569d971725308b5a9c0996bfa5fd9520479f014f325e810aa1db272eb9d"
                    ),
                },
                "mjx_env": {
                    "resolved_path": (
                        "/home/user/openduck_training_20260729/.venv/lib/"
                        "python3.12/site-packages/mujoco_playground/_src/mjx_env.py"
                    ),
                    "relative_path": (
                        ".venv/lib/python3.12/site-packages/"
                        "mujoco_playground/_src/mjx_env.py"
                    ),
                    "sha256": (
                        "c3f1cfe0de036c3ccbba46e8cdd661cb48bfea8f182955298205f17787f53dfe"
                    ),
                },
                "step_source_sha256": (
                    "26571e7510b2837dca07f69890dc26a89695dff4caa1fdc6a0d6736bd22da06b"
                ),
                "step_source_semantics": (
                    "LAX_SCAN_XS_EMPTY_LENGTH_NSUBSTEPS_BODY_REPLACE_CTRL_"
                    "ACTION_THEN_MJX_STEP_RETURN_FINAL_CARRY"
                ),
                "all_files_under_requested_source_root": True,
                "passed": True,
            },
            "probe_input": {
                "seed": 20260809,
                "reset_noise_multiplier": 1.0,
                "initial_state_source": "ENV_RESET_JAX_PRNGKEY_SEED",
                "action_shape": [14],
                "action_dtype": "float32",
                "action_all_zero": True,
            },
            "qualifying_dynamic_state_fields": [
                "qpos",
                "qvel",
                "act",
                "ctrl",
                "time",
                "qacc_warmstart",
            ],
            "dynamic6_exact": True,
            "dynamic6_max_abs_error": 0.0,
            "dynamic6_field_count": 6,
            "derived_diagnostics": {
                "qualification_role": (
                    "NON_QUALIFYING_OBSERVED_DIAGNOSTICS_ONLY"
                ),
                "fields": {
                    "cfrc_int": {"exact": False, "max_abs_error": 0.14},
                    "cfrc_ext": {"exact": False, "max_abs_error": 0.13},
                },
                "all_finite": True,
                "exclusion_is_semantic_not_tolerance": True,
                "numeric_tolerance_used": False,
            },
            "observed_reference_count": 1,
            "passed": True,
        }
        authority_requirement = {
            "dynamic6_exact": True,
            "dynamic6_max_abs_error": 0.0,
            "dynamic6_field_count": 6,
            "dynamic6_field_count_exact": True,
            "saved_dynamic6_substep_count": 10,
            "saved_dynamic6_field_count": 6,
            "saved_dynamic6_field_count_exact": True,
            "saved_dynamic6_all_finite": True,
            "telemetry_force_shape": [2],
            "telemetry_force_shape_valid": True,
            "telemetry_force_all_finite": True,
            "count_totals_qualification_role": (
                "NON_QUALIFYING_FINITE_DIAGNOSTICS_ONLY"
            ),
            "host_count_multiplication_for_qualification": False,
            "numeric_tolerance_used": False,
            "authority_violation_count": 0.0,
            "assertion_token_sum": 0.0,
            "fail_closed_before_output_commit": True,
            "full_nonempty_episode_rows_required": True,
            "wiring_zero_episode_rows_require_compiled_assertion_evidence": True,
        }
        single_authority_runtime = {
            "audit_mode": h4pt.H4_FORWARD_V4_FULL_RUNTIME_AUDIT_MODE,
            "dynamic6_exact": True,
            "dynamic6_max_abs_error": 0.0,
            "dynamic6_field_count": 6,
            "dynamic6_field_count_exact": True,
            "saved_dynamic6_substep_count": 10,
            "saved_dynamic6_field_count": 6,
            "saved_dynamic6_field_count_exact": True,
            "saved_dynamic6_all_finite": True,
            "telemetry_force_shape": [2],
            "telemetry_force_shape_valid": True,
            "telemetry_force_all_finite": True,
            "observed_episode_metric_rows": 5,
            "authority_violation_count": 0.0,
            "assertion_token_sum": 0.0,
            "passed": True,
        }
        if wiring_only:
            single_authority_runtime = {
                "audit_mode": h4pt.H4_FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE,
                "observed_episode_metric_rows": 0,
                "episode_metric_rows_exact_if_observed": True,
                "source_semantic_preflight_passed": True,
                "per_step_compiled_fail_closed_assertion_bound": True,
                "completed_environment_interactions": 40,
                "completed_training_steps": 1,
                "completed_optimizer_updates": 2,
                "progress_reached_final_interaction": True,
                "final_params_all_finite": True,
                "final_metrics_all_finite": True,
                "source_and_teacher_unchanged": True,
                "authority_violation_count": 0.0,
                "assertion_token_sum": 0.0,
                "passed": True,
            }
        config.update(
            {
                "training_contract_id": (
                    v4_spec["wiring_contract"]
                    if wiring_only
                    else v4_spec["contract"]
                ),
                "authorized_iteration_v2_250k_contract_id": None,
                "authorized_iteration_v3_250k_contract_id": None,
                "authorized_iteration_v4_250k_contract_id": v4_spec["contract"],
                "qualification_use": (
                    "WIRING_PREFLIGHT_ONLY_NOT_250K_QUALIFICATION"
                    if wiring_only
                    else "AUTHORIZED_250K_PILOT"
                ),
                "forward_iteration_v2": False,
                "reverse_iteration_v2": False,
                "forward_iteration_v3_touchdown_balance": False,
                "reverse_iteration_v3_no_target_imitation": False,
                "forward_iteration_v4_contact_event_validity_persistence": (
                    expert == "forward"
                ),
                "reverse_iteration_v4_residual_transfer_gain_024": (
                    expert == "reverse"
                ),
                "forward_v4_substep_contact": expert == "forward",
                **(
                    {
                        "forward_v4_source_semantic_preflight": (
                            source_semantic_preflight
                        ),
                        "forward_v4_single_authority_runtime_requirement": (
                            authority_requirement
                        ),
                        "forward_v4_single_authority_runtime_audit_mode": (
                            h4pt.H4_FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                            if wiring_only
                            else h4pt.H4_FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                        ),
                    }
                    if expert == "forward"
                    else {}
                ),
                v4_spec["auth_key"]: v4_auth_config,
                "initialization_source": "V22_BRAX_CHECKPOINT",
                "trusted_h4_parent": None,
                "pinned_v22_parent_tree_sha256": h4pt.PINNED_V22_PARENT_TREE_SHA256,
                "seed": training_contract["seed"],
                "anchor_config": {
                    "physical_primary": curriculum["physical_primary_mps_radps"],
                    "policy_observation_anchor": curriculum[
                        "policy_observation_anchor"
                    ],
                    "stand_probability": curriculum["stand_probability"],
                    "exact_primary_probability": curriculum[
                        "exact_primary_probability"
                    ],
                    "local_probability": curriculum["local_probability"],
                    "local_vx_m_s": curriculum["local_vx_m_s"],
                    "transition_probability": curriculum["transition_probability"],
                    "transition_vx_m_s": curriculum[
                        "transition_vx_uniform_m_s"
                    ],
                },
                "reward_scales": authorization_payload["reward_contract"][
                    "exact_scales"
                ],
                "reset_noise_multiplier": 1.0,
                "backward_residual_scale": 0.24 if expert == "reverse" else 0.12,
                **optimizer,
                "ppo": {**config["ppo"], **optimizer},
            }
        )

    config_path = run_dir / "resolved_config.json"
    _write_json(config_path, config)
    result = {
        "schema_version": 1,
        "status": "WIRING_PASS" if wiring_only else "COMPLETED",
        "hardware_deployment": "PROHIBITED",
        "activity": activity,
        "expert": expert,
        "environment_interactions": 40 if wiring_only else 250_000,
        "optimizer_updates": 2 if wiring_only else 400,
        "final_metrics": {"training/total_loss": 1.0},
        "final_metrics_all_finite": True,
        "final_metrics_nonzero_count": 1,
        "backend_resolution": backend,
        "xla_autotune_policy": xla,
        "debug_callback_preflight": callback,
        "pre_training_device_audits": pre_device_audits,
        "post_training_device_audit": device_audit("post_training_params", 25),
        "checkpoint_compatibility": checkpoint,
        "post_training_checkpoint_audit": post_checkpoint,
        "final_params": {"path": str(params), "sha256": params_sha},
        "source_and_teacher_unchanged": True,
    }
    if v3_spec is not None:
        result.update(
            {
                "training_contract_id": v3_spec["contract"],
                "authorized_iteration_v2_250k_contract_id": None,
                "authorized_iteration_v3_250k_contract_id": v3_spec["contract"],
                "qualification_use": "AUTHORIZED_250K_PILOT",
                "forward_iteration_v2": False,
                "reverse_iteration_v2": False,
                v3_spec["flag"]: True,
                f"{v3_spec['auth_key']}_sha256": v3_spec["auth_sha"],
            }
        )
    if v4_spec is not None:
        assert v4_auth_config is not None
        result.update(
            {
                "training_contract_id": (
                    v4_spec["wiring_contract"]
                    if wiring_only
                    else v4_spec["contract"]
                ),
                "authorized_iteration_v2_250k_contract_id": None,
                "authorized_iteration_v3_250k_contract_id": None,
                "authorized_iteration_v4_250k_contract_id": v4_spec["contract"],
                "qualification_use": (
                    "WIRING_PREFLIGHT_ONLY_NOT_250K_QUALIFICATION"
                    if wiring_only
                    else "AUTHORIZED_250K_PILOT"
                ),
                "forward_iteration_v2": False,
                "reverse_iteration_v2": False,
                "forward_iteration_v3_touchdown_balance": False,
                "reverse_iteration_v3_no_target_imitation": False,
                "forward_iteration_v4_contact_event_validity_persistence": (
                    expert == "forward"
                ),
                "reverse_iteration_v4_residual_transfer_gain_024": (
                    expert == "reverse"
                ),
                "forward_v4_substep_contact": expert == "forward",
                **(
                    {
                        "forward_v4_source_semantic_preflight": (
                            source_semantic_preflight
                        ),
                        "forward_v4_single_authority_runtime_requirement": (
                            authority_requirement
                        ),
                        "forward_v4_single_authority_runtime": (
                            single_authority_runtime
                        ),
                        "forward_v4_single_authority_runtime_audit_mode": (
                            h4pt.H4_FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                            if wiring_only
                            else h4pt.H4_FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                        ),
                    }
                    if expert == "forward"
                    else {}
                ),
                f"{v4_spec['auth_key']}_sha256": v4_auth_config["sha256"],
            }
        )
    result_path = run_dir / "run_result.json"
    _write_json(result_path, result)
    curve_path = run_dir / "training_curve.csv"
    curve_row: dict[str, Any] = {
        "environment_interactions": 40 if wiring_only else 250_000,
        "training/total_loss": 1.0,
    }
    curve_rows = [curve_row]
    if forward_iteration_v4 and not wiring_only:
        length = 20.0
        episode_totals = {
            "episode/length": length,
            "episode/h4/v4_single_authority_dynamic6_exact": length,
            "episode/h4/v4_single_authority_dynamic6_max_abs_error": 0.0,
            "episode/h4/v4_single_authority_dynamic6_field_count": 6.0 * length,
            "episode/h4/v4_single_authority_dynamic6_field_count_exact": length,
            "episode/h4/v4_saved_dynamic6_substep_count": 10.0 * length,
            "episode/h4/v4_saved_dynamic6_field_count": 6.0 * length,
            "episode/h4/v4_saved_dynamic6_field_count_exact": length,
            "episode/h4/v4_saved_dynamic6_all_finite": length,
            "episode/h4/v4_telemetry_force_shape_valid": length,
            "episode/h4/v4_telemetry_force_all_finite": length,
            "episode/h4/v4_single_authority_violation": 0.0,
            "episode/h4/v4_single_authority_assertion_token": 0.0,
        }
        curve_rows = [
            {
                "environment_interactions": step,
                **episode_totals,
            }
            for step in h4pt.H4_FORWARD_V4_FULL_TRAINING_PROGRESS_INTERACTIONS
        ]
        curve_rows.append(curve_row)
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(
            dict.fromkeys(key for row in curve_rows for key in row)
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curve_rows)
    parent_sha = "2" * 64
    manifest = {
        "schema_version": 1,
        "status": "WIRING_PASS" if wiring_only else "COMPLETED",
        "hardware_deployment": "PROHIBITED",
        "run_name": run_name,
        "expert": expert,
        "activity": activity,
        "wiring_only": wiring_only,
        "requested_environment_interactions": 40 if wiring_only else 250_000,
        "source_and_teacher_hashes_pre": source_hashes,
        "source_and_teacher_hashes_post": source_hashes,
        "source_and_teacher_unchanged": True,
        "parent_checkpoint": {
            **({"kind": "V22_BRAX_CHECKPOINT"} if v3_spec or v4_spec else {}),
            "sha256_tree_pre": parent_sha,
            "sha256_tree_post": parent_sha,
            "unchanged": True,
        },
        "resolved_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "canonical_sha256": canonical_json_sha256(config),
        },
        "jax_devices": ["cuda:0"],
        "backend_resolution": backend,
        "xla_autotune_policy": xla,
        "debug_callback_preflight": callback,
        "pre_training_device_audits": pre_device_audits,
        "checkpoint_compatibility": checkpoint,
        "outputs": {
            "final_params": {"path": str(params), "sha256": params_sha},
            "result": {"path": str(result_path), "sha256": sha256_file(result_path)},
            "training_curve": {
                "path": str(curve_path),
                "sha256": sha256_file(curve_path),
            },
        },
    }
    if v3_spec is not None:
        manifest.update(
            {
                "training_contract_id": v3_spec["contract"],
                "authorized_iteration_v2_250k_contract_id": None,
                "authorized_iteration_v3_250k_contract_id": v3_spec["contract"],
                "qualification_use": "AUTHORIZED_250K_PILOT",
                "forward_iteration_v2": False,
                "reverse_iteration_v2": False,
                v3_spec["flag"]: True,
                v3_spec["auth_key"]: {
                    key: value
                    for key, value in v3_auth_config.items()
                    if key in {
                        "path",
                        "sha256",
                        "contract_id",
                        "legacy_reward_config_audit",
                    }
                },
                "parent_checkpoint": {
                    "kind": "V22_BRAX_CHECKPOINT",
                    "sha256_tree_pre": h4pt.PINNED_V22_PARENT_TREE_SHA256,
                    "sha256_tree_post": h4pt.PINNED_V22_PARENT_TREE_SHA256,
                    "unchanged": True,
                },
            }
        )
    if v4_spec is not None:
        assert v4_auth_config is not None
        manifest.update(
            {
                "training_contract_id": (
                    v4_spec["wiring_contract"]
                    if wiring_only
                    else v4_spec["contract"]
                ),
                "authorized_iteration_v2_250k_contract_id": None,
                "authorized_iteration_v3_250k_contract_id": None,
                "authorized_iteration_v4_250k_contract_id": v4_spec["contract"],
                "qualification_use": (
                    "WIRING_PREFLIGHT_ONLY_NOT_250K_QUALIFICATION"
                    if wiring_only
                    else "AUTHORIZED_250K_PILOT"
                ),
                "forward_iteration_v2": False,
                "reverse_iteration_v2": False,
                "forward_iteration_v3_touchdown_balance": False,
                "reverse_iteration_v3_no_target_imitation": False,
                "forward_iteration_v4_contact_event_validity_persistence": (
                    expert == "forward"
                ),
                "reverse_iteration_v4_residual_transfer_gain_024": (
                    expert == "reverse"
                ),
                "forward_v4_substep_contact": expert == "forward",
                **(
                    {
                        "forward_v4_source_semantic_preflight": (
                            source_semantic_preflight
                        ),
                        "forward_v4_single_authority_runtime_requirement": (
                            authority_requirement
                        ),
                        "forward_v4_single_authority_runtime": (
                            single_authority_runtime
                        ),
                        "forward_v4_single_authority_runtime_audit_mode": (
                            h4pt.H4_FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
                            if wiring_only
                            else h4pt.H4_FORWARD_V4_FULL_RUNTIME_AUDIT_MODE
                        ),
                    }
                    if expert == "forward"
                    else {}
                ),
                v4_spec["auth_key"]: {
                    key: value
                    for key, value in v4_auth_config.items()
                    if key
                    in {
                        "path",
                        "sha256",
                        "contract_id",
                        "legacy_reward_config_audit",
                    }
                },
                "parent_checkpoint": {
                    "kind": "V22_BRAX_CHECKPOINT",
                    "sha256_tree_pre": h4pt.PINNED_V22_PARENT_TREE_SHA256,
                    "sha256_tree_post": h4pt.PINNED_V22_PARENT_TREE_SHA256,
                    "unchanged": True,
                },
            }
        )
    manifest_path = run_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "params": params,
        "params_sha": params_sha,
        "manifest": manifest_path,
        "manifest_sha": sha256_file(manifest_path),
        "config": config_path,
        "trusted_run_root": trusted_run_root,
        "source_paths": {
            label: Path(record["path"]) for label, record in source_hashes.items()
        },
    }


def _forward_iteration_v5_bundle_files(tmp_path: Path) -> dict[str, Any]:
    """Promote the real v4 bundle fixture into an exact forward-v5 bundle."""

    files = _bundle_files(
        tmp_path,
        expert="forward",
        forward_iteration_v4=True,
    )
    authorization = _load_runner_v5_authorization("forward")
    spec = h4pt._iteration_v5_spec("forward")
    payload = authorization["payload"]
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    mode_keys = (
        "forward_iteration_v2",
        "reverse_iteration_v2",
        "forward_iteration_v3_touchdown_balance",
        "reverse_iteration_v3_no_target_imitation",
        "forward_iteration_v4_contact_event_validity_persistence",
        "reverse_iteration_v4_residual_transfer_gain_024",
        "forward_v5_contact_pulse_abort_scale_only",
        "reverse_iteration_v5_no_contact_imitation",
    )
    for artifact in (config, manifest, result):
        for mode in mode_keys:
            artifact[mode] = mode == spec["flag"]
        artifact["authorized_iteration_v2_250k_contract_id"] = None
        artifact["authorized_iteration_v3_250k_contract_id"] = None
        artifact["authorized_iteration_v4_250k_contract_id"] = None
        artifact["authorized_iteration_v5_250k_contract_id"] = spec["contract"]
        artifact["training_contract_id"] = spec["contract"]
        artifact["forward_v4_substep_contact"] = True

    config.pop("forward_iteration_v4_contact_event_validity_persistence_authorization")
    manifest.pop(
        "forward_iteration_v4_contact_event_validity_persistence_authorization"
    )
    result.pop(
        "forward_iteration_v4_contact_event_validity_persistence_authorization_sha256"
    )
    config["reward_scales"] = copy.deepcopy(
        payload["reward_contract"]["exact_scales"]
    )
    config[spec["auth_key"]] = {
        "path": str(authorization["path"]),
        "sha256": authorization["sha256"],
        "contract_id": authorization["contract_id"],
        "status": payload["status"],
        "semantic_audit": copy.deepcopy(authorization["semantic_audit"]),
        "bound_causal_inputs": copy.deepcopy(
            authorization["bound_causal_inputs"]
        ),
        "bound_historical_v4_sources": copy.deepcopy(
            authorization["bound_historical_v4_sources"]
        ),
        "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
        "adoption_release_hardware": "PROHIBITED",
    }
    manifest[spec["auth_key"]] = {
        "path": str(authorization["path"]),
        "sha256": authorization["sha256"],
        "contract_id": authorization["contract_id"],
        "bound_historical_v4_sources": copy.deepcopy(
            authorization["bound_historical_v4_sources"]
        ),
    }
    result[f"{spec['auth_key']}_sha256"] = authorization["sha256"]

    source_hashes = {
        label: copy.deepcopy(record)
        for label, record in manifest["source_and_teacher_hashes_pre"].items()
        if not label.startswith("forward_iteration_v4_")
    }
    source_hashes[spec["auth_label"]] = {
        "path": str(authorization["path"]),
        "sha256": authorization["sha256"],
    }
    for source_label in spec["causal_labels"].values():
        bound_key = source_label.removeprefix(spec["prefix"])
        source_hashes[source_label] = copy.deepcopy(
            authorization["bound_causal_inputs"][bound_key]
        )
    for label, relative in h4pt.H4_ITERATION_V4_CAUSAL_SOURCE_PATHS.items():
        source_path = (EXP_ROOT / relative).resolve()
        source_hashes[f"{spec['prefix']}current_source_{label}"] = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        }
    manifest["source_and_teacher_hashes_pre"] = copy.deepcopy(source_hashes)
    manifest["source_and_teacher_hashes_post"] = copy.deepcopy(source_hashes)
    _write_json(files["config"], config)
    _write_json(result_path, result)
    manifest["resolved_config"]["sha256"] = sha256_file(files["config"])
    manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(
        config
    )
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    _write_json(files["manifest"], manifest)
    files["manifest_sha"] = sha256_file(files["manifest"])
    files["source_paths"] = {
        label: Path(record["path"]) for label, record in source_hashes.items()
    }
    return files


_ITERATION_V6_LEGACY_FILENAME_CAUSAL_LABELS = {
    "forward": {
        "iteration_v4_final_params": "forward_iteration_v6_v4_final_params.pkl",
        "iteration_v4_manifest": "forward_iteration_v6_v4_run_manifest.json",
        "iteration_v4_integrated_strict_evaluation": (
            "forward_iteration_v6_v4_h4_integrated_strict_3x6s_v1.json"
        ),
        "rejected_iteration_v5_final_params": (
            "forward_iteration_v6_rejected_v5_final_params.pkl"
        ),
        "rejected_iteration_v5_manifest": (
            "forward_iteration_v6_rejected_v5_run_manifest.json"
        ),
        "rejected_iteration_v5_integrated_strict_evaluation": (
            "forward_iteration_v6_rejected_v5_h4_integrated_strict_3x6s_v1.json"
        ),
    },
    "reverse": {
        "iteration_v3_integrated_strict_evaluation": (
            "reverse_iteration_v6_v3_h4_integrated_strict_3x6s_v1.json"
        ),
        "rejected_iteration_v4_integrated_strict_evaluation": (
            "reverse_iteration_v6_rejected_v4_h4_integrated_strict_3x6s_v1.json"
        ),
        "rejected_iteration_v4_diagnostic_adapter": (
            "reverse_iteration_v6_rejected_v4_diagnostic_adapter.py"
        ),
        "rejected_iteration_v4_diagnostic_adapter_authorization": (
            "reverse_iteration_v6_rejected_v4_diagnostic_adapter_authorization.json"
        ),
        "rejected_iteration_v5_final_params": (
            "reverse_iteration_v6_rejected_v5_final_params.pkl"
        ),
        "rejected_iteration_v5_manifest": (
            "reverse_iteration_v6_rejected_v5_run_manifest.json"
        ),
        "rejected_iteration_v5_integrated_strict_evaluation": (
            "reverse_iteration_v6_rejected_v5_h4_integrated_strict_3x6s_v1.json"
        ),
        "selected_reverse_teacher": (
            "reverse_iteration_v6_selected_reverse_teacher.json"
        ),
    },
}


def _iteration_v6_source_fixture(expert: str) -> dict[str, Any]:
    spec = h4pt._iteration_v6_spec(expert)
    authorization_path = EXP_ROOT / "artifacts" / spec["auth_filename"]
    authorization = load_json_strict(authorization_path)
    causal = authorization["causal_inputs"]
    source_hashes: dict[str, dict[str, str]] = {
        spec["auth_label"]: {
            "path": str(authorization_path.resolve()),
            "sha256": sha256_file(authorization_path),
        }
    }
    bound_causal: dict[str, dict[str, str]] = {}
    runner_prefix = f"{expert}_iteration_v6_"
    assert spec["prefix"] == runner_prefix
    for causal_key in spec["causal_labels"]:
        source_label = f"{runner_prefix}{causal_key}"
        assert spec["causal_labels"][causal_key] == source_label
        causal_path = h4pt._iteration_v6_causal_path(
            experiment_root=EXP_ROOT,
            causal=causal,
            causal_key=causal_key,
            root_key=spec["causal_roots"].get(causal_key),
        )
        record = {"path": str(causal_path), "sha256": sha256_file(causal_path)}
        source_hashes[source_label] = record
        bound_causal[causal_key] = copy.deepcopy(record)
    runner_v5_manifest_label = (
        f"{runner_prefix}rejected_iteration_v5_manifest"
    )
    assert spec["v5_manifest_label"] == runner_v5_manifest_label
    v5_manifest = load_json_strict(
        Path(source_hashes[runner_v5_manifest_label]["path"])
    )
    bound_historical: dict[str, dict[str, Any]] = {}
    for label, relative in h4pt.H4_ITERATION_V4_CAUSAL_SOURCE_PATHS.items():
        manifest_label = f"{spec['v5_source_prefix']}{label}"
        pre = v5_manifest["source_and_teacher_hashes_pre"][manifest_label]
        post = v5_manifest["source_and_teacher_hashes_post"][manifest_label]
        bound_historical[label] = {
            "path": relative,
            "sha256": pre["sha256"],
            "manifest_pre": copy.deepcopy(pre),
            "manifest_post": copy.deepcopy(post),
        }
        current_path = (EXP_ROOT / relative).resolve()
        source_hashes[f"{spec['prefix']}current_source_{label}"] = {
            "path": str(current_path),
            "sha256": sha256_file(current_path),
        }
    mode_keys = (
        "forward_iteration_v2",
        "reverse_iteration_v2",
        "forward_iteration_v3_touchdown_balance",
        "reverse_iteration_v3_no_target_imitation",
        "forward_iteration_v4_contact_event_validity_persistence",
        "reverse_iteration_v4_residual_transfer_gain_024",
        "forward_v5_contact_pulse_abort_scale_only",
        "reverse_iteration_v5_no_contact_imitation",
        "forward_iteration_v6_contact_abort_island_only",
        "reverse_iteration_v6_absolute_full_leg_targets",
    )
    config: dict[str, Any] = {
        key: key == spec["flag"] for key in mode_keys
    }
    config.update(
        {
            "training_contract_id": spec["contract"],
            "authorized_iteration_v2_250k_contract_id": None,
            "authorized_iteration_v3_250k_contract_id": None,
            "authorized_iteration_v4_250k_contract_id": None,
            "authorized_iteration_v5_250k_contract_id": None,
            "authorized_iteration_v6_250k_contract_id": spec["contract"],
            "initialization_source": "V22_BRAX_CHECKPOINT",
            "trusted_h4_parent": None,
            "pinned_v22_parent_tree_sha256": h4pt.PINNED_V22_PARENT_TREE_SHA256,
            "reset_noise_multiplier": 1.0,
            "forward_v4_substep_contact": expert == "forward",
            "backward_residual_scale": 0.0 if expert == "reverse" else 0.12,
            "reward_scales": copy.deepcopy(
                authorization["reward_contract"]["exact_scales"]
            ),
            "wiring_only": False,
            "iteration_v6_core_source": copy.deepcopy(
                source_hashes[
                    f"{spec['prefix']}current_source_h4_training_alignment"
                ]
            ),
        }
    )
    config[spec["auth_key"]] = {
        "path": str(authorization_path.resolve()),
        "sha256": sha256_file(authorization_path),
        "contract_id": spec["contract"],
        "status": "AUTHORIZED_SIMULATION_250K_ONLY",
        "semantic_audit": {
            key: True
            for key in h4pt._iteration_v6_expected_semantic_audit_keys(expert)
        },
        "bound_causal_inputs": bound_causal,
        "bound_historical_v5_sources": bound_historical,
        "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
        "adoption_release_hardware": "PROHIBITED",
    }
    if expert == "forward":
        config["reward_routing_contract"] = copy.deepcopy(
            authorization["reward_routing_contract"]
        )
    else:
        config["action_parameterization_contract"] = (
            copy.deepcopy(authorization["action_parameterization_contract"])
        )
        config["teacher_timing_contract"] = copy.deepcopy(
            authorization["teacher_timing_contract"]
        )
        expected_legacy = {
            "target_imitation": 0.0,
            "contact_imitation": 0.0,
            "tracking_sigma": 0.01,
            "backward_residual_scale": 0.0,
        }
        config[spec["auth_key"]].update(
            {
                "legacy_reward_config_audit": {
                    "expected": copy.deepcopy(expected_legacy),
                    "per_environment": {
                        "train": copy.deepcopy(expected_legacy),
                        "eval": copy.deepcopy(expected_legacy),
                    },
                    "passed": True,
                },
                "h4_parent_checkpoint_allowed": False,
                "v4_gain_inherited": False,
                "v5_parent_checkpoint_inherited": False,
            }
        )
    return {
        "spec": spec,
        "authorization": authorization,
        "config": config,
        "source_hashes": source_hashes,
    }


def _forward_iteration_v6_bundle_files(tmp_path: Path) -> dict[str, Any]:
    files = _bundle_files(
        tmp_path,
        expert="forward",
        forward_iteration_v4=True,
    )
    fixture = _iteration_v6_source_fixture("forward")
    spec = fixture["spec"]
    v6_config = fixture["config"]
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    mode_keys = tuple(
        key
        for key in v6_config
        if key.startswith("forward_iteration_")
        or key.startswith("reverse_iteration_")
        or key == "forward_v5_contact_pulse_abort_scale_only"
    )
    mode_keys = tuple(
        key
        for key in mode_keys
        if isinstance(v6_config[key], bool)
    )
    for artifact in (config, manifest, result):
        for mode in mode_keys:
            artifact[mode] = mode == spec["flag"]
        for version in (2, 3, 4, 5):
            artifact[f"authorized_iteration_v{version}_250k_contract_id"] = None
        artifact["authorized_iteration_v6_250k_contract_id"] = spec["contract"]
        artifact["training_contract_id"] = spec["contract"]
        artifact["forward_v4_substep_contact"] = True

    old_auth = "forward_iteration_v4_contact_event_validity_persistence_authorization"
    config.pop(old_auth)
    manifest.pop(old_auth)
    result.pop(f"{old_auth}_sha256")
    config["reward_scales"] = copy.deepcopy(v6_config["reward_scales"])
    config["reward_routing_contract"] = copy.deepcopy(
        v6_config["reward_routing_contract"]
    )
    manifest["reward_routing_contract"] = copy.deepcopy(
        config["reward_routing_contract"]
    )
    result["reward_routing_contract"] = copy.deepcopy(
        config["reward_routing_contract"]
    )
    for artifact in (config, manifest, result):
        artifact["iteration_v6_core_source"] = copy.deepcopy(
            v6_config["iteration_v6_core_source"]
        )
    config[spec["auth_key"]] = copy.deepcopy(v6_config[spec["auth_key"]])
    manifest[spec["auth_key"]] = {
        "path": config[spec["auth_key"]]["path"],
        "sha256": spec["auth_sha"],
        "contract_id": spec["contract"],
        "bound_historical_v5_sources": copy.deepcopy(
            config[spec["auth_key"]]["bound_historical_v5_sources"]
        ),
    }
    result[f"{spec['auth_key']}_sha256"] = spec["auth_sha"]
    requirement_key = "forward_iteration_v6_reward_routing_runtime_requirement"
    runtime_key = "forward_iteration_v6_reward_routing_runtime"
    requirement = h4pt._iteration_v6_runtime_requirement("forward", "COMPLETED")
    runtime = {
        "audit_mode": h4pt.H4_FORWARD_ITERATION_V6_RUNTIME_AUDIT_MODES[
            "COMPLETED"
        ],
        "expert": "forward",
        "observed_episode_metric_rows": 5,
        "episode_metric_rows_exact_if_observed": True,
        "per_step_compiled_fail_closed_assertion_bound": True,
        "completed_environment_interactions": h4pt.H4_PILOT_INTERACTIONS,
        "completed_training_steps": h4pt.H4_PILOT_TRAINING_STEPS,
        "completed_optimizer_updates": h4pt.H4_PILOT_OPTIMIZER_UPDATES,
        "progress_reached_final_interaction": True,
        "final_params_all_finite": True,
        "final_metrics_all_finite": True,
        "source_and_teacher_unchanged": True,
        "passed": True,
    }
    config[requirement_key] = copy.deepcopy(requirement)
    for artifact in (manifest, result):
        artifact[requirement_key] = copy.deepcopy(requirement)
        artifact[runtime_key] = copy.deepcopy(runtime)
        artifact["iteration_v6_artifact_cross_binding"] = copy.deepcopy(
            h4pt.H4_ITERATION_V6_ARTIFACT_CROSS_BINDING
        )

    curve_path = Path(manifest["outputs"]["training_curve"]["path"])
    with curve_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    added_fields = [
        f"episode/h4/{key.removeprefix('h4_')}"
        for key in h4pt.H4_FORWARD_ITERATION_V6_RUNTIME_INFO_KEYS
    ]
    fieldnames = list(dict.fromkeys([*fieldnames, *added_fields]))
    for row in rows:
        raw_length = row.get("episode/length")
        if raw_length in (None, ""):
            continue
        length = float(raw_length)
        row.update(
            {
                added_fields[0]: str(length),
                added_fields[1]: "0.0",
                added_fields[2]: "0.0",
                added_fields[3]: "0.0",
                added_fields[4]: str(-length),
                added_fields[5]: "0.0",
                added_fields[6]: "0.0",
            }
        )
    with curve_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    source_hashes = {
        label: copy.deepcopy(record)
        for label, record in manifest["source_and_teacher_hashes_pre"].items()
        if not label.startswith("forward_iteration_v4_")
    }
    source_hashes.update(copy.deepcopy(fixture["source_hashes"]))
    manifest["source_and_teacher_hashes_pre"] = copy.deepcopy(source_hashes)
    manifest["source_and_teacher_hashes_post"] = copy.deepcopy(source_hashes)
    for artifact in (config, manifest):
        artifact["forward_iteration_v2_authorization"] = None
        artifact["reverse_iteration_v2_authorization"] = None
    result["forward_iteration_v2_authorization_sha256"] = None
    result["reverse_iteration_v2_authorization_sha256"] = None
    _write_json(files["config"], config)
    _write_json(result_path, result)
    manifest["resolved_config"]["sha256"] = sha256_file(files["config"])
    manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(config)
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    manifest["outputs"]["training_curve"]["sha256"] = sha256_file(curve_path)
    _write_json(files["manifest"], manifest)
    files["manifest_sha"] = sha256_file(files["manifest"])
    files["source_paths"] = {
        label: Path(record["path"]) for label, record in source_hashes.items()
    }
    return files


def _reverse_iteration_v6_bundle_files(tmp_path: Path) -> dict[str, Any]:
    files = _bundle_files(
        tmp_path,
        expert="reverse",
        reverse_iteration_v4=True,
    )
    fixture = _iteration_v6_source_fixture("reverse")
    spec = fixture["spec"]
    v6_config = fixture["config"]
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    mode_keys = (
        "forward_iteration_v2",
        "reverse_iteration_v2",
        "forward_iteration_v3_touchdown_balance",
        "reverse_iteration_v3_no_target_imitation",
        "forward_iteration_v4_contact_event_validity_persistence",
        "reverse_iteration_v4_residual_transfer_gain_024",
        "forward_v5_contact_pulse_abort_scale_only",
        "reverse_iteration_v5_no_contact_imitation",
        "forward_iteration_v6_contact_abort_island_only",
        "reverse_iteration_v6_absolute_full_leg_targets",
    )
    for artifact in (config, manifest, result):
        for mode in mode_keys:
            artifact[mode] = mode == spec["flag"]
        for version in (2, 3, 4, 5):
            artifact[f"authorized_iteration_v{version}_250k_contract_id"] = None
        artifact["authorized_iteration_v6_250k_contract_id"] = spec["contract"]
        artifact["training_contract_id"] = spec["contract"]
        artifact["forward_v4_substep_contact"] = False
        artifact["backward_residual_scale"] = 0.0

    old_auth = "reverse_iteration_v4_residual_transfer_gain_024_authorization"
    config.pop(old_auth)
    manifest.pop(old_auth)
    result.pop(f"{old_auth}_sha256")
    config["reward_scales"] = copy.deepcopy(v6_config["reward_scales"])
    for contract_key in (
        "action_parameterization_contract",
        "teacher_timing_contract",
    ):
        config[contract_key] = copy.deepcopy(v6_config[contract_key])
        manifest[contract_key] = copy.deepcopy(v6_config[contract_key])
        result[contract_key] = copy.deepcopy(v6_config[contract_key])
    for artifact in (config, manifest, result):
        artifact["iteration_v6_core_source"] = copy.deepcopy(
            v6_config["iteration_v6_core_source"]
        )
    config[spec["auth_key"]] = copy.deepcopy(v6_config[spec["auth_key"]])
    config_auth = config[spec["auth_key"]]
    manifest[spec["auth_key"]] = {
        "path": config_auth["path"],
        "sha256": spec["auth_sha"],
        "contract_id": spec["contract"],
        "bound_historical_v5_sources": copy.deepcopy(
            config_auth["bound_historical_v5_sources"]
        ),
        "legacy_reward_config_audit": copy.deepcopy(
            config_auth["legacy_reward_config_audit"]
        ),
        "h4_parent_checkpoint_allowed": False,
        "v4_gain_inherited": False,
        "v5_parent_checkpoint_inherited": False,
    }
    result.update(
        {
            f"{spec['auth_key']}_sha256": spec["auth_sha"],
            "legacy_reward_config_audit": copy.deepcopy(
                config_auth["legacy_reward_config_audit"]
            ),
            "teacher_target_contribution_zero": True,
            "h4_parent_checkpoint_allowed": False,
            "v4_gain_inherited": False,
            "v5_parent_checkpoint_inherited": False,
        }
    )
    requirement_key = "reverse_iteration_v6_decoder_runtime_requirement"
    runtime_key = "reverse_iteration_v6_decoder_runtime"
    requirement = h4pt._iteration_v6_runtime_requirement("reverse", "COMPLETED")
    runtime = {
        "audit_mode": h4pt.H4_REVERSE_ITERATION_V6_RUNTIME_AUDIT_MODES[
            "COMPLETED"
        ],
        "expert": "reverse",
        "observed_episode_metric_rows": 5,
        "episode_metric_rows_exact_if_observed": True,
        "per_step_compiled_fail_closed_assertion_bound": True,
        "completed_environment_interactions": h4pt.H4_PILOT_INTERACTIONS,
        "completed_training_steps": h4pt.H4_PILOT_TRAINING_STEPS,
        "completed_optimizer_updates": h4pt.H4_PILOT_OPTIMIZER_UPDATES,
        "progress_reached_final_interaction": True,
        "final_params_all_finite": True,
        "final_metrics_all_finite": True,
        "source_and_teacher_unchanged": True,
        "passed": True,
    }
    config[requirement_key] = copy.deepcopy(requirement)
    for artifact in (manifest, result):
        artifact[requirement_key] = copy.deepcopy(requirement)
        artifact[runtime_key] = copy.deepcopy(runtime)
        artifact["iteration_v6_artifact_cross_binding"] = copy.deepcopy(
            h4pt.H4_ITERATION_V6_ARTIFACT_CROSS_BINDING
        )

    curve_path = Path(manifest["outputs"]["training_curve"]["path"])
    length = 20.0
    prefix = "episode/h4/v6_reverse_"
    episode_totals = {
        "episode/length": length,
        f"{prefix}decoder_exact": length,
        f"{prefix}decoder_max_abs_error": 0.0,
        f"{prefix}decoder_leg_count": float(np.nextafter(10.0 * length, 0.0)),
        f"{prefix}decoder_leg_count_exact": length,
        f"{prefix}decoder_head_zero_exact": length,
        f"{prefix}teacher_target_contribution_zero_exact": length,
        f"{prefix}residual_authority_scale": 0.0,
        f"{prefix}decoder_all_finite": length,
        f"{prefix}decoder_margin_saturation_count": 1.0,
        f"{prefix}decoder_action_clip_count": 2.0,
        f"{prefix}decoder_guard_lag_max_rad": 0.125,
        f"{prefix}precomposer_call_count": 19.5,
        f"{prefix}precomposer_call_count_exact": length,
        f"{prefix}final_guard_call_count": 20.5,
        f"{prefix}final_guard_call_count_exact": length,
        f"{prefix}decoder_violation": 0.0,
        f"{prefix}decoder_assertion_token": 0.0,
    }
    curve_rows = [
        {"environment_interactions": step, **episode_totals}
        for step in h4pt.H4_FORWARD_V4_FULL_TRAINING_PROGRESS_INTERACTIONS
    ]
    curve_rows.append(
        {"environment_interactions": 250_000, "training/total_loss": 1.0}
    )
    with curve_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(dict.fromkeys(key for row in curve_rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve_rows)

    source_hashes = {
        label: copy.deepcopy(record)
        for label, record in manifest["source_and_teacher_hashes_pre"].items()
        if not label.startswith("reverse_iteration_v4_")
    }
    source_hashes.update(copy.deepcopy(fixture["source_hashes"]))
    manifest["source_and_teacher_hashes_pre"] = copy.deepcopy(source_hashes)
    manifest["source_and_teacher_hashes_post"] = copy.deepcopy(source_hashes)
    for artifact in (config, manifest):
        artifact["forward_iteration_v2_authorization"] = None
        artifact["reverse_iteration_v2_authorization"] = None
    result["forward_iteration_v2_authorization_sha256"] = None
    result["reverse_iteration_v2_authorization_sha256"] = None
    _write_json(files["config"], config)
    _write_json(result_path, result)
    manifest["resolved_config"]["sha256"] = sha256_file(files["config"])
    manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(
        config
    )
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    manifest["outputs"]["training_curve"]["sha256"] = sha256_file(curve_path)
    _write_json(files["manifest"], manifest)
    files["manifest_sha"] = sha256_file(files["manifest"])
    files["source_paths"] = {
        label: Path(record["path"]) for label, record in source_hashes.items()
    }
    return files


def _rewrite_bundle_artifacts(
    files: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    manifest: dict[str, Any],
    result: Mapping[str, Any],
) -> None:
    result_path = Path(manifest["outputs"]["result"]["path"])
    _write_json(Path(files["config"]), config)
    _write_json(result_path, result)
    manifest["resolved_config"]["sha256"] = sha256_file(Path(files["config"]))
    manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(
        config
    )
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    _write_json(Path(files["manifest"]), manifest)


_ITERATION_V6_LOCATION_BUILDERS = (
    ("forward", _forward_iteration_v6_bundle_files),
    ("reverse", _reverse_iteration_v6_bundle_files),
)


def _iteration_v6_expected_artifact_locations_for_test(
    expert: str,
) -> dict[str, frozenset[str]]:
    spec = h4pt._iteration_v6_spec(expert)
    auth_key = spec["auth_key"]
    runtime_key = (
        "forward_iteration_v6_reward_routing_runtime"
        if expert == "forward"
        else "reverse_iteration_v6_decoder_runtime"
    )
    expected = {
        auth_key: frozenset({"config", "manifest"}),
        f"{auth_key}_sha256": frozenset({"result"}),
        f"{runtime_key}_requirement": frozenset(
            {"config", "manifest", "result"}
        ),
        runtime_key: frozenset({"manifest", "result"}),
        "iteration_v6_core_source": frozenset(
            {"config", "manifest", "result"}
        ),
        "iteration_v6_artifact_cross_binding": frozenset(
            {"manifest", "result"}
        ),
    }
    if expert == "forward":
        expected["legacy_reward_config_audit"] = frozenset()
        for key in (
            "reward_routing_contract",
            "forward_v4_source_semantic_preflight",
            "forward_v4_single_authority_runtime_requirement",
            "forward_v4_single_authority_runtime_audit_mode",
        ):
            expected[key] = frozenset({"config", "manifest", "result"})
        expected["forward_v4_single_authority_runtime"] = frozenset(
            {"manifest", "result"}
        )
    else:
        for key in (
            "action_parameterization_contract",
            "teacher_timing_contract",
            "backward_residual_scale",
        ):
            expected[key] = frozenset({"config", "manifest", "result"})
        for key in (
            "legacy_reward_config_audit",
            "h4_parent_checkpoint_allowed",
            "v4_gain_inherited",
            "v5_parent_checkpoint_inherited",
            "teacher_target_contribution_zero",
        ):
            expected[key] = frozenset({"result"})
    return expected


def _load_iteration_v6_artifacts_for_test(
    files: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json_strict(Path(files["config"]))
    manifest = load_json_strict(Path(files["manifest"]))
    result = load_json_strict(Path(manifest["outputs"]["result"]["path"]))
    return config, manifest, result


def _assert_iteration_v6_location_rejects_before_pickle(
    files: Mapping[str, Any],
) -> None:
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="artifact location closure"):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(Path(files["manifest"])),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(
            bundle, files["source_paths"]
        )
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


def test_fixed_seed_and_width_contract_is_exact() -> None:
    assert H4_ACTOR_OBSERVATION_WIDTH == 116
    assert H4_CRITIC_OBSERVATION_WIDTH == 227
    assert H4_ACTION_WIDTH == 14
    assert H4_STRICT_GAIT_SAMPLES == H4_STRICT_PHYSICS_SUBSTEPS + 1 == 3001
    assert H4_STRICT_SEEDS == {
        "forward": (20260809, 20261809, 20262809),
        "reverse": (20260810, 20265810, 20271810),
    }


def test_strict_json_rejects_duplicates_nonfinite_and_overwrite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_json_strict(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_json_strict(nonfinite)
    output = tmp_path / "immutable.json"
    write_new_json(output, {"finite": 1.0})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_new_json(output, {"finite": 2.0})
    assert json_native({"tuple": (np.float32(1.0), np.asarray([2, 3]))}) == {
        "tuple": [1.0, [2, 3]]
    }
    with pytest.raises(ValueError, match="non-finite"):
        json_native((np.float32(np.nan),))


def test_trusted_bundle_checks_all_hashes_before_pickle_restore(tmp_path: Path) -> None:
    files = _bundle_files(tmp_path)
    bundle = validate_trusted_h4_bundle(
        params_path=files["params"],
        manifest_path=files["manifest"],
        expected_params_sha256=files["params_sha"],
        expected_manifest_sha256=files["manifest_sha"],
        trusted_run_root=files["trusted_run_root"],
    )
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="source closure"):
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


def test_arbitrary_pickle_and_incomplete_source_closure_fail_before_load(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(TypeError, match="validated bundle"):
        load_trusted_h4_params(object(), pickle_loader=loader)  # type: ignore[arg-type]
    assert calls == []

    arbitrary = tmp_path / "arbitrary.pkl"
    arbitrary.write_bytes(b"not trusted")
    with pytest.raises(ValueError, match="exact runner output basenames"):
        validate_trusted_h4_bundle(
            params_path=arbitrary,
            manifest_path=tmp_path / "missing.json",
            expected_params_sha256=sha256_file(arbitrary),
            expected_manifest_sha256="0" * 64,
            trusted_run_root=tmp_path,
        )
    assert calls == []

    files = _bundle_files(tmp_path / "closed")
    bundle = validate_trusted_h4_bundle(
        params_path=files["params"],
        manifest_path=files["manifest"],
        expected_params_sha256=files["params_sha"],
        expected_manifest_sha256=files["manifest_sha"],
        trusted_run_root=files["trusted_run_root"],
    )
    missing = dict(files["source_paths"])
    missing.pop("legacy_trainer")
    with pytest.raises(ValueError, match="label closure mismatch"):
        validate_h4_training_source_closure(bundle, missing)
    extra = {**files["source_paths"], "unexpected": Path(__file__)}
    with pytest.raises(ValueError, match="label closure mismatch"):
        validate_h4_training_source_closure(bundle, extra)
    redirected = dict(files["source_paths"])
    redirected["legacy_trainer"] = Path(__file__)
    with pytest.raises(ValueError, match="path mismatch"):
        validate_h4_training_source_closure(bundle, redirected)
    bad_records = copy.deepcopy(dict(bundle.source_hashes))
    bad_records["legacy_trainer"]["sha256"] = "0" * 64
    bad_bundle = replace(
        bundle,
        source_hashes=bad_records,
        source_hashes_canonical_sha256=canonical_json_sha256(bad_records),
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_h4_training_source_closure(bad_bundle, files["source_paths"])
    assert calls == []
    bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
    params, audit = load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == ["pickle_opened"]
    assert audit["pickle_opened_only_after_manifest_validation"] is True
    assert validate_h4_params(params)["passed"] is True

    files["params"].write_bytes(b"tampered")
    calls.clear()
    with pytest.raises(ValueError, match="params changed"):
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


def test_bundle_rejects_params_manifest_and_config_drift(tmp_path: Path) -> None:
    files = _bundle_files(tmp_path)
    with pytest.raises(ValueError, match="params SHA256 mismatch"):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
                expected_params_sha256="0" * 64,
                expected_manifest_sha256=files["manifest_sha"],
                trusted_run_root=files["trusted_run_root"],
        )
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
                expected_params_sha256=files["params_sha"],
                expected_manifest_sha256="0" * 64,
                trusted_run_root=files["trusted_run_root"],
        )
    files["config"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="resolved_config SHA256 mismatch"):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=files["manifest_sha"],
            trusted_run_root=files["trusted_run_root"],
        )


def test_bundle_rejects_gpu_stage_and_curve_provenance_drift(
    tmp_path: Path,
) -> None:
    def validate(files: dict[str, Any]) -> Any:
        return validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )

    gpu_files = _bundle_files(tmp_path / "gpu")
    gpu_config = load_json_strict(gpu_files["config"])
    gpu_config["xla_autotune_policy"]["xla_flags_effective"] = (
        "--xla_gpu_autotune_level=0"
    )
    _write_json(gpu_files["config"], gpu_config)
    gpu_manifest = load_json_strict(gpu_files["manifest"])
    gpu_manifest["resolved_config"]["sha256"] = sha256_file(gpu_files["config"])
    gpu_manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(
        gpu_config
    )
    _write_json(gpu_files["manifest"], gpu_manifest)
    with pytest.raises(ValueError, match="GPU correctness-checked"):
        validate(gpu_files)

    stage_files = _bundle_files(tmp_path / "stage")
    stage_manifest = load_json_strict(stage_files["manifest"])
    stage_result_path = Path(stage_manifest["outputs"]["result"]["path"])
    stage_result = load_json_strict(stage_result_path)
    stage_result["optimizer_updates"] = 399
    _write_json(stage_result_path, stage_result)
    stage_manifest["outputs"]["result"]["sha256"] = sha256_file(stage_result_path)
    _write_json(stage_files["manifest"], stage_manifest)
    with pytest.raises(ValueError, match="run result/manifest binding"):
        validate(stage_files)

    curve_files = _bundle_files(tmp_path / "curve")
    curve_manifest = load_json_strict(curve_files["manifest"])
    curve_path = Path(curve_manifest["outputs"]["training_curve"]["path"])
    curve_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outputs.training_curve SHA256 mismatch"):
        validate(curve_files)


def test_reverse_bundle_requires_both_pinned_composition_hashes(tmp_path: Path) -> None:
    files = _bundle_files(tmp_path, expert="reverse")
    bundle = validate_trusted_h4_bundle(
        params_path=files["params"],
        manifest_path=files["manifest"],
        expected_params_sha256=files["params_sha"],
        expected_manifest_sha256=files["manifest_sha"],
        trusted_run_root=files["trusted_run_root"],
    )
    assert (
        bundle.source_hashes["selected_reverse_teacher"]["sha256"]
        == PINNED_SELECTED_REVERSE_TEACHER_SHA256
    )
    assert (
        bundle.source_hashes["reverse_composition_authorization"]["sha256"]
        == PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
    )


def test_forward_iteration_v2_reconstructs_all_four_causal_sources(
    tmp_path: Path,
) -> None:
    files = _bundle_files(tmp_path, forward_iteration_v2=True)
    bundle = validate_trusted_h4_bundle(
        params_path=files["params"],
        manifest_path=files["manifest"],
        expected_params_sha256=files["params_sha"],
        expected_manifest_sha256=files["manifest_sha"],
        trusted_run_root=files["trusted_run_root"],
    )
    iteration_v2_paths = h4pt._validated_forward_iteration_v2_source_paths(
        config=bundle.config,
        source_hashes=bundle.source_hashes,
    )
    assert set(iteration_v2_paths) == set(
        h4pt.H4_FORWARD_ITERATION_V2_SOURCE_LABELS
    )
    assert all(path.is_file() for path in iteration_v2_paths.values())
    closed = validate_h4_training_source_closure(bundle, files["source_paths"])
    assert closed.source_closure_audit is not None
    assert closed.source_closure_audit["passed"] is True


@pytest.mark.parametrize(
    "drift",
    (
        "missing_label",
        "unknown_label",
        "renamed_label",
        "bound_label",
        "config_path",
        "manifest_path",
        "config_sha",
        "manifest_sha",
    ),
)
def test_forward_iteration_v2_causal_source_drift_fails_closed(
    tmp_path: Path,
    drift: str,
) -> None:
    files = _bundle_files(tmp_path / drift, forward_iteration_v2=True)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    source_hashes = copy.deepcopy(manifest["source_and_teacher_hashes_post"])
    params_label = "forward_iteration_v2_failed_candidate_params"
    bound = config["forward_iteration_v2_authorization"]["bound_causal_inputs"]

    if drift == "missing_label":
        del source_hashes[params_label]
    elif drift == "unknown_label":
        source_hashes["forward_iteration_v2_unexpected"] = copy.deepcopy(
            source_hashes["forward_iteration_v2_authorization"]
        )
    elif drift == "renamed_label":
        source_hashes["forward_iteration_v2_failed_candidate_parameter"] = (
            source_hashes.pop(params_label)
        )
    elif drift == "bound_label":
        bound["failed_candidate_parameter"] = bound.pop(
            "failed_candidate_params"
        )
    elif drift == "config_path":
        bound["failed_candidate_params"]["path"] = bound[
            "failed_candidate_manifest"
        ]["path"]
    elif drift == "manifest_path":
        source_hashes[params_label]["path"] = source_hashes[
            "forward_iteration_v2_failed_candidate_manifest"
        ]["path"]
    elif drift == "config_sha":
        bound["failed_candidate_params"]["sha256"] = "0" * 64
    elif drift == "manifest_sha":
        source_hashes[params_label]["sha256"] = "0" * 64
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(drift)

    _write_json(files["config"], config)
    manifest["resolved_config"]["sha256"] = sha256_file(files["config"])
    manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(
        config
    )
    manifest["source_and_teacher_hashes_pre"] = copy.deepcopy(source_hashes)
    manifest["source_and_teacher_hashes_post"] = copy.deepcopy(source_hashes)
    _write_json(files["manifest"], manifest)
    with pytest.raises(ValueError, match="forward iteration-v2"):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )


def test_reverse_iteration_v2_reconstructs_all_four_causal_sources(
    tmp_path: Path,
) -> None:
    files = _bundle_files(
        tmp_path,
        expert="reverse",
        reverse_iteration_v2=True,
    )
    bundle = validate_trusted_h4_bundle(
        params_path=files["params"],
        manifest_path=files["manifest"],
        expected_params_sha256=files["params_sha"],
        expected_manifest_sha256=files["manifest_sha"],
        trusted_run_root=files["trusted_run_root"],
    )
    iteration_v2_paths = h4pt._validated_reverse_iteration_v2_source_paths(
        config=bundle.config,
        source_hashes=bundle.source_hashes,
    )
    assert set(iteration_v2_paths) == set(
        h4pt.H4_REVERSE_ITERATION_V2_SOURCE_LABELS
    )
    assert all(path.is_file() for path in iteration_v2_paths.values())
    closed = validate_h4_training_source_closure(bundle, files["source_paths"])
    assert closed.source_closure_audit is not None
    assert closed.source_closure_audit["passed"] is True


@pytest.mark.parametrize(
    "drift",
    (
        "missing_label",
        "unknown_label",
        "renamed_label",
        "bound_label",
        "config_path",
        "manifest_path",
        "config_sha",
        "manifest_sha",
    ),
)
def test_reverse_iteration_v2_causal_source_drift_fails_closed(
    tmp_path: Path,
    drift: str,
) -> None:
    files = _bundle_files(
        tmp_path / drift,
        expert="reverse",
        reverse_iteration_v2=True,
    )
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    source_hashes = copy.deepcopy(manifest["source_and_teacher_hashes_post"])
    params_label = "reverse_iteration_v2_failed_candidate_params"
    bound = config["reverse_iteration_v2_authorization"]["bound_causal_inputs"]

    if drift == "missing_label":
        del source_hashes[params_label]
    elif drift == "unknown_label":
        source_hashes["reverse_iteration_v2_unexpected"] = copy.deepcopy(
            source_hashes["reverse_iteration_v2_authorization"]
        )
    elif drift == "renamed_label":
        source_hashes["reverse_iteration_v2_failed_candidate_parameter"] = (
            source_hashes.pop(params_label)
        )
    elif drift == "bound_label":
        bound["failed_candidate_parameter"] = bound.pop(
            "failed_candidate_params"
        )
    elif drift == "config_path":
        bound["failed_candidate_params"]["path"] = bound[
            "failed_candidate_manifest"
        ]["path"]
    elif drift == "manifest_path":
        source_hashes[params_label]["path"] = source_hashes[
            "reverse_iteration_v2_failed_candidate_manifest"
        ]["path"]
    elif drift == "config_sha":
        bound["failed_candidate_params"]["sha256"] = "0" * 64
    elif drift == "manifest_sha":
        source_hashes[params_label]["sha256"] = "0" * 64
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(drift)

    _write_json(files["config"], config)
    manifest["resolved_config"]["sha256"] = sha256_file(files["config"])
    manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(
        config
    )
    manifest["source_and_teacher_hashes_pre"] = copy.deepcopy(source_hashes)
    manifest["source_and_teacher_hashes_post"] = copy.deepcopy(source_hashes)
    _write_json(files["manifest"], manifest)
    with pytest.raises(ValueError, match="reverse iteration-v2"):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )


@pytest.mark.parametrize(
    ("expert", "fixture_kwarg", "expected_prefix"),
    (
        ("forward", "forward_iteration_v3", "forward_iteration_v3_"),
        ("reverse", "reverse_iteration_v3", "reverse_iteration_v3_"),
    ),
)
def test_iteration_v3_bundle_binds_exact_four_causal_sources_before_pickle(
    tmp_path: Path,
    expert: str,
    fixture_kwarg: str,
    expected_prefix: str,
) -> None:
    files = _bundle_files(
        tmp_path / expert,
        expert=expert,
        **{fixture_kwarg: True},
    )
    bundle = validate_trusted_h4_bundle(
        params_path=files["params"],
        manifest_path=files["manifest"],
        expected_params_sha256=files["params_sha"],
        expected_manifest_sha256=files["manifest_sha"],
        trusted_run_root=files["trusted_run_root"],
    )
    assert len(
        [label for label in bundle.source_hashes if label.startswith(expected_prefix)]
    ) == 4
    assert bundle.config["initialization_source"] == "V22_BRAX_CHECKPOINT"
    assert bundle.config["trusted_h4_parent"] is None


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize(
    "drift",
    (
        "top_level_optimizer",
        "ppo_optimizer",
        "reset_noise",
        "execution_contract",
        "missing_causal_source",
        "fresh_parent",
        "result_auth_sha",
        "simultaneous_manifest_result_flag",
    ),
)
def test_iteration_v3_provenance_near_misses_fail_before_pickle(
    tmp_path: Path,
    expert: str,
    drift: str,
) -> None:
    fixture_kwarg = (
        "forward_iteration_v3" if expert == "forward" else "reverse_iteration_v3"
    )
    files = _bundle_files(
        tmp_path / expert / drift,
        expert=expert,
        **{fixture_kwarg: True},
    )
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    spec = h4pt._iteration_v3_spec(expert)
    if drift == "top_level_optimizer":
        config["learning_rate"] = 9.0e-5
    elif drift == "ppo_optimizer":
        config["ppo"]["discounting"] = 0.98
    elif drift == "reset_noise":
        config["reset_noise_multiplier"] = 0.0
    elif drift == "execution_contract":
        config["training_contract_id"] = spec["wiring_contract"]
        manifest["training_contract_id"] = spec["wiring_contract"]
        result["training_contract_id"] = spec["wiring_contract"]
    elif drift == "missing_causal_source":
        label = f"{spec['prefix']}failed_candidate_params"
        del manifest["source_and_teacher_hashes_pre"][label]
        del manifest["source_and_teacher_hashes_post"][label]
    elif drift == "fresh_parent":
        manifest["parent_checkpoint"]["kind"] = "TRUSTED_H4_PARENT"
    elif drift == "result_auth_sha":
        result[f"{spec['auth_key']}_sha256"] = "0" * 64
    elif drift == "simultaneous_manifest_result_flag":
        inactive = (
            "reverse_iteration_v3_no_target_imitation"
            if expert == "forward"
            else "forward_iteration_v3_touchdown_balance"
        )
        manifest[inactive] = True
        result[inactive] = True
    else:  # pragma: no cover
        raise AssertionError(drift)
    _write_json(files["config"], config)
    _write_json(result_path, result)
    manifest["resolved_config"]["sha256"] = sha256_file(files["config"])
    manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(
        config
    )
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    _write_json(files["manifest"], manifest)
    with pytest.raises(ValueError):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )


@pytest.mark.parametrize(
    ("expert", "fixture_kwarg", "expected_prefix"),
    (
        ("forward", "forward_iteration_v4", "forward_iteration_v4_"),
        ("reverse", "reverse_iteration_v4", "reverse_iteration_v4_"),
    ),
)
def test_iteration_v4_bundle_rejects_stale_current_source_before_pickle(
    tmp_path: Path,
    expert: str,
    fixture_kwarg: str,
    expected_prefix: str,
) -> None:
    files = _bundle_files(
        tmp_path / expert,
        expert=expert,
        **{fixture_kwarg: True},
    )
    assert len(
        [
            label
            for label in load_json_strict(files["manifest"])[
                "source_and_teacher_hashes_pre"
            ]
            if label.startswith(expected_prefix)
        ]
    ) == 10
    with pytest.raises(
        ValueError, match="v4 causal source binding drifted for h4_"
    ):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=files["manifest_sha"],
            trusted_run_root=files["trusted_run_root"],
        )


def _set_bundle_v4_semantic_audit(
    files: dict[str, Any], *, expert: str, semantic_audit: dict[str, bool]
) -> None:
    """Replace only the resolved-config audit and restore its manifest binding."""

    spec = h4pt._iteration_v4_spec(expert)
    config = load_json_strict(files["config"])
    config[spec["auth_key"]]["semantic_audit"] = semantic_audit
    _write_json(files["config"], config)
    manifest = load_json_strict(files["manifest"])
    manifest["resolved_config"]["sha256"] = sha256_file(files["config"])
    manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(
        config
    )
    _write_json(files["manifest"], manifest)


@pytest.mark.parametrize(
    ("expert", "fixture_kwarg"),
    (
        ("forward", "forward_iteration_v4"),
        ("reverse", "reverse_iteration_v4"),
    ),
)
def test_iteration_v4_wiring_raw_semantics_pass_but_current_source_rejects(
    tmp_path: Path, expert: str, fixture_kwarg: str
) -> None:
    authorization = _load_runner_v4_authorization(expert)
    semantic_audit = authorization["semantic_audit"]
    required_structural_checks = {
        "top_level_fields_exact",
        "scope_fail_closed",
        "decision_fail_closed",
    }
    assert required_structural_checks <= set(semantic_audit)
    assert set(semantic_audit) == h4pt._iteration_v4_spec(expert)["semantic_keys"]
    assert all(value is True for value in semantic_audit.values())

    files = _bundle_files(
        tmp_path / expert,
        expert=expert,
        wiring_only=True,
        **{fixture_kwarg: True},
    )
    _set_bundle_v4_semantic_audit(
        files,
        expert=expert,
        semantic_audit=copy.deepcopy(semantic_audit),
    )
    with pytest.raises(
        ValueError, match="v4 causal source binding drifted for h4_"
    ):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
            allow_wiring_diagnostic=True,
        )


@pytest.mark.parametrize(
    ("expert", "fixture_kwarg"),
    (
        ("forward", "forward_iteration_v4"),
        ("reverse", "reverse_iteration_v4"),
    ),
)
@pytest.mark.parametrize("drift", ("missing", "false", "extra"))
def test_iteration_v4_real_runner_semantic_audit_near_misses_fail_closed(
    tmp_path: Path,
    expert: str,
    fixture_kwarg: str,
    drift: str,
) -> None:
    semantic_audit = copy.deepcopy(
        _load_runner_v4_authorization(expert)["semantic_audit"]
    )
    if drift == "missing":
        del semantic_audit["scope_fail_closed"]
    elif drift == "false":
        semantic_audit["decision_fail_closed"] = False
    elif drift == "extra":
        semantic_audit["unrecognized_semantic_check"] = True
    else:  # pragma: no cover - the parameter set above is closed.
        raise AssertionError(drift)

    files = _bundle_files(
        tmp_path / expert / drift,
        expert=expert,
        wiring_only=True,
        **{fixture_kwarg: True},
    )
    _set_bundle_v4_semantic_audit(
        files,
        expert=expert,
        semantic_audit=semantic_audit,
    )
    with pytest.raises(ValueError, match="semantic audit drifted"):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
            allow_wiring_diagnostic=True,
        )


def test_forward_v4_wiring_zero_episode_rows_still_rejects_stale_current_source(
    tmp_path: Path,
) -> None:
    files = _bundle_files(
        tmp_path,
        expert="forward",
        forward_iteration_v4=True,
        wiring_only=True,
    )
    with pytest.raises(
        ValueError, match="v4 causal source binding drifted for h4_"
    ):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=files["manifest_sha"],
            trusted_run_root=files["trusted_run_root"],
            allow_wiring_diagnostic=True,
        )


def test_forward_v4_full_curve_without_episode_rows_fails_closed(
    tmp_path: Path,
) -> None:
    files = _bundle_files(
        tmp_path,
        expert="forward",
        forward_iteration_v4=True,
    )
    manifest = load_json_strict(files["manifest"])
    curve_path = Path(manifest["outputs"]["training_curve"]["path"])
    curve_path.write_text(
        "environment_interactions,training/total_loss\n250000,1.0\n",
        encoding="utf-8",
    )
    manifest["outputs"]["training_curve"]["sha256"] = sha256_file(curve_path)
    _write_json(files["manifest"], manifest)
    with pytest.raises(ValueError, match="full run has no exact runtime episode rows"):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )


def test_forward_v4_valid_fractional_counts_still_reject_stale_current_source(
    tmp_path: Path,
) -> None:
    files = _bundle_files(
        tmp_path,
        expert="forward",
        forward_iteration_v4=True,
    )
    manifest = load_json_strict(files["manifest"])
    curve_path = Path(manifest["outputs"]["training_curve"]["path"])
    with curve_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    length = 37.43478260869565
    qualifying_length_keys = (
        "episode/h4/v4_single_authority_dynamic6_exact",
        "episode/h4/v4_single_authority_dynamic6_field_count_exact",
        "episode/h4/v4_saved_dynamic6_field_count_exact",
        "episode/h4/v4_saved_dynamic6_all_finite",
        "episode/h4/v4_telemetry_force_shape_valid",
        "episode/h4/v4_telemetry_force_all_finite",
    )
    for row in rows[:-1]:
        row["episode/length"] = repr(length)
        for key in qualifying_length_keys:
            row[key] = repr(length)
        row["episode/h4/v4_single_authority_dynamic6_field_count"] = (
            "224.6086956521739"
        )
        row["episode/h4/v4_saved_dynamic6_field_count"] = (
            "224.6086956521739"
        )
        row["episode/h4/v4_saved_dynamic6_substep_count"] = (
            "374.3478260869565"
        )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest["outputs"]["training_curve"]["sha256"] = sha256_file(curve_path)
    _write_json(files["manifest"], manifest)
    with pytest.raises(
        ValueError, match="v4 causal source binding drifted for h4_"
    ):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )


def test_forward_v4_full_curve_rejects_false_field_count_exact_total(
    tmp_path: Path,
) -> None:
    files = _bundle_files(
        tmp_path,
        expert="forward",
        forward_iteration_v4=True,
    )
    manifest = load_json_strict(files["manifest"])
    curve_path = Path(manifest["outputs"]["training_curve"]["path"])
    with curve_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["episode/h4/v4_single_authority_dynamic6_field_count_exact"] = (
        "19.0"
    )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest["outputs"]["training_curve"]["sha256"] = sha256_file(curve_path)
    _write_json(files["manifest"], manifest)
    with pytest.raises(ValueError, match="field_count_exact.*drifted"):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )


@pytest.mark.parametrize(
    "drift",
    (
        "truncated_20",
        "missing_training_row",
        "duplicate_training_row",
        "nonmonotonic_training_rows",
        "fractional_interaction",
        "overrun",
        "missing_final_metrics_row",
        "duplicate_final_metrics_row",
    ),
)
def test_forward_v4_full_curve_progress_near_misses_fail_closed(
    tmp_path: Path, drift: str
) -> None:
    files = _bundle_files(
        tmp_path / drift,
        expert="forward",
        forward_iteration_v4=True,
    )
    manifest = load_json_strict(files["manifest"])
    curve_path = Path(manifest["outputs"]["training_curve"]["path"])
    with curve_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6

    if drift == "truncated_20":
        for row in rows:
            if row["environment_interactions"] == "250000":
                row["environment_interactions"] = "20"
    elif drift == "missing_training_row":
        rows.pop(2)
    elif drift == "duplicate_training_row":
        rows.insert(2, copy.deepcopy(rows[1]))
    elif drift == "nonmonotonic_training_rows":
        rows[1], rows[2] = rows[2], rows[1]
    elif drift == "fractional_interaction":
        rows[2]["environment_interactions"] = "150000.5"
    elif drift == "overrun":
        rows[4]["environment_interactions"] = "250001"
    elif drift == "missing_final_metrics_row":
        rows.pop()
    elif drift == "duplicate_final_metrics_row":
        rows.append(copy.deepcopy(rows[-1]))
    else:  # pragma: no cover - parameter list is closed above.
        raise AssertionError(drift)

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest["outputs"]["training_curve"]["sha256"] = sha256_file(curve_path)
    _write_json(files["manifest"], manifest)
    with pytest.raises(ValueError, match="interaction|full curve"):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )


def test_forward_v4_full_curve_final_metrics_are_cross_bound_to_result(
    tmp_path: Path,
) -> None:
    files = _bundle_files(
        tmp_path,
        expert="forward",
        forward_iteration_v4=True,
    )
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    result["final_metrics"]["training/total_loss"] = 2.0
    _write_json(result_path, result)
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    _write_json(files["manifest"], manifest)
    with pytest.raises(ValueError, match="curve final metrics differ"):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize(
    "drift",
    (
        "top_level_optimizer",
        "mode_specific_delta",
        "missing_causal_source",
        "causal_source_sha",
        "fresh_parent",
        "result_auth_sha",
        "simultaneous_mode_flag",
        "prior_contract_id",
    ),
)
def test_iteration_v4_provenance_near_misses_fail_before_pickle(
    tmp_path: Path,
    expert: str,
    drift: str,
) -> None:
    fixture_kwarg = (
        "forward_iteration_v4" if expert == "forward" else "reverse_iteration_v4"
    )
    files = _bundle_files(
        tmp_path / expert / drift,
        expert=expert,
        **{fixture_kwarg: True},
    )
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    spec = h4pt._iteration_v4_spec(expert)
    if drift == "top_level_optimizer":
        config["learning_rate"] = 9.0e-5
    elif drift == "mode_specific_delta":
        if expert == "forward":
            config["forward_v4_substep_contact"] = False
        else:
            config["backward_residual_scale"] = 0.12
    elif drift == "missing_causal_source":
        label = f"{spec['prefix']}source_h4_post_training"
        del manifest["source_and_teacher_hashes_pre"][label]
        del manifest["source_and_teacher_hashes_post"][label]
    elif drift == "causal_source_sha":
        label = f"{spec['prefix']}source_h4_candidate_evaluator"
        manifest["source_and_teacher_hashes_pre"][label]["sha256"] = "0" * 64
        manifest["source_and_teacher_hashes_post"][label]["sha256"] = "0" * 64
    elif drift == "fresh_parent":
        manifest["parent_checkpoint"]["kind"] = "TRUSTED_H4_PARENT"
    elif drift == "result_auth_sha":
        result[f"{spec['auth_key']}_sha256"] = "0" * 64
    elif drift == "simultaneous_mode_flag":
        inactive = (
            "reverse_iteration_v4_residual_transfer_gain_024"
            if expert == "forward"
            else "forward_iteration_v4_contact_event_validity_persistence"
        )
        config[inactive] = True
        manifest[inactive] = True
        result[inactive] = True
    elif drift == "prior_contract_id":
        config["authorized_iteration_v3_250k_contract_id"] = "WRONG_PRIOR"
        manifest["authorized_iteration_v3_250k_contract_id"] = "WRONG_PRIOR"
        result["authorized_iteration_v3_250k_contract_id"] = "WRONG_PRIOR"
    else:  # pragma: no cover
        raise AssertionError(drift)
    _write_json(files["config"], config)
    _write_json(result_path, result)
    manifest["resolved_config"]["sha256"] = sha256_file(files["config"])
    manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(
        config
    )
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    _write_json(files["manifest"], manifest)
    with pytest.raises(ValueError):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )


@pytest.mark.parametrize(
    "drift",
    (
        "preflight_exact",
        "preflight_field_count",
        "preflight_derived_tolerance",
        "preflight_source_path",
        "preflight_source_hash",
        "preflight_step_source",
        "preflight_probe_seed",
        "preflight_probe_action",
        "runtime_error",
        "runtime_field_count_exact",
        "runtime_saved_shape",
        "runtime_telemetry_shape",
        "runtime_violation",
        "runtime_audit_mode",
        "runtime_requirement",
        "manifest_runtime_missing",
    ),
)
def test_forward_iteration_v4_single_authority_near_misses_fail_before_pickle(
    tmp_path: Path, drift: str
) -> None:
    files = _bundle_files(
        tmp_path / drift,
        expert="forward",
        forward_iteration_v4=True,
    )
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    if drift == "preflight_exact":
        for payload in (config, manifest, result):
            payload["forward_v4_source_semantic_preflight"][
                "dynamic6_exact"
            ] = False
    elif drift == "preflight_field_count":
        for payload in (config, manifest, result):
            payload["forward_v4_source_semantic_preflight"][
                "dynamic6_field_count"
            ] = 0
    elif drift == "preflight_derived_tolerance":
        for payload in (config, manifest, result):
            payload["forward_v4_source_semantic_preflight"][
                "derived_diagnostics"
            ]["numeric_tolerance_used"] = True
    elif drift == "preflight_source_path":
        for payload in (config, manifest, result):
            payload["forward_v4_source_semantic_preflight"][
                "source_provenance"
            ]["joystick"]["resolved_path"] = "/tmp/joystick.py"
    elif drift == "preflight_source_hash":
        for payload in (config, manifest, result):
            payload["forward_v4_source_semantic_preflight"][
                "source_provenance"
            ]["mjx_env"]["sha256"] = "0" * 64
    elif drift == "preflight_step_source":
        for payload in (config, manifest, result):
            payload["forward_v4_source_semantic_preflight"][
                "source_provenance"
            ]["step_source_sha256"] = "0" * 64
    elif drift == "preflight_probe_seed":
        for payload in (config, manifest, result):
            payload["forward_v4_source_semantic_preflight"][
                "probe_input"
            ]["seed"] = 20260810
    elif drift == "preflight_probe_action":
        for payload in (config, manifest, result):
            payload["forward_v4_source_semantic_preflight"][
                "probe_input"
            ]["action_all_zero"] = False
    elif drift == "runtime_error":
        for payload in (manifest, result):
            payload["forward_v4_single_authority_runtime"][
                "dynamic6_max_abs_error"
            ] = 1.0e-12
    elif drift == "runtime_field_count_exact":
        for payload in (manifest, result):
            payload["forward_v4_single_authority_runtime"][
                "dynamic6_field_count_exact"
            ] = False
    elif drift == "runtime_saved_shape":
        for payload in (manifest, result):
            payload["forward_v4_single_authority_runtime"][
                "saved_dynamic6_substep_count"
            ] = 9
    elif drift == "runtime_telemetry_shape":
        for payload in (manifest, result):
            payload["forward_v4_single_authority_runtime"][
                "telemetry_force_shape_valid"
            ] = False
    elif drift == "runtime_violation":
        for payload in (manifest, result):
            payload["forward_v4_single_authority_runtime"][
                "authority_violation_count"
            ] = 1.0
    elif drift == "runtime_audit_mode":
        for payload in (manifest, result):
            payload["forward_v4_single_authority_runtime_audit_mode"] = (
                h4pt.H4_FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE
            )
    elif drift == "runtime_requirement":
        for payload in (config, manifest, result):
            payload["forward_v4_single_authority_runtime_requirement"][
                "dynamic6_field_count"
            ] = 5
    elif drift == "manifest_runtime_missing":
        del manifest["forward_v4_single_authority_runtime"]
    else:  # pragma: no cover
        raise AssertionError(drift)
    _write_json(files["config"], config)
    _write_json(result_path, result)
    manifest["resolved_config"]["sha256"] = sha256_file(files["config"])
    manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(
        config
    )
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    _write_json(files["manifest"], manifest)
    with pytest.raises(ValueError, match="single-authority closure"):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )


def test_reverse_iteration_v4_rejects_forward_authority_coupling(
    tmp_path: Path,
) -> None:
    files = _bundle_files(
        tmp_path,
        expert="reverse",
        reverse_iteration_v4=True,
    )
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    config["forward_v4_source_semantic_preflight"] = {"passed": True}
    _write_json(files["config"], config)
    _write_json(result_path, result)
    manifest["resolved_config"]["sha256"] = sha256_file(files["config"])
    manifest["resolved_config"]["canonical_sha256"] = canonical_json_sha256(
        config
    )
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    _write_json(files["manifest"], manifest)
    with pytest.raises(
        ValueError,
        match="must not bind forward single-authority|inactive iteration metadata",
    ):
        validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )


def test_forward_iteration_v5_real_bundle_reuses_exact_v4_authority_closure(
    tmp_path: Path,
) -> None:
    files = _forward_iteration_v5_bundle_files(tmp_path)
    bundle = validate_trusted_h4_bundle(
        params_path=files["params"],
        manifest_path=files["manifest"],
        expected_params_sha256=files["params_sha"],
        expected_manifest_sha256=sha256_file(files["manifest"]),
        trusted_run_root=files["trusted_run_root"],
    )
    bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    _, audit = load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == ["pickle_opened"]
    assert audit["pickle_opened_only_after_manifest_validation"] is True
    assert audit["pickle_opened_only_after_source_closure"] is True


@pytest.mark.parametrize(
    "drift",
    (
        "v6_flag",
        "v6_contract_id",
        "v6_runtime_claim",
        "combined_original_reproduction",
    ),
)
def test_forward_iteration_v5_rejects_grafted_v6_claim_before_pickle(
    tmp_path: Path, drift: str
) -> None:
    files = _forward_iteration_v5_bundle_files(tmp_path / drift)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    if drift in {"v6_flag", "combined_original_reproduction"}:
        for artifact in (manifest, result):
            artifact["forward_iteration_v6_contact_abort_island_only"] = True
    if drift in {"v6_contract_id", "combined_original_reproduction"}:
        for artifact in (manifest, result):
            artifact["authorized_iteration_v6_250k_contract_id"] = (
                h4pt.H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_CONTRACT_ID
            )
    if drift == "v6_runtime_claim":
        for artifact in (manifest, result):
            artifact["forward_iteration_v6_reward_routing_runtime"] = {
                "routing_exact": True
            }
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(
        ValueError,
        match="iteration.*config/manifest/result|inactive iteration metadata",
    ):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


@pytest.mark.parametrize(
    ("label", "fixture_kwargs"),
    (
        ("legacy_forward", {"expert": "forward"}),
        ("legacy_reverse", {"expert": "reverse"}),
        (
            "v2_forward",
            {"expert": "forward", "forward_iteration_v2": True},
        ),
        (
            "v2_reverse",
            {"expert": "reverse", "reverse_iteration_v2": True},
        ),
        (
            "v3_forward",
            {"expert": "forward", "forward_iteration_v3": True},
        ),
        (
            "v3_reverse",
            {"expert": "reverse", "reverse_iteration_v3": True},
        ),
        (
            "v4_forward",
            {"expert": "forward", "forward_iteration_v4": True},
        ),
        (
            "v4_reverse",
            {"expert": "reverse", "reverse_iteration_v4": True},
        ),
    ),
)
@pytest.mark.parametrize(
    "graft",
    (
        "authorization",
        "runtime_contract",
        "reverse_result_claims",
        "artifact_cross_binding",
    ),
)
def test_older_bundle_rejects_grafted_v6_metadata_before_pickle(
    tmp_path: Path,
    label: str,
    fixture_kwargs: Mapping[str, Any],
    graft: str,
) -> None:
    files = _bundle_files(tmp_path / label / graft, **fixture_kwargs)
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    v6 = _iteration_v6_source_fixture("forward")
    v6_spec = v6["spec"]
    v6_auth = v6["config"][v6_spec["auth_key"]]
    if graft == "authorization":
        manifest[v6_spec["auth_key"]] = {
            "path": v6_auth["path"],
            "sha256": v6_auth["sha256"],
            "contract_id": v6_auth["contract_id"],
            "bound_historical_v5_sources": copy.deepcopy(
                v6_auth["bound_historical_v5_sources"]
            ),
        }
        result[f"{v6_spec['auth_key']}_sha256"] = v6_auth["sha256"]
    elif graft == "runtime_contract":
        for artifact in (manifest, result):
            artifact["iteration_v6_core_source"] = copy.deepcopy(
                v6["config"]["iteration_v6_core_source"]
            )
            artifact["reward_routing_contract"] = copy.deepcopy(
                v6["config"]["reward_routing_contract"]
            )
    elif graft == "reverse_result_claims":
        result.update(
            {
                "h4_parent_checkpoint_allowed": False,
                "v4_gain_inherited": False,
                "v5_parent_checkpoint_inherited": False,
                "teacher_target_contribution_zero": True,
            }
        )
    elif graft == "artifact_cross_binding":
        for artifact in (manifest, result):
            artifact["iteration_v6_artifact_cross_binding"] = copy.deepcopy(
                h4pt.H4_ITERATION_V6_ARTIFACT_CROSS_BINDING
            )
    else:  # pragma: no cover
        raise AssertionError(graft)
    _write_json(result_path, result)
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    _write_json(files["manifest"], manifest)
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="inactive iteration metadata"):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


@pytest.mark.parametrize(
    "drift",
    (
        "stripped_requirement",
        "stripped_preflight",
        "stripped_runtime",
    ),
)
def test_forward_iteration_v5_stripped_v4_authority_data_fails_before_pickle(
    tmp_path: Path, drift: str
) -> None:
    files = _forward_iteration_v5_bundle_files(tmp_path / drift)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    if drift == "stripped_requirement":
        weak = {
            "fail_closed_before_output_commit": True,
            "numeric_tolerance_used": False,
        }
        for artifact in (config, manifest, result):
            artifact["forward_v4_single_authority_runtime_requirement"] = (
                copy.deepcopy(weak)
            )
    elif drift == "stripped_preflight":
        stripped_keys = {
            "timing",
            "reference_source",
            "candidate_source",
            "source_provenance",
            "probe_input",
            "qualifying_dynamic_state_fields",
            "derived_diagnostics",
        }
        for artifact in (config, manifest, result):
            for key in stripped_keys:
                artifact["forward_v4_source_semantic_preflight"].pop(key)
    elif drift == "stripped_runtime":
        stripped_keys = {
            "dynamic6_max_abs_error",
            "dynamic6_field_count",
            "saved_dynamic6_substep_count",
            "saved_dynamic6_field_count",
            "telemetry_force_shape",
        }
        for artifact in (manifest, result):
            for key in stripped_keys:
                artifact["forward_v4_single_authority_runtime"].pop(key)
    else:  # pragma: no cover
        raise AssertionError(drift)
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="forward-v4"):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_iteration_v6_authorization_source_and_historical_closure_are_exact(
    expert: str,
) -> None:
    fixture = _iteration_v6_source_fixture(expert)
    spec = fixture["spec"]
    runner_labels = {
        causal_key: f"{expert}_iteration_v6_{causal_key}"
        for causal_key in spec["causal_labels"]
    }
    assert spec["causal_labels"] == runner_labels
    assert spec["v5_manifest_label"] == runner_labels[
        "rejected_iteration_v5_manifest"
    ]
    assert not (
        set(_ITERATION_V6_LEGACY_FILENAME_CAUSAL_LABELS[expert].values())
        & set(fixture["source_hashes"])
    )
    paths = h4pt._validated_iteration_v6_source_paths(
        expert=expert,
        config=fixture["config"],
        source_hashes=fixture["source_hashes"],
    )
    assert set(paths) == set(fixture["source_hashes"])
    assert paths[fixture["spec"]["auth_label"]].name == fixture["spec"][
        "auth_filename"
    ]


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize(
    "drift", ("missing", "false", "extra", "one_ulp", "path", "hash")
)
def test_iteration_v6_contract_drifts_fail_closed(
    expert: str, drift: str
) -> None:
    fixture = _iteration_v6_source_fixture(expert)
    spec = fixture["spec"]
    config = copy.deepcopy(fixture["config"])
    source_hashes = copy.deepcopy(fixture["source_hashes"])
    if drift == "missing":
        source_hashes.pop(next(iter(spec["causal_labels"].values())))
    elif drift == "false":
        semantic = config[spec["auth_key"]]["semantic_audit"]
        semantic[next(iter(semantic))] = False
    elif drift == "extra":
        source_hashes[f"{spec['prefix']}unexpected"] = copy.deepcopy(
            source_hashes[spec["auth_label"]]
        )
    elif drift == "one_ulp":
        if expert == "forward":
            config["reward_scales"]["h4_contact_pulse_40ms"] = float(
                np.nextafter(-1.0, 0.0)
            )
        else:
            config["action_parameterization_contract"][
                "directional_span_fraction"
            ] = float(np.nextafter(0.9, 1.0))
    elif drift == "path":
        source_label = next(iter(spec["causal_labels"].values()))
        bound_key = source_label.removeprefix(spec["prefix"])
        config[spec["auth_key"]]["bound_causal_inputs"][bound_key]["path"] = (
            config[spec["auth_key"]]["path"]
        )
    elif drift == "hash":
        source_label = next(iter(spec["causal_labels"].values()))
        source_hashes[source_label]["sha256"] = "0" * 64
    else:  # pragma: no cover
        raise AssertionError(drift)
    calls: list[str] = []

    def pickle_loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="iteration-v6"):
        h4pt._validated_iteration_v6_source_paths(
            expert=expert,
            config=config,
            source_hashes=source_hashes,
        )
        pickle_loader(None)
    assert calls == []


@pytest.mark.parametrize(
    ("expert", "causal_key", "legacy_alias"),
    [
        (expert, causal_key, legacy_alias)
        for expert, aliases in (
            _ITERATION_V6_LEGACY_FILENAME_CAUSAL_LABELS.items()
        )
        for causal_key, legacy_alias in aliases.items()
    ],
)
@pytest.mark.parametrize(
    "drift", ("missing", "filename_alias", "path", "sha", "cross_swap")
)
def test_iteration_v6_each_runner_causal_binding_drift_fails_before_pickle(
    expert: str,
    causal_key: str,
    legacy_alias: str,
    drift: str,
) -> None:
    fixture = _iteration_v6_source_fixture(expert)
    spec = fixture["spec"]
    config = copy.deepcopy(fixture["config"])
    source_hashes = copy.deepcopy(fixture["source_hashes"])
    source_label = f"{expert}_iteration_v6_{causal_key}"
    assert spec["causal_labels"][causal_key] == source_label
    bound_causal = config[spec["auth_key"]]["bound_causal_inputs"]
    if drift == "missing":
        source_hashes.pop(source_label)
    elif drift == "filename_alias":
        source_hashes[legacy_alias] = source_hashes.pop(source_label)
    elif drift == "path":
        replacement = config[spec["auth_key"]]["path"]
        source_hashes[source_label]["path"] = replacement
        bound_causal[causal_key]["path"] = replacement
    elif drift == "sha":
        source_hashes[source_label]["sha256"] = "0" * 64
        bound_causal[causal_key]["sha256"] = "0" * 64
    elif drift == "cross_swap":
        causal_keys = tuple(spec["causal_labels"])
        other_key = causal_keys[(causal_keys.index(causal_key) + 1) % len(causal_keys)]
        other_label = f"{expert}_iteration_v6_{other_key}"
        source_hashes[source_label], source_hashes[other_label] = (
            source_hashes[other_label],
            source_hashes[source_label],
        )
        bound_causal[causal_key], bound_causal[other_key] = (
            bound_causal[other_key],
            bound_causal[causal_key],
        )
    else:  # pragma: no cover
        raise AssertionError(drift)

    calls: list[str] = []

    def pickle_loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="iteration-v6"):
        h4pt._validated_iteration_v6_source_paths(
            expert=expert,
            config=config,
            source_hashes=source_hashes,
        )
        pickle_loader(None)
    assert calls == []


def _write_reverse_iteration_v6_fractional_curve(
    path: Path, *, exact_count_total: float
) -> float:
    length = 3.5
    prefix = "episode/h4/v6_reverse_"
    row = {
        "environment_interactions": h4pt.H4_WIRING_INTERACTIONS,
        "episode/length": length,
        f"{prefix}decoder_exact": length,
        f"{prefix}decoder_max_abs_error": 0.0,
        f"{prefix}decoder_leg_count": float(np.nextafter(10.0 * length, 0.0)),
        f"{prefix}decoder_leg_count_exact": exact_count_total,
        f"{prefix}decoder_head_zero_exact": length,
        f"{prefix}teacher_target_contribution_zero_exact": length,
        f"{prefix}residual_authority_scale": 0.0,
        f"{prefix}decoder_all_finite": length,
        f"{prefix}decoder_margin_saturation_count": 1.25,
        f"{prefix}decoder_action_clip_count": 0.5,
        f"{prefix}decoder_guard_lag_max_rad": 0.125,
        f"{prefix}precomposer_call_count": 7.25,
        f"{prefix}precomposer_call_count_exact": length,
        f"{prefix}final_guard_call_count": 9.75,
        f"{prefix}final_guard_call_count_exact": length,
        f"{prefix}decoder_violation": 0.0,
        f"{prefix}decoder_assertion_token": 0.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return length


def test_reverse_iteration_v6_fractional_length_uses_device_exact_booleans(
    tmp_path: Path,
) -> None:
    curve_path = tmp_path / "fractional_curve.csv"
    length = _write_reverse_iteration_v6_fractional_curve(
        curve_path, exact_count_total=3.5
    )
    audit = h4pt._iteration_v6_curve_rows(
        curve_path, expert="reverse", wiring_only=True
    )
    assert audit["observed_episode_metric_rows"] == 1
    assert audit["active_reverse_sample_count"] == length
    assert audit["metric_totals"]["decoder_leg_count"] != 10.0 * length


def test_reverse_iteration_v6_fractional_length_rejects_one_ulp_exact_boolean(
    tmp_path: Path,
) -> None:
    curve_path = tmp_path / "fractional_curve_1ulp.csv"
    _write_reverse_iteration_v6_fractional_curve(
        curve_path, exact_count_total=float(np.nextafter(3.5, 0.0))
    )
    with pytest.raises(ValueError, match="decoder routing drifted"):
        h4pt._iteration_v6_curve_rows(
            curve_path, expert="reverse", wiring_only=True
        )


def test_forward_iteration_v6_full_bundle_opens_pickle_only_after_all_gates(
    tmp_path: Path,
) -> None:
    files = _forward_iteration_v6_bundle_files(tmp_path)
    bundle = validate_trusted_h4_bundle(
        params_path=files["params"],
        manifest_path=files["manifest"],
        expected_params_sha256=files["params_sha"],
        expected_manifest_sha256=sha256_file(files["manifest"]),
        trusted_run_root=files["trusted_run_root"],
    )
    bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    _, audit = load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == ["pickle_opened"]
    assert audit["pickle_opened_only_after_manifest_validation"] is True
    assert audit["pickle_opened_only_after_source_closure"] is True


@pytest.mark.parametrize(
    ("expert", "builder"),
    (
        ("forward", _forward_iteration_v6_bundle_files),
        ("reverse", _reverse_iteration_v6_bundle_files),
    ),
)
def test_iteration_v6_real_runner_null_authorization_placeholders_are_not_claims(
    tmp_path: Path, expert: str, builder: Any
) -> None:
    files = builder(tmp_path / expert)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result = load_json_strict(Path(manifest["outputs"]["result"]["path"]))
    for artifact in (config, manifest):
        assert artifact["forward_iteration_v2_authorization"] is None
        assert artifact["reverse_iteration_v2_authorization"] is None
    assert result["forward_iteration_v2_authorization_sha256"] is None
    assert result["reverse_iteration_v2_authorization_sha256"] is None
    core_records = tuple(
        artifact["iteration_v6_core_source"]
        for artifact in (config, manifest, result)
    )
    assert all(set(record) == {"path", "sha256"} for record in core_records)
    assert core_records[0] == core_records[1] == core_records[2]
    assert (
        core_records[0]["sha256"]
        == h4pt.PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256
    )

    bundle = validate_trusted_h4_bundle(
        params_path=files["params"],
        manifest_path=files["manifest"],
        expected_params_sha256=files["params_sha"],
        expected_manifest_sha256=sha256_file(files["manifest"]),
        trusted_run_root=files["trusted_run_root"],
    )
    assert (bundle.status, bundle.activity, bundle.expert) == (
        "COMPLETED",
        "PPO_PILOT_TRAINING",
        expert,
    )


@pytest.mark.parametrize(("expert", "builder"), _ITERATION_V6_LOCATION_BUILDERS)
def test_iteration_v6_artifact_locations_are_exact(
    tmp_path: Path, expert: str, builder: Any
) -> None:
    files = builder(tmp_path / expert)
    config, manifest, result = _load_iteration_v6_artifacts_for_test(files)
    artifacts = {"config": config, "manifest": manifest, "result": result}
    expected = _iteration_v6_expected_artifact_locations_for_test(expert)
    controlled_keys = set(expected)
    for artifact_name, artifact in artifacts.items():
        assert set(artifact) & controlled_keys == {
            key
            for key, locations in expected.items()
            if artifact_name in locations
        }

    bundle = validate_trusted_h4_bundle(
        params_path=files["params"],
        manifest_path=files["manifest"],
        expected_params_sha256=files["params_sha"],
        expected_manifest_sha256=sha256_file(files["manifest"]),
        trusted_run_root=files["trusted_run_root"],
    )
    assert (bundle.status, bundle.activity, bundle.expert) == (
        "COMPLETED",
        "PPO_PILOT_TRAINING",
        expert,
    )


@pytest.mark.parametrize(
    ("expert", "builder", "artifact_name", "controlled_key"),
    [
        (expert, builder, artifact_name, controlled_key)
        for expert, builder in _ITERATION_V6_LOCATION_BUILDERS
        for controlled_key, locations in (
            _iteration_v6_expected_artifact_locations_for_test(expert).items()
        )
        for artifact_name in sorted(locations)
    ],
)
def test_iteration_v6_missing_required_artifact_slot_fails_before_pickle(
    tmp_path: Path,
    expert: str,
    builder: Any,
    artifact_name: str,
    controlled_key: str,
) -> None:
    files = builder(tmp_path / expert)
    config, manifest, result = _load_iteration_v6_artifacts_for_test(files)
    artifacts = {"config": config, "manifest": manifest, "result": result}
    artifacts[artifact_name].pop(controlled_key)
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    _assert_iteration_v6_location_rejects_before_pickle(files)


@pytest.mark.parametrize(
    ("expert", "builder", "artifact_name", "controlled_key", "mutation"),
    [
        (expert, builder, artifact_name, controlled_key, mutation)
        for expert, builder in _ITERATION_V6_LOCATION_BUILDERS
        for controlled_key, locations in (
            _iteration_v6_expected_artifact_locations_for_test(expert).items()
        )
        for artifact_name in ("config", "manifest", "result")
        if artifact_name not in locations
        for mutation in ("valid", "null", "extra", "scalar")
    ],
)
def test_iteration_v6_wrong_artifact_location_fails_before_pickle(
    tmp_path: Path,
    expert: str,
    builder: Any,
    artifact_name: str,
    controlled_key: str,
    mutation: str,
) -> None:
    files = builder(tmp_path / expert)
    config, manifest, result = _load_iteration_v6_artifacts_for_test(files)
    artifacts = {"config": config, "manifest": manifest, "result": result}
    expected = _iteration_v6_expected_artifact_locations_for_test(expert)
    if expected[controlled_key]:
        source_artifact_name = next(iter(sorted(expected[controlled_key])))
        valid_value = copy.deepcopy(
            artifacts[source_artifact_name][controlled_key]
        )
    else:
        assert controlled_key == "legacy_reward_config_audit"
        reverse_fixture = _iteration_v6_source_fixture("reverse")
        valid_value = copy.deepcopy(
            reverse_fixture["config"][reverse_fixture["spec"]["auth_key"]][
                controlled_key
            ]
        )
    if mutation == "valid":
        wrong_value = valid_value
    elif mutation == "null":
        wrong_value = None
    elif mutation == "extra":
        if isinstance(valid_value, dict):
            wrong_value = {**valid_value, "unexpected": True}
        else:
            wrong_value = {"value": valid_value, "unexpected": True}
    elif mutation == "scalar":
        wrong_value = "unexpected-scalar"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    artifacts[artifact_name][controlled_key] = wrong_value
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    _assert_iteration_v6_location_rejects_before_pickle(files)


@pytest.mark.parametrize(
    ("expert", "builder"),
    (
        ("forward", _forward_iteration_v6_bundle_files),
        ("reverse", _reverse_iteration_v6_bundle_files),
    ),
)
@pytest.mark.parametrize("artifact_name", ("config", "manifest", "result"))
def test_iteration_v6_inactive_authorization_non_null_fails_before_pickle(
    tmp_path: Path, expert: str, builder: Any, artifact_name: str
) -> None:
    files = builder(tmp_path / expert / artifact_name)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    if artifact_name == "result":
        result[f"{expert}_iteration_v2_authorization_sha256"] = "0" * 64
    else:
        artifact = config if artifact_name == "config" else manifest
        artifact[f"{expert}_iteration_v2_authorization"] = {
            "inactive_authorization_claim": True
        }
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="inactive iteration metadata"):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(
            bundle, files["source_paths"]
        )
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


@pytest.mark.parametrize(
    ("expert", "builder", "artifact_name", "inactive_key"),
    (
        (
            "forward",
            _forward_iteration_v6_bundle_files,
            "manifest",
            "reverse_iteration_v6_decoder_runtime",
        ),
        (
            "forward",
            _forward_iteration_v6_bundle_files,
            "config",
            "action_parameterization_contract",
        ),
        (
            "reverse",
            _reverse_iteration_v6_bundle_files,
            "result",
            "forward_iteration_v6_reward_routing_runtime",
        ),
        (
            "reverse",
            _reverse_iteration_v6_bundle_files,
            "config",
            "reward_routing_contract",
        ),
    ),
)
def test_iteration_v6_inactive_non_authorization_null_fails_before_pickle(
    tmp_path: Path,
    expert: str,
    builder: Any,
    artifact_name: str,
    inactive_key: str,
) -> None:
    files = builder(tmp_path / expert / inactive_key)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    artifacts = {"config": config, "manifest": manifest, "result": result}
    artifacts[artifact_name][inactive_key] = None
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="inactive iteration metadata"):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(
            bundle, files["source_paths"]
        )
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


def test_iteration_v5_inactive_v6_core_null_fails_before_pickle(
    tmp_path: Path,
) -> None:
    files = _forward_iteration_v5_bundle_files(tmp_path)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    config["iteration_v6_core_source"] = None
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="inactive iteration metadata"):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(
            bundle, files["source_paths"]
        )
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


def test_reverse_iteration_v6_full_bundle_opens_pickle_only_after_all_gates(
    tmp_path: Path,
) -> None:
    files = _reverse_iteration_v6_bundle_files(tmp_path)
    bundle = validate_trusted_h4_bundle(
        params_path=files["params"],
        manifest_path=files["manifest"],
        expected_params_sha256=files["params_sha"],
        expected_manifest_sha256=sha256_file(files["manifest"]),
        trusted_run_root=files["trusted_run_root"],
    )
    bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    _, audit = load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == ["pickle_opened"]
    assert audit["pickle_opened_only_after_manifest_validation"] is True
    assert audit["pickle_opened_only_after_source_closure"] is True


@pytest.mark.parametrize(
    ("expert", "builder"),
    (
        ("forward", _forward_iteration_v6_bundle_files),
        ("reverse", _reverse_iteration_v6_bundle_files),
    ),
)
@pytest.mark.parametrize(
    "drift",
    (
        "missing",
        "false",
        "extra",
        "cross_drift",
        "config_copy",
        "integer_ones",
        "float_ones",
    ),
)
def test_iteration_v6_artifact_cross_binding_drifts_fail_before_pickle(
    tmp_path: Path,
    expert: str,
    builder: Any,
    drift: str,
) -> None:
    files = builder(tmp_path / expert / drift)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    key = "iteration_v6_artifact_cross_binding"
    if drift == "missing":
        manifest.pop(key)
    elif drift == "false":
        manifest[key]["passed"] = False
        result[key]["passed"] = False
    elif drift == "extra":
        manifest[key]["extra"] = True
        result[key]["extra"] = True
    elif drift == "cross_drift":
        result[key]["runtime_requirement_cross_bound"] = False
    elif drift == "config_copy":
        config[key] = copy.deepcopy(h4pt.H4_ITERATION_V6_ARTIFACT_CROSS_BINDING)
    elif drift == "integer_ones":
        manifest[key] = {name: 1 for name in manifest[key]}
        result[key] = {name: 1 for name in result[key]}
    elif drift == "float_ones":
        manifest[key] = {name: 1.0 for name in manifest[key]}
        result[key] = {name: 1.0 for name in result[key]}
    else:  # pragma: no cover
        raise AssertionError(drift)
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(
        ValueError, match="artifact cross-binding|artifact location closure"
    ):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


@pytest.mark.parametrize(
    ("expert", "builder"),
    (
        ("forward", _forward_iteration_v6_bundle_files),
        ("reverse", _reverse_iteration_v6_bundle_files),
    ),
)
@pytest.mark.parametrize(
    "drift",
    ("requirement_bool_as_int", "runtime_bool_as_int", "runtime_int_as_float"),
)
def test_iteration_v6_runtime_types_are_exact_before_pickle(
    tmp_path: Path,
    expert: str,
    builder: Any,
    drift: str,
) -> None:
    files = builder(tmp_path / expert / drift)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    runtime_key = (
        "forward_iteration_v6_reward_routing_runtime"
        if expert == "forward"
        else "reverse_iteration_v6_decoder_runtime"
    )
    requirement_key = f"{runtime_key}_requirement"
    if drift == "requirement_bool_as_int":
        requirement_field = (
            "routing_exact"
            if expert == "forward"
            else "decoder_exact"
        )
        for artifact in (config, manifest, result):
            artifact[requirement_key][requirement_field] = 1
    elif drift == "runtime_bool_as_int":
        for artifact in (manifest, result):
            artifact[runtime_key]["passed"] = 1
    elif drift == "runtime_int_as_float":
        for artifact in (manifest, result):
            artifact[runtime_key]["completed_training_steps"] = 5.0
    else:  # pragma: no cover
        raise AssertionError(drift)
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="iteration-v6.*runtime"):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


@pytest.mark.parametrize(
    ("expert", "builder", "drift"),
    (
        ("forward", _forward_iteration_v6_bundle_files, "routing_true_as_int"),
        ("forward", _forward_iteration_v6_bundle_files, "routing_false_as_int"),
        ("reverse", _reverse_iteration_v6_bundle_files, "action_int_as_float"),
        ("reverse", _reverse_iteration_v6_bundle_files, "action_true_as_int"),
        ("reverse", _reverse_iteration_v6_bundle_files, "timing_int_as_float"),
        ("reverse", _reverse_iteration_v6_bundle_files, "timing_false_as_int"),
        ("reverse", _reverse_iteration_v6_bundle_files, "residual_float_as_int"),
    ),
)
def test_iteration_v6_authorized_contract_types_are_exact_before_pickle(
    tmp_path: Path,
    expert: str,
    builder: Any,
    drift: str,
) -> None:
    files = builder(tmp_path / expert / drift)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    artifacts = (config, manifest, result)
    if drift == "routing_true_as_int":
        for artifact in artifacts:
            artifact["reward_routing_contract"][
                "off_gap_loss_retained_as_telemetry"
            ] = 1
    elif drift == "routing_false_as_int":
        for artifact in artifacts:
            artifact["reward_routing_contract"][
                "legacy_aggregate_contact_pulse_routing_allowed"
            ] = 0
    elif drift == "action_int_as_float":
        for artifact in artifacts:
            artifact["action_parameterization_contract"][
                "nonlinear_exponent"
            ] = 5.0
    elif drift == "action_true_as_int":
        for artifact in artifacts:
            artifact["action_parameterization_contract"][
                "teacher_target_contribution_zero"
            ] = 1
    elif drift == "timing_int_as_float":
        for artifact in artifacts:
            artifact["teacher_timing_contract"]["teacher_table_rows"] = 54.0
    elif drift == "timing_false_as_int":
        for artifact in artifacts:
            artifact["teacher_timing_contract"]["target_guard_changed"] = 0
    elif drift == "residual_float_as_int":
        for artifact in artifacts:
            artifact["backward_residual_scale"] = 0
    else:  # pragma: no cover
        raise AssertionError(drift)
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


def test_reverse_iteration_v6_one_ulp_exact_count_fails_before_pickle(
    tmp_path: Path,
) -> None:
    files = _reverse_iteration_v6_bundle_files(tmp_path)
    manifest = load_json_strict(files["manifest"])
    curve_path = Path(manifest["outputs"]["training_curve"]["path"])
    with curve_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    for row in rows:
        if row.get("episode/length") not in (None, ""):
            row["episode/h4/v6_reverse_decoder_leg_count_exact"] = str(
                float(np.nextafter(float(row["episode/length"]), 0.0))
            )
            break
    with curve_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest["outputs"]["training_curve"]["sha256"] = sha256_file(curve_path)
    _write_json(files["manifest"], manifest)
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="decoder routing drifted"):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


@pytest.mark.parametrize(
    ("expert", "builder"),
    (
        ("forward", _forward_iteration_v6_bundle_files),
        ("reverse", _reverse_iteration_v6_bundle_files),
    ),
)
@pytest.mark.parametrize(
    "drift",
    (
        "truncated_curve",
        "training_steps_4",
        "optimizer_updates_399",
        "progress_false",
        "source_false",
        "params_finite_false",
        "metrics_finite_false",
    ),
)
def test_iteration_v6_full_completion_drifts_fail_before_pickle(
    tmp_path: Path,
    expert: str,
    builder: Any,
    drift: str,
) -> None:
    files = builder(tmp_path / expert / drift)
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    runtime_key = (
        "forward_iteration_v6_reward_routing_runtime"
        if expert == "forward"
        else "reverse_iteration_v6_decoder_runtime"
    )
    if drift == "truncated_curve":
        curve_path = Path(manifest["outputs"]["training_curve"]["path"])
        with curve_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or ())
        episode_index = next(
            index
            for index in range(len(rows) - 1, -1, -1)
            if rows[index].get("episode/length") not in (None, "")
        )
        rows.pop(episode_index)
        with curve_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        manifest["outputs"]["training_curve"]["sha256"] = sha256_file(
            curve_path
        )
    else:
        runtime_field, value = {
            "training_steps_4": ("completed_training_steps", 4),
            "optimizer_updates_399": ("completed_optimizer_updates", 399),
            "progress_false": ("progress_reached_final_interaction", False),
            "source_false": ("source_and_teacher_unchanged", False),
            "params_finite_false": ("final_params_all_finite", False),
            "metrics_finite_false": ("final_metrics_all_finite", False),
        }[drift]
        manifest[runtime_key][runtime_field] = value
        result[runtime_key][runtime_field] = value
    _write_json(result_path, result)
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    _write_json(files["manifest"], manifest)
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


@pytest.mark.parametrize(
    ("expert", "builder"),
    (
        ("forward", _forward_iteration_v6_bundle_files),
        ("reverse", _reverse_iteration_v6_bundle_files),
    ),
)
@pytest.mark.parametrize(
    "drift",
    (
        "all_zero_without_final",
        "duplicate_progress",
        "nonmonotonic_progress",
        "fractional_progress",
        "overrun_progress",
        "missing_final_metrics_row",
        "duplicate_final_metrics_row",
        "result_final_metric_drift",
    ),
)
def test_iteration_v6_bound_curve_completion_drifts_fail_before_pickle(
    tmp_path: Path,
    expert: str,
    builder: Any,
    drift: str,
) -> None:
    files = builder(tmp_path / expert / drift)
    manifest = load_json_strict(files["manifest"])
    curve_path = Path(manifest["outputs"]["training_curve"]["path"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    with curve_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())

    if drift == "all_zero_without_final":
        rows = [
            {**row, "environment_interactions": "0"}
            for row in rows
            if row.get("episode/length") not in (None, "")
        ]
    elif drift == "duplicate_progress":
        rows[1]["environment_interactions"] = rows[0][
            "environment_interactions"
        ]
    elif drift == "nonmonotonic_progress":
        rows[1]["environment_interactions"], rows[2][
            "environment_interactions"
        ] = (
            rows[2]["environment_interactions"],
            rows[1]["environment_interactions"],
        )
    elif drift == "fractional_progress":
        rows[2]["environment_interactions"] = "150000.5"
    elif drift == "overrun_progress":
        rows[4]["environment_interactions"] = "250001"
    elif drift == "missing_final_metrics_row":
        rows.pop()
    elif drift == "duplicate_final_metrics_row":
        rows.append(copy.deepcopy(rows[-1]))
    elif drift == "result_final_metric_drift":
        result["final_metrics"]["training/total_loss"] = 2.0
    else:  # pragma: no cover
        raise AssertionError(drift)

    if drift != "result_final_metric_drift":
        with curve_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        manifest["outputs"]["training_curve"]["sha256"] = sha256_file(
            curve_path
        )
    _write_json(result_path, result)
    manifest["outputs"]["result"]["sha256"] = sha256_file(result_path)
    _write_json(files["manifest"], manifest)
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(
            bundle, files["source_paths"]
        )
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


@pytest.mark.parametrize(
    ("expert", "builder"),
    (
        ("forward", _forward_iteration_v6_bundle_files),
        ("reverse", _reverse_iteration_v6_bundle_files),
    ),
)
@pytest.mark.parametrize("artifact_name", ("config", "manifest", "result"))
@pytest.mark.parametrize(
    "drift", ("null", "missing", "extra", "path", "sha256", "type")
)
def test_iteration_v6_core_source_drifts_fail_before_pickle(
    tmp_path: Path,
    expert: str,
    builder: Any,
    artifact_name: str,
    drift: str,
) -> None:
    files = builder(tmp_path / expert / artifact_name / drift)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    artifacts = {"config": config, "manifest": manifest, "result": result}
    artifact = artifacts[artifact_name]
    if drift == "null":
        artifact["iteration_v6_core_source"] = None
    elif drift == "missing":
        artifact.pop("iteration_v6_core_source")
    elif drift == "extra":
        artifact["iteration_v6_core_source"]["unexpected"] = True
    elif drift == "path":
        replacement = config[
            h4pt._iteration_v6_spec(expert)["auth_key"]
        ]["path"]
        artifact["iteration_v6_core_source"]["path"] = replacement
    elif drift == "sha256":
        artifact["iteration_v6_core_source"]["sha256"] = "0" * 64
    elif drift == "type":
        artifact["iteration_v6_core_source"]["sha256"] = True
    else:  # pragma: no cover
        raise AssertionError(drift)
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError, match="core source|artifact location closure"):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


@pytest.mark.parametrize(
    "drift", ("missing", "false", "extra", "one_ulp", "path", "hash")
)
def test_forward_iteration_v6_bundle_drifts_fail_before_pickle(
    tmp_path: Path, drift: str
) -> None:
    files = _forward_iteration_v6_bundle_files(tmp_path / drift)
    config = load_json_strict(files["config"])
    manifest = load_json_strict(files["manifest"])
    result_path = Path(manifest["outputs"]["result"]["path"])
    result = load_json_strict(result_path)
    requirement_key = "forward_iteration_v6_reward_routing_runtime_requirement"
    runtime_key = "forward_iteration_v6_reward_routing_runtime"
    spec = h4pt._iteration_v6_spec("forward")
    if drift == "missing":
        for artifact in (config, manifest, result):
            artifact[requirement_key].pop("routing_exact")
    elif drift == "false":
        for artifact in (config, manifest, result):
            artifact[requirement_key]["routing_exact"] = False
    elif drift == "extra":
        result[runtime_key]["unexpected"] = 0.0
        manifest[runtime_key]["unexpected"] = 0.0
    elif drift == "one_ulp":
        value = float(np.nextafter(-1.0, 0.0))
        for artifact in (config, manifest, result):
            artifact[requirement_key]["pulse_reward_scale"] = value
    elif drift == "path":
        source_label = next(iter(spec["causal_labels"].values()))
        bound_key = source_label.removeprefix(spec["prefix"])
        config[spec["auth_key"]]["bound_causal_inputs"][bound_key]["path"] = (
            config[spec["auth_key"]]["path"]
        )
    elif drift == "hash":
        source_label = next(iter(spec["causal_labels"].values()))
        manifest["source_and_teacher_hashes_pre"][source_label]["sha256"] = (
            "0" * 64
        )
        manifest["source_and_teacher_hashes_post"][source_label]["sha256"] = (
            "0" * 64
        )
    else:  # pragma: no cover
        raise AssertionError(drift)
    _rewrite_bundle_artifacts(
        files,
        config=config,
        manifest=manifest,
        result=result,
    )
    calls: list[str] = []

    def loader(_stream: Any) -> Any:
        calls.append("pickle_opened")
        return _params()

    with pytest.raises(ValueError):
        bundle = validate_trusted_h4_bundle(
            params_path=files["params"],
            manifest_path=files["manifest"],
            expected_params_sha256=files["params_sha"],
            expected_manifest_sha256=sha256_file(files["manifest"]),
            trusted_run_root=files["trusted_run_root"],
        )
        bundle = validate_h4_training_source_closure(bundle, files["source_paths"])
        load_trusted_h4_params(bundle, pickle_loader=loader)
    assert calls == []


def test_actor116_numpy_inference_and_head_mask_are_exact() -> None:
    params = _params()
    action = infer_h4_action_numpy(
        params, np.zeros(H4_ACTOR_OBSERVATION_WIDTH, dtype=np.float32)
    )
    np.testing.assert_allclose(action, np.tanh(np.float32(0.5)), rtol=0, atol=1e-7)
    masked = mask_h4_head_action(action)
    np.testing.assert_array_equal(masked[5:9], np.zeros(4, np.float32))
    assert compare_policy_outputs(action, action.copy())["passed"] is True
    with pytest.raises(ValueError, match="width 116"):
        infer_h4_action_numpy(params, np.zeros(114, dtype=np.float32))
    broken = copy.deepcopy(params)
    broken[1]["params"]["hidden_0"]["kernel"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_h4_params(broken)


def _safety_episode() -> dict[str, Any]:
    initial = np.asarray(
        [SAFE_INIT_POS[name] for name in ACTUATOR_JOINT_ORDER], dtype=np.float32
    )
    leg_indices = np.asarray(
        [
            index
            for index, name in enumerate(ACTUATOR_JOINT_ORDER)
            if name not in HEAD_JOINTS
        ]
    )
    lower = np.asarray(
        [SAFE_JOINT_LIMITS[ACTUATOR_JOINT_ORDER[index]][0] for index in leg_indices],
        dtype=np.float32,
    )
    upper = np.asarray(
        [SAFE_JOINT_LIMITS[ACTUATOR_JOINT_ORDER[index]][1] for index in leg_indices],
        dtype=np.float32,
    )
    margin_lower = np.add(lower, np.float32(0.05), dtype=np.float32)
    margin_upper = np.subtract(upper, np.float32(0.05), dtype=np.float32)
    desired = np.zeros(14, dtype=np.float32)
    desired[leg_indices] = np.clip(
        initial[leg_indices], margin_lower, margin_upper
    )
    previous_rows = []
    applied_rows = []
    previous = initial.copy()
    for _index in range(300):
        delta = np.clip(
            np.subtract(desired, previous, dtype=np.float32),
            -np.float32(0.04),
            np.float32(0.04),
        )
        applied = np.zeros(14, dtype=np.float32)
        applied[leg_indices] = np.clip(
            np.add(
                previous[leg_indices], delta[leg_indices], dtype=np.float32
            ),
            lower,
            upper,
        )
        previous_rows.append(previous)
        applied_rows.append(applied)
        previous = applied
    applied_targets = np.stack(applied_rows)
    zeros = np.zeros((300, 14), dtype=np.float32)
    control_trace = {
        "source_dtype": "float32",
        "initial_applied_targets": initial,
        "raw_action": zeros,
        "applied_action": zeros,
        "preclip_targets": np.broadcast_to(desired, (300, 14)).copy(),
        "margin_clipped_targets": np.broadcast_to(desired, (300, 14)).copy(),
        "applied_targets": applied_targets,
        "previous_targets": np.stack(previous_rows),
        "joint_qpos": applied_targets.copy(),
    }
    episode = {
        "completed": True,
        "fell": False,
        "duration_s": 6.0,
        "physics_timestep_s": 0.002,
        "completed_control_ticks": 300,
        "completed_physics_substeps": 3000,
        "reset_audit": {
            "comparison_semantics": "SOURCE_DTYPE_FLOAT32_EXACT",
            "exact_safe_init": True,
            "maximum_safe_init_error_rad": 0.0,
            "head_qpos_peak_rad": 0.0,
        },
        "control_trace": control_trace,
        "physics_substep_audit": {
            "sample_count": 3000,
            "contact_sample_count": 3000,
            "qpos_limit_violations": 0,
            "nonfinite_state_samples": 0,
            "height_fall_samples": 0,
            "minimum_height_m": 0.2,
            "upright_fall_samples": 0,
            "minimum_upright": 0.9,
        },
        "guard_call_audit": {
            "control_tick_count": 300,
            "total_guard_calls": 300,
            "guard_call_violation_count": 0,
            "maximum_guard_calls_per_tick": 1,
        },
        "policy_inference_audit": {
            "input_width": 116,
            "output_width": 14,
            "inference_count": 300,
            "nonfinite_observation_count": 0,
            "nonfinite_action_count": 0,
            "post_mask_nonzero_head_count": 0,
        },
    }
    episode["safety_audit"] = rederive_central_safety_audit_from_control_trace(
        episode
    )
    episode["h4_control_contract"] = rederive_h4_control_contract(episode)
    episode["control_trace"] = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in control_trace.items()
    }
    return episode


def _reverse_safety_episode() -> dict[str, Any]:
    episode = _safety_episode()
    episode["expert"] = "reverse"
    episode["physical_command_mps_radps"] = list(
        H4_STRICT_COMMANDS["reverse"]
    )
    initial = np.asarray(
        episode["control_trace"]["initial_applied_targets"], dtype=np.float32
    )
    joint_names = tuple(ACTUATOR_JOINT_ORDER)
    leg_indices = np.asarray(
        [index for index, name in enumerate(joint_names) if name not in HEAD_JOINTS]
    )
    head_indices = np.asarray(
        [index for index, name in enumerate(joint_names) if name in HEAD_JOINTS]
    )
    lower = np.asarray(
        [
            SAFE_JOINT_LIMITS[name][0]
            if name in SAFE_JOINT_LIMITS
            else SAFE_INIT_POS[name]
            for name in joint_names
        ],
        dtype=np.float32,
    )
    upper = np.asarray(
        [
            SAFE_JOINT_LIMITS[name][1]
            if name in SAFE_JOINT_LIMITS
            else SAFE_INIT_POS[name]
            for name in joint_names
        ],
        dtype=np.float32,
    )
    margin_lower = np.add(
        lower[leg_indices], np.float32(0.05), dtype=np.float32
    )
    margin_upper = np.subtract(
        upper[leg_indices], np.float32(0.05), dtype=np.float32
    )
    phase_scale = np.float32(
        h4pt.H4_REVERSE_TEACHER_TABLE_ROWS
        / h4pt.H4_REVERSE_SOURCE_PERIOD_BINS
    )
    source_rate = np.float32(
        h4pt.H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS / float(phase_scale)
    )
    source_before = np.empty(300, dtype=np.float32)
    source_after = np.empty(300, dtype=np.float32)
    source_before[0] = np.float32(
        h4pt.H4_REVERSE_TEACHER_ENTRY_PHASE_BINS / float(phase_scale)
    )
    for index in range(300):
        source_after[index] = np.remainder(
            np.add(source_before[index], source_rate, dtype=np.float32),
            np.float32(h4pt.H4_REVERSE_SOURCE_PERIOD_BINS),
            dtype=np.float32,
        )
        if index + 1 < 300:
            source_before[index + 1] = source_after[index]
    table_phase = np.remainder(
        np.multiply(source_after, phase_scale, dtype=np.float32),
        np.float32(h4pt.H4_REVERSE_TEACHER_TABLE_ROWS),
        dtype=np.float32,
    )
    teacher_targets = h4pt._interpolate_reverse_teacher_float32(
        h4pt._pinned_reverse_teacher_table_float32(), table_phase
    )
    raw_action = np.zeros((300, 14), dtype=np.float32)
    raw_action[:, leg_indices] = np.float32(0.10)
    applied_action = raw_action.copy()
    delayed_action = applied_action.copy()
    bounded = np.clip(delayed_action, np.float32(-1.0), np.float32(1.0))
    proposed = np.asarray(
        teacher_targets.astype(np.float64)
        + float(np.float32(h4pt.H4_REVERSE_RESIDUAL_SCALE))
        * bounded.astype(np.float64),
        dtype=np.float32,
    )
    proposed[:, head_indices] = np.float32(0.0)
    safe_lower = np.add(
        initial,
        np.multiply(
            np.float32(0.9),
            np.subtract(lower, initial, dtype=np.float32),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    safe_upper = np.add(
        initial,
        np.multiply(
            np.float32(0.9),
            np.subtract(upper, initial, dtype=np.float32),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    proposed = np.clip(np.clip(proposed, safe_lower, safe_upper), lower, upper)
    upstream = np.zeros_like(proposed)
    upstream[:, leg_indices] = np.clip(
        proposed[:, leg_indices], margin_lower, margin_upper
    )
    previous_rows: list[np.ndarray] = []
    precomposer_rows: list[np.ndarray] = []
    applied_rows: list[np.ndarray] = []
    previous = initial.copy()
    for index in range(300):
        precomposer = np.add(
            previous,
            np.clip(
                np.subtract(upstream[index], previous, dtype=np.float32),
                -np.float32(0.04),
                np.float32(0.04),
            ),
            dtype=np.float32,
        )
        precomposer[head_indices] = np.float32(0.0)
        final_margin = np.zeros(14, dtype=np.float32)
        final_margin[leg_indices] = np.clip(
            precomposer[leg_indices], margin_lower, margin_upper
        )
        applied = np.zeros(14, dtype=np.float32)
        applied[leg_indices] = np.clip(
            np.add(
                previous[leg_indices],
                np.clip(
                    np.subtract(
                        final_margin[leg_indices],
                        previous[leg_indices],
                        dtype=np.float32,
                    ),
                    -np.float32(0.04),
                    np.float32(0.04),
                ),
                dtype=np.float32,
            ),
            lower[leg_indices],
            upper[leg_indices],
        )
        previous_rows.append(previous)
        precomposer_rows.append(precomposer)
        applied_rows.append(applied)
        previous = applied
    applied_targets = np.stack(applied_rows)
    control_trace = {
        "source_dtype": "float32",
        "initial_applied_targets": initial,
        "raw_action": raw_action,
        "applied_action": applied_action,
        "preclip_targets": proposed,
        "margin_clipped_targets": np.stack(precomposer_rows),
        "applied_targets": applied_targets,
        "previous_targets": np.stack(previous_rows),
        "joint_qpos": applied_targets.copy(),
        "reverse_teacher_source_phase_before": source_before,
        "reverse_teacher_table_phase": table_phase,
        "reverse_teacher_table_targets": teacher_targets.copy(),
        "reverse_action_delay_index": np.zeros(300, dtype=np.int32),
        "reverse_delayed_applied_action": delayed_action,
        "reverse_upstream_margin_targets": upstream,
        "reverse_precomposer_active": np.ones(300, dtype=bool),
    }
    episode["reverse_composition_contract"] = {
        "schema_version": 1,
        "semantics": h4pt.H4_REVERSE_COMPOSITION_TRACE_SEMANTICS,
        "selected_reverse_teacher_sha256": (
            PINNED_SELECTED_REVERSE_TEACHER_SHA256
        ),
        "reverse_composition_authorization_sha256": (
            PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
        ),
        "teacher_table_rows": h4pt.H4_REVERSE_TEACHER_TABLE_ROWS,
        "teacher_entry_phase_preincrement_bins": (
            h4pt.H4_REVERSE_TEACHER_ENTRY_PHASE_BINS
        ),
        "teacher_phase_advance_bins_per_control": (
            h4pt.H4_REVERSE_TEACHER_PHASE_ADVANCE_BINS
        ),
        "source_period_bins": h4pt.H4_REVERSE_SOURCE_PERIOD_BINS,
        "residual_scale": h4pt.H4_REVERSE_RESIDUAL_SCALE,
        "action_delay_min": h4pt.H4_REVERSE_ACTION_DELAY_MIN,
        "action_delay_max_exclusive": (
            h4pt.H4_REVERSE_ACTION_DELAY_MAX_EXCLUSIVE
        ),
        "step_entry_physical_command_x_mps": -0.05,
    }
    episode["control_trace"] = control_trace
    episode["safety_audit"] = rederive_central_safety_audit_from_control_trace(
        episode
    )
    episode["h4_control_contract"] = rederive_h4_control_contract(episode)
    episode["control_trace"] = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in control_trace.items()
    }
    return episode


def _gait_metrics() -> dict[str, Any]:
    return {
        "measurement_complete": True,
        "measurement_schema_version": 2,
        "sample_count": 3001,
        "duration_s": 6.0,
        "physics_timestep_s": 0.002,
        "maximum_timestep_error_s": 0.0,
        "trunk_pose_measurement_source": "synthetic_complete_pose",
        "trunk_yaw_sample_count": 3001,
        "contact_force_sample_count": 3001,
        "contact_velocity_payload_sample_count": 3001,
        "contact_state_source": "normal_force_schmitt",
        "stance_slip_measurement_source": "force_weighted_contact_point_jacobian",
        "steady_linear_tracking_ratio": 1.0,
        "steady_cross_drift_mps": 0.0,
        "uncommanded_yaw_rate_radps": 0.0,
        "uncommanded_heading_drift_rad": 0.0,
        "single_support_rate": 0.4,
        "flight_rate": 0.0,
        "stance_slip_rms_mps": 0.0,
        "stance_slip_p95_mps": 0.0,
        "maximum_per_stance_cumulative_slip_m": 0.0,
        "alternating_touchdown_fraction": 1.0,
        "contact_duty_imbalance": 0.0,
        "step_count_imbalance": 0,
    }


def test_h4_control_contract_uses_bit_exact_guard_output_not_endpoint_rounding() -> None:
    episode = _safety_episode()
    contract = rederive_h4_control_contract(episode)
    assert contract["passed"] is True
    assert contract["guard_output_mismatch_joint_sample_count"] == 0
    assert contract["float32_slew_violation_count"] == 0
    assert contract["checks"][
        "endpoint_delta_within_explicit_float32_rounding_bound"
    ] is True

    tampered = copy.deepcopy(episode)
    original = np.float32(tampered["control_trace"]["applied_targets"][-1][0])
    one_ulp_larger = np.nextafter(original, np.float32(np.inf))
    tampered["control_trace"]["applied_targets"][-1][0] = float(
        one_ulp_larger
    )
    tampered_contract = rederive_h4_control_contract(tampered)
    assert tampered_contract["passed"] is False
    assert tampered_contract["checks"][
        "applied_targets_exact_float32_guard_output"
    ] is False
    assert tampered_contract["guard_output_mismatch_joint_sample_count"] == 1
    assert tampered_contract["float32_slew_violation_count"] == 1
    # A one-ULP endpoint perturbation can remain under the diagnostic rounding
    # bound; exact guard-output equality must still reject it.
    assert tampered_contract["checks"][
        "endpoint_delta_within_explicit_float32_rounding_bound"
    ] is True


def test_reverse_two_stage_composition_has_exact_positive_parity() -> None:
    episode = _reverse_safety_episode()
    contract = rederive_h4_control_contract(episode)
    assert contract["schema_version"] == 3
    assert contract["passed"] is True
    assert contract["reverse_composition"][
        "precomposer_outside_margin_joint_sample_count"
    ] == 1
    assert contract["desired_margin_violation_count"] == 0
    assert contract["authorized_startup_joint_sample_count"] == 1
    assert contract["unauthorized_startup_joint_sample_count"] == 0
    assert rederive_h4_safety_acceptance(episode)["passed"] is True


def test_reverse_teacher_table_one_ulp_drift_is_rejected() -> None:
    episode = _reverse_safety_episode()
    original = np.float32(
        episode["control_trace"]["reverse_teacher_table_targets"][-1][0]
    )
    episode["control_trace"]["reverse_teacher_table_targets"][-1][0] = float(
        np.nextafter(original, np.float32(np.inf))
    )
    contract = rederive_h4_control_contract(episode)
    assert contract["passed"] is False
    assert contract["checks"][
        "reverse_teacher_table_targets_exact_pinned_interpolation"
    ] is False


def test_reverse_teacher_phase_preincrement_drift_is_rejected() -> None:
    episode = _reverse_safety_episode()
    original = np.float32(
        episode["control_trace"]["reverse_teacher_table_phase"][17]
    )
    episode["control_trace"]["reverse_teacher_table_phase"][17] = float(
        np.nextafter(original, np.float32(np.inf))
    )
    contract = rederive_h4_control_contract(episode)
    assert contract["passed"] is False
    assert contract["checks"][
        "reverse_teacher_table_phase_exact_after_preincrement"
    ] is False


def test_reverse_residual_scale_drift_is_rejected() -> None:
    episode = _reverse_safety_episode()
    episode["reverse_composition_contract"]["residual_scale"] = float(
        np.nextafter(
            np.float32(h4pt.H4_REVERSE_RESIDUAL_SCALE),
            np.float32(np.inf),
        )
    )
    contract = rederive_h4_control_contract(episode)
    assert contract["passed"] is False
    assert contract["checks"]["reverse_composition_contract_exact"] is False


def test_reverse_slew_precomposer_drift_is_rejected() -> None:
    episode = _reverse_safety_episode()
    original = np.float32(
        episode["control_trace"]["margin_clipped_targets"][-1][0]
    )
    episode["control_trace"]["margin_clipped_targets"][-1][0] = float(
        np.nextafter(original, np.float32(np.inf))
    )
    contract = rederive_h4_control_contract(episode)
    assert contract["passed"] is False
    assert contract["checks"]["reverse_slew_precomposer_output_exact"] is False


def test_reverse_final_guard_output_drift_is_rejected() -> None:
    episode = _reverse_safety_episode()
    original = np.float32(
        episode["control_trace"]["applied_targets"][-1][0]
    )
    episode["control_trace"]["applied_targets"][-1][0] = float(
        np.nextafter(original, np.float32(np.inf))
    )
    contract = rederive_h4_control_contract(episode)
    assert contract["passed"] is False
    assert contract["checks"][
        "applied_targets_exact_float32_guard_output"
    ] is False


def test_reverse_nonzero_action_delay_is_rejected() -> None:
    episode = _reverse_safety_episode()
    episode["control_trace"]["reverse_action_delay_index"][11] = 1
    contract = rederive_h4_control_contract(episode)
    assert contract["passed"] is False
    assert contract["checks"]["reverse_action_delay_index_exact_zero"] is False


def test_reverse_legacy_trace_without_schema3_composition_fields_fails_closed() -> None:
    episode = _reverse_safety_episode()
    del episode["control_trace"]["reverse_teacher_table_targets"]
    with pytest.raises(
        ValueError,
        match="reverse composition trace incomplete",
    ):
        rederive_h4_control_contract(episode)


def _fake_rederive(_metrics: Any) -> dict[str, Any]:
    return {"passed": True, "checks": {"full_current_p0": True}}


def _strict_episode(seed: int) -> dict[str, Any]:
    episode = {
        **_safety_episode(),
        "seed": seed,
        "segment_id": f"h4_forward_seed{seed}_6s",
        "expert": "forward",
        "physical_command_mps_radps": [0.05, 0.0, 0.0],
        "source_segment_kind": "H4_STRICT_6S",
        "gait_quality_metrics": _gait_metrics(),
        "gait_quality_acceptance": _fake_rederive(None),
    }
    episode["metrics"] = legacy_metrics_from_gait_quality(
        episode["gait_quality_metrics"]
    )
    episode["h4_safety_acceptance"] = rederive_h4_safety_acceptance(episode)
    episode["strict_passed"] = True
    return episode


def _strict_artifact(
    *,
    candidate: dict[str, Any] | None = None,
    source_hashes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seeds = H4_STRICT_SEEDS["forward"]
    training_provenance = {
        "schema_version": 1,
        "training_execution_provider": "JAX_GPU",
        "platform": "gpu",
        "cross_bound_config_manifest_result": True,
        "passed": True,
    }
    sources = source_hashes or {
        "runner": {"path": str(Path(__file__).resolve()), "sha256": "5" * 64}
    }
    candidate_record = candidate or {
        "run_name": "candidate",
        "expert": "forward",
        "status": "WIRING_PASS",
        "activity": "PPO_WIRING_TRAINING",
        "final_params_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "resolved_config_sha256": "3" * 64,
        "source_and_teacher_hashes_sha256": canonical_json_sha256(sources),
        "training_provenance_sha256": canonical_json_sha256(training_provenance),
    }
    candidate_record = copy.deepcopy(candidate_record)
    candidate_record.setdefault(
        "training_provenance_sha256",
        canonical_json_sha256(training_provenance),
    )
    central = {"safe_gait_experts/gait_quality.py": "4" * 64}
    return {
        "schema_version": 1,
        "artifact_kind": STRICT_ARTIFACT_KIND,
        "hardware_deployment": "PROHIBITED",
        "execution_provider": "CPU",
        "candidate": candidate_record,
        "evaluation_contract": {
            "fixed_seeds": list(seeds),
            "physical_command_mps_radps": list(H4_STRICT_COMMANDS["forward"]),
            "duration_s": 6.0,
            "control_timestep_s": 0.02,
            "physics_timestep_s": 0.002,
            "control_tick_count": 300,
            "physics_substep_count": 3000,
            "gait_sample_count": 3001,
            "gait_quality_semantics": (
                "FULL_CURRENT_P0_RECOMPUTED_FROM_N_PLUS_ONE_SUBSTEP_TRACE"
            ),
            "reverse_composition": None,
        },
        "central_hashes": central,
        "episodes": [_strict_episode(seed) for seed in seeds],
        "summary": {
            "passing_seed_count": 3,
            "passing_seeds": list(seeds),
            "all_three_strict_pass": True,
        },
        "runtime_provenance": {
            "execution_provider": "CPU",
            "jax_default_backend": "cpu",
            "jax_devices": [{"description": "TFRT_CPU_0", "platform": "cpu"}],
            "candidate_manifest_sha256": candidate_record["manifest_sha256"],
            "candidate_final_params_sha256": candidate_record[
                "final_params_sha256"
            ],
            "candidate_resolved_config_sha256": candidate_record[
                "resolved_config_sha256"
            ],
            "source_and_teacher_hashes": sources,
            "training_provenance": training_provenance,
            "central_hashes": central,
            "evaluation_source_hashes_pre": {"evaluator.py": "7" * 64},
            "evaluation_source_hashes_post": {"evaluator.py": "7" * 64},
            "pre_post_source_hashes_unchanged": True,
        },
    }


def test_safety_and_full_artifact_are_rederived_not_trusted() -> None:
    seed = H4_STRICT_SEEDS["forward"][0]
    episode = _strict_episode(seed)
    audit = validate_h4_strict_episode(
        episode,
        expert="forward",
        expected_seed=seed,
        gait_quality_rederive=_fake_rederive,
    )
    assert audit["strict_passed"] is True
    tampered = copy.deepcopy(episode)
    tampered["guard_call_audit"]["total_guard_calls"] = 299
    tampered_audit = validate_h4_strict_episode(
        tampered,
        expert="forward",
        expected_seed=seed,
        gait_quality_rederive=_fake_rederive,
    )
    assert tampered_audit["strict_passed"] is False
    assert not tampered_audit["safety"]["checks"]["guard_called_once_per_tick"]
    reset_tampered = copy.deepcopy(episode)
    reset_tampered["reset_audit"]["comparison_semantics"] = "FLOAT64_TOLERANCE"
    reset_audit = validate_h4_strict_episode(
        reset_tampered,
        expert="forward",
        expected_seed=seed,
        gait_quality_rederive=_fake_rederive,
    )
    assert reset_audit["structural_checks"][
        "h4_control_contract_exactly_rederived"
    ] is True
    assert reset_audit["safety"]["checks"]["exact_safe_init_reset"] is True
    assert reset_audit["safety"]["checks"][
        "reset_audit_matches_float32_raw_trace"
    ] is False
    assert reset_audit["strict_passed"] is False
    partial = copy.deepcopy(episode)
    partial["gait_quality_metrics"].pop("measurement_complete")
    partial_audit = validate_h4_strict_episode(
        partial,
        expert="forward",
        expected_seed=seed,
        gait_quality_rederive=_fake_rederive,
    )
    assert partial_audit["strict_passed"] is False
    assert partial_audit["structural_checks"]["gait_measurement_complete"] is False


def test_strict_artifact_requires_exact_n_plus_one_and_fixed_ordered_seeds() -> None:
    seeds = H4_STRICT_SEEDS["forward"]
    artifact = _strict_artifact()
    central = artifact["central_hashes"]
    audit = validate_h4_strict_artifact(
        artifact,
        current_central_hashes=central,
        require_all_three_pass=True,
        gait_quality_rederive=_fake_rederive,
    )
    assert audit["passing_seed_count"] == 3
    with pytest.raises(ValueError, match="runtime provenance failed"):
        validate_h4_strict_artifact(
            artifact,
            current_central_hashes=central,
            current_evaluation_hashes={"evaluator.py": "8" * 64},
            gait_quality_rederive=_fake_rederive,
        )
    wrong_n = copy.deepcopy(artifact)
    wrong_n["episodes"][0]["gait_quality_metrics"]["sample_count"] = 3000
    with pytest.raises(ValueError, match="summary is not exactly recomputed"):
        validate_h4_strict_artifact(
            wrong_n,
            current_central_hashes=central,
            gait_quality_rederive=_fake_rederive,
        )
    wrong_order = copy.deepcopy(artifact)
    wrong_order["episodes"].reverse()
    with pytest.raises(ValueError, match="summary is not exactly recomputed"):
        validate_h4_strict_artifact(
            wrong_order,
            current_central_hashes=central,
            gait_quality_rederive=_fake_rederive,
        )
    wrong_seed = copy.deepcopy(artifact)
    wrong_seed["episodes"][0]["seed"] += 1
    with pytest.raises(ValueError, match="summary is not exactly recomputed"):
        validate_h4_strict_artifact(
            wrong_seed,
            current_central_hashes=central,
            gait_quality_rederive=_fake_rederive,
        )
    wrong_expert = copy.deepcopy(artifact)
    wrong_expert["candidate"]["expert"] = "sideways"
    with pytest.raises(ValueError, match="expert is invalid"):
        validate_h4_strict_artifact(
            wrong_expert,
            current_central_hashes=central,
            gait_quality_rederive=_fake_rederive,
        )


def test_promotion_rejects_any_external_self_reported_baseline(tmp_path: Path) -> None:
    candidate = _strict_artifact()
    baseline = copy.deepcopy(candidate)
    baseline_sources = {
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": "6" * 64,
        }
    }
    baseline["runtime_provenance"]["source_and_teacher_hashes"] = baseline_sources
    baseline["candidate"]["source_and_teacher_hashes_sha256"] = (
        canonical_json_sha256(baseline_sources)
    )
    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    _write_json(candidate_path, candidate)
    _write_json(baseline_path, baseline)
    with pytest.raises(ValueError, match="external/self-reported baseline"):
        build_promotion_evidence(
            candidate_artifact_path=candidate_path,
            baseline_artifact_path=baseline_path,
            bundle=object(),  # rejected by source preflight before bundle use
            current_central_hashes=candidate["central_hashes"],
        )


def test_completed_artifact_requires_bound_integrated_official_v22_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _strict_artifact()
    artifact["candidate"]["status"] = "COMPLETED"
    artifact["candidate"]["activity"] = "PPO_PILOT_TRAINING"
    baseline_source = tmp_path / "official_v22"
    baseline_source.mkdir()
    artifact["official_v22_baseline"] = {
        "source_checkpoint": {
            "kind": "OFFICIAL_FROZEN_V22_BRAX_CHECKPOINT",
            "path": str(baseline_source.resolve()),
            "sha256_tree_pre": h4pt.PINNED_V22_PARENT_TREE_SHA256,
            "sha256_tree_post": h4pt.PINNED_V22_PARENT_TREE_SHA256,
            "unchanged": True,
        },
        "transplant_audit": {
            "method": "OFFICIAL_V22_101_TO_116_ZERO_ROW_TRANSPLANT",
            "source_actor_width": 101,
            "target_actor_width": 116,
            "source_critic_width": 212,
            "target_critic_width": 227,
            "insert_offset": 101,
            "inserted_feature_count": 15,
            "optimizer_updates": 0,
            "checks": {"actor_new_15_rows_exact_zero": True},
            "passed": True,
        },
        "transplanted_params_numeric_sha256": "8" * 64,
        "evaluation_process": (
            "SAME_PROCESS_ENVIRONMENT_CONTRACT_AND_FIXED_SEEDS_AS_CANDIDATE"
        ),
        "optimizer_updates": 0,
        "policy_inference": "BRAX_DETERMINISTIC_NORMAL_TANH_ACTOR116",
        "episodes": copy.deepcopy(artifact["episodes"]),
        "summary": copy.deepcopy(artifact["summary"]),
    }
    monkeypatch.setattr(
        h4pt, "sha256_tree", lambda _path: h4pt.PINNED_V22_PARENT_TREE_SHA256
    )
    audit = validate_h4_strict_artifact(
        artifact,
        current_central_hashes=artifact["central_hashes"],
        gait_quality_rederive=_fake_rederive,
    )
    assert audit["official_v22_baseline_passing_seed_count"] == 3
    missing = copy.deepcopy(artifact)
    missing["official_v22_baseline"] = None
    with pytest.raises(ValueError, match="integrated baseline"):
        validate_h4_strict_artifact(
            missing,
            current_central_hashes=artifact["central_hashes"],
            gait_quality_rederive=_fake_rederive,
        )
    tampered = copy.deepcopy(artifact)
    tampered["official_v22_baseline"]["transplant_audit"]["checks"][
        "actor_new_15_rows_exact_zero"
    ] = False
    with pytest.raises(ValueError, match="transplant drifted"):
        validate_h4_strict_artifact(
            tampered,
            current_central_hashes=artifact["central_hashes"],
            gait_quality_rederive=_fake_rederive,
        )
