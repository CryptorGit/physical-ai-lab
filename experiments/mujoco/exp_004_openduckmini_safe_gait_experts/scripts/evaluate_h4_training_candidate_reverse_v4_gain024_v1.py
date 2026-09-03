"""Exact diagnostic strict evaluator adapter for the frozen reverse-v4 gain-0.24 run.

This adapter never mutates the frozen evaluator or post-training module.  It
clones the small affected call graph with ``types.FunctionType`` and supplies
the candidate's authorized 0.24 residual scale only to runtime expectation and
host rederivation globals.  The original 0.12 base-composition authorization
loader remains unchanged.  Output is diagnostic-only and is never promotion
evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType, ModuleType
from typing import Any, Mapping, Sequence
import hashlib


EXP_ROOT = Path(__file__).resolve().parents[1]
BASE_EVALUATOR_PATH = EXP_ROOT / "scripts" / "evaluate_h4_training_candidate.py"
POST_TRAINING_PATH = EXP_ROOT / "safe_gait_experts" / "h4_post_training.py"
ADAPTER_PATH = Path(__file__).resolve()
ADAPTER_AUTHORIZATION_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_iteration_v4_gain024_strict_evaluator_adapter_v1_authorization.json"
)

ADAPTER_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_"
    "STRICT_EVALUATOR_ADAPTER_V1"
)
TRAINING_CONTRACT_ID = (
    "H4_REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_250K_FROM_V22"
)
DIAGNOSTIC_ARTIFACT_KIND = (
    "openduckmini_h4_reverse_iteration_v4_gain024_strict_evaluation_diagnostic"
)
LEGACY_RESIDUAL_SCALE = 0.12
GAIN024_RESIDUAL_SCALE = 0.24
PINNED_ADAPTER_AUTHORIZATION_SHA256 = (
    "f2531e5962818fdb3b6d9853447c288bb00c52917f6e72ad7b6a0ac5f4f085c0"
)
PINNED_REVERSE_V4_AUTHORIZATION_SHA256 = (
    "93e3a53d5b601987df7a4efb84de5fb0ae499dc0ea0dc93acdbb074d96510312"
)
PINNED_BASE_COMPOSITION_AUTHORIZATION_SHA256 = (
    "082405e34b8a46e7d4a9ccf7b8c0729871fee1eb202b4a1ed8c758b2c7a52900"
)

CANDIDATE_ROOT = (
    EXP_ROOT
    / "artifacts"
    / "h4_iteration_v4_training_runs_20260809_bool_exact_v2"
    / "reverse"
    / "h4_reverse_250k_seed20260810_iteration_v4_residual_transfer_gain_024_level4_v1"
)
TRUSTED_RUN_ROOT = (
    EXP_ROOT / "artifacts" / "h4_iteration_v4_training_runs_20260809_bool_exact_v2"
)
EXPECTED_PARAMS_PATH = CANDIDATE_ROOT / "final_params.pkl"
EXPECTED_MANIFEST_PATH = CANDIDATE_ROOT / "run_manifest.json"
EXPECTED_CONFIG_PATH = CANDIDATE_ROOT / "resolved_config.json"
EXPECTED_RESULT_PATH = CANDIDATE_ROOT / "run_result.json"
EXPECTED_CURVE_PATH = CANDIDATE_ROOT / "training_curve.csv"
EXPECTED_OUTPUT_PATH = CANDIDATE_ROOT / "h4_integrated_strict_3x6s_v1.json"
DEFAULT_SOURCE_ROOT = Path("/home/user/openduck_training_20260729")
DEFAULT_GENERATED_ROOT = EXP_ROOT / "artifacts" / "generated_playground"
DEFAULT_V22_PARENT_CHECKPOINT = Path(
    "/home/user/openduck_training_runs/"
    "calibrated_hybrid_yaw_cost_v22_300m/2026_07_29_154427_10485760"
)

PINNED_CANDIDATE_FILES = {
    "candidate_params": (EXPECTED_PARAMS_PATH, "b0aa6fb639da3b8cc9e9bed27eee73f5a216e357e99ef030f5de07cd79d9e417"),
    "candidate_manifest": (EXPECTED_MANIFEST_PATH, "1ee678f682a8487181d7deaa3ada16e56df3e43b74a2f2f19e3e26bd31cac99c"),
    "candidate_config": (EXPECTED_CONFIG_PATH, "0217cb992b53d25dbc9e7333cc6c9fbbb40e93693f7e11d891bd2b5ee2ca1ef0"),
    "candidate_result": (EXPECTED_RESULT_PATH, "5954054c4f85dae493fed2a04b49ed66ab259af58e7a1896eba32b3430c2a20c"),
    "candidate_training_curve": (EXPECTED_CURVE_PATH, "b05f4fc04c0d23ebb7e7a8c335be0c754d72c089fa56b9b4948593df56ea2cda"),
    "failed_evaluator_stdout": (CANDIDATE_ROOT / "h4_integrated_strict_3x6s_v1.stdout.log", "566fbc64d94d0892ba71d730aa60fbd7e02cd6b79be9d6f1d3d33117997cae7c"),
    "failed_evaluator_stderr": (CANDIDATE_ROOT / "h4_integrated_strict_3x6s_v1.stderr.log", "2ada77a36600433aeeff122ae780f7db8a6605464b85fa59ce291a4589986a16"),
}

PINNED_FROZEN_SOURCES = {
    "h4_candidate_evaluator": (BASE_EVALUATOR_PATH, "c214d086e6d66f6f9f98c7268481899e4133961dcc5355d738d4cd134a82e6ae"),
    "h4_post_training": (POST_TRAINING_PATH, "afdfcf9da43a7a7e5824ce7562c489b5e5e20a32e83af817be9e80d740a27b3f"),
    "h4_training_alignment": (EXP_ROOT / "safe_gait_experts" / "h4_training_alignment.py", "872a11a817bb068e3a0819c0afca12ae9e7f2dbfcc103c6569b9081b8d5fbebb"),
    "h4_runner": (EXP_ROOT / "scripts" / "train_h4_aligned_expert.py", "b15b9692a72deadd34790d442f4ab4263c3f987255173566a62438e0d380da13"),
    "h4_no_ppo_smoke": (EXP_ROOT / "scripts" / "smoke_h4_training_alignment.py", "410924542bac85f70de3a4055f617a85e93eb841cd403f5280699778ac96710d"),
    "central_evaluator": (EXP_ROOT / "scripts" / "evaluate_routed_transitions.py", "31fb8846fc6267f28d032bca164dee2c872bfb484ebe272850100834bf1b1a9b"),
    "central_gait_quality": (EXP_ROOT / "safe_gait_experts" / "gait_quality.py", "b28e1ceb4cb6406411150bbad772a78203b8163bf10adcd79a5f31f83da5f2de"),
    "central_routed_evaluation": (EXP_ROOT / "safe_gait_experts" / "routed_evaluation.py", "f25ed858fbb5753fdcfd9e76f08396d0f09f95ac4696eb2e50cb5c128b80db09"),
    "reverse_v4_authorization": (EXP_ROOT / "artifacts" / "h4_reverse_iteration_v4_residual_transfer_gain_024_authorization.json", PINNED_REVERSE_V4_AUTHORIZATION_SHA256),
    "base_reverse_composition_authorization": (EXP_ROOT / "artifacts" / "h4_reverse_training_composition_authorization_v1.json", PINNED_BASE_COMPOSITION_AUTHORIZATION_SHA256),
}

ADAPTER_SOURCE_KEY = "scripts/evaluate_h4_training_candidate_reverse_v4_gain024_v1.py"
ADAPTER_AUTHORIZATION_SOURCE_KEY = (
    "artifacts/h4_reverse_iteration_v4_gain024_strict_evaluator_adapter_v1_authorization.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json_strict(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"authorization field is missing: {'.'.join(keys)}")
        current = current[key]
    return current


def validate_adapter_authorization_payload(payload: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version",
        "artifact_kind",
        "status",
        "hardware_deployment",
        "authorization",
        "scope",
        "causal_basis",
        "candidate_binding",
        "authorization_bindings",
        "frozen_source_bindings",
        "adapter_contract",
        "provenance_contract",
        "decision",
    }
    if set(payload) != expected_top:
        raise ValueError("adapter authorization top-level closure drifted")
    checks = {
        "schema": payload.get("schema_version") == 1,
        "kind": payload.get("artifact_kind")
        == "openduckmini_h4_reverse_iteration_v4_gain024_strict_evaluator_adapter_authorization",
        "status": payload.get("status")
        == "AUTHORIZED_EXACT_DIAGNOSTIC_STRICT_EVALUATION_ONLY",
        "hardware": payload.get("hardware_deployment") == "PROHIBITED",
        "contract": _nested(payload, "scope", "contract_id") == ADAPTER_CONTRACT_ID,
        "training_contract": _nested(payload, "scope", "training_contract_id")
        == TRAINING_CONTRACT_ID,
        "expert": _nested(payload, "scope", "expert") == "reverse",
        "adapter_path": _nested(payload, "scope", "adapter_source_path")
        == ADAPTER_SOURCE_KEY,
        "diagnostic_kind": _nested(payload, "scope", "evaluation_artifact_kind")
        == DIAGNOSTIC_ARTIFACT_KIND,
        "clone_method": _nested(payload, "scope", "method")
        == "FUNCTIONTYPE_CLONED_GLOBALS_WITHOUT_MODULE_MUTATION",
        "promotion_ineligible": _nested(payload, "scope", "promotion_eligible")
        is False,
        "base_scale": _nested(payload, "causal_basis", "base_composition_maximum_residual_scale")
        == LEGACY_RESIDUAL_SCALE,
        "runtime_scale": _nested(payload, "causal_basis", "candidate_effective_runtime_residual_scale")
        == GAIN024_RESIDUAL_SCALE,
        "host_scale": _nested(payload, "causal_basis", "host_rederivation_residual_scale")
        == GAIN024_RESIDUAL_SCALE,
        "candidate_root": _nested(payload, "candidate_binding", "root_relative_path")
        == str(CANDIDATE_ROOT.relative_to(EXP_ROOT)).replace("\\", "/"),
        "candidate_status": _nested(payload, "candidate_binding", "status")
        == "COMPLETED",
        "candidate_activity": _nested(payload, "candidate_binding", "activity")
        == "PPO_PILOT_TRAINING",
        "candidate_qualification": _nested(payload, "candidate_binding", "qualification_use")
        == "AUTHORIZED_250K_PILOT",
        "candidate_flag": _nested(payload, "candidate_binding", "reverse_iteration_v4_residual_transfer_gain_024")
        is True,
        "candidate_scale": _nested(payload, "candidate_binding", "backward_residual_scale")
        == GAIN024_RESIDUAL_SCALE,
        "standard_output": _nested(payload, "candidate_binding", "standard_output")
        == EXPECTED_OUTPUT_PATH.name,
        "no_overwrite": _nested(payload, "candidate_binding", "overwrite_allowed")
        is False,
        "v4_auth": _nested(payload, "authorization_bindings", "reverse_iteration_v4", "sha256")
        == PINNED_REVERSE_V4_AUTHORIZATION_SHA256,
        "base_auth": _nested(payload, "authorization_bindings", "base_reverse_composition", "sha256")
        == PINNED_BASE_COMPOSITION_AUTHORIZATION_SHA256,
        "base_auth_scale": _nested(payload, "authorization_bindings", "base_reverse_composition", "maximum_residual_scale")
        == LEGACY_RESIDUAL_SCALE,
        "promotion_builder_prohibited": _nested(payload, "adapter_contract", "promotion_builder_or_output_prohibited")
        is True,
        "write_new_only": _nested(payload, "adapter_contract", "write_new_standard_json_only")
        is True,
        "decision_promotion": _nested(payload, "decision", "promotion") == "PROHIBITED",
    }
    expected_authorization = {
        "exact_candidate_diagnostic_strict_evaluation": True,
        "ppo_training": False,
        "promotion_evidence": False,
        "candidate_adoption": False,
        "package_release": False,
        "hardware": False,
    }
    checks["authorization_exact"] = payload.get("authorization") == expected_authorization
    for name, (_path, expected_sha) in PINNED_CANDIDATE_FILES.items():
        record_name = {
            "candidate_params": "final_params",
            "candidate_manifest": "manifest",
            "candidate_config": "resolved_config",
            "candidate_result": "run_result",
            "candidate_training_curve": "training_curve",
        }.get(name)
        if record_name is not None:
            checks[f"{name}_sha"] = _nested(
                payload, "candidate_binding", record_name, "sha256"
            ) == expected_sha
    for name, (_path, expected_sha) in PINNED_FROZEN_SOURCES.items():
        if name in {"reverse_v4_authorization", "base_reverse_composition_authorization"}:
            continue
        checks[f"{name}_sha"] = _nested(
            payload, "frozen_source_bindings", name, "sha256"
        ) == expected_sha
    if not all(checks.values()):
        raise ValueError(f"adapter authorization semantic binding failed: {checks}")


def load_and_validate_adapter_authorization(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved != ADAPTER_AUTHORIZATION_PATH.resolve():
        raise ValueError("adapter authorization path must remain exact")
    actual_sha = sha256_file(resolved)
    if actual_sha != PINNED_ADAPTER_AUTHORIZATION_SHA256:
        raise ValueError(f"adapter authorization SHA256 drifted: {actual_sha}")
    payload = load_json_strict(resolved)
    validate_adapter_authorization_payload(payload)
    return payload


def _verify_file_bindings(
    bindings: Mapping[str, tuple[Path, str]],
) -> dict[str, str]:
    actual: dict[str, str] = {}
    for label, (path, expected_sha) in bindings.items():
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise ValueError(f"pinned file is missing: {label}: {resolved}")
        digest = sha256_file(resolved)
        if digest != expected_sha:
            raise ValueError(f"pinned file SHA256 drifted: {label}: {digest}")
        actual[label] = digest
    return actual


def _validate_candidate_metadata(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    common_checks = {
        "config_training_contract": config.get("training_contract_id")
        == TRAINING_CONTRACT_ID,
        "manifest_training_contract": manifest.get("training_contract_id")
        == TRAINING_CONTRACT_ID,
        "result_training_contract": result.get("training_contract_id")
        == TRAINING_CONTRACT_ID,
        "config_expert": config.get("expert") == "reverse",
        "manifest_expert": manifest.get("expert") == "reverse",
        "result_expert": result.get("expert") == "reverse",
        "manifest_completed": manifest.get("status") == "COMPLETED",
        "result_completed": result.get("status") == "COMPLETED",
        "manifest_activity": manifest.get("activity") == "PPO_PILOT_TRAINING",
        "result_activity": result.get("activity") == "PPO_PILOT_TRAINING",
        "config_qualification": config.get("qualification_use")
        == "AUTHORIZED_250K_PILOT",
        "manifest_qualification": manifest.get("qualification_use")
        == "AUTHORIZED_250K_PILOT",
        "result_qualification": result.get("qualification_use")
        == "AUTHORIZED_250K_PILOT",
        "config_gain024_flag": config.get(
            "reverse_iteration_v4_residual_transfer_gain_024"
        )
        is True,
        "manifest_gain024_flag": manifest.get(
            "reverse_iteration_v4_residual_transfer_gain_024"
        )
        is True,
        "result_gain024_flag": result.get(
            "reverse_iteration_v4_residual_transfer_gain_024"
        )
        is True,
        "config_scale_exact": config.get("backward_residual_scale")
        == GAIN024_RESIDUAL_SCALE,
        "config_v4_auth": _nested(
            config,
            "reverse_iteration_v4_residual_transfer_gain_024_authorization",
            "sha256",
        )
        == PINNED_REVERSE_V4_AUTHORIZATION_SHA256,
        "result_v4_auth": result.get(
            "reverse_iteration_v4_residual_transfer_gain_024_authorization_sha256"
        )
        == PINNED_REVERSE_V4_AUTHORIZATION_SHA256,
    }
    if not all(common_checks.values()):
        raise ValueError(f"exact reverse-v4 candidate metadata drifted: {common_checks}")


def _validate_base_authorization_semantics() -> None:
    base = load_json_strict(
        PINNED_FROZEN_SOURCES["base_reverse_composition_authorization"][0]
    )
    v4 = load_json_strict(PINNED_FROZEN_SOURCES["reverse_v4_authorization"][0])
    checks = {
        "base_maximum_residual_scale": _nested(
            base, "composition_contract", "maximum_residual_scale"
        )
        == LEGACY_RESIDUAL_SCALE,
        "v4_contract": _nested(v4, "scope", "contract_id")
        == TRAINING_CONTRACT_ID,
        "v4_gain_from": _nested(
            v4, "teacher_and_guard_contract", "backward_residual_scale_iteration_v3"
        )
        == LEGACY_RESIDUAL_SCALE,
        "v4_gain_to": _nested(
            v4, "teacher_and_guard_contract", "backward_residual_scale_iteration_v4"
        )
        == GAIN024_RESIDUAL_SCALE,
        "v4_only_delta": _nested(v4, "teacher_and_guard_contract", "only_delta")
        == "backward_residual_scale",
    }
    if not all(checks.values()):
        raise ValueError(f"reverse authorization semantics drifted: {checks}")


def _validate_exact_candidate_files() -> dict[str, str]:
    hashes = _verify_file_bindings(PINNED_CANDIDATE_FILES)
    _validate_candidate_metadata(
        load_json_strict(EXPECTED_CONFIG_PATH),
        load_json_strict(EXPECTED_MANIFEST_PATH),
        load_json_strict(EXPECTED_RESULT_PATH),
    )
    _validate_base_authorization_semantics()
    return hashes


def _clone_function(
    function: FunctionType, *, global_overrides: Mapping[str, Any]
) -> FunctionType:
    if not isinstance(function, FunctionType) or function.__closure__ is not None:
        raise TypeError("adapter requires a closure-free Python function")
    cloned_globals = dict(function.__globals__)
    cloned_globals.update(global_overrides)
    clone = FunctionType(
        function.__code__,
        cloned_globals,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=None,
    )
    clone.__kwdefaults__ = (
        dict(function.__kwdefaults__) if function.__kwdefaults__ is not None else None
    )
    clone.__annotations__ = dict(function.__annotations__)
    clone.__dict__.update(function.__dict__)
    clone.__doc__ = function.__doc__
    clone.__qualname__ = function.__qualname__
    return clone


def _module_contract_snapshot(base: ModuleType, post: ModuleType) -> dict[str, Any]:
    return {
        "base_scale": base.H4_REVERSE_RESIDUAL_SCALE,
        "post_scale": post.H4_REVERSE_RESIDUAL_SCALE,
        "base_make": base._make_environment_and_policy,
        "base_episode": base._run_episode,
        "base_run": base.run_evaluation,
        "base_control": base.rederive_h4_control_contract,
        "base_safety": base.rederive_h4_safety_acceptance,
        "base_validate": base.validate_h4_strict_artifact,
        "post_control": post.rederive_h4_control_contract,
        "post_safety": post.rederive_h4_safety_acceptance,
        "post_episode": post.validate_h4_strict_episode,
        "post_validate": post.validate_h4_strict_artifact,
    }


def _assert_module_contract_unchanged(
    base: ModuleType, post: ModuleType, expected: Mapping[str, Any]
) -> None:
    current = _module_contract_snapshot(base, post)
    if current != dict(expected):
        raise RuntimeError("frozen evaluator/post globals or function references changed")


def build_gain024_call_graph(
    base: ModuleType, post: ModuleType
) -> dict[str, FunctionType]:
    original = _module_contract_snapshot(base, post)
    identity_checks = {
        "base_scale_legacy": original["base_scale"] == LEGACY_RESIDUAL_SCALE,
        "post_scale_legacy": original["post_scale"] == LEGACY_RESIDUAL_SCALE,
        "base_control_is_post": original["base_control"] is original["post_control"],
        "base_safety_is_post": original["base_safety"] is original["post_safety"],
        "base_validate_is_post": original["base_validate"] is original["post_validate"],
    }
    if not all(identity_checks.values()):
        raise RuntimeError(f"frozen evaluator/post import contract drifted: {identity_checks}")

    control = _clone_function(
        post.rederive_h4_control_contract,
        global_overrides={"H4_REVERSE_RESIDUAL_SCALE": GAIN024_RESIDUAL_SCALE},
    )
    safety = _clone_function(
        post.rederive_h4_safety_acceptance,
        global_overrides={"rederive_h4_control_contract": control},
    )
    episode_validator = _clone_function(
        post.validate_h4_strict_episode,
        global_overrides={
            "rederive_h4_control_contract": control,
            "rederive_h4_safety_acceptance": safety,
        },
    )
    artifact_validator = _clone_function(
        post.validate_h4_strict_artifact,
        global_overrides={
            "STRICT_ARTIFACT_KIND": DIAGNOSTIC_ARTIFACT_KIND,
            "validate_h4_strict_episode": episode_validator,
        },
    )
    make_environment = _clone_function(
        base._make_environment_and_policy,
        global_overrides={"H4_REVERSE_RESIDUAL_SCALE": GAIN024_RESIDUAL_SCALE},
    )
    run_episode = _clone_function(
        base._run_episode,
        global_overrides={
            "rederive_h4_control_contract": control,
            "rederive_h4_safety_acceptance": safety,
        },
    )
    run_evaluation = _clone_function(
        base.run_evaluation,
        global_overrides={
            "STRICT_ARTIFACT_KIND": DIAGNOSTIC_ARTIFACT_KIND,
            "_make_environment_and_policy": make_environment,
            "_run_episode": run_episode,
            "validate_h4_strict_artifact": artifact_validator,
        },
    )
    _assert_module_contract_unchanged(base, post, original)
    return {
        "control": control,
        "safety": safety,
        "episode_validator": episode_validator,
        "artifact_validator": artifact_validator,
        "make_environment": make_environment,
        "run_episode": run_episode,
        "run_evaluation": run_evaluation,
    }


def _load_frozen_modules() -> tuple[ModuleType, ModuleType]:
    if str(EXP_ROOT) not in sys.path:
        sys.path.insert(0, str(EXP_ROOT))
    import safe_gait_experts.h4_post_training as post

    spec = importlib.util.spec_from_file_location(
        "exp004_h4_reverse_v4_gain024_adapter_base_evaluator_v1",
        BASE_EVALUATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen evaluator: {BASE_EVALUATOR_PATH}")
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    return base, post


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only strict evaluation of the exact reverse-v4 gain-0.24 "
            "candidate; promotion output is intentionally unsupported."
        )
    )
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--params-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trusted-run-root", type=Path, default=TRUSTED_RUN_ROOT
    )
    parser.add_argument(
        "--adapter-authorization",
        type=Path,
        default=ADAPTER_AUTHORIZATION_PATH,
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument(
        "--v22-parent-checkpoint",
        type=Path,
        default=DEFAULT_V22_PARENT_CHECKPOINT,
    )
    parser.add_argument("--platform", choices=("cpu",), default="cpu")
    return parser


def _resolve_process_start_paths(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "params",
        "manifest",
        "output",
        "trusted_run_root",
        "adapter_authorization",
        "source_root",
        "generated_root",
        "v22_parent_checkpoint",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).resolve())
    args.allow_wiring_diagnostic = False
    args.promotion_evidence_output = None
    return args


def _validate_exact_cli(
    args: argparse.Namespace, *, require_output_absent: bool = True
) -> None:
    checks = {
        "params_path": args.params == EXPECTED_PARAMS_PATH.resolve(),
        "params_sha": args.params_sha256
        == PINNED_CANDIDATE_FILES["candidate_params"][1],
        "manifest_path": args.manifest == EXPECTED_MANIFEST_PATH.resolve(),
        "manifest_sha": args.manifest_sha256
        == PINNED_CANDIDATE_FILES["candidate_manifest"][1],
        "output_path": args.output == EXPECTED_OUTPUT_PATH.resolve(),
        "trusted_root": args.trusted_run_root == TRUSTED_RUN_ROOT.resolve(),
        "adapter_authorization": args.adapter_authorization
        == ADAPTER_AUTHORIZATION_PATH.resolve(),
        "platform_cpu": args.platform == "cpu",
        "wiring_forbidden": args.allow_wiring_diagnostic is False,
        "promotion_forbidden": args.promotion_evidence_output is None,
    }
    if not all(checks.values()):
        raise ValueError(f"exact reverse-v4 gain024 adapter CLI drifted: {checks}")
    if require_output_absent and args.output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {args.output}")


def _adapter_source_hashes() -> dict[str, str]:
    return {
        ADAPTER_SOURCE_KEY: sha256_file(ADAPTER_PATH),
        ADAPTER_AUTHORIZATION_SOURCE_KEY: sha256_file(ADAPTER_AUTHORIZATION_PATH),
    }


def _augment_evaluation_hashes(
    base_hashes: Mapping[str, str], adapter_hashes: Mapping[str, str]
) -> dict[str, str]:
    result = dict(base_hashes)
    for path, digest in adapter_hashes.items():
        if path in result:
            raise ValueError(f"adapter evaluation source key collision: {path}")
        result[path] = digest
    expected_existing = {
        "scripts/evaluate_h4_training_candidate.py": PINNED_FROZEN_SOURCES[
            "h4_candidate_evaluator"
        ][1],
        "safe_gait_experts/h4_post_training.py": PINNED_FROZEN_SOURCES[
            "h4_post_training"
        ][1],
    }
    if any(result.get(path) != digest for path, digest in expected_existing.items()):
        raise ValueError("base evaluator/post source provenance drifted")
    return result


def _validate_six_episode_gain024(artifact: Mapping[str, Any]) -> None:
    candidate = artifact.get("episodes")
    baseline_record = artifact.get("official_v22_baseline")
    baseline = (
        baseline_record.get("episodes")
        if isinstance(baseline_record, Mapping)
        else None
    )
    if not isinstance(candidate, list) or not isinstance(baseline, list):
        raise ValueError("gain024 diagnostic requires candidate and official baseline episodes")
    all_six = [*candidate, *baseline]
    if len(candidate) != 3 or len(baseline) != 3 or len(all_six) != 6:
        raise ValueError("gain024 diagnostic requires exactly six runtime episodes")
    for index, episode in enumerate(all_six):
        if not isinstance(episode, Mapping):
            raise ValueError(f"gain024 episode {index} is not an object")
        composition = episode.get("reverse_composition_contract")
        control = episode.get("h4_control_contract")
        checks = control.get("checks") if isinstance(control, Mapping) else None
        if (
            not isinstance(composition, Mapping)
            or composition.get("residual_scale") != GAIN024_RESIDUAL_SCALE
            or not isinstance(checks, Mapping)
            or checks.get("reverse_composition_contract_exact") is not True
            or any(
                value is not True
                for name, value in checks.items()
                if isinstance(name, str) and name.startswith("reverse_")
            )
        ):
            raise ValueError(
                f"gain024 reverse composition/rederivation failed for episode {index}"
            )


def _augment_and_revalidate(
    *,
    artifact: dict[str, Any],
    bundle: Any,
    central_hashes: Mapping[str, str],
    base_evaluation_hashes: Mapping[str, str],
    adapter_hashes_pre: Mapping[str, str],
    adapter_hashes_post: Mapping[str, str],
    frozen_hashes_pre: Mapping[str, str],
    frozen_hashes_post: Mapping[str, str],
    candidate_hashes_pre: Mapping[str, str],
    candidate_hashes_post: Mapping[str, str],
    validator: FunctionType,
) -> tuple[dict[str, Any], dict[str, str]]:
    if adapter_hashes_pre != adapter_hashes_post:
        raise RuntimeError("adapter/authorization sources changed during evaluation")
    if frozen_hashes_pre != frozen_hashes_post:
        raise RuntimeError("frozen source bindings changed during evaluation")
    if candidate_hashes_pre != candidate_hashes_post:
        raise RuntimeError("candidate or causal input files changed during evaluation")
    augmented = _augment_evaluation_hashes(
        base_evaluation_hashes, adapter_hashes_post
    )
    provenance = artifact.get("runtime_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("strict artifact runtime provenance is missing")
    provenance["evaluation_source_hashes_pre"] = dict(augmented)
    provenance["evaluation_source_hashes_post"] = dict(augmented)
    provenance["evaluation_source_hashes_current"] = dict(augmented)
    provenance["pre_post_source_hashes_unchanged"] = True
    provenance["reverse_v4_gain024_evaluator_adapter"] = {
        "schema_version": 1,
        "contract_id": ADAPTER_CONTRACT_ID,
        "training_contract_id": TRAINING_CONTRACT_ID,
        "scope": "EXPECTATION_AND_HOST_REDERIVATION_ONLY_NO_SIMULATION_MUTATION",
        "method": "FUNCTIONTYPE_CLONED_GLOBALS_WITHOUT_MODULE_MUTATION",
        "base_composition_maximum_residual_scale": LEGACY_RESIDUAL_SCALE,
        "effective_runtime_residual_scale": GAIN024_RESIDUAL_SCALE,
        "host_rederivation_residual_scale": GAIN024_RESIDUAL_SCALE,
        "diagnostic_artifact_kind": DIAGNOSTIC_ARTIFACT_KIND,
        "adapter_source": {
            "path": ADAPTER_SOURCE_KEY,
            "sha256_pre": adapter_hashes_pre[ADAPTER_SOURCE_KEY],
            "sha256_post": adapter_hashes_post[ADAPTER_SOURCE_KEY],
            "unchanged": True,
        },
        "adapter_authorization": {
            "path": ADAPTER_AUTHORIZATION_SOURCE_KEY,
            "sha256_pre": adapter_hashes_pre[ADAPTER_AUTHORIZATION_SOURCE_KEY],
            "sha256_post": adapter_hashes_post[ADAPTER_AUTHORIZATION_SOURCE_KEY],
            "unchanged": True,
        },
        "frozen_source_hashes_pre": dict(frozen_hashes_pre),
        "frozen_source_hashes_post": dict(frozen_hashes_post),
        "candidate_and_causal_hashes_pre": dict(candidate_hashes_pre),
        "candidate_and_causal_hashes_post": dict(candidate_hashes_post),
        "original_module_globals_and_function_references_unchanged": True,
        "promotion_evidence_allowed": False,
        "candidate_adoption_allowed": False,
        "release_allowed": False,
        "hardware_deployment": "PROHIBITED",
    }
    artifact["artifact_kind"] = DIAGNOSTIC_ARTIFACT_KIND
    artifact["promotion_allowed"] = False
    artifact["adoption_allowed"] = False
    artifact["release_allowed"] = False
    artifact["hardware_deployment"] = "PROHIBITED"
    _validate_six_episode_gain024(artifact)
    audit = validator(
        artifact,
        bundle=bundle,
        current_central_hashes=central_hashes,
        current_evaluation_hashes=augmented,
        require_all_three_pass=False,
    )
    summary = artifact.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("strict artifact summary is missing")
    summary["recomputed_validation_passed"] = bool(
        audit.get("passing_seed_count") == summary.get("passing_seed_count")
    )
    validator(
        artifact,
        bundle=bundle,
        current_central_hashes=central_hashes,
        current_evaluation_hashes=augmented,
        require_all_three_pass=False,
    )
    return artifact, augmented


def run_adapter(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, str], Any]:
    _validate_exact_cli(args)
    load_and_validate_adapter_authorization(args.adapter_authorization)
    adapter_hashes_pre = _adapter_source_hashes()
    frozen_hashes_pre = _verify_file_bindings(PINNED_FROZEN_SOURCES)
    candidate_hashes_pre = _validate_exact_candidate_files()
    base, post = _load_frozen_modules()
    frozen_hashes_after_import = _verify_file_bindings(PINNED_FROZEN_SOURCES)
    if frozen_hashes_after_import != frozen_hashes_pre:
        raise RuntimeError("frozen sources changed while importing evaluator")
    graph = build_gain024_call_graph(base, post)
    original_contract = _module_contract_snapshot(base, post)
    try:
        artifact, bundle, central_hashes, evaluation_hashes = graph[
            "run_evaluation"
        ](args)
    finally:
        _assert_module_contract_unchanged(base, post, original_contract)
    adapter_hashes_post = _adapter_source_hashes()
    frozen_hashes_post = _verify_file_bindings(PINNED_FROZEN_SOURCES)
    candidate_hashes_post = _validate_exact_candidate_files()
    artifact, augmented = _augment_and_revalidate(
        artifact=artifact,
        bundle=bundle,
        central_hashes=central_hashes,
        base_evaluation_hashes=evaluation_hashes,
        adapter_hashes_pre=adapter_hashes_pre,
        adapter_hashes_post=adapter_hashes_post,
        frozen_hashes_pre=frozen_hashes_pre,
        frozen_hashes_post=frozen_hashes_post,
        candidate_hashes_pre=candidate_hashes_pre,
        candidate_hashes_post=candidate_hashes_post,
        validator=graph["artifact_validator"],
    )
    if _adapter_source_hashes() != adapter_hashes_post:
        raise RuntimeError("adapter/authorization sources changed before output write")
    _assert_module_contract_unchanged(base, post, original_contract)
    return artifact, augmented, base.write_new_json


def main() -> None:
    args = _resolve_process_start_paths(build_parser().parse_args())
    artifact, evaluation_hashes, write_new_json = run_adapter(args)
    artifact_sha = write_new_json(args.output, artifact)
    result = {
        "diagnostic_strict_artifact": {
            "path": str(args.output),
            "sha256": artifact_sha,
            "artifact_kind": DIAGNOSTIC_ARTIFACT_KIND,
        },
        "adapter_contract_id": ADAPTER_CONTRACT_ID,
        "evaluation_source_hash_count": len(evaluation_hashes),
        "passing_seed_count": artifact["summary"]["passing_seed_count"],
        "all_three_strict_pass": artifact["summary"]["all_three_strict_pass"],
        "promotion_allowed": False,
        "adoption_allowed": False,
        "hardware_deployment": "PROHIBITED",
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    if not artifact["summary"]["all_three_strict_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
