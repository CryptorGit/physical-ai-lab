"""Export a trusted H4 actor116 to ONNX with three-way CPU parity evidence.

The ONNX file contains only the deterministic 116-to-14 Brax policy network.
It is not an adopted runtime.  Head outputs still require the exact post-policy
5:9 mask, and a reverse policy additionally requires the pinned persistent
teacher-plus-residual composition and guard order recorded in the authorization
artifact.  Standalone use, release, and hardware deployment remain prohibited.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.h4_post_training import (  # noqa: E402
    H4_ACTION_WIDTH,
    H4_ACTOR_OBSERVATION_WIDTH,
    H4_CRITIC_OBSERVATION_WIDTH,
    PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256,
    PINNED_SELECTED_REVERSE_TEACHER_SHA256,
    compare_policy_outputs,
    current_source_hashes,
    infer_h4_action_numpy,
    load_trusted_h4_params,
    mask_h4_head_action,
    reconstruct_h4_training_source_paths,
    sha256_file,
    validate_h4_training_source_closure,
    validate_trusted_h4_bundle,
    write_new_json,
)


LEGACY_TRAINER_PATH = EXP_ROOT / "scripts" / "train_expert.py"
H4_RUNNER_PATH = EXP_ROOT / "scripts" / "train_h4_aligned_expert.py"
H4_ALIGNMENT_PATH = EXP_ROOT / "safe_gait_experts" / "h4_training_alignment.py"
H4_POST_TRAINING_PATH = EXP_ROOT / "safe_gait_experts" / "h4_post_training.py"
REVERSE_COMPOSITION_VALIDATOR_PATH = (
    EXP_ROOT / "scripts" / "validate_h4_reverse_training_composition.py"
)
TRUSTED_H4_RUN_ROOT = EXP_ROOT / "artifacts" / "h4_training_runs"
DEFAULT_SOURCE_ROOT = Path("/home/user/openduck_training_20260729")
DEFAULT_GENERATED_ROOT = EXP_ROOT / "artifacts" / "generated_playground"


def _load_legacy_trainer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "exp004_h4_onnx_legacy_trainer", LEGACY_TRAINER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy trainer: {LEGACY_TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a manifest-bound H4 actor116 to simulation-only ONNX."
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
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument(
        "--allow-wiring-diagnostic",
        action="store_true",
        help="Allow export of a 40-interaction WIRING_PASS for interface QA only.",
    )
    return parser


def _build_brax_policy(stack: Any, bundle: Any, params: Any) -> Any:
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


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output.resolve()
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite immutable ONNX/report output")
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("H4 export output must use the .onnx suffix")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    bundle = validate_trusted_h4_bundle(
        params_path=args.params,
        manifest_path=args.manifest,
        expected_params_sha256=args.params_sha256,
        expected_manifest_sha256=args.manifest_sha256,
        trusted_run_root=args.trusted_run_root,
        allow_wiring_diagnostic=args.allow_wiring_diagnostic,
    )
    source_paths = {
        "exporter": Path(__file__).resolve(),
        "h4_post_training": H4_POST_TRAINING_PATH,
        "h4_alignment": H4_ALIGNMENT_PATH,
        "h4_runner": H4_RUNNER_PATH,
        "legacy_trainer": LEGACY_TRAINER_PATH,
        "params": bundle.params_path,
        "manifest": bundle.manifest_path,
        "config": bundle.config_path,
        "result": Path(bundle.manifest["outputs"]["result"]["path"]).resolve(),
        "training_curve": Path(
            bundle.manifest["outputs"]["training_curve"]["path"]
        ).resolve(),
    }
    if bundle.expert == "reverse":
        source_paths["selected_reverse_teacher"] = Path(
            bundle.config["selected_reverse_teacher"]["path"]
        ).resolve()
        source_paths["reverse_composition_authorization"] = Path(
            bundle.config["reverse_composition_authorization"]["path"]
        ).resolve()
    # Load the source stack first so Brax/Flax checkpoint classes exist, but do
    # not open the pickle until the manifest and exact training source closure
    # have both passed.
    trainer = _load_legacy_trainer()
    stack = trainer._load_training_stack(args.source_root.resolve())
    generated = trainer.generated_paths(args.generated_root.resolve())
    trainer._validate_generated_manifest(generated)

    class TeacherArgs:
        backward_gait = None
        backward_left_gait = None
        backward_right_gait = None

    legacy_teacher_gaits = trainer.resolve_teacher_gaits(TeacherArgs(), generated)
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    expected_training_sources = reconstruct_h4_training_source_paths(
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
    bundle = validate_h4_training_source_closure(
        bundle, expected_training_sources
    )
    present_paths = {Path(path).resolve() for path in source_paths.values()}
    for label, path in expected_training_sources.items():
        resolved = Path(path).resolve()
        if resolved not in present_paths:
            source_paths[f"training_{label}"] = resolved
            present_paths.add(resolved)
    source_pre = current_source_hashes(source_paths, root=EXP_ROOT)
    jax = stack["jax"]
    jp = stack["jp"]
    if jax.default_backend() != "cpu" or any(
        device.platform != "cpu" for device in jax.devices()
    ):
        raise RuntimeError("H4 ONNX export parity must run on CPU-only JAX")
    params, restore_audit = load_trusted_h4_params(bundle)
    policy = _build_brax_policy(stack, bundle, params)

    rng = np.random.default_rng(20_260_809)
    observations = np.stack(
        (
            np.zeros(H4_ACTOR_OBSERVATION_WIDTH, dtype=np.float32),
            np.ones(H4_ACTOR_OBSERVATION_WIDTH, dtype=np.float32),
            np.linspace(-1.0, 1.0, H4_ACTOR_OBSERVATION_WIDTH, dtype=np.float32),
            rng.normal(size=H4_ACTOR_OBSERVATION_WIDTH).astype(np.float32),
        )
    )
    numpy_actions = infer_h4_action_numpy(params, observations)
    brax_actions = []
    for index, observation in enumerate(observations):
        action, _ = policy(
            {
                "state": jp.asarray(observation),
                "privileged_state": jp.zeros(H4_CRITIC_OBSERVATION_WIDTH),
            },
            jax.random.PRNGKey(index),
        )
        brax_actions.append(np.asarray(action, dtype=np.float32))
    brax_actions_array = np.stack(brax_actions)
    numpy_brax_parity = compare_policy_outputs(numpy_actions, brax_actions_array)
    if not numpy_brax_parity["passed"]:
        raise RuntimeError(f"H4 NumPy/Brax pre-export parity failed: {numpy_brax_parity}")

    from mujoco_playground.config import locomotion_params
    import onnx
    import onnxruntime
    from playground.common.export_onnx import export_onnx

    ppo_config = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )
    if (
        tuple(ppo_config.network_factory.policy_hidden_layer_sizes)
        != (512, 256, 128)
        or ppo_config.network_factory.policy_obs_key != "state"
    ):
        raise RuntimeError("source ONNX exporter network topology drifted")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(
        params,
        H4_ACTION_WIDTH,
        ppo_config,
        H4_ACTOR_OBSERVATION_WIDTH,
        output_path=str(output_path),
    )
    if not output_path.is_file():
        raise RuntimeError("H4 ONNX exporter did not create its output")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    session = onnxruntime.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("H4 ONNX verification session is not CPU-only")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("H4 ONNX must expose exactly one input and one output")
    model_input = inputs[0]
    model_output = outputs[0]
    if model_input.name != "obs" or model_input.shape != [1, 116]:
        raise ValueError(
            f"H4 ONNX input must be obs[1,116], got {model_input.name} "
            f"{model_input.shape}"
        )
    if model_output.shape != [1, 14]:
        raise ValueError(f"H4 ONNX output must be [1,14], got {model_output.shape}")
    onnx_actions = np.stack(
        [
            session.run(
                [model_output.name],
                {model_input.name: observation.reshape(1, 116)},
            )[0][0]
            for observation in observations
        ]
    ).astype(np.float32)
    numpy_onnx_parity = compare_policy_outputs(numpy_actions, onnx_actions)
    brax_onnx_parity = compare_policy_outputs(brax_actions_array, onnx_actions)
    if not numpy_onnx_parity["passed"] or not brax_onnx_parity["passed"]:
        raise RuntimeError(
            "H4 ONNX parity failed: "
            f"numpy={numpy_onnx_parity}, brax={brax_onnx_parity}"
        )
    masked_actions = mask_h4_head_action(onnx_actions)
    if not np.array_equal(masked_actions[:, 5:9], np.zeros((4, 4), np.float32)):
        raise RuntimeError("H4 ONNX post-inference head mask is not exact zero")

    source_post = current_source_hashes(source_paths, root=EXP_ROOT)
    if source_pre != source_post:
        raise RuntimeError("H4 ONNX sources changed during export")
    try:
        import tensorflow as tf

        tensorflow_version = tf.__version__
        tensorflow_visible_gpus = [
            device.name for device in tf.config.get_visible_devices("GPU")
        ]
    except ImportError:
        tensorflow_version = None
        tensorflow_visible_gpus = []
    if tensorflow_visible_gpus:
        raise RuntimeError("H4 ONNX TensorFlow conversion exposed a GPU")

    reverse_composition = None
    if bundle.expert == "reverse":
        reverse_composition = {
            "mode": "PINNED_PERSISTENT_TEACHER_PLUS_TRAINABLE_RESIDUAL",
            "selected_teacher_sha256": PINNED_SELECTED_REVERSE_TEACHER_SHA256,
            "composition_authorization_sha256": (
                PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
            ),
            "teacher_phase_steps": 54,
            "teacher_cadence_hz": 1.5,
            "teacher_phase_advance_bins_per_control": 1.62,
            "teacher_entry_phase_preincrement_bins": 14.0,
            "maximum_residual_scale": 0.12,
            "standalone_onnx_direct_runtime": "PROHIBITED",
            "required_order": [
                "pinned teacher plus ONNX residual",
                "inner target-margin clamp",
                "0.04 rad per-control reverse precomposer",
                "single final runtime-equivalent target guard",
                "physics",
            ],
        }
    report = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_h4_actor116_onnx_export",
        "hardware_deployment": "PROHIBITED",
        "adoption_allowed": False,
        "release_allowed": False,
        "standalone_direct_runtime_allowed": False,
        "diagnostic_only": bundle.status == "WIRING_PASS",
        "promotion_eligible": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": bundle.candidate_record(),
        "onnx": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "opset": 11,
            "checker_passed": True,
        },
        "interface": {
            "input_name": model_input.name,
            "input_shape": model_input.shape,
            "input_dtype": model_input.type,
            "output_name": model_output.name,
            "output_shape": model_output.shape,
            "output_dtype": model_output.type,
            "deterministic_distribution_mode": "tanh(first_14_of_28_logits)",
        },
        "parity": {
            "probe_count": int(observations.shape[0]),
            "numpy_brax": numpy_brax_parity,
            "numpy_onnx": numpy_onnx_parity,
            "brax_onnx": brax_onnx_parity,
            "all_finite_and_tanh_bounded": bool(
                np.all(np.isfinite(onnx_actions))
                and np.max(np.abs(onnx_actions)) <= 1.0 + 1.0e-6
            ),
        },
        "head_contract": {
            "raw_actor_outputs_retained": 14,
            "post_inference_mask_slice": [5, 9],
            "post_mask_probe_heads_exact_zero": True,
        },
        "reverse_runtime_composition": reverse_composition,
        "runtime_provenance": {
            "execution_provider": "CPUExecutionProvider",
            "onnxruntime_version": onnxruntime.__version__,
            "onnxruntime_available_providers": onnxruntime.get_available_providers(),
            "onnxruntime_session_providers": session.get_providers(),
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [
                {"description": str(device), "platform": device.platform}
                for device in jax.devices()
            ],
            "training_jax_devices": bundle.manifest.get("jax_devices"),
            "training_provenance": dict(bundle.training_provenance),
            "tensorflow_version": tensorflow_version,
            "tensorflow_visible_gpus": tensorflow_visible_gpus,
            "params_restore_audit": restore_audit,
            "source_hashes_pre": source_pre,
            "source_hashes_post": source_post,
            "pre_post_sources_unchanged": True,
        },
    }
    write_new_json(report_path, report)
    return report


def main() -> None:
    report = run_export(build_parser().parse_args())
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
