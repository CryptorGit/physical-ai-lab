#!/usr/bin/env python3
"""Freeze runtime and configuration provenance for the corrected diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import jax
import jaxlib
import mujoco
from brax.training.agents.ppo import checkpoint

from playground.open_duck_mini_v2 import joystick

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from v59_mjx_diagnostic_common import canonical_tree_sha256


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--evaluation-script", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    source = Path(args.source_root)
    env = joystick.Joystick(task="flat_terrain_backlash_calibrated")
    params = checkpoint.load(args.checkpoint)
    command_path = root / "command_manifest.json"
    seed_path = root / "seed_manifest.json"
    d_metadata = json.loads(
        (root / "condition_d_run_metadata.json").read_text(encoding="utf-8")
    )
    s_metadata = json.loads(
        (root / "condition_s_run_metadata.json").read_text(encoding="utf-8")
    )
    teacher_files = [
        source / "playground/open_duck_mini_v2/joystick.py",
        source / "playground/open_duck_mini_v2/data/optimized_backward_gait.json",
        source
        / "playground/open_duck_mini_v2/data/optimized_backward_left_turn_gait.json",
        source
        / "playground/open_duck_mini_v2/data/optimized_backward_right_turn_gait.json",
        source
        / "playground/open_duck_mini_v2/data/polynomial_coefficients_calibrated.pkl",
    ]
    teacher_config = {
        str(path): file_sha(path) for path in teacher_files
    }
    reward_config = env._config.reward_config.to_dict()
    normalizer = {"mean": params[0].mean, "std": params[0].std}
    scene = Path(env._xml_path)
    resolved = {
        "schema_version": 1,
        "diagnostic_only": True,
        "formal_acceptance_eligible": False,
        "enough_episodes": False,
        "backend": "GPU MJX",
        "checkpoint": args.checkpoint,
        "conditions": {
            "D": d_metadata,
            "S": s_metadata,
        },
        "command_manifest_sha256": file_sha(command_path),
        "seed_manifest_sha256": file_sha(seed_path),
        "scene_sha256": file_sha(scene),
        "normalizer_sha256": canonical_tree_sha256(normalizer),
        "teacher_configuration": teacher_config,
        "teacher_configuration_sha256": json_sha(teacher_config),
        "reward_configuration": reward_config,
        "reward_configuration_sha256": json_sha(reward_config),
        "evaluation_script_sha256": file_sha(Path(args.evaluation_script)),
    }
    resolved_path = root / "resolved_evaluation_manifest.json"
    resolved_path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    smi = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    environment = {
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "mujoco_version": mujoco.__version__,
        "mjx_provenance": "mujoco.mjx from mujoco package",
        "cuda_nvidia_smi_header": smi[:4],
        "gpu_model_and_driver": d_metadata["runtime"]["nvidia_smi"],
        "xla_flags": d_metadata["runtime"]["xla_flags"],
        "jax_config": {
            "float_precision": "float32",
            "x64_enabled": d_metadata["runtime"]["x64_enabled"],
            "jit_enabled": d_metadata["runtime"]["jit_enabled"],
            "matmul_precision": d_metadata["runtime"]["matmul_precision"],
            "process_count": d_metadata["runtime"]["process_count"],
            "device_count": d_metadata["runtime"]["device_count"],
            "devices": d_metadata["runtime"]["devices"],
        },
        "hashes": {
            "checkpoint_tree_sha256": "4e522903cfb3edf8dacfc2f5dc5b9510746711360748440c54097483f0ac38f1",
            "onnx_sha256": file_sha(Path(args.onnx)),
            "scene_xml_sha256": file_sha(scene),
            "command_definition_sha256": file_sha(command_path),
            "normalizer_sha256": canonical_tree_sha256(normalizer),
            "teacher_configuration_sha256": json_sha(teacher_config),
            "reward_configuration_sha256": json_sha(reward_config),
            "evaluation_script_sha256": file_sha(Path(args.evaluation_script)),
            "resolved_evaluation_manifest_sha256": file_sha(resolved_path),
        },
        "onnx_path": args.onnx,
        "onnx_role": "provenance only; GPU JAX checkpoint actor was used",
        "scene_path": str(scene),
        "push_training_provenance": {
            "status": "resolved_enabled",
            "code": "joystick.py step: adds sampled vector to floating-base xy qvel",
            "external_force": False,
            "base_velocity_rewrite_or_addition": True,
            "impulse_like": True,
            "episode_time": True,
            "reset_only": False,
            "condition_D": "disabled",
            "condition_S": "enabled with 5-10 s interval and 0.10-0.50 m/s magnitude",
        },
    }
    (root / "environment_manifest.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
