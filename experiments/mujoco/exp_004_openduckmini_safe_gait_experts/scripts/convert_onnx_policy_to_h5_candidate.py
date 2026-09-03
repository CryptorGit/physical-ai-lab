"""Build a diagnostic H5 actor candidate from a compatible 101-wide ONNX policy.

The frozen v22 actor and the calibrated exp003 actor share the Brax MLP
topology.  H5 changes the actor input to 116 features and owns the absolute
target decoder, so a compatible policy can be transferred without changing
the decoder or the frozen evaluator: the fifteen H5-only input rows are
zero-initialized and the ONNX normalizer is extended with zero/one entries.

This tool is deliberately diagnostic.  It does not authorize adoption or
hardware deployment and emits a manifest whose qualification field is the
same non-qualification value accepted by the H5 diagnostic evaluator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


ACTOR_INPUT_WIDTH = 116
SOURCE_INPUT_WIDTH = 101
ACTION_WIDTH = 14
HIDDEN_WIDTHS = (512, 256, 128, 28)
DIAGNOSTIC_QUALIFICATION = "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        tensor.name: np.asarray(numpy_helper.to_array(tensor))
        for tensor in model.graph.initializer
    }


def _load_onnx_actor(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    model = onnx.load(str(path))
    input_shape = [dimension.dim_value for dimension in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dimension.dim_value for dimension in model.graph.output[0].type.tensor_type.shape.dim]
    if input_shape != [1, SOURCE_INPUT_WIDTH] or output_shape != [1, ACTION_WIDTH]:
        raise ValueError(
            f"compatible ONNX must be [1,101]->[1,14], got {input_shape}->{output_shape}"
        )
    tensors = _initializer_map(model)
    mean_candidates = [
        value
        for name, value in tensors.items()
        if value.shape == (SOURCE_INPUT_WIDTH,) and "/sub/" in name
    ]
    reciprocal_candidates = [
        value
        for name, value in tensors.items()
        if value.shape == (SOURCE_INPUT_WIDTH,) and "truediv_recip" in name
    ]
    if len(mean_candidates) != 1 or len(reciprocal_candidates) != 1:
        raise ValueError("could not identify the unique ONNX input normalizer")
    mean = mean_candidates[0].astype(np.float32, copy=True)
    reciprocal_std = reciprocal_candidates[0].astype(np.float32, copy=True)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(reciprocal_std)):
        raise ValueError("ONNX normalizer is non-finite")
    if np.any(reciprocal_std <= 0.0):
        raise ValueError("ONNX reciprocal standard deviation must be positive")

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for node in model.graph.node:
        if node.op_type != "Gemm" or len(node.input) < 3:
            continue
        kernel = tensors.get(node.input[1])
        bias = tensors.get(node.input[2])
        if kernel is None or bias is None:
            continue
        if not layers and kernel.shape != (SOURCE_INPUT_WIDTH, HIDDEN_WIDTHS[0]):
            continue
        expected_input = SOURCE_INPUT_WIDTH if not layers else HIDDEN_WIDTHS[len(layers) - 1]
        expected_output = HIDDEN_WIDTHS[len(layers)]
        if kernel.shape != (expected_input, expected_output) or bias.shape != (expected_output,):
            raise ValueError(
                f"ONNX actor layer {len(layers)} shape drifted: {kernel.shape}, {bias.shape}"
            )
        layers.append((kernel.astype(np.float32, copy=True), bias.astype(np.float32, copy=True)))
    if len(layers) != 4:
        raise ValueError(f"expected four ONNX actor Gemm layers, found {len(layers)}")
    return mean, reciprocal_std, layers


def _replace_actor(
    params: Any,
    mean: np.ndarray,
    reciprocal_std: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray]],
) -> Any:
    candidate = copy.deepcopy(params)
    normalizer, actor, _critic = candidate
    state_mean = np.concatenate((mean, np.zeros(ACTOR_INPUT_WIDTH - SOURCE_INPUT_WIDTH, dtype=np.float32)))
    state_std = np.concatenate((1.0 / reciprocal_std, np.ones(ACTOR_INPUT_WIDTH - SOURCE_INPUT_WIDTH, dtype=np.float32)))
    normalizer.mean["state"] = state_mean
    normalizer.std["state"] = state_std
    actor_layers = actor["params"]
    for index, (kernel, bias) in enumerate(layers):
        target = actor_layers[f"hidden_{index}"]
        if index == 0:
            padded = np.zeros((ACTOR_INPUT_WIDTH, kernel.shape[1]), dtype=np.float32)
            padded[:SOURCE_INPUT_WIDTH] = kernel
            target["kernel"] = padded
        else:
            target["kernel"] = kernel
        target["bias"] = bias
    return candidate


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--domain", choices=("planar", "reverse"), required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    onnx_path = args.onnx.resolve()
    template_path = args.template.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if not onnx_path.is_file() or not template_path.is_file():
        raise FileNotFoundError("ONNX source and H5 template must exist")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite H5 transfer evidence")
    mean, reciprocal_std, layers = _load_onnx_actor(onnx_path)
    with template_path.open("rb") as stream:
        template = pickle.load(stream)
    candidate = _replace_actor(template, mean, reciprocal_std, layers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(candidate, stream, protocol=pickle.HIGHEST_PROTOCOL)
    payload = {
        "schema_version": 1,
        "status": "COMPLETED",
        "hardware_deployment": "PROHIBITED",
        "activity": "H5_DIAGNOSTIC_ACTOR_TRANSFER",
        "expert": args.domain,
        "qualification_use": DIAGNOSTIC_QUALIFICATION,
        "candidate_kind": "H5_COMPATIBLE_ONNX_TRANSFER_NOT_QUALIFIED",
        "source_policy": {
            "path": str(onnx_path),
            "sha256": sha256_file(onnx_path),
            "input_width": SOURCE_INPUT_WIDTH,
            "output_width": ACTION_WIDTH,
            "actor_topology": [SOURCE_INPUT_WIDTH, *HIDDEN_WIDTHS],
        },
        "template_h5_candidate": {
            "path": str(template_path),
            "sha256": sha256_file(template_path),
            "actor_input_width": ACTOR_INPUT_WIDTH,
        },
        "input_adaptation": {
            "source_rows_copied": SOURCE_INPUT_WIDTH,
            "new_h5_rows": ACTOR_INPUT_WIDTH - SOURCE_INPUT_WIDTH,
            "new_actor_rows_exact_zero": True,
            "new_normalizer_mean_exact_zero": True,
            "new_normalizer_std_exact_one": True,
            "critic_preserved_from_template": True,
        },
        "target_space_contract": "OPEN_DUCK_MINI_H5_TARGET_SPACE_ROUTING_V1",
        "actor_authority": "H5_candidate_actor_only; frozen_runtime_decoder_and_guard_unchanged",
        "outputs": {
            "final_params": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
            }
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"params": str(output_path), "manifest": str(manifest_path), "params_sha256": payload["outputs"]["final_params"]["sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
