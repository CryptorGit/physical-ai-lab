"""Export a completed Brax gait expert to ONNX and verify its interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/user/openduck_training_20260729"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params_path = args.params.resolve()
    output_path = args.output.resolve()
    source_root = args.source_root.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not params_path.is_file():
        raise FileNotFoundError(params_path)
    sys.path.insert(0, str(source_root))

    from mujoco_playground.config import locomotion_params
    import onnxruntime
    from playground.common.export_onnx import export_onnx

    with params_path.open("rb") as stream:
        params = pickle.load(stream)
    ppo_config = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(params, 14, ppo_config, 101, output_path=str(output_path))

    session = onnxruntime.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    action = session.run(
        None, {model_input.name: np.zeros((1, 101), dtype=np.float32)}
    )[0]
    if model_input.name != "obs" or model_input.shape != [1, 101]:
        raise ValueError(f"unexpected ONNX input: {model_input.name} {model_input.shape}")
    if model_output.shape != [1, 14] or action.shape != (1, 14):
        raise ValueError(f"unexpected ONNX output: {model_output.shape} / {action.shape}")
    if not np.all(np.isfinite(action)) or np.max(np.abs(action)) > 1.0 + 1e-6:
        raise ValueError("ONNX smoke action is non-finite or outside tanh range")

    report = {
        "schema_version": 1,
        "hardware_deployment": "PROHIBITED",
        "params": {"path": str(params_path), "sha256": sha256(params_path)},
        "onnx": {"path": str(output_path), "sha256": sha256(output_path)},
        "interface": {
            "input_name": model_input.name,
            "input_shape": model_input.shape,
            "output_name": model_output.name,
            "output_shape": model_output.shape,
        },
        "head_contract": (
            "ONNX retains 14 outputs; runtime/training composition must mask indices "
            "5:9 to exact zero after inference"
        ),
        "zero_observation_action": action[0].astype(float).tolist(),
    }
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
