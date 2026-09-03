#!/usr/bin/env bash
# Forward-only simulation evaluation.  It does not open hardware devices.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <seconds> <episodes> <new-output-json-path>" >&2
  exit 2
fi

seconds="$1"
episodes="$2"
output_path="$3"

if [[ -e "${output_path}" ]]; then
  echo "refusing to overwrite immutable evaluation evidence: ${output_path}" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd -- "${script_dir}/.." && pwd)"
workspace_root="$(cd -- "${experiment_root}/../../.." && pwd)"
python_bin="${OPENDUCK_GPU_PYTHON:-/home/user/openduck_training_20260729/.venv/bin/python}"
package_root="${experiment_root}/artifacts/single_policy_deployment_v1"
# The packaged V22 policy has the same SHA-256 as the reviewed runtime source.
# Use the reviewed source scene and reference file for evaluation because the
# exp_003 evaluator resolves its companion motion data relative to that scene.
# This avoids silently copying or regenerating calibration/motion assets.
review_root="${workspace_root}/.openduck_playground_source_review/playground/open_duck_mini_v2"

if [[ ! -x "${python_bin}" ]]; then
  echo "missing executable OPENDUCK_GPU_PYTHON: ${python_bin}" >&2
  exit 2
fi

exec "${python_bin}" \
  "${workspace_root}/experiments/mujoco/exp_003_openduckmini_calibrated_walk/evaluate_official_policy.py" \
  --scene "${review_root}/xmls/scene_flat_terrain_backlash_calibrated.xml" \
  --policy "${package_root}/models/base_v22.onnx" \
  --reference-data "${review_root}/data/polynomial_coefficients.pkl" \
  --seconds "${seconds}" \
  --episodes "${episodes}" \
  --initial-joint-noise 0.03 \
  --initial-base-speed 0.10 \
  --command 0.08 0.0 0.0 \
  --lock-head-targets \
  --mask-head-action-history \
  --output "${output_path}"
