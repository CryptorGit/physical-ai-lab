#!/usr/bin/env bash
# Fresh-process GPU diagnostic only.  This script never authorizes PPO or hardware.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <control-steps: 1|3> <new-output-json-path>" >&2
  exit 2
fi

control_steps="$1"
output_path="$2"
case "${control_steps}" in
  1|3) ;;
  *)
    echo "control steps must be exactly 1 or 3" >&2
    exit 2
    ;;
esac

if [[ -e "${output_path}" ]]; then
  echo "refusing to overwrite immutable diagnostic evidence: ${output_path}" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${OPENDUCK_GPU_PYTHON:-/home/user/openduck_training_20260729/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "missing executable OPENDUCK_GPU_PYTHON: ${python_bin}" >&2
  exit 2
fi

cd -- "${experiment_root}"
exec "${python_bin}" scripts/train_h4_aligned_expert.py \
  --platform gpu \
  --seed 20260823 \
  --expert unified \
  --diagnostic-reward-exploration \
  --h5-unified-command-mapper direct_normalized_v3 \
  --v4-authoritative-primitive-batch-parity-preflight-only \
  --v4-authoritative-primitive-batch-parity-control-steps "${control_steps}" \
  --v4-authoritative-primitive-batch-parity-preflight-output "${output_path}"
