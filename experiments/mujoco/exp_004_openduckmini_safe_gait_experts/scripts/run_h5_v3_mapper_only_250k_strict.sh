#!/usr/bin/env bash
# Strict simulation-only evaluation for the pre-registered V3 250k candidate.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_JSON" >&2
  exit 2
fi

exp_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/user/openduck_training_20260729/.venv/bin/python"
output="$1"
run="$exp_root/artifacts/h5_training_runs_diagnostic_20260811/v3_mapper_only_clean_v22/unified/h5_unified_250k_v3_direct_cleanv22_notarget_v1"
wrapper_root="$exp_root/artifacts/h5_diagnostic_wrappers/h5_unified_250k_v3_direct_cleanv22_notarget_v1"
params="$run/final_params.pkl"
planar_manifest="$wrapper_root/planar/manifest.json"
reverse_manifest="$wrapper_root/reverse/manifest.json"
policy="$exp_root/artifacts/single_policy_deployment_v1/models/base_v22.onnx"

if [[ -e "$output" ]]; then
  echo "refusing to overwrite strict V3 evidence: $output" >&2
  exit 2
fi
for required in "$params" "$planar_manifest" "$reverse_manifest" "$policy"; do
  if [[ ! -f "$required" ]]; then
    echo "required V3 evaluation input is missing: $required" >&2
    exit 2
  fi
done

params_sha="$(sha256sum "$params" | awk '{print $1}')"
planar_manifest_sha="$(sha256sum "$planar_manifest" | awk '{print $1}')"
reverse_manifest_sha="$(sha256sum "$reverse_manifest" | awk '{print $1}')"

exec "$python_bin" "$exp_root/scripts/evaluate_h5_routed_transitions.py" \
  --policy "stand=$policy" \
  --policy "forward=$policy" \
  --policy "reverse=$policy" \
  --policy "lateral_left=$policy" \
  --policy "lateral_right=$policy" \
  --policy "yaw_left=$policy" \
  --policy "yaw_right=$policy" \
  --policy "compound=$policy" \
  --generated-root "$exp_root/artifacts/generated_playground" \
  --output "$output" \
  --seed 20260833 \
  --episodes 1 \
  --seconds 6 \
  --transition-seconds 6 \
  --transition-stand-seconds 2 \
  --warmup-seconds 1.5 \
  --initial-joint-noise-scale 1 \
  --initial-base-speed 0.1 \
  --unified-single-weight \
  --unified-command-mapper direct_normalized_v3 \
  --h5-planar-params "$params" \
  --h5-planar-params-sha256 "$params_sha" \
  --h5-planar-manifest "$planar_manifest" \
  --h5-planar-manifest-sha256 "$planar_manifest_sha" \
  --h5-reverse-params "$params" \
  --h5-reverse-params-sha256 "$params_sha" \
  --h5-reverse-manifest "$reverse_manifest" \
  --h5-reverse-manifest-sha256 "$reverse_manifest_sha" \
  --require-pass
