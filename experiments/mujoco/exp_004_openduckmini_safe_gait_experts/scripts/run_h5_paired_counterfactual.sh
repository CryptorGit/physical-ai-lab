#!/usr/bin/env bash
# Run one pre-registered H5 V2/V3 paired counterfactual arm.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {legacy_h4_compensated|direct_normalized_v3} OUTPUT_JSON" >&2
  exit 2
fi

mapper="$1"
output="$2"
case "$mapper" in
  legacy_h4_compensated|direct_normalized_v3) ;;
  *)
    echo "unsupported paired counterfactual mapper: $mapper" >&2
    exit 2
    ;;
esac

exp_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/user/openduck_training_20260729/.venv/bin/python"
# Strict H5 is fail-closed and never calls this ONNX actor, but the frozen
# evaluator validates all eight legacy role files while constructing its base
# bank.  Pin the existing base-v22 package asset rather than inventing a copy
# under generated assets.
policy="$exp_root/artifacts/single_policy_deployment_v1/models/base_v22.onnx"
params="$exp_root/artifacts/h5_training_runs_diagnostic_20260811/unified/unified/h5_unified_1m_rate2_profile_bc_from250k_v1/final_params.pkl"
planar_manifest="$exp_root/artifacts/h5_diagnostic_wrappers/h5_unified_1m_rate2_profile_bc_from250k_v1/planar/manifest.json"
reverse_manifest="$exp_root/artifacts/h5_diagnostic_wrappers/h5_unified_1m_rate2_profile_bc_from250k_v1/reverse/manifest.json"

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
  --unified-command-mapper "$mapper" \
  --h5-planar-params "$params" \
  --h5-planar-params-sha256 887bbbd5dc6f54fe27b9b6e9437b67e719d92e765a60beee515d050553f7c922 \
  --h5-planar-manifest "$planar_manifest" \
  --h5-planar-manifest-sha256 5230530dbdbe851e2f1ec3fe5b8d75a1a1bac9a606804f281acf006680ebc12b \
  --h5-reverse-params "$params" \
  --h5-reverse-params-sha256 887bbbd5dc6f54fe27b9b6e9437b67e719d92e765a60beee515d050553f7c922 \
  --h5-reverse-manifest "$reverse_manifest" \
  --h5-reverse-manifest-sha256 364d84e7e7df7cd8b7a6a2886aacca189fb7ffbdde992896d6c7dc9c42ecb703
