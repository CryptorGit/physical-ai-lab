#!/usr/bin/env bash
# Pre-registered V3 mapper-only simulation pilot.  No hardware path exists here.
set -euo pipefail

exp_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/user/openduck_training_20260729/.venv/bin/python"
parent_checkpoint="/home/user/openduck_training_runs/calibrated_hybrid_yaw_cost_v22_300m/2026_07_29_154427_10485760"
output_root="$exp_root/artifacts/h5_training_runs_diagnostic_20260811/v3_mapper_only_clean_v22"
run_name="h5_unified_250k_v3_direct_cleanv22_notarget_v1"

if [[ -e "$output_root/unified/$run_name" ]]; then
  echo "refusing to overwrite pre-registered V3 pilot output" >&2
  exit 2
fi

exec "$python_bin" "$exp_root/scripts/train_h4_aligned_expert.py" \
  --expert unified \
  --diagnostic-reward-exploration \
  --unified-development-run \
  --authorize-simulation-training \
  --num-timesteps 250000 \
  --num-envs 1250 \
  --seed 20260823 \
  --learning-rate 0.00005 \
  --entropy-cost 0.001 \
  --clipping-epsilon 0.10 \
  --discounting 0.97 \
  --max-grad-norm 0.5 \
  --observation-mode h4_116_transplant \
  --allow-verified-v22-transplant \
  --parent-checkpoint "$parent_checkpoint" \
  --output-root "$output_root" \
  --run-name "$run_name" \
  --platform gpu \
  --reset-noise-multiplier 1.0 \
  --backward-residual-scale 0.12 \
  --h5-unified-command-mapper direct_normalized_v3
