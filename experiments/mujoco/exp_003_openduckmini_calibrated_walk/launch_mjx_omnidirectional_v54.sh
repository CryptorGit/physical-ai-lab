#!/usr/bin/env bash
set -euo pipefail

workspace_root=/mnt/c/Users/user/workspace/physical-ai-lab
source_root="$workspace_root/.openduck_playground_source_review"
training_venv=/home/user/openduck_training_backward_v23_20260729/.venv
run_root=/home/user/openduck_training_runs/omnidirectional_robust_v54_60m
restore_path=/home/user/openduck_training_runs/calibrated_hybrid_yaw_cost_v22_300m/2026_07_29_154427_10485760

mkdir -p "$run_root"
cd "$source_root"
exec env \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$training_venv/bin/python" -u playground/open_duck_mini_v2/runner.py \
  --task flat_terrain_backlash_calibrated \
  --num_envs 4096 \
  --learning_rate 0.00001 \
  --num_evals 24 \
  --num_timesteps 60000000 \
  --restore_checkpoint_path "$restore_path" \
  --output_dir "$run_root" \
  >>"$run_root/train.log" 2>&1
