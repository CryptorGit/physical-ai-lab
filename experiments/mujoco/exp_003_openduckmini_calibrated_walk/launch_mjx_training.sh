#!/usr/bin/env bash
set -euo pipefail

training_root=/home/user/openduck_training_20260729
run_root=/home/user/openduck_training_runs/calibrated_hybrid_yaw_cost_v22_300m
restore_path=/home/user/openduck_training_runs/calibrated_hybrid_yaw_v21_300m/2026_07_29_153359_20971520

mkdir -p "$run_root"
cd "$training_root"
exec env \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  .venv/bin/python -u playground/open_duck_mini_v2/runner.py \
  --task flat_terrain_backlash_calibrated \
  --num_envs 4096 \
  --learning_rate 0.0001 \
  --num_evals 60 \
  --num_timesteps 300000000 \
  --restore_checkpoint_path "$restore_path" \
  --output_dir "$run_root" \
  >>"$run_root/train.log" 2>&1
