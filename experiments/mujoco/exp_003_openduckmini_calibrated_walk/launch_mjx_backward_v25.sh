#!/usr/bin/env bash
set -euo pipefail

training_root=/home/user/openduck_training_backward_v23_20260729
run_root=/home/user/openduck_training_runs/calibrated_learned_backward_v25_75m
restore_path=/home/user/openduck_training_runs/calibrated_learned_backward_v24_100m/2026_07_29_204749_2621440

mkdir -p "$run_root"
cd "$training_root"
exec env \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  PYTHONPATH="$training_root${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -u playground/open_duck_mini_v2/runner.py \
  --task flat_terrain_backlash_calibrated \
  --num_envs 4096 \
  --learning_rate 0.00003 \
  --num_evals 30 \
  --num_timesteps 75000000 \
  --restore_checkpoint_path "$restore_path" \
  --output_dir "$run_root" \
  >>"$run_root/train.log" 2>&1
