#!/usr/bin/env bash
set -euo pipefail

training_root=/home/user/openduck_training_backward_v23_20260729
run_root=/home/user/openduck_training_runs/omnidirectional_finish_v59_40m
restore_path=/home/user/openduck_training_runs/omnidirectional_stable_v58_60m/2026_07_30_025744_57671680

mkdir -p "$run_root"
cd "$training_root"
exec env \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  PYTHONPATH="$training_root${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -u playground/open_duck_mini_v2/runner.py \
  --task flat_terrain_backlash_calibrated \
  --num_envs 4096 \
  --learning_rate 0.000005 \
  --num_evals 16 \
  --num_timesteps 40000000 \
  --restore_checkpoint_path "$restore_path" \
  --output_dir "$run_root" \
  >>"$run_root/train.log" 2>&1
