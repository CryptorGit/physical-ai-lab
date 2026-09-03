#!/usr/bin/env bash
set -euo pipefail

training_root=/home/user/openduck_training_backward_v23_20260729
run_root=/home/user/openduck_training_runs/omnidirectional_turn_teacher_v57_60m
restore_path=/home/user/openduck_training_runs/coupled_head_original_stand_backward_v45_50m/2026_07_29_235335_47349760

mkdir -p "$run_root"
cd "$training_root"
exec env \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  PYTHONPATH="$training_root${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -u playground/open_duck_mini_v2/runner.py \
  --task flat_terrain_backlash_calibrated \
  --num_envs 4096 \
  --learning_rate 0.00001 \
  --num_evals 24 \
  --num_timesteps 60000000 \
  --restore_checkpoint_path "$restore_path" \
  --output_dir "$run_root" \
  >>"$run_root/train.log" 2>&1
