#!/usr/bin/env bash
set -euo pipefail

workspace=/mnt/c/Users/user/workspace/physical-ai-lab
python_bin=/home/user/workspace/Open_Duck_Playground/.venv/bin/python
experiment="$workspace/experiments/mujoco/exp_003_openduckmini_calibrated_walk"

cd "$workspace"
exec "$python_bin" -u \
  "$experiment/optimize_reference_scales.py" \
  --seconds 20 \
  --maxiter 80 \
  --popsize 8 \
  --seed 50 \
  --target-vx -0.10 \
  --target-yaw 0.0 \
  --velocity-weight 5000 \
  --lateral-weight 5000 \
  --yaw-weight 5000 \
  --roll-pitch-weight 200 \
  --upright-weight 500 \
  --fall-penalty 10000 \
  --max-scale 5 \
  --max-bias 0.35 \
  --max-phase-rate 3 \
  --initial-joint-noise 0.003 \
  --initial-base-speed 0.01 \
  --initial-gait "$experiment/artifacts/optimized_exact_stand_backward_v49.json" \
  --output "$experiment/artifacts/optimized_exact_stand_backward_v50.json" \
  >>"$experiment/artifacts/optimized_exact_stand_backward_v50.log" 2>&1
