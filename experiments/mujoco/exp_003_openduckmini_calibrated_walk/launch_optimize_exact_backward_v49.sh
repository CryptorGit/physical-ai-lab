#!/usr/bin/env bash
set -euo pipefail

workspace=/mnt/c/Users/user/workspace/physical-ai-lab
python_bin=/home/user/workspace/Open_Duck_Playground/.venv/bin/python
experiment="$workspace/experiments/mujoco/exp_003_openduckmini_calibrated_walk"
output="$experiment/artifacts/optimized_exact_stand_backward_v49.json"
log="$experiment/artifacts/optimized_exact_stand_backward_v49.log"

cd "$workspace"
exec "$python_bin" -u \
  "$experiment/optimize_reference_scales.py" \
  --seconds 8 \
  --maxiter 120 \
  --popsize 10 \
  --seed 49 \
  --target-vx -0.10 \
  --target-yaw 0.0 \
  --velocity-weight 1200 \
  --lateral-weight 800 \
  --yaw-weight 800 \
  --roll-pitch-weight 100 \
  --upright-weight 300 \
  --fall-penalty 5000 \
  --max-scale 5 \
  --max-bias 0.35 \
  --max-phase-rate 3 \
  --output "$output" \
  >>"$log" 2>&1
