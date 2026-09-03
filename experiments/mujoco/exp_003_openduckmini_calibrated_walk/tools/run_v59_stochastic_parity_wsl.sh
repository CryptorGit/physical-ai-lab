#!/usr/bin/env bash
set -euo pipefail
cd /home/user/openduck_training_backward_v23_20260729
export PYTHONPATH=/home/user/openduck_training_backward_v23_20260729${PYTHONPATH:+:$PYTHONPATH}
exec .venv/bin/python \
  /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_003_openduckmini_calibrated_walk/tools/export_v59_stochastic_trace.py \
  --checkpoint /home/user/openduck_training_runs/omnidirectional_finish_v59_40m/2026_07_30_031556_33423360 \
  --output /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_003_openduckmini_calibrated_walk/artifacts/v59_stochastic_evaluation_equivalence \
  --master-seed 0 \
  --steps 100
