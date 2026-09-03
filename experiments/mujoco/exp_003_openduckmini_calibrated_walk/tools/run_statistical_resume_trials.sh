#!/usr/bin/env bash
set -euo pipefail

source /home/user/openduck_training_20260729/.venv/bin/activate
cd /mnt/c/Users/user/workspace/physical-ai-lab

python \
  experiments/mujoco/exp_003_openduckmini_calibrated_walk/training/statistical_resume_test.py \
  orchestrate \
  --initial \
  /home/user/openduck_training_backward_v23_20260729/experiments/mujoco/exp_003_openduckmini_calibrated_walk/artifacts/instrumented_training_harness_v1/runs/final_stability_a \
  --raw-root /home/user/openduck_statistical_resume_20260730/trials_v1 \
  --artifact-root \
  /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_003_openduckmini_calibrated_walk/artifacts/statistical_resume_and_null_continuation \
  --trials-per-mode 20 \
  --seed 20260730
