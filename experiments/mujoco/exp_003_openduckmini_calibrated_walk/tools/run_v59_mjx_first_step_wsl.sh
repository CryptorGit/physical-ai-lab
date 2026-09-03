#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_003_openduckmini_calibrated_walk
SOURCE=/home/user/openduck_training_backward_v23_20260729
OUT="$ROOT/artifacts/v59_mjx_first_step_divergence"
TRACE="$ROOT/artifacts/v59_stochastic_evaluation_equivalence/stochastic_traces"

cd "$SOURCE"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"$SOURCE/.venv/bin/python" "$ROOT/tools/export_v59_mjx_diagnostic_state.py" \
  --trace-root "$TRACE" --output "$OUT" --master-seed 0

"$SOURCE/.venv/bin/python" "$ROOT/tools/run_v59_mjx_one_step_diagnostic.py" \
  --input-root "$OUT/inputs" --output-root "$OUT/outputs/gpu_same_process" \
  --label gpu_same_process --repeats 20

"$SOURCE/.venv/bin/python" "$ROOT/tools/run_v59_mjx_one_step_diagnostic.py" \
  --input-root "$OUT/inputs" --output-root "$OUT/outputs/gpu_fresh_process_a" \
  --label gpu_fresh_process_a --repeats 1

"$SOURCE/.venv/bin/python" "$ROOT/tools/run_v59_mjx_one_step_diagnostic.py" \
  --input-root "$OUT/inputs" --output-root "$OUT/outputs/gpu_fresh_process_b" \
  --label gpu_fresh_process_b --repeats 1

JAX_PLATFORMS=cpu "$SOURCE/.venv/bin/python" \
  "$ROOT/tools/run_v59_mjx_one_step_diagnostic.py" \
  --input-root "$OUT/inputs" --output-root "$OUT/outputs/cpu_process" \
  --label cpu_process --repeats 1
