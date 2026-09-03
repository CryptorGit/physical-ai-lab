# Git Inventory Before Instrumentation

Recorded: 2026-07-30 (Asia/Tokyo)

## Main research repository

- Branch: `master`
- HEAD: `3d3f23cb66e62cbee2c1900870c82f4a5973edea`
- The complete `experiments/mujoco/exp_003_openduckmini_calibrated_walk/`
  directory was untracked.
- Unrelated modified and untracked files exist outside this experiment. They are
  excluded from both OpenDuckMini commits.

## Historical training-source repository

- Path: `/home/user/openduck_training_backward_v23_20260729`
- Branch: `main`
- HEAD: `b9be205ac64488c23504ca42e5ec790337adeec3`
- Modified source: six Python files under `playground/`.
- Untracked required configuration: three reverse gait JSON files and two
  calibrated/backlash XML files.
- External calibration artifact:
  `playground/open_duck_mini_v2/data/polynomial_coefficients_calibrated.pkl`
  (2,387,819 bytes, SHA-256
  `47f2cf4beb701e1d84860f22b3da3676f265ebbbc568fd204656a04609562be6`).
  It is deliberately not committed; it is recorded by hash in
  `external_artifact_manifest.json`.
- `.venv` and `smoke.log` are generated local files and are excluded.

## Baseline commit scope

The source-repository baseline commit contains exactly:

- `playground/common/export_onnx.py`
- `playground/common/rewards.py`
- `playground/common/runner.py`
- `playground/open_duck_mini_v2/constants.py`
- `playground/open_duck_mini_v2/joystick.py`
- `playground/open_duck_mini_v2/runner.py`
- three optimized reverse gait JSON profiles
- the calibrated robot XML
- the calibrated/backlash scene XML

The main-repository baseline commit contains the experiment's Python, shell,
PowerShell, JSON configuration, README, analysis tools, and tests, plus the
small v59/v60 provenance reports needed to establish the pre-instrumentation
state. Checkpoints, raw traces, videos, plots, caches, and generated rollout
data are excluded.

## Pre-commit checks

- No credential-like material was found in the selected main-repository files.
- The historical source diff has pre-existing CRLF/trailing-whitespace warnings.
  It is committed byte-for-byte to preserve the executed v60 source; formatting
  normalization is intentionally not mixed into this provenance freeze.
