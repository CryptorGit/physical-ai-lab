$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$env:PYTHONPATH = @(
    (Join-Path $repo "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    (Join-Path $repo "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    (Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src")
) -join ";"

& $python (Join-Path $PSScriptRoot "train_w1a4.py") --mode preflight --headless
if ($LASTEXITCODE) { throw "W1A4 retention beta preflight failed" }

& $python (Join-Path $PSScriptRoot "train_w1a4.py") --mode train --headless
if ($LASTEXITCODE) { throw "W1A4 single persistent training run failed" }
