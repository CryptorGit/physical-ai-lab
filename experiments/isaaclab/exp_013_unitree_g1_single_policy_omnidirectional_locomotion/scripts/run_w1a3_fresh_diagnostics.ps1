$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$env:PYTHONPATH = @(
    (Join-Path $repo "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    (Join-Path $repo "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    (Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src")
) -join ";"
$checkpoint = Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"

& $python (Join-Path $PSScriptRoot "evaluate_w1a3.py") --mode tradeoff --checkpoint $checkpoint --tag "tradeoff_80" --headless
if ($LASTEXITCODE -ne 0) { throw "tradeoff validation failed" }
& $python (Join-Path $PSScriptRoot "diagnose_w1a3_fresh.py") --mode state --headless
if ($LASTEXITCODE -ne 0) { throw "state diagnosis failed" }
& $python (Join-Path $PSScriptRoot "diagnose_w1a3_fresh.py") --mode gradient --headless
if ($LASTEXITCODE -ne 0) { throw "gradient diagnosis failed" }
