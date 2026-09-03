$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$env:PYTHONPATH = @(
    (Join-Path $repo "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    (Join-Path $repo "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    (Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src")
) -join ";"
$out = Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation"

foreach ($label in @("initial", "1", "5", "10", "20", "30", "40", "50", "60")) {
    $checkpoint = Join-Path $out "checkpoints/model_$label.pt"
    & $python (Join-Path $PSScriptRoot "evaluate_w1a4.py") --mode capability --checkpoint $checkpoint --tag "capability_$label" --headless
    if ($LASTEXITCODE) { throw "W1A4 capability evaluation failed at $label" }
}
