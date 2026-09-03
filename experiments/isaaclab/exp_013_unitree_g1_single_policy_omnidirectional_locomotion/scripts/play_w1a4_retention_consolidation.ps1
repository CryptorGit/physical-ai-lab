param(
    [string]$Checkpoint = "",
    [switch]$Headless
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$out = Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation"
if (-not $Checkpoint) {
    $Checkpoint = (Get-Content (Join-Path $out "canonical_walk_parent.json") | ConvertFrom-Json).path
}
$env:PYTHONPATH = @(
    (Join-Path $repo "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    (Join-Path $repo "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    (Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src")
) -join ";"
$arguments = @((Join-Path $PSScriptRoot "play_w1a4_retention_consolidation.py"), "--checkpoint", $Checkpoint)
if ($Headless) { $arguments += "--headless" }
& $python $arguments
