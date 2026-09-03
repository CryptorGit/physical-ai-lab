param(
    [switch]$OfflineOnly
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$Python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
Set-Location $Repo
& $Python ".\experiments\isaaclab\exp_009_unitree_g1_unified_walk_run_student\scripts\run_stage1_offline_diagnostics.py"
if (-not $OfflineOnly) {
    $Checkpoints = @(
        "single_walk_steady.pt",
        "single_run_steady.pt",
        "single_walk_to_run.pt",
        "capacity_small.pt",
        "capacity_medium.pt",
        "capacity_large.pt"
    )
    foreach ($Checkpoint in $Checkpoints) {
        & $Python ".\experiments\isaaclab\exp_009_unitree_g1_unified_walk_run_student\scripts\evaluate_stage1_diagnostics.py" `
          --diagnostic-checkpoint ".\results\exp_009_unitree_g1_unified_walk_run_student\stage1_single_head_interference_diagnosis\checkpoints\$Checkpoint" `
          --headless
    }
}
