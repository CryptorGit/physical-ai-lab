param([switch]$AuditOnly, [switch]$WiringOnly)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$output = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage2_continuous_0_to_2_training"
$stage1 = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage1_single_policy_baseline\stage0_selected_baseline.json"
Push-Location $repo
try {
  & $isaac -p (Join-Path $PSScriptRoot "audit_stage2_protocol.py") --output $output
  if ($LASTEXITCODE -ne 0) { throw "Stage 2 static audit failed closed." }
  if ($AuditOnly) { return }
  $checkpoint = (Get-Content $stage1 -Raw | ConvertFrom-Json).selected.checkpoint_path
  & $isaac -p (Join-Path $PSScriptRoot "train_stage2_continuous.py") --mode wiring --checkpoint $checkpoint --output $output --headless
  if ($LASTEXITCODE -ne 0) { throw "Stage 2 wiring failed." }
  if ($WiringOnly) { return }
  & $isaac -p (Join-Path $PSScriptRoot "train_stage2_continuous.py") --mode pilot --checkpoint $checkpoint --output $output --headless
  $pilotExit = $LASTEXITCODE
  & "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" (Join-Path $PSScriptRoot "finalize_stage2_unstable.py")
  if ($pilotExit -ne 0) {
    $classification = (Get-Content (Join-Path $output "stage2_classification.json") -Raw | ConvertFrom-Json).classification
    if ($classification -ne "GO2_TRAINING_UNSTABLE") { throw "Unexpected Pilot failure." }
    Write-Host "Pilot stopped fail-closed: GO2_TRAINING_UNSTABLE"
  }
} finally { Pop-Location }
