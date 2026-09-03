param([switch]$AuditOnly)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$output = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage1_single_policy_baseline"
Push-Location $repo
try {
  & $isaac -p (Join-Path $PSScriptRoot "audit_go2_baseline.py") --output $output --headless
  if ($LASTEXITCODE -ne 0) { throw "Stage 0 audit failed closed." }
  if ($AuditOnly) { return }
  $selected = Get-Content (Join-Path $output "stage0_selected_baseline.json") -Raw | ConvertFrom-Json
  if ($selected.status -ne "SELECTED") { throw "NO_USABLE_GO2_BASELINE" }
  $checkpoint = $selected.selected.checkpoint_path
  & $isaac -p (Join-Path $PSScriptRoot "evaluate_steady_state.py") --checkpoint $checkpoint --output $output --headless
  if ($LASTEXITCODE -ne 0) { throw "Steady-state evaluation failed closed." }
  & $isaac -p (Join-Path $PSScriptRoot "evaluate_bidirectional_transitions.py") --checkpoint $checkpoint --output $output --headless
  if ($LASTEXITCODE -ne 0) { throw "Transition evaluation failed closed." }
  & "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" (Join-Path $PSScriptRoot "finalize_stage1.py")
} finally { Pop-Location }

