param(
  [switch]$AuditOnly,
  [string]$Device = "cuda:0"
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$output = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage11_tangential_slip_reduction"
Push-Location $repo
try {
  & $isaac -p (Join-Path $PSScriptRoot "prepare_stage11.py")
  if ($LASTEXITCODE -ne 0) { throw "Stage 11 offline contract failed." }
  & $isaac -p (Join-Path $PSScriptRoot "preflight_stage11.py") `
    --num-envs 2048 --batches 10 --device $Device --headless
  $gate = Get-Content (Join-Path $output "preflight_gate.json") -Raw | ConvertFrom-Json
  & $isaac -p (Join-Path $PSScriptRoot "finalize_stage11.py")
  if ($gate.status -ne "PASS") {
    Write-Host "Stage 11 stopped fail-closed: $($gate.status)"
    return
  }
  if ($AuditOnly) { return }
  & $isaac -p (Join-Path $PSScriptRoot "train_stage11.py")
} finally {
  Pop-Location
}
