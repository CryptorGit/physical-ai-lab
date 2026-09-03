param(
  [switch]$SkipRollout,
  [switch]$SkipVisual
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$diagnose = Join-Path $PSScriptRoot "diagnose_stage5_endpoints.py"
$analyze = Join-Path $PSScriptRoot "analyze_stage5_endpoints.py"
$visualize = Join-Path $PSScriptRoot "visualize_stage5_slip.py"
Push-Location $repo
try {
  if (-not $SkipRollout) { & $isaac -p $diagnose --headless }
  & python $analyze
  if (-not $SkipVisual) { & $isaac -p $visualize }
} finally {
  Pop-Location
}
