param(
  [string]$Device = "cuda:0",
  [ValidateSet("C0","C1","C2")][string[]]$Controllers = @("C0","C1","C2"),
  [switch]$SkipPrepare
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$evaluate = Join-Path $PSScriptRoot "evaluate_stage10_heading_controller.py"
$output = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage10_phase_gated_fixed_heading"
Push-Location $repo
try {
  if (-not $SkipPrepare) {
    python (Join-Path $PSScriptRoot "prepare_stage10.py")
    if ($LASTEXITCODE -ne 0) { throw "Stage 10 protocol preflight failed." }
  }
  $chunks = @(
    "steady_0", "steady_1", "steady_2", "steady_3", "steady_4",
    "low_0", "low_1", "low_2", "anchor_0", "anchor_1", "sequence"
  )
  foreach ($controller in $Controllers) {
    foreach ($chunk in $chunks) {
      $raw = Join-Path $output "raw_${controller}_${chunk}.json"
      if (Test-Path -LiteralPath $raw) {
        Write-Output "SKIP $controller $chunk"
        continue
      }
      Write-Output "RUN $controller $chunk"
      & $isaac -p $evaluate --controller $controller --chunk $chunk `
        --num-envs 50 --device $Device --headless
      if ($LASTEXITCODE -ne 0) { throw "Stage 10 chunk failed: $controller $chunk" }
    }
  }
  if ($Controllers.Count -eq 3) {
    python (Join-Path $PSScriptRoot "finalize_stage10.py")
    if ($LASTEXITCODE -ne 0) { throw "Stage 10 finalization failed." }
  }
} finally {
  Pop-Location
}
