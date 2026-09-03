[CmdletBinding(PositionalBinding=$false)]
param()
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$python = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Push-Location $root
try {
  & $isaac -p (Join-Path $PSScriptRoot "build_observability_dataset.py") --headless
  if ($LASTEXITCODE -ne 0) { throw "dataset replay failed" }
  & $python (Join-Path $PSScriptRoot "finalize_dataset_metadata.py")
  if ($LASTEXITCODE -ne 0) { throw "dataset metadata finalization failed" }
  & $python (Join-Path $PSScriptRoot "train_observability_probes.py")
  if ($LASTEXITCODE -ne 0) { throw "probe training failed" }
  & $python (Join-Path $PSScriptRoot "evaluate_observability_probes.py")
  if ($LASTEXITCODE -ne 0) { throw "probe evaluation failed" }
  foreach ($candidate in @("baseline", "walk_expert", "run_expert", "bounded_joint_group", "target_walk_alignment")) {
    & $isaac -p (Join-Path $PSScriptRoot "run_counterfactual_action_audit.py") --counterfactual-candidate $candidate --headless
    if ($LASTEXITCODE -ne 0) { throw "counterfactual audit failed: $candidate" }
  }
  & $python (Join-Path $PSScriptRoot "aggregate_counterfactual_results.py")
  if ($LASTEXITCODE -ne 0) { throw "counterfactual aggregation failed" }
  & $python (Join-Path $PSScriptRoot "finalize_stage0_outputs.py")
  if ($LASTEXITCODE -ne 0) { throw "Stage 0 finalization failed" }
}
finally {
  Pop-Location
}
