[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)] [string]$Checkpoint,
    [Parameter(Mandatory = $true)] [string]$Output,
    [ValidateRange(1, 100)] [int]$EpisodesPerDepth = 10,
    [double]$MaxPrimitiveDepth = 0.1010949334
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($oldPythonPath) { ";$oldPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    & $isaacLabBat -p (Join-Path $PSScriptRoot "evaluate_scripted_crouch.py") --checkpoint (Resolve-Path -LiteralPath $Checkpoint).Path --output $Output --episodes-per-depth $EpisodesPerDepth --max-primitive-depth $MaxPrimitiveDepth --viz none
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $Output "summary.json"))) { throw "Scripted CROUCH evaluation failed" }
}
finally { Pop-Location; $env:PYTHONPATH = $oldPythonPath }
