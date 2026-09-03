[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)] [string]$Checkpoint,
    [Parameter(Mandatory = $true)] [string]$Output,
    [ValidateSet("full", "finite", "refine", "deep")] [string]$SearchMode = "full"
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$flatRunSrc = Join-Path $repositoryRoot "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$arguments = @(
    "-p", (Join-Path $PSScriptRoot "audit_crouch_authority.py"),
    "--checkpoint", (Resolve-Path -LiteralPath $Checkpoint).Path,
    "--output", $Output, "--search-mode", $SearchMode, "--viz", "none"
)
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$srcRoot;$flatRunSrc" + $(if ($oldPythonPath) { ";$oldPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    & $isaacLabBat @arguments
    if ($LASTEXITCODE -ne 0) { throw "CROUCH authority audit failed: $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath (Join-Path $Output "summary.json"))) {
        throw "CROUCH authority audit produced no summary.json"
    }
}
finally { Pop-Location; $env:PYTHONPATH = $oldPythonPath }
