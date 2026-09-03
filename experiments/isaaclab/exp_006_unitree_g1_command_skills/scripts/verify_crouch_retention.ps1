[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)] [string]$Reference,
    [Parameter(Mandatory = $true)] [string]$Checkpoint,
    [Parameter(Mandatory = $true)] [string]$BaselineGate,
    [Parameter(Mandatory = $true)] [string]$Output
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$arguments = @(
    "-p", (Join-Path $PSScriptRoot "verify_crouch_retention.py"),
    "--reference", (Resolve-Path -LiteralPath $Reference).Path,
    "--checkpoint", (Resolve-Path -LiteralPath $Checkpoint).Path,
    "--baseline-gate", (Resolve-Path -LiteralPath $BaselineGate).Path,
    "--output", $Output
)
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$srcRoot" + $(if ($oldPythonPath) { ";$oldPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    & $isaacLabBat @arguments
    if ($LASTEXITCODE -ne 0) { throw "CROUCH retention verification failed: $LASTEXITCODE" }
}
finally { Pop-Location; $env:PYTHONPATH = $oldPythonPath }
