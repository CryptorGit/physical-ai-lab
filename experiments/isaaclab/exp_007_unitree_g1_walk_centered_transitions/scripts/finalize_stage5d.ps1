[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Push-Location $repositoryRoot
try {
    python "$PSScriptRoot\finalize_stage5d.py"
    if ($LASTEXITCODE -ne 0) { throw "Stage 5D finalization failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
}
