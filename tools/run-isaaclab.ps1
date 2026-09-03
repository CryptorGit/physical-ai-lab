param(
    [Parameter(Mandatory = $true)]
    [string]$Script,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = "Stop"

$isaacLabRoot = Join-Path $HOME "workspace\IsaacLab"
$isaacLabBat = Join-Path $isaacLabRoot "isaaclab.bat"

if (-not (Test-Path $isaacLabBat)) {
    throw "Isaac Lab launcher not found: $isaacLabBat"
}

if (-not (Test-Path $Script)) {
    throw "Script not found: $Script"
}

$scriptPath = (Resolve-Path $Script).Path

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$previousPythonWarnings = $env:PYTHONWARNINGS
$env:PYTHONWARNINGS = "ignore::DeprecationWarning,ignore::FutureWarning"

try {
    Push-Location $isaacLabRoot

    & $isaacLabBat -p $scriptPath @ScriptArgs

    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
    $env:PYTHONWARNINGS = $previousPythonWarnings
}

if ($exitCode -ne 0) {
    throw "Isaac Lab process failed with exit code $exitCode"
}

exit 0