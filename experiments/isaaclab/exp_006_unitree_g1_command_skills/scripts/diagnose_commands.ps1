param(
    [Parameter(Mandatory = $true)] [string]$Checkpoint,
    [Parameter(Mandatory = $true)] [ValidateSet("run", "turn", "stop", "crouch", "sequence")] [string]$Stage,
    [string]$Output = "results/exp_006_unitree_g1_command_skills/diagnostics/command.json",
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$ExtraArgs
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$flatRunSrc = Join-Path $repositoryRoot "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$arguments = @("-p", (Join-Path $PSScriptRoot "diagnose_commands.py"), "--checkpoint", (Resolve-Path -LiteralPath $Checkpoint).Path, "--stage", $Stage, "--output", $Output, "--viz", "none") + $ExtraArgs
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$srcRoot;$flatRunSrc" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
Push-Location $repositoryRoot
try { & $isaacLabBat @arguments; if ($LASTEXITCODE -ne 0) { throw "Command diagnostics failed: $LASTEXITCODE" } }
finally { Pop-Location; $env:PYTHONPATH = $previousPythonPath }
