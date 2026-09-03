param(
    [Parameter(Mandatory = $true)] [string]$Checkpoint,
    [ValidateRange(5, 10)] [int]$EpisodesPerCategory = 5,
    [string]$SavedRunSummary = "",
    [string]$Output = "results/exp_006_unitree_g1_command_skills/turn_matrix",
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$ExtraArgs
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$flatRunSrc = Join-Path $repositoryRoot "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$arguments = @(
    "-p", (Join-Path $PSScriptRoot "evaluate_turn_matrix.py"),
    "--checkpoint", (Resolve-Path -LiteralPath $Checkpoint).Path,
    "--episodes-per-category", $EpisodesPerCategory,
    "--output", $Output,
    "--viz", "none"
)
if ($SavedRunSummary) { $arguments += @("--saved-run-summary", (Resolve-Path -LiteralPath $SavedRunSummary).Path) }
$arguments += $ExtraArgs
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$srcRoot;$flatRunSrc" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Push-Location $repositoryRoot
try { & $isaacLabBat @arguments; if ($LASTEXITCODE -ne 0) { throw "TURN matrix evaluation failed: $LASTEXITCODE" } }
finally { Pop-Location; $env:PYTHONPATH = $previousPythonPath }
