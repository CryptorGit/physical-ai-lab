param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [string]$Task = "Isaac-Velocity-Flat-G1-Run-Stage3-Eval-v0",

    [double[]]$Speeds = @(2.3, 2.4, 2.5, 2.6),

    [ValidateRange(1, 1000)]
    [int]$EpisodesPerSpeed = 50,

    [ValidateRange(1, 1000)]
    [int]$ParallelEnvsPerSpeed = 1,

    [ValidateRange(1, 10000000)]
    [int]$MaxSteps = 51000,

    [ValidateRange(0.0, 60.0)]
    [double]$SteadyStateStartS = 2.0,

    [ValidateRange(0, 2)]
    [int]$CurriculumStage = 0,

    [string]$OutputDir = "",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$evaluateScript = Join-Path $experimentRoot "scripts\evaluate.py"
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$userProfile = [Environment]::GetFolderPath("UserProfile")
$isaacLabBat = Join-Path $userProfile "workspace\IsaacLab\isaaclab.bat"

$evaluateArgs = @(
    "-p", $evaluateScript,
    "--checkpoint", $checkpointPath,
    "--task", $Task,
    "--speeds"
) + $Speeds + @(
    "--episodes_per_speed", $EpisodesPerSpeed,
    "--parallel_envs_per_speed", $ParallelEnvsPerSpeed,
    "--max_steps", $MaxSteps,
    "--steady_state_start_s", $SteadyStateStartS,
    "--curriculum_stage", $CurriculumStage,
    "--viz", "none"
)
if ($OutputDir) {
    $evaluateArgs += @("--output_dir", $OutputDir)
}
$evaluateArgs += $ExtraArgs

$previousPythonUtf8 = $env:PYTHONUTF8
$previousPythonIoEncoding = $env:PYTHONIOENCODING
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Push-Location $repositoryRoot
try {
    & $isaacLabBat @evaluateArgs
    if ($LASTEXITCODE -ne 0) {
        throw "G1 running evaluation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    $env:PYTHONUTF8 = $previousPythonUtf8
    $env:PYTHONIOENCODING = $previousPythonIoEncoding
}
