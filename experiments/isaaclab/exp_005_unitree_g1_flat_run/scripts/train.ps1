[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7", "stage8", "stage9", "run")]
    [string]$Stage = "stage1",

    [ValidateRange(1, 65536)]
    [int]$NumEnvs = 1024,

    [ValidateRange(1, 1000000)]
    [int]$MaxIterations = 500,

    [int]$Seed = 42,

    [string]$RunName = "",

    [string]$Checkpoint = "",

    [ValidateRange(0, 2)]
    [int]$CurriculumStage = 0,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$userProfile = [Environment]::GetFolderPath("UserProfile")
$isaacLabBat = Join-Path $userProfile "workspace\IsaacLab\isaaclab.bat"

if (-not (Test-Path -LiteralPath $isaacLabBat -PathType Leaf)) {
    throw "Isaac Lab launcher not found: $isaacLabBat"
}

$taskIds = @{
    stage1 = "Isaac-Velocity-Flat-G1-Run-Stage1-v0"
    stage2 = "Isaac-Velocity-Flat-G1-Run-Stage2-v0"
    stage3 = "Isaac-Velocity-Flat-G1-Run-Stage3-v0"
    stage4 = "Isaac-Velocity-Flat-G1-Run-Stage4-v0"
    stage5 = "Isaac-Velocity-Flat-G1-Run-Stage5-v0"
    stage6 = "Isaac-Velocity-Flat-G1-Run-Stage6-v0"
    stage7 = "Isaac-Velocity-Flat-G1-Run-Stage7-v0"
    stage8 = "Isaac-Velocity-Flat-G1-Run-Stage8-v0"
    stage9 = "Isaac-Velocity-Flat-G1-Run-Stage9-v0"
    run = "Isaac-Velocity-Flat-G1-Run-v0"
}
if (-not $RunName) {
    $RunName = $Stage
}

$trainArgs = @(
    "train",
    "--rl_library", "rsl_rl",
    "--task", $taskIds[$Stage],
    "--external_callback", "g1_flat_run.tasks.register_envs",
    "--viz", "none",
    "--num_envs", $NumEnvs,
    "--max_iterations", $MaxIterations,
    "--seed", $Seed,
    "--run_name", $RunName
)

if ($Checkpoint) {
    $checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
    $sourceRun = Split-Path (Split-Path $checkpointPath -Parent) -Leaf
    $checkpointName = Split-Path $checkpointPath -Leaf
    $resumeRun = ("_resume_{0}_{1}" -f $sourceRun, [IO.Path]::GetFileNameWithoutExtension($checkpointName)) `
        -replace '[^A-Za-z0-9_.-]', '_'
    $resumeDir = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_flat_run\$resumeRun"
    $stagedCheckpoint = Join-Path $resumeDir $checkpointName
    New-Item -ItemType Directory -Path $resumeDir -Force | Out-Null
    if (Test-Path -LiteralPath $stagedCheckpoint) {
        if ((Get-Item $stagedCheckpoint).Length -ne (Get-Item $checkpointPath).Length) {
            throw "Resume staging collision with different file: $stagedCheckpoint"
        }
    }
    else {
        Copy-Item -LiteralPath $checkpointPath -Destination $stagedCheckpoint
    }
    $trainArgs += @(
        "--resume",
        "--load_run", ("^{0}$" -f [regex]::Escape($resumeRun)),
        "--checkpoint", ("^{0}$" -f [regex]::Escape($checkpointName))
    )
}
$trainArgs += $ExtraArgs
if ($Stage -eq "stage5" -and $CurriculumStage -gt 0) {
    $trainArgs += "env.curriculum.speed_ceiling.params.initial_stage=$CurriculumStage"
}

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($previousPythonPath) { "$srcRoot;$previousPythonPath" } else { $srcRoot }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Push-Location $repositoryRoot
try {
    & $isaacLabBat @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "G1 running training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
