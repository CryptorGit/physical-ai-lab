param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [ValidateRange(1, 4096)]
    [int]$NumEnvs = 1,

    [ValidateSet("run", "stage3", "stage4", "stage5", "stage6", "stage7", "stage8", "stage9")]
    [string]$Stage = "stage3",

    [ValidateSet("kit", "none")]
    [string]$Visualizer = "kit",

    [switch]$Video,

    [ValidateRange(1, 1000000)]
    [int]$VideoLength = 200,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$userProfile = [Environment]::GetFolderPath("UserProfile")
$isaacLabBat = Join-Path $userProfile "workspace\IsaacLab\isaaclab.bat"

$taskIds = @{
    run = "Isaac-Velocity-Flat-G1-Run-Play-v0"
    stage3 = "Isaac-Velocity-Flat-G1-Run-Stage3-Play-v0"
    stage4 = "Isaac-Velocity-Flat-G1-Run-Stage4-Play-v0"
    stage5 = "Isaac-Velocity-Flat-G1-Run-Stage5-Play-v0"
    stage6 = "Isaac-Velocity-Flat-G1-Run-Stage6-Play-v0"
    stage7 = "Isaac-Velocity-Flat-G1-Run-Stage7-Play-v0"
    stage8 = "Isaac-Velocity-Flat-G1-Run-Stage8-Play-v0"
    stage9 = "Isaac-Velocity-Flat-G1-Run-Stage9-Play-v0"
}

$playArgs = @(
    "play",
    "--rl_library", "rsl_rl",
    "--task", $taskIds[$Stage],
    "--external_callback", "g1_flat_run.tasks.register_envs",
    "--checkpoint", $checkpointPath,
    "--num_envs", $NumEnvs,
    "--viz", $Visualizer
)
if ($Video) {
    $playArgs += @("--video", "--video_length", $VideoLength)
}
$playArgs += $ExtraArgs

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($previousPythonPath) { "$srcRoot;$previousPythonPath" } else { $srcRoot }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Push-Location $repositoryRoot
try {
    & $isaacLabBat @playArgs
    if ($LASTEXITCODE -ne 0) {
        throw "G1 running playback failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
