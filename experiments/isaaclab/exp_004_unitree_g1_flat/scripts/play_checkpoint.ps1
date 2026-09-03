param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [ValidateRange(1, 4096)]
    [int]$NumEnvs = 1,

    [ValidateSet("kit", "none")]
    [string]$Visualizer = "kit",

    [switch]$Video,

    [ValidateRange(1, 1000000)]
    [int]$VideoLength = 200,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$userProfile = [Environment]::GetFolderPath("UserProfile")
$isaacLabBat = Join-Path $userProfile "workspace\IsaacLab\isaaclab.bat"

if (-not (Test-Path -LiteralPath $isaacLabBat -PathType Leaf)) {
    throw "Isaac Lab launcher not found: $isaacLabBat"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$playArgs = @(
    "play",
    "--rl_library", "rsl_rl",
    "--task", "Isaac-Velocity-Flat-G1-Play-v0",
    "--checkpoint", $checkpointPath,
    "--num_envs", $NumEnvs,
    "--viz", $Visualizer
)
if ($Video) {
    $playArgs += @("--video", "--video_length", $VideoLength)
}
$playArgs += $ExtraArgs

Push-Location $repositoryRoot
try {
    & $isaacLabBat @playArgs
    if ($LASTEXITCODE -ne 0) {
        throw "G1 checkpoint playback failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
