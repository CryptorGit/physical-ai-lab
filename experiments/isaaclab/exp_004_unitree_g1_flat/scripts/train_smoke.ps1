param(
    [ValidateRange(1, 65536)]
    [int]$NumEnvs = 64,

    [ValidateRange(1, 1000000)]
    [int]$MaxIterations = 5,

    [int]$Seed = 42,

    [string]$RunName = "smoke",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$userProfile = [Environment]::GetFolderPath("UserProfile")
$isaacLabBat = Join-Path $userProfile "workspace\IsaacLab\isaaclab.bat"

if (-not (Test-Path -LiteralPath $isaacLabBat -PathType Leaf)) {
    throw "Isaac Lab launcher not found: $isaacLabBat"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Push-Location $repositoryRoot
try {
    & $isaacLabBat train `
        --rl_library rsl_rl `
        --task Isaac-Velocity-Flat-G1-v0 `
        --viz none `
        --num_envs $NumEnvs `
        --max_iterations $MaxIterations `
        --seed $Seed `
        --experiment_name physical_ai_g1_flat `
        --run_name $RunName `
        @ExtraArgs

    if ($LASTEXITCODE -ne 0) {
        throw "G1 smoke training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
