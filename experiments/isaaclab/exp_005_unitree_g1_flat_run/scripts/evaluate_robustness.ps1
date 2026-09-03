param(
    [string]$Checkpoint = ".\logs\rsl_rl\physical_ai_g1_flat_run\2026-07-18_10-51-47_stage9_5mps_cycle_knee010_1024_150\model_5840.pt",

    [ValidateRange(2, 1000)]
    [int]$EpisodesPerCondition = 20,

    [ValidateRange(1, 1000)]
    [int]$ParallelEnvs = 10,

    [ValidateRange(1000, 10000000)]
    [int]$MaxSteps = 5200,

    [string]$OutputRoot = ".\results\exp_005_unitree_g1_flat_run\stage9_robustness_5mps",

    [string[]]$Conditions = @(),

    [switch]$Force
)

$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$evaluateScript = Join-Path $PSScriptRoot "evaluate.ps1"
$summaryScript = Join-Path $PSScriptRoot "summarize_robustness.py"
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$outputRootPath = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputRoot))
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"

if ($EpisodesPerCondition % $ParallelEnvs -ne 0) {
    throw "EpisodesPerCondition must be divisible by ParallelEnvs."
}

$matrix = @(
    @{ Name = "baseline"; Args = @() },
    @{ Name = "friction_080"; Args = @("--friction_scale", "0.8") },
    @{ Name = "friction_120"; Args = @("--friction_scale", "1.2") },
    @{ Name = "mass_090"; Args = @("--mass_scale", "0.9") },
    @{ Name = "mass_110"; Args = @("--mass_scale", "1.1") },
    @{ Name = "com_backward_20mm"; Args = @("--com_shift_x_m", "-0.02") },
    @{ Name = "com_forward_20mm"; Args = @("--com_shift_x_m", "0.02") },
    @{ Name = "stiffness_090"; Args = @("--stiffness_scale", "0.9") },
    @{ Name = "stiffness_110"; Args = @("--stiffness_scale", "1.1") },
    @{ Name = "damping_090"; Args = @("--damping_scale", "0.9") },
    @{ Name = "damping_110"; Args = @("--damping_scale", "1.1") },
    @{ Name = "pd_090"; Args = @("--stiffness_scale", "0.9", "--damping_scale", "0.9") },
    @{ Name = "pd_110"; Args = @("--stiffness_scale", "1.1", "--damping_scale", "1.1") },
    @{ Name = "action_delay_1"; Args = @("--action_delay_steps", "1") },
    @{ Name = "action_delay_2"; Args = @("--action_delay_steps", "2") },
    @{ Name = "external_force_fore_aft"; Args = @("--external_force_axis", "x") },
    @{ Name = "external_force_lateral"; Args = @("--external_force_axis", "y") },
    @{ Name = "small_rough_10mm"; Args = @("--small_rough_terrain") }
)

if ($Conditions.Count -gt 0) {
    $unknown = $Conditions | Where-Object { $_ -notin $matrix.Name }
    if ($unknown) {
        throw "Unknown conditions: $($unknown -join ', ')"
    }
    $matrix = $matrix | Where-Object { $_.Name -in $Conditions }
}

New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null
foreach ($condition in $matrix) {
    $conditionDir = Join-Path $outputRootPath $condition.Name
    $summaryPath = Join-Path $conditionDir "summary.json"
    if ((Test-Path -LiteralPath $summaryPath) -and -not $Force) {
        Write-Host "[skip] $($condition.Name): summary.json already exists"
        continue
    }
    if ($Force -and (Test-Path -LiteralPath $summaryPath)) {
        Remove-Item -LiteralPath $summaryPath -Force
    }
    Write-Host "[run] $($condition.Name)"
    $extra = @("--condition_name", $condition.Name) + $condition.Args
    & $evaluateScript `
        -Checkpoint $checkpointPath `
        -Task "Isaac-Velocity-Flat-G1-Run-Stage9-Eval-v0" `
        -Speeds 5.0 `
        -EpisodesPerSpeed $EpisodesPerCondition `
        -ParallelEnvsPerSpeed $ParallelEnvs `
        -MaxSteps $MaxSteps `
        -SteadyStateStartS 2.0 `
        -OutputDir $conditionDir `
        -ExtraArgs $extra
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        throw "Condition '$($condition.Name)' did not produce summary.json"
    }
}

Push-Location $repositoryRoot
try {
    & $isaacLabBat -p $summaryScript --root $outputRootPath
    if ($LASTEXITCODE -ne 0) {
        throw "Robustness summary aggregation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
