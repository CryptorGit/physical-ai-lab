param(
    [int]$NumEnvs = 64,
    [int]$MaxIterations = 20
)

$ErrorActionPreference = "Stop"

# PythonとPowerShellの文字コードをUTF-8へ統一する。
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

chcp 65001 | Out-Null

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\..\..\.."
$Launcher = Join-Path $ProjectRoot "tools\run-isaaclab.ps1"
$TrainScript = Join-Path $PSScriptRoot "train.py"

$OutputRoot = Join-Path `
    $ProjectRoot `
    "experiments\isaaclab\exp_002_peg_observation_ablation\results\pilot"

$Tasks = @(
    @{
        Name = "baseline"
        Task = "Isaac-PegObservationBaseline-Direct-v0"
    },
    @{
        Name = "no_angvel"
        Task = "Isaac-PegObservationNoAngvel-Direct-v0"
    },
    @{
        Name = "no_linvel"
        Task = "Isaac-PegObservationNoLinvel-Direct-v0"
    },
    @{
        Name = "no_velocity"
        Task = "Isaac-PegObservationNoVelocity-Direct-v0"
    },
    @{
        Name = "position_only"
        Task = "Isaac-PegObservationPositionOnly-Direct-v0"
    }
)

$Seeds = @(42, 43, 44)

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

foreach ($TaskConfig in $Tasks) {
    foreach ($Seed in $Seeds) {
        $RunName = "$($TaskConfig.Name)_seed_$Seed"
        $LogFile = Join-Path $OutputRoot "$RunName.log"

        Write-Host ""
        Write-Host "========================================"
        Write-Host "Run: $RunName"
        Write-Host "Task: $($TaskConfig.Task)"
        Write-Host "Seed: $Seed"
        Write-Host "Iterations: $MaxIterations"
        Write-Host "========================================"

        & $Launcher `
            -Script $TrainScript `
            --task $TaskConfig.Task `
            --num_envs $NumEnvs `
            --max_iterations $MaxIterations `
            --seed $Seed |
            Tee-Object -FilePath $LogFile

        if ($LASTEXITCODE -ne 0) {
            throw "Training failed: $RunName"
        }
    }
}