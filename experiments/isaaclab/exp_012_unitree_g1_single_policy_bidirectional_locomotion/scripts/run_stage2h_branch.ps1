param(
    [Parameter(Mandatory = $true)][string]$Branch,
    [ValidateSet("standard", "completion", "background")][string]$Mode,
    [int]$Horizon,
    [double]$Coefficient,
    [int]$Iterations = 4,
    [int]$SeedOffset = 0,
    [switch]$ForceRecompute,
    [string]$PythonExe = "C:\isaacsim\python.bat",
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$StageRoot = Join-Path $Repo "results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\stage2h_short_horizon_completion_replay_preflight"
$Parent = Join-Path $Repo "results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\stage2e_phase_a_run_acquisition_preflight\checkpoints\model_50.pt"
$BranchRoot = Join-Path $StageRoot "raw\$Branch"
$env:PYTHONPATH = (
    (Join-Path $Repo "experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src"),
    (Join-Path $Repo "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"),
    $Repo,
    $env:PYTHONPATH
) -join ";"
Set-Location $Repo

for ($Iteration = 1; $Iteration -le $Iterations; $Iteration++) {
    $Checkpoint = if ($Iteration -eq 1) {
        $Parent
    } else {
        Join-Path $BranchRoot ("state_{0}.pt" -f ($Iteration - 1))
    }
    $Corpus = Join-Path $BranchRoot ("on_policy_batch_{0}.pt" -f $Iteration)
    if (-not (Test-Path $Corpus)) {
        & $PythonExe (Join-Path $PSScriptRoot "collect_stage2h_on_policy.py") `
            --branch $Branch `
            --shadow-iteration $Iteration `
            --checkpoint $Checkpoint `
            --diagnostic-seed (20271000 + $SeedOffset + $Iteration) `
            --headless `
            --device $Device
        if ($LASTEXITCODE -ne 0) {
            throw "Stage 2H collection failed for $Branch iteration $Iteration"
        }
    }
    $Metrics = Join-Path $BranchRoot ("metrics_{0}.json" -f $Iteration)
    if ($ForceRecompute -or -not (Test-Path $Metrics)) {
        & $PythonExe (Join-Path $PSScriptRoot "run_stage2h_shadow_step.py") `
            --branch $Branch `
            --shadow-iteration $Iteration `
            --checkpoint $Checkpoint `
            --corpus $Corpus `
            --mode $Mode `
            --horizon $Horizon `
            --coefficient $Coefficient `
            --analysis-seed (20272000 + $SeedOffset + $Iteration)
        if ($LASTEXITCODE -ne 0) {
            throw "Stage 2H shadow step failed for $Branch iteration $Iteration"
        }
    }
    $Evaluation = Join-Path $BranchRoot ("eval_{0}_temporary_behavioral_evaluation.csv" -f $Iteration)
    if ($ForceRecompute -or -not (Test-Path $Evaluation)) {
        & $PythonExe (Join-Path $PSScriptRoot "evaluate_stage2h_shadow.py") `
            --branch $Branch `
            --shadow-iteration $Iteration `
            --diagnostic-seed (20273000 + $SeedOffset + $Iteration) `
            --headless `
            --device $Device
        if ($LASTEXITCODE -ne 0) {
            throw "Stage 2H temporary evaluation failed for $Branch iteration $Iteration"
        }
    }
}
