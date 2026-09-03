[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)] [string]$RunDirectory,
    [Parameter(Mandatory = $true)] [ValidateSet("run", "turn", "stop", "crouch", "sequence")] [string]$Stage,
    [ValidateRange(1, 10000)] [int]$Episodes = 50,
    [ValidateRange(1, 10000)] [int]$AblationEpisodes = 10,
    [ValidateSet("45", "full")] [string]$TurnCurriculum = "full",
    [string]$BaselineGate = "",
    [string]$Output = "",
    [string]$IntegratedCheckpoint = ""
)
$ErrorActionPreference = "Stop"
$runDir = (Resolve-Path -LiteralPath $RunDirectory).Path
if (-not $Output) {
    $runName = Split-Path $runDir -Leaf
    $Output = "results/exp_006_unitree_g1_command_skills/checkpoint_sweeps/$Stage/$runName"
}
$checkpoints = Get-ChildItem -LiteralPath $runDir -Filter "model_*.pt" -File | Sort-Object { [int]($_.BaseName -replace '^model_', '') }
if (-not $checkpoints) { throw "No model_*.pt checkpoints found in $runDir" }
foreach ($checkpoint in $checkpoints) {
    $evaluateArgs = @{
        Checkpoint = $checkpoint.FullName; Stage = $Stage; TurnCurriculum = $TurnCurriculum
        Episodes = $Episodes; AblationEpisodes = $AblationEpisodes; BaselineGate = $BaselineGate
        Output = "$Output/$($checkpoint.BaseName)"
    }
    if ($Stage -eq "crouch") { $evaluateArgs.SkipGate = $true }
    & (Join-Path $PSScriptRoot "evaluate_all.ps1") @evaluateArgs
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $($checkpoint.Name)" }
}
if ($Stage -eq "crouch") {
    & python (Join-Path $PSScriptRoot "manage_crouch_sweep.py") --sweep-root $Output --run-dir $runDir --episodes $Episodes --write-manifest
    if ($LASTEXITCODE -ne 0) { throw "CROUCH expected-output inventory is incomplete; all missing files are listed above" }
    foreach ($checkpoint in $checkpoints) {
        $modelRoot = "$Output/$($checkpoint.BaseName)"
        $gateArgs = @(
            "--stage", "crouch", "--turn-curriculum", $TurnCurriculum,
            "--root", $modelRoot, "--baseline", (Resolve-Path -LiteralPath $BaselineGate).Path,
            "--diagnostic", "$modelRoot/command_diagnostic.json",
            "--retention-provenance", "$modelRoot/retention_provenance.json",
            "--output", "$modelRoot/gate.json"
        )
        & python (Join-Path $PSScriptRoot "gate_checkpoint.py") @gateArgs
        if ($LASTEXITCODE -ne 0) { throw "Gate aggregation failed for $($checkpoint.Name)" }
    }
    & python (Join-Path $PSScriptRoot "manage_crouch_sweep.py") --sweep-root $Output --run-dir $runDir --episodes $Episodes --write-manifest
    if ($LASTEXITCODE -ne 0) { throw "CROUCH manifest refresh failed after gate aggregation" }
}
$selection = "$Output/best_checkpoint.json"
$selectionArgs = @("--stage", $Stage, "--evaluations", $Output, "--run-dir", $runDir, "--output", $selection)
if ($Stage -eq "stop" -and (Test-Path -LiteralPath (Join-Path $runDir "model_0.pt"))) {
    $selectionArgs += @("--corrective-parent", "model_0")
}
& python (Join-Path $PSScriptRoot "select_best_checkpoint.py") @selectionArgs
if ($LASTEXITCODE -ne 0) { throw "No checkpoint passed all gates; see per-checkpoint gate.json files" }
if ($IntegratedCheckpoint) {
    if ($Stage -ne "sequence") { throw "-IntegratedCheckpoint is only valid for the sequence stage" }
    $selected = (Get-Content -LiteralPath $selection -Raw | ConvertFrom-Json).selected.checkpoint
    $target = [IO.Path]::GetFullPath($IntegratedCheckpoint)
    New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
    Copy-Item -LiteralPath $selected -Destination $target -Force
    Write-Host "Integrated RUN/TURN/STOP checkpoint: $target"
}
