[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)] [string]$Checkpoint,
    [Parameter(Mandatory = $true)] [ValidateSet("run", "turn", "stop", "crouch", "sequence")] [string]$Stage,
    [ValidateRange(1, 10000)] [int]$Episodes = 50,
    [ValidateRange(1, 10000)] [int]$AblationEpisodes = 10,
    [ValidateSet("45", "full")] [string]$TurnCurriculum = "full",
    [string]$BaselineGate = "",
    [string]$RetentionReference = "",
    [switch]$SkipGate,
    [string]$Output = ""
)
$ErrorActionPreference = "Stop"
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$stem = [IO.Path]::GetFileNameWithoutExtension($checkpointPath)
if (-not $Output) { $Output = "results/exp_006_unitree_g1_command_skills/checkpoints/$Stage/$stem" }
$learned = @{
    run = @("run")
    turn = @("run", "turn")
    stop = @("run", "turn", "stop")
    crouch = @("crouch")
    sequence = @("run", "turn", "stop", "sequence")
}[$Stage]
foreach ($skill in $learned) {
    $reuse = $false
    if ($Stage -eq "crouch" -and $skill -eq "crouch") {
        $summary = "$Output/crouch/normal/summary.json"
        if (Test-Path -LiteralPath $summary) {
            & python (Join-Path $PSScriptRoot "manage_crouch_sweep.py") --summary $summary --checkpoint $checkpointPath --episodes $Episodes --upgrade-compatible
            $reuse = $LASTEXITCODE -eq 0
            if ($reuse) { Write-Host "Reusing validated CROUCH evaluation: $summary" }
        }
    }
    if (-not $reuse) {
        $normalArgs = @{
            Checkpoint = $checkpointPath; Skill = $skill; TurnCurriculum = $TurnCurriculum
            Episodes = $Episodes; CommandAblation = "normal"; Output = "$Output/$skill/normal"
        }
        if ($skill -eq "crouch") { $normalArgs.ExtraArgs = @("--num-envs", "$Episodes") }
        & (Join-Path $PSScriptRoot "evaluate.ps1") @normalArgs
        if ($LASTEXITCODE -ne 0) { throw "Normal evaluation failed for $skill" }
    }
    if ($skill -eq "crouch") { continue }
    foreach ($ablation in @("new_command_zero", "legacy_command_zero", "all_command_zero", "shuffle")) {
        & (Join-Path $PSScriptRoot "evaluate.ps1") -Checkpoint $checkpointPath -Skill $skill -TurnCurriculum $TurnCurriculum -Episodes $AblationEpisodes -CommandAblation $ablation -Output "$Output/$skill/$ablation"
        if ($LASTEXITCODE -ne 0) { throw "$ablation evaluation failed for $skill" }
    }
}
$retention = "$Output/retention_provenance.json"
if ($Stage -eq "crouch") {
    if (-not $BaselineGate) { throw "CROUCH evaluation requires -BaselineGate for frozen retention inheritance" }
    if (-not $RetentionReference) {
        $RetentionReference = Join-Path (Split-Path $checkpointPath -Parent) "model_0.pt"
    }
    $proofPassed = $true
    try {
        & (Join-Path $PSScriptRoot "verify_crouch_retention.ps1") -Reference $RetentionReference -Checkpoint $checkpointPath -BaselineGate $BaselineGate -Output $retention
    }
    catch {
        $proofPassed = $false
        Write-Warning "Frozen-route proof failed; running checkpoint-specific RUN/TURN normal evaluation. $($_.Exception.Message)"
    }
    if (-not $proofPassed) {
        foreach ($skill in @("run", "turn")) {
            & (Join-Path $PSScriptRoot "evaluate.ps1") -Checkpoint $checkpointPath -Skill $skill -TurnCurriculum $TurnCurriculum -Episodes $Episodes -CommandAblation normal -Output "$Output/$skill/normal"
            if ($LASTEXITCODE -ne 0) { throw "Fallback normal evaluation failed for $skill" }
        }
    }
}
$diagnostic = "$Output/command_diagnostic.json"
& (Join-Path $PSScriptRoot "diagnose_commands.ps1") -Checkpoint $checkpointPath -Stage $Stage -Output $diagnostic
if ($LASTEXITCODE -ne 0) { throw "Command diagnostic failed" }
if ($Stage -eq "crouch") {
    $manifestArgs = @("--model-root", $Output, "--checkpoint", $checkpointPath, "--episodes", $Episodes, "--write-manifest")
    & python (Join-Path $PSScriptRoot "manage_crouch_sweep.py") @manifestArgs
    if ($LASTEXITCODE -ne 0) { throw "Expected CROUCH outputs are incomplete; see evaluation_manifest.json" }
}
$gateArgs = @("--stage", $Stage, "--turn-curriculum", $TurnCurriculum, "--root", $Output, "--diagnostic", $diagnostic, "--output", "$Output/gate.json")
if ($BaselineGate) { $gateArgs += @("--baseline", (Resolve-Path -LiteralPath $BaselineGate).Path) }
if ($Stage -eq "crouch") { $gateArgs += @("--retention-provenance", (Resolve-Path -LiteralPath $retention).Path) }
if (-not $SkipGate) {
    & python (Join-Path $PSScriptRoot "gate_checkpoint.py") @gateArgs
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint gate aggregation failed" }
}
