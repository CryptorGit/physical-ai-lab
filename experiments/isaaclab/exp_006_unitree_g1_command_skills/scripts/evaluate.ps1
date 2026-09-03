param(
    [Parameter(Mandatory = $true)] [string]$Checkpoint,
    [ValidateSet("run", "turn", "stop", "crouch", "sequence")] [string]$Skill = "sequence",
    [ValidateRange(1, 10000)] [int]$Episodes = 10,
    [ValidateSet("45", "full")] [string]$TurnCurriculum = "full",
    [ValidateSet("A", "B", "C")] [string]$StopCurriculum = "A",
    [ValidateSet("normal", "new_command_zero", "legacy_command_zero", "all_command_zero", "shuffle", "zero")] [string]$CommandAblation = "normal",
    [ValidateSet("current", "yaw_mask", "yaw_ankle_roll_mask", "lateral_mask", "symmetric")] [string]$StopResidualAblation = "current",
    [double]$StopFeedbackKHeading = 0.0,
    [double]$StopFeedbackKYawRate = 0.0,
    [double]$StopFeedbackKRoll = 0.0,
    [double]$StopFeedbackKRollRate = 0.0,
    [double]$StopFeedbackAlpha = 1.0,
    [double]$StopFeedbackMaxDeltaPerStep = 1.0,
    [double]$StopFeedbackBrakingScale = 1.0,
    [double]$StopFeedbackHoldScale = 1.0,
    [double]$StopFeedbackSingleSupportScale = 1.0,
    [double]$StopFeedbackFlightScale = 1.0,
    [double]$StopFeedbackYawSoftThreshold = [double]::PositiveInfinity,
    [double]$StopFeedbackYawHardThreshold = [double]::PositiveInfinity,
    [ValidateSet("zero", "damping_only")] [string]$StopFeedbackHardGuardMode = "zero",
    [string]$Output = "",
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$ExtraArgs
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$flatRunSrc = Join-Path $repositoryRoot "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$labels = @{ run = "Run"; turn = "Turn"; stop = "Stop"; crouch = "Crouch"; sequence = "Sequence" }
$taskLabel = if ($Skill -eq "turn" -and $TurnCurriculum -eq "full") { "TurnFull" } elseif ($Skill -eq "stop" -and $StopCurriculum -ne "A") { "Stop$StopCurriculum" } else { $labels[$Skill] }
if (-not $Output) { $Output = "results/exp_006_unitree_g1_command_skills/$Skill" }
$evaluationScript = if ($Skill -eq "crouch") { "evaluate_crouch.py" } else { "evaluate.py" }
$argsList = @(
    "-p", (Join-Path $PSScriptRoot $evaluationScript), "--checkpoint", (Resolve-Path -LiteralPath $Checkpoint).Path,
    "--task", "Isaac-Motion-Flat-G1-Command-$taskLabel-Eval-v0", "--episodes", $Episodes,
    "--output", $Output
)
if ($Skill -ne "crouch") { $argsList += @(
    "--command-ablation", $CommandAblation,
    "--stop-residual-ablation", $StopResidualAblation,
    "--stop-feedback-k-heading", $StopFeedbackKHeading,
    "--stop-feedback-k-yaw-rate", $StopFeedbackKYawRate,
    "--stop-feedback-k-roll", $StopFeedbackKRoll,
    "--stop-feedback-k-roll-rate", $StopFeedbackKRollRate,
    "--stop-feedback-alpha", $StopFeedbackAlpha,
    "--stop-feedback-max-delta-per-step", $StopFeedbackMaxDeltaPerStep,
    "--stop-feedback-braking-scale", $StopFeedbackBrakingScale,
    "--stop-feedback-hold-scale", $StopFeedbackHoldScale,
    "--stop-feedback-single-support-scale", $StopFeedbackSingleSupportScale,
    "--stop-feedback-flight-scale", $StopFeedbackFlightScale,
    "--stop-feedback-yaw-soft-threshold", $StopFeedbackYawSoftThreshold,
    "--stop-feedback-yaw-hard-threshold", $StopFeedbackYawHardThreshold,
    "--stop-feedback-hard-guard-mode", $StopFeedbackHardGuardMode
) }
$argsList += @("--viz", "none") + $ExtraArgs
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$srcRoot;$flatRunSrc" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Push-Location $repositoryRoot
try { & $isaacLabBat @argsList; if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $LASTEXITCODE" } }
finally { Pop-Location; $env:PYTHONPATH = $previousPythonPath }
