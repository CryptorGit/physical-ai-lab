[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("run", "turn", "stop", "crouch", "sequence")]
    [string]$Skill = "run",
    [ValidateRange(1, 65536)] [int]$NumEnvs = 1024,
    [ValidateRange(1, 1000000)] [int]$MaxIterations = 500,
    [ValidateSet("45", "full")] [string]$TurnCurriculum = "45",
    [ValidateSet("A", "B", "C")] [string]$StopCurriculum = "A",
    [int]$Seed = 42,
    [string]$RunName = "",
    [string]$WarmStartCheckpoint = "",
    [string]$Checkpoint = "",
    [switch]$NewStage,
    [string]$ParentGate = "",
    [string]$StandingBaseGate = "",
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
if ($WarmStartCheckpoint -and $Checkpoint) {
    throw "Use either -WarmStartCheckpoint (123-D exp_005) or -Checkpoint (152-D exp_006 resume)."
}
if ($NewStage -and -not $Checkpoint) { throw "-NewStage requires -Checkpoint." }
if ($Skill -eq "crouch" -and $NewStage) {
    throw "CROUCH standing-option parents are pre-rebased; do not use -NewStage."
}
if ($TurnCurriculum -ne "45" -and $Skill -ne "turn") { throw "-TurnCurriculum is only valid with -Skill turn." }
if ($StopCurriculum -ne "A" -and $Skill -ne "stop") { throw "-StopCurriculum is only valid with -Skill stop." }
if ($Skill -ne "run") {
    if (-not $ParentGate) { throw "$Skill training is locked until the previous stage passes; provide -ParentGate <gate.json>." }
    $gate = Get-Content -LiteralPath (Resolve-Path -LiteralPath $ParentGate).Path -Raw | ConvertFrom-Json
    $requiredParent = if ($Skill -eq "turn" -and $TurnCurriculum -eq "full") { "turn" } else {
        @{ turn = "run"; stop = "turn"; crouch = "turn"; sequence = "stop" }[$Skill]
    }
    if (-not $gate.eligible_for_best -or $gate.stage -ne $requiredParent) {
        throw "$Skill requires an eligible '$requiredParent' gate; received stage='$($gate.stage)', eligible='$($gate.eligible_for_best)'."
    }
    if ($Skill -eq "turn" -and $TurnCurriculum -eq "full" -and $gate.turn_curriculum -ne "45") {
        throw "Full TURN curriculum requires an eligible 45-degree TURN gate."
    }
    if ($Skill -in @("stop", "crouch") -and $gate.turn_curriculum -ne "full") {
        throw "$($Skill.ToUpper()) remains locked until the full 45/90-degree TURN gate passes."
    }
}
if ($Skill -eq "crouch") {
    if (-not $StandingBaseGate) { throw "CROUCH is locked until a standing-base gate passes; provide -StandingBaseGate." }
    $standingGate = Get-Content -LiteralPath (Resolve-Path -LiteralPath $StandingBaseGate).Path -Raw | ConvertFrom-Json
    if (-not $standingGate.eligible_for_crouch -or $standingGate.gate -ne "standing_base_v1") {
        throw "CROUCH requires an eligible standing_base_v1 gate."
    }
    if (-not $Checkpoint) { throw "CROUCH requires a rebased standing-option -Checkpoint." }
    $checkpointResolved = (Resolve-Path -LiteralPath $Checkpoint).Path
    $standingSidecar = "$checkpointResolved.standing_option.json"
    if (-not (Test-Path -LiteralPath $standingSidecar -PathType Leaf)) {
        throw "CROUCH checkpoint lacks standing-option provenance sidecar: $standingSidecar"
    }
    $standingInfo = Get-Content -LiteralPath $standingSidecar -Raw | ConvertFrom-Json
    if ($standingInfo.standing_candidate -ne $standingGate.candidate -or -not $standingInfo.crouch_initial_residual_bitwise_zero) {
        throw "CROUCH checkpoint does not match the eligible standing-base gate."
    }
}
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$flatRunSrc = Join-Path $repositoryRoot "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
if (-not (Test-Path -LiteralPath $isaacLabBat -PathType Leaf)) { throw "Isaac Lab launcher not found: $isaacLabBat" }

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$srcRoot;$flatRunSrc" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$labels = @{ run = "Run"; turn = "Turn"; stop = "Stop"; crouch = "Crouch"; sequence = "Sequence" }
$taskLabel = if ($Skill -eq "turn" -and $TurnCurriculum -eq "full") { "TurnFull" } elseif ($Skill -eq "stop" -and $StopCurriculum -ne "A") { "Stop$StopCurriculum" } else { $labels[$Skill] }
$task = "Isaac-Motion-Flat-G1-Command-$taskLabel-v0"
if (-not $RunName) { $RunName = $Skill }
$trainArgs = @(
    "train", "--rl_library", "rsl_rl", "--task", $task,
    "--external_callback", "g1_command_skills.tasks.register_envs",
    "--viz", "none", "--num_envs", $NumEnvs, "--max_iterations", $MaxIterations,
    "--seed", $Seed, "--run_name", $RunName
)

$resumeRoot = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_command_skills"
if ($WarmStartCheckpoint) {
    $source = (Resolve-Path -LiteralPath $WarmStartCheckpoint).Path
    $transferRun = "_warmstart_skill_routes_v2_stage4_" + [IO.Path]::GetFileNameWithoutExtension($source)
    $transferDir = Join-Path $resumeRoot $transferRun
    $transferPath = Join-Path $transferDir "model_0.pt"
    New-Item -ItemType Directory -Path $transferDir -Force | Out-Null
    if (-not (Test-Path -LiteralPath $transferPath)) {
        & $isaacLabBat -p (Join-Path $PSScriptRoot "transfer_checkpoint.py") --input $source --output $transferPath
        if ($LASTEXITCODE -ne 0) { throw "Checkpoint transfer failed with exit code $LASTEXITCODE" }
    }
    $trainArgs += @("--resume", "--load_run", ("^{0}$" -f [regex]::Escape($transferRun)), "--checkpoint", "^model_0.pt$")
}
elseif ($Checkpoint) {
    $source = (Resolve-Path -LiteralPath $Checkpoint).Path
    $checkpointName = Split-Path $source -Leaf
    $prefix = if ($NewStage -and $Skill -eq "stop") { "_stage_stop_corrective_v1" } elseif ($NewStage) { "_stage_skill_routes_v2_$Skill" } else { "_resume" }
    $resumeRun = ("{0}_{1}_{2}" -f $prefix, (Split-Path (Split-Path $source -Parent) -Leaf), [IO.Path]::GetFileNameWithoutExtension($checkpointName)) -replace '[^A-Za-z0-9_.-]', '_'
    $resumeDir = Join-Path $resumeRoot $resumeRun
    $staged = Join-Path $resumeDir $checkpointName
    New-Item -ItemType Directory -Path $resumeDir -Force | Out-Null
    if ($NewStage -and -not (Test-Path -LiteralPath $staged)) {
        & $isaacLabBat -p (Join-Path $PSScriptRoot "rebase_stage_checkpoint.py") --input $source --output $staged --stage $Skill
        if ($LASTEXITCODE -ne 0) { throw "Stage checkpoint rebase failed with exit code $LASTEXITCODE" }
    }
    elseif (-not (Test-Path -LiteralPath $staged)) { Copy-Item -LiteralPath $source -Destination $staged }
    elseif (-not $NewStage -and (Get-Item $staged).Length -ne (Get-Item $source).Length) { throw "Resume staging collision: $staged" }
    $trainArgs += @("--resume", "--load_run", ("^{0}$" -f [regex]::Escape($resumeRun)), "--checkpoint", ("^{0}$" -f [regex]::Escape($checkpointName)))
}
$trainArgs += $ExtraArgs

Push-Location $repositoryRoot
try {
    & $isaacLabBat @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "G1 command-skill training failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location; $env:PYTHONPATH = $previousPythonPath }
