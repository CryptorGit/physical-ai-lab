[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("RUN_TURN_RUN", "STAND_CROUCH_STAND", "UNSUPPORTED_RUN_TO_CROUCH", "UNSUPPORTED_STEP_OVER", "UNSUPPORTED_LAND")]
    [string]$Demo,
    [int]$Seed = 20260723,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$artifactRoot = Join-Path $repositoryRoot "artifacts\exp_006_unitree_g1_command_skills\command_system_v1"
$provenancePath = Join-Path $artifactRoot "skill_provenance.json"
$manifestPath = Join-Path $artifactRoot "capability_manifest.json"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"

function Require-Path([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Label is missing: $Path" }
    return (Resolve-Path -LiteralPath $Path).Path
}
function Resolve-RepositoryReference([string]$Reference, [string]$Label) {
    $candidate = if ([IO.Path]::IsPathRooted($Reference)) { $Reference } else { Join-Path $repositoryRoot $Reference }
    return Require-Path $candidate $Label
}
function Show-Contract(
    [string]$Current, [string]$Requested, [string]$CurrentFamily, [string]$RequestedFamily,
    [string]$Base, [string]$Controller, [bool]$Supported, [bool]$Started,
    [string]$Phase, [string]$Reason
) {
    Write-Host "demo_name=$Demo"
    Write-Host "current_controller_state=$Current requested_controller_state=$Requested"
    Write-Host "current_family=$CurrentFamily requested_family=$RequestedFamily"
    Write-Host "active_base_option=$Base active_scripted_controller=$Controller"
    Write-Host "transition_supported=$($Supported.ToString().ToLower()) transition_started=$($Started.ToString().ToLower())"
    Write-Host "sequence_phase=$Phase rejection_reason=$Reason"
}

Require-Path $artifactRoot "command_system_v1 artifact" | Out-Null
Require-Path $provenancePath "skill provenance" | Out-Null
Require-Path $manifestPath "capability manifest" | Out-Null
Require-Path $isaacLabBat "Isaac Lab launcher" | Out-Null
$provenance = Get-Content -LiteralPath $provenancePath -Raw | ConvertFrom-Json
$model31 = Resolve-RepositoryReference $provenance.stage4_and_skill_checkpoint "RUN/TURN checkpoint"
$stage2 = Resolve-RepositoryReference $provenance.stage2_standing_checkpoint "Stage 2 standing checkpoint"
$crouchCandidate = $provenance.crouch_checkpoint
if (-not (Test-Path -LiteralPath $crouchCandidate)) {
    $crouchCandidate = Join-Path $repositoryRoot "artifacts\exp_006_unitree_g1_command_skills\crouch_standing_option_stage2_model4246\model_0.pt"
}
$crouchCheckpoint = Resolve-RepositoryReference $crouchCandidate "CROUCH standing-option checkpoint"

Write-Host "repository_root=$repositoryRoot"
Write-Host "artifact=$artifactRoot"
Write-Host "run_checkpoint=$model31"
Write-Host "stand_checkpoint=$stage2"
Write-Host "crouch_checkpoint=$crouchCheckpoint"
if ($ValidateOnly) { Write-Host "validation=PASS demo=$Demo gui_started=false"; return }

Push-Location $repositoryRoot
try {
    switch ($Demo) {
        "RUN_TURN_RUN" {
            $output = Join-Path $repositoryRoot "results\exp_006_unitree_g1_command_skills\command_system_v1\gui_run_turn_run"
            Show-Contract RUN TURN RUNNING_FAMILY RUNNING_FAMILY stage4_running_base none $true $true RUN_TURN_RUN none
            $command = @("-p", (Join-Path $PSScriptRoot "evaluate_run_turn_run.py"), "--checkpoint", $model31, "--episodes", "1", "--seed", "$Seed", "--viz", "kit", "--run-duration-min", "2.5", "--run-duration-max", "3.0", "--recovery-duration-min", "2.5", "--recovery-duration-max", "3.2", "--output", $output)
            Write-Host "launch_command=$isaacLabBat $($command -join ' ')"
            & $isaacLabBat @command
            if ($LASTEXITCODE -ne 0) { throw "RUN_TURN_RUN GUI failed with exit code $LASTEXITCODE (checkpoint=$model31)" }
            $summary = Get-Content (Join-Path $output "summary.json") -Raw | ConvertFrom-Json
            Write-Host "sequence_result=$($summary.gate_pass) fall=$($summary.fall_rate) saturation=$($summary.saturation_failure_rate) final_heading_rad=$($summary.final_heading_error_rad)"
        }
        "STAND_CROUCH_STAND" {
            Show-Contract STAND CROUCH_SHALLOW STANDING_FAMILY STANDING_FAMILY stage2_standing_base_model_4246 scripted_shallow_v1 $true $true STAND_CROUCH_STAND none
            $command = @("-Depth", "0.09", "-Checkpoint", $crouchCheckpoint)
            Write-Host "launch_command=$(Join-Path $PSScriptRoot 'play_crouch_shallow.ps1') $($command -join ' ')"
            & (Join-Path $PSScriptRoot "play_crouch_shallow.ps1") @command
            if ($LASTEXITCODE -ne 0) { throw "STAND_CROUCH_STAND GUI failed (checkpoint=$crouchCheckpoint)" }
            $summaryPath = Join-Path $repositoryRoot "results\exp_006_unitree_g1_command_skills\crouch_shallow_scripted_v1\gui_0.09\summary.json"
            if (Test-Path $summaryPath) {
                $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
                $skill = $summary.skills.CROUCH_SHALLOW
                Write-Host "sequence_result=$($summary.skill_success_rate) fall=$($summary.fall_rate) saturation=$($skill.saturation_failure_rate) final_depth_error_m=$($skill.depth_error_m)"
            }
        }
        "UNSUPPORTED_RUN_TO_CROUCH" {
            Show-Contract RUN CROUCH_SHALLOW RUNNING_FAMILY STANDING_FAMILY stage4_running_base none $false $false REJECTED CROSS_BASE_FAMILY_TRANSITION_UNRESOLVED
            Write-Host "launch_command=$(Join-Path $PSScriptRoot 'play.ps1') -Checkpoint $model31 -Skill run -NumEnvs 1 -Visualizer kit"
            & (Join-Path $PSScriptRoot "play.ps1") -Checkpoint $model31 -Skill run -NumEnvs 1 -Visualizer kit
            Write-Host "sequence_result=safe_rejection fall=not_evaluated saturation=not_evaluated action_offset=zero"
        }
        "UNSUPPORTED_STEP_OVER" {
            Show-Contract STAND STEP_OVER STANDING_FAMILY UNSUPPORTED stage2_standing_base_model_4246 none $false $false REJECTED OPTIMIZATION_FAILURE
            Write-Host "launch_command=$(Join-Path $PSScriptRoot 'play_standing_base.ps1') -Checkpoint $stage2 -Candidate unsupported_step_over -Seed $Seed"
            & (Join-Path $PSScriptRoot "play_standing_base.ps1") -Checkpoint $stage2 -Candidate unsupported_step_over -Seed $Seed
            Write-Host "sequence_result=safe_rejection fall=not_evaluated saturation=not_evaluated step_over_offset=zero"
        }
        "UNSUPPORTED_LAND" {
            Show-Contract STAND LAND STANDING_FAMILY UNSUPPORTED stage2_standing_base_model_4246 none $false $false REJECTED POSITION_OFFSET_LANDING_CONTROLLER_FAILED
            Write-Host "launch_command=$(Join-Path $PSScriptRoot 'play_standing_base.ps1') -Checkpoint $stage2 -Candidate unsupported_land -Seed $Seed"
            & (Join-Path $PSScriptRoot "play_standing_base.ps1") -Checkpoint $stage2 -Candidate unsupported_land -Seed $Seed
            Write-Host "sequence_result=safe_rejection fall=not_evaluated saturation=not_evaluated landing_offset=zero"
        }
    }
}
finally { Pop-Location }
