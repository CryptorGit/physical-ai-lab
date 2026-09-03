[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("TURN_LEFT_90", "TURN_RIGHT_90", "TURN_S_CURVE", "CROUCH_SHOWCASE", "SAFE_REJECTION", "FULL_REEL")]
    [string]$Showcase,
    [ValidateSet("WORLD_FIXED", "FOLLOW_POSITION", "TOP_DOWN")]
    [string]$Camera = "WORLD_FIXED",
    [int]$Seed = 20260723,
    [switch]$Record,
    [string]$OutputPath = "",
    [ValidateRange(640, 7680)] [int]$Width = 1920,
    [ValidateRange(480, 4320)] [int]$Height = 1080,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$artifactRoot = Join-Path $repositoryRoot "artifacts\exp_006_unitree_g1_command_skills\command_system_v1"
$provenancePath = Join-Path $artifactRoot "skill_provenance.json"
$manifestPath = Join-Path $artifactRoot "capability_manifest.json"
$crouchArtifact = Join-Path $repositoryRoot "artifacts\exp_006_unitree_g1_command_skills\crouch_shallow_scripted_v1"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"

function Require-Path([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Label is missing: $Path" }
    return (Resolve-Path -LiteralPath $Path).Path
}
function Resolve-RepositoryReference([string]$Reference, [string]$Label) {
    $candidate = if ([IO.Path]::IsPathRooted($Reference)) { $Reference } else { Join-Path $repositoryRoot $Reference }
    return Require-Path $candidate $Label
}

Require-Path $artifactRoot "command_system_v1 artifact" | Out-Null
Require-Path $provenancePath "skill provenance" | Out-Null
Require-Path $manifestPath "capability manifest" | Out-Null
Require-Path $crouchArtifact "CROUCH_SHALLOW formal artifact" | Out-Null
Require-Path $isaacLabBat "Isaac Lab launcher" | Out-Null
$provenance = Get-Content -LiteralPath $provenancePath -Raw | ConvertFrom-Json
$runCheckpoint = Resolve-RepositoryReference $provenance.stage4_and_skill_checkpoint "RUN/TURN checkpoint"
$standingCheckpoint = Resolve-RepositoryReference $provenance.stage2_standing_checkpoint "Stage 2 standing checkpoint"
$crouchCheckpoint = Resolve-RepositoryReference $provenance.crouch_checkpoint "CROUCH standing-option checkpoint"

$spec = @{
    TURN_LEFT_90 = @{ Angle = "+90 deg"; Sequence = "RUN 3s -> LEFT 90 -> RUN 4s"; Unsupported = $false }
    TURN_RIGHT_90 = @{ Angle = "-90 deg"; Sequence = "RUN 3s -> RIGHT 90 -> RUN 4s"; Unsupported = $false }
    TURN_S_CURVE = @{ Angle = "+90 deg, -90 deg"; Sequence = "RUN -> LEFT 90 -> RUN -> RIGHT 90 -> RUN (showcase-only multi-turn)"; Unsupported = $false }
    CROUCH_SHOWCASE = @{ Angle = "n/a"; Sequence = "STAND 2s -> CROUCH 0.09m -> HOLD 2s -> STAND -> HOLD 2s"; Unsupported = $false }
    SAFE_REJECTION = @{ Angle = "n/a"; Sequence = "STAND -> request STEP_OVER -> safe rejection"; Unsupported = $true }
    FULL_REEL = @{ Angle = "+90 deg, -90 deg"; Sequence = "LEFT scene | reset/cut | RIGHT scene | reset/cut | CROUCH scene | reset/cut | rejection scene"; Unsupported = $true }
}
$selected = $spec[$Showcase]
if ($Record -and [string]::IsNullOrWhiteSpace($OutputPath)) {
    throw "-Record requires -OutputPath."
}
$resolvedOutputPath = ""
if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
    $outputCandidate = if ([IO.Path]::IsPathRooted($OutputPath)) { $OutputPath } else { Join-Path $repositoryRoot $OutputPath }
    $resolvedOutputPath = [IO.Path]::GetFullPath($outputCandidate)
}

Write-Host "=== EXP_006 SHOWCASE PREFLIGHT ==="
Write-Host "selected_showcase=$Showcase"
Write-Host "selected_camera=$Camera"
Write-Host "camera_yaw_follow=false"
Write-Host "run_checkpoint=$runCheckpoint"
Write-Host "standing_checkpoint=$standingCheckpoint"
Write-Host "crouch_artifact=$crouchArtifact"
Write-Host "crouch_checkpoint=$crouchCheckpoint"
Write-Host "requested_turn_angle=$($selected.Angle)"
Write-Host "run_speed_mps=2.4 turn_speed_mps=2.0"
Write-Host "expected_sequence=$($selected.Sequence)"
Write-Host "contains_unsupported_request=$($selected.Unsupported.ToString().ToLower())"
Write-Host "contains_cross_family_transition=false"
Write-Host "record_enabled=$($Record.IsPresent.ToString().ToLower())"
Write-Host "output_path=$resolvedOutputPath"
Write-Host "viewport_resolution=${Width}x${Height}"
Write-Host "formal_artifact_mutation=false formal_evaluation_mutation=false"
if ($ValidateOnly) {
    Write-Host "validation=PASS gui_started=false"
    return
}

$showcaseScript = Join-Path $PSScriptRoot "play_showcase.py"
$reelScript = Join-Path $PSScriptRoot "assemble_showcase_reel.py"
$resultRoot = Join-Path $repositoryRoot "results\exp_006_unitree_g1_command_skills\showcase_v1"

function Invoke-Scene([string]$Scene, [string]$SceneOutput, [string]$VideoOutput) {
    $command = @(
        "-p", $showcaseScript,
        "--showcase", $Scene,
        "--camera", $Camera,
        "--run-checkpoint", $runCheckpoint,
        "--standing-checkpoint", $standingCheckpoint,
        "--crouch-checkpoint", $crouchCheckpoint,
        "--output", $SceneOutput,
        "--seed", "$Seed",
        "--width", "$Width",
        "--height", "$Height",
        "--viz", "kit"
    )
    if ($Record) {
        $command += @("--record", "--output-path", $VideoOutput)
    }
    Write-Host "scene_launch=$Scene"
    Write-Host "launch_command=$isaacLabBat $($command -join ' ')"
    & $isaacLabBat @command
    if ($LASTEXITCODE -ne 0) { throw "Showcase scene $Scene failed with exit code $LASTEXITCODE" }
}

Push-Location $repositoryRoot
try {
    if ($Showcase -ne "FULL_REEL") {
        $sceneOutput = Join-Path $resultRoot $Showcase.ToLowerInvariant()
        Invoke-Scene $Showcase $sceneOutput $resolvedOutputPath
    }
    else {
        $scenes = @("TURN_LEFT_90", "TURN_RIGHT_90", "CROUCH_SHOWCASE", "SAFE_REJECTION")
        $clips = @()
        for ($index = 0; $index -lt $scenes.Count; $index++) {
            $scene = $scenes[$index]
            $sceneNumber = $index + 1
            $sceneOutput = Join-Path $resultRoot ("full_reel\scene_{0:d2}_{1}" -f $sceneNumber, $scene.ToLowerInvariant())
            $clip = ""
            if ($Record) {
                $directory = Split-Path -Parent $resolvedOutputPath
                $stem = [IO.Path]::GetFileNameWithoutExtension($resolvedOutputPath)
                $clip = Join-Path $directory ("{0}.scene_{1:d2}_{2}.mp4" -f $stem, $sceneNumber, $scene.ToLowerInvariant())
                $clips += $clip
            }
            Write-Host "scene_cut_before=$scene reset=true"
            Invoke-Scene $scene $sceneOutput $clip
        }
        if ($Record) {
            $assemble = @("-p", $reelScript, "--output", $resolvedOutputPath) + $clips
            Write-Host "assemble_command=$isaacLabBat $($assemble -join ' ')"
            & $isaacLabBat @assemble
            if ($LASTEXITCODE -ne 0) { throw "FULL_REEL assembly failed with exit code $LASTEXITCODE" }
        }
    }
}
finally {
    Pop-Location
}
