param(
  [ValidateSet("Stand","Steady","SteadyState","Transition","ReducedSequence","AnchorSequence","FullSequence","Showcase")][string]$Mode = "AnchorSequence",
  [ValidateSet("OpenLoop","AlwaysOn","PhaseGated")][string]$HeadingController = "PhaseGated",
  [double]$TargetSpeed = 1.2,
  [double]$SourceSpeed = 0.0,
  [double]$TargetYawRate = 0.0,
  [double]$RampDuration = 1.5,
  [int]$Seed = 20260901,
  [switch]$RecordVideo,
  [string]$OutputPath = "",
  [string]$Checkpoint = ""
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
if (-not $Checkpoint) {
  $selection = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage11_tangential_slip_reduction\selected_checkpoint.json"
  if (-not (Test-Path $selection)) {
    $selection = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage7_low_speed_gait_stabilization\selected_checkpoint.json"
  }
  if (-not (Test-Path $selection)) {
    $selection = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage4_resumed_optimizer_training\selected_checkpoint.json"
  }
  if (Test-Path $selection) {
    $Checkpoint = (Get-Content $selection -Raw | ConvertFrom-Json).checkpoint
  }
}
if (-not $Checkpoint) {
  $Checkpoint = Get-ChildItem -Path (Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\.pretrained_checkpoints\rsl_rl\Isaac-Velocity-Flat-Unitree-Go2-v0") -Filter checkpoint.pt -Recurse | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $Checkpoint) { throw "Go2 checkpoint unavailable." }
$script = Join-Path $PSScriptRoot "play_exp011_go2_bidirectional.py"
$arguments = @("-p", $script, "--mode", $Mode, "--heading-controller", $HeadingController, "--checkpoint", $Checkpoint, "--target-speed", $TargetSpeed, "--source-speed", $SourceSpeed, "--target-yaw-rate", $TargetYawRate, "--ramp-duration", $RampDuration, "--seed", $Seed)
if ($RecordVideo) { $arguments += @("--record-video", "--output-path", $OutputPath) }
Push-Location $repo
try { & $isaac @arguments } finally { Pop-Location }
