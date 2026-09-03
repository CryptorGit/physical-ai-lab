[CmdletBinding()]
param(
  [ValidateSet("Stand","Walk","Run","Transition","IntegratedSequence","RunAcquisition","YawDiagnosis","YawCancellation")][string]$Mode = "IntegratedSequence",
  [double]$Speed = 0.6,
  [double]$YawRate = 0.0,
  [ValidateSet("Off","On")][string]$Controller = "Off",
  [int]$Seed = 20263021,
  [string]$Checkpoint = ""
)
$ErrorActionPreference = "Stop"
$exp = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repo = (Resolve-Path (Join-Path $exp "..\..\..")).Path
if (-not $Checkpoint) {
  $phaseASelection = Join-Path $repo "results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\stage2e_phase_a_run_acquisition_preflight\selected_phase_a_checkpoint.json"
  $selection = Join-Path $repo "results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\stage2_pilot1_retry1\selected_checkpoint.json"
  if (($Mode -eq "RunAcquisition") -and (Test-Path $phaseASelection)) {
    $relativeCheckpoint = (Get-Content $phaseASelection -Raw | ConvertFrom-Json).checkpoint
    $Checkpoint = Join-Path $repo $relativeCheckpoint
  } elseif ((Test-Path $selection) -and $Mode -notin @("YawDiagnosis", "YawCancellation")) {
    $relativeCheckpoint = (Get-Content $selection -Raw | ConvertFrom-Json).checkpoint
    $Checkpoint = Join-Path $repo $relativeCheckpoint
  } else {
    $Checkpoint = Join-Path $repo "logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
  }
}
$old = $env:PYTHONPATH
$env:PYTHONPATH = "$exp\src;" + (Join-Path $repo "experiments\isaaclab\exp_005_unitree_g1_flat_run\src") + $(if ($old) { ";$old" } else { "" })
try {
  & "$env:USERPROFILE\workspace\IsaacLab\isaaclab.bat" -p (Join-Path $PSScriptRoot "play_exp012.py") `
    --mode $Mode --speed $Speed --yaw-rate $YawRate --controller $Controller --seed $Seed --checkpoint $Checkpoint
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally { $env:PYTHONPATH = $old }
