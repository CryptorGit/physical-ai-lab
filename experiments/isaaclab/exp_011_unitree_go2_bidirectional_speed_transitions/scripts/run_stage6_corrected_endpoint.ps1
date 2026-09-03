param(
  [switch]$PrepareOnly,
  [switch]$SkipFormal,
  [switch]$SkipGuiValidation
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$prepare = Join-Path $PSScriptRoot "prepare_stage6_protocol.py"
$evaluate = Join-Path $PSScriptRoot "evaluate_stage6_corrected.py"
$merge = Join-Path $PSScriptRoot "merge_stage6_partials.py"
$finalize = Join-Path $PSScriptRoot "finalize_stage6.py"
Push-Location $repo
try {
  & $isaac -p $prepare
  if ($PrepareOnly) { return }
  & $isaac -p $evaluate --mode contact-preflight --num-envs 4
  if (-not $SkipFormal) {
    & $isaac -p $evaluate --mode formal-steady --policy official_parent --num-envs 50
    & $isaac -p $evaluate --mode formal-steady --policy stage4_selected --num-envs 50
    & $isaac -p $evaluate --mode formal-transitions --policy official_parent --num-envs 50
    & $isaac -p $evaluate --mode formal-transitions --policy stage4_selected --num-envs 50
    & $isaac -p $evaluate --mode formal-low-speed --policy official_parent --num-envs 20
    & $isaac -p $evaluate --mode formal-low-speed --policy stage4_selected --num-envs 20
    & $isaac -p $merge
    $sequence = Get-Content (
      Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage6_corrected_endpoint_formal\formal_reduced_sequence.json"
    ) -Raw | ConvertFrom-Json
    if ($sequence.reason -eq "SEQUENCE_REQUIRED_AFTER_PARTIAL_MERGE") {
      & $isaac -p $evaluate --mode formal-reduced --policy stage4_selected --num-envs 50
      & $isaac -p $merge
    }
  }
  & $isaac -p $finalize
  if (-not $SkipGuiValidation) {
    $player = Join-Path $PSScriptRoot "play_exp011_go2_bidirectional.ps1"
    & $player -Mode Stand -Seed 20264901
  }
} finally {
  Pop-Location
}
