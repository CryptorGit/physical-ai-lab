param(
  [ValidateSet("Steady","Transition","All")][string]$Family = "All",
  [string]$Device = "cuda:0"
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$script = Join-Path $PSScriptRoot "collect_stage12_rollout.py"
$results = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage12_tangential_slip_reward_directionality\raw"
$steady = @("0p0","0p2","0p3","0p4","0p5","0p6","0p8","1p0","1p2","1p5","2p0")
$pairs = @(
  @("0p0","0p2"), @("0p0","0p4"), @("0p0","0p6"),
  @("0p6","0p4"), @("0p6","0p2"), @("0p6","0p0"),
  @("0p0","1p2"), @("1p2","2p0"), @("2p0","1p2"), @("1p2","0p0")
)
Push-Location $repo
try {
  if ($Family -in @("Steady","All")) {
    for ($index = 0; $index -lt $steady.Count; $index++) {
      $path = Join-Path $results "steady_$($steady[$index])_$($steady[$index]).pt"
      if (-not (Test-Path -LiteralPath $path)) {
        & $isaac -p $script --family steady --condition-index $index --device $Device --headless
        if ($LASTEXITCODE -ne 0) { throw "steady condition $index failed: $LASTEXITCODE" }
      }
    }
  }
  if ($Family -in @("Transition","All")) {
    for ($index = 0; $index -lt $pairs.Count; $index++) {
      $path = Join-Path $results "transition_$($pairs[$index][0])_$($pairs[$index][1]).pt"
      if (-not (Test-Path -LiteralPath $path)) {
        & $isaac -p $script --family transition --condition-index $index --device $Device --headless
        if ($LASTEXITCODE -ne 0) { throw "transition condition $index failed: $LASTEXITCODE" }
      }
    }
  }
} finally {
  Pop-Location
}
