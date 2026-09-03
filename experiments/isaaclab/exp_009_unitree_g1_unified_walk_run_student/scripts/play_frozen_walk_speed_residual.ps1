param(
    [Parameter(Mandatory = $true)][double]$StartSpeed,
    [Parameter(Mandatory = $true)][double]$TargetSpeed,
    [Parameter(Mandatory = $true)][string]$ResidualCheckpoint
)
$ErrorActionPreference = "Stop"
& "$HOME\workspace\IsaacLab\isaaclab.bat" -p `
  (Join-Path $PSScriptRoot "play_frozen_walk_speed_residual.py") `
  --start-speed $StartSpeed `
  --target-speed $TargetSpeed `
  --residual-checkpoint (Resolve-Path $ResidualCheckpoint).Path
