param(
    [Parameter(Mandatory = $true)][double]$StartSpeed,
    [Parameter(Mandatory = $true)][double]$TargetSpeed,
    [Parameter(Mandatory = $true)][string]$StudentCheckpoint
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
& "$HOME\workspace\IsaacLab\isaaclab.bat" -p `
  (Join-Path $PSScriptRoot "play_unified_student.py") `
  --start-speed $StartSpeed `
  --target-speed $TargetSpeed `
  --student-checkpoint (Resolve-Path $StudentCheckpoint).Path
