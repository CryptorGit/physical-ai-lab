param(
  [ValidateSet(2.4,2.6,2.8)][double]$RunSpeed=2.6,
  [string]$TransitionCheckpoint=""
)
$root=(Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Push-Location $root
try {
  $arguments=@("--run-speed",$RunSpeed)
  if ($TransitionCheckpoint) { $arguments+=@("--transition-checkpoint",$TransitionCheckpoint) }
  python (Join-Path $PSScriptRoot "play_walk_to_run_152d.py") @arguments
}
finally { Pop-Location }
