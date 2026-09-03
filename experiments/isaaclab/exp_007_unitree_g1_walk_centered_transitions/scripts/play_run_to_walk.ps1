[CmdletBinding(PositionalBinding=$false)]
param(
  [Parameter(Mandatory=$true)][ValidateSet(2.6,2.8)][double]$RunSpeed,
  [string]$TransitionCheckpoint = "",
  [int]$Seed = 20261301
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
Push-Location $root
try {
  $arguments = @("-p", (Join-Path $PSScriptRoot "play_run_to_walk.py"), "--run-speed", $RunSpeed, "--seed", $Seed)
  if ($TransitionCheckpoint) { $arguments += @("--transition-checkpoint", $TransitionCheckpoint) }
  & $isaac @arguments
  if ($LASTEXITCODE -ne 0) { throw "RUN_TO_WALK GUI failed" }
}
finally {
  Pop-Location
}
