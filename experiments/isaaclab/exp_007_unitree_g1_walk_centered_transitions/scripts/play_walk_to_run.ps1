param([ValidateSet(2.4,2.6,2.8)][double]$RunSpeed=2.6)
$root=(Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Push-Location $root
try { python (Join-Path $PSScriptRoot "play_walk_to_run.py") --run-speed $RunSpeed }
finally { Pop-Location }
