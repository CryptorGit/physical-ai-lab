[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$exp = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repo = (Resolve-Path (Join-Path $exp "..\..\..")).Path
$exp005 = Join-Path $repo "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$old = $env:PYTHONPATH
$env:PYTHONPATH = "$($exp)\src;$exp005" + $(if ($old) { ";$old" } else { "" })
try { & C:\isaacsim\python.bat (Join-Path $PSScriptRoot "audit_parent.py"); if ($LASTEXITCODE) { exit $LASTEXITCODE } }
finally { $env:PYTHONPATH = $old }
