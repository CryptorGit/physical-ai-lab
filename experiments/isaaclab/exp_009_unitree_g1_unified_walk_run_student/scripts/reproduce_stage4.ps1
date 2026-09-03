$ErrorActionPreference = "Stop"
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$base = $PSScriptRoot
& $python "$base\audit_stage4_residual_feasibility.py"
& $python "$base\finalize_stage4.py"
