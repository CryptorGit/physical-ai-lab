$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
Set-Location $repo
& $python "$PSScriptRoot/audit_protection.py" start
& $python "$PSScriptRoot/bootstrap_phase0.py"
& $python -m pytest "$(Split-Path $PSScriptRoot)/tests" -q
& "C:\Users\user\workspace\IsaacLab\isaaclab.bat" -p "$PSScriptRoot/collect_phase1.py" --headless --device cuda:0
& $python "$PSScriptRoot/audit_dataset.py"
& $python "$PSScriptRoot/train_static.py" --model S0
