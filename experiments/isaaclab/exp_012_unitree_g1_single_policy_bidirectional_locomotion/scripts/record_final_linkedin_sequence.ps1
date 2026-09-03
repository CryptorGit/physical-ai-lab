$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\..\..\.."
$env:PYTHONPATH = "$PWD\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src;$PWD\experiments\isaaclab\exp_005_unitree_g1_flat_run\src;$PWD"
$closure = "$PWD\results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\closure"
$selection = Get-Content "$closure\video_seed_selection.json" | ConvertFrom-Json
$seed = [int]$selection.selected_seed
& "C:\isaacsim\python.bat" `
  ".\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\evaluate_closure_video_seeds.py" `
  --record --selected-seed $seed `
  --raw-video "$closure\raw\exp_012_closure_sequence_raw_v6_floor.mp4" `
  --headless --device cuda:0
