$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\..\..\.."
$env:PYTHONPATH = "$PWD\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src;$PWD\experiments\isaaclab\exp_005_unitree_g1_flat_run\src;$PWD"
$selection = Get-Content "$PWD\results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\closure\video_seed_selection.json" | ConvertFrom-Json
& "C:\isaacsim\python.bat" `
  ".\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\evaluate_closure_video_seeds.py" `
  --record --selected-seed ([int]$selection.selected_seed) `
  --raw-video "$PWD\results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\closure\raw\exp_012_closure_sequence_raw_v6_floor.mp4" `
  --device cuda:0
