$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\..\..\..\.."
$env:PYTHONPATH = "$PWD\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src;$PWD\experiments\isaaclab\exp_005_unitree_g1_flat_run\src;$PWD"
$checkpoint = "$PWD\results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\stage2q_final_sequence_integration\raw\dagger_round_2_student.pt"
& "C:\isaacsim\python.bat" `
  ".\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\evaluate_stage2q_sequence.py" `
  --mode final --checkpoint $checkpoint --gui --viz native --device cuda:0
