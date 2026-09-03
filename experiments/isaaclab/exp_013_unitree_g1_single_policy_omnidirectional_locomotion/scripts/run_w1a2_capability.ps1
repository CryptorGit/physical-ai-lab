$ErrorActionPreference="Stop"
$repo=(Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$python="C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$env:PYTHONPATH=@((Join-Path $repo "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),(Join-Path $repo "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),(Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"))-join";"
$eval=Join-Path $PSScriptRoot "evaluate_w1a2.py";$cp=Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints"
foreach($i in @("initial","1","10","20","40","60","80","100","120","140","160")){& $python $eval --mode capability --checkpoint (Join-Path $cp "model_$i.pt") --tag "capability_$i";if($LASTEXITCODE-ne 0){throw "capability $i failed"}}
