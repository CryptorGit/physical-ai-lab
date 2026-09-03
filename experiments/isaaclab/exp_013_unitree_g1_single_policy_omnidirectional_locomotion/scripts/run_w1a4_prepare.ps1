$ErrorActionPreference="Stop"
$repo=(Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$python="C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$env:PYTHONPATH=@((Join-Path $repo "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),(Join-Path $repo "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),(Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"))-join";"
&$python (Join-Path $PSScriptRoot "collect_w1a4_anchors.py") --headless
if($LASTEXITCODE){throw "anchor collection failed"}
$parent=Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
&$python (Join-Path $PSScriptRoot "evaluate_w1a4.py") --mode parent06 --checkpoint $parent --tag parent80_0p6 --headless
if($LASTEXITCODE){throw "parent evaluation failed"}
