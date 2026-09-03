$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$env:PYTHONPATH = @(
    (Join-Path $repo "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")
    (Join-Path $repo "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src")
    (Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src")
) -join ";"
$evaluator = Join-Path $PSScriptRoot "evaluate_w1a.py"
$checkpointDir = Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk/checkpoints"
foreach ($iteration in @("initial", "1", "10", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200")) {
    & $python $evaluator --suite selection --checkpoint (Join-Path $checkpointDir "model_$iteration.pt") --tag "iter_$iteration" --headless
    if ($LASTEXITCODE -ne 0) { throw "Selection evaluation failed at $iteration" }
}
