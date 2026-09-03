$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$env:PYTHONPATH = @(
    (Join-Path $repo "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")
    (Join-Path $repo "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src")
    (Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src")
) -join ";"
$evaluator = Join-Path $PSScriptRoot "evaluate_w1a.py"
$out = Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk"
$selected = Join-Path $out "checkpoints/model_120.pt"
$parent = Join-Path $repo "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
$jobs = @(
    [pscustomobject]@{ Suite = "formal"; Checkpoint = $selected; Tag = "selected" }
    [pscustomobject]@{ Suite = "formal"; Checkpoint = $parent; Tag = "parent" }
    [pscustomobject]@{ Suite = "envelope"; Checkpoint = $selected; Tag = "selected" }
    [pscustomobject]@{ Suite = "continuous"; Checkpoint = $selected; Tag = "selected" }
    [pscustomobject]@{ Suite = "run"; Checkpoint = $selected; Tag = "selected" }
)
foreach ($job in $jobs) {
    & $python $evaluator --suite $job.Suite --checkpoint $job.Checkpoint --tag $job.Tag --headless
    if ($LASTEXITCODE -ne 0) { throw "Formal evaluation failed: $($job.Suite) $($job.Tag)" }
}
