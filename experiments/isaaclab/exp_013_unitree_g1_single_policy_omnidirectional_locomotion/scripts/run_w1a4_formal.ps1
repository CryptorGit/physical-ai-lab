$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$env:PYTHONPATH = @(
    (Join-Path $repo "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    (Join-Path $repo "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    (Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src")
) -join ";"
$out = Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation"
$selected = (Get-Content (Join-Path $out "selected_checkpoint.json") | ConvertFrom-Json).path
$w1a = Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk/checkpoints/model_120.pt"
$w1a2 = Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints"

$jobs = @(
    @("formal", $selected, "selected"),
    @("formal", $w1a, "comparison_w1a"),
    @("formal", (Join-Path $w1a2 "model_80.pt"), "comparison_i80"),
    @("formal", (Join-Path $w1a2 "model_160.pt"), "comparison_i160"),
    @("continuous", $selected, "continuous_selected"),
    @("run", $selected, "run_selected")
)
foreach ($job in $jobs) {
    & $python (Join-Path $PSScriptRoot "evaluate_w1a4.py") --mode $job[0] --checkpoint $job[1] --tag $job[2] --headless
    if ($LASTEXITCODE) { throw "W1A4 formal evaluation failed: $($job -join ', ')" }
}
