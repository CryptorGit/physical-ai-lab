$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$eval = Join-Path $PSScriptRoot "evaluate_w1a3.py"
$out = Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints"
$env:PYTHONPATH = @(
    (Join-Path $repo "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    (Join-Path $repo "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    (Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src")
) -join ";"

$checkpoints = @("initial", "1", "10", "20", "40", "60", "80", "100", "120", "140", "160")
foreach ($iteration in $checkpoints) {
    & $python $eval --mode timeline --checkpoint (Join-Path $out "model_$iteration.pt") --tag "timeline_$iteration" --headless
    if ($LASTEXITCODE -ne 0) { throw "timeline $iteration failed" }
}

$w1a = Join-Path $repo "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk/checkpoints/model_120.pt"
foreach ($item in @(
    @("w1a", $w1a),
    @("w1a2_120", (Join-Path $out "model_120.pt")),
    @("w1a2_140", (Join-Path $out "model_140.pt")),
    @("w1a2_160", (Join-Path $out "model_160.pt"))
)) {
    & $python $eval --mode boundary --checkpoint $item[1] --tag "boundary_$($item[0])" --headless
    if ($LASTEXITCODE -ne 0) { throw "boundary $($item[0]) failed" }
}

foreach ($lambda in @(0.00, 0.25, 0.50, 0.75, 1.00)) {
    $tag = "interp_" + $lambda.ToString("0.00").Replace(".", "p")
    & $python $eval --mode interpolation --lambda-value $lambda --tag $tag --headless
    if ($LASTEXITCODE -ne 0) { throw "interpolation $lambda failed" }
}
