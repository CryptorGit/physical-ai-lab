$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$collector = Join-Path $PSScriptRoot "collect_stage2_dynamic_sensitivity.py"
$merge = Join-Path $PSScriptRoot "merge_stage2_counterfactual_replays.py"
$combine = Join-Path $PSScriptRoot "combine_stage2_matched_samples.py"
foreach ($cycle in 3..4) {
    $suffix = "primary_walk_steady_${cycle}"
    foreach ($sign in @("plus", "minus")) {
        $proc = Start-Process -FilePath $python -ArgumentList @(
            $collector, "--regime", "walk_steady", "--cycle-index", "$cycle", "--sign", $sign,
            "--output-suffix", $suffix, "--viz", "none"
        ) -WorkingDirectory $repo -RedirectStandardOutput (Join-Path $repo "stage2_${suffix}_${sign}.log") `
            -RedirectStandardError (Join-Path $repo "stage2_${suffix}_${sign}.err.log") -WindowStyle Hidden -Wait -PassThru
        if ($proc.ExitCode -ne 0) { throw "collector failed: $suffix $sign" }
    }
    & $python $merge --suffix $suffix --delta 0.02
    if ($LASTEXITCODE -ne 0) { throw "merge failed: $suffix" }
}
& $python $combine
if ($LASTEXITCODE -ne 0) { throw "combine failed" }
