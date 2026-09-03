$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$collector = Join-Path $PSScriptRoot "collect_stage2_dynamic_sensitivity.py"
$merge = Join-Path $PSScriptRoot "merge_stage2_counterfactual_replays.py"
foreach ($item in @(@("delta001", "0.01"), @("delta004", "0.04"))) {
    $suffix, $delta = $item
    foreach ($sign in @("plus", "minus")) {
        $proc = Start-Process -FilePath $python -ArgumentList @(
            $collector, "--regime", "walk_steady", "--cycle-index", "0", "--sign", $sign,
            "--diagnostic-delta", $delta, "--output-suffix", $suffix, "--viz", "none"
        ) -WorkingDirectory $repo -RedirectStandardOutput (Join-Path $repo "stage2_${suffix}_${sign}.log") `
            -RedirectStandardError (Join-Path $repo "stage2_${suffix}_${sign}.err.log") -WindowStyle Hidden -Wait -PassThru
        if ($proc.ExitCode -ne 0) { throw "locality collector failed: $suffix $sign" }
    }
    & $python $merge --suffix $suffix --delta $delta
    if ($LASTEXITCODE -ne 0) { throw "locality merge failed: $suffix" }
}
