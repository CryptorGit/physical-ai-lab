$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$collector = Join-Path $PSScriptRoot "collect_stage2_dynamic_sensitivity.py"
$merge = Join-Path $PSScriptRoot "merge_stage2_counterfactual_replays.py"
$combine = Join-Path $PSScriptRoot "combine_stage2_matched_samples.py"
$progress = Join-Path $repo "stage2_shards.progress.log"

foreach ($regime in @("walk_steady", "run_steady", "walk_to_run")) {
    foreach ($cycle in 0..2) {
        $suffix = "primary_${regime}_${cycle}"
        foreach ($sign in @("plus", "minus")) {
            "START $suffix $sign $(Get-Date -Format o)" | Add-Content $progress
            $proc = Start-Process -FilePath $python -ArgumentList @(
                $collector, "--regime", $regime, "--cycle-index", "$cycle", "--sign", $sign,
                "--output-suffix", $suffix, "--viz", "none"
            ) -WorkingDirectory $repo -RedirectStandardOutput (Join-Path $repo "stage2_${suffix}_${sign}.log") `
                -RedirectStandardError (Join-Path $repo "stage2_${suffix}_${sign}.err.log") -WindowStyle Hidden -Wait -PassThru
            if ($proc.ExitCode -ne 0) { throw "collector failed: $suffix $sign" }
            "DONE $suffix $sign $(Get-Date -Format o)" | Add-Content $progress
        }
        & $python $merge --suffix $suffix --delta 0.02
        if ($LASTEXITCODE -ne 0) { throw "merge failed: $suffix" }
        "MERGED $suffix $(Get-Date -Format o)" | Add-Content $progress
    }
}
& $python $combine
if ($LASTEXITCODE -ne 0) { throw "combine failed" }
"COMPLETE $(Get-Date -Format o)" | Add-Content $progress
