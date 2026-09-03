param(
    [ValidateSet("All", "Standard", "Linearity")]
    [string]$Mode = "All"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaacLab = "C:\Users\user\workspace\IsaacLab\isaaclab.bat"
$script = Join-Path $PSScriptRoot "counterfactual_stage12.py"
$raw = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage12_tangential_slip_reward_directionality\raw"
$speeds = @("0p2", "0p4", "0p6", "1p2", "2p0")
$modes = if ($Mode -eq "All") { @("standard", "linearity") } else { @($Mode.ToLowerInvariant()) }

Push-Location $repo
try {
    foreach ($currentMode in $modes) {
        for ($index = 0; $index -lt $speeds.Count; $index++) {
            $output = Join-Path $raw "counterfactual_$($speeds[$index])_$currentMode.pt"
            if (Test-Path -LiteralPath $output) {
                Write-Host "SKIP $output"
                continue
            }
            & $isaacLab -p $script --speed-index $index --mode $currentMode --device cuda:0 --headless
            if ($LASTEXITCODE -ne 0) {
                throw "Counterfactual replay failed: speed-index=$index mode=$currentMode"
            }
        }
    }
}
finally {
    Pop-Location
}
