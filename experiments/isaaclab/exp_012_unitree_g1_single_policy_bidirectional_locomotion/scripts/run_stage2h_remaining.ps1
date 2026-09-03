param(
    [int]$WaitForPid = 0
)

$ErrorActionPreference = "Stop"
while ($WaitForPid -gt 0 -and (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 10
}

$Runner = Join-Path $PSScriptRoot "run_stage2h_branch.ps1"

# Coefficient screen at the shortest horizon.
& $Runner -Branch R1_A050 -Mode completion -Horizon 1 -Coefficient 0.050 -Iterations 1 -SeedOffset 150 -ForceRecompute
& $Runner -Branch R1_A100 -Mode completion -Horizon 1 -Coefficient 0.100 -Iterations 1 -SeedOffset 175 -ForceRecompute

# Horizon screen at the smallest coefficient.
& $Runner -Branch R2_A025 -Mode completion -Horizon 2 -Coefficient 0.025 -Iterations 1 -SeedOffset 200 -ForceRecompute
& $Runner -Branch R4_A025 -Mode completion -Horizon 4 -Coefficient 0.025 -Iterations 1 -SeedOffset 300 -ForceRecompute

# Completion-specific effect control with a separately collected branch.
& $Runner -Branch RB_A025 -Mode background -Horizon 4 -Coefficient 0.025 -Iterations 1 -SeedOffset 400 -ForceRecompute
