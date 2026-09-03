$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_stage2h_branch.ps1"

& $Runner -Branch R1_A025 -Mode completion -Horizon 1 -Coefficient 0.025 -Iterations 1 -SeedOffset 100
& $Runner -Branch R1_A050 -Mode completion -Horizon 1 -Coefficient 0.050 -Iterations 1 -SeedOffset 200
& $Runner -Branch R1_A100 -Mode completion -Horizon 1 -Coefficient 0.100 -Iterations 1 -SeedOffset 300
& $Runner -Branch RB_A025 -Mode background -Horizon 4 -Coefficient 0.025 -Iterations 1 -SeedOffset 400
& $Runner -Branch R2_A025 -Mode completion -Horizon 2 -Coefficient 0.025 -Iterations 1 -SeedOffset 500
& $Runner -Branch R4_A025 -Mode completion -Horizon 4 -Coefficient 0.025 -Iterations 1 -SeedOffset 600
