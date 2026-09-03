[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)] [string]$SweepRoot,
    [Parameter(Mandatory = $true)] [string]$RunDirectory
)
$ErrorActionPreference = "Stop"
$latestGate = $null
$root = (Resolve-Path -LiteralPath $SweepRoot).Path
$runDir = (Resolve-Path -LiteralPath $RunDirectory).Path
$gates = @(Get-ChildItem -LiteralPath $root -Recurse -Filter gate.json -File)
Write-Host "Run name: $(Split-Path $runDir -Leaf)"
Write-Host "Sweep root: $root"
Write-Host "Gate count: $($gates.Count)"
if ($gates.Count -eq 0) { throw "No gate.json exists for this run; refusing to display a stale result" }
$latestGate = $gates | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
$gate = Get-Content -LiteralPath $latestGate.FullName -Raw | ConvertFrom-Json
$model = Split-Path $latestGate.DirectoryName -Leaf
$checkpoint = Join-Path $runDir "$model.pt"
Write-Host "Gate path: $($latestGate.FullName)"
Write-Host "Checkpoint ancestry: $checkpoint"
$gate | ConvertTo-Json -Depth 12
