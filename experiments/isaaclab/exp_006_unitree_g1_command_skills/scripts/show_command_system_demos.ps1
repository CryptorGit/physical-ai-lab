[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("RUN_TURN_RUN", "STAND_CROUCH_STAND", "UNSUPPORTED_RUN_TO_CROUCH", "UNSUPPORTED_STEP_OVER", "UNSUPPORTED_LAND")]
    [string]$Demo,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$player = Join-Path $PSScriptRoot "play_command_system.ps1"
$demos = @("RUN_TURN_RUN", "STAND_CROUCH_STAND", "UNSUPPORTED_RUN_TO_CROUCH", "UNSUPPORTED_STEP_OVER", "UNSUPPORTED_LAND")
if (-not (Test-Path -LiteralPath $player)) { throw "Command-system player is missing: $player" }
Write-Host "Available command_system_v1 demos:"
for ($index = 0; $index -lt $demos.Count; $index++) { Write-Host "  $($index + 1). $($demos[$index])" }

do {
    $selected = $Demo
    if (-not $selected) {
        $answer = Read-Host "Select demo number/name, or Q to quit"
        if ($answer -match '^[Qq]$') { break }
        if ($answer -match '^\d+$' -and [int]$answer -ge 1 -and [int]$answer -le $demos.Count) { $selected = $demos[[int]$answer - 1] }
        elseif ($demos -contains $answer) { $selected = $answer }
        else { Write-Warning "Unknown demo: $answer"; continue }
    }
    Write-Host "Executing: & `"$player`" -Demo $selected$(if($ValidateOnly){' -ValidateOnly'})"
    & $player -Demo $selected -ValidateOnly:$ValidateOnly
    if ($LASTEXITCODE -ne 0) { throw "Demo failed: $selected" }
    if ($Demo -or $ValidateOnly) { break }
    $selected = $null
} while ($true)
