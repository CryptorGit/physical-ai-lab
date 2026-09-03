[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("All", "StandWalkStand", "WalkToRun26", "WalkToRun28", "StandWalkStandWalkRun26")]
    [string]$Scene = "All",
    [switch]$RecordVideo,
    [string]$OutputPath = ".\media\exp_008_closeout",
    [switch]$Headless,
    [ValidateSet("Tracking", "Fixed")]
    [string]$CameraMode = "Tracking",
    [bool]$ShowFloorGuides = $true,
    [ValidateSet("SIDE", "REAR_QUARTER", "FOLLOW_POSITION")]
    [string]$CameraPreset = "FOLLOW_POSITION"
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) { throw "Isaac Lab launcher missing: $launcher" }
$output = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputPath))
New-Item -ItemType Directory -Force -Path $output | Out-Null
$sceneNames = if ($Scene -eq "All") { @("StandWalkStand", "WalkToRun26", "WalkToRun28") } else { @($Scene) }
$fileNames = @{
    StandWalkStand = "scene1_stand_walk_stand.mp4"
    WalkToRun26 = "scene2_walk_to_run_2p6.mp4"
    WalkToRun28 = "scene3_walk_to_run_2p8.mp4"
    StandWalkStandWalkRun26 = "exp008_stop_walk_stop_walk_run.mp4"
}
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$repositoryRoot\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src;$repositoryRoot\experiments\isaaclab\exp_006_unitree_g1_command_skills\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    foreach ($name in $sceneNames) {
        Write-Host "NEW SCENE / RESET - NOT A LOCOMOTION TRANSITION"
        $telemetryPath = Join-Path $output "$($name)_telemetry.json"
        if (Test-Path -LiteralPath $telemetryPath) {
            Remove-Item -LiteralPath $telemetryPath -Force
        }
        $arguments = @(
            "-p", "$PSScriptRoot\play_exp008_closeout_showcase.py",
            "--scene", $name,
            "--seed", "20260831",
            "--camera-preset", $CameraPreset,
            "--camera-mode", $CameraMode,
            "--telemetry-output", $telemetryPath
        )
        if (-not $ShowFloorGuides) { $arguments += "--no-show-floor-guides" }
        if ($RecordVideo) {
            $arguments += @("--record", "--output-path", (Join-Path $output $fileNames[$name]))
        }
        if ($Headless) { $arguments += "--headless" }
        & $launcher @arguments
        if ($LASTEXITCODE -ne 0) { throw "Showcase scene failed: $name ($LASTEXITCODE)" }
        if (-not (Test-Path -LiteralPath $telemetryPath -PathType Leaf)) {
            throw "Showcase scene did not produce telemetry: $name"
        }
        if ($RecordVideo) {
            $sceneVideo = Join-Path $output $fileNames[$name]
            $normalizedVideo = Join-Path $output "$($fileNames[$name]).30fps.tmp.mp4"
            & ffmpeg -hide_banner -loglevel error -y -i $sceneVideo `
                -vf "fps=30" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p `
                -an $normalizedVideo
            if ($LASTEXITCODE -ne 0) { throw "30 fps normalization failed: $name ($LASTEXITCODE)" }
            Move-Item -LiteralPath $normalizedVideo -Destination $sceneVideo -Force
        }
    }
    if ($RecordVideo -and $Scene -eq "All") {
        python "$PSScriptRoot\assemble_exp008_closeout_showcase.py" `
            --output (Join-Path $output "exp008_g1_state_graph_closeout_showcase.mp4") `
            (Join-Path $output $fileNames.StandWalkStand) `
            (Join-Path $output $fileNames.WalkToRun26) `
            (Join-Path $output $fileNames.WalkToRun28)
        if ($LASTEXITCODE -ne 0) { throw "Combined showcase assembly failed: $LASTEXITCODE" }
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
