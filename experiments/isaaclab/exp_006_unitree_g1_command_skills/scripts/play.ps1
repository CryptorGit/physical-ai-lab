param(
    [Parameter(Mandatory = $true)] [string]$Checkpoint,
    [ValidateSet("run", "turn", "stop", "sequence")] [string]$Skill = "sequence",
    [ValidateRange(1, 4096)] [int]$NumEnvs = 1,
    [ValidateSet("kit", "none")] [string]$Visualizer = "kit",
    [switch]$Video,
    [ValidateRange(1, 1000000)] [int]$VideoLength = 600,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$ExtraArgs
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$flatRunSrc = Join-Path $repositoryRoot "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$labels = @{ run = "Run"; turn = "Turn"; stop = "Stop"; sequence = "Sequence" }
$argsList = @(
    "play", "--rl_library", "rsl_rl", "--task", "Isaac-Motion-Flat-G1-Command-$($labels[$Skill])-Play-v0",
    "--external_callback", "g1_command_skills.tasks.register_envs", "--checkpoint", (Resolve-Path -LiteralPath $Checkpoint).Path,
    "--num_envs", $NumEnvs, "--viz", $Visualizer
)
if ($Video) { $argsList += @("--video", "--video_length", $VideoLength) }
$argsList += $ExtraArgs
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$srcRoot;$flatRunSrc" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Push-Location $repositoryRoot
try { & $isaacLabBat @argsList; if ($LASTEXITCODE -ne 0) { throw "Playback failed: $LASTEXITCODE" } }
finally { Pop-Location; $env:PYTHONPATH = $previousPythonPath }
