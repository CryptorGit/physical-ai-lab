param(
    [string]$RunDirectory = "/home/user/openduck_training_runs/calibrated_hybrid_yaw_cost_v22_300m",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$experiment = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = (Resolve-Path (Join-Path $experiment "..\..\..")).Path
$python = Join-Path $workspace "experiments\mujoco\.venv\Scripts\python.exe"
$evaluator = Join-Path $experiment "evaluate_official_policy.py"
$scene = Join-Path $workspace ".openduck_playground_source_review\playground\open_duck_mini_v2\xmls\scene_flat_terrain_backlash_calibrated.xml"
$outputDirectory = Join-Path $experiment "artifacts\mjx\auto_evaluations"
$statePath = Join-Path $outputDirectory "watcher_state.json"
$watcherLog = Join-Path $outputDirectory "watcher.log"
$passedPath = Join-Path $outputDirectory "PASSED.json"

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

function Write-WatcherLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $watcherLog -Value "[$timestamp] $Message"
}

function Get-LatestOnnx {
    $command = "find '$RunDirectory' -maxdepth 1 -type f -name '*.onnx' -printf '%T@ %p\n' | sort -n | tail -1"
    $line = (wsl.exe -e bash -lc $command | Select-Object -Last 1)
    if (-not $line) {
        return $null
    }
    return ($line -split " ", 2)[1]
}

function Test-TrainingActive {
    $processLine = wsl.exe -e bash -lc "pgrep -af 'runner.py.*flat_terrain_backlash_calibrated' | head -1"
    return [bool]$processLine
}

$lastEvaluated = $null
if (Test-Path -LiteralPath $statePath) {
    try {
        $lastEvaluated = (Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json).last_evaluated
    }
    catch {
        Write-WatcherLog "Ignoring unreadable watcher state: $($_.Exception.Message)"
    }
}

Write-WatcherLog "Watcher started for $RunDirectory"

while ($true) {
    try {
        $latestOnnx = Get-LatestOnnx
        if ($latestOnnx -and $latestOnnx -ne $lastEvaluated) {
            $onnxName = [IO.Path]::GetFileName($latestOnnx)
            $stem = [IO.Path]::GetFileNameWithoutExtension($onnxName)
            $localOnnx = Join-Path $outputDirectory $onnxName
            $resultPath = Join-Path $outputDirectory "$stem.acceptance.json"
            $stdoutPath = Join-Path $outputDirectory "$stem.evaluator.log"

            $localOnnxWsl = $localOnnx.Replace("\", "/")
            if ($localOnnxWsl -match "^([A-Za-z]):(.*)$") {
                $drive = $Matches[1].ToLower()
                $localOnnxWsl = "/mnt/$drive$($Matches[2])"
            }
            wsl.exe -e cp $latestOnnx $localOnnxWsl
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to copy $latestOnnx"
            }

            Write-WatcherLog "Evaluating $latestOnnx"
            $arguments = @(
                $evaluator,
                "--seconds", "30",
                "--episodes", "20",
                "--initial-joint-noise", "0.03",
                "--initial-base-speed", "0.10",
                "--scene", $scene,
                "--policy", $localOnnx,
                "--output", $resultPath
            )
            & $python @arguments *> $stdoutPath
            if ($LASTEXITCODE -ne 0) {
                throw "Evaluator failed for $latestOnnx"
            }

            $result = Get-Content -Raw -LiteralPath $resultPath | ConvertFrom-Json
            $lastEvaluated = $latestOnnx
            @{
                last_evaluated = $lastEvaluated
                evaluated_at = (Get-Date).ToString("o")
                passed = [bool]$result.acceptance.passed
                result = $resultPath
            } | ConvertTo-Json | Set-Content -LiteralPath $statePath

            Write-WatcherLog "Evaluation complete: passed=$($result.acceptance.passed)"
            if ($result.acceptance.passed) {
                @{
                    policy = $localOnnx
                    result = $resultPath
                    passed_at = (Get-Date).ToString("o")
                } | ConvertTo-Json | Set-Content -LiteralPath $passedPath
                Write-WatcherLog "Acceptance checkpoint saved to $passedPath"
                break
            }
        }

        if (-not (Test-TrainingActive)) {
            Write-WatcherLog "Training ended before an acceptance checkpoint was found"
            break
        }
    }
    catch {
        Write-WatcherLog "ERROR: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds $PollSeconds
}

Write-WatcherLog "Watcher stopped"
