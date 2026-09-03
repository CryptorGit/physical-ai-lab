param(
    [string]$RunDirectory = "/home/user/openduck_training_runs/calibrated_discovery_v18_300m"
)

$gpu = nvidia-smi `
    --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu `
    --format=csv,noheader

$wslStatus = wsl.exe -e bash -lc @"
echo PROCESS
ps -eo pid,etime,%cpu,%mem,rss,cmd |
  grep '[p]layground/open_duck_mini_v2/runner.py' || true
echo METRICS
grep -E 'STEP:|reward:|Saving checkpoint' '$RunDirectory/train.log' |
  tail -30 || true
echo LATEST_ONNX
find '$RunDirectory' -maxdepth 1 -type f -name '*.onnx' \
  -printf '%T@ %s %p\n' | sort -n | tail -3
"@

Write-Output "GPU"
Write-Output $gpu
Write-Output $wslStatus
