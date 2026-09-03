# Public official weight: faithful simulation replay

This directory contains a headless, fixed-camera replay of the public
Open Duck Mini policy `BEST_WALK_ONNX_2.onnx` at a constant command
`[vx, vy, wz] = [0.15, 0, 0]`.

The replay deliberately imports `MjInfer` from a clean detached checkout of
the public Open Duck Playground commit
`1842c8f46a67cb5d6b74e5aaf08c8702cde6e74f` (2025-04-02).  Its control loop
preserves the upstream semantics:

- 14 actions and 101 observations;
- `sim_dt=0.002`, control decimation 10;
- target `default_actuator + action * 0.25`;
- upstream 5.24 rad/s motor-target speed limiter.

No local robot calibration, servo offsets, action masking, target guard,
posture correction, or command profile is applied.  The fixed camera is for
recording only; it does not affect physics or observation.

## Result

`official_best_walk_onnx_2_forward_015_fixed_camera.mp4` is a 6-second,
150-frame replay.  `manifest.json` records the exact source paths, public
commit, command, simulation contract, SHA-256 hashes, first policy action,
and rollout measurements.

Observed in the manifest:

- forward displacement: `0.5539323044 m`;
- final root height: `0.1600976318 m` (initial `0.15 m`);
- minimum body-up/world-up cosine: `0.9964885287`;
- no fall event in the complete 6-second recorded rollout.

The PNGs in `inspection_frames/` are fixed-video frames near 0.2, 3.0 and
5.8 seconds and were visually inspected for a standing, alternating gait.

## Reproduce

From the repository root in PowerShell:

```powershell
$py = '/home/user/openduck_training_20260729/.venv/bin/python'
$root = '/mnt/c/Users/user/workspace/physical-ai-lab'
wsl.exe -- env MUJOCO_GL=egl $py `
  "$root/experiments/mujoco/exp_004_openduckmini_safe_gait_experts/scripts/render_official_best_weight_reference_20250402.py" `
  --playground-root "$root/.openduck_playground_reference_20250402" `
  --policy "$root/.openduck_hardware_source_review/BEST_WALK_ONNX_2.onnx" `
  --output-dir "$root/experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/official_best_weight_reference_20250402" `
  --seconds 6 --fps 25 --lookat '0.30,0,0.25' --distance 1.50
```

## Scope and correction

This verifies the published model in the public simulation contract.  It does
not establish real-robot safety or use a real servo calibration.

Earlier exp_004 videos under
`artifacts/official_best_walk_onnx_2_fixed_camera_20260812/` used the local
exp_003 calibrated evaluator instead of this public contract.  They are
retained for audit but are **not valid evidence of the public weight's
performance**.
