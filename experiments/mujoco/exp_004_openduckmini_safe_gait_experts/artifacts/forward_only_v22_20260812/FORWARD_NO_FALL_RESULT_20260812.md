# V22 forward / no-fall result — 2026-08-12

## Revised scope verdict: PASS (simulation only)

The revised scope is one existing frozen V22 policy making forward progress
without falling; a straight world-frame path is explicitly not required.

- Raw run: `forward_only_v22_routed_20x30_v2.json`
  - SHA-256: `2cafce9eaf9ee9b176a741ff699fbc5bc14f2394df4368b788ec05a36c4178f7`
  - 20 independently reset episodes × 30 seconds.
  - Physical command `[+0.05, 0, 0]`; frozen actor observation command
    `[+0.10, 0, 0]`.
  - Minimum / mean robot-local forward velocity: `0.041575` / `0.042563 m/s`.
  - Falls: `0 / 20`; incomplete episodes: `0 / 20`.
  - Joint-qpos, applied-target, target-slew, and non-finite violations: `0`.
  - Head target peak: `0 rad`; maximum flight rate: `0`.
  - Minimum height: `0.182196 m`; minimum upright: `0.983237`.
- Revised-scope evidence verdict: `forward_only_v22_no_fall_verdict_v2.json`
  - SHA-256: `8b7ed1d16584a0d49e7f76e746bb9a850ec060991426eb8e7b32925c7ee2286a`
  - Result: `passed=true`, `failures=[]`.

## Important limitation retained verbatim

The raw run's original strict-gait assessment is `false` for all 20 episodes.
Those checks include strict slip/cadence/trajectory-quality requirements that
are outside the revised forward/no-fall scope.  This report does **not** claim
strict gait-quality acceptance, straightness, or real-robot validation.

Hardware actuation was not attempted; the package remains simulation-only and
the recorded hardware gate is `PROHIBITED`.

## Reproduction

From a Windows PowerShell session with the reviewed WSL environment:

```powershell
wsl.exe -e /home/user/openduck_training_20260729/.venv/bin/python `
  /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_004_openduckmini_safe_gait_experts/scripts/evaluate_v22_forward_only.py `
  --output /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/forward_only_v22_20260812/NEW_RAW_RESULT.json

wsl.exe -e /home/user/openduck_training_20260729/.venv/bin/python `
  /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_004_openduckmini_safe_gait_experts/scripts/validate_v22_forward_no_fall_gate.py `
  --evidence /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/forward_only_v22_20260812/NEW_RAW_RESULT.json `
  --output /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/forward_only_v22_20260812/NEW_REVISED_VERDICT.json
```

Both scripts refuse to overwrite an existing evidence file.
