# exp_013 — Unitree G1 single-policy omnidirectional locomotion

```text
STATUS: ACTIVE
STAGE: 0 — frozen-parent directional baseline
```

The final contract is one checkpoint, one actor, and one Gaussian policy head
conditioned only on body-frame `(vx_cmd, vy_cmd, yaw_rate_cmd, gait_cmd)`.

Stage 0 is evaluation-only. It must not perform PPO, supervised learning,
DAgger, checkpoint updates, reward changes, or production-policy updates.

Run the complete gated baseline:

```powershell
.\scripts\run_stage0.ps1
```

Inspect one frozen actor interactively:

```powershell
.\scripts\play_directional_baseline.ps1
```
