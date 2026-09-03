# H5 PDCA cycle — command-conditioned SE(2) residual pilot

Status: **REJECTED — diagnostic simulation evidence only; no promotion, multi-seed evaluation, deployment package update, or hardware operation.**

## Hypothesis and authorization

The frozen V3 direct-normalized candidate penalized absolute lateral velocity, yaw rate, and heading whenever `vx > 0.02`. That conflicts with commanded lateral, yaw, and mixed `[vx, vy, wz]` motion. The authorized single 250k-interaction H5-V3 pilot changes only these moving-command residuals:

- Translation error is measured orthogonal to the commanded `(vx, vy)` vector.
- Yaw error is actual `wz - commanded wz` when yaw is commanded.
- Heading error is relative to an integrated commanded heading when yaw is commanded.
- Pure forward/reverse reward paths, route sampler, PPO shape and optimizer, target guard, mapper, source weight, calibration/deployment state, and all strict evaluator thresholds remain unchanged.

Authorization: `h5_v3_command_conditioned_se2_alignment_250k_authorization_20260812.json` (SHA-256 `42bb85195d50da19d9f376f7074733090df81ddd210f4d28ce8d533a9b229e79`). It prohibits hardware and requires the frozen 38-segment strict gate before any 1M or multi-seed work.

## Training and preflights

- Completed run: `h5_training_runs_20260812/unified/h5_unified_250k_v3_direct_se2align_cleanv22_v5/`.
- Final policy SHA-256: `54f487dffcbc682803f267df8592640e0093a88e2ce72df809d7d80d93fa38a9`.
- Unified training manifest SHA-256: `bd4b7ad5941b46e7138cc10d1a96c89762de0a9ba80098165e0f93e9c3859486`.
- Exact 250,000 interactions / 400 PPO updates; GPU device and checkpoint audits passed; source/teacher artifacts remained unchanged.
- The reward-only parity preflight held reset and four control steps exactly equal for actor observation, privileged observation, supplied action, control, qpos, qvel, and done. Only reward/diagnostic residuals differed.
- Focused tests passed: 28 passed, 2 skipped for the SE(2), substep-contact,
  and command-contract run; 19 passed for the same-weight provenance and
  command-contract run.

## Strict evidence

The first alias manifests were rejected before simulation because they copied and relabelled the unified training manifest instead of binding the source. `create_h5_sameweight_alias_manifest.py` now emits a minimal, hash-bound `source_candidate` wrapper and rejects an unbound parameter SHA. New wrapper hashes are planar `7180933baad6c31e9da7e2b60b358d493feeee31abb9833387210bbd10b8e35f` and reverse `c86ee2f3242499eaf7ab5981de436379754350a4143c0f5f68f5b55465067f99`.

The evaluator resolved both wrappers to the same unified V3 training manifest and ran the frozen strict configuration: seed `20260833`, one episode, 6 s moving segments, 6 s transitions, 2 s transition stands, 1.5 s warmup, initial joint noise `1.0`, and initial base speed `0.1`.

- New evidence: `h5_unified_250k_v3_direct_se2align_cleanv22_v5_strict_20260812_provenancefix_v1.json` (SHA-256 `253271c4a3b1a98524c64f97856def26bbb030052f0d3bfebbb244102e8ba925`).
- Frozen comparison: `h5_v3_substep_alignment_preflight_20260812_v2/baseline_no_capture.json` (SHA-256 `5a3024ed9caa429fca4baefba754e76d125e7c949a951c132f951f5c925fb8fc`).
- Both use the exact same evaluator configuration. Neither run fell in any of the 38 segments; safety guard, target limits, joint limits, finite-state, and no-legacy-fallback audits passed.
- Strict gait-quality passes regressed from **3 / 38** to **2 / 38**. Required moving commands remain **0 / 12**. The candidate is rejected.

Some local tracking values improved but do not justify promotion:

| Command | baseline → SE(2) primary error (m/s) | baseline → SE(2) yaw error (rad/s) |
| --- | ---: | ---: |
| forward | 0.0070 → 0.0050 | 0.1343 → 0.1095 |
| reverse | 0.0500 → 0.0500 | 0.0015 → 0.0020 |
| forward-left turn | 0.0043 → 0.0034 | 0.1947 → 0.1926 |
| forward-right turn | 0.0252 → 0.0240 | 0.0657 → 0.0489 |

Reverse remains effectively stationary (projected speed approximately zero), and every mixed/reverse moving segment still fails strict gait quality. The dominant repeated failures are debounced contact quality (24 segments), slip RMS/P95 (22 each), touchdown interval regularity (22), left/right step count (20), normal-force p99 (19), contact cadence (19), and SE(2)/backtracking metrics. This is a gait-generation/contact problem, not a manifest, measurement, safety-guard, or fall-stability problem.

## Decision and next checkpoint

Reject this one-change reward pilot. Do not extend it to 1M, extra seeds, 20×30 perturbations, deployment, camera capture, or hardware. The immediate next checkpoint is an independently reviewed single-pilot proposal that addresses the correlated reverse/contact/slip failure without weakening the strict evaluator or reintroducing teacher/table/target authority. Hardware remains `PROHIBITED`.
