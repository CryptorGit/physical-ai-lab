# exp_014 Phase 2-D15 STAND-to-OMNI-WALK start Teacher audit

## Outcome

`EXP014_D15_DIRECT_WMOVE_START_FAIL`. Direct W_MOVE is not authorized as `S_START_OMNI`. The single fixed checkpoint was evaluated read-only; there were no policy updates, alternative checkpoints, routing, or blending.

## STAND starts

The 102 fixed D5 validation recipes produced 101 valid S_HOLD snapshots (99.0196%). Each snapshot was paired with all 34 conditions, yielding 3,468 formal episodes. The one invalid start was retained in end-to-end and safety accounting.

## Direct W_MOVE start

- Conditional WALK acquisition: 0.029121% (1/3434)
- Conditional steady hold: 100.000000% (conditional on the sole acquisition)
- Conditional joint start success: 0.029121%
- End-to-end success: 0.028835%
- Minimum condition joint success: 0.000000%

All 34 conditions were evaluated. Failure was broad rather than confined to rear or yaw sentinels. The dominant recorded failure was yaw acquisition failure (3267 episodes).

## Sentinel conditions

Rear 180° moving-yaw conditions (IDs 24/25), rear-left 135° (22/23), and rear-right 225° (26/27) each had 0% conditional joint success. No sentinel-specific exception or threshold was applied.

## Safety and handoff

- Fall: 2.797001%
- Dangerous slip: 1.585928%
- Impact: 0.000000%
- Velocity saturation: 0.000000%
- Torque saturation: 2.941176%
- NaN/Inf: 0.000000%

The S_HOLD→W_MOVE raw action discontinuity was L2 p50 1.758268, p95 2.298180, maximum 6.830077; cosine p05 was 0.951740. Root discontinuity, contact-buffer corruption, and handoff-attributed new safety failures were all zero, so the physical handoff gate passed despite the action jump.

## Conditional stages

Local-neighborhood evaluation, process parity, held-out preregistration, and held-out sealing were not executed because formal validation failed. This is required by the preregistered D15 ordering.

## Durability and protection

All 3,468 results and hashes were committed in SQLite WAL mode with `synchronous=FULL`; completed-without-result, duplicate-result, and missing-provenance invariants were zero. Two pure offline aggregate runs were bitwise identical. D6–D14, exp_005–exp_013, all checkpoints, datasets, physics, rewards, and contracts were unchanged. Remote push was not performed.
