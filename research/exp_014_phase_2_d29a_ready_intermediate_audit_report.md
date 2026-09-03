# EXP014 Phase 2-D29A historical READY intermediate audit

Primary classification: `EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED`.

## Historical provenance

The read-only provenance table records the existing exp_012 Stage 2N initial checkpoint, exp_012 Stage 2Q DAgger-round-2 checkpoint, and exp_013 W2P1 selected STOP checkpoint. Their SHA-256 values, 124D actor contract, zero velocity/yaw command, gait-0 input, and original evaluator semantics are in `historical_ready_provenance.json`. The old STOP evaluator used strict flight-zero/double-support criteria; a gait-0 actor input is not itself a STOP command.

D28Z remains `EXP014_D28Z_BOUNDED_SOLVER_FAIL`. The new scientific adjudication is `HZ_CONTROL_CONFLICTS_WITH_FIRST_STEP_TASKS`; it does not edit D28Z.

## READY state

Candidate source coverage is in `ready_state_metrics.json`. A candidate is valid only with zero canonical safety failures, no more than 0.10 m two-second XY displacement, mean horizontal speed no more than 0.08 m/s, p95 yaw rate no more than 0.15 rad/s, and at least one reproducible periodicity signal (support switches, load-ratio oscillation, or strict liftoff/touchdown). Raw per-step observations are in `ready_state_replay.csv`/`.json`.

| Candidate | READY-valid sources | Mean speed (m/s) | Max net XY (m) | Mean yaw p95 (rad/s) | Max yaw p95 (rad/s) | Safety-failed sources |
|---|---:|---:|---:|---:|---:|---:|
| exp013_w2p1_stop | 0/8 | 0.0803 | 0.1761 | 0.3393 | 0.4485 | 4/8 |
| stage2n_initial | 0/8 | 0.0653 | 0.1751 | 0.3424 | 0.4522 | 1/8 |
| stage2q_dagger2 | 0/8 | 0.0585 | 0.1428 | 0.3393 | 0.4649 | 1/8 |

## Manifold comparison

The feature contract is physical-only: root/base motion, CoM/DCM, foot pose/velocity, contact force, and support phase. Command/history are not used for proximity. `ready_wmove_manifold_distance.json` records the comparison contract; proximity is diagnostic and is not a capability gate.

## Route comparison

Routes A–D are recorded in `route_comparison.csv`/`.json`: A is HARD_DIRECT, B is READY_DIRECT_SWITCH, C is HARD_READY_WMOVE, and D is the existing READY controller's native minimum-jerk command ramp when supported. Every route uses the same eight logical train-only lanes, seed `20279941`, normal fresh reset, and existing frozen checkpoints. No raw snapshot restore or additional training was used.

| Route | Safe first step | Liftoff | Touchdown | W_MOVE entry | Falls | Dangerous slips | Velocity/torque saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_HARD_DIRECT | 4/8 | 8/8 | 8/8 | 0/8 | 0 | 0 | 1 |
| B_READY_DIRECT_SWITCH | 1/8 | 8/8 | 8/8 | 0/8 | 0 | 0 | 2 |
| C_HARD_READY_WMOVE | 4/8 | 8/8 | 8/8 | 0/8 | 0 | 0 | 2 |
| D_READY_NATIVE_RAMP | 0/8 | 0/8 | 0/8 | 0/8 | 0 | 0 | 2 |

## First step and W_MOVE entry

`first_step_results.json`, `touchdown_results.json`, `wmove_entry_results.json`, and `first_divergence.json` contain per-route source counts and first-failure labels. The D29A positive-control thresholds are preserved: safe liftoff at least 6/8, p95 yaw no more than 1.5 rad/s, maximum clearance no more than 0.10 m, touchdown at least 4/8, and W_MOVE entry confirmation at least 2/8. Candidate READY replay used the historical `Isaac-Exp012-G1-Reverse-PhaseR1-v0` runtime; the four route comparisons used the existing Exp013 directional route runtime with the same frozen actors and normal reset, as recorded in the raw metadata.

## Protection and repository

Starting HEAD: `14fa0ab15676aee67ae19ff16342849873a8cdd6`. D29A output was generated without modifying D28Z or earlier artifacts. Persistent update: `0`; new learned checkpoint: `0`; PPO/CEM: `0`; WBIK/centroidal modification: `0`; raw snapshot restore: `0`; validation/held-out: `0`; RUN integration: `0`; remote push: `false`. The pre-existing worktree status is preserved in `protected_hashes.json`.
