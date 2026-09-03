# EXP014 Phase 2-D29B post-touchdown WALK capture

Primary classification: `EXP014_D29B_POST_TOUCHDOWN_NOT_CAPTURED_BY_EXISTING_WALK`.

## Historical provenance

The D29A Route A replay is the formal baseline.  S_HOLD uses `logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt` (`734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621`), W_MOVE uses exp013 W1B-R2 (`61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`), and WALK_CAPTURE uses exp012 Stage2Q (`66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698`).  Stage2Q was used only with gait input 0 (the frozen WALK branch), command speeds 0.6 and 0.8 m/s, and zero yaw.  No STAND Teacher was treated as a 0.3 m/s specialist.

D29A remains `EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED`; its historical READY route was not adjusted.  D28Z and earlier artifacts remain read-only.

## Stage2Q reference capture and parity

Each speed has 10,000 steady physical states from the original exp012 runtime and 50 deterministic reference states.  The physical-only feature excludes command/history dimensions.  OFF/ON capture parity, common state/action arrays, and the fixed 1e-05 tolerance are in `stage2q_capture_parity.json`.  Capture mutation is required to be zero before routes are interpreted.

## Touchdown and manifold audit

`hard_direct_touchdown_states.json` records the first strict touchdown and +2/+4 states for each Route A source.  `post_touchdown_manifold_audit.json` compares root/base motion, projected gravity, joint state, CoM/DCM relative to support, foot geometry/velocity, contact force, and support phase.  Previous action and command are excluded from physical-state distance; action discontinuity is reported separately.

## Route comparison

| Route | Controller | Safe first step | Touchdown | WALK basin | W_MOVE entry | W_MOVE retention | Falls | Slips | Saturation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_CONTINUE_WMOVE | W_MOVE 0.3 | 3/8 | 8/8 | n/a | 0/8 | 0/8 | 0 | 0 | 1 |
| B_CAPTURE_06 | stage2q 0.6 | 0/8 | 4/8 | 0/8 | 0/8 | 0/8 | 0 | 0 | 1 |
| C_CAPTURE_08 | stage2q 0.8 | 0/8 | 4/8 | 0/8 | 0/8 | 0/8 | 0 | 0 | 1 |
| D_STAGE2N_CONTROL | stage2n 0.6 | 4/8 | 4/8 | 0/8 | 0/8 | 0/8 | 0 | 0 | 1 |

A is `S_HOLD -> W_MOVE 0.3 -> W_MOVE continuation` and is the formal D29A baseline.  B and C switch at the fixed first strict touchdown +2 step to Stage2Q 0.6/0.8, require 10-step basin confirmation within 50 steps, then hard-switch to W_MOVE 0.3 for 75 steps.  D is Stage2N-only negative/control route; it is not ranked above Stage2Q.

The frozen D29A record reports Route A safe-first-step `4/8`; this fresh D29B replay reproduced `8/8` liftoff and touchdown but recorded `3/8` under the unchanged D29A safety definition.  This small run-to-run difference does not alter the D29B decision because both Stage2Q capture routes acquired the basin `0/8`.

The touchdown manifold audit found no Stage2Q reference within the fixed neighborhood p95: normalized distance p50/p95 was `10.912/12.716` at 0.6 m/s and `10.648/11.535` at 0.8 m/s, versus neighborhood p95 `3.976` and `3.151`.  Joint position was the largest median feature-group distance for both speeds.  The W_MOVE distances were also far outside the entry contract (p50 `1247.19`; p95 `1.3395e8`, with phase/support mismatches in affected states).

## Handoff and safety

`action_discontinuity.json` records action L2, cosine, target-jump proxy, torque transient, and contact continuity at every fixed switch.  Across 32 fixed switches, action L2 was p50 `1.9126` / p95 `2.3832`, cosine was p05 `0.9541`, and the joint-target jump proxy was p50 `0.9563` / p95 `1.1916` rad.  Hard safety failure prevents any later controller switch for that source.  `stage2q_basin_results.json` and `wmove_handoff_results.json` preserve the requested 6/8 and 4/8 gates.  No Stage2Q-to-W_MOVE handoff occurred, so retention is `0/0` rather than a successful retention sample.

## Decision

`EXP014_D29B_POST_TOUCHDOWN_NOT_CAPTURED_BY_EXISTING_WALK`.  Evaluate a dynamics-constrained capture segment only over the 0.2–1.0 s post-touchdown interval; do not re-optimize the first step.

This D29B stage does not authorize formal S_START, a new dataset, training, distillation, LEFT/RIGHT expansion, validation, held-out evaluation, or RUN integration.

## Repository protection

Starting HEAD: `600298f1d21acaf7389efd96ede081faa9bd90b9`; ending HEAD: `600298f1d21acaf7389efd96ede081faa9bd90b9`.  Persistent update: `0`; new learned checkpoint: `0`; PPO/CEM: `0`; WBIK/centroidal modification: `0`; raw restore: `0`; validation/held-out: `0`; RUN integration: `0`; remote push: `false`.  The pre-existing dirty/untracked status is preserved in `protected_hashes.json`.
