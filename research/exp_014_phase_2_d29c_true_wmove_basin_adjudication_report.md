# EXP014 Phase 2-D29C true W_MOVE basin adjudication

Primary classification: `EXP014_D29C_EXISTING_WALK_ATTRACTORS_CANNOT_CAPTURE_TOUCHDOWN`.

D29B was executed and is preserved read-only. Its runtime started and ended at `600298f1d21acaf7389efd96ede081faa9bd90b9`; its artifacts were committed by `c6d374c4dc77fd704c4bdac4e7fe02f5ee942141` (`Test exp_014 post-touchdown WALK capture`). Current D29C source-of-truth HEAD is `458cd7470217d611951aa75701dab166e8b33fbe`. D29B's official classification remains `EXP014_D29B_POST_TOUCHDOWN_NOT_CAPTURED_BY_EXISTING_WALK`.

## Existing D29B result

| Route | Safe first step | Touchdown | Legacy W_MOVE entry | Legacy 75-step retention | Falls | Slips | Saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_CONTINUE_WMOVE | 3/8 | 8/8 | 0/8 | 0/8 | 0 | 0 | 1 |
| B_CAPTURE_06 | 0/8 | 4/8 | 0/8 | 0/8 | 0 | 0 | 1 |
| C_CAPTURE_08 | 0/8 | 4/8 | 0/8 | 0/8 | 0 | 0 | 1 |
| D_STAGE2N_CONTROL | 4/8 | 4/8 | 0/8 | 0/8 | 0 | 0 | 1 |

D29B used 32 physics episodes (four routes × eight sources) and no persistent update. The D29B artifact commit is `c6d374c4dc77fd704c4bdac4e7fe02f5ee942141`; its execution starting/ending HEAD was `600298f1d21acaf7389efd96ede081faa9bd90b9`.

## Entry contract reconciliation

D29A reported `0/8` using a ten-row route-speed/yaw/safety check without phase-conditioned state distance or a touchdown requirement. D29B0 reported `7/8` for STAND preconditioning and `5/8` for WALK-zero using a ten-row distance threshold from `ready_wmove_manifold_distance.json`, but no exact phase/contact match. D29B used a post-touchdown/handoff-conditioned ten-row check with nearest medoid distance and contact phase. These are not the same evaluator; the old counts are preserved, and D29C's E0-E3 labels use the common D26T/D26S phase-conditioned reference contract.

The fixed pre-switch audit is control step 99, immediately before the step-100 W_MOVE switch. Across the eight paired sources, the P_STAND versus P_WALK_ZERO physical-state distance has median `184.999` and the previous-action distance has median `3.765`. Contact/air-time/last-contact history mismatched in `8/8`; command and previous-physical-command history mismatched in `0/8`; mode history mismatched in `8/8` (target mode only, with previous mode/time/ramp equal). The difference is therefore not a velocity-command change, but mode-conditioned actor output plus resulting contact/history state. D29A remains a different actor/runtime proxy and cannot establish a gate-only causal comparison.

## Progression

The route table and per-source E0-E3 results are in `route_level_progression.csv/json` and `true_capture_classification.csv/json`. L0 is liftoff, L1 touchdown, L2 common W_MOVE neighborhood crossing, L3 at least three alternating contacts, L4 E3 stable capture, and L5 100-step retention. A ten-step crossing is never called stable capture.

| Route | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|---:|
| A_CONTINUE_WMOVE | 8/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0/8 |
| B_CAPTURE_06 | 4/8 | 4/8 | 0/8 | 4/8 | 0/8 | 0/8 |
| C_CAPTURE_08 | 4/8 | 4/8 | 0/8 | 4/8 | 0/8 | 0/8 |
| D_STAGE2N_CONTROL | 4/8 | 4/8 | 0/8 | 4/8 | 0/8 | 0/8 |
| R_A29A | 8/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0/8 |
| R_A29B0 | 8/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0/8 |
| R_B29B0 | 8/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0/8 |

The D29B0 routes have repeated alternating contacts but no common ten-step phase-conditioned crossing. Their legacy ten-step entries are therefore adjudicated as E2 transient cycle candidates only when the two-stride condition is also met; they are not E3 captures. Isolated row-level near hits remain diagnostic and are not promoted to L2.

## Return map

`touchdown_return_map.csv/json` records TD0-TD5, same-side native reference IDs, phase-conditioned distances, velocity/yaw/DCM, support force, foot placement, and action distance. Raw ratios are diagnostic and are not used as a standalone stop gate.

| Route | TD rows | Median TD distance | Median same-side ratio | Return-map reading |
|---|---:|---:|---:|---|
| A_CONTINUE_WMOVE | 48 | 15690420.952 | 0.737 | CONTRACTING |
| B_CAPTURE_06 | 24 | 1176.907 | 0.796 | CONTRACTING |
| C_CAPTURE_08 | 24 | 1256.476 | 0.734 | CONTRACTING |
| D_STAGE2N_CONTROL | 24 | 47722831.930 | 1.989 | DIVERGING |
| R_A29A | 48 | 15690420.952 | 0.737 | CONTRACTING |
| R_A29B0 | 48 | 6451371.446 | 0.821 | CONTRACTING |
| R_B29B0 | 48 | 14616627.492 | 1.567 | DIVERGING |

## Existing Stage2Q capture

`stage2q_true_capture.json` preserves D29B's 0.6/0.8 routes. CAPTURE_TRUE requires three alternating touchdowns, two complete strides, and safety; legacy ten-step basin reports are not promoted to true capture. `stage2q_wmove_true_handoff.json` contains only true-capture episodes and therefore does not treat a missing true capture as a successful handoff.

| Route | Raw common CAPTURE_TRUE | CAPTURE_TRANSIENT | CAPTURE_FAIL | Official D29B basin |
|---|---:|---:|---:|---:|
| B_CAPTURE_06 | 1/8 | 0/8 | 7/8 | 0/8 (official D29B) |
| C_CAPTURE_08 | 2/8 | 0/8 | 6/8 | 0/8 (official D29B) |

The raw identity-complete route traces contain three source-level ten-step Stage2Q crossings under the common offline recomputation, but the preserved D29B online basin ledger is `0/8` at both speeds. This legacy/raw evaluator discrepancy is retained explicitly and does not satisfy the required `>=6/8` Stage2Q capture gate. No Stage2Q→W_MOVE true handoff was therefore eligible.

## Safety and protection

D29B0's missing identity-complete telemetry was completed by two exact passive replay processes only; source hashes and available summary metrics are in `passive_replay_parity.json`. No D29B0 artifact was overwritten. No new checkpoint, persistent update, PPO/CEM, WBIK/centroidal modification, trajectory optimization, validation, held-out evaluation, raw restore, RUN integration, or remote push was performed. D29A, D29B0, D29B, D6-D29B0, S_HOLD, W_MOVE, Stage2N, and Stage2Q remain protected.

Recommended next action: `post-touchdown dynamics-constrained capture segment`.
