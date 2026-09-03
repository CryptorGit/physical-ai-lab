# EXP014 Phase 2-D27 RIGHT model-based START physics

Classification: `EXP014_D27_MODEL_BASED_START_SAFETY_FAIL`.

## Source and plans

The authorized scope was RIGHT first swing only. Eight fixed D26X plans were identity-checked before physics, with D26X classification preserved as `EXP014_D26X_SINGLE_SIDE_TIMING_REPAIR_PASS`. The exact selected plan rows and source-specific timing classes are in `authorized_plan_manifest.json`; target `RIGHT_000` is episode 187, control step 115. No plan, target, clearance, root reference, or duration was changed.

Fresh lifecycle endpoint gate: **4/8** eligible. Ineligible sources were fail-closed as `SOURCE_ENDPOINT_INELIGIBLE` and did not execute START.

## Weight shift

Phase A diagnostics are in `phase_a_weight_shift.json`. The primary first-step results are 0/8 (`R0 False; R1 False; R2 False; R3 False; R4 False; R5 False; R6 False; R7 False`); load transfer, left/right contact force, CoM/root displacement, and safety are retained per recipe.

## First swing

Phase B records RIGHT unload, liftoff, clearance, LEFT stance error, forward velocity, yaw, and saturation. The gate used the unchanged contact (>5 N), slip, impact, saturation, support-loss, fall, NaN/Inf, forward displacement, yaw, and roll/pitch contracts.

## Landing

Phase C records RIGHT touchdown, landing pose error, vertical velocity, impact force, support transfer, and DCM error. A hard safety event terminated the model-based episode and did not hand off to W_MOVE.

## W_MOVE entry

Entry used the pre-fixed physical-only D26T feature distance: command/history dimensions were excluded, the RIGHT p95 threshold was fixed before physics, and no new or interpolated state was created. Entry acceptance required the fixed velocity/yaw limits, target support phase, and ten-step continuous confirmation. Accepted entries: **0/8**.

## Handoff

Only accepted entries hard-switched to the frozen W_MOVE checkpoint at 0.3 m/s for 75 control steps; no action blending was used. Handoff-retention passes: **0/0** accepted entries. Handoff action jump, cosine, joint-target jump, continuity, velocity/yaw retention, next-side alternation, and safety are in `wmove_handoff_results.json` and `phase_d_wmove_acceptance.json`.

## Tracking and first divergence

`actual_reference_tracking.csv/.json` reports p50/p95/max root, CoM, DCM, stance/swing foot, joint-target, action-rate, and contact timing errors per source and phase. `first_divergence.json` records one primary failure label per episode; timeout alone was not used as a classification.

## Process parity

The first eight episodes ran in `primary`; the same eight recipes and fixed plans ran in an independent fresh `parity` process. Parity classification: **True**. Bitwise equality was attempted first; if needed, the pre-registered absolute/relative tolerance was 1e-05/1e-05. No result-dependent tolerance was introduced.

## Authorization

`exp014_d28_right_start_teacher_expansion_authorization.json` is present only if the full RIGHT route gate passed; otherwise `exp014_d28_not_authorized.json` records the fail-closed result. LEFT first swing, bilateral START, validation, held-out, final S_START authorization, PPO/CEM, persistent update, and new checkpoints remain unauthorized.

## Protection and repository

Protected hashes are in `protected_hashes.json`; protected inputs remained unchanged: true. Persistent update: `0`; new learned checkpoint: `0`; LEFT physics: `0`; remote push: `false`.

Starting HEAD: `d04cc188fd1a178752d7f89c72dd245ccf571cd3`. Ending HEAD before commit: `d04cc188fd1a178752d7f89c72dd245ccf571cd3`.
