# EXP014 Phase 2-D26U — Fresh S_HOLD source and offline START execution

## Classification

`EXP014_D26U_FRESH_SOURCE_CAPTURE_FAIL`

The D26T artifacts were read-only inputs. D26T remains `EXP014_D26T_OFFLINE_START_KINEMATICS_FAIL` and was not overwritten. The strict D26U source gate is fail-closed: `6/8` sources are valid. Recipe IDs with a canonical applied-torque dwell are `[0, 3]`; endpoint support is valid for all captured sources, including the diagnostic support-loss history `[1, 2]` where applicable.

## Fresh S_HOLD sources

All 8 fixed train-only recipes (4 ORIGINAL, 4 MIRRORED) reached a fresh RESET_TO_STAND confirmation and an additional 1.0 s STAND_HOLD endpoint. The identity-complete bundle contains obs_123, obs_141, compatibility obs_143 (141D plus two non-policy padding columns), root/joint state, all body pose/velocity fields, foot/contact history, air-time buffers, CoM/DCM, Jacobians, masses, torque/effort/limit margins, and safety buffers. Capture OFF/ON parity passed for all 8 independent fresh process pairs with capture mutation 0.

Bundle SHA-256: `b164cf1882eac2b45e5f0ee019bf5e21df57dd7c5457fba21d3d30c90caf345f`. Endpoint control steps by recipe are `[125, 132, 129, 125, 128, 125, 130, 129]`; valid recipes are `[1, 2, 4, 5, 6, 7]` and the two invalid recipes are rejected solely by accumulated canonical torque saturation at control step 10. The support-loss records for recipes `[1, 2]` are retained as diagnostics, while all eight endpoints have valid support.

## Transition geometry

T_ref is the canonical swing-duration median: `8` control steps / `0.160000` s. Complete D26S E0 windows: `2308` (`{'LEFT': 1150, 'RIGHT': 1158}`). Aggregate step length p05/p50/p95 = `0.021411` / `0.038857` / `0.056498` m; width = `0.298865` / `0.304350` / `0.317631` m; clearance p50/p75/p90 = `0.057543` / `0.061676` / `0.063748` m; landing vertical velocity p05/p50/p95 = `-0.018956` / `-0.016513` / `-0.014293` m/s. The fixed grid remains 0.30/0.40/0.50 s × 0.8/1.0/1.2 T_ref × p50/p75/p90.

LEFT p05/p50/p95: length `0.019528`/`0.029011`/`0.046041` m, width `0.299930`/`0.304864`/`0.319628` m; RIGHT p05/p50/p95: length `0.028463`/`0.044839`/`0.058953` m, width `0.298254`/`0.304091`/`0.314006` m. Full p05/p25/p50/p75/p90/p95 reductions are in `wmove_step_geometry_statistics.json`.

## Floating-base semantics

D26 WBIK V1 is `FB0_FIXED_WORLD_ROOT`. Its Jacobian columns are joint columns 6:43, its output is 37D q/dq/action, and no generalized root translation variable is solved. CoM world targets are therefore joint-induced differential targets; D26U did not change this protected implementation. The audit is recorded, but the source gate failed before it could authorize a kinematic feasibility claim.

## Compatibility and target semantics

LEFT first swing targets `LEFT_POST_TOUCHDOWN` episode 52/step 111; RIGHT first swing targets `RIGHT_POST_TOUCHDOWN` episode 187/step 115. The source-target table preserves side-specific native states, contact configuration, CoM/DCM gap, joint/action gap, foot displacement, and pelvis reference. No target average, mirror synthesis, or reverse-family mapping was used.

## Offline plans

The fixed ledger registers `432` plans (`8 × 2 × 27`). Because the source-validity gate is `FAIL`, WBIK executed 0 and eligible plans are 0. All 432 entries are `BLOCKED_SOURCE_STATE_INVALID`; this is not classified as WBIK numerical failure, timing mismatch, or source-target geometry incompatibility. No physics was run.

## Coverage and authorization

LEFT coverage: 0/8. RIGHT coverage: 0/8. Mirror tuple coverage: 0/8. Bilateral and single-side D27 authorization are both false. `exp014_d27_not_authorized.json` is emitted. The next action is to resolve the invalid fresh source gate; do not start D27 physics or PPO.

## Repository and protection

Starting HEAD: `a52d092544eb42e090ff531bf78bbbd9fc605762`. Ending HEAD before commit: `a52d092544eb42e090ff531bf78bbbd9fc605762`. D26U added only fresh-capture/finalization scripts, D26U results, and this report. Protected experiments/artifacts, S_HOLD, Stage 2Q, W_MOVE, S_STOP_OMNI, CoM/foot/WBIK/action conversion, datasets, checkpoints, and optimizers were not modified. Persistent policy update: 0. New learned checkpoint: 0. Model-based START physics: 0. Raw snapshot restore: 0. PPO/CEM/validation/held-out/RUN: 0. Remote push: false.
