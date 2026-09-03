# EXP014 Phase 2-D28W limit enforcement and actuator parity

Classification: `EXP014_D28W_HARD_LIMIT_ENFORCEMENT_UNRESOLVED`.

## Existing violations

The read-only P0--P3 populations produced **10763** candidate-limit exceedance rows.  The per-joint names, side, magnitude, duration, q/dq context, q_cmd, phase, source family, and q+2πk audit are in `formal_limit_violation_magnitude.csv/.json` and `revolute_coordinate_wrap_audit.json`.  Candidate limits were taken from the D28V PhysX metadata; processed soft limits remained diagnostic-only.

## Limit enforcement

The isolated diagnostic executed **222/222** fixed probes at offsets [0.01, 0.05, 0.1] for 50 control steps with a 10-step release.  It was root-fixed and direct-initialized only at probe start, and is not START capability physics.  The response slope, recovery, and fixed formal envelope comparison are in `limit_response_classification.json` and `runtime_limit_enforcement_envelope.json`.  Runtime flags are in `runtime_limit_enabled_audit.json`.

## Actuator substeps

D27 exact V2A OFF/ON capture parity: **True**.  The hook cloned existing tensors around `write_data_to_sim`, `sim.step`, and `scene.update`; it added no RNG, inference, sensor refresh, physics step, or control-loop write.  The source-verified implicit contract and requested/computed/applied effort comparisons are in `implicit_actuator_source_contract.json` and `actuator_substep_parity.json`.  PhysX solver constraint force is kept semantically separate from actuator-side approximate effort telemetry.

## Canonical authority V3

The V3 contract separates q_actual, nominal-limit q_kin, and virtual q_cmd.  q_kin uses the nominal PhysX interval and 0.80 velocity ratio; q_cmd has no physical position clamp and is checked only for finite canonical setter parity; effort authority uses the verified implicit actuator clipping contract.

## Corrected authority

Corrected C0--C4 replay was **not executed because the physical contracts did not pass** on the protected D28S 115 rows (36 critical).  Results and critical-window gates are in `corrected_authority_replay.*` and `critical_window_authority_v3.json`.

## V3R3 shadow

Selected formulation: `None`.  `temporary_v3r3_full_trace_shadow.json` is diagnostic-only and records no physics.  No D28X authorization was emitted unless all contract and full-trace gates passed.

## Protection and repository

Protected hash audit: **True**.  D28V and earlier artifacts were read-only.  START capability physics: `0`; isolated limit probe physics is separately labeled diagnostic-only; persistent update: `0`; new checkpoint: `0`; LEFT START: `0`; PPO/CEM/validation/held-out/RUN: `0`; remote push: `false`.

Starting HEAD: `943743fa0bb9a88f2d7e8d1ff479c40ccbac0394`.  Ending HEAD before commit: `943743fa0bb9a88f2d7e8d1ff479c40ccbac0394`.
