# EXP014 Phase 2-D26X geometric-path, timing, and target-set audit

Classification: `EXP014_D26X_SINGLE_SIDE_TIMING_REPAIR_PASS`.

## Velocity failure

The protected D26W trace has **406/432** plans with a first velocity violation above the unchanged ratio gate 0.80. All first violations occur in `FIRST_SWING`; side counts are LEFT 216 and RIGHT 190. The dominant violating joints across violating rows are right_elbow_roll_joint (305), left_elbow_roll_joint (299), left_two_joint (287), left_zero_joint (287), right_zero_joint (287), right_shoulder_pitch_joint (282), left_shoulder_yaw_joint (279), right_two_joint (268). Group incidence is right arm (1132), left arm (1105), left wrist/hand (574), right wrist/hand (555), waist (93); severity is MILD 83, MODERATE 36, SEVERE 287. Full per-plan rows retain step, named joints/indexes/groups, required dq, limits, recipe, side, duration, clearance, medoid, and ratios in `velocity_failure_decomposition.csv/.json`.

The task-family ablation is:

| condition | rows | velocity failures | essential tolerance pass | max ratio | first phase |
|---|---:|---:|---:|---:|---|
| V0_ROOT_STANCE | 16 | 0 | 0/16 | 0.200 | — |
| V1_ADD_COM | 16 | 0 | 0/16 | 0.200 | — |
| V2_ADD_SWING | 16 | 14 | 4/16 | 4.304 | FIRST_SWING 14 |
| V3_ADD_PELVIS | 16 | 14 | 4/16 | 4.304 | FIRST_SWING 14 |
| V4_ADD_TORSO | 16 | 16 | 4/16 | 7.734 | FIRST_SWING 16 |
| V5_FULL | 16 | 16 | 4/16 | 4.577 | FIRST_SWING 16 |

`V3` itself fails the velocity gate, so the diagnosed condition does not satisfy the required C pattern; `NONESSENTIAL_TASK_CAUSES_VELOCITY_FAILURE` is not promoted. D26W remains `EXP014_D26W_OFFLINE_START_STILL_INFEASIBLE`.

## Geometry-only feasibility

Only the planned joint-velocity gate was diagnostically removed. The canonical action contract and every other mandatory gate remained active. Results: `FULLY_ELIGIBLE` 26, `GEOMETRY_FEASIBLE_VELOCITY_FAIL` 93, `GEOMETRY_INFEASIBLE` 313. Geometry-feasible source coverage is LEFT 0/8 and RIGHT 8/8 (the D26W canonical eligible coverage was LEFT 0/8 and RIGHT 5/8). Non-velocity failures are `COM_TASK_INFEASIBLE` 296, `DCM_ENDPOINT_FAIL` 219, and `SWING_REACH_INFEASIBLE` 298.

## Timing contract

`Exp014ModelBasedStartTimingV2` fixes the geometric path first, then derives `T_joint_min = max(abs(delta_q)/(0.80*velocity_limit))`, applies the fixed 1.10 margin, and evaluates FAST/NOMINAL/SLOW at 1.00/1.25/1.50. Root linear/angular velocity remains diagnostic only. Hard maxima are A=1.00 s, B=1.00 s, C=0.60 s, D=0.60 s, total=2.50 s.

| phase | T_joint_min range (s) | T_phase_min range (s) | T_safe_min range (s) |
|---|---:|---:|---:|
| DOUBLE_SUPPORT_SHIFT | 0.005000–0.005000 | 0.300000–0.500000 | 0.330000–0.550000 |
| FIRST_SWING | 0.013213–0.107008 | 0.128000–0.192000 | 0.140800–0.211200 |
| LANDING_AND_CAPTURE | 0.000379–0.002396 | 0.080000–0.096000 | 0.088000–0.105600 |
| WMOVE_ACCEPTANCE | 0.000749–0.004027 | 0.100000–0.100000 | 0.110000–0.110000 |

Exact-medoid replay: **357 plans, 166 eligible**, coverage LEFT 0/8, RIGHT 8/8, mirror tuples 0/8. Failures were `{'ELIGIBLE': 166, 'JOINT_VELOCITY_INFEASIBLE': 191}`. Selected exact-medoid plans use `RIGHT_000`; the RIGHT compatibility rank is 4 and selected timings/durations are: R0 RIGHT_000 rank4 FAST 0.72s; R1 RIGHT_000 rank4 FAST 0.74s; R2 RIGHT_000 rank4 FAST 0.72s; R3 RIGHT_000 rank4 NOMINAL 0.86s; R4 RIGHT_000 rank4 FAST 0.72s; R5 RIGHT_000 rank4 FAST 0.74s; R6 RIGHT_000 rank4 FAST 0.72s; R7 RIGHT_000 rank4 FAST 0.72s. Selected-plan velocity margins range 0.028047–0.185782.

## Target set

`WMove03ValidatedEntryTargetSetV1` uses the read-only D26T validation artifact and 50 validated native references per side (100 total); no new validation or state was created. Five minimum compatibility references plus the medoid control were frozen using train-only robust physical scales, without result labels, future actions, or physics success. The target-set replay has **264 plans, 55 eligible**, within the 288-plan maximum; coverage is LEFT 0/8, RIGHT 5/8, mirror tuples 0/8. Eligible target IDs are RIGHT_000 (10), RIGHT_003 (4), RIGHT_009 (6), RIGHT_014 (6), RIGHT_021 (15), RIGHT_045 (14); target-set velocity margins range 0.061718–0.457676. Failures were `{'JOINT_VELOCITY_INFEASIBLE': 209, 'ELIGIBLE': 55}`. It does not replace the exact-medoid single-side result.

## Offline feasibility

The selected route is `EXACT_MEDOID_TIMING`. `selected_offline_plans_v4.json` contains one deterministic eligible plan per authorized source/side: R0 RIGHT_000 rank4 FAST 0.72s; R1 RIGHT_000 rank4 FAST 0.74s; R2 RIGHT_000 rank4 FAST 0.72s; R3 RIGHT_000 rank4 NOMINAL 0.86s; R4 RIGHT_000 rank4 FAST 0.72s; R5 RIGHT_000 rank4 FAST 0.74s; R6 RIGHT_000 rank4 FAST 0.72s; R7 RIGHT_000 rank4 FAST 0.72s. Its coverage is LEFT 0/8, RIGHT 8/8, mirror tuples 0/8. The exact-medoid replay has 166 eligible rows on RIGHT and the target-set replay has 55 eligible rows on RIGHT; LEFT has no eligible rows in either route. All replay rows retain task errors, DCM error, action continuity, target rank, timing, and velocity margin.

## Authorization

`exp014_d27_model_based_start_physics_authorization.json` is present with `authorized: True`, scope `RIGHT`, and basis `EXACT_MEDOID_TIMING`. It authorizes only RIGHT selected-side diagnostic physics in D27; LEFT remains unauthorized. No model-based START physics was executed in D26X.

## Protection

D26W/D26T/D26S/D26U, S_HOLD, Stage 2Q, W_MOVE, S_STOP_OMNI, WBIK V1/V2/V2A, checkpoints, optimizers, datasets, physics/control parameters, and existing classifications remained read-only. Persistent update: `0`; new learned checkpoint: `0`; model-based START physics: `0`; raw restore: `0`; PPO/CEM: `0`; validation/held-out: `0`; RUN integration: `0`; remote push: `false`. Hash audit: `protected_hashes.json`.
