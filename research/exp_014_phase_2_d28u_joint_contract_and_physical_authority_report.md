# Exp014 Phase 2-D28U joint contract and physical centroidal authority audit

Classification: `EXP014_D28U_JOINT_LIMIT_CONTRACT_UNRESOLVED`

D28U is offline-only.  No physics, policy update, checkpoint, PPO, CEM, validation, held-out, LEFT START, or RUN integration was executed.

## Joint contract

The D25, D26X, D28R name contracts matched: `True`. Mapping used joint names, not array position alone. The runtime processed limits matched across D25/D26U/D28R: `True`. Separate raw USD hard-limit serialization was unavailable; the audited runtime contract is the D26U `data.soft_joint_pos_limits` capture with G1 soft factor 0.9.

All 37 policy joints were physically actuated revolute joints in the G1 actuator expressions; the wrist/hand joints were not mimic or fixed joints. Their low/zero `action_scale` entries are preserved as action-interface offsets/nominal values, not interpreted as passive joints.

## Positive controls

- `P0_S_HOLD_fresh_endpoint`: 8 states; strict empty fraction 0.0; recovery empty fraction 0.0; q-outside fraction 0.10810810810810811; q_cmd-outside count 36; outward motion 0.0777027027027027; recovery positive-control status `FAIL_OR_UNAVAILABLE`.
- `P1_S_HOLD_formal_rollout`: 6400 states; strict empty fraction 0.0; recovery empty fraction 0.0; q-outside fraction 0.12764780405405404; q_cmd-outside count 45329; outward motion 0.07652449324324324; recovery positive-control status `FAIL_OR_UNAVAILABLE`.
- `P2_W_MOVE_formal_rollout`: 20000 states; strict empty fraction 0.0; recovery empty fraction 0.0; q-outside fraction 0.15038378378378378; q_cmd-outside count 136361; outward motion 0.08157027027027026; recovery positive-control status `FAIL_OR_UNAVAILABLE`.
- `P3_D27_actual_V2A_trace`: 119 states; strict empty fraction 0.0; recovery empty fraction 0.0; q-outside fraction 0.1185555303202362; q_cmd-outside count 945; outward motion 0.07245060186236657; recovery positive-control status `FAIL_OR_UNAVAILABLE`.

The strict D28S interval is empty whenever the declared one-step re-entry requirement conflicts with the current state. The diagnostic monotone-recovery bound removes only the outward-motion requirement for an already outside state and is not adopted as a runtime contract.

## Centroidal columns

- left_hip_pitch_joint (left leg): A_hz p50 0.232799, velocity-normalized p50 7.44956.
- right_hip_pitch_joint (right leg): A_hz p50 0.179615, velocity-normalized p50 5.74768.
- torso_joint (waist): A_hz p50 0.297483, velocity-normalized p50 9.51945.
- left_hip_roll_joint (left leg): A_hz p50 0.200627, velocity-normalized p50 6.42007.
- right_hip_roll_joint (right leg): A_hz p50 0.19259, velocity-normalized p50 6.16289.
- left_shoulder_pitch_joint (left arm): A_hz p50 0.127187, velocity-normalized p50 6.7409.
- right_shoulder_pitch_joint (right arm): A_hz p50 0.124969, velocity-normalized p50 6.62337.
- left_hip_yaw_joint (left leg): A_hz p50 0.0695974, velocity-normalized p50 2.22712.
- right_hip_yaw_joint (right leg): A_hz p50 0.0678601, velocity-normalized p50 2.17152.
- left_shoulder_roll_joint (left arm): A_hz p50 0.0383784, velocity-normalized p50 2.03405.
- right_shoulder_roll_joint (right arm): A_hz p50 0.0498225, velocity-normalized p50 2.64059.
- left_knee_joint (left leg): A_hz p50 0.06557, velocity-normalized p50 1.3114.
- right_knee_joint (right leg): A_hz p50 0.0485424, velocity-normalized p50 0.970848.
- left_shoulder_yaw_joint (left arm): A_hz p50 0.00335782, velocity-normalized p50 0.177965.
- right_shoulder_yaw_joint (right arm): A_hz p50 0.00482633, velocity-normalized p50 0.255796.
- left_ankle_pitch_joint (left leg): A_hz p50 0.00227732, velocity-normalized p50 0.120698.
- right_ankle_pitch_joint (right leg): A_hz p50 0.00144821, velocity-normalized p50 0.0767551.
- left_elbow_pitch_joint (left arm): A_hz p50 0.0482074, velocity-normalized p50 2.55499.
- right_elbow_pitch_joint (right arm): A_hz p50 0.0475323, velocity-normalized p50 2.51921.
- left_ankle_roll_joint (left leg): A_hz p50 0.000372309, velocity-normalized p50 0.0197324.
- right_ankle_roll_joint (right leg): A_hz p50 0.000341105, velocity-normalized p50 0.0180785.
- left_elbow_roll_joint (left arm): A_hz p50 9.60137e-05, velocity-normalized p50 0.00508873.
- right_elbow_roll_joint (right arm): A_hz p50 0.000110385, velocity-normalized p50 0.00585039.
- left_five_joint (left wrist/hand): A_hz p50 0.000141941, velocity-normalized p50 0.00326465.
- left_three_joint (left wrist/hand): A_hz p50 0.000205196, velocity-normalized p50 0.0047195.
- left_zero_joint (left wrist/hand): A_hz p50 0.000174859, velocity-normalized p50 0.00402175.
- right_five_joint (right wrist/hand): A_hz p50 0.000180963, velocity-normalized p50 0.00416215.
- right_three_joint (right wrist/hand): A_hz p50 9.58649e-05, velocity-normalized p50 0.00220489.
- right_zero_joint (right wrist/hand): A_hz p50 0.000182153, velocity-normalized p50 0.00418951.
- left_six_joint (left wrist/hand): A_hz p50 2.30141e-05, velocity-normalized p50 0.000529325.
- left_four_joint (left wrist/hand): A_hz p50 2.91132e-05, velocity-normalized p50 0.000669603.
- left_one_joint (left wrist/hand): A_hz p50 0.000111928, velocity-normalized p50 0.00257435.
- right_six_joint (right wrist/hand): A_hz p50 1.83141e-05, velocity-normalized p50 0.000421225.
- right_four_joint (right wrist/hand): A_hz p50 1.55506e-05, velocity-normalized p50 0.000357665.
- right_one_joint (right wrist/hand): A_hz p50 0.000180249, velocity-normalized p50 0.00414572.
- left_two_joint (left wrist/hand): A_hz p50 1.76204e-05, velocity-normalized p50 0.00040527.
- right_two_joint (right wrist/hand): A_hz p50 2.03142e-05, velocity-normalized p50 0.000467226.

## Formulations

- `G0_ALL_JOINTS_STRICT`: critical all-constraint gate fractions by source `{'4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0}`; H_z improvement >=20% fractions by source `{'4': 1.0, '5': 1.0, '6': 1.0, '7': 1.0}`; feasible rows 0/115; median improvement 0.9837045313613313; max wrist/hand use fraction 0.0.
- `G1_ALL_JOINTS_RECOVERY`: critical all-constraint gate fractions by source `{'4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0}`; H_z improvement >=20% fractions by source `{'4': 1.0, '5': 1.0, '6': 1.0, '7': 1.0}`; feasible rows 0/115; median improvement 0.9982425555017295; max wrist/hand use fraction 0.5765497922589459.
- `G2_FREEZE_WRIST_HAND`: critical all-constraint gate fractions by source `{'4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0}`; H_z improvement >=20% fractions by source `{'4': 1.0, '5': 1.0, '6': 1.0, '7': 1.0}`; feasible rows 0/115; median improvement 0.9982424325984306; max wrist/hand use fraction 6.370818636409268e-13.
- `G3_FREEZE_WRIST_HAND_AND_ARMS`: critical all-constraint gate fractions by source `{'4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0}`; H_z improvement >=20% fractions by source `{'4': 0.3333333333333333, '5': 0.3333333333333333, '6': 0.3333333333333333, '7': 0.3333333333333333}`; feasible rows 0/115; median improvement -2.4710104918996096; max wrist/hand use fraction 3.733952068132982e-14.
- `G4_LEGS_WAIST_ARMS_WITH_COLUMN_SCALING`: critical all-constraint gate fractions by source `{'4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0}`; H_z improvement >=20% fractions by source `{'4': 1.0, '5': 0.7777777777777778, '6': 0.8888888888888888, '7': 1.0}`; feasible rows 0/115; median improvement 0.9997894600759608; max wrist/hand use fraction 1.5340113059904114e-07.

## Root cause

The selected interpretation is: name/order/runtime processed limits matched, but the raw USD hard-limit field was not captured while formal states or q_cmd values were outside the processed soft-limit contract

## Temporary V3R2

Temporary shadow created: `False`. Physics applied: `0`.

## Next action

capture and reconcile the USD hard-limit and runtime soft-limit contracts, then rerun the D28R shadow; no physics is authorized in D28U

## Repository

Starting HEAD `5e283ca22fc51c66999541e102d507de1c7983f7`; ending HEAD `5e283ca22fc51c66999541e102d507de1c7983f7`.
