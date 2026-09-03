# EXP014 Phase 2-D28X scope-aware centroidal authority

Classification: `EXP014_D28X_ACTIVE_LIMIT_ENFORCEMENT_UNRESOLVED`.

## Ambiguous directions

D28W's 12 ambiguous directions were matched by joint name and evaluated against C3/C4 active sets.  The manifest records A_hz columns, formal penetrations, critical-window occurrences, D28S/D28U solver usage, and active/pass-through status.  The maximum formal violation audit is in `formal_maximum_violation_audit.json`.

| Joint | Direction | Group | C3 relevance | C4 relevance | A_hz p50 | |A_hz|·vlim p50 | Formal max penetration |
|---|---|---|---|---|---:|---:|---:|
| left_hip_pitch_joint | lower | left leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.232799 | 7.44956 | 0 |
| left_hip_pitch_joint | upper | left leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.232799 | 7.44956 | 0 |
| right_hip_pitch_joint | lower | right leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.179615 | 5.74768 | 0 |
| right_hip_pitch_joint | upper | right leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.179615 | 5.74768 | 0 |
| left_hip_roll_joint | lower | left leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.200627 | 6.42007 | 3.10242e-05 |
| left_hip_roll_joint | upper | left leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.200627 | 6.42007 | 0 |
| right_hip_roll_joint | lower | right leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.19259 | 6.16289 | 0 |
| right_hip_roll_joint | upper | right leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.19259 | 6.16289 | 0.000190496 |
| left_knee_joint | upper | left leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.06557 | 1.3114 | 0 |
| right_knee_joint | upper | right leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.0485424 | 0.970848 | 0 |
| left_ankle_pitch_joint | lower | left leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.00227732 | 0.120698 | 0.011176 |
| right_ankle_pitch_joint | lower | right leg | AMBIGUOUS_ACTIVE | AMBIGUOUS_ACTIVE | 0.00144821 | 0.0767551 | 0.00142962 |

Active unresolved directions: 10 (rows across C3/C4: 20).

## Targeted enforcement

The targeted probe used only AMBIGUOUS_ACTIVE directions, fixed root, the same asset/PhysX/implicit actuator/dt/decimation, 0.01/0.05/0.10 rad offsets, 100 hold steps, and 200 release steps.  It executed diagnostic physics only.  Results and fixed classifications are in `targeted_extended_probe_results.json` and `active_limit_resolution.json`.

| Joint | Direction | Response slope | Outward velocity gate | Release gate | Classification |
|---|---|---:|---|---|---|
| left_hip_pitch_joint | lower | 0.000116766 | False | True | ACTIVE_LIMIT_STILL_AMBIGUOUS |
| left_hip_pitch_joint | upper | 0 | False | True | ACTIVE_LIMIT_STILL_AMBIGUOUS |
| right_hip_pitch_joint | lower | 0 | False | True | ACTIVE_LIMIT_STILL_AMBIGUOUS |
| right_hip_pitch_joint | upper | 0 | True | True | ACTIVE_LIMIT_ENFORCED |
| left_hip_roll_joint | lower | -0.000517876 | True | True | ACTIVE_LIMIT_ENFORCED |
| left_hip_roll_joint | upper | -0.00217059 | False | True | ACTIVE_LIMIT_STILL_AMBIGUOUS |
| right_hip_roll_joint | lower | -0.000705191 | False | True | ACTIVE_LIMIT_STILL_AMBIGUOUS |
| right_hip_roll_joint | upper | -0.000666888 | False | True | ACTIVE_LIMIT_STILL_AMBIGUOUS |
| left_knee_joint | upper | 0 | False | True | ACTIVE_LIMIT_STILL_AMBIGUOUS |
| right_knee_joint | upper | 0 | False | True | ACTIVE_LIMIT_STILL_AMBIGUOUS |
| left_ankle_pitch_joint | lower | 0 | False | True | ACTIVE_LIMIT_STILL_AMBIGUOUS |
| right_ankle_pitch_joint | lower | 7.52386e-06 | False | True | ACTIVE_LIMIT_STILL_AMBIGUOUS |

The maximum formal violation was: left_ankle_pitch_joint lower at source=P1_S_HOLD_formal_rollout, episode=49, step=76, phase=S_HOLD, penetration=0.0111760497 rad, probe-envelope ratio=1.4726677243524293.  Its formal population had no matching formal substep capture; the report records that limitation rather than fabricating a substep trajectory.

## Pass-through contract

Pass-through joints reused the same D27 V2A q_cmd field exactly.  No nominal-pose replacement, q=0 assumption, projection, or command mutation was used.  D28W's resolved implicit actuator parity is retained.

Pass-through positive controls: `True`; C3/C4 q_cmd bitwise mismatch count: 0/0; D28W actuator effort parity: PASS.

## Scope-aware authority

C3/C4 replay is conditional on all active limit directions being resolved.  Because 10 authority-relevant directions remained ambiguous, the corrected replay was not executed (`NOT_EXECUTED_CONTRACT_UNRESOLVED`).  The protected 115/36-step identity remains recorded and no START capability physics was run.

## V3R4 shadow

Selected formulation: `None`.  No V3R4 shadow was created because the active-limit prerequisite failed; no physics authorization is implied.

## Protection

D28W and earlier protected inputs unchanged: `True`.  Persistent update: `0`; new checkpoint: `0`; START capability physics: `0`; LEFT START: `0`; PPO/CEM/validation/held-out/RUN: `0`; remote push: `false`.

Starting HEAD: `f2a15c27214f673c1073d3e1944bcfbeb8358bda`.  Ending HEAD before commit: `f2a15c27214f673c1073d3e1944bcfbeb8358bda`.
