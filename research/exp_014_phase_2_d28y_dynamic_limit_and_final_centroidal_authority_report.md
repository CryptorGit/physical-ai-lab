# EXP014 Phase 2-D28Y dynamic limit and final centroidal authority

Classification: `EXP014_D28Y_DYNAMIC_LIMIT_INVARIANCE_UNRESOLVED`.

## Dynamic limit invariance

Calibration used 64 directions: 62 D28W proven directions plus 2 D28X resolved directions. The 10 D28X test directions were excluded from tolerance construction.

```text
epsilon_drift = 7.6692892e-06 rad/control-step
epsilon_growth = 8.14975649e-05 rad
epsilon_oscillation = 0.000200500689 rad
```

| Joint | Direction | Coupling slope | Max |drift| | Max terminal growth | Release | Classification |
|---|---|---:|---:|---:|---|---|
| left_hip_pitch_joint | lower | 2.88252e-05 | 2.6491e-08 | 9.65595e-06 | True | DYNAMICALLY_ENFORCED_COMPLIANT |
| left_hip_pitch_joint | upper | -9.77125e-08 | 0 | 0 | True | DYNAMICALLY_ENFORCED_COMPLIANT |
| right_hip_pitch_joint | lower | 0 | 0 | 0 | True | DYNAMICALLY_ENFORCED_COMPLIANT |
| left_hip_roll_joint | upper | -0.00114451 | 0 | 0.000231028 | True | DYNAMIC_ENFORCEMENT_UNRESOLVED |
| right_hip_roll_joint | lower | 0.000791374 | 3.55584e-06 | 0.000167489 | True | DYNAMIC_ENFORCEMENT_UNRESOLVED |
| right_hip_roll_joint | upper | -0.000551453 | 1.49012e-07 | 5.74142e-05 | True | DYNAMICALLY_ENFORCED_COMPLIANT |
| left_knee_joint | upper | 0 | 0 | 0 | True | DYNAMICALLY_ENFORCED_COMPLIANT |
| right_knee_joint | upper | 0 | 0 | 0 | True | DYNAMICALLY_ENFORCED_COMPLIANT |
| left_ankle_pitch_joint | lower | 0 | 0 | 0 | True | DYNAMICALLY_ENFORCED_COMPLIANT |
| right_ankle_pitch_joint | lower | 0 | 0 | 0.000418156 | True | DYNAMIC_ENFORCEMENT_UNRESOLVED |

## Velocity-gate retrospective

The old 0.001 rad/s pointwise gate is retrospective only: `0/10` directions pass. The terminal-drift/net-drift gate passes `10/10` and position-invariance classification passes `7/10`; the pointwise gate is therefore not necessary for the bounded-position interpretation.

## Formal penetration

The maximum is `left_ankle_pitch_joint / lower`, P1 episode 49 step 76, `0.0111760497 rad`; interpretation: `TRANSIENT_CONTACT_LOADED_COMPLIANCE`.

## Authority contract

V5 status: `NOT_CREATED_DYNAMIC_LIMIT_CONTRACT_UNRESOLVED`. The candidate separates q_actual, monotone non-worsening q_kin, bounded dq_kin, virtual q_cmd, effort authority, and exact D27 V2A pass-through; it was not instantiated because the active-limit closure gate was `False`. Pass-through positive controls are `True`.

## C3/C4 authority

Scope replay status: `NOT_EXECUTED_DYNAMIC_LIMIT_CONTRACT_UNRESOLVED`; selected formulation: `None`; physics: `0`.

## Protection

D28X and earlier protected inputs unchanged: `True`. New physics: `0`; persistent update: `0`; checkpoint: `0`; LEFT START: `0`; PPO/CEM/validation/held-out/RUN: `0`; remote push: `false`.

Starting HEAD: `c4caa085f69865c944dac02a9d444f26a9c9385b`. Ending HEAD before commit: `c4caa085f69865c944dac02a9d444f26a9c9385b`.
