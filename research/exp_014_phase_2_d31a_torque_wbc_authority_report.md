# EXP014 Phase 2-D31A torque WBC authority

## Torque interface

- Direct effort API: `set_joint_effort_target_index` available.
- Original actuator: `q_cmd = default_q + 0.5 * normalized_action`; runtime `control_dt=0.02s`, physics dt `0.005s`, decimation `4`.
- Same rigid-body/contact/limit settings were retained.
- Direct effort API was available, but separate intervention equivalence was not authorized after Q5 failed; no implicit-PD parameters were changed.

## Offline authority

The inverse-dynamics QP used runtime mass/inertia/Jacobian data with hard dynamics, stance acceleration, unilateral/friction/CoP, torque, velocity, and joint-limit nonworsening constraints. All Q0-Q6 source probes were contact-wrench infeasible.

| Probe | Result | Sources |
|---|---|---:|
| Q1 CoM/DCM | CONTACT_WRENCH_INFEASIBLE | 0/8 |
| Q2 pelvis/yaw | CONTACT_WRENCH_INFEASIBLE | 0/8 |
| Q3 H_z | CONTACT_WRENCH_INFEASIBLE | 0/8 |
| Q4 swing | CONTACT_WRENCH_INFEASIBLE | 0/8 |
| Q5 combined | CONTACT_WRENCH_INFEASIBLE | 0/8 |

Combined authority gate: **0/8**, required `>=6/8`.

## One-step physics

Not authorized because the combined authority gate failed. No positive WBC torque intervention is claimed.

## TD1 positive control

Not authorized. No WBC TD1 result is promoted; no TD4 or retention was executed.

## Comparison with D28/D30

D28 position-level centroidal authority exposed task conflict, and D30B bounded position residual reachability was `0/8`. D31A direct effort injection was available, but the hard-contact inverse-dynamics QP was infeasible for all 8 sources, so torque-level combined authority was not demonstrated.

## Classification

`EXP014_D31A_TORQUE_WBC_CONTACT_AUTHORITY_FAIL`

## Recommended next action

`contact-model/inverse-dynamics reconciliation`

## Repository

Starting HEAD: `454f4b0371d4e46703f2530b890ad63a11e0bfb1`; execution ending HEAD remains unchanged.
Protected paths/checkpoints unchanged; persistent update `0`, new checkpoint `0`, PPO/CEM/trajectory search/Student/RUN/validation/held-out `0`, remote push `false`.
