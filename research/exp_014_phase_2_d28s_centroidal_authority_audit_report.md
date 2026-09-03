# exp_014 Phase 2-D28S — centroidal-yaw authority audit

## Scope and result

This stage used the D28R `capture_on` body trace read-only. It analyzed the
same protected D26X-selected R4–R7 plans and D28 V3 contract. No physics,
policy update, checkpoint, PPO, CEM, validation, held-out evaluation, LEFT
route, or RUN integration was executed.

The result is:

```text
EXP014_D28S_JOINT_AUTHORITY_INSUFFICIENT
```

The key distinction is that the centroidal `H_z` row remains controllable in
the post-task nullspace, but the actual D28R state cannot satisfy the frozen
one-step joint-position and `0.80 * velocity_limit` authority bounds. The
canonical raw action itself remains unbounded; this is a physical position /
velocity authority failure, not an added action clip.

## Rank and conditioning

The audit window was the START rows through the step before the protected D27
first safety failure:

| source | analysis rows | first failure step | critical rows |
|---|---:|---:|---:|
| R4 | 31 | 160 | 9 |
| R5 | 28 | 154 | 9 |
| R6 | 26 | 157 | 9 |
| R7 | 30 | 160 | 9 |

The first `abs(yaw_rate) > 0.15 rad/s` occurred at the beginning of each
START window, so the requested ±8-step window was clipped at the START
boundary. The complete step identity is in `critical_window_authority.json`.

Across all 115 analysis rows, with the fixed D28 SVD tolerance `1e-8`:

| matrix | rank | conditioning summary |
|---|---:|---:|
| `J_stance` | 6 | condition p50 13.74, p95 14.08 |
| `[J_stance; J_com]` | 9 | condition p50 94.74, p95 108.16 |
| `[J_stance; J_com; J_swing]` | 15 | condition p50 289.24, p95 777.12 |
| `[J_stance; J_com; J_swing; J_pelvis]` | 15 | rank deficient relative to its 18 rows |
| `A_hz` | 1 | rank 1, row condition 1.0 |
| `[hard-task stack; A_hz]` | 16 | H_z row is not zero after stacking |

The post-task H_z authority was present at every analyzed step:

| nullspace | H_z row norm p50 | minimum | H_z authority rank | row-space overlap p50 |
|---|---:|---:|---:|---:|
| N0 after stance | 0.45096 | 0.37533 | 1 | 0.594 |
| N1 after stance + CoM | 0.40618 | 0.23589 | 1 | 0.695 |
| N2 after + swing | 0.31705 | 0.11142 | 1 | 0.824 |
| N3 after + pelvis | 0.31705 | 0.11142 | 1 | 0.969 |

Therefore:

- A (no post-task H_z freedom): not established.
- B (H_z rank zero or centroidal row numerically invalid): not established.
- The high N3 overlap and the rank/conditioning degradation show a strong
  linear interaction with the first-step task stack, but not rank loss.

The numerical matrices, `b` vectors, root contributions, and step identity
are retained losslessly in `task_jacobian_numeric.npz` and its manifest.

## Bounded authority: F0–F4

F0 was the protected D27 V2A baseline. F1 reproduced the protected D28 V3
scalar H_z residual with the fixed all-one metric. F2, F3, and F4 used the
deterministic bounded active-set least-squares diagnostic with fixed damping
`1e-4`, tolerance `1e-9`, and maximum 148 iterations.

| formulation | H_z result | bounded critical-window result |
|---|---|---|
| F0 V2A | reference | velocity ratio p95 about 0.20; position gate remains false at the divergent actual state |
| F1 current V3 | median improvement only about 0–0.01% | no source passed the H_z gate |
| F2 H_z nullspace-only | unbounded diagnostic improvement ≈100% | 0/115 feasible bounded steps |
| F3 bounded lexicographic | unbounded diagnostic improvement ≈100% | 0/115 feasible bounded steps |
| F4 bounded H_z-first | unbounded diagnostic improvement ≈100% | 0/115 feasible bounded steps |

The F2–F4 candidate values are marked solver-infeasible and are not counted
as eligible minima. The bounded failure is due to an empty combined interval,
not a numerical NaN or a changed limit. The affected interval is formed from:

```text
abs(dq) <= 0.80 * velocity_limit
q_min <= q_current + dq * dt + endpoint_feedforward <= q_max
```

The exact lower and upper bounds for every step are in
`joint_bounds_numeric.npz`; the solver status and blocking joints are in
`joint_authority_blockers.json`.

## Task conflict

The unbounded F4 diagnostic can drive the model-predicted H_z residual near
zero, but doing so after stance preserves the mandatory first-step tasks only
partially. In the critical windows, the unbounded H_z-first pelvis residual
was approximately 0.77–0.83 p50 rad/s and 1.18–1.29 p95 rad/s across R4–R7.
CoM and swing residuals were smaller, while stance remained near its hard
solution. This confirms a task interaction, but it is not promoted to the
formal `HZ_CONTROL_CONFLICTS_WITH_FIRST_STEP_TASKS` classification because the
bounded F4 problem has no feasible point at any critical step. The frozen
task contract was not changed.

## Joint authority

The dominant concrete blockers were named joints, not anonymous indices:

```text
left_four_joint
left_six_joint
right_four_joint
right_six_joint
```

They are in the left/right wrist/hand groups. Additional right-hand bounds
appeared in some rows (`right_five_joint`, `right_three_joint`). The repeated
condition is that the endpoint feedforward plus the divergent actual q state
requires a corrective dq outside the simultaneous position and velocity
interval. For example, at R4 control step 129:

```text
left_six_joint:  lower -18.4, upper -66.33 rad/s
left_four_joint: lower -18.4, upper -52.50 rad/s
right_six_joint: lower 46.54, upper 18.4 rad/s
right_four_joint: lower 74.17, upper 18.4 rad/s
```

These are empty intervals. The canonical action remains unbounded, so the
failure is not caused by an artificial `[-1,1]` action cap. It is the frozen
joint-position/velocity authority contract applied to the actual D28R state.

The unbounded H_z-first required velocity ratio was also above the 0.80 gate
in parts of the critical windows (p95: R4 3.71, R5 1.69, R6 1.11, R7 3.11),
but the empty position/velocity intervals appear earlier and are sufficient
to fail the bounded problem. Nullspace unit-H_z directions were dominated by
waist and arm groups after N3; wrist contributions were small. No D28 joint
participation weights were changed.

## Multi-step authority

Because N3 rank was nonzero but one-step useful bounded authority was absent,
the offline 2/4/8-step diagnostic was run using only the recorded
time-varying Jacobians and bounds. It did not use future physics state and was
not a runtime controller.

```text
2-step: 36 starts, 0 feasible sequences
4-step: 36 starts, 0 feasible sequences
8-step: 36 starts, 0 feasible sequences
```

All sequence attempts were marked `NO_FEASIBLE_SEQUENCE`; no terminal
improvement was credited.

## Temporary V3R

`Exp014BoundedCentroidalWBIKV3R` was not created as a physics candidate. The
per-source critical-window bounded gate was 0/4 sources, and each source had
0/9 feasible F2/F3/F4 critical steps. The D28 V3 contract, DARE gain, task
priority, H_z target, target, timing, and action contract remain protected.

## Classification and authorization

Main classification:

```text
EXP014_D28S_JOINT_AUTHORITY_INSUFFICIENT
```

No D28T bounded centroidal physics preflight is authorized. The appropriate
next branch is a separate dynamics-constrained trajectory-optimization or
torque-level WBC study; D28S does not alter the D28 contract or return to PPO
or action search.

## Repository and protection

```text
starting HEAD: ab99ee6323516be3821e9b706be85a9dd65af7aa
physics executed: 0
persistent update: 0
new checkpoint: 0
remote push: false
```

The D28R trace, D28 contracts, D26X plans, WBIK V1/V2/V2A/V3, S_HOLD,
Stage 2Q, W_MOVE, S_STOP_OMNI, checkpoints, datasets, and unrelated dirty
state were read-only/protected. The corresponding snapshot is recorded in
`protected_hashes.json`.
