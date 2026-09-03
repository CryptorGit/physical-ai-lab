# exp_009 Stage 2 — Dynamics-sensitive distillation

## Scope

Stage 2 tested only the `CLOSED_LOOP_SENSITIVITY_DOMINATED` hypothesis. The
student remained the 85,925-parameter `UnifiedWalkRunStudent123`; no PPO,
reward optimization, DAgger, PCGrad, adapter, skill input, multi-head model, or
teacher update was used. Stage 1 remains classified as
`MULTIPLE_FAILURE_MODES`.

## Counterfactual sensitivity

The accepted dataset contains 9,185 branch states:

- WALK steady: 4,167
- RUN steady: 2,502
- WALK_TO_RUN: 2,516

Each perturbation sign was run in a fresh Isaac application for every regime
and cycle. The same reset seed, source route, physical environment ID, and
prebranch teacher actions were replayed. Only samples with root, joint, and
velocity state inside the frozen tolerances were retained; no state setter,
teleport, snapshot injection, or simulator differentiation was used.

The official finite difference was ±0.02 normalized action (±0.01 rad after
the 0.5 action scale), with outcomes at 1, 2, 4, and 8 control steps. A
±0.01/±0.04 subset confirmed finite but nonlinear local response
(finite-difference norm ratios 1.377 and 0.651 versus the official delta).

The largest mean continuous sensitivities were hip roll, hip pitch, and hip
yaw. Ankle roll remained phase-dependent and nonzero, but was not manually
selected: right and left ankle-roll mean sensitivity norms were 0.744 and
0.412. Double support had the largest average sensitivity. The loss used a
state/regime/natural-phase-conditioned local table and data-derived contact
topology changes.

## Loss calibration and WALK-only result

Calibration used the frozen Stage 1 standard-Huber WALK-only diagnostic
student, because strict WALK-teacher initialization has numerically zero WALK
error and cannot define contribution ratios. On the fixed train split,
calibration was performed once before training:

- `lambda_dynamic = 1.565777711935205`
- `lambda_contact = 1179.59570271916`
- action/action-delta definition unchanged from Stage 0
- dynamic/contact median targets: 0.5/0.5 relative to action 1.0

The selected diagnostic checkpoint was epoch 5
(`beca6f5300fa8f7cb5a17235eb8dd4568e83b2bdf0b68d141f42e27ca9a1ecb9`).
Its deterministic WALK success rates over 50 episodes per speed were:

| speed | Stage 2 success |
|---:|---:|
| 0.6 m/s | 26% |
| 0.8 m/s | 0% |
| 1.0 m/s | 0% |
| 1.2 m/s | 0% |

Overall success was 6.5%, versus 6.0% for the Stage 1 WALK-only baseline.
Held-out mean action error remained small (`0.000216`), including small
phase-conditioned ankle-roll errors, but closed-loop retention did not recover.

## Gate and conclusion

The WALK-only gate is `WALK_DYNAMICS_LOSS_FAIL`; therefore mixed unified
distillation and the reverse diagnostic were not executed.

Final classification:

```text
DYNAMICS_LOSS_NO_EFFECT
```

The tested local linear, one-step-action sensitivity penalty does not resolve
H4. The next single method is **short-horizon nonlinear rollout supervision**.
H3 remains documented but was intentionally not modified in this stage.
