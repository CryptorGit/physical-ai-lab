# EXP014 Phase 2-D3 Dedicated STAND Specialist Report

## Classification

`EXP014_D3_PARENT_PILOT_NO_IMPROVEMENT`

The registered stop rule fired: neither fixed parent improved its validation practical-STAND rate after 20 PPO updates. Formal curriculum training, held-out evaluation, reset-boundary labeling, and authorization were therefore not executed.

## Parents

P0 was exp_007 `model_4246.pt` (SHA-256 `734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621`), originally 98% settle/hold with 2% fall. P1 was exp_012 Stage 2Q (SHA-256 `66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698`), originally 99% practical moving-to-stop with 0% fall. Both were expanded to the 141D actor contract with exact legacy-column copies, zero new columns, and `max difference = 0.0`.

## Horizon diagnosis

| Parent | Hold | Practical STAND | Fall | Slip | Speed mean | Yaw mean | Settling p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 2 s | 55.88% | 0.00% | 0.00% | 0.06586 | 0.07773 | 1.00 s |
| P0 | 3 s | 94.12% | 1.96% | 1.96% | 0.05087 | 0.05759 | 0.94 s |
| P0 | 4 s | 97.06% | 1.96% | 1.96% | 0.03903 | 0.04456 | 0.96 s |
| P0 | 6 s | 97.06% | 2.94% | 1.96% | 0.02953 | 0.03140 | 0.98 s |
| P1 | 2 s | 66.67% | 0.98% | 0.98% | 0.06778 | 0.07454 | 0.96 s |
| P1 | 3 s | 95.10% | 0.98% | 1.96% | 0.04874 | 0.05402 | 0.94 s |
| P1 | 4 s | 98.04% | 0.00% | 0.00% | 0.03445 | 0.03530 | 0.92 s |
| P1 | 6 s | 96.08% | 2.94% | 1.96% | 0.02890 | 0.03238 | 1.19 s |

The main failure is the two-second averaging boundary, not an inability to converge by six seconds. Residual speed and yaw dominate; fall/slip remain small.

## Parent pilot

Both pilots used 476 train recipes, 24 rollout steps, seed 20278901, fixed 1.5e-5 learning rate, and 20 updates (456,960 PPO interactions total). P0 moved from 60.78% to 51.96%; P1 moved from 67.65% to 67.65%. P1 ranked ahead at update 20 but was not selected because it did not improve over its own initial checkpoint. Held-out was not used.

## Reward

Exp014StandRewardV1 exactly mirrors the parent Stage-2 reward family: continuous XY and yaw tracking, vertical velocity, roll/pitch angular velocity, flat orientation, torque, acceleration, action-rate, foot-air/slide, joint-limit and joint-deviation terms, plus fall termination penalty. It already supplies continuous zero-command XY/yaw gradients. V1 formal plateau testing was not reached and Reward V2 was neither authorized nor used; additional-term gradient contribution is therefore not applicable.

## Training and validation

C1--C4 formal training did not start. Formal updates/interactions are 0, the one-update stability and early guards are not applicable, failure strata were not frozen, validation checkpoint selection did not occur, and no specialist checkpoint exists.

## Held-out, boundary labels, and roles

Held-out remained unopened and no fallback occurred. Reset steps 0--3 were not labeled; label count is 0 and `Exp014DedicatedStandBoundaryLabelsV1` was not created. No `S_HOLD` was authorized. `S_STOP` remains exp_012 Stage 2Q and `W_MOVE` remains exp_013 W1B-R2; the three-state role comparison was not run.

## Repository and protection

Starting HEAD: `5fd85977de719605a9ebbfe6bb647bf8a01274b3`. Protection status: `PASS`. Existing exp_005--exp_013 state, exp_014 datasets/splits/manifests/checkpoints, physics, and evaluators were not modified. Unified Student training, DAgger Dataset V2, RUN integration, and OMNI-RUN were all zero. No remote push was performed.
