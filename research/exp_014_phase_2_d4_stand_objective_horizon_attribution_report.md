# EXP014 Phase 2-D4 STAND Objective/Horizon Attribution

## Outcome

Classification: `EXP014_D4_SETTLE_HOLD_CONTRACT_AUTHORIZED`. Route E alone is authorized. The existing formal metric and D3 classification are unchanged.

## Training horizon and GAE

The D3 pilot is `MIXED_OR_ASYNCHRONOUS`. Its 20-second episodes continue across 24-step (0.48-second) rollout boundaries; done environments reset asynchronously and validation events force full resets. GAE stops at every rollout boundary and uses `critic(obs_after_step_24)` for all later value propagation. The identical 100-step reward/value sequence gave step-0 mean advantages H24=-0.0036, H50=0.0693, and H100=0.0877.

The critic has horizon-dependent error: signed bias is -0.1575 at t=0, -0.3398 at t=4, and 0.7044 at t=50. It therefore underestimates early settling return but overestimates the later state; classification is `VALUE_HORIZON_MULTIPLE_ERRORS`.

## Formal-window decomposition

| Window | XY mean | XY p95 | abs yaw mean | abs yaw p95 | within both |
|---|---:|---:|---:|---:|---:|
| W0 0.00-0.48s | 0.1874 | 0.4846 | 0.2256 | 0.5795 | 6.05% |
| W1 0.48-1.00s | 0.0543 | 0.1025 | 0.0585 | 0.2498 | 76.85% |
| W2 1.00-1.50s | 0.0277 | 0.0587 | 0.0239 | 0.0309 | 98.51% |
| W3 1.50-2.00s | 0.0104 | 0.0193 | 0.0082 | 0.0178 | 99.69% |

W0 is the primary failing window. W2/W3 are already stable.

## Settle versus hold

- Existing two-second whole-window practical STAND: 58.82%
- RESET_TO_STAND diagnostic: 96.08%
- STAND_HOLD diagnostic: 99.02%
- Fall / dangerous slip: 0.98% / 0.00%

This satisfies the preregistered Route E condition: transition and hold independently exceed 95%, while only the reset-inclusive whole-window average fails.

## Reward and policy-gradient attribution

The V1 XY and yaw tracking terms are present and continuous. Their combined gradient norm is 14.1721, versus total 13.8799 (ratio 1.021). Regularization opposes settling (cosine -0.990), but its norm is only 0.7234, or 5.10% of settling. Counterfactual R_ALL, R_SETTLE_ONLY, and R_NO_ACTION_REG all point in an aligned settling direction. Reward underweighting is therefore not supported as the primary cause.

## Temporary horizon updates

| Clone | interactions | KL | clip | gradient max | formal | reset-to-stand | hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| U24 | 11424 | 0.0130 | 20.04% | 25.27 | 66.67% | 98.04% | 99.02% |
| U50 | 23800 | 0.0088 | 12.13% | 20.48 | 63.73% | 96.08% | 98.04% |
| U100 | 47571 | 0.0098 | 13.79% | 27.54 | 68.63% | 97.06% | 97.06% |
| M24 | 45696 | 0.0091 | 12.93% | 21.63 | 65.69% | 97.06% | 98.04% |
| M50 | 47600 | 0.0090 | 12.85% | 16.33 | 67.65% | 97.06% | 99.02% |
| M100 | 47545 | 0.0100 | 14.07% | 29.73 | 67.65% | 96.08% | 98.04% |

Maximum long-horizon gain was 1.96%, below the preregistered 3pp clear-improvement threshold, and reset-to-stand did not improve. Route H is not authorized.

## Root cause and authorization

Primary: `STAND_EVALUATOR_CONFLATES_SETTLE_AND_HOLD`. Secondary: `STAND_GAE_BOOTSTRAP_MISMATCH`, `STAND_VALUE_FUNCTION_HORIZON_ERROR`, and a small `STAND_REGULARIZATION_GRADIENT_CONFLICT`. Route E versions a `RESET_TO_STAND + STAND_HOLD` capability contract and retains the old two-second whole-window metric as a diagnostic. It does not retroactively pass D3.

No persistent PPO update, checkpoint creation, reward/config edit, held-out evaluation, DAgger work, Student work, or RUN integration occurred.
