# exp_014 Phase 2-D6 omnidirectional stop Teacher report

## Existing routes

All 34 conditions were audited from common W_MOVE snapshots. R0 and R3 had 0% STOP_ACQUISITION. R1 Stage 2Q reached 70.83% acquisition and 69.73% joint success, with 30.09% fall and 27.98% dangerous slip. R2 direct S_HOLD reached 69.53% joint success. R4 was physically successful (100% acquisition/hold/joint, 0% fall/slip), but failed the mandatory internal handoff gate: W_MOVE→Stage 2Q at step 25 had action L2 p95 2.3052 (>0.5) and cosine p05 0.9644 (<0.98). Therefore no read-only global route passed.

## Failure attribution

R0/R3 failed by no deceleration at the 75-step acquisition deadline. R1/R2 remained direction-dependent with fall/slip failures. R4's first failure is `ACTION_DISCONTINUITY` at step 25; its root/contact state remained continuous because the switch was in the same episode.

## Specialist training

`Exp014OmnidirectionalStopSpecialistV1` was initialized from W_MOVE. The 124D→141D expansion had max output difference 0, old columns/hidden/output copied, and 17 new columns zero. `Exp014OmniStopRewardV1` reused existing reward families; XY/yaw weights were 8/4. Settling-gradient/total was 1.025 and regularization cosine was -0.998. The one-update gate passed (KL 0.00646, all-step KL 0.04624, clip 0.0761, mean-action shift 0.02258, NaN/Inf 0).

C1 ran 40 updates, 1,904,000 interactions, 100-step rollouts, 150-step episodes, LR 1.5e-5 fixed, and gradient clipping 10. STOP_ACQUISITION stayed 0% at updates 0/1/10/20/40. The C1 progression gate failed, so C2-C4 were not entered. Five D6-only diagnostic checkpoints were retained in raw results; none is selected or authorized.

## Validation

The 16-direction, moving-yaw, and pure-yaw matrix was evaluated at each required checkpoint reached. At update 40 aggregate and minimum-condition acquisition/joint success were 0%; fall/slip were 0%/0%. No checkpoint met the validation gate.

## Held-out and parity

The specialist held-out set was not opened, no fallback was used, and process parity was not run because no checkpoint was selected. An earlier R4 read-only probe is explicitly invalidated and is not used for specialist selection or authorization.

## Labels and authorization

R4 labels captured before its internal handoff gate was calculated are retained as raw diagnostic data only and invalidated for authorization. `S_HOLD` and `W_MOVE` remain unchanged and authorized in their prior scopes. `S_STOP_OMNI` is not authorized, so Causal DAgger Dataset V2 remains denied and was not built.

## Classification

`EXP014_D6_STOP_SPECIALIST_VALIDATION_FAIL`

The only recommended next experiment is a worst direction/yaw-neighborhood diagnosis. Do not build DAgger Dataset V2.

## Protection

exp_005-exp_013, existing exp_014 datasets/checkpoints, S_HOLD, W_MOVE, S_STOP_FORWARD, physics, and splits were not modified. Student/DAgger/RUN integration counts are zero. No remote push occurred.
