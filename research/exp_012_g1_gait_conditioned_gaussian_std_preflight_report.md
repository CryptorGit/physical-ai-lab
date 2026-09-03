# exp_012 Stage 2L — gait-conditioned Gaussian std preflight

## Result

**GAIT_CONDITIONED_STD_STATIC_PASS_CLOSED_LOOP_FAIL**

Stage 2K mean parameters remain bitwise unchanged. The shared-std KL failure is explained by covariance mismatch:
the std term contributes 0.133 (WALK) and 0.201 (RUN). Endpoint teacher std reduces total KL to 0.014–0.023.

## Static representation

The single log-space gait-conditioned head passes every static endpoint in both KL directions with zero std error.
Serialization, intermediate interpolation, one-step ramp continuity, and the single-weight audit pass.

## Closed-loop

Deterministic traces are byte-identical to Stage 2K. Under S100, however, WALK gait is not robust: the WALK teacher
itself yields WALK_LIKE 29%, and the conditioned student yields
0%. RUN 2.4/2.6 remains periodic
99%/100%.
RUN→WALK stochastic acquisition is 68%, below the 90% gate.

## Next

One method only: **stochastic gait-endpoint robustness diagnosis with frozen mean and teacher std**.
