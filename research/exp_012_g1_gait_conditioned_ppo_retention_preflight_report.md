# EXP 012 Stage 2N — gait-conditioned PPO endpoint-retention preflight

## Outcome

Classification: `GAIT_CONDITIONED_PPO_MULTIPLE_FAILURES`.

The integrated parent, critic initialization, mixed Adam-moment mapping, strict LR synchronization,
anchor collection, beta preflight, and first PPO update all passed. The minimum passing one-update
coefficient was `beta=0.01`. The single authorized continuation stopped at iteration 4 because WALK
anchor KL reached `0.066552`, above the `0.05` early guard. Rollout KL remained `0.014060`; this was
semantic endpoint drift, not PPO numerical instability.

## Parent and optimizer

The Stage 2K mean is bitwise identical. WALK/RUN std use calibrated multipliers `0.30/0.65`.
The critic copies RUN-teacher moments/weights for corresponding 123D parameters and zero-initializes
the gait column. Adam step starts at 105,000 and LR at 1.5e-5.

## Selection

The initial integrated checkpoint was selected because updated checkpoints violated the consecutive
endpoint-retention guard. It retains the already audited deterministic and stochastic endpoint/toggle
behavior and has exact reference KL zero. No production policy was updated.
