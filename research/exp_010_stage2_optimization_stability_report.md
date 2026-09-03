# exp_010 Stage 2: optimization stability preflight

## Decision

Stage 2 is classified **OPTIMIZATION_FAILURE_MULTIPLE**. Pilot 2 is prohibited and
`POST_RUN_WALK_V1` is closed as **NO_GO**.

## Instability timeline

The first PPO update was already non-local: approximate KL was
541.339, clip fraction was 0.950,
and deterministic mean-action KL on replay states was
942.349. Safety behavior
degraded at iteration 2, while std was only
0.2077. By durable iteration 77, value loss
was 7.206e+21, critic gradient norm was
2.097e+17, and std mean/max were
0.7113/
0.7375.

## Reward directionality

Across 384 replayed segments, return/progress Spearman was
0.8852, but normalized
advantage/progress Spearman was -0.1416.
The latter fails the frozen >=0.20 gate. The pooled return correlation is largely
checkpoint-severity separation and is not consistently reproduced within
checkpoints. The frozen reward also contains no explicit POST_RUN_WALK progress
term, periodic-RUN suppression term, or completion bonus.

## Exploration

The entropy-gradient fraction was 0.1721 on
average and 0.2322 at maximum, far below the 0.60
entropy-driven threshold. Policy-loss gradients dominate log-std. Std growth
correlates with later failures, but is not the initiating failure because the mean
policy and behavior moved unsafely at the first update.

## Critic and advantage

The critic/advantage audit is **CRITIC_ADVANTAGE_UNSTABLE**. Return and value scales
grow by orders of magnitude, and the pooled advantage/progress correlation is
negative. This independently blocks a fixed-std Pilot 2.

## Shadow intervention

The fixed-batch shadow intervention was not run. Its frozen prerequisites require
directional reward, stable critic/advantage, and std as the primary failure. All
three prerequisites failed. No diagnostic or production optimizer update was
performed in Stage 2.

## Protection

Pilot 2 iterations: 0. Production checkpoint updates: 0. Reward, contracts,
architectures, source distribution, capability manifest, exp_005 through exp_009,
exp_010 Stage 1, and Isaac Lab were not modified.
