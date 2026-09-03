# EXP_011 Go2 First-Update PPO Stability Diagnosis

## Status

```text
STAGE:
STAGE_3_FIRST_UPDATE_STABILITY_DIAGNOSIS

CLASSIFICATION:
FIRST_UPDATE_FRESH_OPTIMIZER_MISMATCH

PILOT READINESS:
PILOT_READY_WITH_SINGLE_STABILITY_FIX

NEXT:
resume checkpoint optimizer state
```

No Stage 3 Pilot was run. Stage 1 and Stage 2 results, the official checkpoint,
and `model_1_unstable.pt` were not modified.

## Identity and fixed batch

Stage 2 did not serialize its rollout storage. Under the explicit diagnostic
exception, the pre-update rollout was recaptured once using the same official
checkpoint, environment, seed `20260911`, 2,048 environments, and 24 steps.
The batch contains 49,152 samples and has SHA-256
`998b963b008e3f996119ca1a5d9862d94f7d7e76fc05449b61c8ba740630f389`.
No optimizer update was made during capture.

The no-update test replays observations in the original chronological
`24 × 2048` forward shape. Policy mean, derived log-std, value, and action
log-prob reproduce exactly. Entropy differs only by `4.77e-7`, below its
diagnostic tolerance. Serialization and reload give the same result.

The action distribution is a state-independent diagonal Gaussian. RSL-RL stores
the standard deviation directly as `distribution.std_param`; log-std is
`log(std_param)`. There is no minimum, maximum, or distribution clamp. The raw
sampled action and its log-prob are stored before environment stepping. Neither
the RSL wrapper nor the joint-position action term clips this task's action.
The environment then applies
`default_joint_position + 0.25 × raw_policy_action`.

The PPO ratio contract is:

```text
ratio = exp(new_log_prob(saved_raw_action) - old_log_prob(saved_raw_action))
```

Advantages are normalized globally over the rollout batch. RSL-RL's adaptive
schedule uses analytical Gaussian `KL(old || new)`. The Stage 2 field named
`approximate_kl` was also computed with that analytical KL, not with a sampled
approximation.

## Exact KL

Applying the isolated Stage 2 `model_1_unstable.pt` to the recaptured old-policy
observations gives:

| Measure | Value |
|---|---:|
| Stage 2 reported KL | 0.51294 |
| Recomputed exact `KL(old || new)` | 0.50986 |
| Exact `KL(new || old)` | 0.51278 |
| Symmetric KL | 0.51132 |
| Sample-based approximation | 0.50014 |
| Ratio clip fraction | 0.78202 |

The absolute difference between reported and recomputed exact KL is `0.00307`.
The estimator is therefore classified `KL_ESTIMATOR_CONSISTENT`.

Mean movement contributes `0.50955`, or `99.94%`, of exact KL. Standard
deviation movement contributes `0.000318`, or `0.06%`. The distribution change
is `ACTOR_MEAN_UPDATE_DOMINATED`, not log-std dominated.

The largest joint contributions are:

| Joint | Exact KL |
|---|---:|
| RL calf | 0.09433 |
| FR calf | 0.07194 |
| FL hip | 0.06245 |
| FL thigh | 0.04703 |
| FL calf | 0.04213 |

Per-cohort exact KL is `0.16121` for ZERO_HOLD, `0.65278` for STEADY_SPEED,
`0.32997` for ACCELERATION, and `0.89550` for DECELERATION.

## Optimizer diagnosis

Stage 2 created a fresh Adam optimizer at learning rate `1e-3`, with no moments
and scheduler step zero. The official checkpoint contains optimizer state for
all 17 parameters:

```text
checkpoint iteration: 999
Adam step count:       20,000
terminal LR:           0.0003901844231062339
first-moment norm:     0.11252
second-moment norm:    0.01852
```

All shadow conditions used the same fixed batch and diagnostic mini-batch order.
The exact historical Stage 2 mini-batch permutation was not serialized, so the
isolated unstable checkpoint—not S0—is the source of truth for the historical
`0.51294` result.

| Shadow | Exact KL | Clip fraction | Gate |
|---|---:|---:|---|
| S0 fresh optimizer | 5.21573 | 0.93376 | FAIL |
| S1 freeze log-std | 5.35055 | 0.93439 | FAIL |
| S2 freeze actor mean | 0.00152 | 0.00680 | PASS |
| S3 critic only | ~0 | 0 | PASS |
| S4 restore checkpoint Adam | 0.01381 | 0.19289 | PASS, preferred |
| Fresh Adam at terminal LR | 0.05144 | 0.44096 | PASS |

Only the fresh optimizer condition fails while restored Adam passes both the
formal and preferred shadow gates. Terminal LR alone is safer than production
but misses the preferred KL and clip references. By the required causal
precedence, the primary diagnosis is `FRESH_OPTIMIZER_STATE_MISMATCH`.

## Advantage and critic

Each cohort contributes exactly 12,288 samples. DECELERATION has the largest raw
advantage spread (`std 0.28993`) and the largest actor-gradient norm share
(`41.28%`), followed by STEADY_SPEED (`31.13%`). No cohort exceeds the required
50% dominance condition. Cohort-wise diagnostic normalization still produces
KL `3.63194` and clip fraction `0.95756`; it does not resolve the update.
The result is `COHORT_BALANCED`, with deceleration recorded as the largest
secondary contributor.

Critic explained variance is `0.84689`; its diagnostic value loss is `0.03895`
and gradient norm is `0.66360`. It is classified `CRITIC_STABLE`.

## Conclusion

The Stage 2 KL is real, uses the correct old/new direction, and is not caused by
action clipping, log-prob serialization, log-std movement, cohort normalization,
or critic collapse. The immediate distribution movement is actor-mean
dominated, but the causal intervention shows that discarding the converged
checkpoint's Adam state is the primary reason that the first update is unsafe.

Exactly one intervention qualifies:

```text
resume checkpoint optimizer state
```

Reward, curriculum, learning rate, entropy, and log-std must not be changed at
the same time. Stage 3 authorizes a future Pilot with that single stability fix;
it does not execute it.
