# EXP012 Stage 2A — first-update PPO ratio/clipping diagnosis

## Metric semantics

The two KL values do not measure the same state distribution. `0.03938` is exact
diagonal-Gaussian KL(old||new), summed over 37 actions, on the reset observation
batch captured before rollout. `0.20244` is the same exact KL definition on all
24,576 rollout observations after the 20 optimizer steps. The reported clip
fraction `0.72396` is the fraction of samples whose **joint** probability ratio
`exp(sum_37(new_logp-old_logp))` lies outside `[0.8, 1.2]`.

The smaller reset-state KL therefore cannot invalidate the rollout-state KL.
The high-dimensional sum amplifies joint clipping: many individually modest
joint changes accumulate in log-probability space.

## Rollout integrity

The original immutable rollout and exact minibatch permutation were not saved.
One permitted diagnostic recollection produced 24,576 samples (1,024 envs × 24
steps), with zero nonzero yaw commands and cohort counts 4,944 ZERO, 4,944 WALK,
4,896 RUN, and 9,792 SEQUENCE. Canonical field hashes are in
`immutable_rollout_hashes.json`; the raw tensors remain local and untracked.

Stored actions, means, and standard deviations independently reconstruct the old
joint log probability exactly. A fresh policy forward differs by at most
`2.2888e-5`, above the preregistered `1e-6` tolerance, so the strict provenance
gate is fail-closed even though no action-clipping/storage confusion was found.

## Shadow replay

The diagnostic clone executed the prescribed 20 optimizer steps (Adam
85,000→85,020), but did not match the official iteration-1 actor, critic, or std.
Consequently its per-step trace is diagnostic-only. It nevertheless exposed the
decisive resume defect: optimizer restore set LR to `2.25e-5`, while
`PPO.learning_rate` remained `0.001`; the adaptive-KL block wrote `0.001` back
before the first step. The first shadow step already reached exact KL `0.408`
and clip fraction `0.785`, so this was not solely a late-epoch accumulation.

Applying the official iteration-1 checkpoint to the recollected rollout gives
exact KL `0.20289` and clip `0.72306`, close to but not within `1e-5` of the
recorded `0.20244/0.72396`. The initial-observation KL reproduces as `0.0393815`.
The rollout KL is 99.75% actor-mean contribution; std contributes about 0.00050.

## Ratio diagnosis

For the official checkpoint on the recollected rollout, ratio median is `0.841`,
p95 `2.214`, p99 `3.602`, with lower/upper clip fractions `46.4%/25.9%`. All
cohorts show broad clipping (about 71–75%); WALK has the largest exact KL
(`0.238`), but no isolated cohort explains the result. Alternative minibatch
orders were not run after official-order shadow mismatch because that would
confound rollout/RNG mismatch with order sensitivity.

## Classification

Primary: `PPO_FIRST_UPDATE_TRUE_DISTRIBUTION_SHIFT`.

Secondary: reported-KL state-scope mismatch, 37D ratio accumulation,
actor-mean-dominated update, immediate ratio explosion, shadow mismatch, strict
old-logprob tolerance failure, and runtime learning-rate restore mismatch.
The prior `EXP012_FIRST_UPDATE_UNSTABLE` result is retained unchanged.

## Gate disposition and readiness

`IMPLEMENTATION_FIX_REQUIRED`

`EXP012_PILOT1_RETRY_BLOCKED_BY_IMPLEMENTATION`

Single next action: fix the resume contract so the PPO runtime learning-rate
attribute is synchronized from the restored optimizer, then rerun only the
immutable one-update shadow-equivalence diagnostic. No Pilot retry is authorized.

## Repository

Starting HEAD: `61941d1cabbc626834cf8df144bb00b3154198bf`.
Only Stage 2A implementation, tracked aggregates, contracts, and this report are
included. Pre-existing unrelated dirty paths were preserved. No remote push was
performed.
