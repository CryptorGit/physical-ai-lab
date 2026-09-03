# EXP012 Stage 2B — PPO runtime LR resume fix

## Resume contract

`Exp012StrictPPOResumeContract` treats restored optimizer param-group LR as the
only resume source of truth. The optimizer restored `2.25e-05` while the
old PPO runtime field held `0.001`; the adapter synchronizes the PPO/adaptive
scheduler current LR before rollout or optimization. All six offline unit tests
pass, including fresh-training separation, ambiguous groups, missing optimizer
state, and serialization. The first patched optimizer step used `2.25e-05`.

## Causal comparison

Both paths used rollout SHA `33812a8767cddcaa819d2448f4c78e14eb1e02e53c75c90819bb03268def39fa` and the same stored permutation.
U0 wrote `0.001` before its first step; U1 used `2.25e-05`.
First-step KL fell from `0.288926` to
`0.000146`, and first-step clip from
`75.10%` to `0.00%`.
Final KL fell from `0.185594` to
`0.015432` (91.7% reduction); final clip fell
from `70.91%` to `22.99%`
(47.9 percentage points). Ratio p99 fell from
`3.376` to `1.518`.

## Shadow equivalence

U1-A and a separate fresh-process U1-B matched bitwise after all 20 optimizer
steps. Actor, critic, std, and Adam moment maximum differences are all zero;
Adam ends at 85,020. Rollout and permutation hashes match.

## Patched stability

Final rollout KL is `0.015432`; maximum over all steps is
`0.018701`. Joint clip is
`22.99%`, mean-action shift `0.0829`,
critic gradient remains finite, value loss is `0.034733`, and
NaN/Inf count is zero. Every cohort is below KL 0.20 and clip 0.60.

## Continuation integrity

The diagnostic patched state was saved and reloaded without another optimizer
step. Actor/critic/std and optimizer hashes match; Adam is 85,020. Optimizer,
runtime, and scheduler LR all reload as `7.59375e-05`.
The parent uses identity normalizers, so no independent normalizer state exists.

## Classification

`PPO_RUNTIME_LR_RESUME_FIX_PASS`

Gate disposition: `IMPLEMENTATION_FIX_VALIDATED`.

Pilot readiness: `EXP012_PILOT1_RETRY_READY_AFTER_RESUME_FIX`.

Next: retry Pilot 1 once with this strict-resume contract and all other settings
unchanged. No Pilot retry was executed in Stage 2B.

## Repository and protection

Starting HEAD: `e2f7bea3b1acb82edf5fc968908b342762edebec`.
The parent, Pilot initial, and Pilot iteration-1 checkpoint hashes remain
unchanged. Stage 2B modifies only the exp_012 local resume path, its README,
diagnostic implementation/results, and this report. RSL-RL site-packages,
Isaac Lab core, rewards, curriculum, network, observations, and actions are
unchanged. New Pilot interactions and production policy updates are zero.
Pre-existing unrelated dirty paths were preserved, and no remote push occurred.
