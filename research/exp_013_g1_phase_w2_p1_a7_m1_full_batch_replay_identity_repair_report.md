# exp_013 Phase W2-P1-A7-M1 full-batch replay identity repair

## Outcome

Classification: `FULL_BATCH_REPLAY_IDENTITY_REPAIRED_AND_REAUTHORIZED`.

M0 and R1 were each exactly reproducible. R1 diverged from M0 at `T_POST_RESET/T_ZERO_COMMAND`, before the first teacher physics step: all 1,024 teacher observations/actions differed, while 11 environments later crossed formal-stop identity. Environment 207 moved from 0.00947 m/s and 0.00234 rad/s (M0 PASS) to 0.37568 m/s and 0.71188 rad/s with fall/slip (R1 FAIL).

## Root cause

The task, physics, randomization configuration, teacher checkpoint, command values, and external dependency hashes did not change. The R1 production wrapper constructed/restored its trainable actor, critic, optimizer, std, and serialization state before `env.reset`; the S0/M0 reference lifecycle did not. That deterministic pre-reset implementation drift changed reset-derived robot/history observations. It is not threshold jitter, ID remapping, or GPU nondeterminism.

## Repair

`Exp013FormalStopReplayRecipeV2` uses a fresh collector process. It constructs the environment and stop teacher, performs standard reset, zeroes commands, completes the 150-step deterministic stop roll-in, verifies the S0/M0 identity, and only then loads/switches to the current policy for masked collection. V1, the state pool, masks, physics, gate, and splits remain unchanged.

Two fresh runs for each mirror sign matched the M0 formal-state hashes and every captured masked rollout tensor exactly. The resulting 48,864 compact valid samples therefore reproduce the independently authorized M0 loss, gradient, temporary update, invalid-sample invariance, split isolation, and mirror parity exactly.

## Authorization

A7-R1 is reauthorized for one future run using `Exp013FormalStopReplayRecipeV2` plus `Exp013AcceptedEnvMaskedPPOV1`. No PPO update, policy checkpoint, teacher, formal rear-start evaluation, or promotion was produced in M1.

## Evidence summary

- Reference reproduction: the detached M0 worktree reproduced all 7,168 inventory rows and semantic hashes twice, including batch accepted counts `1018/998/1006/1004/1009/1007/1005`. R1 reproduced `1017` accepted states and the same 11 mismatches in three fresh launches.
- First divergence: M0/R1 observations and deterministic teacher actions already differ at control step 0, after reset and explicit zero-command assignment but before the first physics step. Mapping, pool ordering, split assignment, teacher bytes, and command values are unchanged.
- RNG/reset: R1 restored Python, NumPy, torch CPU, and CUDA RNG state before reset. No evidence supports threshold jitter, command RNG drift, event-config drift, dependency drift, or GPU nondeterminism. Reset-derived robot/history state is the first observable divergence.
- Teacher/stepping: the exp_012 Stage 2Q checkpoint, identity normalizer, deterministic-mean mode, action scaling, simulation dt, decimation, reset-to-step order, and sensor/task sources match. The teacher action differs because its observation differs.
- V2 reauthorization: the repaired pre-policy-load hashes are `07c0d9c...6008` (inventory schema) and `e6bd1d...14f1` (masked-capture schema). Both mirror passes contained 24,432 valid samples. Two fresh runs per sign and the M0 tensors matched with maximum difference 0 for observation, action, reward, done, old log probability, value, valid mask, final value, and train mask.
- Masked PPO: effective sample count is 48,864; compact loss, gradient, and temporary update differences are 0; invalid-sample perturbation remains PASS; split leakage is 0; mirror residual is 0. The reference one-update exact KL is 0.000585189, clip fraction 0, gradient norm 6.16032, and value loss 0.00298923.

## Protection audit

All datasets, labels, splits, manifests, overlays, S0 pool artifacts, V1 replay artifacts, M0 mask artifacts, existing checkpoints and optimizers, rewards, and physics remain unchanged. M1 created no persistent policy checkpoint and executed no persistent PPO update, formal rear-start evaluation, or canonical promotion. No remote push was performed.

## Existing results

S0 and A1-A6 remain valid unchanged. M0 remains valid under its reference path and is superseded by V2 only as the production replay contract. The R1 identity-fail result remains valid; its training never started.
