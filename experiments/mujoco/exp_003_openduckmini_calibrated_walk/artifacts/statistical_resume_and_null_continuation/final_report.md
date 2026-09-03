# OpenDuckMini Statistical Resume and Null Continuation Report

## 1. Executive Summary

Checkpoint payload Gate: **`CHECKPOINT_PAYLOAD_BIT_EXACT_PASS`**.

Statistical resume Gate: **`STATISTICAL_RESUME_FAIL`**.

All 20 U and 20 R trials completed without crash, nonfinite value, fall, or
termination. Nevertheless, 10 of 12 primary endpoints exceeded the
pre-registered absolute standardized-effect limit of 0.25. Resume-mode
two-step return was lower, tracking RMSE was higher, and Adam/PPO endpoint
distributions shifted. Null continuation was therefore not started.

## 2. Scope and Non-Claims

This phase changed no reward, yaw objective, teacher, sampler, curriculum,
network, PPO hyperparameter, scene, domain-randomization distribution, MJX
segment/scatter operation, batch-size policy, or deterministic XLA setting.

The statistical Gate used a four-environment diagnostic checkpoint. It
preserves the production batched GPU control/physics/update mechanism but is
not a production-size or performance-training run. No v52/v59/v60 policy was
promoted.

## 3. Backend Nondeterminism Contract

Serialized state/model/RNG/controller payloads remain bit-exact requirements.
GPU MJX batch outputs, batched trajectories, gradients, and post-update
parameters are evaluated statistically. The exception is limited to the
previously established `smooth.crb -> segment_sum -> scatter-add` batch path.

See `backend_nondeterminism_contract.md`.

## 4. Source and Parent Provenance

- Local experiment branch: `exp/openduckmini-statistical-resume`.
- Frozen protocol commit: `0818f871a4498b054dc2d7b068ebbe362f8186d5`.
- Main HEAD before branch: `334a0907750b2a56bf226bafc290966d061e3b4c`.
- Shared branch HEAD after execution: `cac17ed64cf7d8d5ece84764d1f8593c8c109fe3`;
  unrelated exp_013 commits `97d0358` and `cac17ed` were appended by other
  workspace activity after protocol freeze.
- WSL training source: `338451f33e687ea3edcda8a2c2cdcbc8a7b4bda0`.
- v52 parent actor/critic/normalizer match the parent used by v60.
- Parent has no optimizer state; fresh Adam would be required for null
  continuation.

No push or tag was created.

## 5. Checkpoint Payload Round-Trip

Result: **`CHECKPOINT_PAYLOAD_BIT_EXACT_PASS`**.

| Component | Before | After |
| --- | --- | --- |
| complete training state | `d140cd2e...3512` | `d140cd2e...3512` |
| randomized MJX model | `c42fd740...f83f` | `c42fd740...f83f` |
| serialized payload self-check | `0f58512c...aaf1` | `0f58512c...aaf1` |

All numeric leaves, shapes, dtypes, PyTree paths, and canonical bytes matched.

## 6. Statistical Resume Protocol

- U: four uninterrupted updates, 20 independent processes.
- R: two updates, full checkpoint save, process exit, fresh process/load, two
  updates; 20 trials.
- Pair ordering alternated U/R and R/U.
- Every process warmed and synchronized a disposable state, then reloaded the
  measured checkpoint.
- Every trial began with the same state hash and saved RNG.
- All trials used old objective and identical source/config.

One update here is a harness update boundary; the frozen diagnostic profile
executes four Adam minibatch steps per boundary. Each endpoint therefore
contains four harness updates / sixteen Adam steps.

## 7. Uninterrupted Distribution

U completed 20/20 with zero crash/nonfinite/fall/termination.

Selected means:

| Metric | U |
| --- | ---: |
| actor delta L2 | 0.399084 |
| critic delta L2 | 0.630757 |
| Adam first moment norm | 0.237289 |
| tracking RMSE | 0.726843 |
| two-step mean return | -0.571019 |

## 8. Resumed Distribution

R completed 20/20 with zero crash/nonfinite/fall/termination.

Selected means:

| Metric | R |
| --- | ---: |
| actor delta L2 | 0.398687 |
| critic delta L2 | 0.636042 |
| Adam first moment norm | 0.238913 |
| tracking RMSE | 0.733773 |
| two-step mean return | -0.604351 |

R required two process initializations and averaged 69.5 seconds versus 35.2
seconds for U. Wall time is not a performance endpoint.

## 9. Statistical Resume Decision

**`STATISTICAL_RESUME_FAIL`**.

Passing checks:

- 20 completed trials per mode;
- zero nonfinite values;
- equal crash rate;
- actor delta median ratio `0.997985`;
- critic delta median ratio `1.012296`;
- fall and termination difference CI upper bounds `0`;
- command-distribution TV `0`;
- no resumed-only failure class.

Failing check:

- 10/12 primary metrics exceeded `|standardized effect| <= 0.25`.

Notable effects were actor delta `-0.540`, critic delta `+0.492`, Adam second
moment `-0.606`, entropy `+0.582`, two-step return `-0.614`, and tracking RMSE
`+0.443`. The return and Adam-second-moment difference CIs excluded zero.

## 10. Null Continuation Protocol

The registered protocol was 3 logical seeds × 2 backend replicates × 250k
interactions, conditional on `STATISTICAL_RESUME_PASS`.

That condition was not met. Status: `NOT_RUN`.

## 11. Command Exposure

Statistical resume exposure was identical between modes:

- total variation of the 625-bin `P(vx,vy,yaw,head)` distribution: `0`;
- samples per mode: 640;
- formal command matches: 0;
- `OFF_GRID`: 640.

This is consistent with a continuous training sampler. Null-continuation
exposure was not generated.

## 12. Optimizer and Gradient Dynamics

Resume shifted optimizer distributions despite bit-exact loaded payload:

- Adam first moment mean: `0.237289 -> 0.238913`;
- Adam second moment mean: `0.000554185 -> 0.000551286`;
- critic delta mean: `0.630757 -> 0.636042`.

The audit establishes a mode-associated distribution shift but does not prove
whether the immediate cause is process runtime state, compilation/runtime
history, or post-boundary GPU batch scheduling.

## 13. 50k Results

Not run. Null continuation was blocked before training.

## 14. 100k Results

Not run. Null continuation was blocked before training.

## 15. 250k Results

Not run. No null-continuation checkpoint was created.

## 16. Parent Retention

Not evaluated. No claim can be made that old-objective continuation maintains
or degrades v52.

## 17. Seed and Backend-Replicate Variance

The resume experiment quantified 20 process replicates under one logical seed.
It found measurable U/R mode effects beyond the registered bound.

The planned three-seed/two-replicate null experiment was not run, so
within-seed versus between-seed retention variance is unavailable.

## 18. Null Continuation Decision

`NOT_RUN_DUE_TO_STATISTICAL_RESUME_FAIL`.

Neither `NULL_CONTINUATION_STABLE` nor `NULL_CONTINUATION_UNSTABLE` is assigned.

## 19. v60 Postmortem Implications

Classification remains **`UNRESOLVED`**.

The statistical resume failure shows that a fresh-process checkpoint boundary
can shift short-run optimizer/rollout distributions. v60 Arm C did not retain
the telemetry needed to test whether that mechanism occurred there, and the
null fresh-Adam continuation was not authorized. Therefore v60 degradation
cannot be classified as fresh-optimizer, objective-specific, command-exposure,
or seed-sensitive failure from this evidence.

## 20. Next Allowed Single Change

Run one diagnostic factor only:

> Compare update-boundary in-process save/load against the existing
> fresh-process R mode using the same checkpoint bytes.

This isolates the process boundary/runtime state from serialization. Reward,
optimizer, PPO, environment, and trial budget must remain fixed. Null
continuation and reward redesign remain blocked.

## 21. Explicit Non-Claims

- No null-continuation policy was trained.
- No 50k/100k/250k evaluation was run.
- No conclusion about v52 retention or fresh Adam stability is made.
- No formal acceptance or real-robot evaluation was run.
- No policy was promoted.
- `clip_fraction` and episode-return standard deviation were not emitted by the
  frozen harness telemetry; they were not fabricated. The registered primary
  Gate used the available exact endpoint vector.
