# Statistical Resume Equivalence Report

## Decision

`STATISTICAL_RESUME_FAIL`

The checkpoint payload round-tripped bit-exactly and all 40 trials completed
without crash, nonfinite value, fall, or termination.  Resume nevertheless
failed the pre-registered distribution Gate because 10 of 12 primary metrics
had `|standardized mode effect| > 0.25`.

## Protocol

- Initial state: the same four-environment complete checkpoint for every trial.
- U: four uninterrupted optimizer updates, 20 fresh processes.
- R: two updates, state-complete save, process exit, fresh process/load, two
  updates; 20 trials.
- Ordering: pair 0 U/R, pair 1 R/U, alternating thereafter.
- Every process compiled on a disposable loaded state, synchronized, reloaded
  the measured state, then ran.
- Logical seed and serialized RNG were identical in all trials.
- The four-environment profile retains the production GPU JIT, domain-model
  vmap, per-environment randomized MJX model, native batched physics, and PPO
  update path. It is not a production-scale or performance run.

Here, one registered “update” is one harness update boundary. With the frozen
identity profile it contains four Adam minibatch steps, so each endpoint is
four harness updates / sixteen Adam steps.

## Completion and integrity

| Item | U | R |
| --- | ---: | ---: |
| Trials completed | 20/20 | 20/20 |
| Crashes | 0 | 0 |
| Nonfinite count | 0 | 0 |
| Fall rate | 0 | 0 |
| Termination rate | 0 | 0 |

All 40 measured initial state hashes equal
`d140cd2ee858d7c321ffbd11a8fe2ffdc764a9131f16c4e39774e79063233512`.

## Main distribution differences

| Metric | U mean | R mean | R-U | Standardized effect | 95% bootstrap CI of R-U |
| --- | ---: | ---: | ---: | ---: | --- |
| actor delta L2 | 0.399084 | 0.398687 | -0.000397 | -0.540 | [-0.000830, 0.000058] |
| critic delta L2 | 0.630757 | 0.636042 | +0.005286 | +0.492 | [-0.001274, 0.011727] |
| Adam first moment | 0.237289 | 0.238913 | +0.001624 | +0.420 | [-0.000739, 0.003918] |
| Adam second moment | 0.000554185 | 0.000551286 | -0.000002900 | -0.606 | [-0.000005771, -0.000000050] |
| policy loss | -0.002548 | -0.006863 | -0.004314 | -0.548 | [-0.009108, 0.000442] |
| value loss | 1.098547 | 1.113493 | +0.014947 | +0.519 | [-0.002090, 0.032113] |
| approximate KL | 0.0194266 | 0.0195058 | +0.0000793 | +0.485 | [-0.000199, 0.000363] |
| entropy loss | 0.0292464 | 0.0293577 | +0.0001113 | +0.582 | [0.000000005, 0.000233504] |
| two-step mean return | -0.571019 | -0.604351 | -0.033333 | -0.614 | [-0.064899, -0.000044] |
| tracking RMSE | 0.726843 | 0.733773 | +0.006930 | +0.443 | [-0.002754, 0.016285] |
| fall rate | 0 | 0 | 0 | 0 | [0, 0] |
| termination rate | 0 | 0 | 0 | 0 | [0, 0] |

The two-step return is a fixed-horizon diagnostic sum, not an episode return.

## Passing sub-Gates

- actor delta median R/U ratio: `0.997985`;
- critic delta median R/U ratio: `1.012296`;
- fall-rate difference CI upper bound: `0`;
- termination-rate difference CI upper bound: `0`;
- joint command distribution total variation: `0`;
- resumed-only failure classes: none.

All 640 samples per mode were `OFF_GRID`, as expected for the continuous
training sampler. Formal 19-command IDs were not used as the primary exposure
metric.

## Failure interpretation

Checkpoint bytes and loaded state are not the source of mismatch. The measured
fresh-process resume path changes the endpoint distribution beyond the
pre-registered effect-size bound, including statistically one-sided shifts in
Adam second moment, entropy loss, and two-step return. The experiment does not
yet distinguish process-boundary/runtime-state effects from the post-boundary
GPU batch schedule.

Null continuation is therefore prohibited. The next minimal diagnostic should
isolate one factor: compare an in-process update-boundary save/load against the
fresh-process R mode while keeping the same checkpoint bytes.
