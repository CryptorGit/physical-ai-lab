# OpenDuckMini v59 Corrected 15-Second Diagnostic

## 1. Executive Summary

**Diagnostic decision: `DIAGNOSTIC_COMPLETE`.** Both scheduled matrices
(19 commands × 5 seeds × 750 control steps) ran on GPU MJX and produced raw
logs, complete episode-start snapshots, command/seed manifests, failure
matrices, reward contributions, legacy comparisons, and counterfactual
surfaces.

Condition D had no termination in 95 episodes. Condition S had 28 falls, so
67/95 physical episodes survived 15 seconds; terminated episodes were
truncated at first termination for metric aggregation. Walking failure is a
diagnostic result, not an incomplete-evaluator condition.

The three central findings are:

1. legacy reverse-static behavior was partly evaluation-induced, but corrected
   reverse-straight remains severely deficient;
2. yaw-only overshoot is real and stationary at roughly 2.15–2.26× in D;
3. legacy maximum reverse-right 5/5 falls do not reproduce in D (0/5), but one
   S seed still falls (1/5).

## 2. Scope and Non-Claims

This is not formal acceptance. Every output is:
`formal_acceptance_eligible=false`, `enough_episodes=false`, and
`diagnostic_only=true`. The required 20 seeds × 30 seconds were not run.
v59 remains `diagnostic_not_qualified`, cannot be promoted, and cannot be
deployed to hardware.

No CPU trajectory is mixed into these results. No training, optimizer update,
reward/teacher/sampler/curriculum/XML change, legacy rerun, or production-code
change was performed.

## 3. Checkpoint and Runtime Provenance

The actor is the GPU JAX checkpoint path at v59 step 33,423,360. ONNX was used
only for provenance.

| Item | SHA-256 |
| --- | --- |
| checkpoint tree | `4e522903cfb3edf8dacfc2f5dc5b9510746711360748440c54097483f0ac38f1` |
| diagnostic ONNX | `cb5e70d999a17abcad007abae1b7d81e494bcf673dd24dbd17260267b56b7763` |
| calibrated/backlash scene | `9819c81c0eecbe751662a24029976bdb765da28e0a8f5a9854470a0611fee73f` |
| command manifest | `e3bbae7befcafdeb77479d2b3e252bec8f1c81cfdf05119959d388794029399d` |
| checkpoint normalizer | `6451894e65924401f234cbe16782fee4ad6ecea980ea146bf558411248e2749c` |
| teacher configuration | `4f16fdbe5976ebb941f6f674229417c6eaedacc2f2931e95ccbdacd8687ba703` |
| reward configuration | `15ac018d9d93615db31150a5537daa815097c44e4b92d14ef7763d1014668cb3` |
| evaluator | `a826e5914392ebbabfa619b2155f5bafb76a094f0e6260c18088e08df5229476` |
| resolved evaluation manifest | `154528286c32e1346536377b8d754f71515f79ae597dd0d8eb5d0eb331abd04b` |

Runtime was Python 3.12.3, JAX/jaxlib 0.5.3, MuJoCo/MJX 3.11.0,
float32, x64 disabled, JIT enabled, matmul precision `highest`, one process,
and one RTX 5090 Laptop GPU. See `environment_manifest.json`.

## 4. Corrected Evaluation Pipeline

Both conditions use the calibrated/backlash scene, checkpoint observation
normalizer/order, native teacher routing, teacher + residual composition,
nonlinear scaling, head coupling, action history/delay, motor speed limit,
joint clamp, and ten MJX substeps.

Body commands are the frozen legacy 19 values in base/body coordinates. Four
head commands come from each seed's training-compatible reset command and stay
fixed for 15 seconds. `head_locked=false` is not treated as locomotion failure.
The acceptance requirement for zero head target conflicts with v59's training
contract.

Condition D uses a reference model, a fixed reset state, zero observation
noise, deterministic actor, delay index zero, and no push. Condition S uses
native reset/domain/noise/delay/stochastic-action sampling.

The S `push` is not an external force. At an episode-time interval of 5–10
seconds, the code adds a sampled 0.10–0.50 m/s vector to floating-base xy
velocity. It is impulse-like, occurs during the episode, and is not reset-only.
Its training provenance is resolved and enabled in S.

## 5. Condition D Results

All 95 episodes survived 15 seconds.

| Primary command class | Commands |
| --- | ---: |
| TRACKING_OK | 2 |
| LINEAR_UNDERSHOOT | 7 |
| YAW_OVERSHOOT | 5 |
| MIXED_FAILURE | 5 |

Only stand and maximum reverse-right were modal `TRACKING_OK`; one of the five
maximum reverse-right seeds still had yaw-overshoot as its per-seed class.
There were no falls, NaNs, or other terminations.

## 6. Condition S Results

There were 28 falls in 95 episodes across 12 commands; 67 episodes survived
15 seconds.

- 6 falls happened before 5 seconds, before any possible configured push.
- 9 falls had no push event.
- 19 falls occurred after at least one velocity impulse.

Thus S instability is not explained by push alone. It reflects sensitivity to
the combined training-compatible reset/model/noise/delay/stochastic-action
condition. This evaluation does not isolate those factors from one another.

## 7. 19-Command Failure Matrix

The complete 38-row D/S matrix is `failure_matrix.csv`. Diagnostic thresholds:

- no motion: linear speed `<0.02 m/s` and `|yaw|<0.05 rad/s`;
- yaw over/undershoot: response ratio outside 0.75–1.25;
- linear under/overshoot: projected speed ratio outside 0.6–1.4;
- lateral drift: orthogonal mean speed above `0.04 m/s`;
- command-level primary class is `FALL` if any seed falls.

These are diagnostic classifications, not acceptance gates.

## 8. Backward-Straight Analysis

The legacy path reported effective static behavior (`vx=-0.000262`, 5/5
no-motion proxy). Corrected D is not static: mean no-motion duration is only
0.032 seconds. It moves in the wrong mixture:

| Window/metric | vx | vy | yaw |
| --- | ---: | ---: | ---: |
| first 2 s | -0.00234 | +0.02108 | -0.20639 |
| 5–15 s | -0.01206 | +0.03263 | -0.16748 |
| full 15 s | -0.01110 | +0.03125 | -0.16827 |

The desired vx is `-0.10`. Teacher action is active (RMS 0.713), yet the
15-second speed ratio is only 0.111.

- `backward_start_failure`: yes—the first two-second mean has only 2.3% of
  requested backward speed despite transient threshold crossings.
- `backward_tracking_failure`: yes—5–15 seconds remains near 12% of command.
- `backward_directional_drift`: yes—lateral and yaw escape exceed the achieved
  backward component.

S shows a different pathology. Its first-two-second mean vx is `-0.0790`, so
some randomized starts initiate reverse propulsion, but all five reverse
episodes fall at 1.38, 1.76, 8.68, 9.66, or 11.24 seconds.

## 9. Yaw-Only Analysis

Yaw-only overshoot persists with teacher routing inactive:

| Condition | left mean / ratio | right mean / ratio | falls |
| --- | --- | --- | ---: |
| D | +1.291 / 2.152× | -1.358 / 2.263× | 0/10 |
| S, until termination | +1.075 / 1.791× | -1.109 / 1.849× | 6/10 |

It is not a startup transient. D 5–15 second means remain +1.293 and -1.358
rad/s. Left/right overshoot is approximately symmetric in magnitude.

Per-head-component correlations can be large with five samples, but head,
reset, model, delay, and noise covary by seed. No head-causality conclusion is
drawn.

## 10. Maximum Backward-Right-Turn Analysis

Legacy: 5/5 falls. Corrected D: 0/5 falls, mean vx `-0.0610`, mean yaw
`-0.3432`. Corrected S: 1/5 falls; the other four survive.

The S fall occurs at 4.18 seconds without a push. For that episode:

1. flight first appears at 2.20 s;
2. backward-speed error exceeds 0.2 m/s at 2.76 s;
3. `|pitch|` exceeds 0.3 rad at 3.64 s and 0.5 rad at 4.02 s;
4. terminal pitch is -0.908 rad and terminal vx is -0.818 m/s;
5. contact-foot-slip peak is 0.581 m/s, with transient joint/target-limit
   activation.

The evidence supports a mixed support/backward-tracking collapse. It does not
prove a unique initiating cause; early yaw error and saturation coexist.

## 11. Legacy vs Corrected Classification

- Reverse-straight complete static behavior: **mixed**. Missing teacher routing
  made legacy near-static, but corrected D retains severe tracking/drift.
- Yaw-only overshoot: **policy/objective-induced**. It persists under corrected
  routing and scene.
- Maximum reverse-right 5/5 falls: **mostly evaluation-induced**. D resolves
  them completely; S retains 1/5 stochastic instability.
- Legacy raw time series are unavailable, so onset/slip/support comparisons are
  marked insufficient rather than reconstructed.

Per-command labels are in `legacy_vs_corrected.csv`.

## 12. Reward-Term Contributions

`reward_term_summary.csv` contains mean, cumulative return, p05/p95, and
active-step ratio for all 21 active v59 terms. In D yaw-left:

- command_progress: +77.459 per step;
- command_yaw_error: -11.977;
- tracking_ang_vel: +0.863;
- yaw_translation: -3.981.

The raw environment metrics are already weighted. The evaluator's diagnostic
array multiplied `abs(weight)` a second time; analysis reverses this logged
factor and restores the cost sign before producing the canonical reward CSV.
A regression test checks both scale and sign.

There is no active base-height or vertical-velocity reward term in this
resolved v59 config. Those physical signals are logged but no nonexistent
reward contribution is invented. Actuator force is available; separate joint
torque and contact force are marked unavailable.

## 13. command_progress Counterfactual Surface

For ±0.6 rad/s yaw-only, holding empirical non-yaw contributions fixed:

| yaw ratio | tracking yaw | yaw error | command_progress | yaw-related total |
| ---: | ---: | ---: | ---: | ---: |
| 1.0× | 10.000 | 0.000 | 36.000 | 46.000 |
| 2.0× | 0.001 | -7.200 | 72.000 | 64.801 |
| 3.5× | ~0 | -45.000 | 126.000 | **81.000** |
| 4.0× | ~0 | -64.800 | 144.000 | 79.200 |

`command_progress` alone maximizes at the tested 4× boundary. All yaw-related
terms and empirical total maximize at 3.5×. The 2× state is favored over exact
1× by +18.801 per step before dt.

The three compound surfaces (forward+yaw, forward-left+yaw, and maximum
reverse-right) also select linear ratio 1.0 but yaw ratio 3.5 for total reward.
Full grids are in `command_progress_counterfactual.csv`.

## 14. Evaluation-Induced Failures

The largest evaluation-induced artifacts were:

- removal of reverse teacher routing, producing apparent complete reverse
  static behavior;
- non-calibrated scene/motor path, contributing to maximum reverse-right 5/5
  falls.

Correcting these does not make v59 successful; it changes the observed failure
from “static/no controller” to weak/drifting or unstable locomotion.

## 15. Policy-Induced Failures

Corrected D demonstrates persistent linear undershoot in forward, reverse,
lateral, and diagonal commands. Reverse-straight retains lateral/yaw escape.
S demonstrates broad instability: 28/95 falls, including nine without push.

These failures remain after controller/evaluator equivalence corrections and
are therefore not legacy-routing artifacts.

## 16. Objective-Induced Failures

Yaw overshoot has direct mathematical and trajectory evidence:

- corrected D runs at 2.15–2.26× for 15 seconds;
- the unbounded dot-product contribution makes 2× better than 1×;
- the resolved yaw-objective grid peaks at 3.5×.

This is the highest-confidence objective-induced failure.

## 17. Remaining Unknowns

- Historical checkpoint episodes cannot be reconstructed.
- S combines reset, domain, noise, delay, stochastic actor, and push; this
  matrix does not assign the 28 falls to a single source.
- Five correlated head samples cannot establish head causality.
- The single maximum reverse-right S fall has multiple precursors.
- Contact force and separate joint torque were not available.

## 18. Recommended Next Single Change

Run at most one pilot:

- parent: v52 adopted package actor checkpoint `v45_step_47349760`
  (`52205E…F3C`) with frozen v52 reverse profiles;
- change: only replace the unbounded `command_progress` yaw dot-product with a
  command-centered bounded yaw objective;
- fixed: scene, observation, teacher, sampler, curriculum, network, optimizer
  hyperparameters, head routing, and evaluation manifests;
- budget: at most 5M interactions, with a 1M diagnostic checkpoint;
- gate: D yaw response ratios 0.8–1.2 in both directions, no D falls, and no
  stop/linear degradation;
- No-Go: ratio remains >1.25, any D fall, or material linear/fall regression.

No second simultaneous change is recommended. The reverse and stochastic
failures need a separately isolated follow-up after the yaw objective test.

## 19. Explicit Non-Claims

This report does not claim formal acceptance, robustness qualification,
hardware readiness, v59 promotion, or that one reward change will solve all
failures. It does not use the term “strong disturbance evaluation” for this
matrix.

