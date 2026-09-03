# v59 Evaluation Equivalence Parity Report

## 1. Executive Summary

**Deterministic evaluation-equivalence gate: PASS.**  For v59 step
33,423,360, identical raw observations produced matching deterministic actor
outputs in JAX, an independent NumPy MLP, and ONNX Runtime.  An independent
NumPy implementation of v59 action composition produced the same final motor
targets as the historical MJX environment.

The legacy formal result is not a valid measurement of the training control
path.  Its first structural divergence is the non-calibrated scene; for
reverse commands, the first command-dependent controller divergence is the
teacher-routing gate.  The corrected 2-second smoke removes the legacy static
contact pattern in reverse, but it still has poor reverse tracking and large
yaw drift.  Yaw-only overshoot remains.  The 15-second maximum
backward-right fall is unresolved because this phase intentionally ran only
the authorized 2-second diagnostic smoke.

This PASS applies only to deterministic control-path equivalence.  It is not a
performance qualification, v59 remains `diagnostic_not_qualified`, and it is
not eligible for hardware.

## 2. Checkpoint and ONNX Provenance

| Artifact | Path / identity | Evidence |
| --- | --- | --- |
| Training/evaluation checkpoint | `/home/user/openduck_training_runs/omnidirectional_finish_v59_40m/2026_07_30_031556_33423360` | checkpoint step and audit tree SHA-256 `4e522903cfb3edf8dacfc2f5dc5b9510746711360748440c54097483f0ac38f1` |
| Adjacent evaluation ONNX | same path plus `.onnx` | SHA-256 `cb5e70d999a17abcad007abae1b7d81e494bcf673dd24dbd17260267b56b7763` |
| Video diagnostic ONNX | `artifacts/omnidirectional_v59_step_8355840_diagnostic.onnx` | different step and artifact; not used here |

The checkpoint actor has input 101, hidden layers 512/256/128 with swish,
and output 28 (14 location plus 14 scale parameters).  ONNX returns
`tanh(location)`.  Numerical equivalence against checkpoint inference over
500 observations, including joint-wise errors, independently confirms that
the adjacent ONNX contains the checkpoint actor weights and normalizer to
float32 precision.  See `comparison_tables/actor_parity.csv` and
`actor_joint_max_error.csv`.

The historical source file used by the run was
`/home/user/openduck_training_backward_v23_20260729/playground/open_duck_mini_v2/joystick.py`;
its SHA-256 is
`e9aff77b249dacc937a6d06caacc4c1450f250b9d413792b30890019510ada41`.
`PYTHONPATH` is explicitly fixed in `tools/run_v59_parity_wsl.sh`.  An initial
attempt without that setting resolved a different editable checkout and was
discarded before producing accepted results.

## 3. Training Control Pipeline

| Stage | Implementation | Input → output | dtype / frame / processing |
| --- | --- | --- | --- |
| Command | `Joystick.step`, `sample_command` | 7 → 7 | float32; body `vx,vy,yaw`, four head commands; no ramp |
| Reverse routing | `Joystick.step` | command → boolean | active exactly when `vx < -0.02` |
| Teacher phase | `_get_backward_parameters`, `step` | scalar phase → scalar + 2-D sin/cos | periodic; yaw selects/blends left/right profiles |
| Teacher action | `_get_optimized_backward_reference` | phase,yaw → reference frame, then 14 targets | actuator order; reverse only |
| Observation | `_get_obs` | MJX data/info → state 101 | gyro 3, accel 3, command 7, backlash-added joint offset 14, joint velocity 14, three action histories 42, previous target 14, contacts 2, phase 2 |
| Normalization | Brax `running_statistics.normalize` | 101 → 101 | checkpoint Welford mean/std; float32; no clamp argument |
| Actor | Brax policy network | 101 → 28 → 14 | swish; tanh deterministic mode |
| Direct composition | `Joystick.step` | actor 14 → target 14 | nonlinear calibrated directional spans |
| Reverse composition | `Joystick.step` | teacher 14 + actor 14 → target 14 | residual scale 0.12, yaw-dependent; head indices 5:9 |
| Delay | `Joystick.step` | action history → action 14 | training range indices 0..2; deterministic trace fixed to index 0 |
| Motor limits | `Joystick.step` | target 14 → target 14 | velocity clamp, coupled head envelope, measured joint limits |
| Physics | `mjx_env.step` | target 14 → state | calibrated/backlash XML; control 0.02 s, simulation 0.002 s |

The exact serialized trace fields and full arrays are in five NPZ files under
`golden_traces/`.  Each has 100 rows.  Observation shape is `(100,101)`;
qpos is `(100,31)`.  No aggregate-only substitution was used.

## 4. Legacy Evaluation Pipeline

The legacy path is documented stage-by-stage in
`training_vs_legacy_pipeline.md`.  It uses
`scene_flat_terrain.xml`, synthesizes a 7-D observation from a 3-D command,
has default positive-yaw lateral compensation, and does not reproduce the
training delay/backlash path.

Most importantly, `OfficialPolicyEvaluator.calibrated_hardware` is derived
from the scene filename.  Because the formal JSON records the non-calibrated
scene, the reverse periodic teacher branch is disabled and actor output is
treated as a standalone action.

## 5. First Divergence Point

The first unconditional divergence is **scene selection**, before observation
construction:

```text
training: scene_flat_terrain_backlash_calibrated.xml
legacy:   scene_flat_terrain.xml
```

For `C2` and `C4`, the first command-dependent controller divergence is
**teacher routing**:

```text
training: vx < -0.02 → optimized periodic teacher + actor residual
legacy:   non-calibrated scene → teacher disabled → actor standalone
```

There is no need to invoke a numerical tolerance to expose either divergence.

## 6. Actor Inference Parity

| Comparison | Maximum absolute error | Threshold | Result |
| --- | ---: | ---: | --- |
| JAX vs independent NumPy checkpoint MLP | `4.3958426e-7` | `1e-6` | PASS |
| JAX vs ONNX Runtime | `6.8545341e-7` | `1e-5` | PASS |

There was no threshold-crossing step.  The largest per-joint ONNX error was
`6.8545341e-7`; all 14 joint results are in
`comparison_tables/actor_joint_max_error.csv`.  Shapes are 14 in all three
paths and the order is the calibrated actuator order listed in that CSV.

The audit harness initially omitted `running_statistics.normalize` when
constructing the JAX inference network and correctly failed at C0 step 0.
That harness result was rejected.  The corrected construction matches
Brax `ppo.train`, and the final artifact was regenerated from scratch.

## 7. Teacher Routing Parity

`C2_backward` and `C4_backward_right_max` have `teacher_mode=1` for all 100
steps; C0, C1, and C3 have it disabled.  Reverse teacher action norms are
nonzero:

| Command | min norm | max norm |
| --- | ---: | ---: |
| C2 backward | 2.154133 | 3.091997 |
| C4 backward-right | 2.304048 | 3.116572 |

The corrected evaluator intentionally invokes the immutable historical
teacher generator rather than copying its gait data into a second production
implementation.  Routing, phase, and the full teacher array are serialized at
each step.  The independent comparison begins at residual composition.

## 8. Action Composition Parity

`tools/v59_parity_common.py` independently reproduces directional action
scaling, reverse residual scales, speed limiting, coupled-head limiting, and
joint clamps using NumPy.  Across 500 teacher-forced control states, its final
target differs from the historical MJX environment by at most
`6.0796738e-8`, below the `1e-6` gate.

For reverse commands, actor output is never used as the standalone target.
`teacher_action`, `actor_residual_scaled`, `combined_action_pre_clip`, and
`combined_action_post_clip` are all retained in the NPZ traces.

## 9. Motor Target Parity

| Quantity | Result |
| --- | --- |
| Maximum absolute motor-target error | `6.0796738e-8` |
| Required threshold | `1e-6` |
| First failing step | none |
| Joint order | 14-joint calibrated actuator order |

This comparison is teacher-forced: the composition calculation consumes the
same command, phase/reference, previous target, and actor output as the MJX
step.  It therefore isolates controller processing from the subsequent
physics backend.

## 10. Deterministic Smoke Results

Five commands × one fixed seed × 100 steps (2.0 s) completed with no NaN/Inf
and no immediate fall.  Teacher routing was active only for C2 and C4.
`smoke_results/training_equivalent_5x1x2s.csv` is diagnostic metadata, not
formal evaluation.

| Command | mean vx | mean vy | mean yaw | fall |
| --- | ---: | ---: | ---: | --- |
| C0 stand | 0.004665 | 0.000906 | -0.000176 | no |
| C1 forward | 0.055018 | 0.002268 | -0.009744 | no |
| C2 backward | -0.008570 | 0.037158 | -0.290734 | no |
| C3 yaw +0.6 | 0.095137 | -0.034976 | 1.239695 | no |
| C4 back-right -0.3 | -0.041663 | 0.018860 | -0.172533 | no |

## 11. Legacy vs Corrected A/B Results

The A/B table is
`comparison_tables/legacy_vs_training_equivalent_5x1x2s.csv`.  Both runs use
the semantic home pose, zero initial joint noise, zero initial base velocity,
the same ONNX/checkpoint weights, five commands, and 2 seconds.  Exact numeric
state identity is impossible because the XMLs have different qpos dimensions
and the backends differ; this limitation prevents interpreting small dynamic
differences as controller errors.

- Backward straight: legacy mean `vx=-0.0000046` with full double support;
  corrected mean `vx=-0.00857`, with substantial lateral/yaw motion.  The
  exact static behavior is legacy-path induced, but correct reverse tracking
  is still absent.
- Yaw-only: legacy `1.1455` and corrected `1.2397 rad/s` for a `0.6 rad/s`
  command.  Overshoot remains after routing correction.
- Maximum backward-right: neither path falls within 2 seconds.  No claim about
  the reported 15-second fall is possible from this smoke.

## 12. Evaluation-Induced Failures

Confirmed evaluation-induced:

- omission of reverse teacher+residual routing;
- use of non-calibrated/non-backlash scene;
- legacy reverse-straight static/full-support pattern in the 2-second A/B.

The old 5×15-second measurements combine those differences with policy and
physics behavior and must not be called formal v59 policy performance.

## 13. Policy-Induced Failures

Confirmed under the training-equivalent deterministic path:

- yaw-only overshoot (mean `1.2397` for command `0.6`);
- poor backward-straight tracking (`vx=-0.00857` for command `-0.1`) plus
  unintended yaw/lateral motion.

These observations diagnose retained behavior; they do not authorize reward,
network, teacher, sampler, or curriculum changes in this phase.

## 14. Remaining Unknowns

- Whether maximum backward-right falls under corrected 15-second or
  30-second evaluation.
- Whether seeded stochastic streams can be made bit-identical across JAX and
  CPU MuJoCo.  Training uses JAX PRNG splits for action delay, IMU/noise,
  pushes, reset pose/velocity, and command sampling; legacy CPU evaluation
  does not implement the same source or sample order.
- Backlash/delay stochastic parity.  Deterministic backlash state is included,
  but randomized delay/noise was intentionally excluded from the golden gate.
- Long-horizon performance, recovery, external-force robustness, and formal
  acceptance.  None were run.

## 15. Next Gate

The next permitted evaluation-only gate is a separately approved,
training-equivalent horizon test that first records stochastic source, seed,
sample order, and values.  It should test maximum backward-right long enough
to classify the old fall and use 20×30 seconds only when moving to a formal
qualification phase.

No learning experiment should start from this parity result.  If a later
performance phase is authorized, the smallest evidence-supported isolated
experiment is the previously proposed yaw/progress objective correction, with
all routing and scene choices fixed by this harness.  v52 remains the adopted
version meanwhile.
