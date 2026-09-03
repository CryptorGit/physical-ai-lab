# v59 Stochastic / Seeded Evaluation Parity Report

## 1. Executive Summary

**Overall stochastic gate: FAIL.**

The required distinction is:

```text
controller stochastic sample-injection parity: PASS
historical checkpoint-time episode reconstruction: FAIL
native same-backend closed-loop bit reproducibility: FAIL
stochastic 5×3×2 s wiring smoke: FAIL (one termination)
```

All 15 sample-injection cases reproduced the noisy observation, normalized
observation, JAX actor sample, delay-buffer output, teacher composition and
motor target within their thresholds.  There is no sample-injection
controller divergence.

Nevertheless, two independently executed GPU MJX closed loops initialized by
the same JIT reset, same JAX keys and same injected policy samples differ from
the first physics step.  One C2 backward case terminates at 0.98 seconds.
These violate the stated overall PASS conditions.  v59 remains
`diagnostic_not_qualified`, prohibited from hardware and prohibited from
promotion.

## 2. Scope and Non-Claims

The audit uses five commands, environment indices 0–2 derived from training
master seed 0, and 100 control steps (2 seconds).  It covers stochastic
controller and environment wiring only.

It does not claim walking success, robustness, hardware readiness, exact
reconstruction of an episode active at checkpoint time, or equivalence of
JAX, NumPy and MuJoCo random-number bit patterns.  The prior deterministic
artifacts were not modified.

## 3. Checkpoint Provenance

| Artifact | Identity |
| --- | --- |
| checkpoint | `/home/user/openduck_training_runs/omnidirectional_finish_v59_40m/2026_07_30_031556_33423360` |
| checkpoint tree SHA-256 | `4e522903cfb3edf8dacfc2f5dc5b9510746711360748440c54097483f0ac38f1` |
| ONNX | checkpoint path plus `.onnx` |
| ONNX SHA-256 | `cb5e70d999a17abcad007abae1b7d81e494bcf673dd24dbd17260267b56b7763` |
| historical `joystick.py` SHA-256 | `e9aff77b249dacc937a6d06caacc4c1450f250b9d413792b30890019510ada41` |

No checkpoint, ONNX, normalizer, teacher or production evaluator was changed.

## 4. Random Source Inventory

The complete machine-readable inventory is `random_source_manifest.json`.
Confirmed active source groups are:

1. PPO tanh-normal policy sampling: 14 standard-normal values per rollout
   control step.
2. Fixed per-environment model randomization: floor friction, actuated
   frictionloss, armature, torso inertial position, all body masses, torso
   mass offset, model qpos0 and actuator kp/bias.
3. Reset randomization: base x/y, base yaw, 14 actuator positions and six base
   velocity components.
4. Reset command/head sampling and push-interval sampling.
5. Per-step action delay index.
6. Per-step push direction and magnitude; their samples are drawn every step
   even when the interval gate is false.
7. Observation noise: gyro, accelerometer, gravity, joint position and joint
   velocity.
8. IMU gravity-history delay index.
9. Per-step command candidate sampling.  The candidate is computed but not
   applied during this 100-step trace.

Confirmed absent or deterministic:

- no actuated-joint initial qvel noise;
- no random teacher phase initialization;
- no random backlash initialization;
- no sampled command duration;
- no environment-level action noise or motor noise;
- no terrain geometry sampling;
- no external-force sampler—the “push” directly changes base xy qvel;
- action and IMU buffers start as zeros.

## 5. Seed Derivation

The v59 launch omitted a seed override, so Brax PPO used master seed 0:

```text
PRNGKey(0)
→ split(global_key, local_key)
→ fold_in(local_key, process_id=0)
→ split(next_local, key_env, eval_key)
→ split(key_env, 4096)
→ environment key by environment_index
```

The domain randomizer and reset independently receive the same environment
key array.  This creates correlated, but separate, functional split lineages.
The audit uses environment indices 0, 1 and 2.

Policy rollout keys at checkpoint time are unavailable.  Audit policy keys
are explicitly namespaced with
`fold_in(environment_key, 0x59334233)` and are not represented as historical
samples.  See `seed_derivation.md`.

## 6. Sampling Order

Reset order:

```text
xy → yaw → actuator qpos → base qvel → command(8 child keys)
→ push interval → gyro → accel → gravity/IMU-index(shared child key)
→ joint position noise → joint velocity noise
```

Control-step order:

```text
PPO policy sample
→ split(info.rng, retained/push-theta/push-magnitude/action-delay)
→ update/select action buffer
→ optional qvel push
→ teacher/direct action composition
→ speed/head/joint limits
→ MJX physics
→ gyro → accel → gravity/IMU-index
→ joint position noise → joint velocity noise
→ split(retained next RNG, command candidate key)
```

Adding a parent-stream draw shifts all subsequent environment samples.
`sample_command` child draws do not mutate the retained parent key.

## 7. Sample-Injection Architecture

`StochasticInputs` and the pure NumPy replay primitives live in
`tools/v59_stochastic_common.py`.  The evaluator does not draw replacement
values.  It consumes:

- recorded initial qpos/qvel offsets;
- recorded 101-D additive observation-noise vector;
- recorded 14-D policy standard-normal sample;
- recorded integer delay index and 42-element buffer;
- recorded backlash qpos state and teacher phase.

Each injected value is logged in the trace metadata or JSONL event stream with
source ID, sample index, value, shape, dtype, timing and consumer.

## 8. Initial-State Parity

Injection reconstructs:

```text
qpos = home_qpos + recorded_qpos_offset
qvel = zero_qvel + recorded_qvel_offset
```

Maximum error across all 15 cases is exactly `0.0`.  Base linear/angular
velocity and actuator position perturbations are stored in every NPZ.  The
initial phase is deterministically zero and the 42-element action buffer and
9-element IMU buffer are zero-filled.

## 9. Observation-Noise Parity

Actual order:

```text
physical state/sensors
→ body/sensor coordinate extraction
→ deterministic accelerometer x bias +1.3
→ actuator qpos + passive backlash qpos
→ additive sensor/joint noise
→ assemble 101-D observation
→ checkpoint normalization
```

Noise is before normalization and there is no pre-normalization clamp.

| Comparison | Maximum error | Gate |
| --- | ---: | ---: |
| injected noisy observation | `5.9604645e-8` | `1e-6` |
| normalized observation | `9.5367432e-7` | `1e-6` |

Gravity noise and its delay buffer are updated, but projected gravity is not
part of this 101-D actor observation.  It therefore has no actor consumer in
v59, despite consuming RNG and updating `imu_history`.

## 10. Teacher and Actor Parity

The injected JAX policy standard-normal sample reproduces the sampled actor
residual exactly: maximum error `0.0`.

The prior deterministic phase already proved JAX versus independent NumPy
actor location output.  Recomputing the stochastic scale and action entirely
in NumPy produces up to approximately `0.0044` action difference because
float32 logits/softplus error is amplified by the sampled scale; this is
retained as `actor_numpy_diagnostic_max_abs_error`, not hidden by a tolerance
and not used as the training-compatible JAX injection gate.

Teacher phase maximum float32 reconstruction error is `4.7683716e-7`
(`<=1e-6`).  Teacher action uses the immutable historical generator and is
serialized in full.  Reverse routing is active for C2 and C4 only.

## 11. Delay-Buffer Parity

- Unit: control steps, not simulator substeps.
- Sample: one integer shared by all 14 joints every control step.
- Values: 0, 1 or 2; `maxval=3` is exclusive.
- Buffer: 3 rows × 14 joints, initialized to zero.
- Update: roll one action row, prepend current actor action, then select row.
- Application: before nonlinear direct scaling or reverse
  teacher+residual composition, and therefore before clipping.

Observed over 1500 steps:

| delay | count |
| ---: | ---: |
| 0 | 525 |
| 1 | 500 |
| 2 | 475 |

All delay buffers and selected delayed actions match exactly.  The explicit
`a0..a4` unit test verifies expected sequences for delays 0, 1 and 2.

## 12. Backlash-State Parity

Backlash is not a controller hysteresis object and has no random direction
state.  The calibrated XML inserts ten passive hinge joints, nominally limited
to ±0.0087266463 rad, with damping 0.01, zero frictionloss and armature 0.01.

The controller order is:

```text
delayed actor action → target composition/clamp → motor target
→ MJX physics containing passive backlash joints
```

Backlash does not transform the motor target; before/after target difference
is exactly zero.  Its qpos is added to the corresponding actuator qpos before
joint-position observation noise.  Full qpos and sign of backlash qvel are
recorded as state/direction diagnostics.

The maximum absolute backlash qpos observed is `0.03501116 rad`, exceeding the
nominal joint range during the stochastic trajectory.  This is a physics
constraint/state observation, not a separate randomized controller state.
It is not silently clipped in the audit.

## 13. Motor-Target Parity

The injected path independently applies:

```text
delay selection
→ calibrated nonlinear action scaling or reverse teacher+residual
→ motor velocity limit
→ coupled head envelope
→ measured joint limits
```

Maximum combined-action and final motor-target error are both
`1.1920929e-7`, below the respective `1e-5` and `1e-6` gates.  There is no
command/seed injection divergence.

## 14. Native Seed Reproducibility

PRNG and reset diagnostics:

- same JAX key produces identical samples;
- the same JIT-compiled reset called twice gives exact qpos, qvel, observation
  and retained RNG (`max error 0`);
- sample counts, shapes, parameters and timing are stable.

Closed-loop result:

- repeated GPU MJX execution from those identical states and injected samples
  is not bit-exact;
- first difference is at physics step 0 for all 15 cases;
- divergence grows through the closed loop, reaching maximum differences
  `qpos=1.0318`, `qvel=10.7322`, observation `95.7225` and motor target
  `0.8479`.

The XLA autotuner logged rejected candidate kernels with NaN during
compilation.  This is recorded in `run.err`, but the audit does not claim it
is the cause of runtime non-reproducibility.  The demonstrated fact is the
step-0 bit inequality under the current GPU MJX execution.

Different PRNG libraries were not required to match JAX values.

## 15. Stochastic Smoke Results

All 15 cases contain complete metadata and no NaN/Inf.  Fourteen complete two
seconds without termination.  `C2_backward`, environment index 1 terminates
at control step 49 (`0.98 s`).

This is not a success-rate estimate.  It is a wiring smoke failure under the
specified “no immediate fall” gate.  Reference velocities are retained only
as diagnostics in `stochastic_command_results.csv`.

## 16. First Divergence Points

Sample-injection controller comparison:

```text
initial state                 PASS
observation after noise       PASS
normalized observation        PASS
teacher phase/action          PASS
actor residual                PASS
delay buffer/action           PASS
backlash state injection      PASS
combined action               PASS
motor target                  PASS
first divergence              none
```

Native closed-loop rerun:

```text
JAX reset and PRNG samples    PASS, exact
first GPU MJX physics step    FAIL at step 0
subsequent controller state   diverges as a downstream consequence
```

## 17. Missing Historical Provenance

The checkpoint does not contain:

- checkpoint-time MJX `env_state`;
- 4096 retained `info.rng` keys;
- reset generation/count or episode count;
- environment position within the 1000-step episode;
- PPO rollout sampling keys;
- checkpoint-time domain-randomized model assignment.

Therefore the episode being executed at historical training step 33,423,360
cannot be reconstructed.  The audit reconstructs the documented seed
derivation for a fresh reset only.

## 18. PASS / FAIL Decision

| Gate | Decision |
| --- | --- |
| random-source inventory | PASS |
| sample generation/order/consumer | PASS |
| initial-state injection | PASS |
| observation/noise/normalization injection | PASS |
| teacher/actor/delay/backlash/motor injection | PASS |
| controller stochastic parity | **PASS** |
| native same-backend exact closed-loop reproduction | **FAIL** |
| 5×3×2-second no-termination smoke | **FAIL** |
| historical episode reconstruction | **FAIL** |
| overall stochastic parity | **FAIL** |

The result is not “all stochastic processing failed.”  The controller path is
reproducible when samples and state are injected.  The overall gate fails on
native closed-loop exactness, one early termination and missing historical
state provenance.

## 19. Next Allowed Phase

Do not begin a performance or training experiment.  The next allowed work is
a minimal evaluation-infrastructure diagnostic:

1. repeat a single MJX physics step from a serialized identical state/action
   on CPU and GPU separately;
2. identify the first differing MJX field before closing the loop;
3. determine whether deterministic GPU/XLA settings or CPU evaluation can
   provide same-backend replay without changing the training controller.

Only after native replay is stable should the 5×3×2-second stochastic wiring
gate be repeated.  No 15-second, 20×30-second, external-force or hardware
evaluation is authorized by this result.
