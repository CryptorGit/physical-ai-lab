# v59 MJX First-Step Numerical Divergence Diagnostic

## 1. Executive Summary

**Classification: `BIT_EXACT_PASS`.** With the current v59 MJX runtime,
the isolated ten-substep physics call produced bit-identical output for D0,
D1a, and D2 across 20 reload-and-repeat measurements in one GPU process and
across two fresh GPU processes. Every numeric PyTree leaf, contact structure,
constraint count, termination decision, next normalized observation, and next
motor target matched exactly.

CPU and GPU are not bit-identical. Their differences are reported separately
and do not negate GPU repeatability. D1a has the largest CPU/GPU downstream
effect: next normalized-observation max error `0.762585` and next motor-target
max error `0.00588855`. Contact pairs and termination remain unchanged.

The prior native closed-loop failure therefore cannot be attributed to
nondeterminism in this isolated, fixed-input `mjx_env.step` call. This result
does not identify the earlier closed-loop divergence source.

## 2. Scope and Non-Claims

Measured cases were D0 (C0 STAND, environment index 0), D1a (C2 backward,
environment index 1, episode start), and D2 (C4 maximum backward-right-turn,
environment index 0). Each input contains the complete reset MJX Data tree,
the per-environment randomized Model, the recorded action and motor target,
and recorded stochastic samples.

D1b was not measured. The prior step-48 trace saved qpos/qvel but not complete
MJX Data (`qacc_warmstart`, contact/constraint solver state) or a step-local
complete randomized model. Filling those fields would be guesswork.

This is not a performance evaluation, historical episode reconstruction,
reward audit, policy improvement, deployment qualification, or evidence for
promoting v59.

## 3. Runtime Environment

GPU measurements used Python 3.12.3, JAX/jaxlib 0.5.3, MuJoCo/MJX 3.11.0,
float32 data, x64 disabled, JIT enabled, matmul precision `highest`, and one
RTX 5090 Laptop GPU. `XLA_FLAGS` was empty. The WSL driver report was
610.43.02 with Windows KMD 610.62 and CUDA UMD 13.3.

The complete relevant settings and source hashes are in
`environment_manifest.json`. Unrelated environment variables were not copied
because they can contain credentials and do not configure this physics path.

## 4. Serialized Input Provenance

The inputs were recreated from the recorded master-seed derivation and
domain-randomized model assignment used by the preceding stochastic parity
phase. They are new forward episode-start states, not historical states.

| Case | Source | Data SHA-256 | Model SHA-256 | Motor-target SHA-256 |
| --- | --- | --- | --- | --- |
| D0 | `C0_stand_seed0`, reset | `ac1bd1b1…ea0777` | `aee8da04…5994ab` | `1df25a5d…a87ce2` |
| D1a | `C2_backward_seed1`, reset | `d24db39b…2fe1f` | `b2985d4b…3133a5` | `b9c9f3ce…dcf843` |
| D2 | `C4_backward_right_max_seed0`, reset | `a37d8cc8…d28fce4` | `aee8da04…5994ab` | `29a26135…a61f0` |

Full hashes, action hashes, and domain-parameter hashes are in
`serialized_input_hashes.json`. The diagnostic reloads each pickle for every
measurement, so no preceding output can mutate the next input.

## 5. Comparison A: Same-Process GPU

Each case was measured 20 times after compilation and a discarded warm-up.
All 60 measurements were bit-identical for all 84 dynamic numeric leaves plus
five static MJX structure counters. Each field had one unique bit pattern and
each case had one unique contact structure.

**First divergent field: none. Maximum error: 0.**

## 6. Comparison B: Fresh-Process GPU

Fresh process A and fresh process B independently compiled and warmed the same
function, then reloaded the same input files. All fields were bit-identical for
all three cases.

**First divergent field: none. Maximum error: 0.**

## 7. Comparison C: CPU vs GPU

CPU and GPU differed in 44 numeric fields per case. In a dependency-ordered
view of the final Data tree, the earliest observable differing field was
`_impl.cinert` in the kinematics/velocity group:

| Case | `_impl.cinert` max error | qpos max | qvel max | qacc max |
| --- | ---: | ---: | ---: | ---: |
| D0 | 2.98023e-8 | 3.68245e-6 | 2.37584e-4 | 4.76837e-2 |
| D1a | 5.17443e-6 | 2.67621e-4 | 8.10916e-2 | 12.2527 |
| D2 | 2.19792e-7 | 6.47223e-5 | 1.51197e-2 | 2.91986 |

The largest raw absolute difference was `29.0595093` in D1a
`_impl.cfrc_int`. It is an internal force-like quantity; comparing its
magnitude directly with position or velocity fields is invalid.

Because `mjx.step` is compiled/fused and the public result is the final Data
after ten substeps, this experiment identifies the earliest **observable
output group**, not the exact causal GPU/CPU kernel instruction. The latter is
not inferred.

## 8. First Divergent Field

- Same-process GPU: none.
- Fresh-process GPU: none.
- CPU vs GPU: `_impl.cinert`, final-state kinematics/velocity output group.
- Input `ctrl`, motor target, model, randomized parameters, applied forces,
  and serialized Data hashes were fixed before execution.

## 9. Numerical Error Magnitude

GPU repeat errors are exactly zero. For CPU/GPU, the largest dimensionless
ratios using available scales were:

| Case | actuator qpos / joint range | actuator qvel / configured motor velocity | base position / model extent | base velocity / commanded speed |
| --- | ---: | ---: | ---: | ---: |
| D0 | 9.48e-7 | 9.47e-6 | 1.82e-7 | n/a (zero command) |
| D1a | 5.76e-5 | 7.34e-3 | 2.61e-5 | 6.19e-2 |
| D2 | 7.82e-6 | 7.92e-4 | 4.12e-6 | 1.52e-2 |

No independent nominal contact-force scale was available, so no fabricated
relative contact-force metric is reported.

## 10. Contact and Constraint Structure

All comparisons retained the same fixed contact count (`ncon=12`), constraint
count (`nefc=86`), active contact mask, normalized geom pairs, foot-contact
classification, and solver-iteration state. CPU/GPU contact distances changed
numerically (largest: D1a `8.461997e-6`) without changing active/inactive
classification.

This is a numerical-only backend divergence, not a discrete-event divergence.
Joint-limit activation is not exposed as a separate serialized MJX discrete
field and is marked unavailable rather than inferred.

## 11. Termination and Fall Decisions

Termination/fall was false in both members of every comparison. No NaN or Inf
occurred. The comparison therefore found no one-step termination bifurcation.

## 12. Next-Observation Effect

The same saved observation-noise sample was applied to both outputs.

- GPU same-process and fresh-process: raw, noisy, and normalized observations
  were bit-identical (`max error = 0`).
- CPU/GPU normalized-observation max errors were D0 `3.22640e-4`, D1a
  `0.762585`, and D2 `0.0855683`.

Teacher phase and teacher action were identical because those values depend on
the fixed command/phase logic rather than the differing physics fields.

## 13. Next-Motor-Target Effect

The same next policy sample, delay sample, phase, history, and teacher
composition were used.

- GPU same-process and fresh-process: next actor residual, combined action,
  delay buffer, and motor target were bit-identical (`max error = 0`).
- CPU/GPU next motor-target max errors were D0 `8.16584e-6`, D1a
  `0.00588855`, and D2 `0.000385771`.

Thus CPU/GPU backend differences can be controller-visible, especially for
D1a, but no such effect exists between repeated GPU runs in this audit.

## 14. Repeated One-Step Distribution

For every GPU numeric field and case:

- min and max across repeated executions were identical elementwise;
- repeat-axis standard deviation and maximum span were zero;
- number of unique bit patterns was one;
- number of unique contact structures was one.

The per-field distribution is in `repeated_gpu_one_step.csv`.

## 15. XLA Autotuner Observations

The preceding stochastic phase logged:

```text
buffer_comparator.cc:156] Difference at 0: nan, expected 28.6147
gpu/autotuning/gemm_fusion_autotuner.cc:1137] Results do not match the reference. This is likely a bug/unexpected loss of precision.
```

The warning did not recur in any isolated one-step process here. It is
preserved in `environment_manifest.json` but is not assigned causal status.
No XLA or deterministic flag was added.

## 16. Historical Provenance Limitation

`historical_episode_reconstruction = unavailable`.

Reason: the checkpoint did not serialize env_state, per-environment RNG,
rollout key, reset generation, episode index, or randomized model assignment.
For D1b specifically, the trace also lacks the complete step-48 MJX Data and
solver/contact state. This limitation is separate from new forward evaluation.

## 17. Classification

**`BIT_EXACT_PASS`**

The definition is met: 20 GPU repeats from each reconstructible serialized
input were identical across every field. The earlier closed-loop
bit-exact failure is real as an earlier observation, but it did not reproduce
when the physics call was isolated and its complete input fixed.

C2 seed 1 termination at 0.98 seconds remains a performance candidate, not a
physics-parity failure. D1a was non-terminated and bit-exact; D1b is
unavailable. There is no evidence here tying the termination to GPU one-step
nondeterminism.

## 18. Next Allowed Phase

A corrected forward performance diagnostic may proceed in a separate,
explicitly authorized phase. This result does not authorize v59 promotion,
formal 19-command evaluation, deployment, or hardware testing.

