# OpenDuckMini Batched Domain-Randomized MJX Rollout Divergence Audit

## 1. Executive Summary

Primary classification: **`BATCH_ONLY_MJX_DIVERGENCE`**. Gate: **`TRUE_BATCH_NONDETERMINISM_CONFIRMED`**.

Fresh serialized input, fixed saved motor target, no policy sampling, no reset, no donation, and explicit synchronization still diverge in native batched GPU MJX. Each of the four environments is bit-exact in isolation (20/20 each), while batch 2 diverges in 10/20 outputs and batch 4 in 19/20 outputs. The first difference is:

```text
environment index: 0
physics stage: fwd_position -> smooth.crb
PyTree leaf: _impl/crb
element: [1, 2]
float32: 2.177311897277832 -> 2.177312135696411
absolute error: 2.384185791015625e-7
```

The divergent operation is the reverse body-tree child-to-parent composite-inertia accumulation implemented with `jax.ops.segment_sum`, lowered to scatter-add. It occurs before collision, contact construction, and the constraint solver. The earlier Adam moment difference is downstream, not causal.

## 2. Scope and Non-Claims

This audit executed one physics step only and no optimizer update. It did not change reward, teacher, sampler, curriculum, network, PPO, scene XML, or domain-randomization distribution. It did not add deterministic XLA flags.

The immutable checkpoint contains **4 environments**, not 4096. The training contract in the prior harness used 1250 environments. Therefore only batch sizes 1, 2, and 4 are same-payload comparisons; larger synthetic batches were deliberately not represented as the same serialized state.

## 3. Source and Runtime Provenance

- Repository branch/HEAD at diagnostic execution: `master` / `b992569d95bbd06bcfe8fe20c9319babd3c41553`.
- The user-listed final audit `c6fb2af` remains in history; later unrelated exp_013 commits moved HEAD.
- WSL training source: `338451f33e687ea3edcda8a2c2cdcbc8a7b4bda0`.
- Python 3.12.3, JAX/jaxlib 0.5.3, MuJoCo 3.11.0, Brax 0.14.2.
- GPU backend: NVIDIA GeForce RTX 5090 Laptop GPU, CUDA UMD 13.3, x64 disabled, JIT enabled, one process/device.
- No `XLA_FLAGS`, JAX, or CUDA environment override was present.

The worktree was already broadly dirty/untracked outside this audit. No unrelated file was modified and no commit, push, or tag was made. See `environment_manifest.json` and `runtime.json`.

## 4. Complete Input Identity

Canonical host bytes were produced only after synchronization and host transfer. Every measured non-donated call reloaded the same `state.pkl` from disk.

| Component | SHA-256 |
| --- | --- |
| serialized payload | `42a8c0ec928c372e9d89a101613f7e45dade8fa191ad08ddbb1c77d5b9414b56` |
| complete training state | `d140cd2ee858d7c321ffbd11a8fe2ffdc764a9131f16c4e39774e79063233512` |
| MJX Data | `f990b14cb300b10909ceae5e9569537fabf3649d138ca6d983d9677869cfd67b` |
| randomized MJX Model | `c42fd740abd8c247c1c24236e956a784293eeae9e231757587e38eb56911983f` |
| RNG bundle | `7c952be9ac5d6f49824db9043c7817d4caf53d27427d26275742e36a4acb7f4c` |
| command state | `f81a9dbe2ddc3cefbd1042e3cff3922cc82f39f7d108dfae75d79ac80ff97163` |
| controller state | `bd28c4960862a09cec5a524b93ed91cfbf06596e56ee301eff28ea1cdbdb63de` |

Host hashes were unchanged before/after calls. Evidence: `input_identity.json`, `batch_size_ladder.csv`.

## 5. Donation／Aliasing Audit

Fresh-load/non-donated batch 4 diverged in 19/20 outputs. A separate diagnostic using production-style donation, with disposable inputs never reused, also diverged in 19/20. Unbatched calls remained 20/20 exact.

`INPUT_MUTATION`, `BUFFER_DONATION_REUSE`, and `ASYNC_ALIASING` are rejected for the measured result. Decision: `NO_INPUT_ALIASING_FOUND`. Evidence: `donation_aliasing_audit.md`, `production_donation_results.csv`.

## 6. Policy and Action Injection

P0 injected saved `motor_targets` directly into MJX; policy inference, policy sampling, and controller recomputation were bypassed. P0 still diverged at `smooth.crb`. Therefore policy/RNG divergence is excluded.

P1/P2 were not needed after P0 established a physics-path divergence; saved policy-noise samples were not present in this checkpoint. Evidence: `action_injection_results.csv`.

## 7. Reset／Continuing Environment Split

All four serialized environments have `done=false` and episode step 10. The isolated test did not invoke wrappers or reset scatter. The continuing-only group diverged. The checkpoint has no reset environments, so reset-only and mixed-group comparisons are unavailable.

Reset/masked update is not required for this divergence, though reset-only behavior remains untested. Evidence: `reset_group_results.csv`.

## 8. Batch-Size Ladder

One simulator step, always from a fresh payload:

| Batch | Bit-exact outputs | Divergent outputs |
| ---: | ---: | ---: |
| 1 | 20 | 0 |
| 2 | 10 | 10 |
| 4 | 1 | 19 |

The minimum observed divergent batch size is **2**. Sizes above four are marked not applicable because the checkpoint contains four environments. Evidence: `one_simulator_step/batch_size_ladder.csv`.

## 9. Unbatched vs Batched MJX

Environments 0, 1, 2, and 3 each reproduced bit-exactly in 20/20 unbatched runs. The same four-environment batched function diverged in 19/20 runs.

Concatenating independently compiled unbatched results does not equal the batched result; this comparison includes compilation/layout differences and is supporting evidence, not the temporal first-divergence proof. Classification: `BATCH_ONLY_DIVERGENCE`. Evidence: `unbatched_repeatability.csv`, `batched_repeatability.csv`, `unbatched_vs_batched.csv`.

## 10. Domain-Randomized Model Ablation

Divergence persisted for:

- M0: one saved model replicated over all environments;
- M1: original per-environment randomized models;
- M2: a different single saved randomized model replicated;
- M3: models varied while data/action were identical replicas;
- M4: model, data, and action all identical replicas.

M4 diverged in 19/20 outputs. Model diversity, data diversity, and their cross-product are therefore not necessary. Primary classification is not `DOMAIN_MODEL_BATCH_DIVERGENCE`. Evidence: `model_batch_ablation.csv`.

## 11. Batch Permutation Test

Identity, reverse, two fixed random permutations, and parity ordering all produced non-bit-exact repeated batches. After inverse permutation, numerical outputs can differ from the identity reference, so batch ordering/position affects observed output.

However, the identity order itself is already repeat-to-repeat nondeterministic. The permutation experiment therefore supports batch-position sensitivity but cannot independently estimate an adjacency effect. Evidence: `batch_permutation_results.csv`.

## 12. First Divergent Environment

The smallest reproducible case is batch 2. In all position-stage runs that diverged at the earliest stage, the first divergent environment was **index 0**. Later runs can first expose differences in env 1 after upstream amplification; this does not replace the earliest stage result.

Evidence: `position_stage_leaf_results_batch2.csv`.

## 13. First Divergent PyTree Leaf

`kinematics`, `com_pos`, `camlight`, and `tendon` are bit-exact. `smooth.crb` first changes `_impl/crb`, env 0, element `[1,2]`, by `2.384185791015625e-7`.

`smooth.crb` calls reverse `scan.body_tree`; child inertia is accumulated into parents by `jax.ops.segment_sum`. The narrowed HLO contains scatter-add chains on arrays with an explicit batch dimension. Evidence: `position_stage_results_batch2.csv`, `position_stage_leaf_results_batch2.csv`, `hlo/batch2_crb_filtered_hlo.txt`, `jaxpr/batch2_crb_filtered_jaxpr.txt`.

## 14. Contact／Constraint Dependence

All four input environments have 12 contacts and 86 constraints. Divergence occurs before collision/contact generation. In one-step comparisons, contact geom IDs/pairs, contact count, constraint type structure, and constraint count remain exact; only numerical fields such as `efc_force` can diverge later.

Thus the first cause is not a contact/constraint discrete-event change or solver accumulation. Evidence: `contact_group_results.csv`, `one_simulator_step/first_divergent_leaf.csv`.

## 15. JAXPR／HLO Findings

The source-level operation is:

```text
smooth.crb
  -> scan.body_tree(reverse=True)
  -> jax.ops.segment_sum(child_inertia, parent_index)
  -> scatter-add lowering
```

The narrowed batch-2 HLO contains multiple scatter operations with update windows carrying the batch axis. This establishes the numerical operation where divergence starts. It does not by itself prove the exact GPU scheduling mechanism or atomic execution order; that kernel-level detail remains unknown.

## 16. Instrumentation Identity Retest

Not rerun. No test-harness/state-handling defect was found to fix. Rerunning five optimizer updates would not satisfy the user’s no-optimizer-update constraint and would not remove the proven fixed-action batch-physics divergence.

## 17. Exact Resume Retest

Not rerun. `RESUME_GATE_REOPENED` requires a corrected harness/input/reset defect and subsequent exact pass. Instead, the audit met `TRUE_BATCH_NONDETERMINISM_CONFIRMED`. The previous exact-resume FAIL remains in force. Evidence: `resume_retest_results.csv`.

## 18. Primary Root-Cause Classification

**`BATCH_ONLY_MJX_DIVERGENCE`**

Required evidence is present:

- fresh canonical serialized input every run;
- donation/aliasing excluded;
- fixed action still diverges;
- unbatched environments reproduce;
- batched execution diverges;
- first env/leaf/stage identified.

Secondary observation: batch position/order influences outputs. Rejected primary alternatives: test-harness artifact, policy/RNG, reset/masked update, domain-model batching, contact-solver batching, input provenance mismatch.

## 19. Remaining Unknowns

- The precise GPU kernel scheduling/atomic mechanism selecting a scatter-add accumulation order.
- Reset-only behavior, because this checkpoint contains zero reset environments.
- Same-payload behavior above batch 4; synthesizing environments would not be the same training state.
- Whether a different JAX/MJX/CUDA version or deterministic reduced batching removes the effect.
- The current fixed-action audit did not recompute the controller observation; the prior harness’s first controller-visible difference remains `0.007628441`.

Training telemetry must use `P(vx, vy, yaw, head)` bins as its primary exposure measure. Formal 19-command IDs remain evaluation contracts; `OFF_GRID` is expected under a continuous training sampler.

## 20. Next Allowed Phase

The null-continuation and reward experiments remain blocked. Planning is allowed for:

1. deterministic reduced-batch training;
2. alternative single-device batching;
3. CPU/alternative-backend small causal tests;
4. a multi-seed statistical protocol that explicitly abandons bit-exact resume.

None was executed in this phase.
