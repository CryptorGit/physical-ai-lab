# GPU authoritative primitive batch-parity protocol

Status: **implementation preflight passed; GPU parity result still absent.**

## 2026-08-12 T=1 compilation-observable execution record

`v4_authoritative_primitive_gpu_compile_observable_t1_v23` repeated the fixed
T=1 command in a fresh process.  It changed only the launcher environment by
setting `JAX_LOG_COMPILES=1`, retained fresh immutable stdout/stderr/result
paths, and imposed a five-minute `TERM` timeout.  It did not change the
physics primitive, actions, seed, B=1/B=2 topology, source, thresholds, XLA
flags, cache settings, PPO authorization, or hardware prohibition.

The attempt reached and completed all three relevant XLA compilations:

- canonical B=2 reset: 23.228038311 s;
- B=1 `batched_rollout`: 20.427497149 s;
- B=2 `batched_rollout`: 19.952487469 s.

It then produced no result JSON before timeout status 124.  This narrows the
silent interval to B=2 warm-up execution or a later raw-data operation, but
does not identify an exact source line.  It is a **timeout non-result**, not
a raw-parity result.  The hash-bound attempt record is
`v4_authoritative_primitive_gpu_compile_observable_t1_v23_attempt_record.json`;
the retained compiler log is
`v4_authoritative_primitive_gpu_compile_observable_t1_v23.stderr.log`.

## 2026-08-12 T=1 stage-observable execution record

`v4_authoritative_primitive_gpu_stage_observable_t1_v24` added only opt-in,
host-side stderr stage records around the pre-existing blocking calls.  The
runner source hash was pinned before launch and proved identical after it.
The primitive, actions, seed, B=1/B=2 topology, raw comparator, thresholds,
XLA flags, cache settings, PPO prohibition, and hardware prohibition were
unchanged.

The trace completed `probe_reset` and `canonical_initial_reset`, then emitted
`warm_b1:start`.  JAX logged the start of the B=1 `batched_rollout`
compilation, but no B=1 compilation completion or later stage record before
the fixed five-minute timeout (status 124).  No JSON was written.  This is
another **timeout non-result**.  It proves that the current GPU preflight can
stall before a single B=1 warm-up completes; it does not prove raw parity or
permit T=3, B=2/T=20, PPO, deployment, or hardware.

The prelaunch source/argv manifest is
`v4_authoritative_primitive_gpu_stage_observable_t1_v24_prelaunch.json`; the
hash-bound outcome is
`v4_authoritative_primitive_gpu_stage_observable_t1_v24_attempt_record.json`.

## 2026-08-12 T=1 execution record

`v4_authoritative_primitive_gpu_canonical_b1_b2_t1_fresh_process_v22` was
launched in a fresh WSL process with the fixed T=1 command in this protocol.
It remained in initial JAX/XLA compilation for 15 minutes and five seconds,
using one CPU core, with zero stdout, zero stderr, no JSON output, and no PPO
or checkpoint artifact.  The exact Linux Python PID `284` was then sent
`SIGTERM`; it exited without writing evidence.

This is a **timeout non-result**, not a pass or a parity failure.  The lack of
a result does not clear T=1, T=3, B=2/T=20 collector parity, H5 contact PPO,
or hardware.  The retained logs are:

- `v4_authoritative_primitive_gpu_canonical_b1_b2_t1_fresh_process_v22.stdout.log`
- `v4_authoritative_primitive_gpu_canonical_b1_b2_t1_fresh_process_v22.stderr.log`

## Purpose and boundary

This is a no-PPO diagnostic for the previously observed GPU `B=2/T=20` raw
collector failure.  It compares lane 1 of a canonical `B=2` reset against the
same lane sliced into a `B=1` input.  Each control tick applies the upstream
`mjx.step` primitive exactly ten times through
`v4_authoritative_primitive_step(data.replace(ctrl=action), mjx.step)`.

It does not call `env.step`, reward computation, info/telemetry collection,
H5 sidecar scoring, checkpoint writing, PPO training, deployment, serial I/O,
or hardware.  It does invoke `env.reset` to obtain the canonical initial
state, so its scope is *canonical-reset plus primitive-physics-only*.

The control implementation is guarded by `--platform gpu`, an exact resolved
JAX GPU backend, the exact capture `mjx_model`, the upstream
`joystick.mjx_env.mjx.step` identity, finite in-range actions, zero head
controls, raw-byte same-arm checks, raw-byte cross-batch checks, and source
hashes for the runner, alignment module, scene, manifest, joystick wrapper,
MJX wrapper, and upstream `mjx.step` source.  Failed comparisons serialize
leaf paths and raw first-difference evidence; no tolerance is used.

## Triggering evidence

`capture_b2t20_fresh_process_v6.json` remains the controlling GPU negative
evidence.  Its collector same-arm and cross-boundary raw comparisons failed;
the first force/slip trace divergence is tick 2, lane 1, substep 9.  CPU
collector v15 and CPU sidecar v16 do not change this GPU gate.

## Implementation verification

The diagnostic dispatch was moved out of scalar validation and into `run()`:
after the required environment exists, before the unused evaluation environment
is constructed, and before the PPO checkpoint boundary.  This statement does
not claim that Brax PPO modules were never imported by the earlier training
stack loader; the binding is that no PPO **execution** may occur.

Focused verification, after the move and fail-closed guards:

```text
python -m py_compile scripts/train_h4_aligned_expert.py
python -m pytest tests/test_h5_sidecar_quality.py tests/test_train_h4_aligned_expert.py \
  -k "v4_authoritative_primitive or v4_collector_trace or raw_array_difference or stablehlo or sealed_trace" -q

13 passed, 274 deselected
```

## Execution order

Use a fresh Linux process, fixed seed `20260823`, unified expert, diagnostic
reward exploration, direct-normalized-v3 mapping, GPU platform, and an unused
immutable output path.

1. Run T=1.  Its only action row is exactly zero, followed by ten primitive
   physics substeps.  A T=1 failure is strong evidence of a primitive GPU
   batch-lowering/execution defect.  A T=1 pass proves only this nominal,
   zero-action primitive case.
2. Only if T=1 passes, run T=3.  T=3 adds the fixed finite nonzero action rows
   already audited against `model.actuator_ctrlrange`; it still does not clear
   T=20, `env.step`, collector trace topology, GPU PPO, gait quality, or
   deployment.
3. Preserve every JSON/log result, including failures.  Never overwrite a
   prior output path.

The diagnostic must retain `hardware_deployment=PROHIBITED` and
`NOT_A_TRAINING_CANDIDATE` regardless of pass/fail.  GPU PPO remains blocked
until exact GPU collector B=2/T=20 parity and structural StableHLO provenance
are independently established.
