# CPU sealed-trace H5 sidecar decision — 2026-08-12

## Decision

The CPU-only H5 quality sidecar is verified **only** as a pure calculation over
a sealed V4 collector trace.  It is not connected to environment reward,
simulation training, GPU execution, checkpoint selection, deployment, or
hardware.  Its sole next authority is `CPU_SIDECAR_ONLY_NO_PPO`.

The prior combined collector-plus-sidecar attempt remains a failed design
gate.  It re-lowered and re-executed the collector before evaluating the
sidecar, so it cannot establish that quality scoring is independent from the
physics runtime.  Do not reinterpret that failure as a pass.

## Immutable evidence

| Artifact | SHA-256 | Result |
| --- | --- | --- |
| `capture_b2t20_cpu_sealed_trace_parent_fresh_process_v15.json` | `48aa2c42e6bf7c0a29e165636c780def295ae8a375edda0110aacbcdcdd06bbe` | CPU V4 collector: 21/21 raw checks pass, no PPO |
| `capture_b2t20_cpu_sealed_trace_parent_fresh_process_v15.npz` | `a2d9d9771bc2664317432c05cda4802a3209d7907d357bbe1b2a6595d63171a0` | Sealed six-array trace export |
| `h5_sidecar_sealed_trace_cpu_fresh_process_v16.json` | `4000b3d307dfdb4dd96055dc42098922aff5a27a310202af94c3cf0289603477` | Pure NumPy sidecar: 17/17 checks pass |

The v15 trace repeat raw-tree SHA-256 is
`e6d321f99df0ee539582305d8e1b105f34f8c75a7a63f0b1b111762e1a423ac7`,
identical to the earlier CPU collector pass.  Its sealed field-bundle SHA-256
is `444750a6b2b1f2fe565a6e028ccaa46ed3c1d8f9c26865864412a18d63a5c02d`.

v16 verifies the parent JSON SHA, the `.npz` file SHA, field names, shapes,
dtype-and-byte digests, and the ordered field-bundle digest before scoring.
It then proves raw-exact repeatability, unroll-boundary equivalence, 200-sample
debounce continuity, asynchronous terminal reset, reset-time handling, and a
single ordered three-term reward delta.  Its execution record has environment
instances, simulator calls, PPO calls, and checkpoint writes all equal to zero.

## Strict boundaries retained

- GPU raw collector parity remains failed (`capture_b2t20_fresh_process_v6.json`);
  it is not interchangeable with CPU evidence and does not authorize PPO.
- The prior combined attempt (`capture_b2t20_cpu_sidecar_semantic_hlo_fresh_process_v12.stderr.log`)
  failed before sidecar scoring.  It is retained as negative evidence.
- Fresh CPU StableHLO dumps v13/v14 differed only in the process-local integer
  assigned to the one `@xla_python_cpu_callback` registration.  The
  location/callback-handle SHA is diagnostic evidence, **not** a structural
  MLIR canonicalization and is not a training authority.
- The parent binds H4 alignment, H5 substep, and H5 sidecar module bytes.  Any
  byte change requires a new collector parent and a new sealed sidecar result.

## Reproduction

Run the collector in the Linux environment with CPU only, fixed seed `20260823`,
expert `unified`, diagnostic reward exploration, direct-normalized-v3 command
routing, `--v4-substep-collector-trace-preflight-only`, and fresh JSON/NPZ
output paths.  Then run:

```text
python scripts/run_h5_sidecar_sealed_trace_preflight.py \
  --parent artifacts/v4_substep_collector_trace_preflight_20260812/capture_b2t20_cpu_sealed_trace_parent_fresh_process_v15.json \
  --output artifacts/v4_substep_collector_trace_preflight_20260812/<fresh-sidecar-output>.json
```

The output must be `CPU_PURE_H5_SIDECAR_SEALED_TRACE_NO_PPO_PASS` with every
check true.  A pass is still CPU-sidecar-only; it cannot be supplied to any PPO
or hardware command.

## Known unrelated repository failure

The complete local pair of test modules currently reports `256 passed,
23 failed, 1 skipped`.  The failures are pre-existing fail-closed H4
authorization drift: the current
`safe_gait_experts/h4_training_alignment.py` SHA is
`217f9fef0e66f621b556bbaa69e77588cd9890831d34fab8b017f335d048a4aa`,
while frozen iteration-v2/v3/v4/v6 authorization expects historical core SHA
`5da1d3a8a2c505a5ce4bc6621f76dd3031070cdb467a4cde96b4ed3c23190c02`.
No historical authorization was rewritten and this mismatch remains a block
for those older paths.  The focused sidecar suite passes: `9 passed, 1 skipped`.

## Next checkpoint

Before any PPO integration, create a parsed, structural StableHLO canonical
representation or otherwise remove the process-local callback registry from the
collector proof without erasing operations, types, attributes, constants,
effects, layouts, or custom-call semantics.  Separately, GPU B=2/T=20 raw
collector parity must pass same-arm, baseline, capture-vs-baseline, and trace
checks exactly.  Neither condition is satisfied today.
