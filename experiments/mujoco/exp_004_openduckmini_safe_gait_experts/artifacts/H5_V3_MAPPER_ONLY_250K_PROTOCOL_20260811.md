# H5 V3 direct-command 250k mapper-only protocol

Status: **PRE-REGISTERED SIMULATION-ONLY DIAGNOSTIC.** This is neither a
qualification run nor an authorization for hardware, promotion, release, or
deployment.

## Decision being tested

The valid V2/V3 paired counterfactual established two facts:

- replacing the command mapper at inference for the existing V2-trained actor
  is rejected (3 / 38 strict segments in both arms; V3 strongly regresses pure
  forward heading); and
- the V2 positive-`vx` cross-axis coupling is nevertheless a semantic defect
  for a physical `[vx, vy, wz]` command-conditioned policy.

This run tests only whether the actor can learn the direct, axis-separable V3
observation contract from the clean V22 initialization. It is not a reward,
teacher, target decoder, guard, simulator, command-range, or curriculum change.
It uses the existing 13 signed locomotion/turn anchors with independent
continuous per-axis jitter `[0.92, 1.08]`; it intentionally does not widen to a
full Cartesian velocity box in the same experiment. A full-distribution
curriculum, if justified, is a separate follow-up after this mapper isolation
gate.

## Fixed source and training input

- Parent kind: `V22_BRAX_CHECKPOINT`.
- Parent path:
  `/home/user/openduck_training_runs/calibrated_hybrid_yaw_cost_v22_300m/2026_07_29_154427_10485760`.
- Required parent tree SHA-256:
  `fe35e5ee932dc0ba70c1c32f3e410ea469d229e69cab43ed85f34aefe9505f1f`.
- Reference clean V2 run (for matching non-mapper conditions only):
  `h5_training_runs_diagnostic_20260811/unified/unified/h5_unified_1m_affine_mapper_v2_v22_init`.
  Its `h5_targetspace_seed` is null and its parent is the same V22 checkpoint.
- No `--h5-seed-*` argument, no H3 trace, no target table, no seed params,
  no H4 parent, and no promotion evidence may be supplied.
- Command contract:
  `OPEN_DUCK_MINI_H5_UNIFIED_COMMAND_ROUTING_V3_DIRECT_NORMALIZED` with mapper
  `direct_normalized_v3`. At every mapper call, the policy command must equal
  `diag(2, 5/3, 2) * [vx, vy, wz]` (plus four zero container entries).
- Seed `20260823`; GPU; `250000` environment interactions; exactly five
  50,000-interaction training steps; `1250` environments; learning rate
  `5e-5`; entropy cost `1e-3`; clipping epsilon `0.10`; discount `0.97`;
  maximum gradient norm `0.5`; reset-noise multiplier `1.0`; reverse residual
  scale `0.12`.
- The H5 legacy diagnostic reward configuration, target decoder, target margin,
  slew guard, assets, network architecture, and all other defaults stay equal
  to the clean V2 reference. The **only** intentional behavioral/configuration
  delta is the versioned V3 mapper; the shorter 250k duration is the prescribed
  diagnostic stage, not a performance comparison with the V2 1M run.

## Immutable output and invocation

The runner refuses a pre-existing output directory. It must write only under:

`artifacts/h5_training_runs_diagnostic_20260811/v3_mapper_only_clean_v22/unified/h5_unified_250k_v3_direct_cleanv22_notarget_v1/`

Run from the experiment root in WSL:

```bash
bash scripts/run_h5_v3_mapper_only_250k.sh
```

The run manifest and resolved configuration are the source of truth. They must
bind the V3 contract and canonical mapper, exact clean-parent tree hash, final
params hash, source pre/post hash snapshots, and `hardware_deployment:
PROHIBITED`.

## Post-training gate

Before any longer run, create two same-weight diagnostic wrappers with
`build_h5_diagnostic_candidate_manifest.py --source-manifest RUN_MANIFEST`, then
run the strict actor-only 38-segment suite with 6-second moving segments and
2-second transition stands. The evaluator must report:

```bash
bash scripts/run_h5_v3_mapper_only_250k_strict.sh \
  artifacts/h5_unified_250k_v3_direct_cleanv22_notarget_v1_1x6s_stand2s_20260811.json
```

- V3 training contract and V3 evaluation mapper are compatible;
- both aliases bind the same params, source training manifest, and resolved
  configuration;
- zero legacy fallback, exactly one final guard per control tick, finite state,
  no applied target/joint-limit violations, and complete gait measurements;
- all 38 strict segments pass all frozen safety, tracking, transition, contact,
  force, slip, and smoothness gates.

Any failure rejects this 250k candidate; do not start 1M, 20x30 perturbations,
MP4 capture, deployment packaging, or hardware testing from it. A 1M extension
is eligible only after this exact candidate is 38 / 38 and its source hashes
and V3 mapper provenance are validated. Hardware remains prohibited regardless
of the result.
