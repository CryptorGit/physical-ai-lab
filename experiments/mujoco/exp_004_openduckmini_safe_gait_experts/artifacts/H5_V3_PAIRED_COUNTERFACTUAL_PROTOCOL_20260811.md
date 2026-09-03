# H5 V2/V3 paired counterfactual protocol

Status: **PRE-REGISTERED SIMULATION-ONLY DIAGNOSTIC.** Neither arm can be
adopted, released, or deployed to hardware. The candidate was trained under
V2, so the V3 arm is an out-of-distribution counterfactual, not a V3 policy
qualification.

## Purpose

Measure whether the historical positive-`vx` observation coupling contributes
to the unified H5 actor's forward and transition failures, without changing
reward, target decoding, final guard, assets, weight, reset, or schedule. V2
remains immutable historical replay; V3 is
`OPEN_DUCK_MINI_H5_UNIFIED_COMMAND_ROUTING_V3_DIRECT_NORMALIZED`.

## Fixed inputs

- Same planar and reverse candidate: `final_params.pkl`, SHA-256
  `887bbbd5dc6f54fe27b9b6e9437b67e719d92e765a60beee515d050553f7c922`.
- Same two diagnostic-wrapper manifests used by the prior one-weight H5 run;
  strict actor-only execution, one identical actor alias in both H5 domains,
  no target table/profile/teacher, and legacy fallback count zero.
- Seed `20260833`; one episode; independent and moving transition segments
  `6.0 s`; transition stand segments `2.0 s`; warm-up `1.5 s`; initial joint
  noise scale `1.0`; initial base speed `0.10 m/s`.
- Frozen target margin `0.050 rad`, target slew `2.0 rad/s` (`0.040 rad` per
  control tick), one final guard call per control tick, frozen evaluator,
  scene, assets, and safety/quality acceptance predicates.

## Arms and output paths

| Arm | Mapper | Immutable command contract | Output |
| --- | --- | --- | --- |
| Historical replay | `legacy_h4_compensated` | `OPEN_DUCK_MINI_H5_COMMAND_ROUTING_V2` | `h5_unified_1m_v2_paired_1x6s_stand2s_20260811.json` |
| Counterfactual | `direct_normalized_v3` | `OPEN_DUCK_MINI_H5_UNIFIED_COMMAND_ROUTING_V3_DIRECT_NORMALIZED` | `h5_unified_1m_v3_direct_counterfactual_1x6s_stand2s_20260811.json` |

## Frozen interpretation and gates

Both artifacts must apply every unchanged strict safety, command tracking,
transition, and gait-quality gate; behavioral promotion would require 38/38
strict segments, but is not authorized by this protocol. The counterfactual
must prove a V3 policy-command error at most `1e-12` at every control tick.
Both artifacts must prove zero legacy fallback and exactly one final-guard call
per control tick.

The pairing verifier must require exact H5 control-trace SHA-256 equality for
the independently reset commands unchanged by V3: `stand`, `reverse`,
`lateral_left`, `lateral_right`, `yaw_left`, `yaw_right`,
`reverse_turn_left`, and `reverse_turn_right`. A trace hashes requested and
effective physical commands, actor-observation command, raw action, preclip
candidate target, margin-clipped desired target, guarded target, post-control
`qpos`, and post-control `qvel` at every control tick. Transition cases are
history-dependent after a positive-`vx` segment and are not controls.

A poor V3 arm rejects only inference-time remapping of this V2-trained weight.
It cannot retain V2 as the future training contract. A subsequent V3 training
pilot must use a separately pinned no-teacher V22 parent and no H3/table seed.
