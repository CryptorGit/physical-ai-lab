# H5 PDCA cycle — contact terms and pre-guard feasibility

Status: **REJECTED — simulation-only diagnostic evidence; no promotion or hardware deployment.**

This record covers two controlled H5 unified PPO screens performed after the
strict actor-only baseline.  The acceptance thresholds, runtime target guard,
H5 command mapping, action decoder, seed, PPO settings, evaluator seed, and
1×6 suite definition were not relaxed.

## Baseline

- Strict evidence: `h5_unified_1m_rate2_profile_bc_from250k_1x6s_20260811.json`
  - SHA-256: `c227a667939f8c05a62e19a0742b53c154eff17187b09692c18cb911cc8708bc`
  - strict gait-quality segments: 0 / 38
  - pre-guard inward-margin violations: 5,803 samples / 38 affected segments
  - forward normal-force p99: 5.024 body-weight fractions

## Experiment A — force-tail and contact-pulse only

- Training manifest:
  `h5_training_runs_diagnostic_20260811/strict_force_tail_pulse/unified/h5_unified_250k_force_tail_pulse_v1/run_manifest.json`
  - SHA-256: `e57e48b5e47bbee91398eb39cd0d68948b1520e71b7400262064f671663bf190`
- Strict evidence: `h5_unified_250k_force_tail_pulse_1x6s_20260811.json`
  - SHA-256: `9d95001e66f5dbdba6c4a3288ed3d6c3edcd2ee3051e076793d74e4ecc875014`
- Only changed reward coefficients: `h4_total_normal_force_tail: 0 -> -1`,
  `h4_contact_pulse_40ms: 0 -> -1`.
- Result: strict gait-quality segments 4 / 38, but 40 ms
  `debounce_all_windows_quality` remained 0 / 24 and slip RMS/P95 remained
  2 / 24.  Forward normal-force p99 regressed to 5.208 and pre-guard margin
  violations rose to 8,142.

Decision: reject; do not extend to 1M, multi-seed, 20×30, or deployment.

## Experiment B — pre-guard target/action feasibility only

- Training manifest:
  `h5_training_runs_diagnostic_20260811/strict_target_feasibility/unified/h5_unified_250k_target_feasibility_v1/run_manifest.json`
  - SHA-256: `72f867072ecd6248867d4cd9c651a6b2d72a071c71abee348aff9d61c7ba0111`
- Strict evidence: `h5_unified_250k_target_feasibility_1x6s_20260811.json`
  - SHA-256: `14c1024e03e67ddb30040020af162c8a611008af1175a4ba1ba678fb68d37385`
- Hypothesis: penalizing target inward-margin excess and actor action overflow
  before the unchanged final guard would reduce the raw target saturation
  observed in every baseline segment.
- The temporary implementation was unit-tested (including JAX outward
  gradients) and then reverted after evaluation.  The source hashes used by
  this training/evaluation remain bound in the run manifests.
- Hard reject criteria: at least 90% fewer pre-guard margin violations,
  no desired/applied violation, forward p99 no greater than 5.024, and no
  strict-quality/tracking/contact regression.
- Result: 8,021 pre-guard violations across all 38 segments, forward p99
  5.757, strict gait-quality segments 3 / 38, debounce 0 / 24, and slip
  RMS/P95 2 / 24.

Decision: reject and revert the code change.  The current source hashes for
`h4_training_alignment.py`, `train_h4_aligned_expert.py`, and
`test_h5_target_contract.py` were restored to their pre-experiment values;
`python -m pytest -q tests/test_h5_target_contract.py` passes 11 tests.

## Experiment C — exact H3 applied-target replay, no tuning

The limited H3 diagnostic artifact
`h3_exact_home_baseline_trace_forward_yawleft_5s_v1.json` was not treated as
training data.  Its source status is `DIAGNOSTIC_ONLY_NOT_ADOPTED` and its
hardware status is `PROHIBITED` (source SHA-256
`32fa783e4c589b0beb85afe3ba0a8b738ad3cef7025c88890ec40e7dc0bdb2f2`).

`scripts/explore_h5_target_program.py` now has a diagnostic-only
`recorded_h3_trace` mode (source SHA-256
`22c3fb3116bb615663aa7d0d8384364698ff8eb0c8fd1d22cd284d4752d0ed99`).
It accepts only that diagnostic status, requires the exact matching route,
checks a finite 14-wide target row, proves H5 decode round-trip error at most
`1e-12` rad with all actions inside `[-1, 1]`, and forbids amplitude, phase,
smoothing, profile, or table tuning.  During a blended route it advances the
recorded sequence once per control tick and supplies the same row to both
blend inputs, so a 250-row target trace cannot be accidentally consumed as
two actor calls per tick.  The final version also pins source hash/schema/
artifact kind, exact-home reset and startup audits, source command/timing,
250 consecutive ticks, and zero head targets; runs H5 strict-actor-only with
legacy fallback disabled; captures candidate and physically applied targets
at every tick; and independently rederives strict segment acceptance.

The H3 forward and yaw-left rows were representable without clipping:
maximum H5 action magnitude was 0.878 and 0.764 respectively; maximum
decoder round-trip error was `1.11e-16` rad.  Both fixed 5 s / 250 tick
replays completed without a fall, joint-limit violation, applied target limit
violation, or target-slew violation.  This establishes target-space and
physics-path compatibility only.

The two paths without the `strict_v2` suffix are retained only as preliminary
invalid diagnostics: they did not yet prove strict-actor-only isolation,
per-tick applied-target fidelity, or independent acceptance.  The two
`strict_v2` artifacts below are authoritative.  In both, all 250 candidate
and physically applied targets matched the source within `1.11e-16` rad,
strict actor mode was true, and legacy fallback count was zero.

- Forward evidence: `h5_h3_recorded_target_replay_forward_5s_strict_v2_20260811.json`
  (SHA-256 `7630cf1ac341ae912c45a76e3e000d2b7ebcc8934036a83731382ee9ca13b48e`)
  - strict gait quality: false; 21 failing checks
  - normal-force p99: 4.852 body-weight fractions
  - stance-slip RMS/P95: 0.02495 / 0.05487 m/s
  - uncommanded yaw rate / endpoint heading: 0.12090 rad/s / 0.66087 rad
- Yaw-left evidence: `h5_h3_recorded_target_replay_yaw_left_5s_strict_v2_20260811.json`
  (SHA-256 `b8b0f8f45a9c6045c869ae1374f656d66e58ac6962651b46ef98f286606b04f8`)
  - strict gait quality: false; 20 failing checks
  - normal-force p99: 6.182 body-weight fractions
  - stance-slip RMS/P95: 0.03716 / 0.08536 m/s
  - failing checks include contact debounce, slip, force, cadence, yaw
    tracking, and yaw-only SE(2) drift

Decision: both authoritative screens are `REJECTED_AS_H5_SEED`. Reject this
H3 trace as an H5 strict teacher, BC seed, checkpoint, or deployment artifact.
The formal H3 20×30 release is aggregate evidence, not a per-tick
116-observation/guarded-target dataset; even a future exact export cannot be
presumed qualified after the available exact forward/yaw target rows failed
the unchanged H5 gate.

## Experiment D — direct command mapper, current-weight OOD ablation

The unified command contract deliberately inherited H4's positive-vx
compensation: a physical pure forward command `(0.05, 0, 0)` is shown to the
actor as `(0.10, -0.018, -0.170)`.  This mixes lateral and yaw requests into
a pure forward observation, while reverse has no such coupling.  It is a
semantic mismatch with the requested continuous physical `[vx, vy, wz]`
contract, but replacing it for the already-trained weight is an OOD test, not
a deployable change.

Added, without changing the legacy default:

- `h5_unified_direct_policy_command` and its XP counterpart: direct
  normalization only `(2 vx, 5/3 vy, 2 wz)`.
- `--unified-command-mapper {legacy_h4_compensated,direct_normalized}` on the
  strict evaluator.  `direct_normalized` is explicitly recorded as
  training-mapper-incompatible and remains simulation-only.
- Contract tests now cover legacy and direct pure-axis behavior.  The H5
  target contract test suite passed 13 tests.

Source SHA-256: `h5_command_contract.py`
`a838c06b4c52809fefd2c81a942ce418a0b6fd8926ebb11153a611755c6051e1`;
`evaluate_h5_routed_transitions.py`
`9254c1ad7c4c947798b16974faf95c1528b0b75fc01dbbde53a6e0f4b8181434`.

The same 1M unified weight was evaluated under the same 1×6 seed, initial
noise, initial base speed, 6 s, and transition configuration as the baseline:

- Evidence: `h5_unified_1m_direct_normalized_1x6s_20260811.json`
  (SHA-256 `36e80d8ef21f72a69e19e39516d8b50723bea7d888a42079ffa5c74eff97d64b`)
  - strict quality segments: 0 / 38; suite false; no promotion
  - pure-forward policy command: `(0.10, 0, 0)` exactly
  - forward force p99 improved `5.024 -> 4.804` body-weight fractions and
    slip RMS improved `0.02764 -> 0.02444` m/s
  - but current-weight forward heading drift regressed `0.055 -> 0.692` rad,
    cross endpoint error `0.036 -> 0.151` m, and strict quality failures
    increased `13 -> 25`
  - reverse is unchanged, confirming the ablation only changes positive-vx
    command observations

Decision: reject mapper swapping for the current 1M weight.  The ablation
does not exonerate the legacy coupling as a root semantic defect; it proves
that the existing actor has learned that coupling.  Any direct mapper path
requires a separately trained one-weight candidate with unchanged safety and
quality gates.

## Current dominant failure and next checkpoint

The recurring failure is not an unrepresented scalar penalty or missing H3
target-space expressivity: the H5 actor gait lacks debounced alternating
contact, low stance slip, and bounded force/heading behavior.  Do not run
another reward-scale sweep and do not promote any H3/MPC/FSM/table source as
a teacher.

The next checkpoint is a structural intervention proposal that directly
addresses the correlated contact/slip/force failure modes while retaining one
velocity-conditioned H5 actor and the unchanged final guard.  It must decide
whether a direct normalized command curriculum is justified, state the exact
causal hypothesis and pre-registered 1×6 rejection gate, and only then start
a new training run.  Hardware remains `PROHIBITED` throughout.

## Experiment E — versioned V2/V3 paired command-contract counterfactual

The prior direct-map screen used a V2-named artifact and one-second transition
stands.  A direct map is a distinct single-policy command contract, not a
runtime option on V2.  V2 is therefore frozen for historical replay and the
new axis-separable contract is
`OPEN_DUCK_MINI_H5_UNIFIED_COMMAND_ROUTING_V3_DIRECT_NORMALIZED`.

Protocol: `H5_V3_PAIRED_COUNTERFACTUAL_PROTOCOL_20260811.md`.  Both arms use
the same current 1M actor (`887bbbd5…c922`), same strict actor-only H5 bridge,
same seed/noise/base speed/6-second moving segments, and **2.0-second**
transition stands so every gait-quality measurement is complete.  V2 uses
`legacy_h4_compensated`; V3 uses `direct_normalized_v3`.  No teacher, table,
profile, target authority, reward, guard, or deployment path was changed.

Evidence:

- V2: `h5_unified_1m_v2_paired_1x6s_stand2s_20260811.json`, SHA-256
  `df860117d3050ffc6e44891e5fbdc007590fa54172b855835b357a3cf26ee45a`.
- V3 counterfactual:
  `h5_unified_1m_v3_direct_counterfactual_1x6s_stand2s_20260811.json`,
  SHA-256
  `918d47d98468184937c593bce7e8fee23874d784fac8fd314de932c7bbe5d4f0`.
- Pair verifier:
  `h5_unified_v2_v3_paired_counterfactual_1x6s_stand2s_20260811_v2.json`,
  SHA-256
  `cf350ac223610810f448eb03f87508c4401e0d012453a1c5b52ff5c74d46520f`.
  The earlier no-suffix verifier output is retained as an invalid tooling
  diagnostic: it incorrectly assumed that flattened independent suites held
  one segment.  It is not evidence; the `v2` verifier checks the preserved
  per-case guard audits.

The valid pair proves: all 38 gait-quality measurements are complete; legacy
fallback is zero; one final guard call occurs per control tick; V3 command
fidelity is exactly zero error; and raw action, candidate/desired/applied
targets, and post-control `qpos`/`qvel` trace hashes are exactly equal for all
eight independently reset V3-unaffected cases (stand, reverse, both lateral,
both yaw, and both reverse-turn commands).

Result: strict segments are **3 / 38** in both arms.  Pure forward under V3
does reduce force p99 `5.024042 -> 4.803662` body-weight fractions and stance
slip RMS `0.027643 -> 0.024440 m/s`, but regresses endpoint heading error
`0.055435 -> 0.691790 rad`, adds debounce/cadence/alternation failures, and
does not improve the strict count.  Thus inference-time V3 remapping of this
V2-trained actor is **REJECTED**.  This rejects only the V2-weight
counterfactual; it does not restore V2 as a future command contract.

Implementation provenance: `h5_command_contract.py`
`01b55aa32d3f530fa8546f31edfacf6150db272ba679d046b61b887156a106a5`,
`evaluate_h5_routed_transitions.py`
`5e36ae35a2d44a65a2edf75a88f6a7ccc256b2a8129f0ed812d65a31646e42bc`,
`evaluate_routed_transitions.py`
`c9f445344cbedbc83ba3aca7d814824017531446a8b6ebcf3fc1bd66671087e9`,
and `verify_h5_paired_counterfactual.py`
`93d1659010c71f7873f47fdfc949655060c019448f90e881fc2a3584bb5e90d3`.
The H5 contract test suite now passes 15 tests.

Next: retain V3 as the only future unified command contract, pin the clean
no-teacher V22 parent (`fe35e5…5f1f`), and pre-register a single 250k V3
mapper-only PPO pilot.  Do not reuse H3 traces, a 54-row target table, or any
H5 target-space distilled seed.  Hardware remains `PROHIBITED`.

## Experiment F — clean-V22 V3 mapper-only 250k pilot

Protocol: `H5_V3_MAPPER_ONLY_250K_PROTOCOL_20260811.md`. This is the first
candidate trained, rather than only evaluated, with the direct V3 command
contract. It uses the clean V22 checkpoint—not the current V2 actor—as parent
and supplies no H5 seed bundle, target table, H3 trace, H4 parent, or promotion
evidence. The single deliberate behavioral change relative to the clean V2
reference is mapper `direct_normalized_v3`; the 250k stage is the prescribed
diagnostic gate.

- Training run:
  `h5_training_runs_diagnostic_20260811/v3_mapper_only_clean_v22/unified/h5_unified_250k_v3_direct_cleanv22_notarget_v1/`.
  - final params SHA-256:
    `d9ff9552f7ba62cc86ecf0bd92b33dfec153aadd6a4c0101af2e946dfc553f41`.
  - run manifest SHA-256:
    `673eec20bb7f782e3aa47b8ee79d0f82c61db65e41cbc495243df8ccabfd8252`.
  - resolved config SHA-256:
    `e1a2d4a2e73f22958c0ae9ed9941e1153654f555618e35612896c4c58fbe39f5`.
  - result SHA-256:
    `8693fa882af403ad8d892a1ad461d8e2da93a5eec495327a249047f2f27eb366`.
  - completed exactly 250,000 interactions / five 50,000-interaction steps on
    `cuda:0`; parent and source snapshots are both recorded unchanged.
  - the resolved configuration binds
    `OPEN_DUCK_MINI_H5_UNIFIED_COMMAND_ROUTING_V3_DIRECT_NORMALIZED`, canonical
    mapper `direct_normalized_v3`, null `h5_targetspace_seed`, and clean V22
    parent tree SHA-256
    `fe35e5ee932dc0ba70c1c32f3e410ea469d229e69cab43ed85f34aefe9505f1f`.

- Strict evidence:
  `h5_unified_250k_v3_direct_cleanv22_notarget_v1_1x6s_stand2s_20260811.json`.
  - SHA-256:
    `ab6cb84e67f03ad11702929797994f15c817227ce2f59777a131904f2f101947`.
  - strict actor-only, one unified weight in both aliases, zero legacy fallback,
    V3 training/evaluation mapper compatibility true, and V3 command-contract
    error exactly zero across all 4,900 guarded control ticks.
  - all 38 segments completed and no segment fell. The applied target and joint
    limit protections remained active; this does **not** override failed gait
    quality or pre-guard-margin evidence.
  - strict gait-quality result: **3 / 38**, identical count to the paired V2
    baseline. The only passing segments are transition stand segments after
    reverse, lateral-left, and forward-turn-left: **0 / 12 required moving
    locomotion commands pass**.
  - dominant failures remain contact debounce (24 / 38), stance-slip RMS/P95
    (22 / 38 each), force p99 (19 / 38), left/right step count (20 / 38), and
    touchdown regularity (23 / 38). Pure forward has force p99 4.700 body
    weights, slip RMS/P95 0.02592 / 0.05792 m/s, cross error 0.1921 m, and
    endpoint yaw 0.8537 rad, so it is not an acceptable forward gait.

Decision: **REJECT the 250k V3 mapper-only candidate.** Do not run its 1M
extension, multi-seed evaluation, 20x30 perturbation campaign, MP4 capture,
deployment packaging, or hardware tests. The direct command semantics remain
the correct versioned interface; this result says that changing only the mapper
does not repair the contact/force/slip gait failure. The next intervention must
be selected against that dominant physical gait failure, not by weakening a
quality gate or reintroducing H3/table teachers.
