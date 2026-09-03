# Exp014 Phase 2-D28R: passive centroidal trace and gated RIGHT feedback

## Executive result

The D27 exact V2A runtime was recaptured with a passive body-trace hook for R4--R7. Capture-off versus capture-on parity passed at the fixed `1e-5` tolerance, and the identity-complete trace passed durability, CoM reconstruction, and centroidal-momentum-matrix validation.

The gated actual-state WBIK V3 shadow did not pass for any source. All four sources failed the required 20% predicted `H_z` error improvement and stance/position gates; therefore D28R centroidal-feedback physics was correctly not started (`0/4` primary, `0/4` replay).

**Classification:** `EXP014_D28R_SHADOW_PREFLIGHT_FAIL`

## Trace parity

Two independent fresh D27 processes were used:

| mode | controller | scope | result |
|---|---|---|---|
| `B0_CAPTURE_OFF` | protected V2A | R4--R7 exact D27 route | baseline |
| `B1_CAPTURE_ON` | same V2A | R4--R7 plus detached body reads | parity PASS |

The four source endpoint hashes, reference hashes, action hashes, physics-state hashes, contact-event hashes, and failure classifications matched. All 47 common arrays were within tolerance with maximum absolute difference `0`; `capture_mutation=0`, classification matches `4/4`.

The captured body contract contains 44 rigid bodies, 37 joints, total mass `32.23892676584296 kg`, body-local CoM offsets, local inertia tensors, world body-CoM pose/velocity/orientation, world inertia, and `[44,6,43]` Jacobians with root columns `[0:6]` and joint columns `[6:43]`. Each R4--R7 has 160 active control steps. Contact-force history length is 3, air-time buffers are available, and the D26 sole polygon is reused read-only. The bundle is atomic and SHA-256 durable:

`86916dabf446002b2ded3c4abf8567599ca14d370bbc47273ea288ca81c326d7`

The D12 audit passed: missing steps `0`, duplicate steps `0`, missing mandatory fields `0`, non-finite active fields `0`, NPZ reader hash match, metadata reader match, and body-recipe identity match.

Source-gate audit retained the D28 result `SOURCE_GATE_CONTRACT_MISMATCH` for the historical D26V-versus-D27 comparison, with the mismatch confined to R0--R3. R4--R7 are contract matches; B0/B1 endpoint eligibility and endpoint hashes match for all four authorized sources.

## Centroidal validation

Whole-body CoM was reconstructed from body-local CoM offsets transformed into world coordinates. Against the runtime D26 CoM fields:

- maximum position difference: `0.0 m`
- maximum velocity difference: `0.0 m/s`
- mass sum: `32.23892676584296 kg`

Direct momentum was computed in the world frame about the whole-body CoM:

`H_i = R_i I_i,local R_i^T omega_i + (r_i-c) x m_i(v_i-c_dot)`

The runtime Jacobian audit established that the captured PhysX linear Jacobian rows already reproduce body-CoM velocity. Applying an additional origin-to-CoM correction would double-count the lever arm, so the protected D28 primitive was left unchanged and the analysis passed the body-CoM point adapter.

The direct body sum and `A(q)v` matrix result passed:

- median relative error: `8.318899609589443e-08`
- p95 relative error: `2.678911848156328e-07`
- `H_z` sign agreement: `1.0`
- near-zero exclusions: `0`
- NaN/Inf: `0`

Selected direct-trace diagnostics (`|H_z|` p95/max, `|dH_z/dt|` p95/max) were:

| source | `|H_z|` p95/max (Nms) | `|dH_z/dt|` p95/max (Nms²) |
|---|---:|---:|
| R4 | `0.3217 / 1.4199` | `11.3894 / 61.7806` |
| R5 | `2.1137 / 6.1179` | `16.9079 / 90.7063` |
| R6 | `0.3618 / 2.7721` | `7.5173 / 54.1401` |
| R7 | `0.3254 / 1.4636` | `10.3418 / 34.5632` |

## Causality

The D27 yaw-divergence window was fixed as the first `|yaw rate| > 0.15 rad/s` sample plus/minus eight steps. In all four sources the first `H_z` threshold, `dH_z/dt` spike, and yaw threshold occurred at the same first START-window sample; the stored classification is `COUPLED_SAME_STEP` for R4--R7. Because yaw was already above the threshold when the START window became analyzable, this trace does not support the stronger claim that centroidal momentum strictly precedes yaw.

Absolute `H_z` contribution fractions in that fixed window were:

| source | upper body (arms + waist/torso + wrist/hand) | stance leg | swing leg | right arm |
|---|---:|---:|---:|---:|
| R4 | `40.9%` | `31.1%` | `16.4%` | `14.2%` |
| R5 | `50.6%` | `13.4%` | `23.6%` | `18.2%` |
| R6 | `50.0%` | `12.9%` | `24.9%` | `23.7%` |
| R7 | `45.7%` | `27.3%` | `15.2%` | `17.8%` |

These are measured contribution fractions, not an intervention result. The protected D28 joint-participation contract states that upper-body attribution is not determinable from the prior trace, so D28R correctly kept participation weighting disabled and used all joint weights `1.0`.

Formal contact yaw moment was **not available**. The runtime supplied net foot-force history and support polygons but no contact points and no contact torque tensor. No ankle/body-origin force proxy was used as a formal yaw moment. Accordingly, contact-yaw-moment attribution remains unresolved.

## Shadow preflight

The shadow used the protected `Exp014CentroidalMomentumAwareWBIKV3` and `Exp014RightStartCentroidalFeedbackV1` contracts, fixed DARE gain, fixed CoP projection, fallback `H_z_target=0`, fixed contact phases, and fixed swing replanning. It was evaluated on 119 captured START rows without applying actions to physics.

| source | rows | solver success | H_z gate | stance gate | CoM/DCM gate | velocity gate | position gate | source PASS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R4 | 32 | 100% | 0% | 0% | 100% | 68.8% | 0% | no |
| R5 | 29 | 100% | 0% | 6.9% | 100% | 62.1% | 0% | no |
| R6 | 27 | 100% | 0% | 3.7% | 100% | 66.7% | 0% | no |
| R7 | 31 | 100% | 0% | 0% | 100% | 58.1% | 0% | no |

The median predicted `H_z`-error improvement was only `0.00044`, `0.000067`, `0.00114`, and `0.00056` for R4--R7 respectively; none approached the fixed 20% requirement. Solver and canonical-action/finite gates were valid, but the V3 shadow introduced stance degradation and joint-position-limit violations, with velocity authority also exceeding the `0.80` gate on a substantial fraction of rows. This is a preflight failure, not a physics outcome.

## Physics and safety

D28R centroidal-feedback physics was not executed because the per-source shadow gate failed. Therefore:

- primary physics: `0/4`
- independent fresh replay physics: `0/4`
- D28R support-shift/liftoff/touchdown/entry/handoff: not evaluated
- new D28R slip/saturation/support-loss/fall counts: not applicable

The retained D27 baseline remains: R4--R7 formed LEFT support dominance and RIGHT liftoff, but RIGHT touchdown and W_MOVE entry were `0/4`; D27 yaw maxima were approximately `19.27`, `15.01`, `17.49`, and `20.14 rad/s` for R4--R7. These values are baseline context only and are not reported as D28R feedback results.

Because no D28R physics pair was launched, D28R primary/fresh process parity is `NOT_APPLICABLE`. Capture-off/on parity is PASS at the fixed `1e-5` tolerance.

## Authorization and next action

`exp014_d29_right_start_teacher_expansion_authorization.json` was not created. `exp014_d29_not_authorized.json` records no authorization. The next authorized methodology step is a separate audit of WBIK V3 task rank, centroidal-map conditioning, and joint velocity authority; do not launch centroidal-feedback physics until a per-source shadow preflight passes. No LEFT START, target/timing/gain modification, PPO/CEM, validation, held-out, checkpoint, persistent update, raw restore, or RUN integration was performed.

## Repository

- starting HEAD: `b292ad54e4003dd00abf6350d97e95e9b57e6541`
- D28/D27 protected artifacts: read-only; hashes recorded in `protected_hashes.json`
- persistent update: `0`
- new learned checkpoint: `0`
- LEFT START physics: `0`
- remote push: `false`

