# Phase 2-D28 — centroidal causality audit and RIGHT START feedback pilot

Classification: `EXP014_D28_CENTROIDAL_MOMENTUM_INTERFACE_FAIL`.

## Dynamics diagnosis

D27 R4–R7 contact yaw moments were reconstructed from the persisted ankle-roll body-origin/contact-force proxy. Exact contact points and contact torques were not persisted, so those values are diagnostic proxies only. Root yaw-rate, CoM/root tracking, swing overshoot, slip, and saturation timing were retained. Whole-body `H_z`, per-body angular-momentum terms, and upper-body/leg momentum fractions could not be reconstructed because the D27 NPZ does not contain `body_pos, body_quat, body_com_pos, body_com_vel, body_ang_vel, body_com_ang_vel, body_jacobians, body_inertias, body_com_quat`. No inferred `H_z` was substituted.

The source-gate audit is `SOURCE_GATE_CONTRACT_MISMATCH`: D26V uses an endpoint last-50 window, while D27 freezes cumulative fresh-process safety flags at the endpoint. D28 therefore retains only D27-eligible R4–R7 for the authorized scope.

## Centroidal controller

`Exp014CentroidalMomentumAwareWBIKV3` and `Exp014RightStartCentroidalFeedbackV1` contracts were defined in the new D28 output. They preserve V2A, the RIGHT_000 target, canonical `q_cmd = default_q + 0.5 * raw_action`, and fixed physics parameters. The DARE-derived discrete LIPM gain and swing p90/p05 bounds were fixed before any D28 outcome. Entry `H_z` was unavailable, so the contract records the required fixed `H_z_target = 0` fallback. The joint participation metric was not activated because the required upper-body contribution audit was unavailable.

## Shadow preflight

Synthetic centroidal matrix tests pass, including mass sum, mirror sign, static pose, finite-difference linearity, and direct body-sum versus matrix comparison. The required D27 per-step V2A/V3 shadow action gate is **FAIL** because body Jacobian, inertia, body-local CoM velocity/angular velocity, and body pose fields are absent from the saved D27 trace. Physics was consequently not started.

## Physics and safety

Primary R4–R7 physics episodes: **0/4**. Fresh replay episodes: **0/4**. Weight shift, liftoff, yaw reduction, clearance, touchdown, W_MOVE entry, handoff, slip, saturation, support loss, and fall are all `NOT_EVALUATED` in D28; D27 baseline values remain read-only in its own artifacts.

## Process parity

Not run because the mandatory prephysics gate failed. The registered numeric tolerance remains `1e-05` and was not relaxed.

## Authorization and protection

`exp014_d29_not_authorized.json` is emitted. RIGHT expansion, landing repair, and dynamics-constrained optimization are not authorized by this D28 result. Persistent update `0`; new checkpoint `0`; LEFT physics `0`; PPO/CEM/validation/held-out/RUN `0`; remote push `false`. D6–D27 artifacts, S_HOLD, Stage 2Q, W_MOVE, S_STOP_OMNI, V1/V2/V2A, RIGHT_000, and the canonical action contract were unchanged.

Starting HEAD: `7fb59fdd6e93ce08b154cb8dab6b8be801619f41`. Ending HEAD before commit: `7fb59fdd6e93ce08b154cb8dab6b8be801619f41`.
