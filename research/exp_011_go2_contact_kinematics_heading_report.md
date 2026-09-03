# exp_011 Go2 contact kinematics and heading — Stage 9

## Outcome

**Classification:** `GO2_CONTACT_KINEMATICS_NOT_PRIMARY`

**Heading interpretation:** `ABSOLUTE_HEADING_UNOBSERVABILITY_REMAINS`

**Next:** `phase-gated fixed-heading command controller diagnosis`

No PPO update, reward optimization, checkpoint mutation, external heading feedback,
or production promotion occurred.

## Contact telemetry

PhysX `RigidContactView` provides world contact position/normal, scalar normal
force, separation, friction force/application point, and count/start-index
buffers. Dedicated FL/FR/RL/RR views use the ground collision prim as their only
filter. Units are SI; force values are returned after the API's `dt=0.005`
conversion. Direct relative velocity, manifold ID, and a stable contact-point ID
are not exposed. The resolved dynamic friction coefficient is 0.6.

Foot surface velocity is computed as `v_b + omega_b × (p_c-x_b)` and projected
onto the measured contact tangent plane. All eight synthetic contract checks pass.

## Tangential motion

Stage 7 selected steady summaries:

| speed (m/s) | tangential speed p95 (m/s) | friction utilization p95 | net yaw moment mean (N·m) |
|---:|---:|---:|---:|
| 0.1 | 0.667 | 1.268 | 0.006 |
| 0.2 | 2.485 | 1.999 | 0.145 |
| 0.3 | 2.954 | 1.264 | 0.191 |
| 0.4 | 3.889 | 1.562 | 0.327 |
| 0.5 | 5.248 | 1.632 | 0.304 |
| 0.6 | 6.776 | 1.630 | 0.362 |
| 0.7 | 8.879 | 1.589 | 0.352 |

True tangential surface motion and high-utilization samples exist. Therefore the
Stage 6 contact-point displacement cannot be dismissed as a pure rolling artifact.
However, left/right tangential-speed asymmetry has pooled Spearman
`0.077` with heading slope. The relationship changes sign
or weakens across speed.

## Legacy migration

Contact centroid migration, foot-link-origin motion, and true tangential surface
speed are stored separately. Stable point IDs are unavailable, so same-patch and
patch-replacement events cannot be identified exactly; foot-local centroid and
point-count changes are diagnostic proxies only. Rolling/rocking candidates
(anchor displacement >3 cm with tangent speed <0.05 m/s) exist, especially near
zero command, but do not explain the low-speed heading failure globally.

## Contact yaw moment

Normal and friction forces are evaluated at their respective PhysX application
points about the root COM. Net moment has pooled Spearman
`0.185` with heading slope. Some individual speeds show
moderate/strong coupling, but it is not stable across speed/checkpoint.

## Regression comparison

| model | R² | adjusted R² | CV RMSE |
|---|---:|---:|---:|
| legacy displacement | 0.001 | -0.001 | 0.0134 |
| tangential velocity | 0.016 | 0.015 | 0.0133 |
| contact yaw moment | 0.031 | 0.029 | 0.0132 |
| combined | 0.055 | 0.047 | 0.0133 |

The maximum R² is `0.055`. Contact kinematics is physically real but
does not explain enough of the remaining heading drift to be the primary causal
target. Absolute heading remains absent from the 48D observation.

## Stage 4 versus Stage 7

The low-speed curriculum improves falls and changes contact dynamics, but neither
tangential-slip asymmetry nor yaw-moment asymmetry becomes a stable cross-speed
predictor of heading. This separates fall stabilization from residual heading
control.

## Classification and next action

`GO2_CONTACT_KINEMATICS_NOT_PRIMARY`. The single next method is a
`phase-gated fixed-heading command controller diagnosis`. This is a diagnostic command-layer test, not a PPO Pilot and not
the unsafe always-on feedback tested in Stage 8.

## Protection

Stage 1–8 artifacts, all checkpoints, and `GO2_ENDPOINT_EVALUATION_V1` remain
unchanged. PPO updates and reward optimization are zero. No remote push occurred.
