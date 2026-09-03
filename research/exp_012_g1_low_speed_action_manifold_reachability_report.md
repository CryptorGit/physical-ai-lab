# exp_012 Stage 2J — Low-speed WALK action-manifold reachability

## Result

**LOW_SPEED_WALK_MANIFOLD_NOT_LOCALLY_REACHABLE**

The 1.2 m/s WALK trajectory has a statistically significant return advantage over both RUN policies, so low-speed
reward indifference is rejected. Nevertheless, the WALK and RUN physical-state distributions are effectively
disjoint (nonlinear AUROC 0.999986), and bounded perturbations do not
provide a reliable local bridge.

## Positive controls

| policy | gait | flight fraction | stride frequency | return | fall |
|---|---:|---:|---:|---:|---:|
| W0 Stage 2 | WALK_LIKE | 0.035 | 1.403 Hz | 25.026 | 0.0% |
| R0 Stage 4 | PERIODIC_RUNNING | 0.480 | 6.217 Hz | 22.104 | 0.0% |
| R1 selected | PERIODIC_RUNNING | 0.481 | 6.253 Hz | 22.185 | 0.0% |

The gait distinction is physical, not classifier-only: RUN has about 48% flight and almost no double support, while
WALK has 3.5% flight, predominantly single support, lower vertical velocity, and lower pitch.

## Reward landscape

W0 exceeds R1 by 2.840 return
(bootstrap 95% CI 2.710 to
2.973). `safe_periodic_flight` is exactly zero on every 1.2 m/s sample.
The advantage comes from the unchanged base objective, notably yaw tracking, angular/vertical motion, orientation,
air-time, slide, acceleration, and action-rate terms.

## State, action, value, and gradient

WALK-vs-R1 mean-action L2 distance on WALK states is 2.957; the difference is
distributed across the action vector, with the largest lower-body terms at ankle and knee joints. WALK actions are
far off-policy under R1: only 0.16% are inside the diagnostic ratio-valid
range, clip fraction is 99.95%, and ESS is
0.56%. Consequently the positive valid-sample advantage cannot establish a
reliable on-policy WALK direction. The BC direction is nearly orthogonal to the base-reward gradient
(cosine -0.0088); BC is diagnostic only and was not used for learning.

## Reachability

Fresh-process prefix hashes matched for 1024/1024 single-step
counterfactuals and 1024/1024 short-sequence
counterfactuals. One-step success was 0/1024. Four-step bounded
random shooting produced 1/1024 candidate, or
0.098%, below the 5% partial-reachability gate. No parameter or checkpoint was updated.

## History and interpretation

After convergence at current command 1.2, source-history classification is `HISTORY_NOT_IDENTIFIABLE`. This does not
imply that current gait is hidden: WALK and RUN current physical states are directly distinguishable. Rather, the
RUN policy maps the histories onto the same RUN attractor. The inference is **WALK_MANIFOLD_DYNAMICAL_BARRIER**.

## Next

Perform exactly one next method: **single-policy gait-transition latent/input preflight**. No gait input, reward, recurrent state, or curriculum change was
made in Stage 2J.
