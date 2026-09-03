# exp_012 Stage 2K — Single-policy gait-command latent preflight

## Result

**GAIT_LATENT_REPRESENTATION_FAIL**

The selected 124D student (step 14000, SHA `d0c46acdc2d3a5793d7dc8d6ae9e47f741ff0353fb1aef8c9ff993e71ea8bec3`) represents the deterministic WALK
and RUN means in one checkpoint and demonstrates complete closed-loop gait authority. Formal PASS is withheld because
the preregistered Gaussian KL gate fails: one state-independent student std cannot simultaneously reproduce the
different WALK and RUN teacher exploration distributions.

## Static endpoints

Mean action MSE is 0.00031–0.00035 and cosine is at least 0.99986. Gaussian KL is
0.149 for WALK and 0.220–0.217
for RUN, above the 0.05 contract.

## Closed-loop authority

- 1.2 m/s, gait=0: WALK_LIKE 100%, fall 0%, flight 3.6%.
- 1.2 m/s, gait=1: PERIODIC_RUNNING 100%, fall 0%, flight 47.8%.
- 2.4/2.6 m/s gait=1: periodic 100%/100%,
  completion fires 5671/5653.

Paired fresh-process initial observations match exactly. Gait switch accuracy is 100%.

## Toggle diagnostic

WALK→RUN and RUN→WALK both acquire the target gait in 100/100 episodes with fall 0%.
Mean transition times are 0.558s and
0.980s after the command ramp.

## Interpretation and next

The scalar gait input has real lower-body authority and the deterministic endpoint hypothesis is supported. The sole
formal blocker is distributional std representation, not mean capacity or closed-loop dynamics.

Next single method: **gait-conditioned Gaussian-std representation preflight**.
