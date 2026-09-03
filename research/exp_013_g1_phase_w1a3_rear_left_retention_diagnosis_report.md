# exp_013 Phase W1A3 rear-left retention diagnosis

This was a frozen-checkpoint diagnosis: PPO updates, checkpoint writes, reward changes, curriculum
continuation, and production selection changes were all zero.

## Timeline and tradeoff

Primary classification: `REAR_LEFT_CHECKPOINT_SELECTION_TRADEOFF`.

The saved all-direction capability timeline contains an existing tradeoff candidate:
`80`. It retains
16/16 at 0.3 m/s and reaches
5/16 at 0.6 m/s. This diagnosis does not change selection.
Fresh 50-episode validation confirms iteration 80 at 16/16 and 5/16, with both forward anchors
at 100%, fall 0%, and dangerous slip 0.55%.

247.5 degrees first drops below 90% at iteration 100; 225 degrees remains at or above 90% through
iteration 120 and drops during E4. Mirror 112.5/135-degree controls remain intact.

## Failure and boundary

In the fresh iteration-160 decomposition, all 13 failures at 225 degrees and all 8 failures at
247.5 degrees are direction-error failures. Vector MAE remains about 0.12/0.11 m/s; gait, yaw,
heading, fall, impact, and saturation do not explain the regression. The fine map shows a localized
213.75-258.75-degree direction-accuracy hole rather than global loss of low-speed walking.

## Exposure, state, and action

The fixed sampler reconstruction gives comparable low-speed exposure to rear-left and mirror bins;
there is no evidence of contract-level undersampling. Historical per-bin return and advantage were
not persisted, so those fields remain null rather than inferred. Matched rollouts show no worsening
of mirror contact-sequence agreement. State/action differences rise modestly, concentrated in hand
and ankle outputs, without a discrete action-manifold discontinuity.

## Gradient, critic, and interpolation

Fresh iteration-120 PPO actor-gradient cosines are +0.325/+0.110 for low versus high speed in the
same rear-left direction, and -0.039/+0.045 versus the combined expansion target. This is not a
strong, consistent conflict. Critic value bias is small (at most about 0.17 in the reported
rear-left conditions), so critic miscalibration is not primary.

Diagnostic interpolation has joint-capability regions: lambda 0.25/0.50/0.75 retain 16/16 at
0.3 m/s while reaching 6/6/7 directions at 0.6 m/s. Temporary actors were deleted and no
checkpoint was created or adopted.

## Artifact interpretation

- W1A remains the all-direction 0.3 m/s WALK artifact.
- W1A2 remains the improved 0.6 m/s expansion artifact (30% to 75% average) with localized
  225/247.5-degree low-speed loss.
- Neither is the final omnidirectional policy.

## Next

Only: **select the existing tradeoff checkpoint as the parent for a low-speed-retention consolidation preflight**.
