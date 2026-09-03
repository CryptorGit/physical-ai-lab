# exp_009 Stage 5 — Base/action-manifold compatibility

## Result

**PIECEWISE_OR_PHASE_CONDITIONED_BASE_REQUIRED**

Stage 4's `RESIDUAL_PARAMETERIZATION_INADEQUATE` result remains unchanged.  The
large target was not caused by comparing an unclipped actor tensor with an
applied action.  No policy clipping occurred, the action term has no configured
clip, all teachers share the 37-joint order and scale 0.5, and the large
difference remains after conversion to the actual joint-position target.

## Pipeline

The WALK actor emits a normalized 37-D position action.  RUN and WALK_TO_RUN
emit `running_base + 0.25*tanh(skill residual)`.  Isaac applies
`default_joint_position + 0.5*normalized_action`.  The actual global previous
action is shared in columns 86:123 during every common-state cross-forward.
Observation normalization is disabled for all three actors.

## Common-state compatibility

100,000 frozen physical observations were used, with 10,000 states for every
WALK speed, RUN speed, and WALK_TO_RUN occupancy third.  The WALK base exactly
matches WALK but has zero Level-2 full-vector coverage on RUN and nearly all
transition groups.  The internal RUN base contains RUN and all transition
groups within its existing 0.25 route, yet does not contain WALK even at Level
2.  Therefore no single existing base reaches 99% full-vector coverage over all
regimes.

The dominant cross-base differences include ankle pitch, ankle roll, knee, and
upper-body coordinates rather than a single joint.  The difference subspaces
need rank 12–14 for 99% variance, so the correction is not one-dimensional,
although it is not full-rank either.

## Transition endpoints

At the early transition occupancy, WALK_TO_RUN differs from the WALK teacher by
0.413 normalized action on average and has 0% Level-2 full-vector coverage.
At acceptance, it is close to RUN: Level-2 full-vector coverage is 100%.
This confirms an endpoint migration toward the RUN manifold, not one common
base spanning both endpoints.

## Closed-loop diagnostic

The 240-episode diagnostic started every candidate from the same valid
WALK@1.2 occupancy.  WALK held 0.6/1.2 but failed to track RUN commands and fell
at 2.8.  RUN internal/full bases held the tested commands in this short
diagnostic, but their action targets still differed materially from the formal
WALK teacher and safety dwell gates were not claimed.  WALK_TO_RUN endpoint
anchors were not treated as steady controllers.

## Next

The single next research method is **continuous phase-conditioned base
morphing** between frozen WALK and RUN manifolds.  Stage 5 does not implement
that method, expand a residual bound, train a controller, or change capability.
