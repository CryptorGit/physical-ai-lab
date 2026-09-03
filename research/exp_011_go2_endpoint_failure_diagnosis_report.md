# EXP_011 Go2 Endpoint Failure Diagnosis Report

## Status

```text
PRIMARY:
GO2_ENDPOINT_EVALUATOR_MISMATCH_PRIMARY

SECONDARY:
REAL_SUSTAINED_SLIP
STAND_METRIC_NOT_GO2_APPROPRIATE
REAL_LOW_SPEED_GAIT_BIFURCATION
```

No PPO update, reward change, curriculum change, checkpoint change, or physics
change was performed.

## Slip

The Stage 1--4 evaluator marks contact from the maximum 3D force in contact
history above 5 N, takes the maximum world-frame planar velocity of the four
foot rigid-body origins, averages that maximum over the episode, and calls the
episode dangerous when the mean exceeds 0.55 m/s. It does not fail on a single
event and requires no minimum contact or contiguous slip duration.

Foot names, robot indices, sensor indices, SI units, and world frame are
consistent. However, `body_lin_vel_w` describes the foot rigid-body origin,
not the instantaneous ground collision point. The official generic
`feet_slide` function uses the same body-origin velocity with a 1 N threshold,
but the official Go2 Flat reward configuration does not include that term;
its weighted contribution is exactly zero.

Contact-boundary-only events account for
`50.6%` of slip-positive events, below the
80% boundary-dominated criterion. At 0.4/0.6/1.2/2.0 m/s the Stage 4 contact
fractions above 0.5 m/s are
`26.8%, 34.5%, 44.9%, 49.7%`.
Thus the historical binary 100% is not a one-step artifact. It is accompanied
by sustained foot-link-origin motion. Because contact-point velocity was not
available, the report preserves that geometric limitation rather than
overclaiming literal surface sliding.

Paired Stage 4-minus-parent existing-slip mean differences at
0.4/0.6/1.2/2.0 m/s are
`+0.088, +0.012, -0.085, +0.216` m/s.
The effect is mixed rather than a uniform Stage 4 regression.

## Stand

Current Isaac Lab explicitly exposes `root_quat_w.torch` as **xyzw**.
The Stage 1--4 evaluator unbound it as **wxyz**. This is the direct cause of
the reported near-pi roll values during successful stance.

With the correct contract, Stage 4 zero-command results are: fall
`0.0%`, roll p95
`0.089` rad, pitch p95
`0.045` rad, gravity tilt p95
`0.101` rad, and settle-after-2s
height range p95 `0.0074` m.
The original height range included the reset/settling transient and was
`0.0827` m. The settled
nominal-relative tilt deviation is
`0.0081` rad.
This supports `STAND_METRIC_NOT_GO2_APPROPRIATE`, not a real Stage 4 standing
posture failure.

## Low speed

The Stage 4 fall rates at 0.0--0.7 m/s are:
`0.0:0%, 0.1:0%, 0.2:18%, 0.3:14%, 0.4:2%, 0.5:2%, 0.6:2%, 0.7:0%`.
The unstable band is 0.2--0.5 m/s, with the highest failure at 0.2--0.3,
then recovery by 0.7 m/s. At 0.4 m/s, non-fallen yaw p95 remains
`0.364` rad,
so the heading result is not produced only by the fallen tail. The fixed-seed
visual failure shows a genuine collapse. Contact distributions shift from
stand-like stepping toward irregular locomotion in this band, supporting
`REAL_LOW_SPEED_GAIT_BIFURCATION`.

## Gait classifier

Across 1,000 paired diagnostic episodes, the historical classifier labels
70.1% IRREGULAR; 279 of those are independently classified as stand-like
stepping. Its pair synchrony measures equality rather than alternating-phase
opposition, so it is not a reliable quadruped gait identity metric for these
near-full-duty traces. It remains diagnostic-only.

## Classification and next action

The primary result is `GO2_ENDPOINT_EVALUATOR_MISMATCH_PRIMARY` because the proven
quaternion contract bug has precedence. Real sustained foot-link motion and a
real low-speed gait bifurcation are retained as secondary physical findings.

The one next action is:

```text
freeze a corrected Go2-specific endpoint evaluation protocol and rerun Stage 4 formal evaluation without retraining
```

No checkpoint is promoted and no training Pilot is authorized by Stage 5.
