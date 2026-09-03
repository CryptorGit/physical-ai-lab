# EXP 012 G1 speed-conditioned yaw cancellation preflight

## Scope

Stage 1B evaluated the frozen `model_4246.pt` parent only. No PPO update,
policy gradient, reward optimization, state injection, or checkpoint change was
performed.

## Controller

`G1SpeedConditionedYawBiasCancellerV1` uses commanded forward speed only. Its
fixed table is:

| speed (m/s) | yaw offset (rad/s) |
|---:|---:|
| 0.6 | -0.0248 |
| 0.8 | -0.0742 |
| 1.0 | -0.0920 |
| 1.2 | -0.1233 |

The table is piecewise-linear above 0.6 m/s, held at the 1.2 m/s value above
that speed, and limited to ±0.15 rad/s. Output is strictly zero at or below
0.4 m/s. Activation between 0.4 and 0.6 m/s and temporal
activation/deactivation use minimum jerk over 0.50 s. All offline unit tests
passed.

## STAND equivalence

The parent checkpoint, G1 asset, 123D observation, action scale 0.5, physics
step 0.005 s, control step 0.02 s, decimation 4, plane terrain, reset base
configuration, and termination base configuration agree with exp_007.
The reported 2% versus 6% falls are not the same protocol sample: exp_007 used
seed 20260723 and a state-based 0.4 s settle streak followed by an 8 s hold,
whereas the yaw diagnosis used seed 20261101, a fixed 8 s episode, and a
heading reference at 2 s. Both historical results remain intact.

In the paired Stage 1B zero-speed evaluation, both controller conditions had
0% falls and heading p95 0.0122 rad. Policy command and action traces were
bitwise identical, controller offset was exactly zero, and saturation was
identical. The controller therefore caused no STAND interference.

## Steady cancellation

| speed | yaw bias off | yaw bias on | reduction | heading p95 off | heading p95 on | fall off/on |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | +0.0200 | +0.0004 | 97.9% | 0.1245 | 0.0192 | 4% / 4% |
| 0.8 | +0.0564 | -0.0026 | 95.4% | 0.3075 | 0.0285 | 0% / 0% |
| 1.0 | +0.0631 | +0.0001 | 99.8% | 0.3746 | 0.0373 | 0% / 0% |
| 1.2 | +0.0866 | -0.0058 | 93.3% | 0.5104 | 0.0594 | 2% / 2% |

Mean absolute yaw bias fell from 0.05654 to 0.00223 rad/s, a 96.1% reduction.
Speed-MAE changes were between -0.0011 and +0.0018 m/s, lateral-velocity
changes were at most +0.0021 m/s, and long-dwell saturation remained 0%.
The tiny flight-fraction changes at 1.0 and 1.2 m/s occurred in contact modes
already present in the paired baseline and were not classified as a new
contact instability.

## Transition cancellation

For the fixed
`0→0.6→0.8→1.0→1.2→1.0→0.8→0.6→0` sequence, completion remained 100% and
falls remained 0%. Heading p95 improved from 0.9870 to 0.1065 rad. Mean
speed MAE changed from 0.07905 to 0.07966 m/s. Final speed was 0.0041 m/s and
final stand hold was 100% with the controller. Activation and deactivation
action jumps remained within the controller-off action-rate p95. Long-dwell
saturation stayed at 14% in both conditions; the controller did not worsen it.

## PPO semantic audit and amendment

The policy observation and yaw-tracking reward consume the same command-manager
yaw target. A negative cancellation offset intended to produce physical yaw
rate zero would therefore be interpreted by unchanged PPO reward semantics as
a target for negative physical yaw. There is no existing separated
policy-command/reward-target interface.

Pilot 1 is consequently amended as follows:

```text
training yaw-rate command: 0
external heading controller: OFF
speed-conditioned canceller: OFF
parent yaw-tracking reward: unchanged
```

The fixed canceller remains a frozen-checkpoint evaluation candidate. The
current Stage 2 command implementation contains phase-gated heading behavior
and must not be run until a future authorized stage applies the frozen
amendment. The Stage 1B open-loop parent still exceeds the 0.12 rad heading
threshold at moving speeds; reducing that bias is a Pilot objective, not a
pre-existing parent capability claim.

## Classification

`G1_SPEED_CONDITIONED_YAW_CANCELLATION_PASS`

Pilot readiness: `EXP012_PILOT1_READY_OPEN_LOOP_YAW`

The earlier broad-matrix `MOVING_YAW_RATE_NOT_CONTROLLABLE` result is retained.
Stage 1B establishes the narrower fact that the preregistered operating-point
feedforward table safely cancels the observed bias.

## Next

Run Pilot 1 with yaw-rate command fixed at zero and all external yaw
controllers disabled.
