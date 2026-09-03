# exp_011 Go2 low-speed heading diagnosis — Stage 8

## Outcome

**Classification:** `LOW_SPEED_HEADING_MULTIPLE_CAUSES`

**Pilot readiness:** `PILOT_NOT_READY`
**Next:** `PILOT_NOT_READY`

No PPO update, reward optimization, checkpoint mutation, or production controller adoption occurred.

## Observation contract

The 48D policy observation is body linear velocity (3), body angular velocity (3),
projected gravity (3), velocity command `vx/vy/yaw-rate` (3), relative joint
position (12), joint velocity (12), and previous action (12). It contains no
absolute world yaw, initial/target heading, heading error, world position, or
lateral path error. The policy can suppress instantaneous yaw-rate bias, but it
cannot directly observe the sign or magnitude of accumulated world-heading error.

## Heading decomposition

| speed (m/s) | fall | heading p95 (rad) | mean yaw rate (rad/s) | drift slope (rad/s) |
|---:|---:|---:|---:|---:|
| 0.2 | 2% | 0.182 | -0.0017 | -0.0036 |
| 0.3 | 2% | 0.123 | -0.0057 | -0.0075 |
| 0.4 | 0% | 0.105 | -0.0039 | -0.0058 |
| 0.5 | 0% | 0.108 | 0.0026 | 0.0004 |
| 0.6 | 0% | 0.117 | 0.0049 | 0.0028 |

Stage 7 reduces the Stage 4 low-speed fall band, but small signed yaw-rate biases
still accumulate. Drift is not fall-dominated and the low-speed direction is not
systematic at the 80% criterion. Oscillation is small relative to accumulated
drift. Transition phase analysis shows error in source/target holds as well as
ramps, so this is not a ramp-only failure.

## Left-right and slip

Left/right action means and phases are unequal, especially in the front leg pairs,
but strict offline mirror equivariance was not executed because the observation
mirror mapping could not be verified. Therefore actor asymmetry is observed, not
established as the primary cause. Drift direction is not consistently one-sided.

Contact-point slip asymmetry has strong speed-conditioned Spearman correlation
with signed drift at 0.2 (-0.752), 0.3
(-0.893), and 0.4 m/s
(-0.773). The pooled coefficient is moderate because
the sign/magnitude changes across speed. Correlation is evidence of coupling, not
proof of causation.

Initial roll/pitch correlations are -0.032
and 0.021. Initial yaw is
removed by the corrected heading reference. The contact sensor reports no stable
support at the exact reset frame (`0000`) for every sampled reset, so a causal
initial support-phase classification cannot be made from that frame.

## Yaw controllability

Small yaw commands are monotonic and sign-correct at all four probe speeds;
signed response fraction is 100%, and
the diagnostic class is `YAW_RATE_CONTROLLABLE`. At 0.2 m/s, positive yaw probes caused
5% falls; this asymmetry is retained as a safety caveat.

## Frozen heading-feedback upper bound

This is diagnostic only and was not adopted:

| condition | open heading p95 | feedback heading p95 | open fall | feedback fall | gate |
|---|---:|---:|---:|---:|---|
| 0.2 | 0.170 | 0.043 | 0% | 5% | FAIL |
| 0.3 | 0.122 | 0.033 | 0% | 0% | PASS |
| 0.4 | 0.104 | 0.031 | 0% | 0% | PASS |
| 0.5 | 0.125 | 0.027 | 0% | 0% | PASS |
| 0.6 | 0.141 | 0.029 | 0% | 0% | PASS |
| 0.6_to_0 | 0.118 | 0.088 | 0% | 0% | PASS |
| 0.6_to_0.4 | 0.119 | 0.031 | 0% | 0% | PASS |
| 0_to_0.4 | 0.116 | 0.193 | 0% | 0% | FAIL |
| 0_to_0.6 | 0.133 | 0.180 | 0% | 0% | FAIL |

Feedback improves five steady conditions and two deceleration transitions, but
creates a 5% fall rate at 0.2 m/s and worsens `0→0.4` and `0→0.6` heading.
Consequently it does not isolate a safe single command-layer fix.

## Interpretation

Absolute heading unobservability is a necessary structural limitation, while
speed-conditioned contact-point slip coupling is an independent physical
contributor. Action asymmetry remains unresolved, and the fixed feedback upper
bound is non-uniform. The evidence therefore supports multiple causes rather than
a safe single intervention. `PILOT_NOT_READY` is the only permitted next action.

Official-parent steady comparison is complete. Several official-parent transition
raw-contact chunks ended inside the Isaac/PhysX contact telemetry path; Stage 4 and
Stage 7 transition datasets are complete. This comparator limitation is recorded
in `gate.json` and is not hidden by zero-filled metrics.

## Protection

Stage 1–7 artifacts and all three checkpoints remain unchanged. The
`GO2_ENDPOINT_EVALUATION_V1` hash remains
`d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908`.
PPO updates and reward optimization are zero. No remote push was performed.
