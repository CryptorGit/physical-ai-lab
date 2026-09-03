# EXP 012 G1 yaw-rate controllability diagnosis

## Status

**Classification:** `G1_YAW_BIAS_SPEED_CONDITIONED_CANCELABLE`

**Pilot readiness:** `EXP012_PILOT1_NOT_READY`
**Next single action:** speed-conditioned yaw-bias cancellation controller preflight

This stage diagnoses the frozen `model_4246.pt` parent only. It does not test
whether the proposed unified STAND/WALK/RUN policy can be learned. Pilot 1 was
not run; PPO updates, policy gradients, and reward optimization are all zero.

## Command contract

The 123D policy observation exposes the velocity command unchanged at indices
9–11: `vx=9`, `vy=10`, and `yaw_rate=11`. The command scale is 1.0 and the
command is expressed in the robot base-frame SE(2) convention. There is no
observation-side clipping or normalization (`Identity`). Live policy-observation
and command-log comparisons both had maximum absolute error 0.

The observed base angular velocity is body-frame, while the parent yaw tracking
reward compares command yaw rate with world-frame root angular velocity z.
Heading decoding uses the parent/Isaac Lab `wxyz` quaternion contract. No
index, scale, sign, frame-routing, or displayed-command mismatch was found.

The source training distribution was recovered from the parent config:
`vx=[0,2.2] m/s`, `vy=[-0.1,0.1] m/s`, and yaw rate
`[-0.2,0.2] rad/s`; heading command was disabled. The 2% standing environments
set all three velocity commands to zero. Consequently, exact zero-speed plus a
nonzero yaw command was not represented: this parent was not trained for true
turn-in-place.

## Open-loop heading

With yaw-rate command fixed at zero, the walking speeds 0.6–1.2 m/s remained
fall-free and had heading p95 between 0.062 and 0.066 rad. Nevertheless the
strict Pilot gate did not pass:

| target (m/s) | fall | speed MAE | mean yaw bias (rad/s) | heading p95 (rad) | long-dwell saturation |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 6% | 0.043 | -0.0530 | 0.510 | 6% |
| 0.6 | 0% | 0.098 | +0.0203 | 0.062 | 0% |
| 0.8 | 0% | 0.102 | +0.0557 | 0.066 | 6% |
| 1.0 | 0% | 0.081 | +0.0634 | 0.064 | 18% |
| 1.2 | 0% | 0.076 | +0.0882 | 0.066 | 12% |

The STAND condition therefore is not merely a turn-in-place limitation:
open-loop STAND itself failed fall, heading, and saturation requirements.
`OPEN_LOOP_HEADING_SUFFICIENT_FOR_PILOT1` is false.

Long-dwell saturation follows the existing G1 definition, not normalized actor
output magnitude: joint velocity utilization ≥95% for 0.05 s or effort
utilization ≥95% for 0.20 s.

## Yaw response

Walking response was monotonic and had a positive affine gain, but it was
offset by an increasingly positive speed-dependent bias:

| speed (m/s) | bias b (rad/s) | gain k | R² | Spearman | cancellation command (rad/s) |
|---:|---:|---:|---:|---:|---:|
| 0.6 | +0.0191 | 0.772 | 0.966 | 0.976 | -0.0248 |
| 0.8 | +0.0530 | 0.715 | 0.987 | 0.990 | -0.0742 |
| 1.0 | +0.0647 | 0.703 | 0.948 | 0.970 | -0.0920 |
| 1.2 | +0.0865 | 0.702 | 0.988 | 0.990 | -0.1233 |

The positive and negative side gains were similar. The command path is thus
locally responsive, but small negative commands do not always reverse the
actual yaw rate: sign accuracy fell from 92.5% at 0.6 m/s to 50% at 1.2 m/s.
At 1.2 m/s even `-0.10 rad/s` left a small positive actual yaw rate, consistent
with the fitted `-0.123 rad/s` cancellation estimate. All fitted cancellation
commands for 0.6–1.2 m/s remain inside the parent's trained ±0.20 rad/s range.

The original all-or-nothing moving controllability gate remains failed because
of sign accuracy, a 5-point fall increase at one 0.6 m/s command, and
long-dwell saturation above 5% at 0.8–1.2 m/s. This does not negate the affine
evidence that the bias is locally cancelable; it means a controller must pass a
separate safety preflight before Pilot 1.

## Contact phase and left/right symmetry

A state-injection or ordinary-reset phase counterfactual was not substituted.
No G1 fresh-process prefix replay contract was preregistered, so the formal
pulse result is `PHASE_COUNTERFACTUAL_NOT_EXECUTED`.

For the allowed normal-rollout diagnostic, differential gain between +0.05 and
-0.05 rad/s was computed within each observed support phase. Gain coefficient
of variation was 0.058 at 0.6 m/s and 0.096 at 1.2 m/s, with no sign reversal.
This supports `YAW_RESPONSE_PHASE_INVARIANT` for response gain. Phase changes
the instantaneous gait yaw component, but it does not explain the systematic
command-response slope.

Left/right force and duty-factor differences existed, but their episode-level
correlations with yaw bias were weak (absolute correlations 0.13–0.23).
Leg action pairs show modest hip/ankle differences; large raw discrepancies in
hand/finger joints are not treated as locomotion-yaw evidence. The available
telemetry therefore does not support left/right action or contact asymmetry as
the primary cause.

## Interpretation and next action

The failure decomposes as follows:

- A — command pipeline mismatch: rejected.
- B — turn-in-place absent from parent training: true, but STAND open-loop also
  fails, so it is not an isolated explanation.
- C/F — speed-dependent positive yaw bias that small commands cannot always
  cancel: primary.
- D — response gain dominated by contact phase: not supported by the
  phase-conditioned differential statistic.
- E — a consistent yaw bias: supported, increasing with speed.
- G — open-loop is already sufficient for Pilot 1: rejected by the frozen gate.

The single next method is a **speed-conditioned yaw-bias cancellation
controller preflight** using the fitted commands within the original training
range. It must validate STAND handling, falls, speed tracking, and long-dwell
saturation before any unified-policy Pilot. Pilot 1 remains unexecuted and not
ready.

## Repository protection

Starting HEAD was `bfa32181ccd81cc2b0b16b65b5cd6b5d7ed2e737`.
The parent SHA-256 remained
`734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621`.
Existing exp_005–exp_011 and prior exp_012 results were not modified by this
stage. Pre-existing unrelated dirty paths were preserved and excluded from the
commit. No remote push was performed.
