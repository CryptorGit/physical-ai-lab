# EXP013 G1 Phase W1B-C2 shared yaw endpoint evaluator

## Outcome

Classification: `EXP013_W1B_C2_ENDPOINT_PASS_ACQUISITION_PARTIAL`.

The frozen W1B-R2 iteration 200 actor and `MonotonicPositiveYawCalibrationV1`
pass the canonical static/dynamic endpoint gate. No policy parameter,
checkpoint, reward, curriculum, sampler, physics, or calibration was changed.

## Shared contract

`Exp013YawEndpointEvaluator` applies the same physical-target contract to
static trajectories and the dynamic final hold (6-12 s): sign of mean yaw plus
MAE (pure <=0.15 rad/s, moving <=0.20 rad/s), translation/gait constraints,
and safety. `Exp013YawAcquisitionEvaluator` reports transition acquisition but
does not enter endpoint PASS/FAIL.

Unit tests, C1 static regression, D4 dynamic parity, negative controls, and
fresh-process replay all pass. Static/dynamic mean pass-rate difference is
0.778 pp; paired
disagreement is 0.889%; negative-control
false PASS is 0.331%.

## Formal results

- zero-yaw 0.3 m/s: 16/16
- forward 0.6 / 1.2: 100.0% / 99.96%
- pure yaw -0.3 / +0.3: 100.0% / 98.0%
- static moving turns: 24/24
- dynamic final endpoints: 36/36
- translation/yaw independence: 10/10

All endpoint conditions pass. The slow tail remains an acquisition diagnostic:
the maximum condition-level p95 for 0.20 s sustained acquisition is
7.60 s from ramp start, concentrated in rear transitions.

## Safety and symmetry

Aggregate fall 0.089%, dangerous slip
1.251%, impact
0.000%, and long-dwell saturation
0.000%. Dynamic mirror maximum
success difference is 7.0 pp
and mean yaw-MAE difference is 0.0120 rad/s.

## Canonical artifact

The canonical yaw-conditioned WALK artifact is the single W1B-R2 iteration 200
checkpoint (`61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`), the fixed positive-yaw calibration, and the shared
endpoint evaluator. Acquisition timing and random compound transitions move to
Phase W2; they do not invalidate the established endpoint.
