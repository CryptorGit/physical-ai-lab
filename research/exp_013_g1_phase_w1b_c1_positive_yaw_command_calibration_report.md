# exp_013 Phase W1B-C1 positive-yaw command calibration

## Outcome

Classification: `EXP013_W1B_C1_CORE_PASS_DYNAMIC_PARTIAL`.

`MonotonicPositiveYawCalibrationV1` keeps non-positive yaw unchanged and maps positive
physical targets to actor input with a fixed 1.50 gain. It is a command-interface
calibration; the frozen actor remains the only source of joint actions.

## Static formal results

- positive pure/moving conditions: 18/18 PASS
- original moving-turn matrix: 24/24 PASS
- translation/yaw independence: 10/10 PASS
- zero-yaw 0.3 m/s: 16/16 PASS
- forward 0.6 / 1.2: 100.0% / 94.0%
- negative-yaw retention: PASS

## Dynamic findings

The static yaw core is repaired, but the zero-crossing requirement is not met:
the minimum target-sign acquisition across the prescribed sequences is
88.3%. Backward positive/negative curve conditions also
remain outside the dynamic success gate. The 60-second random diagnostic achieved
2.0% simultaneous episode success,
physical-target yaw MAE 0.104 rad/s, vector MAE
0.068 m/s, and 2.0% falls.

## Safety and symmetry

- aggregate formal fall: 0.056%
- dangerous slip: 1.084%
- impact: 0.000%
- long-dwell saturation: 0.000%
- mean mirrored yaw-MAE difference: 0.0117 rad/s
- maximum mirror success difference: 2.0 points

## Interpretation

The fixed calibration is sufficient for the static pure-yaw, moving-turn, and
independence core. It is not promoted because the dynamic zero-crossing gate is
not met and the sampled forward 1.2 retention result is 94% versus the required
95%. The latter uses a bitwise-native zero-yaw path, so it is not a calibration
regression. The canonical artifact therefore remains W1A2 iteration 80.

No policy parameter, checkpoint, optimizer, reward, curriculum, sampler, robot,
physics, Isaac Lab core, or RSL-RL package was changed.
