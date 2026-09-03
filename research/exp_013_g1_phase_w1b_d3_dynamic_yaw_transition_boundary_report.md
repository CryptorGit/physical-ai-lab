# exp_013 Phase W1B-D3 dynamic yaw transition boundary diagnosis

## Outcome

Classification: `DYNAMIC_EVALUATOR_WINDOW_MISMATCH`.

The current command pipeline is P1: physical yaw is minimum-jerk interpolated,
then the asymmetric calibration is applied at each step. Actor input is continuous
at zero but its first derivative changes with gain 1.0/1.5.

## Boundary findings

- ramp duration 0.25--4.0 s: every duration passed 5/10 to 5/10 conditions
- zero dwell 0--2.0 s: every dwell passed 5/10 to 5/10
- profiles C1/C2/C3/C4: {'C1': 5, 'C2': 5, 'C3': 5, 'C4': 5}
- history STATIC/SAME/OPPOSITE/ZERO: {'STATIC': 5, 'SAME_SIGN': 5, 'OPPOSITE_SIGN': 5, 'ZERO': 5}
- backward final-hold instantaneous-sign fraction: 0.769--1.000

The failure persists even when starting statically at the final yaw target. This
rules out ramp length, zero dwell, slope discontinuity, and opposite-sign history
as primary explanations. The static formal evaluator passes these same endpoints
because it evaluates mean yaw sign and yaw MAE. The dynamic evaluator instead
requires instantaneous yaw to retain the command sign for 95% of the entire
episode, including gait-periodic yaw oscillation, ramp, and prior history.

## System, contact, and random diagnostics

- mean first-order model R2: 0.442
- contact-start success spread: 0.058; contact phase is secondary
- worst random segment class: zero_to_positive (18.2%)
- forward 1.2 mean across 100x50 episodes: 99.96%
- probability of <=94%: 0.0%

No training, policy/checkpoint update, production command shaper, reward,
curriculum, sampler, network, robot, physics, or core-library change was made.
