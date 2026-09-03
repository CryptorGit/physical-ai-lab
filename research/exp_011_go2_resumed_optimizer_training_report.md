# EXP_011 Go2 Resumed Optimizer Training Report

## Status

```text
CLASSIFICATION:
GO2_ENDPOINT_FAILURE_MULTIPLE

NEXT:
endpoint failure diagnosis before another pilot
```

## Optimizer resume

The official checkpoint actor, critic, state-independent Gaussian standard
deviation, normalizer, Adam parameter mapping, moments, step counters, learning
rate, and source iteration were strictly restored. Adam contains 17 parameter
states at step 20,000 with learning rate `0.0003901844231062339`.
First- and second-moment norms are `0.11251891` and
`0.01851612`. Pre-update model identity is bitwise.

## First update

Stage 2 exact KL/clip were `0.50986 / 0.78202`. Stage 4 exact KL/clip are
`0.01453 / 0.20186`.
The first update passes both formal and preferred stability gates, confirming
the Stage 3 optimizer-state diagnosis in a real Pilot.

## Training

Stage 4 completed `300` local iterations and
`14745600` interactions. The selected checkpoint is local
iteration `50` with SHA-256 `e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea`.

## Formal results

Zero-command tracking succeeded in all 50 episodes without a fall, but the
formal gate is `False` because roll/pitch p95 and base-height range
failed their fixed thresholds.

Steady-state support: `{"0.4": false, "0.6": false, "0.8": false, "1.0": false, "1.2": false, "1.5": false, "2.0": false}`.

Transition gates: `{"0.0->1.2": false, "1.2->0.0": false, "1.2->2.0": false, "2.0->1.2": false}`.

Reduced sequence gate: `False`.

Optimizer stabilization and locomotion endpoint quality are interpreted
separately. Speed acquisition and zero-fall transition reachability were
retained, while the fixed dangerous-slip metric failed every moving condition;
0.4 m/s also failed fall and heading limits. These independent endpoint
failures produce `GO2_ENDPOINT_FAILURE_MULTIPLE`. No Stage 4 checkpoint is promoted to
production by this report.
