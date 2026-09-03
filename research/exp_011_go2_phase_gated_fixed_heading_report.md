# exp_011 Go2 phase-gated fixed-heading controller — Stage 10

## Outcome

**Classification:** `PHASE_GATED_FIXED_HEADING_PASS`

**Production status:** `DIAGNOSTIC_CANDIDATE`

**Next:** `freeze phase-gated fixed-heading command controller and select tangential-slip reduction as the next isolated Pilot target`

## Controller

The Stage 7 iteration-50 checkpoint is frozen. The command layer computes
`wrap(reference-current yaw)`, applies `Kp=1.0`, clips at `±0.10 rad/s`, and
multiplies by a non-learned phase gate. The gate is disabled during source hold,
speed ramp, and target acquisition; it activates once through a 0.5 s
minimum-jerk profile after 0.5 s continuous target acquisition. All offline
contract tests pass.

## Steady low speed

Mean 0.2–0.6 m/s heading p95 changes from `0.1332` rad (C0) to
`0.0254` rad (C2). All C2 low-speed steady heading gates:
`True`. Steady safety gates: `True`.

| speed | C0 heading p95 | C1 heading p95 | C2 heading p95 | C2 fall | C2 MAE |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.209 | 0.052 | 0.047 | 0% | 0.025 |
| 0.3 | 0.174 | 0.032 | 0.033 | 0% | 0.016 |
| 0.4 | 0.126 | 0.025 | 0.020 | 0% | 0.024 |
| 0.5 | 0.080 | 0.019 | 0.013 | 0% | 0.029 |
| 0.6 | 0.076 | 0.018 | 0.013 | 0% | 0.031 |
| 1.2 | 0.188 | 0.037 | 0.026 | 0% | 0.032 |
| 2.0 | 0.283 | 0.060 | 0.045 | 0% | 0.036 |

## Transitions and anchors

All low-speed transition gates: `True`. Anchor retention:
`True`. Sequence retention: `True`.
Always-on transition degradation avoided: `True`.
The measured rate of C2 feedback activation before acquisition is
`0.0000`.

| transition | C0 heading p95 | C1 heading p95 | C2 heading p95 | C2 completion | C2 fall |
|---|---:|---:|---:|---:|---:|
| 0→0.2 | 0.192 | 0.078 | 0.081 | 100% | 0% |
| 0→0.4 | 0.104 | 0.046 | 0.062 | 100% | 0% |
| 0→0.6 | 0.083 | 0.035 | 0.057 | 100% | 0% |
| 0.6→0.4 | 0.106 | 0.024 | 0.034 | 100% | 0% |
| 0.6→0.2 | 0.197 | 0.052 | 0.070 | 100% | 0% |
| 0.6→0 | 0.112 | 0.080 | 0.105 | 100% | 0% |

| anchor transition | C2 completion | acquisition | hold | fall |
|---|---:|---:|---:|---:|
| 0→1.2 | 100% | 100% | 100% | 0% |
| 1.2→2 | 100% | 100% | 100% | 0% |
| 2→1.2 | 100% | 100% | 100% | 0% |
| 1.2→0 | 100% | 100% | 100% | 0% |

The C2 anchor sequence completes at
`100%`, with fall
`0%` and final stand
`100%`.

## Contact non-regression

Tangential-relative-motion non-regression: `True`. Contact telemetry
remains diagnostic-only and no contact penalty was introduced.

## GUI validation

Steady 0.4 m/s, 0→0.4, and AnchorSequence completed with tracking camera and
floor guides enabled. The installed headless runtime used the console overlay
fallback; no public video is claimed.

## Protection

PPO updates, reward optimization, and policy gradients are zero. All protected
checkpoint and endpoint-protocol hashes match. No production manifest was
changed and no remote push occurred.
