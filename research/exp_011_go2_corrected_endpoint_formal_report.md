# EXP_011 Go2 Corrected Endpoint Formal Report

## Protocol correction

`GO2_ENDPOINT_EVALUATION_V1` was canonicalized and frozen before any formal
rollout. Its SHA-256 is
`d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908`. Isaac Lab `root_quat_w` is decoded
once as xyzw. Zero-command quality excludes the first 1.0 s; fixed-speed
quality excludes the first 2.0 s. Heading uses a wrapped atan2 error around a
frozen window reference.

Four independent read-only PhysX sensors associate FL/FR/RL/RR with the ground.
The formal slip metric uses a normal-force-weighted actual contact-point
centroid in world XY, excludes two boundary steps, and fails an interval at
more than 0.03 m anchor displacement or more than 0.30 m/s for five control
steps. Quaternion and synthetic slip test suites pass.

## Legacy invalidation

Stage 1--4 posture values are retained as
`LEGACY_INVALID_QUATERNION_DECODE`; their height range is
`LEGACY_INCLUDES_RESET_SETTLING`; their slip value is
`LEGACY_FOOT_LINK_ORIGIN_MOTION_NOT_CONTACT_POINT_SLIP`. No old result,
classification, report, or checkpoint was deleted or rewritten. Checkpoint
identity, optimizer stability, speed/fall telemetry, and transition
acquisition remain valid.

## Corrected STAND

Selected hold success is `100%`, fall
`0%`, root speed mean/p95
`0.003/0.026` m/s,
yaw-rate p95 `0.075` rad/s, roll/pitch/tilt p95
`0.076/0.032/0.080`
rad, post-settle height-range p95 `0.0098` m,
and physical-slip episode rate `38%`.
The corrected STAND gate is `FAIL`.

## Corrected steady state

| command | actual | MAE | fall | heading p95 | tilt p95 | physical slip | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.4 | 0.365 | 0.057 | 8% | 0.650 | 2.687 | 100% | PARTIAL |
| 0.6 | 0.603 | 0.049 | 0% | 0.188 | 0.190 | 100% | PARTIAL |
| 0.8 | 0.833 | 0.042 | 0% | 0.132 | 0.079 | 100% | PARTIAL |
| 1.0 | 1.042 | 0.043 | 0% | 0.182 | 0.074 | 100% | PARTIAL |
| 1.2 | 1.246 | 0.046 | 0% | 0.211 | 0.070 | 100% | PARTIAL |
| 1.5 | 1.548 | 0.048 | 0% | 0.224 | 0.067 | 100% | PARTIAL |
| 2.0 | 2.029 | 0.034 | 0% | 0.309 | 0.066 | 100% | PARTIAL |

Gait labels are GO2_GAIT_CLASSIFIER_V1 diagnostics only and do not affect any
gate.

## Corrected transitions

| direction | completion | acquisition | target hold | fall | heading p95 | physical slip | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 0_to_1.2 | 100% | 100% | 100% | 0% | 0.218 | 100% | FAIL |
| 1.2_to_0 | 100% | 100% | 100% | 0% | 0.224 | 100% | FAIL |
| 1.2_to_2 | 100% | 100% | 100% | 0% | 0.341 | 100% | FAIL |
| 2_to_1.2 | 100% | 100% | 100% | 0% | 0.262 | 100% | FAIL |

The directional-asymmetry audit retains separate reset, acceleration, and
deceleration endpoints; no dominant high-speed gait/flight retention is
reported after 2.0 to 1.2 m/s.

## Reduced sequence

Execution: `False`. Gate:
`False`. Reason when skipped:
`zero, required steady endpoints, or formal transitions not all SUPPORTED`.

## Low-speed diagnostic

| command | fall | heading p95 | physical slip | gait counts |
|---:|---:|---:|---:|---|
| 0.1 | 0% | 0.090 | 80% | {'STAND_LIKE': 20} |
| 0.2 | 25% | 1.145 | 100% | {'CRAWL_LIKE': 14, 'FALL': 5, 'STAND_LIKE': 1} |
| 0.3 | 10% | 0.782 | 100% | {'CRAWL_LIKE': 16, 'FALL': 2, 'IRREGULAR': 2} |
| 0.4 | 0% | 0.262 | 100% | {'CRAWL_LIKE': 19, 'IRREGULAR': 1} |
| 0.5 | 0% | 0.177 | 100% | {'CRAWL_LIKE': 20} |
| 0.6 | 0% | 0.157 | 100% | {'CRAWL_LIKE': 20} |

The 0.1--0.6 m/s diagnostic remains outside the formal capability grid except
for 0.4 and 0.6 m/s. It preserves initialization failures and the Stage 5
stand-like-to-locomotion bifurcation rather than averaging them away.

## Classification and next action

Classification: `GO2_CORRECTED_ENDPOINT_FAILURE_MULTIPLE`.

The single next action is:

```text
select one endpoint failure by safety severity before training
```

Stage 6 performed zero PPO updates, zero reward optimization, and no training
interaction. Neither checkpoint was modified or promoted.
