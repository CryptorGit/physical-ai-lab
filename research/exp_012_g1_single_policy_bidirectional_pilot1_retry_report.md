# EXP 012 — G1 single-policy bidirectional Pilot 1 retry

## Outcome

The single authorized retry completed all 300 PPO iterations with the Stage 2B
strict-resume LR contract. The update path was stable and a single checkpoint
retained the complete WALK range. Formal evaluation nevertheless found both a
narrow STAND regression and a large RUN_LOW/safety failure:

```text
G1_SINGLE_POLICY_MULTIPLE_FAILURES
```

The next isolated method is:

```text
multi-regime gradient interference diagnosis
```

No Pilot 2 or second retry was executed.

## Integrity

- Starting HEAD: `3e30caef16d4b732c03c200103a0991f82757d5d`
- Parent SHA-256: `734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621`
- Parent contract: 123D observation, 37D joint-position action, scale 0.5
- Adam states / step: 17 / 85,000
- Restored optimizer/runtime/scheduler LR: `2.25e-5`
- Resume implementation: `Exp012StrictPPOResumeContract`
- Yaw command: zero for every training sample
- External yaw controllers, teacher/expert actions, action blending, and
  checkpoint switching: zero

The curriculum stayed at `ZERO_HOLD/WALK_STEADY/RUN_HOLD/
BIDIRECTIONAL_SEQUENCE = 20/20/20/40%`. Parent rewards were unchanged; the only
additional term was exp_005 `safe_periodic_flight`, gated at requested
`vx >= 2.3m/s`.

## Training and LR integrity

- Environments / rollout: 1,024 / 24 steps
- Iterations / interactions: 300 / 7,372,800
- Seed: 20261021
- PPO epochs / mini-batches: 5 / 4

First-update hard gates passed:

| Metric | Result |
|---|---:|
| Exact Gaussian KL old→new | 0.01878 |
| Maximum KL over 20 optimizer steps | 0.01914 |
| Joint clip fraction | 0.27254 |
| Mean-action shift | 0.09278 |
| Critic gradient / value loss | 0.17955 / 0.02988 |
| NaN / Inf | 0 |
| First / final update LR | 2.25e-5 / 7.59375e-5 |

The early guard passed. Optimizer/runtime/scheduler LR equality held for all
300 iterations. Adaptive KL ranged from `1.0e-5` to `1.70859375e-4`, always
starting from the restored state and without manual tuning.

Periodic-running validation remained weak: mean 2.4/2.6 success was 0% at the
parent, 10% at iteration 10, 35% at iterations 50/100/200/250, and 15% at
iteration 300. Later updates also increased falls. This is not a case where the
last checkpoint should be selected automatically.

## Frozen checkpoint selection

The predefined precedence selected iteration 100 before formal evaluation:

- Checkpoint SHA-256:
  `8d8afac60cafbd4adf0b98469fab01f711f32771a40899653d962cc08a5d8143`
- Actor hash:
  `fcd1f33b98b474176ff1bfc68a24ba31c5d1d4cd7f4c867f1b27f292271b3fff`
- Adam step: 87,000
- LR: `7.59375e-5`

Gait events were counted only inside each target quality window. Source-RUN
flight events were not allowed to make a later 1.2 m/s hold look like running.

## Formal abilities

All results use 50 deterministic episodes per condition, seed root 20263021,
zero yaw command, no controller/canceller, and the selected checkpoint only.

### STAND — FAIL

- settle / hold: 94%
- fall: 6%
- speed MAE: 0.0057 m/s
- heading p95: 0.0278 rad
- final double support: 94%
- dangerous slip / saturation: 0% / 0%

Tracking and posture were good when upright, but the 95% hold and 2% fall gates
were missed.

### WALK — PASS

- 0.6 / 0.8 / 1.0 / 1.2 success: 100% / 100% / 98% / 100%
- overall: 99.5%
- fall: 0%
- speed MAE: 0.038–0.110 m/s
- heading p95: 0.050–0.055 rad
- dangerous slip / saturation / impact: 0%

### RUN_LOW — FAIL

- periodic running at 2.4 / 2.6: 64% / 22%
- aggregate: 43%
- fall: 34% / 74%
- speed MAE: 0.129 / 0.410 m/s
- heading p95: 0.462 / 0.760 rad
- dangerous slip: 16% / 48%
- impact failure: 6% / 12%

The failure is not merely a gait-label miss: safety, heading, speed at 2.6,
slip, and impact gates also failed.

### Directional transitions

- `0→0.6`: PASS, 100%
- `0.6→1.2`: PASS, 100%
- `1.2→2.4`: FAIL, periodic acquisition 68%
- `1.2→2.6`: FAIL, periodic acquisition 78%
- `2.4→1.2`: FAIL, WALK acquisition 66%, fall 34%
- `2.6→1.2`: FAIL, WALK acquisition 42%, fall 58%
- `1.2→0.6`: PASS, 100%
- `0.6→0`: PASS, 100%

### Integrated sequence — FAIL

Sequence completion was 38%, fall 6%, and final STAND 94%. RUN-hold segment
success was 60% at upward 2.4, 86% at 2.6, and 82% at downward 2.4 m/s. The
single-weight audit passed: one checkpoint/actor hash, zero expert calls, and
zero checkpoint switches.

## Regression and hysteresis

WALK was retained, but STAND narrowly regressed and RUN caused severe falls.
Long-dwell saturation stayed at 0%; degradation concentrated in fall, heading,
slip, and impact at high speed.

After quality-window correction, target-hold gait statistics no longer
misattribute source-RUN flight to 1.2 m/s. Detailed endpoint telemetry is stored
for reset, upward, and both downward 1.2 m/s paths. The old Stage 1B yaw
cancellation table was not applied; the selected policy's bias changed and the
old table would over-correct.

## Repository protection

Existing exp_005–exp_011 results, prior exp_012 stages/checkpoints, Isaac Lab
core, installed RSL-RL, capability manifest, and production artifacts were not
changed. Pre-existing unrelated dirty paths were preserved and excluded from
the commit. Remote push was not performed.
