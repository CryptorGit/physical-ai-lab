# v60 Bounded Yaw Paired Pilot

## Result

Decision: **STOP_AT_1M**.  This is a diagnostic causal pilot, not an adoption
or hardware-transfer decision.

## Matched design

Both arms restored the same v52/v45 normalizer, actor and critic, used fresh
zero-state Adam with seed 20260730, and executed exactly 1,000,000 environment
interactions (1,600 optimizer updates).  Their step-0 parameter hashes match.
The only reward difference is the yaw contribution inside command_progress.

## Primary yaw result (Condition D)

| Controller | Left ratio | Right ratio | Mean yaw MAE | Yaw falls |
| --- | ---: | ---: | ---: | ---: |
| parent v52 | 0.681 | 1.100 | 0.456 | 8 |
| Arm C 1M | 0.825 | 1.123 | 0.389 | 3 |
| Arm T 1M | 0.504 | 1.077 | 0.337 | 0 |

Arm T reduced the mean yaw MAE by 13.2% versus Arm C, below
the required 50%.  Its left response became an undershoot (0.504×),
while the right response was 1.077×.  The left/right
gap therefore also fails the 0.15 gate.

## Stochastic yaw result

Arm C yaw-only falls: 7/10.  Arm T yaw-only falls:
8/10.  Treatment did not satisfy the not-worse gate.

## Retention

Deterministic treatment retention falls: 0.  Stochastic
3-seed treatment/control falls over the seven retention commands:
4/6.  Commands exceeding the 10% primary
linear-speed degradation test: C01_forward, C09_forward_yaw_left.

## Decision and non-claims

The bounded objective changed yaw behavior, but it did not produce the required
bilateral, 50%-MAE improvement and it increased yaw-only stochastic falls.
Training stops at 1M; 5M and the 19-command final pilot were not run.  No claim
is made about solving linear undershoot, backward initiation/tracking, domain
randomization, push recovery, formal acceptance, adoption, or hardware safety.

## Instrumentation limitations

The installed Brax callback did not expose the exact rollout command tensor or
optimizer state.  Training term metrics and seeds are saved, but an exact
rollout command histogram and resumable Adam state are unavailable.  Three
failed control attempts caused by WSL libcuda host-boundary crashes are retained
under `failed_runs/` and are excluded from the causal comparison.
