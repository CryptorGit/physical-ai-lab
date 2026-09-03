# Exp 011 Stage 2 — continuous Go2 0–2.0 m/s training report

## Training

Stage 2 used one implementation only: a single continuous Go2 policy with a
symmetric steady-state, acceleration, and deceleration command curriculum.

The parent was the official Isaac Lab
`Isaac-Velocity-Flat-Unitree-Go2-v0` checkpoint with SHA-256
`32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0`.
Actor, critic, distribution/log-std, and observation-normalizer state were
loaded strictly. Optimizer state was not loaded; the unchanged official PPO
optimizer configuration was initialized fresh.

The warm-start audit passed every bitwise check:

- actor and critic state dictionaries matched the parent;
- deterministic actions and stochastic distribution means matched bitwise;
- distribution parameters and log-std matched bitwise;
- the Identity observation normalizer had no missing state;
- optimizer state count was zero before and after loading.

The training environment was registered only by importing the exp_011 package
as `Isaac-Exp011-Go2-Bidirectional-0To2-v0`. Observation remained 48D, action
remained 12D position action with scale 0.25, and the 48→128→128→128→12
network was unchanged. Physics dt 0.005 s, control dt 0.020 s, decimation 4,
terrain, friction, PD gains, termination, reward terms, and reward weights were
unchanged.

The 100,000-segment pre-training audit produced exactly 25% ZERO_HOLD, 25%
STEADY_SPEED, 25% ACCELERATION, and 25% DECELERATION. Acceleration and
deceleration each had 25,000 samples. Pair max/min ratios were 1.0 and the
steady-speed ratio was 1.00028. No negative, lateral, yaw, or 2.5 m/s training
command was sampled. Baseline and Stage 2 reward hashes were identical:
`cc8b4fc34263c132e24555d67d58b92fc41e6f58b0cac0b72fe00a34e2a78ce4`.

The 16-environment, two-update wiring run completed and carries no performance
claim. Its small-batch KL warning was retained as wiring-only telemetry.

Pilot 1 was configured for 2,048 environments, 300 iterations, seed 20260911,
24 rollout steps per environment, and the unchanged official PPO settings. It
stopped after the mandatory first update:

| Metric | First update | Gate |
|---|---:|---:|
| Approximate KL | 0.51294 | ≤0.20 |
| Clip fraction | 0.78412 | diagnostic |
| KL from initial policy | 0.21220 | diagnostic |
| Mean-action L2 shift | 0.24545 | ≤2.0 |
| Actor gradient norm | 1.00000 | diagnostic |
| Critic gradient norm | 0.45404 | ≤1e6 |
| Value loss | 0.04737 | ≤1e8 |
| Policy loss | 0.10458 | diagnostic |
| Entropy | 4.71199 | diagnostic |
| log-std mean / max | -1.02549 / -0.75771 | finite |
| NaN / Inf | 0 | 0 |

The first-update approximate KL exceeded the immutable threshold. The run
therefore stopped as `GO2_TRAINING_UNSTABLE`; no threshold was relaxed and no
restart was attempted. Completed interaction count was 49,152
(2,048 × 24 × 1), rather than the planned 14,745,600 interactions.

Two audit checkpoints exist:

- initial warm-start:
  `511d11557e81ffd410f6b7c06d2304fd1f8eb4f4dc61c5554a2b3ebc0a94f03e`;
- rejected iteration-1 unstable checkpoint:
  `ee862ff9922b2c33721f0d3e02814bfd472fef68549b9fb06ad542f1d5915f27`.

The unstable checkpoint was not validation-eligible and was not selected.
Iterations 25–300 were not created because fail-closed termination occurred
first.

## Zero command

Formal seed 20262901 evaluation was not run because no validation-selected
stable checkpoint exists. Stage 1 remains the protected reference: hold success
86%, fall 14%, classification STAND FAIL. No Stage 2 improvement claim is made
for hold, fall, speed, contact, slip, or saturation.

## Steady-state

Formal 0.4–2.0 m/s evaluation was not run. Consequently there are no Stage 2
speed-error, gait, fall, heading, slip, saturation, or SUPPORTED/PARTIAL/
UNSUPPORTED claims. Stage 1 values remain reference-only and were not
overwritten.

## Bidirectional transitions

Formal 0→1.2, 1.2→2.0, 2.0→1.2, and 1.2→0 evaluations were not run. The Stage 1
diagnostics showing 1.2↔2.0 reachability were not reused as Stage 2 evidence.
Directional asymmetry and endpoint hysteresis were not evaluated because the
training stability prerequisite failed.

## Reduced sequence

The reduced 0→0.6→1.2→2.0→1.2→0.6→0 sequence was not run. Required checkpoint
validation and steady endpoint support were unavailable. No reset, checkpoint
switch, or unsupported command execution was used to manufacture a result.

The conditional 2.5 m/s extrapolation was also not run.

## Classification

`GO2_TRAINING_UNSTABLE`

This classification is triggered solely by the frozen first-update approximate
KL gate. It is not a locomotion, slip, heading, transition, or sequence
classification because formal policy evaluation was correctly blocked.

## Next

First-update PPO stability diagnosis with the frozen Stage 2 contract.

This is the only recommended next action. It must determine why an unchanged
official optimizer configuration produces an excessive first update after the
required optimizer reset. Pilot 2, reward changes, or modular experts are not
authorized by this Stage.

## Repository

Starting HEAD was `3132bc57e4d29c44ddc3da7b9d20016f1c9ad900`.
Stage 1 result files and report retained their starting SHA-256 values.
Experiments exp_005 through exp_010, capability manifests, production
artifacts, the official checkpoint, and Isaac Lab tracked core were unchanged.
The pre-existing exp_006, OpenDuck, artifact, and media dirty state was
preserved and excluded from staging. No remote push was performed.
