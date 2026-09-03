# EXP 012 G1 single-policy bidirectional Pilot 1

## Outcome

Pilot 1 started from the audited `model_4246.pt` parent with its matching Adam
state, but stopped fail-closed after the first PPO update.

Classification:

```text
EXP012_FIRST_UPDATE_UNSTABLE
```

No learning-rate change, fresh optimizer, threshold relaxation, retry, Pilot 2,
validation selection, or formal evaluation was performed.

## Integrity

- Parent SHA-256:
  `734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621`
- Observation/action contract: 123D / 37D joint-position target, scale 0.5.
- Actor and critic: bitwise equal to the parent before the update.
- Exploration std and normalizer: bitwise equal.
- Deterministic action: bitwise equal.
- Adam parameter mapping: strict match, 17 states.
- Adam step before training: 85,000.
- Restored learning rate: `2.25e-5`.
- Teacher/expert actions, routing, action blending, and checkpoint switches: 0.

The Stage 1B training amendment was applied. Across all 24,576 rollout samples,
the policy-observation yaw command and unchanged yaw-reward target were exactly
zero. External heading control, phase-gated heading control, and the
speed-conditioned yaw canceller were all disabled.

## Wiring clone

The independent 16-environment clone verified environment construction, tensor
shapes, reset, strict optimizer resume, logging, and checkpoint writing. It was
not used as the parent of the formal run and carries no performance claim.

## First update

The 1,024-environment formal Pilot processed one 24-step rollout:

```text
interactions:             24,576
exact analytical KL:      0.0393813
reported rollout KL:      0.2024432
clip fraction:            0.7239583
mean-action L2 shift:      0.1439008
actor gradient norm:       0.9999996
critic gradient norm:      0.2323755
value loss:                0.0341687
entropy:                  39.6387558
std mean / max:            0.9690589 / 1.4228476
NaN / Inf:                 0
```

Exact KL was inside the 0.20 hard limit and preferred 0.05 reference. The
reported rollout KL exceeded 0.20, and the clip fraction exceeded 0.50. The
hard gate therefore failed even though model parameters and critic statistics
remained finite.

## Checkpoints

Only fail-closed diagnostic checkpoints were persisted:

- Initial:
  `371876f89ebc5a1d3ebac5f57be361745a038ad7b4ca243fe730b852e8e7431b`
- Iteration 1:
  `1221c1ea154206941d99f7c009bf4f4cdfa14057706ce036e0e01f50174b8879`

Neither is selected or eligible for production. Iterations 10 through 300 were
not executed.

## Capability and formal evaluation

Validation checkpoint selection, STAND, WALK, RUN 2.4/2.6, directional
transitions, integrated sequence, directional hysteresis, single-weight
sequence audit, and post-training yaw diagnostics were not executed. Running
them after a first-update safety failure would violate the preregistered
fail-closed protocol.

Consequently, this run makes no claim about the central single-policy
locomotion hypothesis. It only establishes that the attempted Pilot-1 PPO
update contract was not safe under the fixed gate.

## Next

Use one diagnostic method:

```text
diagnose first-update PPO ratio and clipping on the frozen initial rollout
before another Pilot
```

The diagnosis must not change the learning rate, optimizer, reward,
curriculum, network, or yaw contract.
