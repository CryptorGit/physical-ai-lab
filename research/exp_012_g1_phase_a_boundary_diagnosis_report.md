# exp_012 G1 Phase A boundary diagnosis

## Scope

Stage 2F is a frozen-checkpoint diagnosis. It performed no PPO continuation,
Phase A extension, Phase B run, reward edit, curriculum edit, optimizer step,
or checkpoint write. The selected checkpoint remained iteration 50
(`4edbb595e28e24dc09cf39e8245c7be1b1bebf792798a73af2e562075d0fe952`).

## Completion reproduction

The historical 241 training completions are available only as iteration-level
counts. Stage 2E did not retain their observations, policy means, sampled
actions, or log probabilities, so the report records
`TRAINING_COMPLETION_ACTION_TRACE_NOT_AVAILABLE` and does not infer them.

In the new frozen sweep, deterministic mean actions produced **0** valid
completion events at every checkpoint. S100 produced **27** events and
S150 produced **237** events; both reproduced completion across all
five swept checkpoints. Completion density increased with exploration noise,
while falls also increased sharply. This passes `COMPLETION_EXPLORATION_ONLY`.

## Action distance

Across 264 valid stochastic completion landings, the overall
Mahalanobis median was 8.993. Within S100,
completion median was 6.203 versus
6.013 for failed landings; within
S150 it was 9.124 versus
9.029. Successful landing actions
are therefore not single-step outliers inside a fixed noise distribution.
Exploration changes the trajectory enough to enter the completion event, but
the final landing action itself is typical for that exploration level.

## RUN boundary and reward quality

The registered 10 s fine sweep shows broad high-speed safety failure rather
than a robust narrow deterministic completion basin. The preserved Stage 2E
8 s result still contains a sharp 2.4 to 2.5 m/s boundary
(-55.0 periodic points and
55.0 fall points), but the longer
sweep exposes that 2.4 m/s is not durable.

The gait classifier and reward intentionally answer different questions.
Episode-level periodic flight can pass while every landing fails one or more
strict duration, memory, alternation, speed, tilt, or vertical-speed gates.
This is a `PERIODIC_GAIT_REWARD_QUALITY_GAP`, not an implementation mismatch.

## Heading

The frozen Stage 1B table reduced signed yaw bias, but its maximum paired fall
improvement was only 6.0 points and it was inconsistent
across speeds. Heading is `HIGH_SPEED_HEADING_SECONDARY`; it does not explain
the absent deterministic completion.

## Gradient consolidation

On the full 175,000-sample S100 rollout, completion density was
0.00857% and the completion actor
gradient was only
0.0738% of total.
An 8x virtual density still reached only
0.590%;
16x was the first tested factor above 1%.

The restored Adam direction had completion-descent cosine
0.023, but zero-moment Adam was
also only 0.087. This fails
the preregistered Adam-history causal gate. The primary problem is event
density and mean-policy consolidation, not optimizer history alone.

## Classification

**PHASE_A_BOUNDARY_MULTIPLE_CAUSES**

Secondary findings: PHASE_A_COMPLETION_EXPLORATION_ONLY, PHASE_A_COMPLETION_NOT_CONSOLIDATED_IN_MEAN_POLICY, PHASE_A_PERIODIC_GAIT_REWARD_QUALITY_GAP, PHASE_A_COMPLETION_SIGNAL_TOO_SPARSE, HIGH_SPEED_HEADING_SECONDARY, ADAM_MOMENT_ORTHOGONAL.

## Phase B readiness

**PHASE_B_NOT_READY.** Deterministic completion remains zero and is not
reproduced by two checkpoints.

## Next

One method only: **event-stratified on-policy minibatch construction
preflight**. It is not executed in Stage 2F.

## Repository

Starting HEAD: `03a92043d2c685ff48a51321f838b0f929761fa4`. Protected experiments and Stage 0-2E results
were not modified. New training checkpoints: 0. Production policy updates: 0.
Remote push: false.
