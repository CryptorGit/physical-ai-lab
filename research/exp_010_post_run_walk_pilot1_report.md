# exp_010 POST_RUN_WALK Pilot 1

## Question

Can a distinct full-action `POST_RUN_WALK` steady-state expert retain safe
1.2 m/s locomotion for eight seconds after RUN-cycle termination and first
WALK-compatible contact, without matching the original WALK expert?

## Frozen protocol

The source route used the frozen STAND, STAND_TO_WALK, WALK, WALK_TO_RUN, and
RUN_LOW controllers, followed by frozen Stage 8C `model_10.pt`. The
`PostRunWalkExpert152` was a strict deep copy of that checkpoint and produced a
full 37-D action. Source preparation used no gradient and no PPO storage.
Handoff preserved the physical environment ID, global previous action, contact
history, and sensor history; no setter, teleport, or state copy was used.

The steady-state reward contained only 1.2 m/s tracking, heading, upright,
stable support, excessive-flight suppression, slip, impact, saturation, fall,
and action rate. It contained no original-WALK action imitation or acceptance
bonus.

## Result

`POST_RUN_WALK_STATE_FAIL`

Source formation remained reliable (mean 90.06% of physical environments, with
all selected 512-member cohorts satisfying the source contract) and all chosen
segments reached the first compatible contact. The initial actor nevertheless
returned to periodic RUN in 100% of deterministic episodes. Across durable
checkpoints `first_post_update`, 10, 25, 50, and 75, acquisition and eight-second
hold were 0% for both 2.6 and 2.8 m/s sources.

Learning ended periodic RUN but did not create a safe low-speed attractor.
Earlier checkpoints were dominated by excessive flight; later checkpoints were
dominated by long-dwell saturation and increasing heading error. At iteration
77 the online deterministic-equivalent outcome remained 0% acquisition and
100% saturation. Value loss, KL, and gradient norms diverged while trainable
exploration standard deviation rose from 0.20 to 0.71 mean. The next iteration
crossed the frozen 0.75 maximum and stopped fail-closed. No model after the last
durable checkpoint was adopted.

## Decision

Pilot 1 does not justify either `POST_RUN_WALK → STAND` or
`POST_RUN_WALK → original WALK`; both remain blocked until the state itself
passes. Pilot 2 was not run. The single next action is an optimization-stability
preflight that explains the exploding value/KL/std behavior before deciding
whether the experiment's second and final Pilot is warranted.

No capability manifest or production artifact was changed.
