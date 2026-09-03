# EXP 012 — G1 single-policy bidirectional locomotion

## Outcome

The experiment stopped fail-closed at the required parent heading-response
preflight. The classification is `G1_YAW_RATE_NOT_LOCALLY_CONTROLLABLE`. Pilot 1, checkpoint
selection, formal evaluation, and 2.8 m/s diagnostics were not run.

## Parent

The exact parent is `model_4246.pt`, SHA-256
`734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621`.
Its actor is 123→256→128→128→37 and its critic is 123→256→128→128→1.
The 17-state Adam mapping is present at step 85,000 with learning rate
2.25e-5. Strict actor, critic, std, deterministic-action, and optimizer
mapping identity passed in the 16-environment wiring run.

## Curriculum and reward

The preregistered distribution is ZERO_HOLD/WALK_STEADY/RUN_HOLD/SEQUENCE
= 20/20/20/40. A 100,000-schedule audit passed all ±1% and 1.10 ratio gates.
The parent Stage 2 reward is retained; the only resolved semantic addition is
the existing exp_005 Stage 4 `safe_periodic_flight` term, statically gated to
requested vx ≥2.3 m/s.

## Heading preflight

At 0.6 m/s the parent responded monotonically and with the requested yaw-rate
sign. At 0.0 m/s and 1.2 m/s it did not. STAND showed 5–10% falls in some
small-yaw conditions, and 1.2 m/s failed the negative-command sign contract
and showed a 5% fall rate at +0.10 rad/s. This violates the explicit
precondition, so the phase-gated controller cannot be treated as frozen and
safe for this Pilot.

## Scientific interpretation

This result does not test or refute the unified-policy hypothesis. It isolates
an earlier prerequisite failure: the chosen parent does not have the required
local yaw-rate command controllability across STAND, WALK 0.6, and WALK 1.2.
The next single method is a G1 yaw-rate controllability diagnosis before any
single-policy Pilot.

## Repository

Starting HEAD: `60028d13a5534527835e215c37106ea107585b39`.
Existing exp_005–exp_011 results, capability manifests, production artifacts,
Isaac Lab core, and the parent checkpoint were not modified. Remote push is
false. Unrelated pre-existing dirty paths remain untouched.
