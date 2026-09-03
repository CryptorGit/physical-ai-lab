# EXP 013 Phase W1B-D1: yaw/translation interference diagnosis

## Scope

This is a read-only diagnosis of the canonical W1A2 iteration 80 actor and the diagnostic W1B
iteration 1 actor. No PPO update, checkpoint write, reward/curriculum change, or promotion was made.
The existing W1B classification remains `EXP013_W1B_TRAINING_UNSTABLE`.

## Early-guard parity

The online probe used deterministic mean actions, not WALK exploration. Its contract nevertheless
differs materially from the fresh evaluator: it reuses the training W1B environment after rollout
and update, keeps inherited observation corruption and disturbance events active, interleaves
22 conditions modulo 1024 environments, continues the training RNG stream, and records only an
aggregate PASS count. The fresh evaluator creates a new DirectionalBaseline environment with
observation corruption, base-force, and push events disabled, uses block allocation, and starts a
new RNG stream.

Fresh parent/iteration-1 results were 16/16 and 16/16. The reconstructed online
paths were parent 11/16, iteration 1 11/16, and warm iteration 1 16/16.
The 100-batch deterministic variance study gives P(PASS≤11) =
0.000 for iteration 1. Order/reset testing
classified contamination as `NO_REPRODUCIBLE_STATE_CONTAMINATION`.

Early-guard classification: `ONLINE_EARLY_GUARD_EVALUATOR_MISMATCH`.

## Parent yaw surface

At 0.3 m/s, front/left sectors favor negative yaw (mean success 1.000 versus positive
0.140), while rear-right sectors reverse that relation (positive 0.750 versus
negative 0.487). Pure +0.3 rad/s remains weak while pure -0.3 rad/s is strong. Linear fits
show direction-dependent gain/dead-zone behavior rather than one global positive-yaw deficit.

## Mirror, contact, gradient, and critic

The robot mirror map is uniquely derived from live joint pairing, limits, default pose, PD gains,
and action scale. Actor counterfactual mirror classification is `POLICY_YAW_ACTION_ASYMMETRY`. Formal
episodes begin in flight; unsupported phases were not inferred. The mean zero-yaw versus
yaw actor-gradient cosine is -0.0256. The 24-step critic mean absolute value bias is
0.1012; neither critic nor reward wiring explains the direction reversal as a primary cause.

Yaw/translation classification: `PARENT_DIRECTION_CONDITIONAL_YAW_ASYMMETRY`.

## Conclusion

Combined classification: `W1B_FALSE_EARLY_STOP_WITH_PARENT_YAW_ASYMMETRY`.

The canonical translation-only WALK artifact remains W1A2 iteration 80
(`bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244`).
W1B iteration 1 remains diagnostic-only and is not promoted.

Next action (not executed here): repair online/fresh evaluation parity, then rerun the original W1B curriculum once from canonical W1A2 iteration 80.
