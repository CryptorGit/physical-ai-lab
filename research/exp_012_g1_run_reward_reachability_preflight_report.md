# exp_012 G1 RUN reward reachability preflight

## Scope

Stage 2D is diagnostic-only. It used frozen checkpoints, long-horizon evaluation,
offline reward replay, positive controls, and diagnostic gradients. It performed
zero PPO updates and saved no training checkpoint.

## Implementation equivalence

exp_005 Stage 4 and exp_012 resolve to the same
`SafePeriodicFlightReward` callable and the same command, speed, tilt,
vertical-speed, flight-duration, landing-alternation, precursor-cap, completion,
and overlong-flight parameters. Semantic difference count is **0**.

The persistent per-environment state is `_was_in_flight`, `_flight_duration`,
`_event_precursor_reward`, and `_last_landing_foot`. It resets only for actual
environment reset IDs. A PPO rollout boundary does not reset it. Continuous and
24-step chunked replay matched exactly (maximum reward difference 0).

Contact order resolved to left ankle then right ankle, threshold 1 N. The exp_005
Stage 4 positive control validates landing-edge and alternation detection.

## Exposure

Stage 2C did not terminate before reaching RUN: RUN_HOLD samples were essentially
100% at requested speed >=2.3 m/s, and SEQUENCE samples were 98.4-99.2% in the
RUN band. A 24-step batch spans only 0.48 s, so it cannot independently certify a
0.5 s dwell, but it sampled already-active RUN segments.

The Stage 2D long-horizon protocol used 50 deterministic episodes per condition,
10 s direct/ramped and 18 s bidirectional trajectories.

## Gate cascade

At iteration 100, long-horizon evaluation produced 273 landing
candidates, 2 takeoff-precursor steps, 11
safe-flight reward steps, and **0 completions**. The dominant
first failures were precursor speed and tilt. Candidate events generally failed
multiple gates; this is not a one-threshold near miss.

## Positive controls

The exp_005 Stage 3 negative control produced 0 completion events. The
exp_005 Stage 4 positive control produced 5601 completion events under the
exp_012-resolved telemetry/reward path. A stochastic Stage 4 trace produced
2115 completion samples.
Thus the shared implementation can detect and reward periodic alternation.

## Gradient strength

At iteration 100, the base component gradient norm was
45.0879; precursor/run-specific was
0.1226 (0.272%
of base); completion was exactly 0 because no completion event existed. On the
positive-control trajectory, completion gradient norm was
19.9682, versus precursor
4.1695. Completion is learnable once
reached, but exp_012 never enters that event basin.

## Classification

**RUN_PRECURSOR_ONLY_NO_COMPLETION_BASIN**

Secondary findings are `RUN_ACTION_MANIFOLD_NOT_REACHED` and
`RUN_REWARD_SIGNAL_TOO_WEAK_RELATIVE_TO_BASE`. The primary label follows the
direct observation that precursors and safe flight exist while alternating
completion remains zero across parent/iteration 50/100/300 diagnostics.

## Next

**two-stage single-policy RUN-acquisition continuation preflight**

Use the same single checkpoint: Phase A concentrates on the 2.3-2.6 m/s
transition band to acquire safe periodic RUN, then Phase B returns it to joint
ZERO/WALK/RUN/SEQUENCE retention. No runtime checkpoint switching is proposed.
