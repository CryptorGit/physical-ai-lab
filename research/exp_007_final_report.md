# exp_007 final report

## Final classification

```text
exp_007:
PARTIAL_SUCCESS_ASYMMETRIC_STATE_GRAPH

RUN_TO_WALK v1:
RUN_TO_WALK_V1_NO_GO
```

The experiment is a partial success, not a global No-Go. Independent
steady-state experts, both STAND/WALK directions, and the WALK-to-RUN
transition were established. The unavailable capability is the reverse
RUN-to-WALK v1 edge.

## Achieved

- Frozen STAND, WALK, and RUN_LOW steady-state experts.
- Formal `STAND_TO_WALK` and `WALK_TO_STAND` edges.
- State-contract-conditioned STAND/WALK round trips.
- Limited formal `WALK_TO_RUN` support from WALK 1.2 m/s to RUN_LOW 2.6 and
  2.8 m/s.
- Live in-place environment-ID cohorts, transition-only storage and GAE,
  ordered `selected_env_ids` gather/scatter, and frozen-controller protection.
- For RUN-to-WALK: valid RUN source generation, safe in-place handoff,
  RUN-cycle termination, and entry into WALK-compatible contact.

## Not achieved

- Persistent target-WALK basin retention after RUN.
- WALK contract acquisition for 0.40 seconds.
- WALK expert takeover and a full RUN-to-WALK edge.
- Graph-based `RUN -> WALK -> STAND` STOP.
- A bidirectional `STAND <-> WALK <-> RUN` graph.
- A full integrated locomotion sequence.

## RUN_TO_WALK v1 evidence

The parameter-free hard switch was rejected at Stage 8A: full-edge success was
46.7% from RUN 2.6 m/s and 40.0% from RUN 2.8 m/s, with long-dwell saturation
above 50%.

The learned transition infrastructure passed its live handoff, storage, GAE,
gradient, and protection gates. Stage 8C then completed its frozen 100-update
Pilot 1. Every deterministic checkpoint terminated the RUN cycle and detected
WALK-compatible contact in 100% of episodes, but achieved:

```text
WALK contract acquisition  0%
transition completion      0%
WALK takeover              0%
full edge                  0%
timeout                  100%
```

The Stage 8D parent re-evaluation used 40 valid episodes at 2.6 m/s and 40 at
2.8 m/s. Every trajectory reached exactly seven consecutive WALK-valid steps:
0.14 seconds versus the required 20 steps/0.40 seconds. Contact invalidated the
streak 1,539 times; speed invalidated it four times; heading, flight, and
safety accounted for none.

The existing WALK-acquisition-progress term fired in 100% of near-contract
episodes, but its correlation with maximum streak was zero because every
episode plateaued at the same seven-step maximum. Therefore doubling the
weight could not rank trajectories toward longer basin retention. The strict
reachability gate stopped Pilot 2 before any optimizer update.

The formal closure is:

```text
primary:
TARGET_WALK_BASIN_RETENTION_FAILURE

supporting:
WALK_ACQUISITION_REWARD_NOT_REACHABLE
```

This is not merely an infrastructure failure. The infrastructure succeeded;
the learned controller did not retain the target basin.

## Final capability graph

```text
STAND
  ↕
WALK
  └── WALK_TO_RUN ──→ RUN_LOW
```

- WALK commands: 0.6, 0.8, 1.0, 1.2 m/s.
- RUN_LOW steady-state commands: 2.4, 2.6, 2.8 m/s.
- WALK_TO_RUN targets: 2.6 and 2.8 m/s.
- WALK_TO_RUN 2.4 m/s: not supported.
- RUN_TO_WALK: No-Go v1.
- Graph-based STOP: blocked by RUN_TO_WALK.
- Full bidirectional locomotion graph: not achieved.

## Research findings

1. Specialized state experts plus independent directed transition experts can
   establish formal `STAND <-> WALK` connectivity.
2. Real WALK occupancy can be connected safely to an independent RUN expert at
   2.6 and 2.8 m/s.
3. Transition directions are not symmetric.
4. RUN-cycle termination and entry contact are insufficient; the target
   expert's acceptance basin must be retained.
5. A semantically meaningful intermediate state does not guarantee
   bidirectional connectivity. Contact phase, observation sufficiency, and
   action-distribution compatibility matter.
6. Strict staged gates prevented a reward-weight-only experiment that lacked
   a ranking signal.

## Next research pivot

The recommended successor is:

```text
exp_008_phase_aware_locomotion_transitions
```

Before new reinforcement learning, use Stage 8D trajectories to determine
whether the next contact break and required correction can be predicted from
the existing 152D observation. Predictability indicates an optimization
problem; failure indicates partial observability. Only then compare explicit
contact/history features, a GRU, or an explicit phase estimator. A larger
alternative is unified WALK/RUN trajectory distillation.

No successor work is implemented or authorized by this report.
