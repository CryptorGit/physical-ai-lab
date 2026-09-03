# POST_RUN_WALK protocol

The physical source route is:

`RESET → STAND → STAND_TO_WALK → WALK@1.2 → WALK_TO_RUN@2.6/2.8 →
RUN_LOW contract → frozen Stage 8C model_10 RUN-cycle termination →
first WALK-compatible contact`.

The same physical environment ID is retained. State setters, teleports,
snapshots, state copying, source-prefix storage, and controller blending are
forbidden. The actually applied Stage 8C action remains the global previous
action at handoff.

`PostRunWalkExpert152` is a strict deep copy of Stage 8C `model_10.pt`. It
produces the full 37-D action and is optimized as a separate steady-state
expert. It is not a residual and has no WALK action-alignment objective.

The target contract is safe 1.2 m/s locomotion with speed error at most
0.20 m/s, heading error at most 0.12 rad, no periodic RUN, no excessive
flight, stable alternating/single support, and no fall, dangerous slip,
impact failure, or long-dwell saturation. Acquisition requires 0.4 seconds;
success requires 8.0 continuous seconds.

Stage 2 is a read-only optimization-stability preflight over Pilot 1 logs,
rollout replay, and fixed checkpoints. Its `OPTIMIZATION_FAILURE_MULTIPLE`
gate prohibits Pilot 2: reward/advantage directionality and critic/mean-policy
stability fail independently, while exploration std is contributory but not
the initiating failure. `POST_RUN_WALK_V1` is closed as `NO_GO`.
