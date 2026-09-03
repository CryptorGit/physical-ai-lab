# exp_014 Phase 2-D16 dedicated START specialist report

## Outcome

`EXP014_D16_FORWARD_START_FAIL`. Training stopped at update 40 because C1_FORWARD acquisition remained 0%. C2–C5, trained formal-34 selection, local-neighborhood evaluation, process parity, and held-out sealing were not executed.

## Architecture and parity

The single composite actor contains the frozen 124D W_MOVE base and a trainable 141→512→512→256→37 residual. The residual head was strictly zero initialized and bounded by `0.50*tanh`. Initial action and post-1.5-second steady action were bitwise identical to W_MOVE (maximum difference 0). W_MOVE's tensor hash was unchanged.

## Reward and stability

Velocity/yaw weights were 6.0/8.0. Their combined gradient was 85.16% of the total; the yaw gradient norm was 41.901733. Regularization/tracking norm ratio was 0.4737%. Update 1 passed: exact KL 0.000324, max KL 0.001263, clip fraction 0.000000, final-action shift 1.716262, fall 0%, slip 0.0924%, and matching temporary/persistent tensor hashes.

## C1 result

Forty updates produced 1,904,000 interactions. At the fixed D15 validation starts, forward acquisition, conditional steady hold, joint success, and end-to-end success were all 0%. Fall was 4.9020%, dangerous slip 10.7843%, and torque saturation 17.6471%. Residual saturation dwell was 0%, so the preregistered ±0.75 repair was not authorized.

The stopping checkpoint's active residual L2 mean was 0.111332, maximum absolute component 0.164873, with exact W_MOVE parity after 1.5 seconds.

## Protection

The 476 train recipes yielded 473 valid source snapshots; all three invalid sources remained recorded but were excluded from PPO as required. Validation retained all 102 D15 snapshots with 102/102 identical hashes and 101 valid starts. No prior checkpoint, dataset, physics, reward config, command/observation contract, D6–D15 artifact, RUN system, or integrated Student was changed. Remote push was not performed.
