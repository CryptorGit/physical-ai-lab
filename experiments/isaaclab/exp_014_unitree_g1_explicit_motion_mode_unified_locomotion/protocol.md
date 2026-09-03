# Registered protocol

The command contract is `Exp014ExplicitMotionModeCommandV1` and the Phase 1 dataset is `Exp014StandOmniWalkTrajectoryDatasetV1`. Splits are recipe/episode level (70/15/15), validation alone selects a checkpoint, and held-out is opened once after selection. S0 is attempted first; S1 is authorized only after an S0 static-gate failure, and S2 only after S1 failure. Closed-loop failure after a static pass authorizes DAgger horizons 8, 16, and 32. Phase 3 is authorized only after full Phase 2 passage with at least two hours remaining.

Stop immediately on protected-path drift, split/future/teacher-ID leakage, non-finite values, material collisions at identical full input, post-hoc gate changes, or held-out fallback. Every retry changes one registered variable and a failure class receives at most two retries.
