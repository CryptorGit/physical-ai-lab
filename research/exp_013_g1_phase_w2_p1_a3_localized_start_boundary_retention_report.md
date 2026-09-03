# Exp 013 Phase W2-P1-A3 — localized start-boundary trajectory retention preflight

## Outcome

Classification: `START_BOUNDARY_NO_JOINT_STATIC_SOLUTION`.

The preregistered two-step boundary-retention grid produced no candidate that simultaneously passed B0/B1 and every existing static group. Consequently, validation physical branches, zero-command retention, held-out authorization, formal closed-loop evaluation, DAgger, checkpoint creation, and promotion were not authorized.

## Immutable boundary views

The read-only `G_START_BOUNDARY_2STEP` view uses time index 0 (B0, exact-zero physical and actor command) and time index 1 (B1, first minimum-jerk nonzero command) from every accepted start episode. There are 2,373 episodes and 4,746 index entries: 3,786 train, 480 validation, and 480 held-out. `G_START_NONBOUNDARY` references time indices 2–54. No tensor was copied or reserialized, and all 13 chunk hashes match the resolved immutable manifest.

## Probe grid

The grid used boundary weights 2.5%, 5%, 10%, and 15%; learning rates 2e-5, 5e-5, and 1e-4; checkpoints at 250, 500, 1,000, and 2,000 optimizer steps; Adam; seed 20278121; and gradient clipping at 10. This gives 12 in-memory runs and 48 evaluated candidates. Persistent checkpoint writes were zero.

Every candidate passed B1. None passed B0 (`MSE <= 0.001`). The best B0 result was `0.0426297` at W15/LR1e-4/500, but it regressed steady stop to `0.0030798` and stop recovery to `0.0020401`. The closest candidate that retained all existing groups was W5/LR1e-4/500: B0 `0.0590198`, B1 `0.00001419`, stop recovery `0.0009072`, steady stop `0.0004383`, and every moving/nonboundary group PASS.

## Static trade-off

Low boundary weights (2.5–5%) preserve stop, moving, and nonboundary-start groups but leave B0 between approximately 0.059 and 0.065. Higher weights (10–15%) improve B0 only to approximately 0.043–0.052 while pushing stop recovery and/or steady stop above 0.001. Additional steps within the preregistered 2,000-step horizon do not close the gap.

For the closest existing-retention candidate, B0 loss contributions are 89.07% upper body, 10.91% lower body, and 0.02% waist. Nevertheless, A2 physical ablations established that upper-body-only or lower-body-only substitution cannot enter the safe basin. The remaining error is therefore whole-body and cannot be authorized through joint-specific weighting in this stage.

## Gradient conflict

At the frozen step-37,000 initialization, boundary-gradient cosine is -0.923 against steady stop and -0.863 against stop recovery. After the closest existing-retention probe these become -0.998 and -0.981. Boundary versus moving retention is positive but small (0.086 after the probe); boundary versus start nonboundary is -0.466. Increasing boundary fit directly suppresses the stop-maintenance/recovery solution.

## Physical and held-out authorization

The protocol requires simultaneous existing-group and B0/B1 static PASS before any physical branch. Candidate count was zero, so no A3 physical rollout occurred. A2 positive controls are retained as reference only: the frozen student one-step branch produced 89.60% endpoint and 9.29% falls; canonical/W1B two-step produced 99.52% endpoint and 0.46% falls; four-step produced 99.75% endpoint and 0.25% falls. These controls were not used to bypass the A3 static gate.

Held-out data were not evaluated, no fallback candidate was selected, and no candidate state was serialized.

## Interpretation and next action

The B1 first-nonzero state is already representable with high accuracy. The unsolved component is B0: it supplies a future-start W1B label at an observation whose current command and state also support stop maintenance. Under the unchanged objective, increasing B0 retention necessarily degrades steady-stop and recovery imitation before B0 approaches its gate.

The candidate remains W2-P1-R2 step 37,000, diagnostic-only. Closed-loop authorization is not granted and W1B-R2 iteration 200 remains canonical.

One next action only: retain fail-closed status and diagnose/version the B0 stop/start label contract before any formal localized integration. Architecture, command contract, physical gates, and immutable dataset remain unchanged in this preflight.
