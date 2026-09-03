STATUS:
STAGE_2_PILOT_1_RETRY1

PRIMARY HYPOTHESIS:
One speed-conditioned G1 policy can jointly retain STAND and WALK,
acquire periodic RUN_LOW, and execute bidirectional transitions
without checkpoint switching.

TRAINING YAW CONTRACT:
yaw-rate command fixed at zero.
All external yaw controllers disabled.

YAW CANCELLER:
Frozen-checkpoint diagnostic only.
Not used during PPO training.

DISTINCTION FROM EXP_009:
No post-hoc teacher distillation.
No teacher identity.
No expert router.
One continued checkpoint from the exp_005 Stage 2 walking parent.

This package owns the experiment registration, command curriculum, frozen
heading controller, reward composition, training, evaluation, and playback.
It imports the unchanged exp_005 task only as the provenance source for the
parent reward and the existing Stage 4 periodic-running reward.

STAGE 2A RESULT:
The first Pilot 1 update used a mismatched runtime learning rate.
The optimizer restored 2.25e-5, but runtime state overwrote it with 0.001.

STAGE 2B:
Synchronize all runtime and scheduler LR state from the restored optimizer
before any optimizer or adaptive-KL operation.

PILOT RETRY:
Not executed in Stage 2B.

RESUME FIX:
Runtime and scheduler learning-rate state are synchronized from
the strictly restored optimizer before any rollout or adaptive-KL update.

TRAINING YAW CONTRACT:
Yaw-rate command fixed at zero.
All external yaw controllers disabled.

SINGLE-WEIGHT CONTRACT:
One actor checkpoint.
No expert router, checkpoint switching, teacher action, or action blending.

STAGE 2 PILOT 1 RETRY 1 RESULT:
The single authorized retry completed 300 iterations. WALK was retained, but
STAND narrowly regressed and periodic RUN_LOW plus the run-boundary transitions
did not meet the formal gates. Classification:
G1_SINGLE_POLICY_MULTIPLE_FAILURES.

## Stage 2I — Reverse single-policy continuation Phase R1

STAGE 2H:
Completion-event reuse from the WALK parent was closed as NO EFFECT.

STAGE 2I:
Reverse single-policy continuation from the exp_005 Stage 4
RUN-capable checkpoint.

TARGET:
Recover 1.2m/s WALK and bidirectional WALK↔RUN
while preserving periodic RUN in one actor checkpoint.

RUNTIME:
No checkpoint switching, router, teacher, or action blending.

## Project closure

```text
STATUS:
CLOSED

PROJECT-LEVEL CLASSIFICATION:
EXP_012_CLOSED_WITH_SINGLE_POLICY_LOCOMOTION_SUCCESS_AND_STRICT_STAND_LIMITATION
```

Single-checkpoint WALK/RUN gait selection and bidirectional transition were
achieved. The final Stage 2Q actor retains WALK from 0.6 to 1.2 m/s and RUN
from 1.2 to 2.6 m/s, including WALK→RUN, RUN acceleration/deceleration, and
RUN→WALK. It reaches a practical near-stop after the sequence.

The strict static-contact STAND definition remains unresolved: small stepping
and contact oscillations violate the zero-flight/final-double-support gate.
Stage 2R specialist positive controls also failed that same strict gate. This
project-level closure does not overwrite any prior stage classification.

Final report:
`research/exp_012_g1_single_policy_bidirectional_locomotion_final_report.md`
