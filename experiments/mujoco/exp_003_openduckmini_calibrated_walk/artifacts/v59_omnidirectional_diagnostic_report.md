# v59 omnidirectional retraining report (not qualified)

Date: 2026-07-30

## Outcome

The all-direction curriculum and video render completed, but no v54-v59
checkpoint met the release criteria.  The existing v52 release remains the
deployment candidate.  Do not deploy the v59 diagnostic policy.

## Training performed

- v54: 7.9M steps, rejected after static-collapse behavior.
- v55: 2.6M steps, rejected because the turn teacher was being overwritten.
- v56/v57: 13.1M combined diagnostic steps with the real v52 turn teachers,
  rejected because negative tracking costs incentivized early termination.
- v58: 60.3M steps with symmetric dense tracking and an explicit termination
  penalty.  Stable checkpoints emerged, but reverse motion collapsed to stand.
- v59: 41.8M continuation steps from the best stable v58 checkpoint with a
  stronger command-progress reward.  Motion returned, but yaw overshot and
  reverse-right remained unstable.

The final two-stage v58/v59 training therefore contains 102.1M new steps.

## Strong-perturbation evaluation

Evaluation used 19 commands, five seeds, 15 seconds, 0.03 rad initial joint
noise, and up to 0.1 m/s initial base speed.

The best-reward v59 checkpoint (33,423,360 steps) still had:

- 20% falls for in-place yaw left;
- 20% falls for in-place yaw right;
- 100% falls for reverse + yaw right 0.3;
- in-place yaw magnitude about 1.17-1.22 rad/s for a 0.6 rad/s command;
- compound yaw magnitude about 0.82-0.93 rad/s for a 0.3 rad/s command;
- reverse straight speed approximately 0 m/s.

Authoritative evaluation:
`v59_eval_step_33423360_19x5_15s.json`

## Diagnostic artifacts

- Policy: `omnidirectional_v59_step_8355840_diagnostic.onnx`
- Policy SHA-256:
  `F6EC2ACC78D725C3D0264F78BFAF6C4BC3605A64EC54069B21FAEADBAF8003A3`
- Combined 19-pattern video:
  `videos_v59_omnidirectional_diagnostic_not_qualified/all_19_directions_v59_diagnostic_not_qualified.mp4`
- Video SHA-256:
  `CBF9CD3C3552ED9BF4C4C48206F3FD3E83265D1E694E48B3D6BF68966246A2CC`

All 19 nominal six-second renders completed without falling.  This is a
visual diagnostic only and does not override the perturbation failures above.

## Required next iteration

The next training run should separate linear-progress and yaw-progress scales,
add a command-transition/ramp state to the observation, and train recovery
from a settled stand before enabling the reverse periodic teacher.  A policy
must not be released until all 19 commands and the transition sequence pass
the strong-perturbation suite with zero falls and bounded tracking error.
