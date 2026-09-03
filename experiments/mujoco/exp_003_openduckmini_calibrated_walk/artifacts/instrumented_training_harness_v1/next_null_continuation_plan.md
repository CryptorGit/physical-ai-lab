# Next Null-Continuation Plan

Status: **designed, not authorized while HARNESS_FAIL remains**.

- Parent: v52 package (v45 actor plus v50/v52 reverse profiles and the frozen
  calibrated teacher/residual route).
- Objective: old objective, unchanged.
- Seeds: at least 3.
- Pilot budget: 250k environment interactions per seed.
- Update-boundary checkpoints: 0, 50k, 100k, 250k. Filenames and manifests must
  contain the actual interaction count if an update crosses a threshold.
- Fixed: network, optimizer hyperparameters, sampler, teacher, curriculum,
  scene, domain randomization, delay, head route, PPO configuration.
- Observe: parent retention, between-seed reproducibility, initial fresh-Adam
  update/moment norms, exact `P(vx,vy,yaw,head)`, valid PPO samples by command
  and yaw sign, deterministic/stochastic falls.
- Gate: all three seeds retain the parent within the predeclared linear/yaw/fall
  tolerances and the harness exact-resume/identity gates have already passed.
- No-Go: any deterministic retention fall, >10% primary linear degradation,
  unstable stochastic fall increase, material yaw-sign exposure imbalance, or
  failure of checkpoint/restart identity.

Do not return to yaw-objective redesign until this null continuation is stable.
