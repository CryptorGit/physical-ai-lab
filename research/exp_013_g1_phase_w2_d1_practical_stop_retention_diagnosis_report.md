# exp_013 Phase W2-D1 practical-stop retention diagnosis

## Outcome

The combined classification is **W2_PARENT_STOP_CAPABILITY_NOT_ESTABLISHED**.
The canonical parent already fails the formal practical-stop yaw contract.  Its translation
stops, but gait-period yaw oscillation remains.  Iteration 5 also changes the residual signed
yaw bias and therefore degrades the legacy quick-guard score, but it does not erase an
established formal practical-stop capability.

## Early guard

The clean quick evaluator uses 8 START and 8 STOP conditions, 20 deterministic episodes per
condition, yaw-zero sources only, a 3 s source hold, 1.5 s minimum-jerk ramp, and about 4.5 s
endpoint window.  STOP yaw is computed as `abs(mean signed yaw)`, whereas the formal practical
stop contract uses `mean(abs(yaw))`.  This permits gait-cycle cancellation.

- Parent legacy guard stop: 99.00%
- Iteration 5 legacy guard stop: 33.62%
- Parent formal stop: 0.00%
- Iteration 5 formal stop: 0.21%

The exact saved iteration-5 guard result is 219/320 (68.4375%): START is 160/160
and STOP is 59/160. A preregistered 200-batch stratified bootstrap gives a parent
aggregate mean of 99.49% (0% below 70%) and an iteration-5 mean of 67.02% (91%
below 70%). This is not a substitute for 200 extra simulator batches, and the
artifact labels that limitation. It is enough to show that ordinary episode
resampling does not make the parent and iteration-5 legacy scores exchangeable.

## Parent baseline and failure decomposition

Parent translation-stop success is 100.00%; yaw-stop
success is 0.00%. Mean final speed is
0.0463 m/s, while mean absolute yaw is
0.1057 rad/s.  Failures are therefore yaw residual plus periodic
stepping, not fall, slip, impact, or translation drift.

The exp_012 Stage 2Q forward positive control passes 98.00%
under the same evaluator, with mean speed 0.0095 m/s and mean
absolute yaw 0.0078 rad/s.  This validates that the threshold is
reachable in the shared physics/evaluator protocol.

## Timeline

Available policy artifacts are parent, iteration 1, and iteration 5. Iterations 2–4 are
telemetry-only and were not regenerated. The saved quick metric declines from 99.69% parent to
99.38%, 95.63%, 84.38%, 87.19%, and 68.44% at iterations 1–5. Static locomotion retention
remains intact. Formal mean-absolute-yaw stopping is absent at the parent, so this curve is not
evidence of losing an already-established formal stop skill.

## Time/profile diagnostics

Extending final hold from 2 to 12 seconds never produces a parent formal stop
(0% at every boundary); iteration 5 remains between 0% and 0.33%. Parent mean
absolute yaw stays at 0.1049–0.1060 rad/s and iteration 5 at 0.1064–0.1067
rad/s. This is endpoint-not-reached, not a four-second acquisition delay.

Changing the ramp from 0.25 to 4.0 seconds also leaves parent success at 0%
and iteration-5 success at 0–0.5%. Direct, yaw-first, translation-first,
two-stage-speed, and magnitude-only profiles all fail to recover the formal
stop. They retain mean absolute yaw near 0.106 rad/s. These profiles are
counterfactual only and were not adopted.

## Direction and yaw

The parent formal failure occurs for all 8 directions and all source yaw signs.
The legacy signed-mean score is 98.50%, 98.88%, and 99.63% for negative, zero,
and positive source yaw. Iteration 5 changes these to 41.88%, 35.13%, and
23.88%, respectively. Directional legacy scores span 29–39% at iteration 5,
but the formal mean-absolute-yaw failure is common rather than confined to one
direction/yaw attractor.

## Exposure and optimization

T1 assigns 40% steady retention, 30% start/stop, and 30% speed-change sequences. Within the
start/stop group, a Bernoulli split gives 15% start and 15% stop in expectation. Per-sequence
counts and condition-resolved advantages were not persisted, so they are marked
`not_recorded`, not inferred. Aggregate training gradients and reward terms remain finite and
stable, but cannot establish a start/stop-specific gradient conflict.

## State/action and positive control

The parent has no formal-success stop class, so a successful-versus-failed
parent stop-manifold AUROC is not evaluable. The exp_012 forward positive
control demonstrates a distinct low-motion endpoint: 98% success, 0.0095 m/s
mean translation speed, and 0.0078 rad/s mean absolute yaw.

Interpolating toward the exp_012 action for 1/2/4/8 control steps does not make
the W2 parent locally enter that endpoint. All 12,000 branch trials remain at
0% formal stop; the strongest intervention increases rather than reduces the
residual yaw. This is diagnostic-only and is not a teacher, runtime action
source, or recommended controller.

## Protection

No PPO continuation, shadow update, checkpoint creation, reward/curriculum/gate modification,
controller, teacher, action blend, or checkpoint switch was performed. Existing stages,
checkpoints, optimizers, samplers, physics, Isaac Lab, and RSL-RL remain unchanged. No remote
push was performed.

## Next

Run one practical-stop endpoint acquisition preflight from the canonical yaw-conditioned WALK
parent before restarting full W2.
