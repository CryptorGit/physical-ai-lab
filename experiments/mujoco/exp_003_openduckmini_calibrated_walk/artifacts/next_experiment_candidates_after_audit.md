# Next experiment candidates after the v52–v59 audit

These are designs only. No training, evaluator modification, or policy promotion was performed.

## Parent selection

Keep **v52 as the adopted package**. For any learned-policy pilot, use the exact actor checkpoint that v52 packages (`v45_step_47349760`, SHA-256 `52205E…F3C`) together with the frozen v52 reverse profiles and calibrated/backlash scene. Do not use v59 as an automatic parent.

Reason: v54–v57 are separate failed branches; v58 is stable only under a configuration whose direct reverse output collapses; v59 adds a mathematically overshoot-seeking objective. v52 is the only candidate with a recorded, calibrated/backlash, hybrid-controller acceptance history, although its 0.003-rad/0.01-m/s reverse-only v52 validation is weaker than the later 0.03-rad/0.1-m/s release records and does not imply hardware readiness.

## Experiment A — evaluation-equivalence counterfactual (zero training)

- Parent: v59 step 33,423,360 for diagnosis only; v52 remains adopted.
- One changed item: evaluator actuator/scene path. Run the exact calibrated/backlash model and the same reverse teacher+residual routing used during training. Remove the legacy positive-yaw lateral observation compensation for this counterfactual.
- Fixed: ONNX bytes, 19 commands, seeds 0–4, 15 s, initial noise, initial velocity, normalization, fall gates.
- Training budget: 0.
- Pilot: deterministic seed 0, 3 commands only: reverse straight, yaw left, maximum reverse-right; 2 s, then 6 s if wiring hashes and observation snapshots agree.
- Formal evaluation: 19×5×15 s only after pilot equivalence checks pass.
- Success gate: first-step observation and motor target match a frozen MJX reference within numeric tolerance; no scene/command argument omitted from output JSON; reverse result changes in the direction predicted by teacher routing.
- No-Go: any observation-order, normalization, action-history, phase, backlash, or target mismatch; any attempt to interpret this diagnostic as v59 qualification.
- Failure interpretation: if matched routing still gives reverse static/falls, the cause is not the current evaluator scene gate alone and the teacher/residual controller itself is unstable.

## Experiment B — yaw-objective isolation pilot

- Parent: v52 package actor checkpoint (`v45_step_47349760`), not v59.
- One changed item: replace the shared, unbounded `command_progress` scalar with a yaw objective whose optimum is at commanded yaw. Linear reward, sampler, teacher, observations, network, randomization and termination remain frozen.
- Fixed: exact calibrated/backlash source snapshot, seed 0, 4096 envs, PPO hyperparameters except the learning rate chosen before launch, v52 teacher profiles.
- Training budget: wiring smoke only, then 2.62M-interaction pilot; hard cap 5.24M if the first gate passes.
- Pilot: evaluate yaw-only ±0.6 and forward+yaw ±0.3 at 3 seeds×10 s plus reverse straight and maximum reverse-right sentinels.
- Formal evaluation: matched-pipeline 19×5×15 s and a separate explicitly timed push-recovery suite whose force/impulse event is logged.
- Success gate: yaw overshoot ratio 0.8–1.2 for every yaw command, zero pilot falls, no degradation greater than 0.02 m/s in the linear sentinels.
- No-Go: overshoot ratio >1.5 at the first checkpoint, any 3/3 fall condition, or reward-term telemetry missing.
- Failure interpretation: if overshoot remains with a command-centered yaw optimum, observation/action routing or teacher asymmetry is more likely than reward competition.

## Experiment C — reverse actor-role isolation

- Parent: v52 package actor checkpoint and frozen v52 profiles.
- One changed item: choose and enforce one reverse control contract: actor-as-residual with the teacher present in both evaluation and deployment. Do not simultaneously change reward, sampler, or observation.
- Fixed: v52 reward baseline for the pilot, all PPO parameters, command distribution, scene, delays/noise, profiles, and evaluation conditions.
- Training budget: 0 for controller-equivalence smoke, then at most 2.62M interactions if residual adaptation is required.
- Pilot: STAND-start reverse straight and reverse yaw ±0.1/±0.3, seeds 0–2, 10 s; record teacher target, residual, final motor target and phase per step.
- Formal evaluation: 19×5×15 s plus a logged STAND→reverse transition sequence without reset.
- Success gate: reverse straight speed ratio ≥0.6, direction cosine ≥0.8, maximum reverse-right zero falls, and exact actor/teacher/motor-target traceability.
- No-Go: direct ONNX targets are evaluated while training still forces teacher targets; missing per-step routing trace; any v59 promotion.
- Failure interpretation: failure under identical residual routing implicates teacher/profile asymmetry or insufficient stabilization, not missing standalone transition learning.

No ramp-observation experiment is recommended now: the audited path has no command ramp. No capacity expansion is recommended: capacity has not been isolated from the objective and routing defects.
