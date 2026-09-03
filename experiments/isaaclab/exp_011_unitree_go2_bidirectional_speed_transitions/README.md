STATUS:
STAGE_1_BASELINE

PRIMARY HYPOTHESIS:
A single continuous speed-conditioned Go2 policy can support
bidirectional acceleration and deceleration without expert switching.

G1 RESULTS:
PROTECTED / NOT MODIFIED

## Stage 7 low-speed gait stabilization

Stage 7 resumes the Stage 4 selected checkpoint and its matching optimizer state.
The only causal change is a low-speed command curriculum; reward, network,
observation, action, physics, and PPO settings remain frozen. Formal evaluation
continues to use the unchanged `GO2_ENDPOINT_EVALUATION_V1` protocol.

# Exp 011 — Unitree Go2 bidirectional speed transitions

This experiment audits the official Isaac Lab Go2 flat-velocity environment and
checkpoint, then evaluates one deterministic continuous speed-conditioned policy.
There is no expert routing, checkpoint switching, action blending, residual,
scripted motion, state injection, reward change, or PPO update.

## Stage 1 result

Classification: `GO2_STEADY_STATE_ENVELOPE_INSUFFICIENT`.

The official policy strictly loads with a 48D observation and 12D position-action
contract. Zero-command hold fails (86% success, 14% fall). Nonzero tracking is
useful diagnostically through 2.0 m/s, but no tested nonzero point passes every
frozen steady-state safety gate; 2.5 m/s is also outside the documented training
range and misses the speed-error gate. Consequently no transition is formally
gate-eligible and neither full nor limited sequence is executed.

The only recommended next step is a new single continuous 0–2.0 m/s Go2 base
policy. Stage 2 robustness evaluation is not authorized until STAND and required
steady endpoints pass.

## Reproduction

```powershell
.\experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions\scripts\run_stage1_baseline.ps1 -AuditOnly
.\experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions\scripts\run_stage1_baseline.ps1
.\experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions\scripts\play_exp011_go2_bidirectional.ps1 -Mode FullSequence -Seed 20260901
```

Formal outputs are under
`results/exp_011_unitree_go2_bidirectional_speed_transitions/stage1_single_policy_baseline/`.

## Stage 2 result

Classification: `GO2_TRAINING_UNSTABLE`.

The official checkpoint warm-start passed bitwise actor, critic, deterministic
action, distribution, log-std, and normalization checks. Optimizer state was
intentionally not inherited. The 100,000-segment command-distribution audit and
the zero-difference reward/config freeze audits also passed.

Pilot 1 stopped fail-closed after its first 2,048-environment PPO update:
approximate KL was 0.51294 against the frozen 0.20 maximum, with clip fraction
0.78412 and KL from the initial policy 0.21220. No threshold was relaxed.
Iterations 2--300, checkpoint validation, formal evaluation, reduced sequence,
2.5 m/s diagnostics, and selected-checkpoint GUI playback were not run.

The only recommended next action is first-update PPO stability diagnosis with
the frozen Stage 2 contract. Pilot 2 is not authorized by this result.

## Stage 3 result

Classification: `FIRST_UPDATE_FRESH_OPTIMIZER_MISMATCH`.

The saved-batch identity and Gaussian KL estimator checks pass. The Stage 2
distribution change is real and 99.94% actor-mean dominated, but the causal
shadow comparison identifies fresh Adam state as the primary failure. Restoring
the official checkpoint optimizer state yields exact KL 0.01381 and clip
fraction 0.19289 on the same batch and minibatch order.

Readiness is `PILOT_READY_WITH_SINGLE_STABILITY_FIX`. The only permitted next
change is `resume checkpoint optimizer state`; Stage 3 did not run that Pilot.

## Stage 4

Stage 4 restores the official checkpoint actor, critic, standard deviation,
normalizer, Adam moments, Adam step 20,000, terminal learning rate, and source
iteration 999. The environment, reward, symmetric command curriculum, PPO
hyperparameters, seed, 2,048 environments, and 300-local-iteration budget are
identical to Stage 2.

The first real resumed update passes with exact KL 0.01453 and clip fraction
0.20186, confirming the Stage 3 optimizer-state diagnosis. Checkpoint selection
and formal endpoint evaluation remain separate from this optimizer result.

Stage 4 completed all 300 local iterations (14,745,600 interactions) and selected
iteration 50 by the frozen validation precedence. The formal classification is
`GO2_ENDPOINT_FAILURE_MULTIPLE`: zero-command tracking and all four directional
transition acquisitions completed without falls, but the fixed stand posture
limits failed, every moving condition failed the dangerous-slip metric, and
0.4 m/s also failed fall and heading limits. The reduced sequence and 2.5 m/s
extrapolation were therefore not run. No Stage 4 checkpoint was promoted to a
production artifact.

The single recommended next action is `endpoint failure diagnosis before
another pilot`.

## Stage 5 result

Classification: `GO2_ENDPOINT_EVALUATOR_MISMATCH_PRIMARY`.

Stage 5 performed paired deterministic diagnostics only. It found that current
Isaac Lab exposes `root_quat_w.torch` in `xyzw` order while the historical
exp_011 evaluator decoded it as `wxyz`; this directly explains the false
near-pi roll result during successful zero-command stance. Reset settling also
dominated the historical height range, while the settle-after-2s height range
and nominal-relative tilt were stable.

The moving-foot result is not merely a one-step boundary artifact. Both the
official parent and Stage 4 policy show sustained foot-link-origin motion under
the contact-history mask, and the official Go2 Flat reward does not include a
feet-slide term. Stage 4 does not uniformly worsen the paired severity.

The 0.2--0.5 m/s band contains real fall/yaw instability and stand-like
stepping/locomotion bifurcation; the fixed-seed 0.4 m/s failure was visually
confirmed. The only next action is to freeze a corrected Go2-specific endpoint
evaluation protocol and rerun Stage 4 formal evaluation without retraining.

## Stage 6 endpoint evaluation protocol

Stage 1–4 endpoint posture metrics used an invalid quaternion ordering
and are retained only as legacy reproducible results.

Stage 6 uses `GO2_ENDPOINT_EVALUATION_V1` with:

- xyzw quaternion decoding
- fixed settling windows
- corrected heading handling
- contact-point-based physical slip

The protocol is frozen and hashed before formal rollout. Stage 6 evaluates the
unchanged Stage 4 selected checkpoint and official parent with paired seeds;
it performs no PPO update, reward change, curriculum change, or checkpoint
mutation.
