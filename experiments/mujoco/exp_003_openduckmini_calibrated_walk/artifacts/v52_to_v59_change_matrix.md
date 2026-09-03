# v52–v59 change matrix

Audit date: 2026-07-30. “Confirmed” means a launcher, log, manifest, checkpoint, or current source supports the entry. Historical source-level changes are marked unverified because the run directories contain neither a resolved config nor a source snapshot and the entire experiment directory is untracked.

## Version and lineage matrix

| Version | Actual object | Parent | Requested / actual new interactions | LR | Selected evidence checkpoint | Confirmed result | Historical code change status |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| v52 | Hybrid controller package; **not a learned checkpoint** | v45 actor at 47,349,760 plus v50/v52 optimized reverse gaits | N/A (CMA-ES reference optimization) | N/A | `v45_step_47349760.onnx`, SHA-256 `52205E…F3C` | Calibrated/backlash reverse package passed its recorded 3-command, 20-seed, 30-s test at only 0.003-rad/0.01-m/s initialization | Confirmed by `artifacts/openduckmini_backward_v52_manifest.json` |
| v54 | New PPO branch | v22 step 10,485,760 | 60M / 10,485,760 | 1e-5 | step 7,864,320 | Direct-policy evaluator collapsed to stand for essentially all commands | “reverse-only sampler diagnosis / robust sampler” is not recoverable as an exact diff |
| v55 | New PPO branch | v45 step 47,349,760 | 80M / 2,621,440 | 1e-5 | step 2,621,440 | Forward/turn motion present; reverse direct policy static; reverse turns unstable | Report says turn teacher was overwritten; exact source snapshot absent |
| v56 | New PPO branch | v45 step 47,349,760 | 100M / 7,864,320 | 1e-5 | step 5,242,880 | Reverse direct policy mostly static; maximum reverse turns fall | Report says v52 turn teachers restored; exact source snapshot absent |
| v57 | Separate restart, not continuation of v56 | v45 step 47,349,760 | 60M / 10,485,760 | 1e-5 | step 7,864,320 | Broad instability and falls | Same negative-return regime is supported by train return, but the exact diff is absent |
| v58 | Separate restart, not continuation of v57 | v45 step 47,349,760 | 60M / 60,293,120 | 1e-5 | step 60,293,120 | Returns become positive; direct reverse collapses to stand; right-yaw falls remain | Reported signed-reward/termination fix is consistent with return shift but not source-version provable |
| v59 | Policy/critic/normalizer continuation with **fresh Adam state** | v58 step 57,671,680 | 40M / 41,779,200 | 5e-6 | formal step 33,423,360; video step 8,355,840 | yaw overshoot; direct reverse static; maximum reverse-right 5/5 falls | Current source mtime is 03:00:48, immediately before v59; current `command_progress=100` is attributable to v59, but no immutable snapshot exists |

The actual v58+v59 count is `60,293,120 + 41,779,200 = 102,072,320` environment interactions. It is one actor/critic/normalizer lineage, but not one uninterrupted optimizer run: Brax restores normalizer, policy and value parameters and initializes a new Adam state and a new `env_steps=0`.

## PPO and learned components

| Item | v52 | v54–v58 | v59 |
| --- | --- | --- | --- |
| Actor | v45 policy used inside hybrid controller | Restored actor, then all 14 outputs trained | v58 actor restored, all outputs trained |
| Critic | Not part of runtime package | Restored at branch start; privileged observation | v58 critic restored |
| Teacher | Runtime periodic reverse gait | Training environment forces periodic reverse leg targets; actor supplies residual | Same current routing |
| Command encoder | None; raw 7-vector enters MLP | None; raw 7-vector | Same |
| Observation normalization | Embedded v45 normalizer in ONNX | Brax running stats, restored then updated | v58 stats restored then updated |
| Action scaling | Runtime hybrid: periodic reverse target plus configured residual | Non-reverse nonlinear measured-range map; reverse periodic target + residual | Same current source |
| Optimizer | N/A | Adam, new state on every launcher | Adam, new state despite “continuation” |
| Learning rate | N/A | 1e-5 | 5e-6 |
| Entropy | N/A | 0.005 | 0.005 |
| Discount | N/A | 0.97 | 0.97 |
| GAE lambda | N/A | Brax default 0.95 | 0.95 |
| Rollout | N/A | 20 actions; 4096 envs | Same |
| Actor / critic MLP | Runtime actor 101→512→256→128→28 distribution params | Same; critic uses same hidden sizes | Same |

Evidence: launcher scripts, each `train.log` PPO block, current `playground/common/runner.py`, and installed Brax `ppo/train.py:706-733`.

## Policy observation contract (101 dimensions)

All entries describe the audited current v59 training source. There is no resolved historical observation schema for v54–v58.

| Signal | Present / dim | Frame and transform | Noise / delay / stack |
| --- | ---: | --- | --- |
| Base linear velocity | **No** in actor; 3-D local velocity is critic-only | body/local sensor | critic privileged only |
| Base angular velocity | gyro, 3 | local/body; no explicit clamp | uniform ±0.1; no gyro delay |
| Projected gravity | **No** in actor; critic-only | body frame | uniform ±0.1 and random 0–2 action-step delay, critic sees clean |
| Accelerometer | 3 | sensor frame; x receives +1.3 | uniform ±0.05; no delay |
| Joint position | 14 | actuator qpos plus backlash displacement, minus home | per-joint uniform: hips ±0.03, knees ±0.05, ankles ±0.08; head scale defaults to zero |
| Joint velocity | 14 | actuator velocity ×0.05 | pre-scale noise ±2.5 rad/s |
| Previous actions | 42 | raw policy actions at t−1,t−2,t−3 | explicit three-frame action history |
| Raw command | 7 | `[vx, vy, yaw, neck_pitch, head_pitch, head_yaw, head_roll]`; velocity command is body frame | no noise, normalization only through running stats |
| Filtered command | No | — | — |
| Ramp progress | No | — | no ramp exists |
| Previous command | No | — | — |
| Motor target | 14 | previous actuator target in radians | no added noise |
| Foot contact | 2 | Boolean left/right | no noise |
| Gait phase | 2 | cosine/sine of periodic phase | phase continues across command changes; not reset |
| Disturbance state | No | push velocity impulse is hidden | — |
| Frame stack | Only three previous actions | no state frame stack | `history_len=0` |

No actor observation clamp is implemented before Brax running-stat normalization. Training action delay samples indices 0–2 (`randint maxval=3`); evaluation applies no delay.

## Command contract

- Configured continuous bounds: `vx∈[-0.15,0.15] m/s`, `vy∈[-0.2,0.2] m/s`, `yaw∈[-1,1] rad/s`.
- Contrary to the README deployment statement that head targets are forced to zero, current training samples all four head commands over calibrated ranges and the 14-action actor controls them. The formal v59 evaluation reports nonzero head peaks and fails the `head_locked` gate for 19/19 commands.
- Current v59 sampler chooses one of 24 modes. Stop is 1/24. Modes explicitly cover axis forward/backward, lateral, yaw, forward compounds, reverse+yaw, reverse lateral+yaw and continuous modes. Exact probability at a single maximum corner remains zero for continuous random variables.
- Command is held until `state.info["step"] > 500`, i.e. approximately 10.02 s at 50 Hz, then changes abruptly without simulation reset, phase reset, or ramp.
- No rollout command histogram or effective-episode histogram was saved. Therefore actual `P(vx,vy,yaw)`, maximum-backward×maximum-right counts and post-command survival exposure are unavailable.

Formal 19 commands, in body-frame units, are:

| ID | vx | vy | yaw rad/s |
| --- | ---: | ---: | ---: |
| stop | 0 | 0 | 0 |
| forward / backward | +0.10 / −0.10 | 0 | 0 |
| left / right | 0 | +0.10 / −0.10 | 0 |
| yaw left / right | 0 | 0 | +0.60 / −0.60 |
| forward left / right | +0.07 | +0.05 / −0.05 | 0 |
| forward yaw left / right | +0.07 | 0 | +0.30 / −0.30 |
| forward-left-yaw-left / mirror | +0.07 | +0.04 / −0.04 | +0.30 / −0.30 |
| backward yaw left | −0.07 | 0 | +0.10,+0.20,+0.30 |
| backward yaw right | −0.07 | 0 | −0.10,−0.20,−0.30 |

## Current reward inventory

Each term is multiplied by its weight, summed, multiplied by `dt=0.02`, then clipped to `[-100,10000]`. “Historical version” is not asserted where no snapshot exists.

| Term | Formula / active condition | Weight | Sign |
| --- | --- | ---: | --- |
| tracking_lin_vel | `exp(-((vx−vx*)²+(|vy−vy*|−tol)_+²)/sigma)`; sigma 0.005 reverse, 0.02 otherwise; multiplier 4 reverse, 2 forward, 4 yaw-only | 10 | + |
| tracking_ang_vel | `exp(-(yaw−yaw*)²/0.04)` | 10 | + |
| torques | `Σ torque²` | −0.0005 | − |
| action_rate | `Σ(a_t−a_t-1)²` | −0.1 | − |
| stand_still | joint absolute position/velocity deviation; only command norm <0.01 | −0.5 | − |
| alive | 1 at stop, 0.25 while moving | +10 | + |
| imitation | reference velocity/contact rewards minus joint errors; moving only | +2 | mixed raw term, positive scale |
| target_imitation | reverse commanded-leg target MSE to teacher | −2 | − |
| target_limits | squared target-limit violation | −20 | − |
| orientation | squared projected-gravity x/y | −100 | − |
| contact_imitation | contact match fraction; moving only | +2 | + |
| feet_slip | contacted-foot horizontal speed sum; moving only | −2 | − |
| unexpected_contact | contacts outside teacher schedule; moving only | −1 | − |
| swing_clearance | normalized swing-foot clearance; moving only | +2 | + |
| yaw_translation | horizontal speed² for yaw-only | −200 | − |
| knee_extension | squared shortfall below 0.30 rad | −10 | − |
| head_frame_contact | squared head constraint violation | −200 | − |
| command_velocity_error | `||(vx,vy)−(vx*,vy*)||²` | −50 | − |
| command_yaw_error | `(yaw−yaw*)²` | −20 | − |
| command_progress | `v_xy·cmd_xy + yaw*cmd_yaw` | +100 | + |
| termination | `done` | −2000 | − |
| base height, vertical velocity, flight, explicit fall | not present as separate reward terms | 0 / absent | — |
| backward-specific velocity/yaw/lateral/progress terms | implemented but weights zero | 0 | inactive |

The negative-reward termination issue cannot be assigned to an exact introduction/fix commit. The logs do prove a regime change: v54–v57 evaluation returns remain approximately −1,149 to −5,779; v58 starts at −175 and reaches +192 after the lower clip/termination-penalty regime described by current source. Per-term traces were not logged, so actual termination-term firing and return immediately before/at/after a fall cannot be reconstructed.

## Teacher, curriculum, and disturbance

- Reverse command (`vx < −0.02`) causes the environment to replace the direct policy motor target with the optimized periodic reverse/turn reference plus policy residual. This is actuator routing, not merely imitation reward.
- Left and right teacher profiles are chosen by yaw sign and blended by `|yaw|/0.2`; right blend is capped at 0.90.
- Teacher, reward, and policy all receive the same training `info["command"]`. The actor observes raw command and phase, but not whether a hidden velocity impulse just occurred.
- There are no curriculum stages or stage-promotion gates. Resets start from the home/stand keyframe with ±0.03-rad joint noise and small base velocity. A reverse command can be sampled at reset, but the periodic teacher immediately owns the reverse leg trajectory; the actor is not trained to create a standalone STAND→reverse limit cycle.
- Training adds a random horizontal **velocity impulse** (not a force) in world x/y every 5–10 s, magnitude 0.10–0.50 m/s in current source.
- Domain randomization covers floor friction 0.5–1.0, friction loss ±10%, armature +0–5%, torso COM ±0.05 m, link mass ±10%, torso mass ±0.1 kg, qpos0 ±0.03 rad and PD gain ±10%. No explicit latency randomization beyond action/IMU delays is recorded.
