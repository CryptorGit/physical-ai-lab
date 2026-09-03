# Current state audit: v52 to v59

Audit date: 2026-07-30  
Decision state: **v52 remains adopted; v59 remains `diagnostic_not_qualified` and must not be deployed.**

## 1. Executive Summary

The v59 failure cannot be attributed to one clean policy experiment because the provenance and evaluation contracts are not frozen. The experiment directory has zero Git-tracked files, no v52–v59 commits/tags, and the checkpoint metadata has empty custom metadata. v54–v59 launchers all trained a 101-observation/14-action PPO policy, but v54 and v55–v58 are separate branches from different parents, v57 is not a continuation of v56, and v59 restores v58 policy/value/normalizer with a **fresh optimizer state**.

Three conclusions are evidence-backed:

1. **Primary audit defect — evaluation/training wiring mismatch (HIGH).** Training used `scene_flat_terrain_backlash_calibrated.xml`, random delays/noise, and replaced reverse motor targets with a periodic teacher plus residual. The authoritative v59 JSON records `scene_flat_terrain.xml`; the scene-name gate therefore disables the reverse teacher path. Evaluation also applies a legacy positive-yaw `vy=-0.06` observation compensation that is not recorded in JSON. Thus reverse-static and “strong disturbance” results are not a clean measurement of the trained controller.
2. **Primary behavioral cause — the v59 shared progress objective rewards yaw overshoot (HIGH).** Current source uses `-20(ω−c)² + 100ωc` in addition to tracking rewards. This dense part alone has stationary optimum `ω=3.5c`, not `ω=c`. The formal JSON measures 1.95–2.03× overshoot for yaw-only and 2.73–3.11× for forward+yaw, consistent with the objective.
3. **Primary reverse cause — the actor was trained as a reverse residual, then evaluated as a direct controller (HIGH).** For `vx < −0.02`, training forces the teacher’s periodic leg target and adds a small actor residual. Direct ONNX evaluation on a non-calibrated scene removes that mechanism. The 5/5 static reverse-straight result therefore does not prove a network-capacity failure or a missing ramp state.

The reported “strong external disturbance” suite is misnamed. It applies no force and no timed push. It sets one initial world-frame base velocity with magnitude uniformly sampled from 0 to 0.1 m/s. Recovery time and pre/post-disturbance behavior cannot be computed.

The JSON also applies the repository’s 20-seed/30-s release criteria to a 5-seed/15-s run, so `enough_episodes` is false for all 19 commands by construction. In addition, all 19 commands fail `head_locked`: current training samples/actuates four head commands despite the README’s zero-head deployment contract.

## 2. Confirmed Facts

- Root repository: branch `master`, HEAD `5518ccecf73e7a471db1ef004b9d89e26560529a`; unrelated modified/untracked files already existed. Evidence: `git status --short --branch`.
- The entire target directory is untracked (`git ls-files …` returns 0). No matching Git tags or commits exist. Exact historical code diffs are unavailable.
- v52 is a hybrid controller package, not a single learned “v52 checkpoint”: v45 actor + v50 straight reverse gait + v52 turn gaits. Evidence: `artifacts/openduckmini_backward_v52_manifest.json`.
- v54 restores v22 step 10,485,760; v55–v58 each independently restore v45 step 47,349,760; v59 restores v58 step 57,671,680. Evidence: `launch_mjx_omnidirectional_v54.sh` through `v59.sh`.
- Every PPO log reports seed-default behavior (Brax default seed 0), 4096 environments, 20-step rollout, 0.97 discount, 0.005 entropy, 32 minibatches, four update epochs, and 101 observations. Evidence: each WSL run’s `train.log` plus installed Brax signature `seed=0`.
- Control is 50 Hz (`ctrl_dt=0.02`) with 0.002-s MuJoCo timestep and 10 simulator substeps per policy action. Evidence: current `joystick.py` config and `base.py`.
- Actor and critic hidden sizes are 512/256/128. Actor uses `state`; critic uses `privileged_state`. Evidence: resolved `brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")`.
- Brax checkpoint restore loads normalizer, policy and value parameters but creates a fresh Adam optimizer state and resets `env_steps`. Evidence: installed `brax/training/agents/ppo/train.py:706-733`.
- v58 used 60,293,120 environment interactions and v59 used 41,779,200, for exactly 102,072,320. Evidence: final checkpoint names and train logs.
- The combined diagnostic video hash is exactly `CBF9CD3C3552ED9BF4C4C48206F3FD3E83265D1E694E48B3D6BF68966246A2CC`.
- The copied diagnostic ONNX is byte-identical to v59 step 8,355,840, hash `F6EC2ACC78D725C3D0264F78BFAF6C4BC3605A64EC54069B21FAEADBAF8003A3`.
- The formal failure JSON evaluates a different checkpoint, v59 step 33,423,360. The report correctly distinguishes the diagnostic-video policy from the best-reward formal-evaluation policy.

## 3. Unverified or Contradictory Claims

| Claim | Audit status | Evidence / contradiction |
| --- | --- | --- |
| “v54–v59 changes are versioned” | Contradicted | Target has zero tracked files; source review repo is dirty at a single grafted upstream commit. |
| “v58/v59 is a 102.1M-step continuation” | Partly true | One policy lineage, but v59 starts fresh optimizer/env-step state. It is the sum of two environment-interaction counters, not one checkpoint’s counter. |
| “strong external disturbance” | Contradicted | `evaluate_official_policy.py:347-353` initializes base velocity once; no force body, start time, duration, repetition, or recovery event exists. |
| “v58/v59 used the same evaluation plant as training” | Contradicted | JSON records `scene_flat_terrain.xml`; training launcher uses task `flat_terrain_backlash_calibrated`. |
| “negative-reward fall incentive was fixed in v58” | Plausible but incompletely proven | Return regime changes sharply and current source has negative reward clip plus termination penalty. No v57/v58 source snapshots or per-term traces prove exact introduction/fix or term firing. |
| “turn teacher overwrite was fixed in v56” | Incompletely proven | Current source loads v52 turn profiles and v56/v57 report names agree; no frozen v55/v56 code diff exists. |
| “command ramp state should be added” | Unsupported for current failure | Neither training nor formal evaluation implements a command ramp. Commands change abruptly. |
| “nominal 19/19 proves walking” | Rejected | Six-second, no-noise, home-pose renders only prove render completion without the renderer’s fall threshold firing. |
| “19×5×15 s can meet the attached release criteria” | Contradicted | `acceptance_criteria.json` requires 20 seeds and 30 s; all 19 `enough_episodes` checks are false. |
| “head targets are forced to zero in this training path” | Contradicted | Current sampler draws four head commands, actor has 14 outputs, and all 19 formal checks fail `head_locked`. |

## 4. Repository and Checkpoint Provenance

### Repository state

The source review repository is at upstream commit `b9be205ac64488c23504ca42e5ec790337adeec3` with modified `joystick.py`, reward/export/runner files, and untracked calibrated models and gait files. `joystick.py` mtime is `2026-07-30 03:00:48 +09:00`, just before v59 began at 03:02, so current source is useful for v59 but is not an immutable record.

### Lineage and hashes

Checkpoint “tree SHA-256” below is a deterministic audit hash over sorted `(relative path length, relative path, file length, file bytes)` entries. Orbax does not define a directory hash.

| Version | Parent | Selected checkpoint | Tree SHA-256 | Exported ONNX SHA-256 |
| --- | --- | --- | --- | --- |
| v52 | v45 actor + optimized gaits | v45 step 47,349,760 | checkpoint tree not copied into release | `52205e…f3c` |
| v54 | v22 step 10,485,760 | 7,864,320 | `c5f3a5…c253` | `1b796b…604e` |
| v55 | v45 step 47,349,760 | 2,621,440 | `044146…bfe0` | `6bfb97…6a9` |
| v56 | v45 step 47,349,760 | 5,242,880 | `76e09c…854e` | `37c067…5c01` |
| v57 | v45 step 47,349,760 | 7,864,320 | `6c1d68…0174` | `58aa27…7142` |
| v58 | v45 step 47,349,760 | 60,293,120 | `fc4770…1178` | `755b1e…e178` |
| v59 | v58 step 57,671,680 | formal 33,423,360 | `4e5229…38f1` | `cb5e70…7763` |

All `_CHECKPOINT_METADATA` files contain empty `custom_metadata`; they do not identify parent, config, source commit, seed, or data contract. ONNX counterfactual inference on 20 fixed random observations confirms v55/v56/v57 step-0 exports are functionally identical to the v45 parent and v59 step-0 is functionally identical to its v58 parent (maximum absolute output difference 0).

### Step accounting

- One PPO training step consumes `256 batch_size × 20 unroll × 32 minibatches = 163,840` environment interactions.
- v58: `60,293,120 / 163,840 = 368` PPO training steps.
- v59: `41,779,200 / 163,840 = 255` PPO training steps.
- Total optimizer minibatch updates: `(368+255) × 4 epochs × 32 = 79,744`.
- Policy inference count during rollout: 102,072,320 (one action per environment interaction).
- Simulator step count: `102,072,320 × 10 = 1,020,723,200`.

Therefore “102.1M” is the sum of v58 and v59 **parallel-environment interactions**, not a single checkpoint counter, optimizer updates, or simulator steps.

## 5. v52–v59 Change Matrix

The detailed matrix is in `artifacts/v52_to_v59_change_matrix.md`. The defensible high-level sequence is:

- v52: packaged hybrid actor+feedforward controller.
- v54: separate v22-parent omnidirectional branch; static collapse under direct-policy evaluator.
- v55: new v45-parent branch; stopped at 2.62M.
- v56/v57: two separate v45-parent diagnostic branches, not a continuation pair.
- v58: new v45-parent branch; return regime improves and survives 60.29M, but direct reverse output remains static.
- v59: v58-policy continuation at half LR with new shared progress scale; movement increases and yaw overshoot becomes severe.

Because resolved configs/source snapshots are missing, claims about exact sampler, teacher and reward edits are reported as claims unless current source, launchers, and log behavior independently support them.

## 6. Training Distribution Audit

Current v59 source samples 24 discrete mode families over continuous command magnitudes. It explicitly includes backward-only and backward+yaw families, so a purely “reverse commands were impossible to draw” explanation is false for current code. However:

- No actual rollout histogram is logged.
- No 3-D joint histogram `P(vx,vy,yaw)` exists.
- Exact maximum corners have probability zero under continuous sampling.
- Effective exposure after early termination is not recorded.
- Commands last about 10.02 s and change abruptly; episodes last 20 s.

Resets use the home/stand pose, ±0.03-rad joint noise, base velocity components uniform in ±0.05 m/s, and random world yaw. Reverse can be drawn at reset. The reverse teacher begins immediately at phase zero, so the actor observes a STAND-start state but does not own the nominal reverse leg transition.

Training disturbance is a hidden horizontal world-frame velocity impulse every 5–10 s, magnitude 0.10–0.50 m/s in current source. Domain randomization includes friction, masses, COM, armature, joint zero and PD gain. This is substantial randomization; “no disturbance training” is contradicted.

## 7. Reward and Teacher Audit

### Reward objective

The full term inventory and formulas are in the change matrix. The critical v59 terms are:

```text
-50 * ||v_xy - c_xy||²
-20 * (yaw - c_yaw)²
+100 * (v_xy·c_xy + yaw*c_yaw)
```

For yaw-only, differentiating the dense yaw terms gives:

```text
d/dyaw [-20(yaw-c)² + 100*yaw*c] = -40(yaw-c) + 100c
stationary yaw = 3.5c
```

The positive progress term is unbounded in velocity/yaw until dynamics or other penalties dominate. It does not allow linear motion to directly compensate yaw inside one scalar error, but it does put both into a shared total reward and independently rewards overshoot in both channels. This supports H1’s objective-conflict outcome, though not its exact “same gate error compensation” wording.

Offline per-trajectory reward reconstruction is impossible because only episode means are stored. Symbolic counterfactuals are still decisive for the dense yaw part:

- correct yaw `ω=c`: dense contribution `100c²`;
- static `ω=0`: `−20c²`;
- dense optimum `ω=3.5c`: `225c²`;
- thus overshoot receives 2.25× the correct dense contribution before bounded tracking/other terms.

### Negative-return termination

Current source permits negative per-step reward (`clip min=-100`) and adds `termination=-2000×done`, multiplied by 0.02, i.e. nominal −40 at termination before total clipping. This removes the earlier zero-clipped negative-cost pathology and directly penalizes termination. Training logs show v54–v57 strongly negative episode returns and v58 moving into positive returns.

What is **not** proven: the exact version where the sign/clip issue was introduced, the exact v58 source diff, actual termination-term activation frequency, or return immediately before/at/after fall. No per-term TensorBoard extraction tied to a frozen config or raw trajectories exists.

### Teacher consistency

For reverse, teacher selection, reward command, and training policy command all read the same `info["command"]`. Teacher action is a joint-position target in actuator coordinates, built from measured-range reference profiles. At `vx<−0.02`, the environment switches the actual motor targets to teacher legs plus actor residual. This means the exported actor alone is not the trained reverse controller.

No command ramp exists. Gait phase is observed as cos/sin but is not reset on command change. There are no curriculum stages or promotion conditions.

## 8. Evaluation Pipeline Integrity

### Nominal video

All 19 numerical commands are listed in the change matrix and CSV. The renderer uses:

- duration 6.0 s;
- deterministic home pose;
- no seed-dependent initialization;
- no joint noise or base-speed perturbation;
- no command ramp;
- no timed disturbance;
- calibrated/backlash scene by default;
- positive-yaw policy observation `vy -= 0.06`;
- reverse periodic feedforward enabled unless explicitly disabled.

Thus the video is a hybrid-render diagnostic, not a direct learned-policy test. All 19 files contain 150 frames at 25 fps and the combined file duration is 114 s.

### Formal JSON

The formal v59 JSON uses commands in the same body-frame units, seeds 0–4, 15 s, ±0.03-rad joint initialization and one random initial base velocity up to 0.1 m/s. But it records:

```text
scene = scene_flat_terrain.xml
action_scale = 0.25
```

Consequences:

1. `calibrated_hardware = "calibrated" in scene.stem` is false.
2. Reverse teacher/feedforward routing is disabled.
3. The plant has no calibrated backlash joints and uses a different included robot model.
4. Training joint observation adds backlash displacement; evaluator does not.
5. Training action/IMU delays and observation noise are absent.
6. Training fall gate is upright <0.65 or root height <0.12 (plus head constraint); evaluator stops at upright <0.25 or height <0.08.
7. The default `positive_yaw_lateral_compensation=0.06` changes the policy observation but is not serialized in JSON.
8. The attached acceptance logic requires 20 seeds and 30 s, so a 5-seed/15-s run cannot satisfy it; all 19 commands fail `enough_episodes`.
9. Every command fails the zero-head-target gate. Current training samples head commands and does not implement the README’s claimed head lock.

There is no evidence of degree/radian confusion, yaw sign reversal, x/y swap, backward sign reversal, recurrent state, or ONNX observation-order permutation. ONNX is feed-forward. The evaluator’s nominal observation concatenation order matches the 101-D current actor schema at a field level, but model/backlash, noise, delay, command compensation, and control routing do not match.

The evaluator performs no timed push and cannot compute recovery. A matched minimal smoke would be justified, but was not run because current production evaluator code must remain unchanged and the existing evidence already establishes the mismatch.

## 9. 19-Command Failure Decomposition

Full seed-aggregate data are in `artifacts/v59_19_command_failure_matrix.csv`. P95 fields are p95 across the five episode-level means, not per-step p95.

| Failure family | Commands | Result |
| --- | --- | --- |
| Reverse straight static | (−0.10,0,0) | mean vx −0.000262 m/s; speed ratio 0.0031; no-motion proxy 5/5; both feet essentially always in contact |
| Small reverse turns static | (−0.07,0,+0.10), (−0.07,0,−0.10), (−0.07,0,−0.20) | no-motion proxy 5/5 |
| Yaw-only overshoot/fall | (0,0,±0.60) | +1.169/−1.221 rad/s; 1.95/2.03×; one fall each, earliest 2.34/1.92 s |
| Forward+yaw overshoot | (+0.07,0,±0.30) | +0.934/−0.836 rad/s; 3.11/2.79×; no falls |
| Diagonal forward+yaw overshoot | (+0.07,±0.04,±0.30) | +0.921/−0.820 rad/s; 3.07/2.73× |
| Maximum reverse-left | (−0.07,0,+0.30) | no falls, but vx has wrong sign (+0.0276); yaw 0.345 |
| Maximum reverse-right | (−0.07,0,−0.30) | 5/5 falls; earliest 2.08 s; mean yaw has wrong sign (+0.155) |
| Lateral asymmetry | left/right ±0.10 | left speed ratio 0.029 versus right 0.312; right command induces −0.102 rad/s unintended yaw |

### Requested special comparisons

1. **Backward straight static:** confirmed in all seeds, but only under the mismatched direct-policy/non-calibrated evaluator.
2. **Yaw excessive:** confirmed for every ±0.3 compound yaw and ±0.6 yaw-only condition.
3. **Maximum backward-right fall:** 5/5, while maximum backward-left is 0/5; strong asymmetry.
4. **Left/right turning asymmetry:** yaw-only magnitudes are similar, but right falls earlier and maximum reverse-right flips yaw sign and falls.
5. **Forward/backward asymmetry:** forward vx is 0.0395 m/s; backward is −0.00026 m/s. Training reverse actuator routing makes the direct comparison confounded.
6. **6 s versus 15 s:** video uses calibrated hybrid routing and no perturbation, so duration is not the only changed variable. The data do not isolate cumulative drift.
7. **Pre/post disturbance recovery:** unavailable; no post-start disturbance exists.

Torque, foot slip, joint velocity limits, exact action-clipping count, roll/pitch time series, no-motion duration and support sequence are unavailable. No substitute was fabricated.

## 10. Nominal vs Strong-Disturbance Gap

The gap is not evidence that a nominal policy merely lacks robustness:

- nominal video: calibrated/backlash hybrid controller, exact home pose, no perturbation, 6 s;
- formal JSON: non-calibrated/non-backlash direct actor, randomized initialization, 15 s.

At least five variables change simultaneously. Moreover, the formal perturbation is weaker than current training impulses and occurs only at initialization. Persistent reverse static and yaw overshoot are nominal control/objective/routing failures, not recovery failures.

## 11. Ranked Root-Cause Hypotheses

| Rank | Cause hypothesis | Evidence | Counterevidence | Confidence | Next minimal test |
| ---: | --- | --- | --- | --- | --- |
| 1 | Evaluator/deployment wiring mismatch | JSON scene; scene-name gate; teacher routing; unrecorded vy compensation; delay/noise/gate differences | Field-level 101-D order and ONNX normalization path appear compatible | HIGH | 2-s matched MJX-vs-evaluator observation/target trace, no training |
| 2 | Reward objective conflict | `command_progress=100` and yaw error −20 imply dense optimum 3.5× command; observed 1.95–3.11× | Other bounded rewards and dynamics reduce actual optimum | HIGH | command-centered yaw-objective pilot from v52 actor |
| 3 | Teacher/direct-policy inconsistency and actor-role mismatch | Reverse teacher owns leg targets in training; formal scene disables it; direct reverse static 5/5 | Actor does see raw command and phase and may learn residual stabilization | HIGH | identical hybrid routing in matched evaluator |
| 4 | Curriculum / initial-state insufficiency | No staged curriculum; actor never owns nominal STAND→reverse transition | Reverse commands can start at reset from stand | MEDIUM | log teacher/residual/target trace on STAND-start reverse |
| 5 | Sampler joint coverage shortage | No saved joint histogram/effective exposure | Current v59 code has explicit reverse-turn modes and 4096 envs | INSUFFICIENT_DATA | instrument histogram only, no reward change |
| 6 | Disturbance robustness shortage | Formal falls exist | Training has larger repeated impulses; formal has only mild initial velocity; persistent static/overshoot precede recovery question | LOW | only after evaluator matching, add logged timed impulse |
| 7 | Command-ramp partial observability | No ramp state observed | No ramp exists; commands are abrupt and phase is observed | LOW | none until a ramp is actually introduced |
| 8 | Network capacity shortage | No capacity ablation | 512/256/128 actor; defects already explained by wiring/objective | INSUFFICIENT_DATA | do not test before ranks 1–3 |
| 9 | Optimization shortage | v59 return continues improving | objective optimum is wrong and optimizer state restarts; more steps can worsen overshoot | INSUFFICIENT_DATA | objective-isolated short pilot |

Main causes are ranks 1–3. Rank 4 is secondary. Ranks 5, 8, and 9 are unverified, not co-equal causes.

## 12. Missing Instrumentation

- Immutable source diff, dependency lock, resolved config and command-line args per run.
- Parent checkpoint path/hash and optimizer-restore semantics in checkpoint metadata.
- Actual command 3-D histogram, duration and survived exposure.
- Per-step trajectory, action, motor target, teacher target, phase and observation snapshots.
- Per-term reward values and cumulative contributions, especially termination.
- Explicit perturbation event body/frame/type/time/duration/magnitude and recovery labels.
- Roll/pitch, base-height, vertical-velocity, foot-slip, torque, joint-limit and action-clipping traces.
- Evaluation arguments such as positive-yaw compensation in output JSON.

Without these, offline reward correlation, observation-collision search, per-step p95 and recovery-time analysis cannot be performed.

## 13. Recommended Parent Checkpoint

Maintain the **v52 hybrid controller package** as adopted. If a learned pilot is later authorized, use its exact v45 actor parent (`v45_step_47349760`, SHA-256 `52205E…F3C`) with frozen v52 profiles and a frozen calibrated/backlash source snapshot.

Do not parent from v54–v57 (failed independent branches), v58 (reverse direct collapse), or v59 (overshoot-seeking objective and failed formal suite). This is a simulation-research parent recommendation, not hardware authorization.

## 14. Minimal Next Experiments

Detailed protocols are in `artifacts/next_experiment_candidates_after_audit.md`:

1. Zero-training evaluation-equivalence counterfactual.
2. Single-change yaw-objective isolation pilot from the v52 package actor.
3. Single-change reverse actor-role/hybrid-routing isolation.

No ramp-state, larger-network, or longer-training experiment is justified by current evidence.

## 15. Explicit No-Go Conditions

- Do not deploy v59 or label it hardware-ready.
- Do not promote nominal 19/19 render completion to formal walking success.
- Do not start long training until evaluator/training equivalence is proven and frozen.
- Do not combine reward, observation, sampler, teacher, curriculum, and network changes in one run.
- Do not call the present suite “strong external force recovery”; it has no timed external force.
- Do not compare v52 calibrated hybrid results with v59 non-calibrated direct-policy results as if only checkpoint changed.
- Do not call a 5×15-s run a release-criteria pass/fail test while the attached criteria require 20×30 s.
- Do not proceed toward deployment while all 19 commands fail the zero-head-target contract.
- Do not proceed without per-run source/config hashes, parent hash, reward telemetry, command histogram, and per-step evaluation trace.
- Keep v52 adopted and v59 `diagnostic_not_qualified`.
