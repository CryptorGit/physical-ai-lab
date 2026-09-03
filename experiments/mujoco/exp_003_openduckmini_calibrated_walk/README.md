# Calibrated Open Duck Mini locomotion

## 2026-07-30 omnidirectional retraining

An additional 102.1M-step v58/v59 all-direction continuation was completed
after diagnosing the reverse-only command sampler, overwritten turn teachers,
and a negative-reward early-termination incentive.  It produced a complete
19-pattern diagnostic video, but it did not pass the strong-perturbation
release suite.  v52 remains the deployment candidate.  See
`artifacts/v59_omnidirectional_diagnostic_report.md`; do not deploy the v59
diagnostic ONNX.

This experiment keeps two concepts separate:

- the user-confirmed straight-leg physical pose is model `q=0`;
- learning resets to the shallow, compliant `SAFE_INIT_POS`.

The deployment contract controls the ten leg joints. The official policy
interface remains 101 observations and 14 actions, but the four head targets
are forced to zero. Leg targets are clipped to the measured mechanical limits,
the MuJoCo limits, and a 32-count safety margin from the STS3215
absolute-position boundary.

## Release candidate status

`calibrated_hybrid_policy_v22.onnx` is the simulation-qualified release
candidate. It combines the learned policy for standing, forward/lateral
walking, and turning with the optimized periodic feedforward gait for reverse.
The deployed observation path applies a `-0.06 m/s` lateral compensation only
for positive-yaw policy observations. Requested commands are not changed.

The final acceptance run covers seven commands, 20 seeds per command, and
30 seconds per episode with `0.03 rad` initial joint noise and up to `0.1 m/s`
initial base velocity. A second run covers 12 compound commands, including
diagonal forward motion, forward turning, and three magnitudes of left/right
reverse turning. A 20-seed continuous run also changes directly between
forward, reverse, reverse turning, stop, lateral motion, and in-place turning
without resetting the simulation. All acceptance checks pass. The runtime
target audit passes 8,100 samples with no joint-limit violation, no STS3215
absolute boundary crossing, and zero head targets.

Authoritative release records:

- `artifacts/final_release_manifest.json`
- `artifacts/final_release_acceptance.json`
- `artifacts/final_compound_command_acceptance_v2.json`
- `artifacts/final_command_transition_acceptance_v2.json`
- `artifacts/runtime_integrated_compound_torque_off_audit_v2.json`

This qualifies the controller for staged hardware testing, not unattended
floor operation. Head motors 30–33 remain disabled. The remaining gates are
deployment checksum verification, torque-off observation replay, an airborne
low-torque leg test, supported standing, and short low-speed walking.

## Current training path

The earlier Stable-Baselines3 environment remains useful for diagnostics, but
it converged to standing. The main training path now uses the official Open
Duck Playground MJX/Brax environment in WSL2:

- 4,096 simultaneous backlash-enabled environments;
- CUDA JAX on the RTX 5090 Laptop GPU;
- 300 million training steps;
- the measured shallow-crouch pose as the home keyframe;
- all policy targets clipped to measured operational joint limits;
- official polynomial gait references affine-mapped into those limits;
- explicit stop-command samples;
- head actuation locked to zero.

Build and validate the generated calibrated scene and reference data:

```powershell
& '..\.venv\Scripts\python.exe' build_calibrated_playground.py
```

Start the persistent WSL training host:

```powershell
$scriptPath = '/mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_003_openduckmini_calibrated_walk/launch_mjx_training.sh'
Start-Process wsl.exe -ArgumentList @('-e', 'bash', $scriptPath) -WindowStyle Hidden
```

Inspect the live process, reward checkpoints, and GPU load:

```powershell
.\training_status.ps1
```

The simulation and staged hardware gates are defined in
`acceptance_criteria.json`. No policy may be sent to the floor-standing robot
until every simulation gate passes.

Run the environment and short training contract:

```powershell
& '..\.venv\Scripts\python.exe' train.py --smoke
```

The smoke checkpoint proves that the environment and calibration contract
train end-to-end. It is not suitable for deployment. A longer policy must be
evaluated in MuJoCo before any hardware export.

GPU policy optimization with parallel MuJoCo workers:

```powershell
& 'C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe' `
  train_parallel.py --n-envs 24
```

The default curriculum trains 24 independent CPU MuJoCo processes. PPO
optimization uses CUDA with 98,304 samples per rollout, 32,768-sample
minibatches, and 2048/2048/1024 actor and critic networks. It first trains
standing for one million steps, then walking for five million steps.

## 2026-07-29 validation history

- Straight-leg physical zero is model `q=0`.
- The shallow reset pose remains within every measured operational limit.
- The reset pose held for 600/600 control steps without termination.
- A 24-environment benchmark reached 4,281–6,139 simulation steps/s.
- RTX 5090 Laptop GPU utilization reached 97% during PPO updates; total
  observed VRAM usage reached 8.6 GB.
- PPO walking attempts that started without a reference gait converged to
  standing still.
- Early apparent forward motion was partly caused by uncontrolled head joints.
  The environment now holds the head and antennas at zero using low-torque PD.
- The best head-held periodic reference completed 600/600 steps, moved
  0.0459 m in 12 seconds, and kept upright cosine above 0.910.
- The official policy contract was verified as 101 observations and 14
  actions. The previous custom ten-action trainer was therefore not equivalent
  to the original runtime.
- The stock deep-crouch keyframe was found to exceed the measured limits of
  both knees, both ankles, and the left hip pitch.
- A generated calibrated keyframe and reference-motion set now fit all ten
  measured leg limits with a 10% trajectory margin.
- The calibrated MJX environment completed reset and GPU step smoke tests
  without termination.
- Runtime calibration tests pass 10/10.

The older rendered reference remains a diagnostic visualization. Use the
release manifest above, rather than an intermediate training checkpoint, for
staged hardware validation.

Render the current diagnostic gait:

```powershell
& 'C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe' `
  render_reference_gait.py
```
