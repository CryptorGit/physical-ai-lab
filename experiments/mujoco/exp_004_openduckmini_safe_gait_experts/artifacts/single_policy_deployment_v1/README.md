# OpenDuck Mini single-policy deployment package

Package ID: `exp004-single-policy-deployment-20260811-v1`

This package contains one executable locomotion weight:
`models/base_v22.onnx`. Every route uses that same weight. The backward gait
files are measured/calibrated periodic target overlays, not direction-specific
policies and not additional neural-network weights. They are retained because
the V22 actor alone does not produce a sufficiently safe reverse trajectory
under the calibrated hardware actuator model.

The package is simulation-qualified and hardware-pending. It does not enable
torque from this workstation.

The declared command envelope is `vx=[-0.10,+0.10] m/s`,
`vy=[-0.16,+0.16] m/s`, `wz=[-1.0,+1.0] rad/s`. The exact 19 qualified
command cases, including the six backward-yaw arcs, are the `qualified_cases`
in `manifest.json`; no command outside that list should be sent before a new
qualification run.

## Contents and contracts

- `manifest.json`: exact command, observation/action, joint/servo mapping,
  calibration provenance, guard values, and file hashes.
- `models/base_v22.onnx`: single command-conditioned ONNX actor, input `obs`
  `[1, 101]`, output `[1, 14]`.
- `calibration/`: copied 2026-07-29 user-confirmed raw leg calibration and the
  exp004 safety contract plus the byte-identical authoritative `duck_config`
  (`start_paused=true`, `imu_upside_down=false`). These values are never
  regenerated or zeroed.
- `motion/`: the three hash-pinned calibrated backward profiles.
- `simulation/` and `evidence/`: the exact scene/reference, formal 20x30
  evidence, exp003 primitive/compound/transition records, and fixed-camera
  19-pattern MP4 used by the package.
- `deployment/offline_target_sanity.py`: hash, sign, offset, raw-count,
  head-lock, and target-limit audit. It never opens a serial device.
- `deployment/hardware_test_plan.md`: staged human-controlled test protocol.

The conversion is:

```text
policy action
 -> V22 action-to-model target (action_scale=0.25)
 -> desired target guard (0.050 rad inward margin)
 -> one 2.0 rad/s slew at 50 Hz (<=0.040 rad/tick)
 -> model joint target
 -> direction sign
 -> measured zero offset from 2026-07-29 raw counts
 -> physical servo target / STS3215 raw position
```

Head actions and head torque remain exactly zero. Absolute position commands
that cross the STS3215 0/4095 boundary must be rejected and the group disabled.

## Offline verification

From the repository root:

```powershell
wsl.exe -- /home/user/openduck_training_20260729/.venv/bin/python \
  /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/single_policy_deployment_v1/deployment/offline_target_sanity.py \
  --package-root /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/single_policy_deployment_v1
```

Expected result: `passed=true`, `failure_count=0`, with
`hardware_io_opened=false` and `torque_enabled=false`.

The launcher is also fail-closed by default:

```bash
python deployment/run_v22_hardware.py
```

Only after a human has secured the robot and completed the staged plan may the
operator intentionally add `--enable-hardware
--confirmation I_HAVE_SECURED_OPEN_DUCK_MINI_AND_EMERGENCY_STOP`. The live
launcher still requires an explicit `duck_config.json` and runtime script and
passes the package's one weight plus all three measured backward profiles to
the pinned runtime.

## Simulation reproduction

The formal evaluator must receive the same `models/base_v22.onnx` path for all
eight role names. It must use the exp004 generated root and the frozen central
evaluator. The release protocol is 20 episodes × 30 s, master seed 20260808,
transition 30 s, transition stand 5 s, warmup 1.5 s, initial joint noise 1.0,
and initial base speed 0.10. Any H5 diagnostic PPO weight is excluded from this
package and cannot replace the single V22 weight.

Exact formal-evaluation invocation from PowerShell (the output is intentionally
outside this hash-pinned package):

```powershell
$exp = "/mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_004_openduckmini_safe_gait_experts"
$weight = "$exp/artifacts/single_policy_deployment_v1/models/base_v22.onnx"
wsl.exe -- /home/user/openduck_training_20260729/.venv/bin/python "$exp/scripts/evaluate_routed_transitions.py" `
  --policy "stand=$weight" --policy "forward=$weight" --policy "reverse=$weight" `
  --policy "lateral_left=$weight" --policy "lateral_right=$weight" `
  --policy "yaw_left=$weight" --policy "yaw_right=$weight" --policy "compound=$weight" `
  --generated-root "$exp/artifacts/generated_playground" --seed 20260808 `
  --episodes 20 --seconds 30 --transition-seconds 30 --transition-stand-seconds 5 `
  --warmup-seconds 1.5 --initial-joint-noise-scale 1.0 --initial-base-speed 0.10 `
  --backward-residual-scale 0 `
  --output "$exp/artifacts/exp004_single_policy_v22_formal20x30_rerun.json"
```

Exact fixed-camera MP4 regeneration invocation:

```powershell
$exp = "/mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_004_openduckmini_safe_gait_experts"
$pkg = "$exp/artifacts/single_policy_deployment_v1"
wsl.exe -- /home/user/openduck_training_20260729/.venv/bin/python `
  /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_003_openduckmini_calibrated_walk/render_all_release_patterns.py `
  --seconds 6 --fps 25 --width 640 --height 480 `
  --scene "$pkg/simulation/scene_flat_terrain_backlash_hardware_safe_calibrated.xml" `
  --policy "$pkg/models/base_v22.onnx" --reference-data "$pkg/simulation/reference.pkl" `
  --backward-left-turn-gait "$pkg/motion/optimized_backward_left_turn_gait.json" `
  --backward-right-turn-gait "$pkg/motion/optimized_backward_right_turn_gait.json" `
  --output-dir "$exp/artifacts/fixed_camera_v22_release_repro" `
  --combined-name exp004_single_policy_v22_fixed_camera_all_19.mp4 `
  --backward-residual-scale 0
```

The formal evidence is simulation-only; it is not evidence that the robot is
powered or that servo targets have been physically verified.

## Hardware handoff

Do not run the live runtime until the staged test plan is executed by a person
with the robot physically secured and an emergency stop available. The
calibrated runtime uses the leg bus `/dev/ttyACM0`, head bus `/dev/serial0`,
1,000,000 baud, BNO055 over I2C, and active-low foot switches on D22/D27.
The runtime must load the measured leg offsets and directions documented in
`calibration/contract.json` and must match the byte-pinned runtime config's
`start_paused`, `imu_upside_down`, and complete `joints_offsets` map; a default,
missing, or mismatched `duck_config.json` is a stop condition.

Current workstation evidence: read-only SSH to the `openduck` host timed out,
and the last captured host audit reported no matching serial motor device.
Therefore hardware verification is intentionally open, not claimed as PASS.
