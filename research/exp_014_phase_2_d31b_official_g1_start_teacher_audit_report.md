# EXP014 Phase 2-D31B official G1 start-teacher audit

## Scope and execution

- Starting HEAD: `9b20c92880f5d37dfef690078aff0c48c5e075a0`
- Execution HEAD: `9b20c92880f5d37dfef690078aff0c48c5e075a0`
- Native command: `C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe C:\Users\user\workspace\physical-ai-lab\experiments\isaaclab\exp_014_unitree_g1_explicit_motion_mode_unified_locomotion\scripts\run_phase2_d31b_official_g1_start_teacher_audit.py --headless --viz none`
- Task candidate: `Isaac-Velocity-Flat-G1-v0`
- Seeds: `0, 1, 2, 3, 4, 5, 6, 7`
- Classification: **`EXP014_D31B_NO_OFFICIAL_PRETRAINED_G1_START_TEACHER`**

The audit searched installed Isaac Lab source/package/cache locations only. The
registered default G1 velocity-flat task is an environment candidate, not a
pretrained policy. Repository experiment checkpoints were explicitly excluded
from the official-policy set.

## Discovery

- Official pretrained checkpoint count: `0`
- Agile locomotion asset paths found: `1`
- Task registry evidence files: `6`

Because no eligible official pretrained checkpoint was found, native policy
rollouts, OFF/ON parity, START directions, STOP, L0-L5, and the S_HOLD to
official-teacher diagnostic replay were not run. No START authorization is
claimed.

## Safety and preservation

No training, PPO, CEM, search, WBC, torque, trajectory optimization, reward,
Student, RUN, or validation procedure was executed. Native environment/config/
robot/action/physics were not modified. Unrelated dirty and untracked state was
preserved: `True`.

See the JSON artifacts in
`results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31b_official_g1_start_teacher_audit/`
for the complete machine-readable inventory, contracts, gates, and hashes.
