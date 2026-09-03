# EXP014 Phase 2-D31B-R official Agile remote retrieval

## Scope and execution

- Starting HEAD: `c66f7cb22798f5a41f7c0ce73ef3d4afd2bea2a4`
- Observed execution-start HEAD: `012ebed9ebabc04176659e32eca3bf36db1fd54c`
- Execution HEAD: `012ebed9ebabc04176659e32eca3bf36db1fd54c`
- Native command: `"C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" "C:\Users\user\workspace\physical-ai-lab\experiments\isaaclab\exp_014_unitree_g1_explicit_motion_mode_unified_locomotion\scripts\run_phase2_d31br_official_agile_remote_retrieval.py" --headless --viz none`
- Official environment: `Isaac-PickPlace-Locomanipulation-G1-Abs-v0`
- Seeds: `0, 1, 2, 3, 4, 5, 6, 7`
- D31B classification preserved unchanged: `EXP014_D31B_NO_OFFICIAL_PRETRAINED_G1_START_TEACHER`
- D31B-R classification: **`EXP014_D31BR_OFFICIAL_POLICY_RUNTIME_FAIL`**

## Official source and gate

The source expression was resolved from the installed Isaac Lab locomanipulation
configuration and `ISAACLAB_NUCLEUS_DIR`; no URL or third-party source was
guessed.  `check_file_path` and `retrieve_file_path` were used through the
installed Isaac Python.  Remote status: `remote_available`.
Retrieval status: `retrieved`.  Read-only structure status:
`readable_torchscript`.

## Native result

Native capture status: `runtime_fail`.  Native START was gated on
successful retrieval and TorchScript structure audit.  No training, PPO, CEM,
WBC, search, Student, RUN, validation, or held-out evaluation was executed.
No EXP014 cross-runtime replay was attempted.

## Preservation

Unrelated dirty and untracked state preserved: `True`.  D6-D31B,
checkpoints, S_HOLD, W_MOVE, and physics were not modified.

Machine-readable artifacts are in
`results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31br_official_agile_remote_retrieval/`.
