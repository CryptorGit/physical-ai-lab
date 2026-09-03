# EXP014 Phase 2-D31B-R2 official policy runtime repair

## Result

- Requested starting HEAD: `012ebed9ebabc04176659e32eca3bf36db1fd54c`
- Observed execution-start HEAD: `5ed0a97994db8bb1376ca788334df852ec1997b9`
- Execution HEAD: `5ed0a97994db8bb1376ca788334df852ec1997b9`
- Classification: **`EXP014_D31BR2_OFFICIAL_POLICY_LOAD_PASS_START_NOT_RUN`**
- Official task: `Isaac-PickPlace-Locomanipulation-G1-Abs-v0`
- Official checkpoint SHA-256: `f04a58b834057eb1c9f38350dc12feaf929ff2cc7d5b75d2871e23811b775dde` (500080 bytes; referenced read-only)
- Native forward result: `fail`
- Unrelated dirty/untracked state preserved: `True`

## Runtime diagnosis

The canonical IsaacLab launcher was tested in a fresh process using
`C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe`.  Bare `python`/`py` resolve to a separate system runtime
whose package set does not contain the installed IsaacLab modules and whose
Torch build is not the official IsaacLab CUDA build.  This explains the
historical runtime split and is the evidence-based root cause; no package
replacement or persistent PATH edit was performed.

The canonical TorchScript load, CPU/CUDA probes, launcher probe, and minimal
Isaac probe pass.  Native locomanipulation construction reaches the official
environment but stops before reset because the installed Windows runtime lacks
`pink`; IsaacLab's own installer/setup metadata skips the pin-pink stack on
Windows.  No unsupported dependency installation was attempted.

The historical WinError 1114 probe is recorded verbatim in
`original_runtime_failure.txt` and the JSON matrix.  If it did not recur in
the repaired process, that is recorded as a non-reproduction rather than
invented as a new failure.

## Scope and safety

Only process-local diagnostics and the official installed launcher were used.
No checkpoint was converted, resaved, fine-tuned, or committed.  No PPO, CEM,
WBC, search, Student, RUN, validation, cross-runtime replay, or physics/PD/
friction/timing/robot/D6-D31B modification was performed.  The complete
machine-readable ledger is in
`results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31br2_official_policy_runtime_repair/`.
