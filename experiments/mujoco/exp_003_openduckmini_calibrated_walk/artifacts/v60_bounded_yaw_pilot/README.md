# v60 Bounded Yaw Objective Causal Pilot

Status: **STOP_AT_1M**. This is a diagnostic pilot only. Neither v59 nor the
v60 artifacts are qualified for adoption, hardware transfer, or formal
acceptance.

The matched control and treatment arms restored the same v52 package training
parent (v45 normalizer, actor and critic), initialized fresh Adam identically,
used seed `20260730`, and each executed exactly 1,000,000 environment
interactions. Their step-0 parameter tree SHA-256 is identical:
`5981cee4c606a0ea87c1d426b8090532837e7dbac9c00d86c2a6141f7db82bc3`.
The only intended reward difference is the yaw contribution inside
`command_progress`.

The static bounded-objective contract passed, but the learned causal gate did
not. Under Condition D, Arm T produced a left yaw response ratio of `0.504`
and right ratio of `1.077`; its mean yaw MAE improvement over Arm C was only
`13.2%`, not the required `50%`. Under Condition S, yaw-only falls were `8/10`
for Arm T versus `7/10` for Arm C. Forward and representative forward+yaw
retention also degraded by more than 10% relative to the parent. No 5M
continuation was run.

Key files:

- `source_provenance.md`, `baseline_source_manifest.json`: frozen source and runtime.
- `parent_checkpoint_manifest.json`: v52/v45 parent, normalizer, actor, critic and optimizer initialization hashes.
- `old_objective_contract.json`, `new_objective_contract.json`: exact static objectives and pre-training gate.
- `training_contract.json`: automated Arm C/T diff.
- `control_training_manifest.json`, `treatment_training_manifest.json`: completed 1M runs.
- `paired_training_report.md`, `paired_training_report.json`: causal result and gates.
- `yaw_primary_results.csv`, `retention_results.csv`: episode-level evaluation.
- `checkpoint_comparison.csv`, `reward_term_summary.csv`: command and reward summaries.
- `evaluations/`: complete GPU MJX snapshots and raw logs for parent/C/T, D/S.
- `failed_runs/`: excluded WSL `libcuda.so` host-boundary failures and kernel evidence.

Important provenance limitation: the installed Brax callback did not expose a
resumable Adam state or the exact rollout command tensor. The completed causal
runs were uninterrupted, but exact rollout command histograms and 250k/500k
checkpoints for both matched arms are unavailable. The valid live checkpoints
are step 0 and step 1M; no claim is made about selecting an intermediate
checkpoint.
