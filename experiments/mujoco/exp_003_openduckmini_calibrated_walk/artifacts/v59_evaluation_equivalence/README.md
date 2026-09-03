# v59 Evaluation Equivalence Artifacts

This directory is a diagnostic-only parity audit for checkpoint step
33,423,360.  It is not a formal performance evaluation and does not qualify
v59 for deployment.

## Reproduction

From the historical WSL checkout:

```bash
bash /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_003_openduckmini_calibrated_walk/tools/run_v59_parity_wsl.sh
```

The script loads the checkpoint and ONNX read-only, disables stochastic
inputs, creates five 2-second traces, compares JAX, independent NumPy, and
ONNX actor inference, and compares an independent NumPy implementation of
action composition with the MJX environment's emitted motor targets.

Files under `golden_traces/` contain every array element per control step.
`comparison_tables/` contains actor and motor-target errors.
`smoke_results/` contains diagnostic smoke metadata and outcomes.
