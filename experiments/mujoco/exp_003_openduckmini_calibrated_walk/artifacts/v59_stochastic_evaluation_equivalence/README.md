# v59 Stochastic Evaluation Equivalence

This directory audits v59 step 33,423,360 with training-time environment
noise, delay, reset randomization, domain randomization, passive backlash
joints, disturbance sampling, and PPO policy sampling.

The result is deliberately split:

- controller sample-injection parity: **PASS**;
- exact historical checkpoint-time episode reconstruction: **FAIL**;
- native closed-loop same-backend bit reproducibility: **FAIL**;
- overall stochastic gate: **FAIL**.

This is not a performance evaluation.  v59 remains
`diagnostic_not_qualified`, is not promoted, and must not be deployed.

## Reproduce

```bash
bash /mnt/c/Users/user/workspace/physical-ai-lab/experiments/mujoco/exp_003_openduckmini_calibrated_walk/tools/run_v59_stochastic_parity_wsl.sh
```

The run is fixed to five commands, environment indices 0–2, and 100 control
steps.  It performs no optimizer or checkpoint writes.

Each case has:

- a compressed full-array NPZ trace;
- JSON metadata containing reset and domain samples;
- JSONL containing every per-step random sample event.

The accepted numerical decisions are in `stochastic_parity_report.json`,
`sample_injection_results.csv`, and `native_seed_reproducibility.csv`.
