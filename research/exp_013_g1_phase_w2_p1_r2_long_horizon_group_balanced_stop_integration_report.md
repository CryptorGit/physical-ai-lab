# exp_013 Phase W2-P1-R2 long-horizon group-balanced stop integration

## Outcome

Classification: `EXP013_W2_P1_R2_VALIDATION_SELECTED_HELDOUT_FAIL`.

The resolved dataset identity and exact canonical 2,000-step prefix parity passed. One persistent canonical-parent balanced-only run completed all 40,000 optimizer steps and wrote the preregistered 81 checkpoints. Validation produced 20 joint-pass checkpoints; the first was step 10000. The immutable validation rank selected step 37000 rather than the latest checkpoint.

## Prefix and training

- Prefix tensor hash: `d734a11b1c8d64da34498e3e02216e3c655629f425cc155ccf1b01d51e4aaa70` (exact D3 match)
- Prefix trace hash: `32ee73375473e7cd96322f661163f8f17e74e569b061929316bf73974783bee1` (exact D3 match)
- Optimizer: Adam, fixed LR 2e-4, seed 20277717, pool seed 20276049
- Group weights: 25/25/25/25
- NaN/Inf and hard numerical guards: PASS

## Validation selection

- Selected step: 37000
- Selected checkpoint SHA-256: `29f1acfb257111b7462b4c781b56992668759f84ab17133f30c3cfa37c6b7e93`
- Start MSE: 0.0009793625
- Stop-recovery MSE: 0.0009766348
- Steady-stop MSE: 0.0000465463
- Worst moving subgroup MSE: 0.0000446342
- Validation exact-zero MSE: 0.0668302551; nonzero start MSE: 0.0000172554

## Held-out authorization

The selected checkpoint was then evaluated once on held-out data. No fallback was permitted or performed. All groups except START_RETENTION passed. START_RETENTION MSE was 0.0011614304 (threshold 0.001) with cosine 0.9997312427. Its held-out exact-zero subset contained 171 samples with MSE 0.0669293329; nonzero start MSE was 0.0000172335.

This failure terminates the authorization chain before closed-loop evaluation. Static practical stop, moving retention rollouts, transition matrices, safety/symmetry rollout gates, and DAgger are explicitly not executed. Existing D3 held-out analysis was acknowledged, but checkpoint selection in this stage used validation only.

## Protection

Existing datasets, labels, split, manifests, checkpoints, optimizers, sampler, reward, physics, calibration, evaluators, Isaac Lab core, and RSL-RL package remained unchanged. The only new policy files are W2-P1-R2 scheduled/selected student artifacts. Runtime teacher use and remote push are zero.
