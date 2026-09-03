# Exp 013 Phase W2-P1-A7-M0 accepted-environment masked PPO preflight

Classification: `ACCEPTED_ENV_MASKED_PPO_CONTRACT_PASS`.

Seven 1024-environment source batches were freshly replayed twice without snapshot restore; both raw inventory artifacts were byte-identical. S0 did not retain a rejected-plus-accepted full-1024 tensor hash, so parity is reported in two non-overstated parts: all 6,144 retained pool IDs and per-state semantic hashes match S0, and the newly computed full-1024 hashes match across both M0 fresh processes. Immutable masks assign every environment exclusively to train, validation, held-out, or rejected/unselected. Invalid environments remain under the exp_012 zero-command teacher as simulator housekeeping only.

The prototype compacts valid train indices before GAE normalization, PPO losses, statistics, gradients, and the temporary update. FULL_MASKED and COMPACT_REFERENCE differ by loss 0, gradients 0, and updated tensors 0. Extreme invalid-sample perturbations leave the compact tensors unchanged by 0, loss/statistics by 0, and updated tensors by 0. Split/rejected/roll-in/post-terminal leakage is zero.

Two independent fresh-process batch-0 passes captured the actual negative/positive rear-yaw W1B rollouts with identical initial-state and policy hashes and no optimizer update between passes. They provide 48,864 valid samples. The temporary update passed: exact KL 0.000585189, clip fraction 0, mean-action shift 0.00248375, gradient norm 6.16032, and NaN/Inf 0. No persistent PPO run or checkpoint was created.
