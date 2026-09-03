# exp_014 Phase 2-D13R seed-contract correction report

## Result

Classification: **EXP014_D13R_CANONICAL_SEED_CORRECTED_AND_SEALED**. The earliest result-blind D12 commitment takes precedence. The three lowercase hexadecimal inputs, concatenated without whitespace, newline, or delimiter, reproduce seed `1940027935` twice bitwise using D12 source lines 123-124. D13's later five-element rule remains a failed historical artifact and was not used.

The result-blind manifest contains 680 unique episodes, exactly 20 for each of 34 conditions. Every perturbation axis has per-condition marginal counts of six or seven; all episode, recipe, snapshot, trajectory, planned-state-hash, generator-seed, and perturbation-key overlaps are zero. No outcome field exists.

Episode manifest SHA-256: `b37eccb1211a39f87f8fe7326c13b88ce80bc378c8b19f750adca55bfee69f1b`. Sealed payload SHA-256: `c6ef724da6fcafb25eb5c7d6a7b0b1ade17deb5cd4051a7fa16172c9465b9cfa`. The replacement is `SEALED_UNOPENED`, access count zero. No Isaac Lab import, simulation, physics, actor inference, checkpoint deserialize, or outcome calculation occurred.

SQLite WAL/FULL parent-owned persistence passed its synthetic transaction preflight; six of six crash tests and all 16 suite tests passed. D14 is `AUTHORIZED_ONE_TIME_ACCESS` for S1 step 30000 only, with the frozen gates and no fallback.
