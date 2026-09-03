# exp_013 Phase W2-P1-A5 — versioned four-step trajectory overlay preflight

## Outcome

**VERSIONED_4STEP_POSITIVE_CONTROL_FAIL**

The A4 in-memory candidate reproduced deterministically with tensor hash `db65a3069d665b8012fd9d264b7fd54e629a22d25b05a9ff793e23bfc549ac5f` and exact stored-metric parity. The preregistered PC2 positive control did not pass all condition gates, so candidate-visited collection, V3 overlay creation, probe training, held-out evaluation, and all downstream authorization were intentionally not run.

## Runtime positive controls

| Profile | Endpoint | Acquisition | Fall | Slip | Min condition endpoint | Min condition acquisition |
|---|---:|---:|---:|---:|---:|---:|
| PC0 candidate only | 93.85% | 90.25% | 5.35% | 4.69% | 47.00% | 28.50% |
| PC1 B0 stop + B1-B2 W1B | 99.33% | 92.56% | 0.65% | 0.35% | 95.00% | 13.50% |
| PC2 B0 stop + B1-B4 W1B | 99.73% | 92.65% | 0.27% | 0.04% | 99.00% | 18.00% |
| PC3 B0-B4 W1B | 99.71% | 92.92% | 0.29% | 0.10% | 99.00% | 19.50% |

PC2 satisfies aggregate endpoint, aggregate acquisition, fall, slip, impact, and every condition endpoint gate. It fails the required per-condition acquisition gate: direction 180°, yaw +0.3 reaches 18.0%, and direction 180°, yaw -0.3 reaches 22.5%, below the required 85%.

## Interpretation

Four W1B actions solve the safety and endpoint problem but do not establish the preregistered acquisition behavior for rear moving-yaw targets. Collecting an overlay from this positive control would therefore encode a protocol whose required capability has not been demonstrated. The fail-closed outcome is a protocol-level positive-control failure, not evidence about V3 representational feasibility.

## Protection

The immutable base dataset, labels, split, manifests, V2 overlay, checkpoints, and optimizers remain unchanged. No V3 `.pt` overlay was created. No persistent student, PPO, DAgger, formal full closed-loop evaluation, or canonical promotion occurred.
