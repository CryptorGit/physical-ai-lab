# exp_009 Stage 3 — Nonlinear rollout supervision

## Outcome

**SURROGATE_NOT_TRUSTWORTHY.** The fixed three-member nonlinear residual MLP ensemble was trained on grouped teacher trajectories plus live Stage 0/1/2 student WALK occupancy. It passed contact/support/gait classification and finite uncertainty handling, but failed the predeclared physical-state and action-ranking gates.

## Dataset and model

- Dynamics pairs: 1,034,060
- Live student occupancy: 147,520 steps from 922 episodes
- Strict matched bounded perturbation branches: 9,185
- Ensemble: 3 × (160 → 512 → 512 → 256 → 96), ELU
- State setters, teleport, snapshot injection, and Isaac backpropagation: none

## Trust gates

- One-step normalized physical MAE: 0.1294 (required ≤ 0.05)
- Contact macro-F1: 0.9890
- Support accuracy: 0.9882
- Gait accuracy: 1.0000
- Eight-step normalized RMSE: 0.5906 (required ≤ 0.25)
- Ranking Spearman: 0.3962 (required ≥ 0.70)
- Pairwise ranking: 0.6379 (required ≥ 0.80)
- Unsafe inversion: 0.1134 (required ≤ 0.10)

## Decision

No student checkpoint, WALK-only optimization, mixed distillation, or reverse diagnostic was run. The next single design is a **frozen WALK base with a continuous speed-conditioned residual/adapter**. This ends the single-head loss-redesign line without weakening the trust gate.
