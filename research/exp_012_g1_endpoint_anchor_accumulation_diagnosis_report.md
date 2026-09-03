# EXP-012 Stage 2O — Endpoint-anchor accumulation diagnosis

## Outcome

The Stage 2N anchor implementation is semantically correct. The main diagnosis is
`ADAM_HISTORY_SUPPRESSES_ANCHOR`. At beta 0.10 the raw anchor term reached 45.6% of the PPO
gradient norm, while the effective Adam update remained nearly orthogonal to the
anchor. No tested beta kept every endpoint below KL 0.03 for five updates.

## Anchor implementation

The loss is exact diagonal-Gaussian KL(reference||current), includes mean and std,
weights the four endpoints equally, and is evaluated for every PPO minibatch.
The reference is detached and the loss sign is correct. Its unavoidable first-step
gradient is zero because current and reference policies initially coincide.

## Drift and causal comparisons

Stage 2N WALK KL rose 0.01857, 0.03395, 0.04699, and 0.06655 over iterations 1–4.
WALK crossed 0.03 first. Fixed 1.5e-5 LR reduced beta-0.10 five-update WALK KL
from 0.05472 to
0.03838, but still failed the 0.03 gate.
Thus adaptive LR amplifies drift but is not sufficient as the primary cause.

The endpoint-specific PPO gradients are not simply a WALK-vs-RUN opposition:
RUN-1.2 and WALK-1.2 gradient cosine stays small and positive. RUN-1.2 nevertheless
projects away from the frozen reference at multiple updates. Critic value biases
are modest relative to returns, so the critic is not primary.

The frozen anchor and current endpoint-state distributions are separable
(maximum diagnostic AUROC 0.99997), but replacing the static anchor with a
current-state reference KL did not solve retention: its five-update WALK static
KL was 0.06177 and current-state KL was 0.06872. Coverage mismatch therefore
exists but is not the causal fix.

At beta 0.10, the median raw combined-gradient cosine to the anchor after the
anchor emerged was 0.164, while the median effective Adam-update cosine to the
anchor was 0.003. This optimizer-direction mismatch is the strongest isolated
cause. Endpoint closed-loop evaluation was fail-closed after every beta branch
missed the analytic KL gate; no missing Stage 2N checkpoint was regenerated.

## Current best artifact

The initial Stage 2N checkpoint remains the current best integrated gait artifact:
deterministic endpoints, calibrated stochastic endpoints, bidirectional toggles,
and single-weight operation all pass. Continued PPO semantic retention is not stable.

## Protection

All shadow branches were limited to five updates and discarded. No persistent
checkpoint or optimizer was written, no production policy was updated, protected
experiments and cores were unchanged, and no remote push was performed.
