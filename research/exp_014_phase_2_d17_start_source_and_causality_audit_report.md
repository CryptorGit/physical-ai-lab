# EXP014 Phase 2-D17 START source-state and causality audit

## Outcome

**EXP014_D17_YAW_REWARD_CAUSALITY_FAIL**. D15 and D16 remain unchanged. No persistent policy update or checkpoint was created.

## Source geometry

The 64 S_HOLD train sources are a distinct manifold from the 10,240-state `W_MOVE_FORWARD_BASIN_REFERENCE_V1`: normalized nearest-state distance mean 27.5945, nearest-action L2 mean 4.2002, cosine mean 0.8339, and contact mismatch 100.0%.

## Physical divergence

Direct W_MOVE reduced basin distance by 95.3% but achieved 0% sustained acquisition because yaw oscillated (p95 0.3904). R40 also acquired 0%; its first major yaw spike occurred at step 12 and its minimum basin distance 18.7921 rose again to 21.8303 by step 75.

No ramp, previous-action, or gate-duration counterfactual produced any acquisition. Extending the gate therefore does not explain D16. The one-update yaw-only probe did not produce yaw acquisition.

## Gradient causality

Yaw gradient norm grew from 17.5933 in W0 to 61.7968 in W3, a 3.51x increase after the physical yaw breakdown. Large aggregate gradient was therefore late/noncausal for the required initial weight shift.

## Reachability and prior artifacts

25-step ±0.50, 50-step ±0.50, and 25-step ±0.75 CEM action-sequence probes all achieved 0% registered success on 32 train sources. Existing exp013 A5-A8 trajectories use different source-state/roll-in contracts and fail either rear-yaw, local-neighborhood, or retention gates; no reusable S_HOLD-source oracle was established.

## Decision

Per registered precedence, command semantics, existing/search oracle, and gate duration were rejected first. The next single experiment is an early-phase yaw/weight-shift objective redesign. More PPO updates and curriculum expansion are not authorized.

## Protection

exp_005-exp_013, D6-D16 artifacts, checkpoints, datasets, optimizer, physics, reward config, and formal contracts were not changed. Persistent updates, new checkpoints, RUN, Causal DAgger V2, and remote push are zero.
