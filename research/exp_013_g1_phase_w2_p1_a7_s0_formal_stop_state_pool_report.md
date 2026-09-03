# Exp 013 Phase W2-P1-A7-S0 formal-stop state initialization

## Outcome

Classification: `FORMAL_STOP_REPLAY_POOL_PASS`. A public-tensor snapshot is not authorized: hidden PhysX contact state is unavailable and both pre-step and continuation parity failed. The deterministic live-roll-in replay recipe is authorized for one future A7 rerun. No PPO or policy training occurred in S0.

## Existing contract

The historical W2-P1 start collector used standard reset, zero command, and the deterministic exp_012 Stage 2Q actor for 3.0 seconds (150 control steps). It never restored a saved full simulator state. Historical start acceptance was evaluated at the moving endpoint. S0 preserves that implementation and adds an explicit final-2-second practical-stop acceptance window for the new versioned pool.

## Pool

The run attempted 7168 episodes and found 7047 formal-stop PASS states; the first 6144 were retained as train 4096, validation 1024, held-out 1024. The pool contains 24 chunks of 256 states. Whole-pool semantic SHA-256 is `1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853`. A second fresh process reproduced all accepted IDs, batch semantic hashes, split assignment, and the whole-pool semantic hash exactly.

The generation-run temporal contact-summary accumulator did not initialize its prior-contact sample, so contact-switch/flight/double-support summary values from that run are marked `NOT_AVAILABLE`, not interpreted as zero. This instrumentation issue did not participate in acceptance and did not affect the captured contact tensors or replay hashes; the retained reproduction code fixes it.

## Snapshot and replay

Snapshot pre-step observation difference was 3.88659, and teacher action difference was 4.76589; both exceed 1e-8. Same-process continuation maximum observation difference was 22.0854; fresh-process was 9.2866. PhysX warm-start, manifold, solver impulse, and broadphase/narrowphase internal states are unavailable through the public API.

The replay recipe fixes seed 20278501, 1024 environments, batch/reset order, zero commands, teacher SHA, and 150 control steps. Its independent reproduction was exact. LIVE_ROLLIN and REPLAY_RECIPE are consequently the same executable operation; the 24-condition parity artifact records zero inferred difference from exact recipe identity. Snapshot start evaluation was correctly skipped after its prerequisite failed.

## Prior validity and protection

W2-P1 and A1-A6 used live roll-in and remain `VALID_UNCHANGED`; no prior result is invalidated. Existing datasets, labels, splits, manifests, overlays, checkpoints, optimizers, physics, reward, and evaluator code were not changed. New policy checkpoint: 0. A7 PPO: 0. Canonical promotion: 0. Remote push: false.
