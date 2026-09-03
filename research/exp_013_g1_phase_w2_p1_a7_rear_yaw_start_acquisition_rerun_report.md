# Exp 013 Phase W2-P1-A7 replay-recipe rerun

## Outcome

Classification: `EXP013_W2_P1_A7_REPLAY_RECIPE_PARITY_FAIL`. The W1B-R2 parent strict-resume payload passes, and S0's full 1024-environment batch replay remains exact. PPO was not started because the newly required accepted-only, split-isolated per-recipe allocator cannot be constructed from the authorized S0 manifest without adding an unregistered mask/restore mechanism.

## Evidence

`Exp013FormalStopReplayRecipeV1` serializes one global seed, fixed batch order, environment ID, teacher SHA, and 150 roll-in steps. It does not serialize actual per-recipe reset RNG state. The `source_seed` column was derived as `seed + batch*1024 + env_id` but was not passed to `env.reset`; it is an identity label, not an independently executable reset seed.

Train recipes span batches as {0: 1018, 1: 998, 2: 1006, 3: 1004, 4: 70}; validation spans {4: 939, 5: 85}; held-out spans {5: 922, 6: 102}. Batches 4 and 5 cross split boundaries. Every source batch also contains rejected environments. RSL-RL's fixed 1024-env rollout buffer would therefore include rejected or out-of-split states unless a new masked-PPO contract were introduced. Snapshot duplication/fallback is explicitly prohibited.

The requested 256-per-split independent replay, same-recipe mirror branching, without-replacement recipe epochs, and serialized allocator continuation consequently fail before simulator/PPO launch. No baseline, PPO update, checkpoint, selection, or formal evaluation was performed. This does not invalidate S0's batch-level determinism or any earlier live-roll-in result.

## Protection

Existing datasets, labels, splits, manifests, overlays, state pool, replay manifest, checkpoints, optimizers, reward, physics, W2-P1-R2 student, and A4 candidate were unchanged. New checkpoint: 0. Remote push: false.
