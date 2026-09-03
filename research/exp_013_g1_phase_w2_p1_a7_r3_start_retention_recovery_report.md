# Exp 013 Phase W2-P1-A7-R3 start-retention recovery

## Outcome

Classification: `EXP013_W2_P1_A7_R3_TRAINING_UNSTABLE`. No label-generation teacher was authorized.

## Existing checkpoint rescue

All 11 A7-R2 checkpoints were evaluated over 24 validation conditions with 200 episodes each. No checkpoint was eligible. The best minimum acquisition was 75.0% at update 10.

## Diagnosis

At update 75, 315°/+0.3 had endpoint 100.0%, 0.10 s acquisition 100.0%, and 0.20 s acquisition 19.0%. Translation, direction, and gait each sustained PASS; yaw sustained PASS was 19.0%, with 22.33 resets and 0.1595 s longest PASS. The mirrored 45°/-0.3 condition acquired at 100.0%. The primary limiter was yaw-rate oscillation.

## Localized continuation

The single fixed continuation used A7-R2 update 75, LR 5e-6, seed 20278631, unchanged reward, and the preregistered 30/25/25/20 command mixture. Update 1 temporary/persistent tensors matched exactly. Update 3 triggered the mandatory early guard: rear -yaw acquisition 95.0%, rear +yaw 70.0%. Training stopped; no held-out evaluation or horizon sweep was run.

## Protection

All protected datasets, labels, splits, manifests, overlays, state pools, replay/mask contracts, prior checkpoints/optimizers, reward, and physics remain unchanged. No V3 overlay, canonical promotion, or remote push occurred.
