# exp_013 Phase W2-P1-A7-R2 rear-yaw start teacher report

## Outcome

Classification: `EXP013_W2_P1_A7_R2_RETENTION_FAIL`. The single authorized 150-update masked-PPO run completed, and validation selected update 75. Rear 0.3 m/s with yaw -0.3/+0.3 reached held-out acquisition 0.993/0.997 with endpoint 1.000/1.000 and no fall or slip. The teacher gate nevertheless failed closed because the full start matrix passed 23/24 conditions: 315 degrees with yaw +0.3 retained endpoint 0.993 but acquisition was only 0.277.

## Identity and training contract

Replay V2 reproduced 6,144/6,144 S0 accepted IDs and semantic hashes across all seven full batches, including all 11 previously divergent environments. The environment mask hash is `0e32a2b41eae4996c1ec6acf7ef929c473af76e9685e14c2f12f738e1b9e6fb6`. The masked one-update input contained 24,432 samples per mirror sign (48,864 total), with zero loss, gradient, and updated-tensor difference from M0; invalid and non-train leakage was zero. Parent actor, critic, optimizer, Identity normalizer, sampler/RNG, curriculum, and empty mirror queue restored strictly after replay identity.

Update 1 exactly reproduced KL 0.000585189, clip fraction 0, gradient norm 6.16032, and value loss 0.00298923. The corrected complete-unit accumulator used at least 24,576 valid samples per update. All 150 updates completed with 7,367,609 PPO-valid interactions, 171,417,600 teacher roll-in env-steps, 49,293,312 prefix env-steps, and 229,902,336 total simulator env-steps. The initial incomplete update-5 attempt was excluded before any formal update was committed; its diagnostic evidence is isolated under `raw/invalid_update_005_effective_batch_attempt`.

## Validation selection

Update 75 was frozen using validation only. Both rear 0.3 yaw conditions had 0.99 acquisition, mandatory endpoint/safety and static retention passed, and fresh-process next-collection tensors matched bitwise for both mirror signs. No held-out fallback was used. Checkpoint SHA-256: `1cf290ace57bd9be4aeb0199a41b643b8604757bd3b788f2c98cec17e3f65028`.

## Held-out authorization

Rear target, pure-yaw starts, static endpoint retention, symmetry, and aggregate safety passed. The full 24-condition start gate failed at 315 degrees/yaw +0.3 (acquisition 27.7%; endpoint 99.3%; fall/slip 0). Accordingly `rear_yaw_start_teacher.json` was not created, StartBoundaryTrajectoryOverlayV3 was not reopened, and no canonical runtime promotion occurred.

## Protection audit

Base datasets, labels, splits, manifests, overlays, the formal stop pool, V1/V2 replay contracts, the masked-PPO contract, existing checkpoints/optimizers, reward, physics, W2-P1-R2 step 37000, and the A4 candidate were not changed. Only A7-R2 code and artifacts were created. No remote push was performed.
