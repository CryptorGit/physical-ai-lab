# Exp 013 Phase W2-P1-A7-R1 rear-yaw start teacher masked-PPO run

Classification: `EXP013_W2_P1_A7_R1_MASK_CONTRACT_IDENTITY_FAIL`.

The mandatory identity gate stopped the run before PPO collection or any optimizer update. Three fresh launches reproduced the same batch-0 mismatch: the live replay selected 1017 formal-stop environments while the authorized M0 mask selected 1018, with 11 environment identities disagreeing. The differences are physical, not threshold rounding. For example, environment 207 changed from M0 speed/yaw 0.00947/0.00234 and PASS to R1 speed/yaw 0.37568/0.71188 with fall and slip.

Because the authorized contract requires per-state semantic identity and exact accepted IDs before the persistent run, no policy update, training checkpoint, checkpoint selection, held-out evaluation, or rear-yaw teacher artifact was produced. The parent, S0 replay recipe, M0 masks, datasets, labels, overlays, reward, and physics remain unchanged. Snapshot restore and unmasked PPO were not used.
