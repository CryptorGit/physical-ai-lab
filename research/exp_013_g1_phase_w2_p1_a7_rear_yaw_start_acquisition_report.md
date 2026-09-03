# Exp 013 Phase W2-P1-A7 rear-yaw start acquisition preflight

## Outcome

Classification: `EXP013_W2_P1_A7_STOP_STATE_RESTORE_FAIL`.

The W1B-R2 iteration-200 checkpoint is intact (`61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`). Its actor, critic, optimizer, Identity normalizer, sampler RNG, command RNG, curriculum state, and empty pending-mirror queue are serialized. Adam state is exactly step 8,000, so the strict parent resume gate passes.

The required formal-stop initial-state contract cannot be reproduced. The historical W2-P1 collection source resets the simulator and runs the exp_012 teacher for three seconds; it does not restore a saved full simulator state. The immutable chunks contain observations, commands, actions, contacts, and outcomes, but omit root state, complete joint simulator state, rigid-body/contact history, episode-manager state, and randomization state. They therefore cannot serve as a contact-consistent simulator restore pool.

Per the preregistered fail-closed rule, parent baseline rollout, one-update preflight, the 150-iteration PPO continuation, checkpoints, selection, and formal evaluation were not run. No teacher artifact was created.

## Protection

No existing dataset, label, split, manifest, overlay, checkpoint, optimizer, reward, physics, W2-P1-R2 student, or A4 candidate was changed. New checkpoint count is zero; no push was performed.
