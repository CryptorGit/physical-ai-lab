# Exp014 Phase 2-D24C S_HOLD source and safety parity audit

## Result

Main classification: `EXP014_D24C_SNAPSHOT_RESTORE_CONTRACT_BUG`.

The canonical 100-episode fresh lifecycle reproduced RESET_TO_STAND at 99%, conditional STAND_HOLD at 100%, and joint success at 99%, with zero canonical joint-velocity saturation. The current harness therefore has S_HOLD parity.

The exact 320-environment D24B boundary replay reproduced 0/320 continuous-valid sources: all environments were outside threshold at post-restore steps 0 and 1, 120/320 were valid at step 2, 316/320 at step 3, and 320/320 at step 4. The committed snapshot contains physical state, actions, command state, and episode length, but not contact history, air-time/last-contact buffers, actuator state, termination buffers, or observation history. It is `R2_PHYSICAL_STATE_ONLY` and cannot establish a formal START source by raw restore. An isolated replay reached 64/64 after one S_HOLD warm-up step.

D24B's torque metric is not the canonical S_HOLD saturation metric. Canonical D5 uses joint velocity divided by velocity limit, threshold `>0.95`, five-step dwell. D24B used clipped applied torque divided by effort limit and `>5` dwell. A ratio of exactly 1.0 is actuator clipping, not by itself a canonical S_HOLD unsafe event.

The conditional exact Stage2Q 10 s replay reproduced the historical flight-fraction classification in 100/100 episodes and 80 first-step events, but strict successes and canonical E2 safe demonstrations were both zero. Stage2Q's native prestart is therefore not an available model-based bridge target under this audit.

The only next experiment selected by precedence is a fresh-lifecycle S_HOLD-to-Stage2Q transfer audit using `Exp014FreshS_HOLDSourceLifecycleV2` and the canonical saturation metric. No START transfer, bridge, D25, persistent update, checkpoint, validation, held-out access, or remote push occurred in D24C.
