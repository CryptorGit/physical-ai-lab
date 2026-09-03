# Exp014 Phase 2-D24A existing START Teacher transfer audit

## Outcome

Main classification: `EXP014_D24A_STAGE2Q_NATIVE_REPRODUCTION_FAIL`.

The frozen Stage 2Q checkpoint reproduced its historical native gait classification on 100/100 train-only episodes. The stricter D24A gate did not pass: only 37.00% completed the forward-velocity 25-step confirmation, torque saturation dwell occurred in 31.00%, and safe joint success was 29.00%. Falls, dangerous slips, impacts, velocity saturation, and non-finite states were zero.

This does not revise the committed exp012 claim. It establishes that the old gait-classification success contract is not sufficient for the new D24A START-source prerequisite.

## Native first-step dynamics

An identifiable first single-support/forward-displacement event occurred in 77/100 episodes. Its median time was 1.320 s (p05 1.000 s, p95 2.100 s), measured from the native reset timeline. All observations, actions, states, contacts, forces, and foot traces were durably stored in the native NPZ; per-episode classifications were committed to SQLite with WAL/FULL settings.

## Conditional branches

Because the native mandatory gate failed, protocol section 6 required an immediate stop. R0 through R4, the S_HOLD-source transfer, Stage 2Q to W_MOVE handoff, successful-demonstration construction, and temporary static distillation were not executed. Their artifacts contain explicit `NOT_EXECUTED` reasons; no missing values were inferred.

## Protection

No policy or optimizer was updated, no checkpoint was created, the fixed validation 102 snapshots and all held-out data were untouched, and no RUN integration or remote push occurred. Existing D6-D23 artifacts and the frozen S_HOLD, Stage 2Q, W_MOVE, and S_STOP_OMNI checkpoints were not modified. Unrelated pre-existing working-tree changes were preserved.

## Next

The only recommended next experiment is D25 phase-conditioned low-speed STEP continuation. Direct Teacher distillation and demonstration-based DAgger are not authorized because the prerequisite native gate did not establish a usable demonstration source.
