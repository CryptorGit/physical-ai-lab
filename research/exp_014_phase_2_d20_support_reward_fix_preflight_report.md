# Exp014 Phase 2-D20 — support reward correction and preflight replay

## Outcome

Primary classification: `EXP014_D20_D18_REPLAY_IDENTITY_FAIL`. The two implementation defects were corrected in the versioned `Exp014OmnidirectionalStartRewardV2R1` implementation, and all 21 synthetic regression tests passed. D18R persistent training is **not authorized** because the exact D18 captured rollout was not persisted.

## Corrections

The target now ramps from 0 to 0.7 through 0.35 s, remains 0.7 until 0.75 s, and is independent of the 0.50–0.75 s weight decay. Load transfer is masked to zero unless either foot's canonical contact-force norm exceeds 5 N. No reward weight, sigma, architecture, command, optimizer, observation, or physics setting changed.

## Regression tests

The 11 D19 tests and 10 D20 additions passed (21/21). At 0.60 s the target is 0.7 while weight remains positive. Both zero-support fixtures produce exactly zero load reward, while valid 50/50 and mirrored 85/15 fixtures retain the intended maxima/invariance.

## Replay identity

D18 stored aggregate preflight metrics, calibration, stability, references, and initial parity. It did not store the captured observation/action/physics-state records, their hashes, or episode IDs. The required exact replay therefore cannot be established. A replacement rollout was not collected; Isaac Lab, actor inference, and policy updates were not run. Consequently the preventive-yaw reproduction, corrected support probe, full V2R1 stability probe, and corrected gradient calibration are all `NOT_EXECUTED`.

## Authorization

D18R is not authorized. The next experiment should preregister and durably persist an identity-complete train-only captured rollout under the unchanged D18 conditions, then run the corrected term decomposition and signed-support audit. Persistent PPO remains prohibited.

## Protection

D6–D19 artifacts and all existing checkpoints/datasets were read-only. Persistent updates: 0. New persistent checkpoints: 0. Actor-input/formal-gate/reward-weight changes: 0. Remote push: false.
