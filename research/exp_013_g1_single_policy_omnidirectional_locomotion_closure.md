# exp_013 closure

## Final classification

`EXP013_CLOSED_WITH_SINGLE_POLICY_OMNIDIRECTIONAL_LOCOMOTION_SUCCESS_AND_SINGLE_ACTOR_STOP_RESTART_INTEGRATION_UNRESOLVED`

単一方策による全方向歩行・旋回は成功した。完全停止能力も別方策では成立した。しかし、停止維持・全方向再発進・moving yaw・stop recoveryを一つのactorへ統合する課題は、試験した観測・履歴・教師軌道・PPO条件の範囲では未解決として終了した。

## Canonical artifacts

- exp_013: W1B-R2 iteration 200, SHA-256 `61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`
- exp_012 transition actor: Stage 2Q, SHA-256 `66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698`
- yaw calibration: physical yaw <= 0 uses x1.0 actor input; physical yaw > 0 uses x1.5 actor input

## Supported capabilities

- exp_013 one actor / one checkpoint: continuous vx/vy, zero-yaw translation in 16 directions at 22.5° intervals, pure yaw in both directions, and moving yaw.
- exp_012 Stage 2Q only: WALK/RUN bidirectional transitions, speed transitions, and practical stop.

## Unsupported or unresolved capabilities

- One actor jointly retaining stop maintenance, safe restart in all directions, moving yaw, and stop recovery.
- Strict static stand.
- A claim that the exp_013 actor alone performs WALK/RUN gait transitions.

## Protected artifacts

Existing exp_005 through exp_012 outputs, the exp_012 closure, all existing exp_013 stage reports, checkpoints, optimizers, datasets, labels, splits, manifests, overlays, raw state pools, ReplayRecipeV1/V2, MaskedPPOV1, rewards, physics, architecture, and evaluators were treated as immutable. Closure work consists only of read-only playback, media, reports, and closure manifests.

## Deliverables

- Final report: `research/exp_013_g1_single_policy_omnidirectional_locomotion_final_report.md`
- Video: `media/exp_013_g1_omnidirectional_and_motion_transitions_linkedin.mp4`
- LinkedIn posts: `research/exp_013_linkedin_post_ja.md`, `research/exp_013_linkedin_post_en.md`
- Abstract: `research/exp_013_g1_single_policy_omnidirectional_locomotion_abstract_en.md`
- Artifact manifest: `results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/closure/final_artifact_manifest.json`
- Reproduction commands: `results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/closure/reproduction_commands.ps1`

## Repository closure

- Starting HEAD: `cf82e217fe90fc4857f3bd37bb608760999ebe29`
- Commit: resolve with `git rev-parse HEAD` after the closure commit
- Annotated tag: `exp013-single-policy-omnidirectional-closed`
- Remote push: false
