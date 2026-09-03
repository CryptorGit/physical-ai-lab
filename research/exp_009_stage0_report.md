# exp_009 Stage 0 — Unified WALK/RUN distillation

Classification: `DISTILLATION_FAIL_INTERFERENCE`.

The selected diagnostic checkpoint is `C:\Users\user\workspace\physical-ai-lab\results\exp_009_unitree_g1_unified_walk_run_student\stage0_multiteacher_distillation\checkpoints\epoch_10.pt` (`9b98c94d8143568cfa64625ccb6b3f7cd26147518ceb8aac44149c0605722fa8`). The dataset contains 1,880,660 actual frozen-teacher steps. The student is a single 123D→37D ELU policy; it receives a continuous command and no teacher/skill identity. No PPO, reward optimization, teacher update, production promotion, or capability-manifest change was performed.

## Retention

- WALK gate: False
- RUN gate: False
- WALK_TO_RUN gate: False

## Reverse diagnostic

Reverse transition emerged: False. This remains diagnostic-only.

## Next action

Re-diagnose the single-head student architecture.
