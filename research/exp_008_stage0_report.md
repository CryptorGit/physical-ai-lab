# exp_008 Stage 0 — observability and controllability

## Scope

This diagnostic used the frozen exp_007 Stage 8C `model_10.pt`; it performed
zero PPO iterations, zero optimizer updates, and no reward, actor, production
observation, artifact, or capability change.

## Dataset

- Episodes: 2048 (1024 at 2.6 m/s and 1024 at 2.8 m/s)
- Rows: 201882
- 20-step successes: 0 (the target of 200 was not reachable with the frozen policy)
- Failed segments: 2048
- Break reasons: {'contact': 1579, 'heading': 41, 'safety': 2, 'speed': 426}
- Launch phases: {'flight': 671, 'left': 726, 'right': 651}
- Split: complete episode/reset-seed/source-speed/checkpoint groups, 60/20/20; no step leakage

## Observability

The primary timing-ablated static 152D probe achieved AUROC 0.982
and AUPRC 0.871 at prevalence 0.026 for
contact break within three steps. Removing timing fields did not reduce AUROC.
However, ridge time-to-break MAE was 5.18
steps, above the preregistered 1.5-step limit. The 16-step GRU reached AUROC
0.989, an improvement of only 0.008.
Legacy 123D AUROC was 0.978; explicit phase upper-bound AUROC
was 0.981 with MAE 5.16.

Fixed classification: **BREAK_NOT_PREDICTABLE**. Near-term binary risk is
highly rankable, but the complete preregistered observability gate (including
exact time-to-break) is not met.

## Controllability

Fresh Isaac applications replayed identical reset seeds, source routes, and
prebranch actions. Comparisons used only identical physical env IDs at identical
branch ages/steps. Root, joint, and velocity errors were exactly zero for all
accepted comparisons; no state was copied.

No candidate produced a safe 20-step contract:

- baseline: 0/512
- frozen WALK: 0/512
- frozen RUN: 0/512
- bounded joint groups: 0/512
- bounded target-WALK alignment: 0/512

Fixed classification: **NO_LOCAL_CORRECTION_FOUND**.

## Stage 0 decision

**UNIFIED_WALK_RUN_DISTILLATION**

The audited local corrections did not enter the target basin, while neither
history nor explicit phase passed the full observability gate. The next single
implementation should therefore be a unified WALK/RUN trajectory-distillation
study, not a production GRU or a reward-only continuation.
