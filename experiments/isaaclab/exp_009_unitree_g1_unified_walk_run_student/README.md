# exp_009_unitree_g1_unified_walk_run_student

Stage 0 distills the frozen WALK, RUN_LOW, and WALK_TO_RUN teachers into one
continuous-speed-conditioned 123D policy. It uses no PPO, reward objective,
teacher update, production capability change, or runtime controller switch.

The student is diagnostic-only until a later formal stage.

## Stage 0 result

`DISTILLATION_FAIL_INTERFERENCE`

The 1,880,660-step offline dataset and one permitted DAgger round were
completed. RUN behavior was the easiest regime to imitate, but no checkpoint
retained the formal WALK, RUN, and WALK_TO_RUN gates together. The DAgger
checkpoint further degraded safety and retention, so it was not selected.
The unified student remains diagnostic-only and no capability manifest or
production artifact was changed.

```powershell
.\experiments\isaaclab\exp_009_unitree_g1_unified_walk_run_student\scripts\reproduce_stage0.ps1
```
## Stage 1: single-head interference diagnosis

Stage 1 is diagnostic-only. It freezes the Stage 0 dataset/split and separates
raw capacity, hidden-mode aliasing, regime-gradient interference, and
closed-loop dynamic sensitivity. It creates no PPO updates, reward changes, or
production capability.

The resulting gate is `MULTIPLE_FAILURE_MODES`: regime gradients and sequential
forgetting are real, but the more immediate failure survives task isolation.
A WALK-only clone with near-zero held-out one-step error still loses the WALK
attractor, while restoring only the teacher ankle-roll actions recovers much of
the 1.0/1.2 m/s retention. The next single design is therefore a
dynamics-sensitive distillation objective with short-horizon contact/state
matching; it is not implemented in Stage 1.

## Final status

`CLOSED_NO_GO_UNIFIED_ACTION_MANIFOLD`

Stages 2–6 tested the remaining single-controller hypotheses without changing
the protected teachers or production capability:

- the local dynamics-sensitive loss did not restore WALK retention;
- the nonlinear rollout surrogate failed its accuracy and action-ranking gates;
- a frozen-WALK residual could not represent RUN or WALK_TO_RUN inside the
  existing bounded residual;
- the action-pipeline audit confirmed that the large WALK/RUN difference is
  present at the actual applied joint-target level;
- scalar and fixed joint-group WALK/RUN-base morphs could not represent the
  WALK_TO_RUN teacher inside the existing bound. The oracle WALK_TO_RUN morph
  was already RUN-side at transition entry.

The negative result is scoped to the unified action-manifold and morphing
approaches evaluated in exp_009. The frozen WALK, RUN_LOW, and WALK_TO_RUN
teachers remain unchanged. No exp_009 diagnostic checkpoint is a production
capability.

The next experiment is `exp_010_unitree_g1_post_run_walk_attractor`, which
tests an independent post-RUN low-speed attractor rather than forcing
RUN-derived states into the original WALK expert's acceptance basin.
