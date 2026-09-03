# exp_009 Stage 4 — Frozen WALK speed residual

## Classification

**RESIDUAL_PARAMETERIZATION_INADEQUATE**

The frozen WALK base was preserved bitwise on all 594,360 formal-WALK dataset samples. The controller performs no residual addition when the fixed speed gate is zero.

The existing formal exp_006 residual envelope (per-joint ±0.25) was then applied without tuning. Scalar target coverage was train 39.6698%, validation 39.6984%, and test 39.6435%; complete 37D sample coverage was 0% in every split. This is far below the frozen 99.5/99/99% gate.

The global absolute target p99.5 was 5.363 and max was 7.298, compared with the fixed 0.25 bound.

No residual distillation, closed-loop retention, intermediate-speed evaluation, or reverse diagnostic was run. The negative result indicates that RUN/WALK_TO_RUN actions are not a bounded ±0.25 correction around this WALK base. The single next action is to close residual v1 and audit base/action-manifold compatibility before authorizing another method.
