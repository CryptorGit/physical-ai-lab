# Phase 2-D26V — START-source endpoint eligibility correction and prescribed-floating-base WBIK V2

Classification: **EXP014_D26V_OFFLINE_START_KINEMATICS_FAIL**.

## Source eligibility

The D26U full-lifecycle classification remains `EXP014_D26U_FRESH_SOURCE_CAPTURE_FAIL` and its artifact is read-only. The D26V START endpoint contract evaluated the last 50 control steps. Endpoint-eligible recipes: **8/8**; capture OFF/ON parity: **PASS**. Recipe 0 and 3 retain their D26U torque event as pre-acquisition diagnostics only; the replayed endpoint windows contain no fall, slip, impact, support loss, velocity saturation, torque saturation, or nonfinite event.

## WBIK V2

`Exp014PrescribedFloatingBaseHierarchicalWBIKV2` is `FB1_PRESCRIBED_ROOT_REFERENCE`. The six root Jacobian columns contribute prescribed root twist to stance-foot, swing-foot, and CoM tasks; only the 37 joint columns are solved and converted to the unchanged 37D action interface. Hierarchy: stance-foot 6D, CoM xyz and swing-foot 6D, torso/nominal/action-rate regularization. Unit tests: **PASS**; independent-process determinism: **PASS**.

## Offline plans

The fixed grid registered **432** plans. WBIK V2 executed **432** plans and found **0** eligible plans. Physics execution was 0. Dominant failure classes: `{"ACTION_BOUND_INFEASIBLE": 432}`. Timing diagnosis: `ACTION_BOUND_INFEASIBLE_PRECEDES_TIMING`.

## Coverage and authorization

LEFT coverage: **0/8** recipes; RIGHT coverage: **0/8**; mirror-equivalent tuple coverage: **0/8**. Authorization: **none**. No model-based START physics was run in D26V.

## Protection and repository

Starting HEAD: `ab59baf29f16b78f5724122e164b4e06aa201de5`; ending HEAD at artifact generation: `ab59baf29f16b78f5724122e164b4e06aa201de5`. Protected D26U/D26T/D26S artifacts, S_HOLD, Stage 2Q, W_MOVE, S_STOP_OMNI, D26 WBIK V1, physics/PD/friction/robot/dt/decimation were not modified by this stage. Persistent update 0; new checkpoint 0; PPO/CEM/validation/held-out/RUN 0; remote push false.

The complete ledger, task-error rows, endpoint audit, V1/V2 comparison, protection hashes, and reproduction commands are in the D26V result directory.
