# Exp014 Phase 2-D28V hard-limit and actuator authority audit

Classification: `EXP014_D28V_RUNTIME_HARD_LIMIT_UNRESOLVED`.

D28U and all earlier stages were read-only. No physics, policy update, checkpoint, PPO, CEM, validation, held-out, LEFT START, or target/action/gain/timing change was executed.

## Limit hierarchy

Runtime metadata status: `PASS`; PhysX/USD hard-limit metadata candidate: `True`; formal q_actual enforcement gate: `False`; authority-ready hard-limit contract: `False`. Existing formal q_actual violations: `10763` at tolerance `1e-06` rad. Processed limits remain diagnostic soft/evaluation limits and were not applied to q_cmd.
Raw USD joint rows: `37/37`; PhysX runtime rows: `37/37`. Source of truth: `L2_PHYSX_HARD_LIMIT`.

## Formal states

- `P0_S_HOLD_fresh_endpoint`: states `8`; q_actual hard-limit violations `0`; strict empty `0`; monotone recovery empty `0`.
- `P1_S_HOLD_formal_rollout`: states `6400`; q_actual hard-limit violations `1183`; strict empty `0`; monotone recovery empty `0`.
- `P2_W_MOVE_formal_rollout`: states `20000`; q_actual hard-limit violations `9511`; strict empty `0`; monotone recovery empty `0`.
- `P3_D27_actual_V2A_trace`: states `119`; q_actual hard-limit violations `69`; strict empty `0`; monotone recovery empty `0`.

## q_cmd semantics

The canonical route is q_actual (simulation state) -> q_kin (physical WBIK candidate) -> q_cmd (virtual implicit-actuator target). Isaac Lab's position action and articulation setter source path contains no q_cmd hard-position projection; q_cmd position limits are therefore non-canonical.

## Actuator contract

Actuator parity status: `ACTUATOR_MODEL_STATE_MISSING`. The inspected model is the runtime implicit actuator path; fixed-tolerance torque parity was not admitted because the hard-limit positive-control gate failed. D27's macro-step trace does not contain the final decimation substep q/dq at which the persisted actuator telemetry was evaluated.

## Corrected authority

- `C0_ALL_JOINTS`: {"status": "NOT_RUN"}
- `C1_FREEZE_WRIST_HAND`: {"status": "NOT_RUN"}
- `C2_SCALED_ALL_JOINTS`: {"status": "NOT_RUN"}
- `C3_SCALED_FREEZE_WRIST_HAND`: {"status": "NOT_RUN"}
- `C4_SCALED_LEGS_WAIST`: {"status": "NOT_RUN"}

## V3R3

Temporary shadow created: `False`; selected formulation: `None`; physics applied: `0`.

## Next action

resolve PhysX hard-limit enforcement against existing q_actual violations; an isolated hard-limit enforcement probe is required but was not executed in D28V; do not run authority or physics

## Repository

Starting HEAD `90f541296dfbe8864b5b746f73306956dfbb6e21`; ending HEAD before commit `90f541296dfbe8864b5b746f73306956dfbb6e21`. Protected input changes: `False`.
