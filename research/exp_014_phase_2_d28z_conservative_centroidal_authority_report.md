# EXP014 Phase 2-D28Z conservative centroidal authority

Classification: `EXP014_D28Z_BOUNDED_SOLVER_FAIL`.

## Conservative contract

The V6 candidate uses q_actual as the measured state, q_kin_next=q_current+dt*dq for active variables, nominal-limit containment when q_current is inside, and non-worsening signed penetration when q_current is outside. q_cmd is q_kin_next plus the protected feedforward offset and is not position-clipped. Pass-through commands are exact D27 V2A q_cmd values.

Sanity contract pass: `True` (`S0=PASS`, `S1=PASS`, `S2 outward motion rejected`, empty intervals `0`); pass-through pass: `True` (`C3=14`, `C4=24` pass-through joints, q_cmd mismatch `0`, actuator parity `PASS`). The three D28Y unresolved directions were preserved without reclassification and were not used as a V6 closure gate.

## C3/C4 replay

- `B2_C3_SCOPE_NULLSPACE`: solver success `112/115`; critical H_z improvement `9/9` for each of R4–R7, but critical mandatory gate `0/9` for each source. Median H_z improvement over the full trace was `0.99999927`; failures were predominantly `STANCE_TASK_CONFLICT` with `SWING` co-failure.
- `B3_C4_SCOPE_NULLSPACE`: solver success `115/115`; critical H_z improvement was R4 `1/9`, R5 `2/9`, R6 `1/9`, R7 `1/9`; critical mandatory gate `0/9` for every source. Full-trace median H_z improvement was `-1.13589664`.
- Active penetration worsening was `0` in both candidates. Critical active velocity-ratio maxima were `0.2091` (C3) and `0.1347` (C4); critical effort-ratio maxima were `0.4669` in both candidates.

## H_z-first diagnostic

- `B4_C3_HZ_FIRST_DIAGNOSTIC`: critical H_z improvement `9/9` for R4–R7, but critical mandatory gate `0/9` for every source; median improvement `0.99999996`. Diagnostic only.
- `B5_C4_HZ_FIRST_DIAGNOSTIC`: critical H_z improvement `9/9` for R4–R7, but critical mandatory gate `0/9` for every source; median improvement `0.99999943`. Diagnostic only. These results establish that H_z reduction is available only by violating the first-step hard-task contract.

## Root cause

H_z-first/task conflict established: `True`. The active-set solver also returned `SOLVER_NUMERICAL_FAILURE` on 3 B2 full-trace rows and 14 B5 diagnostic rows; under the specified precedence this makes `BOUNDED_SOLVER_FAIL` the main classification. The conservative V6 constraint itself did not block the critical rows: penetration worsening was zero, and no effort/velocity limit was the dominant critical failure. Unresolved PhysX directions were not used as a closure criterion; V6 only prevents worsening the measured penetration.

## V3R6 shadow

Selected formulation: `None`; status `NOT_CREATED`; independent-process determinism `True`; physics `0`. Runtime shadow authorization is therefore not issued. The no-go artifact records `UNRESOLVED_SOLVER_BLOCKED`, rather than falsely closing the branch on the unresolved PhysX directions.

## Protection

Protected D28R/D28S/D28W/D28Y inputs unchanged: `True`. New physics/probe `0`; persistent update `0`; checkpoint `0`; LEFT START `0`; PPO/CEM/validation/held-out/RUN `0`; remote push `false`.

Starting HEAD: `0ff6846360c88d8329dbb819b685a1261ab5ec44`. Ending HEAD before commit: `0ff6846360c88d8329dbb819b685a1261ab5ec44`.
