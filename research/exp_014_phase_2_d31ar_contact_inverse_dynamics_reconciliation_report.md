# EXP014 Phase 2-D31A-R — contact/inverse-dynamics reconciliation

## Scope

Fresh Isaac/PhysX Route A used P0 S_HOLD followed by native W_MOVE for R0-R7.
The protected D31A WBC authority result was not rerun or changed. Runtime
state, body Jacobians, mass/inertia-derived dynamics, net foot forces, and
read-only actuator contracts were captured around the first strict TD0.

## Availability

The runtime exposes body Jacobians, masses, inertias, q/dq/root state, applied
and computed torques, and net foot forces. The contact-point, normal-impulse,
and penetration APIs were `NOT_AVAILABLE`;
these quantities are therefore recorded as `NOT_AVAILABLE`, not inferred.
When the backend folds decimation into one simulator update, exact internal
physics-substep offsets are likewise explicitly unavailable.

## Reconstruction

Continuous force and impulse residuals, point-force ankle proxies, aggregate
sole wrenches, finite-difference contact kinematics, friction witnesses, CoP
witnesses, and the F0-F6 feasibility ladder are emitted in the result
directory. The sole polygon and actual rigid-contact wrench feasibility remain
unadjudicated.

## Scientific adjudication

Classification: `EXP014_D31AR_MULTIPLE_CONTRACT_MISMATCHES`. Authority supported: **no**. Authority
status: **NOT_YET_ADJUDICATED**. Official D31A classification remains
`EXP014_D31A_TORQUE_WBC_CONTACT_AUTHORITY_FAIL`.

## Repository

Starting HEAD: `9b20c92880f5d37dfef690078aff0c48c5e075a0`; execution HEAD: `9b20c92880f5d37dfef690078aff0c48c5e075a0`.
No protected artifact, checkpoint, runtime setting, or unrelated worktree
state was modified.
