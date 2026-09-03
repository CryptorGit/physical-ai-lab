# EXP014 Phase 2-D26W action semantics and endpoint feedforward audit

Classification: `EXP014_D26W_OFFLINE_START_STILL_INFEASIBLE`.

## Canonical action contract

The fresh runtime positive control measured `agent.clip_actions = None`, wrapper clipping `None`, and `JointPositionAction.cfg.clip = None`. The runtime mapping is `q_cmd = default_joint_position + 0.5 * raw_action`; the target buffer is written directly. D26V's `[-1,1]` evaluator is therefore a contract mismatch.

S_HOLD positive control used 100 fresh reset streams for 150 steps; P0 source probes were True, and W_MOVE current/next medoid probes were True / True. Wrapper mutation was 0.0 and term raw parity was 0.0.

## Command offsets

Source and target offsets are kept per recipe and per side. The mapper uses the source offset at `u=0` and the corresponding LEFT/RIGHT target offset at `u=1`; it does not average or mirror targets. Endpoint parity passed with max errors 0.0 and 0.0.

## Original failures

D26V had 432/432 plans fail the artificial normalized bound. First-violation decomposition is in `d26v_first_violation_decomposition.json`; duration counts were 144/144 at each registered multiplier. Under the canonical unbounded runtime contract, the action-bound failure disappears, but the protected WBIK trace retains substantial joint-velocity and task/endpoint failures.

## Offline replay

The fixed 432-plan ledger was replayed read-only. D26V original eligibility was 0/432; A0 canonical-bound-only eligibility was 26/432; V2A endpoint-feedforward eligibility was 26/432. A0 coverage was LEFT 0/8, RIGHT 5/8, mirror tuples 0/8. V2A coverage was LEFT 0/8, RIGHT 5/8, mirror tuples 0/8. D26V's original action-bound failure was 144/144 at each registered duration; after removing only that evaluator bound, the remaining A0/V2A dominant failure was JOINT_VELOCITY_INFEASIBLE (406 plans), with 26 plans passing all retained gates.

## Task ablation

T0 through T5 were evaluated on the fixed `SHIFT0.40 / SWING1.0 / C75` diagnostic tuple per source and lead. The canonical runtime bound is unbounded, so no task family creates a canonical action-bound failure; the D26V `[-1,1]` diagnostic remains recorded separately. T0 had 0/16 D26V-bound diagnostic failures, T1 8/16, T2 16/16, and T3/T4/T5 16/16; the essential CoM/swing additions, not the priority-2 regularizers alone, are where the diagnostic bound violations appear.

## Authorization

D27 is not authorized. No model-based START physics was executed. The remaining work is to separate source-target geometry from joint-velocity/task authority; do not change W_MOVE, the fixed grid, reward, or begin PPO.

## Protection

D26U and D26V artifacts were read-only; protected hashes before/after are recorded in `protected_hashes.json`. Persistent policy update: `0`; new checkpoint: `0`; physics: `0`; raw restore: `0`; PPO/CEM: `0`; validation/held-out: `0`; RUN integration: `0`; remote push: `false`.
