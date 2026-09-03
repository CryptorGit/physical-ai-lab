# EXP014 Phase 2-D30B nonlinear post-touchdown reachability

## Local dynamics

D30B uses deterministic direct shooting with the protected D30A 4D
`WMoveCaptureActionBasisV1`, native p95 coefficient bounds, and no neural or
local surrogate. The phase tube is `WMove03PhaseTubeV1` with 50 LEFT and 50
RIGHT D26T references.

## Baseline

The exact fresh D29B Route-A `S_HOLD -> W_MOVE -> TD0` lifecycle is the
baseline for source-specific R0-R7. Prefix parity is required before any
candidate result; all 128 coordinate candidates were replayed in fresh
processes against the same source indexing.

## Capture MPC

No MPC, PPO, CEM, random search, Bayesian search, reward, training, Student,
or RUN integration is used. D30B evaluates fixed minimum-jerk phase-block
controls by fresh PhysX direct shooting.

## Stable capture

Best capture count: `0`; stable count:
`0`; release count: `0`;
100-step retention count: `0`.
Per-source `d2`, `d4`, `d4/max(d2,eps)`, yaw, velocity, effort, safety, and
evaluation counts are stored in `stable_capture_results.json` and the source
candidate ledger. No source reached TD4 in the selected no-go result, so d2/d4
and post-TD4 release values remain explicitly null rather than fabricated.

## Handoff

The hard switch to W_MOVE and 8-step zero release are required before the
100-step pure W_MOVE retention gate.

## Failure decomposition

- CANDIDATE_ADAPTER_FAILED:20
- CANDIDATE_ADAPTER_FAILED:112
- support_loss
- support_loss
- support_loss
- torque_saturation
- NO_STABLE_CAPTURE

## Classification

`EXP014_D30B_MULTIPLE_FAILURES`

## Recommended next action

Do not authorize transfer; retain the registered failure and redesign only after review.

## Repository

Starting HEAD: `454f4b0371d4e46703f2530b890ad63a11e0bfb1`; ending HEAD: `9b20c92880f5d37dfef690078aff0c48c5e075a0`. D30B paths are
filtered from pre-existing status. Protected hashes and the protection audit
are in `protected_hashes.json` and `protection_audit.json`.
