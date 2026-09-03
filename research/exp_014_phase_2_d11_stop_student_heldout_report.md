# exp_014 Phase 2-D11 sealed held-out report

## Result

Classification: **EXP014_D11_HELDOUT_RUNTIME_INTERRUPTED**. Pre-open identity, sealed hash, split integrity, and protected-path checks passed. The one-time ledger was written before the sealed payload was deserialized.

All 579 sealed episode IDs across 34 conditions were attempted exactly once and batch 2 was committed to the access ledger. The simulation context then terminated the process before episode metrics were serialized. The ledger therefore has zero unevaluated episodes, but no recoverable formal outcomes. Re-running completed held-out episodes is prohibited, so moving-start, STOP, hold, joint, safety, condition-minimum, and handoff results are reported as unavailable rather than inferred.

No checkpoint was selected or changed, no fallback or training was performed, and `Exp014DistilledOmnidirectionalStopSpecialistV1` is **NOT_AUTHORIZED**. The sealed access count is one. The next permissible work is a lifecycle/result-serialization audit that does not reopen or rerun the held-out split.
