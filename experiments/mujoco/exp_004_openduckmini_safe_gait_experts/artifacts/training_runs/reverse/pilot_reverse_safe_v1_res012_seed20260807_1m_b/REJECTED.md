# REJECTED — do not route, package, or deploy

Run `pilot_reverse_safe_v1_res012_seed20260807_1m_b` completed 1,000,000 training interactions, but its learned residual policy failed the recorded exact-safe formal evaluation and was not adopted.

- Routing: disabled
- Package adoption: prohibited
- Hardware deployment: **PROHIBITED**
- Research evidence retention: allowed

The selected straight-reverse simulation route is frozen `v22` plus `optimized_reverse_exact_safe_v1`, residual scale `0.0`, capped at command `[-0.075, 0.0, 0.0]`.

See `rejection_manifest.json` for the machine-readable decision and `../../../../RESULTS.md` for the experiment report.
