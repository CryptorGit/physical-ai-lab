# exp_014 Phase 2-D14 D11R durable held-out report

## Result

Classification: **EXP014_D14_OMNI_STOP_REPLACEMENT_HELDOUT_PASS**. The sealed replacement was opened once for the sole frozen S1 step-30000 candidate. All 680 results were committed by the parent-owned SQLite WAL/FULL store before simulation-context teardown; offline aggregation was bitwise-identical in two runs.

## Capability

- Moving-start validity: 97.3529% (662/680)
- Conditional STOP acquisition: 99.8489%
- Conditional S_HOLD: 100.0000%
- Conditional joint success: 99.8489%
- End-to-end success: 97.2059%
- Minimum condition joint success: 95.0000%
- Fall/slip/impact: 0.1471% / 0.1471% / 0.0000%
- Velocity/torque saturation: 0.1471% / 0.2941%

No fallback, training, PPO, DAgger, RUN integration, checkpoint change, or original-D11 reopen occurred.
