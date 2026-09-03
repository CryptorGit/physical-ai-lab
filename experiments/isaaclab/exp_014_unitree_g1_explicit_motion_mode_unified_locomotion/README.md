# EXP 014 — Explicit motion-mode unified locomotion

This experiment distills the formally scoped EXP 012/013 specialists into one feed-forward actor with one Gaussian head. Runtime routing, teacher queries, checkpoint switching, and action blending are forbidden. The frozen 124D observation is retained verbatim and 17 causal mode/history features are appended.

See `protocol.md` for gates and `scripts/run_exp014.ps1` for the fail-closed execution order. Results are written only below `results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion`.

## Current outcome

The registered run is classified `EXP014_STATIC_PASS_PHYSICAL_FAIL`. The 141D contract, 680 dual-mode pairs, dataset integrity gates, and the S0 static joint-solution gate passed. Two DAgger rounds improved practical STAND but did not meet physical Phase 2 gates; consequently RUN and OMNI-RUN phases were not authorized. No physical capability is promoted from this checkpoint.

## Pause status

Status: `PAUSED at D29C`

Resume: `post-touchdown stable WALK capture`

Latest report: `research/exp_014_pause_report_d29c.md`
