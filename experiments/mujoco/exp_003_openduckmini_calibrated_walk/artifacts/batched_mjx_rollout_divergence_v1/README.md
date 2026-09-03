# Batched MJX Rollout Divergence v1

Result: `BATCH_ONLY_MJX_DIVERGENCE`; gate: `TRUE_BATCH_NONDETERMINISM_CONFIRMED`.

The canonical four-environment checkpoint is bit-exact when each environment is stepped alone. The native batched GPU MJX path diverges from batch size two onward even with a fixed saved motor target, fresh disk reload, no donation, no reset, and identical model/data/action copies. The first difference is in `fwd_position -> smooth.crb -> _impl/crb`, before collision and constraint solving.

Start with `divergence_report.md` and `divergence_report.json`. CSV files contain the raw repeat, ablation, permutation, and leaf-level comparisons. The immutable 18.16 MB checkpoint is referenced by SHA-256 under `serialized_inputs/` rather than duplicated.
