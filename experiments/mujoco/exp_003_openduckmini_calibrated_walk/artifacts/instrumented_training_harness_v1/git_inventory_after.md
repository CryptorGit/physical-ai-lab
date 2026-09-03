# Git Inventory After Instrumentation

- Main baseline commit: `8ae5105caf1af90eb200ffef7a96e985d655614b`
- WSL historical training-source baseline:
  `338451f33e687ea3edcda8a2c2cdcbc8a7b4bda0`
- Harness implementation commit:
  `1c2b26d6f32091e81146da125bc5ffc94b40ba4d`
- Branch: `master`
- Push: not performed
- Tag: not created

All source, scripts, tests, schemas, and small reports created for this harness
are tracked. Checkpoints, raw rollout payloads, caches, videos, large CSVs, and
the immutable tar archive are ignored and referenced by SHA-256.

An unrelated concurrent commit
`e3d396aed8fbf1383eccd2f2aee71de8b21bce89` advanced the repository HEAD
after the harness implementation commit. It does not modify the exp_003
harness. It was preserved and not included as evidence of harness behavior.
Unrelated dirty files elsewhere in the workspace were also preserved.
