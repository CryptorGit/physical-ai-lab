# H5 V3 transform-ladder time-bound record (2026-08-12)

## Scope

These two executions were attribution diagnostics only. They used the sealed
T=1 fixed-quality-replay ablation; they did not invoke PPO, write a checkpoint,
or perform hardware I/O. The H5 V3 candidate remains rejected.

The fixed-quality replay removes the second post-physics `mjx.forward` replay,
but preserves the H5 arithmetic and the additional graph topology. Its earlier
B=2 raw-repeat witness still diverged despite identical inputs, restored global
state, and exactly one trace. The smaller nonzero divergence was therefore a
failure, not a tolerance-qualified pass.

## Inputs shared by both attempts

- Failing B=2 lane 1 was extracted as the exact S=1 input.
- Canonical entry whole-tree digest:
  `8f7f592f2f14d6f20558db7ecc73f95f457143e7f304b6d53e3b84d5180af0a5`.
- Canonical entry data digest:
  `71150f151e5c3beb4199293f8de20da7d749f432d951f6929e077b9f452822f0`.
- `ladder_treatment_eager_s1_fixed_replay_fresh_process_v1.json` is the
  successful control point: scalar eager repeat is raw-array equal on that
  exact input.

## Time-bounded inconclusive attempts

| Mode | Started (JST) | Stop condition | Result file | Log |
| --- | --- | --- | --- | --- |
| `jit_scalar_s1` | 2026-08-12 04:24 | no result or stdout after more than 7 min | not created | `ladder_treatment_jit_scalar_s1_fixed_replay_fresh_process_v1.stdout.log` (0 bytes) |
| `vmap_b1_s1` | 2026-08-12 04:32 | no result or stdout after 7 min 49 s; exact Windows parent PID 5404 stopped | not created | `ladder_treatment_vmap_b1_s1_fixed_replay_fresh_process_v1.stdout.log` (0 bytes) |

The specific parent process was stopped only after checking that its result
path was absent, its stdout log was empty, and it was the process owning the
listed diagnostic command. No generic process kill was used. A subsequent WSL
process check found no remaining `train_h4_aligned_expert.py` process.

## Interpretation and next gate

This record does **not** classify either JIT mode as a pass or fail. It must
not replace the existing B=2 raw-array failure or be used as authorization for
PPO. The next design gate is architectural: prove a top-level physics rollout
whose treat/control compiled physics program and outputs are byte-identical,
then run any H5 quality calculation as a non-writing sidecar outside that
rollout. If the existing PPO stack cannot consume sidecar rewards without
joining the H5 graph to the rollout, H5 V3 must be rolled back rather than
trained under a weakened determinism gate.
