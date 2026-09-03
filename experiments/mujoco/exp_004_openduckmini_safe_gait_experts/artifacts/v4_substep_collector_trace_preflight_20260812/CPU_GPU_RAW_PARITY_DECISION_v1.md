# V4 collector raw-parity backend attribution

Status: **CPU collector boundary PASS; GPU B=2 boundary FAIL; PPO and hardware remain prohibited.**

## Fixed evidence

- GPU B=2/T=20 fresh-process preflight:
  `capture_b2t20_fresh_process_v6.json`, SHA-256
  `a3a59adcd356a36c843f59b3f3b045f18cde306821ceb072bfea3066d3f1159d`.
  It fails same-arm capture and baseline replay, capture-vs-baseline final/history,
  and trace raw equality.  The first physical divergence is batch lane 1 and
  propagates from `data.qpos`, `data.qvel`, and `data.qacc`.
- CPU B=2/T=20 fresh-process preflight:
  `capture_b2t20_cpu_fresh_process_v7.json`, SHA-256
  `c86d651cffaf1b9c5587069a76c90b5a2c0e63106e80ffddf92f7c2704eea65d`.
  Every raw-equality check passes: repeated collector input/full/core/trace,
  baseline repeated core, capture-vs-baseline initial/final/history core, and
  the 20 x 2 x 10 x 2 force/slip trace shape/finite/nonnegative checks.

Both executions bind the same scene and `h4_training_alignment.py` SHA-256
`217f9fef0e66f621b556bbaa69e77588cd9890831d34fab8b017f335d048a4aa`.
The H5-V3 flags were disabled, no H5-substep token appeared in the collector
StableHLO, and no PPO, checkpoint, or run directory was created.

## Decision

The evidence attributes the raw failure to the GPU B=2 execution path, not to
the V4 trace output or H5 sidecar scoring.  The CPU result is a limited
architecture-boundary proof only.  It does **not** authorize GPU PPO,
qualification, promotion of the rejected H5 250k candidate, deployment, or
hardware.

The only permitted next experiment is a CPU no-PPO pure-sidecar checkpoint:
consume the immutable V4 trace after collection, prove terminal/reset/debounce
causality and one-time reward scaling, and prove that no physics state is
modified.  If that check fails, retire H5 to offline evaluation only.  GPU PPO
remains blocked unless a separate exact B=2 GPU collector remedy is established
without tolerances.
