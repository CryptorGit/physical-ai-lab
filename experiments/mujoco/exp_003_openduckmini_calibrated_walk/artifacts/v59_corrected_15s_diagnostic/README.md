# v59 corrected 15-second omnidirectional diagnostic

This directory contains a GPU-MJX-only diagnostic of v59 step 33,423,360
through the training-compatible calibrated/backlash controller path.

It is diagnostic-only:

- `formal_acceptance_eligible = false`
- `enough_episodes = false`
- `diagnostic_only = true`
- v59 remains `diagnostic_not_qualified`

Condition D disables observation noise and push, fixes reset/model state, and
uses a deterministic actor. Condition S uses the historical environment's
native reset, domain, observation-noise, delay, stochastic actor, and
episode-time base-velocity impulse path.

The canonical conclusions are in `diagnostic_report.md`. Per-episode numbers
are in the condition CSVs; compressed full-step arrays are in `raw_logs/`;
complete batched episode-start states and randomized models are in
`episode_snapshots/`.

