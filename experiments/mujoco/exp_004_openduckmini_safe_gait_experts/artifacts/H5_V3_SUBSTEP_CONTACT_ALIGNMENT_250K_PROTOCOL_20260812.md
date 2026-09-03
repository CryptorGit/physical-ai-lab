# H5 V3 2 ms contact-quality alignment protocol

Status: **PRE-REGISTERED PREFLIGHT ONLY — no training is authorized by this
document.**  Hardware deployment remains `PROHIBITED`.

## Why this checkpoint exists

The clean-V22, direct-V3 mapper-only 250k candidate is rejected by
`h5_unified_250k_v3_direct_cleanv22_notarget_v1_1x6s_stand2s_20260811.json`
(SHA-256
`ab6cb84e67f03ad11702929797994f15c817227ce2f59777a131904f2f101947`).
It completes all 38 segments without a fall, but passes only 3/38 strict
gait-quality segments and passes none of the 12 required moving commands.
The recurrent failures are contact-debounce robustness, stance slip,
normal-force tail, cadence, and alternation.

The rejected force-tail/contact-pulse scalar experiment must not be repeated
as a coefficient sweep.  The present hypothesis is narrower and falsifiable:

> The existing H5 training path records a 2 ms contact trajectory but its
> quality losses do not preserve the strict evaluator's four causal debounce
> windows and do not aggregate slip/normal-force quality over every retained
> 2 ms sample.  Making those *measurements* identical, while leaving actor,
> command distribution, decoder, simulator, guard, and all pre-existing
> reward scales unchanged, is required before another PPO pilot can test
> whether the actor can learn an acceptable gait.

This is not a claim that the current actor lacks a periodic input.  The actor
already receives `cos(phase)`, `sin(phase)`, command, contact, normal-force,
and force-weighted tangential-foot-speed channels in its 116-vector.  No CPG,
recurrent architecture, new actor feature, H3 profile/table, target-space
seed, or teacher is authorized here.

## Frozen inputs and non-negotiable invariants

- Parent, if a later pilot is authorized: clean V22 checkpoint
  `/home/user/openduck_training_runs/calibrated_hybrid_yaw_cost_v22_300m/2026_07_29_154427_10485760`,
  tree SHA-256
  `fe35e5ee932dc0ba70c1c32f3e410ea469d229e69cab43ed85f34aefe9505f1f`.
- Command contract: `OPEN_DUCK_MINI_H5_UNIFIED_COMMAND_ROUTING_V3_DIRECT_NORMALIZED`;
  mapper `direct_normalized_v3`; physical command remains `[vx, vy, wz]` and
  policy command remains `[2 vx, (5/3) vy, 2 wz]`.
- One 116-wide actor and one parameter file serve both aliases.  The final
  guard remains exactly once per 20 ms control tick.  Target limits, 0.05 rad
  margin, 2 rad/s slew, head lock, source closure, and all strict evaluator
  thresholds are immutable.
- No hardware connection, serial operation, torque command, calibration edit,
  H3 target/profile authority, 54-row table, target-space seed, H4 parent, or
  reward-scale search is allowed.

## Required preflight implementation and evidence

The preflight must use the existing 10 × 2 ms post-physics replay, without a
second physics trajectory.  It must retain the following values at every
sample: raw per-foot force-Schmitt contact, per-foot normalized normal force,
total normalized normal force, and force-weighted tangential foot speed.

1. Calculate causal contact state independently for **10, 20, 30, and 40 ms**
   using the strict evaluator's semantics: a change begins at its first raw
   sample and commits only when elapsed sampled time reaches the applicable
   window.  Incomplete terminal pending transitions are right-censored, just
   as evaluator segment finalization does not invent a touchdown/liftoff.
2. From the same samples calculate per-window touchdown sequence, contact,
   single-support, flight, and alternation summaries.  From force-qualified
   samples calculate per-foot/all-foot slip RMS, p95/tail accounting, stance
   cumulative distance, steady normal-force band, and normal-force tail.
   These are measurement-aligned diagnostics; they must not weaken or replace
   `gait_quality.py` acceptance.
3. Evaluate the frozen V3 250k weight action-for-action with measurement
   telemetry disabled and enabled.  Dynamic MJX leaves, raw action, desired
   target, guarded target, qpos, and qvel must be exactly equal.  A telemetry
   change that affects a physics state, action, target, guard count, or source
   trajectory is a hard failure.
4. Export raw 2 ms arrays from a fresh, deterministic simulation-only V3
   preflight.  Re-run `GaitQualityAccumulator` over those arrays and prove
   equality of the four per-window debounce summaries and the finite-sample
   slip/force aggregates.  A stored aggregate JSON alone is insufficient.
5. Run pure NumPy and JAX tests covering all four windows, Schmitt hysteresis,
   state continuation across a control boundary, simultaneous transitions,
   stance start/end, p95/tail accounting, terminal censoring, finite values,
   and no-action-change parity.
6. Run a phase-shift/phase-zero **observation-only** diagnostic on the frozen
   V3 actor.  It may establish whether the actor uses its existing phase
   input, but it must not add a feature or alter a candidate.  A phase result
   cannot be used to bypass the contact-quality preflight.

All preflight artifacts must bind source hashes, frozen params SHA,
deterministic seeds, direct-V3 mapper fidelity, exact guard count, and
`hardware_deployment=PROHIBITED`.

## Decision gate after preflight

Only if every required equality and test passes may a **single** fresh pilot
be proposed, with unique output name
`h5_unified_250k_v3_substep_contact_alignment_cleanv22_notarget_v1`.
The proposed pilot may change only the newly measurement-aligned contact and
all-substep normal-force losses, each at the pre-registered scale `-1.0`;
all existing reward scales and every other training setting remain exactly the
clean V3 mapper-only values.  This is explicitly not permission to run the
pilot yet.

If later authorized, the 250k candidate is rejected unless its unchanged
strict actor-only 1×6 s / 2 s-transition-stand gate reports:

- 38/38 strict gait-quality segments, including all 24 moving segments and
  all 12 required locomotion commands;
- all four debounce windows pass their three robustness checks for every
  moving segment;
- all moving segments pass slip RMS, slip p95, and per-stance cumulative slip;
- all segments pass normal-force p99 and steady-force checks; and
- zero fall, nonfinite, fallback, command/target/joint/slew violation, mapper
  error, or guard-count mismatch.

Any failure stops this feed-forward PPO reward line.  It does not authorize a
1M continuation, multi-seed run, 20×30 perturbation campaign, MP4, packaging,
or hardware test.  Those stages remain available only to a candidate that
passes the stated 250k gate.
