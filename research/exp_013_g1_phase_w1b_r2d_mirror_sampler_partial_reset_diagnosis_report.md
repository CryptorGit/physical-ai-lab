# exp_013 Phase W1B-R2D: mirror sampler partial-reset diagnosis

## Scope and protection

This stage is read-only with respect to policy training. No PPO update, checkpoint
creation, optimizer update, curriculum/reward change, or production promotion was
performed. Existing exp_005–exp_012 and exp_013 Stage 0 through W1B-R1 artifacts
were not modified. The repository started at
`9225fcf77e910c389a41a5784d8d67f7c8899ac5`; no remote push was performed.

## Failure reconstruction

W1B-R1 failed during iteration 15, in Y1, before the iteration-15 optimizer update.
The exception followed this path:

```text
train_w1b.py
→ train_w1a.py rollout
→ ManagerBasedRLEnv.step / _reset_idx
→ CommandManager.reset
→ MirrorPairedW1BCommand._resample_command
```

`w1b_command.py:31-32` tests `len(env_ids) % 2` and raises:

```text
RuntimeError: W1B requires an even environment population for exact mirror pairing
```

The original rollout step, reset count, reset IDs, reset-mask hash, termination
counts, and RNG hash were not recorded. The reset count is nevertheless proven to
have been positive and odd by the executed predicate. The failure happens before
the sampler consumes RNG or writes commands. A structurally exact reproduction
using the same sampler code path and an odd mask produced the same exception at the
same source predicate. It is not an exact replay of the original mask because that
mask is unavailable.

## Existing sampler contract

The sampler splits each reset call into equal base and mirror halves:

```text
base:   (vx,  vy,  yaw, gait)
mirror: (vx, -vy, -yaw, gait)
```

Its present implementation therefore makes three implementation assumptions:

1. every reset call has even cardinality;
2. both pair members are assigned during the same reset call;
3. base and mirror IDs occupy the two equal halves of that call.

Only exact mirror correspondence, matching curriculum group/speed/vx, opposite
vy/yaw, and exactly-once assignment are research requirements. Same-call pairing,
sorted IDs, adjacent pair IDs, and stable environment pairing are not required.
Arbitrary asynchronous partial resets cannot guarantee an even reset cardinality,
so same-call pairing is an undefined boundary rather than a valid environment
invariant.

The current checkpoints save actor, critic, optimizer, iteration, and information
metadata. They do not save sampler RNG, environment RNG, command buffer, pair/event
counters, partial-reset state, or a pending mirror command. The present sampler has
no pending state to serialize.

## Boundary tests

The current sampler was tested for every reset count from 0 through 64, plus
127/128, 255/256, 511/512, 1023/1024. Nine ID patterns covered contiguous,
sorted/unsorted random, alternating, paired/unpaired, mixed-reason, and full-reset
cases. Y1 zero-yaw, moving-yaw, and pure-yaw groups were tested, with sampler-only
coverage of Y2–Y4 as well.

| Boundary class | Cases | Failures |
|---|---:|---:|
| Even reset count | 1,332 | 0 |
| Odd reset count | 1,296 | 1,296 |

The failure is independent of ID ordering, reset pattern, and curriculum phase.
The first failing cardinality is one; zero and every tested even count succeed.

## Required pairing semantics

Exact pairing in the same reset call is neither necessary nor compatible with
independent episode termination. The required guarantee should be:

- exact mirrored counterparts over a bounded rolling window;
- every reset environment receives exactly one command;
- no additional environment is reset;
- counterpart commands retain group, speed, vx, and RNG provenance;
- pending imbalance is bounded to one command and one eligible reset event;
- phase transitions cannot carry an incompatible command.

The recommended unit is therefore a bounded rolling window, while retaining exact
command-pair identity—not merely matching aggregate histograms.

## Candidate comparison

### C0 CURRENT_EVEN_ONLY

Even-path behavior is valid, but the first odd reset deterministically aborts
training. It is not usable with asynchronous resets.

### C1 DROP_OR_DUPLICATE_ONE

It can avoid the exception, but either leaves an environment unassigned or
duplicates a command. On the 100,000-event stream it missed 49,867 assignments.
This violates exactly-once assignment and biases the training population.

### C2 SELF_MIRROR_FILLER

It assigns all environments, but injects `vy=0, yaw=0` on every odd event. In the
simulated stream this produced a 1.5371% filler mass
(`TVD=0.015371`, `KL=0.015491`) and changes curriculum group proportions.

### C3 FORCED_PARTNER_RESET

It preserves command pairs but added 49,867 resets. Those extra resets shorten
episodes, discard valid transitions, corrupt termination labels, and change the
on-policy state distribution. It is incompatible with training-semantics
preservation.

### C4 PENDING_MIRROR_QUEUE

This candidate consumes any pending mirror first. If no command is pending, even
calls delegate to the old sampler unchanged. For an odd call it samples a complete
pair batch, assigns the final base command, and queues its exact mirror for the next
eligible reset slot. The queue is FIFO, has maximum length one, and preserves pair
identity and RNG provenance.

### C5 ROLLING_BALANCE_RESERVOIR

It can balance aggregate signs over a window, but does not necessarily preserve the
exact sampled counterpart. It is more complex and weakens the mirror-pair contract
without a demonstrated benefit.

## Distribution, parity, and determinism

Because W1B-R1 did not record reset masks, the recorded-stream test could not be
performed literally. A deterministic 100,000-call synthetic asynchronous reset
stream was used instead. It contained 49,867 odd calls and 3,244,133 environment
assignments. This limitation does not affect the exact source-level failure proof,
but the stream is not claimed to reproduce the empirical termination distribution.

C4 processed every environment exactly once, used no forced reset, reached a
maximum queue length of one and a maximum/mean queue age of one event, and ended
with zero residual after drain. Including the drained queue, its command
distribution had zero measured TVD/KL/Wasserstein proxy from the intended paired
stream.

On a separate stream containing 100,000 exclusively even reset events, C4 with an
empty queue matched the old sampler bitwise for:

- command tensors;
- pair IDs;
- RNG state after every event;
- curriculum counters.

Both command streams had SHA-256
`c86a74a377ccb6995559ea96bcdff12935997e458ab34578bec127473e74dd64`.

On the odd/mixed prototype stream, C4 produced identical command, queue, RNG, and
pair-metric state on same-process repeat, fresh-process repeat, and serialized
midstream resume. This is a prototype result only; no production sampler change was
made.

At a curriculum boundary, a pending command must be consumed before activating the
new phase. Carrying or reclassifying it would contaminate the new phase; dropping
it would break exact pairing. The transition barrier and both requested/active
phase IDs must be serialized.

## Resume-point audit

The latest available checkpoint is W1B-R1 iteration 10:

```text
path:
results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/
phase_w1b_r1_evaluation_parity_corrected_rerun/checkpoints/model_10.pt

SHA-256:
a5d20b51d3398dd75ecb6832e559684f5eeb5432b830120c980bbcea84842934

Adam step:
4200
```

It restores policy and optimizer state only. There is no iteration-14 checkpoint,
iteration-15 rollout-start state, or failure-prestate. Because sampler/environment
RNG, command buffers, rollout position, and partial-reset sequence were not saved,
iteration 11–14 cannot be reproduced bitwise from iteration 10, and exact
iteration-15 continuation is impossible.

For an exact repaired run, restart from the canonical W1A2 iteration-80 parent and
require bitwise equality with W1B-R1 through the first odd reset. This is a restart
of the original prefix, not a continuation from iteration 10. No restart was
executed in this stage.

## Classification

```text
MIRROR_SAMPLER_ASYNC_RESET_CONTRACT_MISSING
```

The direct manifestation is the even-count assertion, but the primary defect is
that same-call pairing has no defined semantics for asynchronous odd reset calls.
It is therefore broader than an isolated assertion typo.

## Recommended repair

```text
DETERMINISTIC_PENDING_MIRROR_QUEUE
```

The next stage should implement only this repair, serialize queue/RNG/pair/event
and phase-transition state, re-run the even-path bitwise and odd-path
serialize/resume gates, and then rerun W1B exactly once from canonical W1A2
iteration 80. It must not force partner resets or change reward, curriculum,
network, evaluation gates, or PPO configuration.

## Current artifact interpretation

- Evaluation parity repair: PASS.
- Training tensor parity: PASS.
- W1B numerical stability: PASS through iteration 14.
- Iteration-10 zero-yaw retention: 16/16 PASS.
- Iteration-10 moving-turn matrix: 18/24 PASS.
- Iteration-10 fall rate: 0%.
- Runtime failure: mirror sampler odd partial-reset boundary.
- Policy failure: not established.
- Canonical promotion: none.
- Canonical parent remains W1A2 iteration 80,
  SHA-256 `bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244`.
