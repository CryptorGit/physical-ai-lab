# Instrumented Training Harness v1 Audit

## 1. Executive Summary

Decision: **HARNESS_FAIL**.

The harness can serialize and reload the complete update-boundary learner,
environment, RNG, controller, and randomized-model state. It also exposes exact
command/PPO exposure and optimizer diagnostics without control-step host
callbacks. However, the mandatory identity gates fail:

- 2+2 resumed updates differ from 4 uninterrupted updates.
- Uninstrumented and instrumented 5-update outputs differ.
- More fundamentally, two invocations of the same compiled update function in
  one process, from the same in-memory state, already differ.

The first observed divergence in that focused probe is after the first batched
MJX physics step. Command, initial observation, first sampled action, and RNG
were bit-exact; the following observation differed by max 0.007628441, reward
by 0.00006297, and the next action by 0.000210762. Thus checkpoint I/O is not
the first cause, and instrumentation cannot be causally cleared while the
native baseline is itself non-repeatable.

No reward experiment or null continuation is authorized.

## 2. Scope and Non-Claims

Only the old unbounded yaw objective was used. Tests used four environments,
two rollout steps, one PPO epoch, four minibatches, and at most five harness
updates (40 interactions). They are wiring tests, not performance training.
No reward, teacher, sampler, curriculum, network, PPO hyperparameter, scene, or
domain-randomization change is claimed. No policy is promoted.

## 3. Git and Source Provenance

- Main source freeze: `8ae5105caf1af90eb200ffef7a96e985d655614b`
- Historical WSL training source freeze:
  `338451f33e687ea3edcda8a2c2cdcbc8a7b4bda0`
- Initial harness implementation:
  `1c2b26d6f32091e81146da125bc5ffc94b40ba4d`
- Parent checkpoint tree:
  `81158e0444a90012c07755a055304ba6e7203fb59a71807c4d6d4129ac07af1d`
- Scene SHA-256:
  `9819c81c0eecbe751662a24029976bdb765da28e0a8f5a9854470a0611fee73f`
- Old reward SHA-256:
  `8f128dddf2591e2e68e194bfa6edcc26b0110a3bc0574ba7e311513ddf5f5cd7`

The baseline archive hash and external checkpoint hashes are recorded in
`baseline_commit.json` and `external_artifact_manifest.json`.

## 4. Brax PPO Path and State Ownership

The installed Brax 0.14.2 learner state contains actor, critic, Optax state,
normalizer, and environment-step count. Environment state and loop RNG exist
inside the training loop but are not exposed by the existing callback. The
standard Brax checkpoint stores normalizer/policy/value, not the full live
learner and environment continuation state.

The harness therefore minimally rehosts the installed ordering:

environment/reset/randomization → policy rollout → running-normalizer update →
GAE/loss → seeded permutation → all minibatches → update-boundary state.

See `training_pipeline.md` and `state_capability_matrix.csv`.

## 5. Checkpoint Contract

Each `state.pkl` stores:

- actor, critic, normalizer, complete Optax Adam state;
- learner, rollout, and evaluation keys;
- explicit interaction, Adam-step, harness-update, reset, episode, and
  per-environment episode-step counters;
- complete batched environment/MJX Data state and `info` dictionary, including
  per-env RNG, commands/head fields, action history/delay buffer, teacher phase,
  and controller state;
- environment keys and exact batched domain-randomized MJX model.

The final small-run payload is 18,164,049 bytes; raw numeric leaves are
6,261,240 bytes. Integrity is checked by payload, state-tree, and model-tree
SHA-256. `checkpoint_schema.json` is the machine-readable contract.

## 6. Resume Identity

Uninterrupted 4 updates and 2+checkpoint+new-process+2 updates both reached:

- 32 environment interactions;
- 16 Adam minibatch updates;
- identical randomized-model tree.

They were not bit-exact. The first flattened differing leaf was Adam first
moment `mu/policy/params/hidden_0/bias`; max absolute error
0.00023505363787990063 (1,622 differing bytes/elements as reported by the leaf
comparator).

A separate fresh-process 2-vs-2 repeat also differed at the same Adam leaf
(max 0.000008559582056477666), proving the failure is not specific to
checkpoint serialization.

## 7. First Observable Divergence

A same-process probe reused one compiled update function and the exact same
in-memory input state twice:

| Point | Result |
| --- | --- |
| resolved command | bit-exact |
| reset/initial observation | bit-exact |
| first sampled actor action | bit-exact |
| RNG/key state | bit-exact |
| observation after first batched MJX step | max error 0.007628441 |
| first-step reward | max error 0.00006297 |
| second action | max error 0.000210762 |

This probe identifies the first *controller-visible* divergence, not the first
internal MJX leaf. The exact kinematics/contact/solver leaf was not instrumented
in this phase and is not guessed. Prior fixed-input single-state GPU tests
remain valid for their selected states; this new failure covers the batched,
domain-randomized training reset state.

## 8. Behavior-Preserving Instrumentation

Five uninstrumented and five instrumented updates did not end bit-exact. The
first compared leaf was again Adam `mu/policy/hidden_0/bias`, max
0.0004627953239833005.

Because the uninstrumented native update is already non-repeatable from
identical input, this A/B result cannot causally attribute the difference to
telemetry. Nevertheless, the required identity proof is absent, so the
instrumentation gate fails. The telemetry code itself consumes no RNG and
stays outside the loss/update dataflow, but structural purity is not a
substitute for the required empirical identity.

## 9. Command and Effective PPO Exposure

Device-side aggregation records C00–C18 plus an explicit OFF_GRID class. The
continuous training sampler usually produces off-grid values; they are not
falsely assigned to the nearest formal command. Metrics include rollout,
survival, termination/fall, command starts, valid-advantage samples,
minibatch-input counts, flattened `P(vx,vy,yaw,head)`, reward-term sums, actual
velocity sums, advantage sums/squares, and pre-update surrogate contribution.

The example is in `command_exposure_example.csv`. This proves tensor access and
exact counting for the executed rollout, not that historical v60 exposure can
be recovered.

## 10. Optimizer and PPO Diagnostics

Per update the harness records policy/value/entropy loss, KL, explained
variance, actor/critic/global gradient norms, update norm, parameter norms,
Adam first/second moment norms, effective Adam step scale, learning rate, and
nonfinite counts. See `optimizer_diagnostics_example.csv`.

Post-update command attribution is not claimed: the per-command policy
contribution is explicitly the pre-update `rho=1` surrogate. This limitation is
recorded in `metric_schema.json`.

## 11. Intermediate Checkpoint Schedule

The threshold scheduler unit test passes for 0/50k/100k/250k/500k/1M and saves
the actual interaction count when an update crosses a threshold. Zero-step and
nonzero update checkpoints round-trip and verify hashes. Actual 250k/500k/1M
training was prohibited and not run.

## 12. Host Boundary Stability

Final-code instrumented runs A/B/C each completed five updates without a WSL
`libcuda.so` crash:

- crash count: 0/3;
- host transfers: 5 per run, 37,640 bytes total per run;
- checkpoint save time: 0.0199–0.0228 seconds;
- no control-step callback.

This is limited evidence; three successes do not establish the root cause of
the earlier v60 crashes. The first compilation observed an XLA autotuner result
mismatch warning. The warning is preserved as an observation and is not
asserted to cause the numerical divergence.

## 13. v60 Postmortem

Arm C itself degraded, so fresh-Adam continuation is not established as a
neutral control. Actor L2 delta from parent was 4.9436 (C) and 5.1744 (T);
critic delta was 12.6140 and 11.8249. Arm T’s left/right response was
0.5037x/1.0771x, but exact training command, advantage, optimizer, and RNG
telemetry did not exist.

The left/right mechanism is therefore
`UNRESOLVED_DUE_TO_MISSING_TRAINING_TELEMETRY`. See `v60_postmortem.md`.

## 14. Gate Decision

| Gate | Result |
| --- | --- |
| source/config/tests tracked | PASS |
| complete state serializable | PASS |
| 2+2 equals 4 | **FAIL** |
| instrumentation identity | **FAIL / not isolatable** |
| exact command exposure | PASS |
| effective PPO counts by command | PASS |
| optimizer state/update norms | PASS |
| arbitrary threshold scheduler | PASS |
| RNG/env/domain state saved | PASS |
| three small runs without crash | PASS |

Overall: **HARNESS_FAIL**.

## 15. Next Allowed Work

The next allowed work is a bounded diagnostic of batched, domain-randomized MJX
same-input one-step repeatability to locate the first internal divergent leaf
and discrete contact/constraint change. It must not alter production XLA/MJX
settings initially.

The 250k old-objective null-continuation experiment is designed in
`next_null_continuation_plan.md` but must not run until exact resume and
instrumentation identity pass.
