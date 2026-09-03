# Training Pipeline Audit

| Stage | Implementation | JIT/device boundary | RNG | State/checkpoint |
| --- | --- | --- | --- | --- |
| entrypoint | `playground/common/runner.py`, `ppo.train` | host builds loop | master seed | standard callback sees only params |
| environment | `joystick.Joystick`, `wrapper.wrap_for_brax_training` | reset/step JIT | environment key | full state was not standard-saved |
| domain randomization | `playground/common/randomize.py:domain_randomize` | model batched on device | reset-derived key | harness saves exact randomized model |
| reset/sampler | `joystick.py:reset`, command sampling helpers | device | per-env `info.rng` | embedded in full env state |
| actor/rollout | Brax PPO policy and scan | device | rollout/action keys | transitions and sidecar stay device-side |
| GAE | `brax.training.agents.ppo.losses.compute_gae` | device | none | aggregate only |
| minibatches | Brax PPO permutation/reshape | device | learner split keys | RNG and update boundary saved |
| PPO update | `compute_ppo_loss`, Optax Adam | device | loss/entropy keys | actor, critic, Adam, normalizer saved |
| checkpoint | `training/checkpointing.py` | explicit block, then host | none | complete update-boundary PyTree + model |
| telemetry | `training/device_metrics.py` | reduced on device, one transfer/update | none | small JSON payload |

The installed Brax 0.14.2 `TrainingState` contains optimizer state, network
parameters, normalizer state, and environment-step count. The stock high-level
API keeps environment state and loop RNG as local variables and its normal
checkpoint path saves only normalizer/policy/value parameters. “Not visible to
the callback” therefore did not mean “not present internally.” The harness
minimally rehosts the installed loop ordering to return these values; it does
not change PPO equations or minibatch order.

One harness update means one rollout collection followed by every configured
PPO epoch/minibatch update. Resume is supported only at that boundary.
