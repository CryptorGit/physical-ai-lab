# v59 Seed Derivation and RNG Lineage

## Historical training root

The v59 launch did not pass a `seed` override.  Brax PPO therefore used its
default master seed `0`.

```text
PRNGKey(0)
├─ global_key
│  ├─ policy initialization key
│  └─ value initialization key
└─ local_key
   └─ fold_in(process_id=0)
      ├─ next local optimizer/rollout stream
      ├─ key_env
      └─ eval_key
```

Training used one process and 4096 environments.  Environment reset keys are:

```python
key_envs = jax.random.split(key_env, 4096)
environment_seed = key_envs[environment_index]
```

The domain-randomization wrapper independently calls the same
`jax.random.split(key_env, 4096)`.  Consequently an environment's fixed model
randomization and its initial reset start from numerically identical keys,
although each consumer maintains its own functional split lineage.  This is a
confirmed correlation in the historical implementation.

## Environment reset lineage

For each environment key, `Joystick.reset` performs six sequential parent/child
splits:

1. base x/y offset;
2. base yaw;
3. 14 actuator qpos offsets;
4. six base qvel components;
5. command key, internally split eight ways;
6. push interval.

It then calls `_get_obs`, which performs five further sequential splits:
gyro, accelerometer, gravity, joint position and joint velocity.  The gravity
child key is reused for the IMU delay integer; it is not split again.

## Per-control-step environment lineage

`Joystick.step` first performs one four-way split:

```text
new info.rng, push_theta_key, push_magnitude_key, action_delay_key
```

After physics, `_get_obs` consumes the five observation splits described
above.  Finally `info.rng` is split into the retained next key and a command
candidate key.  `sample_command` splits that child eight ways even during the
first 500 steps, when `jp.where` retains the old command.

Adding or deleting any parent-stream split shifts all following environment
samples.  Adding a draw inside `sample_command` does not shift the retained
parent stream, but changes the candidate's internal values.

## PPO policy sampling lineage

Policy exploration is outside `Joystick`.  Each PPO training step derives
`key_generate_unroll`; nested scans split this key for an unroll and then once
per control step.  The policy uses the resulting child key to draw a
14-element standard normal vector.

The checkpoint stores network and normalizer parameters, but not the PPO
training key, environment state, environment PRNG keys, reset counts, or
episode counts.  Therefore the exact stochastic episode active at step
33,423,360 cannot be reconstructed.

## Audit trace lineage

The stochastic audit uses master seed 0 and the historical derivation above,
selecting environment indices 0, 1 and 2.  It uses those exact keys for domain
randomization and reset.  Fixed C0–C4 commands replace the sampled reset
command after the draw has been consumed.

Because the historical policy rollout key at checkpoint time is absent, the
audit derives a clearly namespaced policy key with
`jax.random.fold_in(environment_seed, 0x59334233)`.  Those policy samples are
fully logged and injected; they are not claimed to be samples from a
historical v59 episode.

Each audit case is duplicated with the same key in the same JAX `vmap` call.
Exact equality of those paired states establishes native same-backend seeded
reproducibility without requiring NumPy or MuJoCo to emit JAX's bit pattern.

## Reset and termination caveats

`BraxAutoResetWrapper` restores the initially cached state on termination
rather than calling `Joystick.reset` and drawing a new key.  Separately,
`num_resets_per_eval=1` causes host-side resets between training epochs using <!-- gitleaks:allow -->
keys derived by splitting the prior `key_envs`.  The checkpoint does not store
which reset generation or episode each environment occupied.  JIT and vmap do
not change functional PRNG ordering, but environment index controls which
pre-split key is received.
