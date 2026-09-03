# RNG Lineage

`PRNGKey(seed)` is split into global and local keys. Global is split for the
policy and value initialization. Local is split into learner, environment reset,
and evaluation keys. The environment key is split by environment index.

At every harness update the learner key produces the epoch key and next learner
key. The epoch key produces SGD, rollout, and unused parity-compatible branches.
Rollout collection splits per scan batch and per control step; each step splits
the policy-sampling key. SGD splits by epoch, permutation, gradient pass, and
loss/entropy evaluation in the same structural order as Brax 0.14.2.

Per-environment stochastic state is carried in `environment_state.info["rng"]`.
The checkpoint also saves `environment_keys`, reset generation, episode
indices/steps, the complete environment state, and the exact randomized MJX
model. Telemetry performs no random split and consumes no key.

Historical v60 keys cannot be reconstructed because those runs did not
serialize learner/rollout keys, environment state, reset generation, or
randomized model assignment.
