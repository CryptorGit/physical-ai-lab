# Backend nondeterminism contract

Bit-exact equality remains mandatory for serialized checkpoint bytes, actor,
critic, normalizer, optimizer, all RNG state, environment/controller state,
domain-randomized model, unbatched controller calculations, and fixed-input
unbatched MJX.

Bit-exact equality is not required for GPU MJX batches of size two or larger,
batched trajectories, their gradients, or post-update parameters. These are
evaluated using pre-registered U/R distributions, all-run disclosure,
bootstrap confidence intervals, standardized mode effects, failure rates, and
joint command-exposure distance.

The exception is limited to the established GPU batch path:

```text
fwd_position -> smooth.crb -> reverse body_tree
             -> jax.ops.segment_sum -> scatter-add
```

It does not relax checkpoint payload integrity or permit dropped trials.
