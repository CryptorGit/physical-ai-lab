# v59 MJX first-step divergence diagnostic

This directory contains a read-only, production-independent one-step MJX
diagnostic for v59 step 33,423,360. It starts from immutable serialized
episode-start inputs and compares:

- GPU repeats in one process;
- GPU runs in two fresh processes;
- CPU against GPU.

The result is `BIT_EXACT_PASS` for GPU repeatability in the three reconstructible
states. CPU and GPU differ numerically, but those backend differences are kept
separate from GPU repeatability. See `first_step_report.md` for the bounded
claim and `serialized_input_hashes.json` for input identity.

`D1b` was not synthesized: the prior trace did not save the complete MJX Data
tree at the last normal step before termination.

