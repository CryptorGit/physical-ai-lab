# Donation / Aliasing Audit

- Every measured call reloaded `state.pkl` from disk, copied host leaves, performed `device_put`, executed one step, called `block_until_ready()`, then copied results to host.
- The canonical input hash was identical before and after every non-donated call.
- Non-donated fixed-target batch-4 runs diverged in 19/20 outputs.
- A diagnostic donated-data variant also diverged in 19/20 outputs. Its input was disposable and was never reused.
- Unbatched environment 0–3 each reproduced bit-exactly in 20/20 runs.
- The same device object was not reused after donation for any result used in classification.

Decision: `NO_INPUT_ALIASING_FOUND`. Donation is neither necessary nor sufficient for the observed divergence.
