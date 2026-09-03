# Statistical Resume and Null Continuation

This directory records a pre-registered statistical replacement for
bit-exact batched-GPU resume. Null continuation is conditional on
`STATISTICAL_RESUME_PASS`; it must not start when that gate fails.

No reward, teacher, sampler, curriculum, network, PPO, scene, domain
randomization, MJX scatter/segment operation, or deterministic XLA setting is
changed by this protocol.

Results:

```text
checkpoint payload round-trip: CHECKPOINT_PAYLOAD_BIT_EXACT_PASS
statistical resume:            STATISTICAL_RESUME_FAIL
null continuation:             NOT_RUN
```

Start with `final_report.md`, then use `resume_equivalence_report.md` and the
all-run CSV files for numerical evidence.
