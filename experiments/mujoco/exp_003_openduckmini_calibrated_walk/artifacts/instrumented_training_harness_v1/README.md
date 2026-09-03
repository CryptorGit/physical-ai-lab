# instrumented_training_harness_v1

This directory records the pre-experiment training observability and
update-boundary resume audit. It is not a policy release and contains no new
reward experiment.

Decision: **HARNESS_FAIL**. The payload contract is complete and loadable, but
the required bit-exact gates are not met. A same-process repeated-input probe
showed identical command, reset observation, first action, and RNG, followed by
a difference after the first batched MJX physics step. Consequently neither the
2+2/4 resume identity nor instrumentation identity can be proven under the
native training backend.

Large `state.pkl` files remain outside Git in:

`/home/user/openduck_training_backward_v23_20260729/experiments/mujoco/exp_003_openduckmini_calibrated_walk/artifacts/instrumented_training_harness_v1/runs/`

Their hashes are in `external_artifact_manifest.json`; small copied manifests
and telemetry are under `run_metadata/`.

No continuation or reward experiment is authorized until the native batched
MJX rollout repeatability issue is isolated and the exact-resume and
instrumentation-identity gates pass.
