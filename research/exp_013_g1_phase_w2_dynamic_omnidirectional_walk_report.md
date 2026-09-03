# Exp 013 Phase W2 — Dynamic omnidirectional WALK transitions

## Outcome

The sole authorized W2 persistent run stopped at iteration 5 under the preregistered
early guard. The primary classification is `EXP013_W2_TRAINING_UNSTABLE`. The stopping metric
was start/stop quick success 68.4375%, below the required 70%. No retry, resume, seed
change, or second persistent run was performed.

## Parent

- W1B-R2 iteration 200, SHA-256 `61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`
- `MonotonicPositiveYawCalibrationV1`
- shared endpoint/acquisition evaluators retained
- actor, critic, optimizer, normalizer and sampler state restored bitwise
- artifact Adam step: 8000; fixed LR: 1.5e-5
- WALK exploration alpha 0.30; log-std branches frozen

## Command pipeline and sampler

Physical vx/vy/yaw remained the reward and evaluation targets. Actor observations
used physical vx/vy and calibrated yaw (positive ×1.50; zero/negative identity).
The complete transition sequence is mirrored by `(vx, vy, yaw) -> (vx, -vy, -yaw)`.
The pending sequence queue is FIFO, bounded to one, serialized with timers and RNG.
Boundary/property tests passed, mixed odd/even determinism passed, and 100,000
single-segment reset events matched the W1B-R2 steady path bitwise.

## Training

T1 began as specified, but the run reached only 5/250 iterations (122,880
interactions). The one-update preflight passed: exact KL 0.011691, all-step maximum
KL 0.011691, clip fraction 0.178345, mean-action shift 0.027237, NaN/Inf 0.
Iterations 1–4 passed the corrected clean guard. Iteration 5 retained zero-yaw
16/16, forward 0.6/1.2 at 100%, static moving turns 24/24 and fall 0%, but start/stop
quick success declined to 68.4375%.

## Formal evaluation

Formal selection and all W2 transition matrices were not run because the mandatory
early guard made the run ineligible. This is recorded as `NOT_RUN`, not inferred.
The initial checkpoint is retained only as the best reference candidate; it is not
a W2 promotion.

## Safety and artifact interpretation

At the stopping guard: fall 0%, dangerous slip 0.1724%, impact 0%. The W1B-C2
yaw-conditioned endpoint artifact remains canonical. No dynamic WALK canonical
artifact was promoted. RUN was not trained and is outside this gate.

## Protection

Existing stages, checkpoints, optimizers, reward weights, network, physics,
Isaac Lab and installed RSL-RL were not modified. Only W2-specific code and
artifacts were produced. Remote push was not performed.

## Next

Perform one diagnostic only: iteration-3-to-5 practical-stop retention regression
boundary diagnosis. Starts remained 100%, while direction-wise stop success fell
to 30–50%; isolate that regression before any further training is authorized.
