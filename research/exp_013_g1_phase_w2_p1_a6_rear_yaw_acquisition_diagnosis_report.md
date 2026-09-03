# Exp 013 Phase W2-P1-A6 rear-yaw acquisition diagnosis

## Outcome

Main classification: `W1B_REAR_START_CAPABILITY_PARTIAL`.

The committed A5 PC2 result remains the formal reference: endpoint 99.729%, acquisition 92.646%, fall 0.271%, slip 0.042%. Rear 180°/-0.3 and 180°/+0.3 both reached 100% endpoint with 0% fall, but acquisition was only 22.5% and 18.0%.

The A4 candidate reproduced twice in-process and once in a fresh process with tensor hash `db65a3069d665b8012fd9d264b7fd54e629a22d25b05a9ff793e23bfc549ac5f`. The independent A6 PC2 trace replay did not meet the requested zero-delta parity (aggregate acquisition drift +0.002708); therefore A5 committed metrics are preserved as authoritative and replay metrics are identified separately.

## Component diagnosis

Translation vector, direction, and gait conditions pass early. Rear acquisition resets are governed by yaw sustained-pass oscillation. Low-speed direction angle is noisy but ceases to be limiting before the rear acquisition deadline.

Factorial controls isolate the interaction: rear 0.3/yaw 0 passes at about 98.5%; forward 0.3/±yaw passes at about 99.5%; pure yaw passes at 81–99%; rear 0.3±yaw falls to roughly 16–29%. Reducing rear speed to 0.15 materially improves the result.

## Teacher horizon and timing

No 2/4/6/8/12/16-step horizon, FULL_RAMP, FULL_START, or W1B_ONLY profile achieves rear±yaw authorization. W1B_ONLY itself remains around 24.0% at 3 seconds and 37.8% at 6 seconds. This refutes a four-step takeover-only explanation.

Extending the PC2 deadline improves acquisition gradually, but only about half of rear trials pass by 8 seconds. The failure is oscillatory rather than a simple late-but-stable crossing. Ramp durations from 0.75 to 3.0 seconds do not resolve it.

## Contract note

The formal evaluator combines translation vector MAE, direction, yaw magnitude/sign, caller-provided gait and safety, with a 0.20 s (10-step) sustained window. A5's stage-local positive-control helper used vector/yaw/safety rather than all formal evaluator components. No thresholds or physical gates were changed here.

## Protection

No dataset, label, split, manifest, overlay, checkpoint, optimizer, network, reward, sampler, physics, calibration, or evaluator contract was changed. No student training, V3 overlay, formal closed-loop authorization, DAgger, PPO, or promotion was performed.
