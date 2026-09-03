# EXP013 Phase W1B-D2 yaw-rate tracking boundary diagnosis

## Outcome

Primary classification: `POSITIVE_YAW_GLOBAL_GAIN_BIAS`.

The selected W1B-R2 actor is monotonic on both yaw signs. At the native +0.30 input its positive
response is too small, but a diagnostic global gain of 1.50 passes
9/9 target-0.30 pure/moving conditions
with maximum fall 0.0%. Direction-specific oracle inputs span
0.375 to
0.525 rad/s.

## Timeline and response boundary

Positive yaw did not regress after an early peak. It remained near zero through the first half of
training and improved late: pure +0.30 reached approximately +0.135 at iteration 160 and +0.160 at
iteration 200. Pure -0.30 was already supported and ends near -0.300. The same late improvement is
direction dependent in magnitude: native +0.30 remains weakest for 90, 135, and 180 degree
translation. Iteration 200 is the best existing checkpoint under the required zero-yaw, forward,
and safety retention constraints; no selection change is made in this diagnostic.

The selected actor has a monotonic positive response, with fitted gain depending somewhat on
translation state but no hard saturation near the target. At 0.3 m/s the diagnostic positive
command required for +0.30 ranges from roughly +0.38 to +0.59 rad/s. Commands around +0.45 to
+0.53 reach the target across the tested directions, while +0.7 tends to overshoot. The native
+0.30 failure is therefore an input-to-response gain/offset boundary, not absence of a physically
reachable right-turn gait.

## Translation unlock

At the native +0.30 input, adding a small translation unlocks only some directions; it does not
provide a universal turn-in-place solution. In contrast, a +0.50 actor input succeeds for pure yaw
without translation. This rejects a turn-in-place-only dynamical barrier as the primary cause.

## Exposure, reward, advantage, and gradients

The serialized training artifacts establish mirror-balanced command sampling, but per-bin rollout
return, advantage, and minibatch inclusion telemetry were not retained for all 200 iterations;
unavailable fields are explicitly `not_recorded` in the exposure audit. Fresh on-policy diagnosis
shows comparable yaw and translation reward on both signs. Pure positive/negative 24-step
advantages are 0.0352 / 0.0531, and neither
positive-advantage rate nor critic bias indicates positive-only suppression.

The pure negative/positive total-gradient cosine is -0.1575. Several mirrored
direction pairs also show sign-dependent gradient opposition, but this is secondary evidence:
the frozen policy reaches the requested positive rate safely through command magnitude alone,
without any parameter update.

## Counterfactual controls

The full mirrored-policy wrapper is not a valid positive control: maximum fall is
100.0%. Short 1-8-step mirrored-action
interpolation reaches at most 3.0% success and does not establish a retained alternative
basin. These results reject mirrored runtime control and do not indicate local action-manifold
reachability.

## Action, contact, and state analysis

Mean-action mirror asymmetry remains measurable, and mirrored positive/negative state populations
are partially separable (linear AUROC about 0.84-0.91 in the failing groups). Contact timing and
support fractions differ after mirroring, but no joint, action-saturation, torque, or contact-force
limit blocks the calibrated positive response. The direct mirrored wrapper falls in every tested
condition, so it is not a valid runtime remedy or evidence that a simple mirrored action basin can
be entered locally.

## Artifact status

This stage creates no checkpoint and performs no optimizer step. W1B-R2 remains a diagnostic
yaw-capable WALK artifact; W1A2 iteration 80 remains the canonical translation-only WALK parent.
No command calibration or mirrored wrapper is adopted.
