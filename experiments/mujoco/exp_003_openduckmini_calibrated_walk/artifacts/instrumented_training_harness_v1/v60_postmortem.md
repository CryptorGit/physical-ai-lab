# v60 Postmortem

## Supported

- Arm C and Arm T used the same parent/fresh-Adam initialization hash.
- Parameter movement was substantial. Actor L2 delta from parent was 4.9436
  (C) and 5.1744 (T); C-to-T actor distance was 6.6083. Critic L2 delta was
  12.6140 (C), 11.8249 (T), and C-to-T 11.4740.
- The largest actor movement was hidden_1 kernel: 3.9267 (C) and 4.1145 (T).
- Arm C itself was not a neutral retention control: deterministic yaw-only
  falls were 3/10, stochastic yaw-only falls 7/10, and forward retention failed
  the >10% degradation gate for C01 and C09.
- Arm T changed the deterministic yaw response asymmetrically: left 0.5037x,
  right 1.0771x, absolute difference 0.5734. Mean yaw MAE improved only 13.2%.

## Partially supported

The objective change affected the learned parameter tree and yaw behavior, but
the missing optimizer/RNG/rollout telemetry prevents attribution of individual
updates or exposure imbalance. Evaluation outcomes establish the asymmetry;
they do not establish its training-time mechanism.

## Unresolved due to missing training telemetry

- Exact `P(vx, vy, yaw, head)` exposure and effective PPO sample counts.
- Positive/negative-yaw advantage, loss, and reward contribution by update.
- Fresh Adam first-update magnitude and optimizer moment history.
- Head-command-conditioned training asymmetry.
- The update at which Arm C or T first lost retention.

Final classification for the left/right asymmetry cause:
`UNRESOLVED_DUE_TO_MISSING_TRAINING_TELEMETRY`.
The existence of Arm C degradation is `SUPPORTED`; reward-only causality beyond
the matched final outcome is `PARTIALLY_SUPPORTED`.
