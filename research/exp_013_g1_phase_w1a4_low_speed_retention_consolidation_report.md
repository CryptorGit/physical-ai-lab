# exp_013 Phase W1A4 low-speed-retention consolidation preflight

W1A2 iteration 80, SHA `bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244`, was restored bitwise with its critic,
optimizer (Adam step 4000), Identity normalizer, and fixed 1.5e-5 learning rate.
The WALK exploration contract remained alpha 0.30 with WALK/RUN log-std frozen.

The low-speed reference archive contains 1,920 deterministic episodes across 16
directions and 0.25/0.30/0.35 m/s (12,288 train and 3,072 holdout observations).
The fresh iteration-80 0.6 m/s audit fixed 11 failed sectors before the preflight.

The prescribed four beta branches each completed exactly five shadow updates:

- beta 0.00: holdout KL 0.02536, 0.3 m/s 4/16, 225° 60%, 247.5° 15%
- beta 0.01: holdout KL 0.01156, 0.3 m/s 5/16, 225° 40%, 247.5° 25%
- beta 0.03: holdout KL 0.00532, 0.3 m/s 5/16, 225° 30%, 247.5° 10%
- beta 0.10: holdout KL 0.00185, 0.3 m/s 5/16, 225° 45%, 247.5° 25%

All branches made nonzero PPO updates and remained numerically stable, but none
retained the required 0.3 m/s 16/16 quick gate or the 225°/247.5° ≥90% gates.
Therefore no beta was selected. In accordance with the hard stop contract, the
60-iteration persistent PPO run, early guard, checkpoint generation, capability
timeline, formal W1A4 matrix, continuous diagnostic, and RUN diagnostic were not
executed.

Formal classification: `EXP013_W1A4_RETENTION_COEFFICIENT_NOT_FOUND`. New persistent checkpoint count is
zero and no production policy was updated. W1-series speed expansion is closed.
The canonical WALK parent is frozen as W1A2 iteration 80, SHA `bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244`,
with 0.3 m/s 16/16, 0.6 m/s 5/16, forward 0.6/1.2 at 100%, fall 0%, and
dangerous slip 0.55%. The only next action is **Phase W1B:
yaw-conditioned omnidirectional WALK**.
