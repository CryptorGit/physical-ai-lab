# exp_013 Phase W1B yaw-conditioned omnidirectional WALK report

Canonical W1A2 iteration 80 (`bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244`) strict resume passed with actor/critic/optimizer bitwise restoration, Adam step 4000, Identity normalizer, fixed LR 1.5e-5, alpha_walk 0.30, and frozen WALK/RUN std.

Reward audit passed: body-frame vx/vy tracking weight 2.0 and body yaw-rate tracking weight 1.0 are unchanged and sign-symmetric. Parent boundary showed pure yaw -0.3 success 100% (MAE 0.109) versus +0.3 success 0% (MAE 0.218), with zero falls.

The one-update preflight passed (exact KL 0.00957, clip 0.142). The single authorized persistent run then stopped at iteration 1 because zero-yaw 0.3 m/s quick PASS was 11/16, below the hard minimum 12/16. Fall 0.00%, slip 0.20%, impact 0.00%, forward 0.6/1.2 100%/100%, and yaw-sign correctness all remained within guard limits.

Fresh read-only capability evaluation found both initial and iteration-1 checkpoints at 16/16 zero-yaw conditions, but the online guard result is authoritative and no second run or additional yaw curriculum is allowed. Formal W1B, path, random, and RUN suites were not executed after the stop.

Classification: `EXP013_W1B_TRAINING_UNSTABLE`. W1B is not promoted. Canonical translation-only WALK remains W1A2 iteration 80. The single next action is **yaw/translation interference diagnosis**.
