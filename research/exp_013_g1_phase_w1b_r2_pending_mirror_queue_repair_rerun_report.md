# exp_013 Phase W1B-R2 pending-mirror queue repair and rerun

## Outcome

The deterministic pending-mirror FIFO queue passed even/odd/reset/serialization/distribution gates. The sole persistent W1B rerun completed 200 iterations (4,915,200 interactions). Selected checkpoint: iteration 200, SHA `61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`.

## Sampler repair

Even reset calls preserve the legacy base/mirror assignment and RNG stream bitwise. Odd calls assign K+1 base commands and K mirrors, carrying the final exact mirror for the next positive reset event. Queue length and age are bounded at one. Phase changes consume old-phase pending commands before generating new-phase commands. Final base/mirror counts were 116,743/116,743; queue and residual were zero. All 13 checkpoints passed fresh-process sampler restoration.

## Formal evaluation

Zero-yaw 0.3 m/s passed 16/16; forward 0.6/1.2 success was 100.0%/98.0%. Pure yaw -0.3/+0.3 success was 100.0%/2.0%, with MAE 0.092/0.162 rad/s. Moving-turn core passed 21/24 and translation/yaw independence 8/10.

Safety across formal episodes: fall 0.00%, excessive tilt 0.14%, dangerous slip 1.14%, impact 0.00%, long-dwell saturation 0.00%. Symmetry pass: False.

## Classification

`EXP013_W1B_R2_YAW_RATE_PARTIAL`

Canonical promotion: False. Next action: **yaw-rate tracking boundary diagnosis**. W1B-R2 remains a WALK specialist and is not the final integrated WALK/RUN policy.
