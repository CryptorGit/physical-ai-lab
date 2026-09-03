# exp_013 Phase W1B-R1 evaluation-parity-corrected rerun report

## Outcome

Evaluator parity and iteration-1 training tensor parity both passed. The sole persistent rerun then stopped after 14 completed updates when the unchanged W1B mirror-paired command sampler received an odd-cardinality partial-reset env-id set. No retry or curriculum/reward change was made.

## Evaluation parity

`Exp013DirectionalCapabilityEvaluator` reuses the protected fresh DirectionalBaseline evaluator as the only metric and success implementation. P1/P2/P3 were isolated processes with deterministic mean actions, corruption/push/external force disabled, block allocation, and seed 20274021. Parent and old iteration 1 were 16/16 on both quick and 50-episode formal checks; maximum metric difference was 0.

## Parent and training

W1A2 iteration 80 `bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244` restored actor, critic, optimizer, Identity normalizer, and Adam step 4000. Exploration remained alpha_walk 0.30 with frozen WALK/RUN std; reward and original Y1–Y4 curriculum were unchanged. The first-update preflight passed. Official iteration 1 actor/critic/optimizer tensors were bitwise identical to old W1B iteration 1. The clean early guard passed iterations 1–10; fall remained 0 in the noisy training rollouts.

The run completed 14/200 iterations (344,064 interactions) in Y1. It stopped during iteration 15 because exact mirror pairing rejected an odd partial-reset population. Only W1B-R1 initial, iteration 1, and iteration 10 checkpoints persist.

## Read-only diagnostic at selected available checkpoint

The diagnostic selection is iteration 10, SHA `a5d20b51d3398dd75ecb6832e559684f5eeb5432b830120c980bbcea84842934`. Zero-yaw 0.3 m/s passed 16/16; forward 0.6/1.2 success was 100.0%/100.0%. Pure yaw -0.3/+0.3 success was 100.0%/0.0%, with MAE 0.097/0.222 rad/s. Moving-turn core passed 18/24 and independence 7/10. These are diagnostic only because 200 iterations were not completed.

Formal diagnostic safety: fall 0.00%, excessive tilt 0.00%, dangerous slip 0.14%, impact 0.00%, long-dwell saturation 0.00%. Symmetry pass: False.

## Classification and artifact

Classification: `EXP013_W1B_R1_TRAINING_UNSTABLE`.

W1B-R1 is not promoted. Canonical WALK remains W1A2 iteration 80. W1B-R1 is not a final integrated policy. The single next action is **mirror-paired W1B command sampler partial-reset boundary diagnosis**.
