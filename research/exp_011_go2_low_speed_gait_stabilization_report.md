# exp_011 Go2 low-speed gait-stabilization report

## Training

- Parent: Stage 4 iteration 50, `e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea`.
- Strictly resumed actor, critic, std, normalizer, Adam moments, step 21,000, LR 0.00026012294873748923, and scheduler state.
- Only the command distribution changed: ZERO 15%, LOW STEADY 35%, LOW TRANSITION 30%, ANCHOR 20%. Reward semantic difference: 0.
- 200 iterations, 9,830,400 interactions; first exact KL 0.01293, clip fraction 0.18264; stability PASS.
- Selected checkpoint: iteration 50, `d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd`.

## Zero retention

- Hold 100%, fall 0%, speed mean 0.0026 m/s.
- Heading p95 0.059 rad; tilt p95 0.071 rad.
- Contact-point slip remains diagnostic (46%); non-slip STAND retention PASS.

## Low-speed steady state

| speed | parent fall | Stage 7 fall | heading p95 | MAE | gait |
|---:|---:|---:|---:|---:|---|
| 0.2 | 24% | 2% | 0.212 | 0.034 | CRAWL_LIKE:47, FALL:1, IRREGULAR:2 |
| 0.3 | 24% | 0% | 0.165 | 0.022 | CRAWL_LIKE:49, IRREGULAR:1 |
| 0.4 | 6% | 2% | 0.119 | 0.030 | CRAWL_LIKE:49, FALL:1 |
| 0.5 | 0% | 0% | 0.148 | 0.026 | CRAWL_LIKE:50 |
| 0.6 | 2% | 0% | 0.175 | 0.027 | CRAWL_LIKE:50 |

The fall band contracted substantially; heading remains outside 0.12 rad at 0.2, 0.3, 0.5, and 0.6 m/s.

## Low-speed transitions

| direction | completion | acquisition | hold | fall | heading p95 | gate |
|---|---:|---:|---:|---:|---:|---|
| 0.6→0 | 100% | 100% | 100% | 0% | 0.117 | PASS |
| 0.6→0.2 | 100% | 100% | 100% | 0% | 0.224 | FAIL |
| 0.6→0.4 | 100% | 100% | 100% | 0% | 0.140 | FAIL |
| 0→0.2 | 100% | 100% | 100% | 0% | 0.185 | FAIL |
| 0→0.4 | 100% | 100% | 100% | 0% | 0.123 | FAIL |
| 0→0.6 | 100% | 100% | 100% | 0% | 0.166 | FAIL |

## Anchor retention and migration

- 1.2/2.0 steady and all four primary transition retention checks: PASS.
- Anchor sequence completion 100%, fall 0%, checkpoint switches 0.
- No new fall band appeared at 0.5–0.7 m/s. Contact-point displacement and contiguous-duration non-regression checks PASS.
- Slip remains an independent unresolved failure; it was not part of this Pilot's optimization target.

## Classification

`GO2_LOW_SPEED_GAIT_STABILIZED_PARTIAL`

Next: **low-speed failure diagnosis v2**. No Pilot 2 is run in this stage.

Stage 7 does not constitute final exp_011 capability PASS.
