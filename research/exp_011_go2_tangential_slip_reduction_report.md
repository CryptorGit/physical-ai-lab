# exp_011 Go2 tangential-slip reduction — Stage 11

## Reward preflight

Stage 7 iteration 50 (`d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd`) was resumed strictly with its matching Adam
state (step 22,000, learning rate 0.00026012294873748923). Actor, critic, std,
normalizer, deterministic action, and optimizer mapping passed the strict identity
audit. The only new semantic reward term was `go2_contact_tangential_slip`.
It uses PhysX contact points and foot-surface velocity `v + omega × r`, a causal
`F_n > 5 N` / contact-age mask, force weighting, and the frozen robust score.
The preflight signal and runtime gates passed. Throughput was
105.6% of Stage 7 and the one-shot calibrated weight was
`0.00559195994498`. The signal was non-zero in
49.9% of samples and
its Spearman correlation with stable-contact tangential-speed p95 was
0.862.

## Training

The frozen Stage 7 curriculum and Stage 10 phase-gated heading controller were
retained. Training completed 200 iterations / 9,830,400 interactions with seed
20261001. First-update exact KL was
0.01513; maximum exact KL was 0.01983 and
NaN/Inf count was zero. The pre-formal validation rule selected iteration
0 with SHA-256 `e7c6eb71b943369360686deeb376881161c6f78ce108ee29d89040a6a6ae464f`. This is the
bitwise-identical initial/Stage 7 actor: no trained checkpoint outranked it.

## Slip and capability retention

Zero command: completion 100.0%, fall 0.0%,
heading p95 0.0404 rad, speed MAE 0.0014 m/s,
and dangerous-slip episode rate 24.0%.

| speed (m/s) | fall | heading p95 | speed MAE | dangerous episodes | dangerous time | tangent p95 (m/s) | friction p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0% | 0.040 | 0.001 | 24% | 0.002 | 0.062 | 1.025 |
| 0.2 | 0% | 0.045 | 0.024 | 100% | 0.131 | 0.825 | 1.004 |
| 0.3 | 0% | 0.032 | 0.015 | 100% | 0.208 | 1.255 | 1.000 |
| 0.4 | 0% | 0.020 | 0.022 | 100% | 0.247 | 1.658 | 1.000 |
| 0.5 | 0% | 0.018 | 0.024 | 98% | 0.276 | 1.933 | 1.000 |
| 0.6 | 0% | 0.017 | 0.026 | 98% | 0.311 | 2.166 | 1.000 |
| 0.8 | 0% | 0.016 | 0.028 | 98% | 0.368 | 2.776 | 1.000 |
| 1.0 | 0% | 0.022 | 0.026 | 100% | 0.420 | 3.499 | 1.000 |
| 1.2 | 0% | 0.029 | 0.027 | 100% | 0.452 | 4.071 | 1.000 |
| 1.5 | 0% | 0.032 | 0.032 | 100% | 0.457 | 4.702 | 1.000 |
| 2.0 | 0% | 0.053 | 0.030 | 100% | 0.440 | 5.149 | 1.000 |

| transition | completion | acquisition | target hold | fall | heading p95 | dangerous episodes |
|---|---:|---:|---:|---:|---:|---:|
| 0->0.2 | 100% | 100% | 100% | 0% | 0.095 | 100% |
| 0->0.4 | 100% | 100% | 100% | 0% | 0.067 | 98% |
| 0->0.6 | 100% | 100% | 100% | 0% | 0.066 | 100% |
| 0.6->0.4 | 100% | 100% | 100% | 0% | 0.043 | 100% |
| 0.6->0.2 | 100% | 100% | 100% | 0% | 0.060 | 100% |
| 0.6->0 | 100% | 100% | 100% | 0% | 0.101 | 100% |
| 0->1.2 | 100% | 100% | 100% | 0% | 0.054 | 100% |
| 1.2->2 | 100% | 100% | 100% | 0% | 0.081 | 100% |
| 2->1.2 | 100% | 100% | 100% | 0% | 0.088 | 100% |
| 1.2->0 | 100% | 100% | 100% | 0% | 0.130 | 100% |

Across the registered 0.2–2.0 m/s speeds, the median dangerous-time reduction was
0.0% and the median tangential-speed-p95 reduction was
0.0%. Full slip gate pass: **False**. Capability
retention: **True**; integrated sequence completion:
100.0%; sequence fall:
0.0%; sequence heading p95:
0.083 rad; final stand:
100.0%.

No legacy contact-anchor displacement or foot-link-origin velocity was used by
the reward. Friction utilization and continuous contact severity remain reported
as diagnostics.

## Reward-exploitation audit

No reward exploitation guard fired. Final rollout flight fraction was
0.000529, stable-contact fraction was
0.834, and speed MAE was
0.030 m/s. There was no stopping/speed avoidance, sustained
flight increase, duty-factor collapse, or one-foot failure migration in the
selected result.

## Classification

`GO2_TANGENTIAL_SLIP_NO_EFFECT`

## Next

`tangential-slip reward directionality diagnosis before Pilot 2`

Stage 11 remains diagnostic: no capability manifest or production artifact was
updated, and no remote push was performed.
