# exp_012 two-stage single-policy RUN acquisition — Phase A

## Integrity

Phase A strictly resumed the Stage 2 retry selected iteration-100 checkpoint
`8d8afac60cafbd4adf0b98469fab01f711f32771a40899653d962cc08a5d8143` with Adam step 87,000 and restored LR
`7.59375e-05`. Runtime, scheduler, and optimizer LR matched.
The base and `SafePeriodicFlightReward` semantics, PPO configuration, network,
observation, action, and physics were unchanged. Yaw command was zero and all
external controllers were off.

## Training

The single actor completed 100 iterations / 2,457,600 interactions. First-update
exact KL was 0.012057, maximum per-step KL was
0.020637, and clip fraction was
0.1811; all stability gates passed.

The first stochastic completion fired at iteration 11. Across
training, 241 completions occurred in 69
iterations. Peak density was 0.0335% at iteration
54, below the registered 0.05% gate. Frozen
deterministic evaluation produced completion fires in 0 of 10 checkpoints.

## Selected checkpoint

Iteration 50, SHA `4edbb595e28e24dc09cf39e8245c7be1b1bebf792798a73af2e562075d0fe952`, was selected by the
registered ordering after all deterministic completion densities tied at zero;
it had the strongest 2.4 m/s periodic-running result.

| speed | periodic | fall | speed MAE | heading p95 | slip | impact | saturation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.3 | 65% | 35% | 0.154 | 0.349 | 25% | 10% | 0% |
| 2.4 | 85% | 15% | 0.126 | 0.436 | 10% | 5% | 0% |
| 2.5 | 30% | 70% | 0.401 | 0.696 | 50% | 10% | 0% |
| 2.6 | 50% | 50% | 0.360 | 0.695 | 45% | 10% | 0% |

## Gradient emergence

At the selected checkpoint, precursor/base gradient ratio was
0.1106%, completion/base was
0.1004%, and run-specific/total was
0.1342%. Completion/total cosine
was 0.242. The effective Adam
descent-direction alignment was 0.010;
optimizer moments were retained, not reset.

## Retention

Selected-checkpoint STAND, WALK 0.6, WALK 1.2, and WALK_TO_STAND success were all
100%, with zero falls in these diagnostic sets. No catastrophic retention
collapse occurred.

## Classification

**SINGLE_POLICY_RUN_COMPLETION_EMERGED_PARTIAL**

Phase A clearly moved the gait toward periodic RUN (2.4 m/s 85%, 2.6 m/s 50%),
but completion remained too sparse, unsafe at several speeds, and absent in
frozen deterministic evaluation.

## Phase B

**NOT READY.** Phase B was not executed and no joint-retention protocol was
frozen. The single next action is **Phase A boundary diagnosis**.
