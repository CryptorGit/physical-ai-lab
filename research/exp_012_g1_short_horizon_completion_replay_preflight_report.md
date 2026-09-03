# exp_012 Stage 2H — Short-horizon completion replay preflight

## Status

`SHORT_HORIZON_COMPLETION_REPLAY_NO_EFFECT`

`PHASE_B_NOT_READY`

This stage used only disposable actor/optimizer clones. It created no persistent
training checkpoint and made no production policy update.

## Replay data

The source policy was the Phase A iteration-50 checkpoint
`4edbb595e28e24dc09cf39e8245c7be1b1bebf792798a73af2e562075d0fe952`
(Adam step 88,000). Each branch collected its own fresh, on-policy S100 data
under the unchanged Phase A command distribution. Initial completion collections
contained 67–83 events for the main branches. The background control required a
second batch because its first batch contained only 60 events; its merged corpus
contained 130 events in 126 episodes.

Replay units preserve one episode's temporal order from at least 15 control
steps before takeoff through at least 10 steps after completion landing, capped
at 60 steps. Consecutive overlapping cycles are merged. The buffer is FIFO,
holds at most 256 unique completion windows, has maximum age four shadow
iterations, and permits at most one reuse per iteration and four total uses.

In the four-iteration R1/A0.025 branch, eligible windows were 53, 77, 40, and
64. Drops were dominated by the joint clip-fraction gate; KL, ratio-p99, and ESS
were also audited per window. Enough eligible data remained for every auxiliary
step.

## Auxiliary gradient

The applied auxiliary-gradient ratio remained below the 10% cap:

| Branch | Horizon | Coefficient | Effective replay/on-policy norm | Replay vs unsafe cosine |
|---|---:|---:|---:|---:|
| R1/A0.025, iter 1 | 1 | 0.025 | 5.7% | -0.033 |
| R1/A0.025, iter 4 | 1 | 0.025 | 2.3% | +0.390 |
| R2/A0.025 | 2 | 0.025 | 2.1% | +0.313 |
| R4/A0.025 | 4 | 0.025 | 4.4% | +0.208 |
| RB/A0.025 | 4 | 0.025 | 3.5% | -0.013 |

A0.050 and A0.100 exceeded the gradient cap (21.1% and 17.9%) and were
fail-closed before the auxiliary step. No coefficient was silently rescaled.

## Stability

All applied branches passed the shadow hard gate. Final exact old-to-new KL was
0.012–0.033, joint clip fraction was 0.177–0.333, mean-action shift remained
well below 2.0, critic gradients and value losses remained finite and far below
their limits, and NaN/Inf count was zero. The strict-resume LR contract remained
active; the first standard optimizer step began at `7.59375e-5`. Auxiliary
updates changed the actor mean only; critic, std, and normalizer were frozen.

## Loss cross-effect

The central causal gate failed. Completion-window holdout loss worsened in every
applied completion-replay condition:

- R1/A0.025 across four iterations: +2.99%, +1.95%, +2.57%, +1.42%.
- R2/A0.025: +7.05%.
- R4/A0.025: +10.10%.

R2 also worsened unsafe loss by 12.7%. The compute-matched background auxiliary
update worsened completion loss by 31.0%, confirming that an extra actor step by
itself is not beneficial. Completion replay therefore did not provide a
completion-specific loss advantage over either standard PPO or matched
background computation.

## Consolidation and behavior

The tracked Stage 2H comparison uses the first 20 pre-registered episodes per
speed and action mode.

| Branch/final shadow iteration | Deterministic completion, all speeds | S100 completion, all speeds | 2.4 periodic | 2.4 fall |
|---|---:|---:|---:|---:|
| R0 standard, iter 1 | 0 | 2 | 0% | 35% |
| R1/A0.025, iter 4 | 0 | 1 | 0% | 40% |
| R2/A0.025, iter 1 | 0 | 5 | 0% | 50% |
| R4/A0.025, iter 1 | 0 | 5 | 0% | 60% |
| RB/A0.025, iter 1 | 0 | 3 | 0% | 50% |

No branch produced deterministic completion. R1 did not retain its transient
S100 improvement over four shadow iterations, and no branch produced periodic
running at 2.4 m/s. The higher S100 counts in R2/R4 coincided with materially
higher fall rates and with worse completion holdout loss, so they do not satisfy
the replay-specific causal or safety gates. The final no-replay evaluation after
R1 iteration four likewise showed no consolidated mean-policy behavior.

Mean-action distance was tracked through importance-ratio/KL drift of replay
windows. Older R1 windows were increasingly removed by the clip gate rather
than pulling the mean policy into a stable completion basin.

## Retention

At R1 iteration four, STAND, WALK 0.6, WALK 1.2, and WALK_TO_STAND were each
20/20 successful. Other replay branches retained 95–100% on these diagnostics.
Retention was not the limiting failure; RUN completion consolidation and RUN
safety were.

## Interpretation

The proposed short-horizon auxiliary objective is not supported. It was
numerically stable and sufficiently on-policy under the registered KL/ratio/ESS
gates, but it consistently moved held-out completion loss in the wrong direction.
Increasing replay age did not repair that direction, and apparent stochastic
event-count increases were coupled to higher falls without deterministic or
periodic consolidation.

Fresh-process reproduction of a “best” replay contract was not run because no
candidate passed the causal and safety gates. This is a fail-closed application
of the protocol, not a resource failure.

## Classification and next action

Main classification:

`SHORT_HORIZON_COMPLETION_REPLAY_NO_EFFECT`

Selected replay contract: none.

Phase B readiness:

`PHASE_B_NOT_READY`

The single next method is:

`close completion-event reuse route and pivot to reverse single-policy continuation from the exp_005 Stage 4 RUN-capable parent`

That future route would still produce one final actor checkpoint; it does not
introduce runtime expert switching.

## Protection

Stages 0–2G, exp_005–exp_011, all formal checkpoints and optimizer states,
reward/curriculum/network/observation/action/physics, Isaac Lab/RSL-RL core,
capability manifests, and production artifacts were not changed. Raw rollouts,
buffers, and temporary clone states remain outside the commit. Remote push was
not performed.
