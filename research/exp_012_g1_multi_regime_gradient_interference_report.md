# EXP 012 G1 Multi-Regime Gradient Interference Diagnosis

## Result

Stage 2C is classified **RUN_REWARD_REACHABILITY_FAIL**. The selected iteration-100 policy does
not satisfy the pre-registered evidence for multi-regime actor-gradient interference.
The RUN actor-gradient norm is 7.644, or
1.81x the reconstructed combined
norm. Its cosines with ZERO/WALK/SEQUENCE are
-0.055 /
0.336 /
0.285.

## Gradient balance and interference

All 11 durable checkpoints were evaluated using 24,576 samples each (6,144 per
cohort). At iteration 100, every cohort has a positive projection onto the formal
20/20/20/40 combined actor gradient: ZERO 0.133,
WALK 0.142, RUN
0.641, and SEQUENCE
0.892. The corresponding negative
minibatch-conflict rates for RUN versus ZERO/WALK/SEQUENCE are
50.0% /
18.8% /
25.0%;
none reaches the 60% confirmation gate.

Conflict becomes more visible only by iteration 300: RUN-vs-WALK cosine is
-0.325 and its minibatch negative
rate is 81.2%.
That late conflict does not explain why the selected iteration-100 checkpoint
already failed safe RUN.

## Layer, joint, critic

At iteration 100, the strongest aggregate opposition is localized in the output
mean head against ZERO; std gradients remain aligned. Joint-local conflict is
real but distributed across ankle, hip, knee, arm, and hand joints rather than a
single actuator group. The top-ten list is frozen in `top_run_conflict_joints.json`.
RUN critic explained variance is 0.593;
global, cohort-local, and unnormalized advantage comparisons do not turn the
selected full-vector result into strong interference. Critic/advantage scaling is
therefore secondary.

## Reward reachability

Across all checkpoints, precursor and short safe-flight events occur, but inferred
alternating-landing completion events total **0**. At iteration 100 only
0.295%
of RUN-command samples emit any run-specific reward. This explains how a sizable
RUN gradient can exist—base speed/fall/regularization terms still contribute—while
the intended periodic-running direction remains weakly specified. Formal behavior
matches this: periodic RUN is 64% at 2.4 m/s and 22% at 2.6 m/s, with 34%/74% falls.

## One-step cross effects

The initial restored-Adam combined step improves all four cohort losses. At
iteration 100 its loss changes are ZERO +0.001022,
WALK +0.008648, RUN +0.010302,
and SEQUENCE +0.016336. This differs from the
instantaneous positive gradient projections because restored Adam moments encode
training history; it is retained as a secondary optimizer-history finding, not
used to claim current-gradient conflict.

## Next

One method only: **RUN reward reachability and gradient-strength preflight**. No PPO continuation, Pilot 2, reward or
curriculum change, checkpoint write, PCGrad, or MGDA was executed in Stage 2C.

## Protection

All checkpoint hashes match the prior manifest. Production policy updates: 0.
New training checkpoints: 0. Isaac Lab and RSL-RL core: unchanged. Remote push:
false.
