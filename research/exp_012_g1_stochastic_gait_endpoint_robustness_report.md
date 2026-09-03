# exp_012 Stage 2M — stochastic gait endpoint robustness

## Result

**SAFE_GAIT_CONDITIONED_EXPLORATION_WINDOW_FOUND**

The frozen Stage 2K mean supports a nontrivial safe exploration window. The limiting student boundaries are
WALK alpha=0.40 and RUN alpha=0.70; the selected inside-boundary pair is
alpha_walk=0.30, alpha_run=0.65.

At alpha=1, the WALK teacher retains WALK_LIKE only 22% and the student
0%. At the calibrated WALK candidate the student retains
100%. Thus teacher std is an exploration parameter, not a closed-loop
endpoint-safety contract.

Candidate paired authority is 97%; WALK→RUN and RUN→WALK
acquisition are 100%/100%, with
fall 0%/1%.

Next single method: **gait-conditioned PPO endpoint-retention fine-tuning preflight using calibrated gait-specific exploration multipliers**.
