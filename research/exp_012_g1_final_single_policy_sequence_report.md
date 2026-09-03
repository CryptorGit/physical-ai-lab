# EXP 012 Stage 2Q — Final single-policy sequence integration

## Outcome

Stage 2Q is classified **G1_FINAL_STAND_STOP_FAIL**. The selected one-checkpoint actor preserves every
WALK/RUN endpoint and every gait/RUN transition, but the Stage 2 WALK/STAND teacher's zero-speed
behavior retains small stepping/contact oscillations. Two pre-authorized DAgger rounds reproduced
rather than removed that behavior.

## Student and data

- Parent: Stage 2N initial checkpoint (`04b43e…d121`)
- Architecture: `124 → 256 → 128 → 128 → 37`
- Frozen gait-conditioned std: `alpha_walk=0.30`, `alpha_run=0.65`
- Selected supervised base step: 17,500
- DAgger: 2 rounds, 500 episodes and 5,000 steps per round
- Selected SHA-256: `66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698`
- Static held-out: all ten endpoint/toggle conditions PASS; worst MSE
  `0.000234`, minimum cosine above 0.99989.

## Closed-loop endpoints

WALK 0.6/0.8/1.0/1.2 and RUN 1.2/2.4/2.6 each achieved 100% gait success and 0% fall.
RUN completion fired 5,662 times at 2.4 m/s and 5,645 times at 2.6 m/s. STAND had 0% fall
and 0.0447 m/s speed MAE, but zero-flight and 95% double-support gates were 0%;
mean double support was 75.1%.

## Transitions

STAND→WALK, WALK→RUN, RUN acceleration, RUN deceleration, and RUN→WALK were each 100%
with 0% fall. WALK→STAND reached 0.0414 m/s final speed and 0% fall, but failed the formal
zero-flight/double-support completion gate.

## Integrated sequence

- Formal completion: 0%
- Fall: 5%
- Final speed mean: 0.0547 m/s
- Heading p95 mean: 0.0804 rad
- Dangerous slip / impact / long-dwell saturation: 2% /
  0% / 0%

The locomotion body of the sequence succeeds, but initial/final STAND contact gates prevent any
episode from satisfying the formal all-segment completion predicate. Candidate stochastic
evaluation was correctly skipped because the deterministic gate did not pass.

## Runtime and protection

Evaluation uses one selected checkpoint, one mean actor, and one gait-conditioned Gaussian head.
Runtime teacher/expert/router calls, checkpoint switches, and action blends are all zero. No PPO,
reward, physics, teacher-checkpoint, previous-stage, or production-artifact changes were made.
