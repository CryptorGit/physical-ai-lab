# exp_013 Phase W2-P1 practical-stop endpoint acquisition preflight

## Outcome

Classification: `EXP013_W2_P1_STATIC_REPRESENTATION_FAIL`.

The exp_012 teacher compatibility audit passed, and the preregistered latest passing switch was
`SW3_ZERO_TARGET`. It passed all 24 source direction/yaw conditions with aggregate success
98.71% and minimum condition success
94.00%. SW2 passed
21/24 and SW1 passed
11/24.

## Dataset and supervised integration

- Stop recovery: 7200 episodes
- Steady formal stop: 990 accepted / 1000 attempted
- Moving retention: 10100 episodes
- Start retention: 2373 accepted / 2400 attempted
- Supervised run: 25,000 optimizer steps completed
- Selected checkpoint: step 20000 under the preregistered ordering
- Std heads: frozen; critic/PPO: unused

The hard static representation gate failed: [('START_RETENTION', 0.0012912879465147853, 0.9996974468231201)]. The moving-retention, stop-recovery, and
steady-stop groups passed, but the selected student's START_RETENTION mean action MSE remained
above 0.001. Thresholds were not changed. Therefore no closed-loop student evaluation and no
DAgger round were authorized.

## Interpretation

The teacher demonstrates that a closed-loop stop basin exists and is reachable from every tested
moving direction/yaw state when switching at the zero target. This supervised mixture did not,
however, satisfy the preregistered simultaneous offline representation contract for start retention.
It would be invalid to infer practical-stop acquisition or moving retention from imitation loss
alone, so all closed-loop/formal outputs are explicitly marked not executed.

The canonical artifact remains W1B-R2 iteration 200 (`61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`). The W2-P1 student is a
diagnostic candidate only and is not promoted.

## Protection

No existing checkpoint, optimizer, sampler, reward, physics, Isaac Lab core, or RSL-RL package was
modified. No PPO was run. No remote push was performed.
