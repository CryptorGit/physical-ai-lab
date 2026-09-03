# One Policy, Two Gaits: Closing My Unitree G1 Locomotion Study

I have closed a simulation study on controlling a Unitree G1 through **stand → walk → run → walk → stop** with one neural policy.

The runtime contract was deliberately strict: one checkpoint, one actor, no expert router, no checkpoint switching, and no action blending.

The key lesson was that speed alone did not specify gait. At the same 1.2 m/s target, the walking and running policies produced physically different contact patterns: walking used about 3.5% flight time at roughly 1.4 Hz, while running used about 48% flight time at roughly 6.2 Hz. Their state distributions were nearly disjoint, and bounded local action perturbations almost never crossed from the running basin into the walking basin.

I therefore added one scalar gait command, independent of speed:

```text
gait=0 → WALK
gait=1 → RUN
```

The resulting single mean network selected both gaits at the same speed and achieved:

- WALK at 0.6–1.2 m/s: 100%
- RUN at 1.2–2.6 m/s: 100%
- WALK→RUN: 100%
- RUN→WALK: 100%

The policy also accelerated, decelerated, returned to walking, and reached an average final speed of about 0.055 m/s.

One limitation remains. The strict standing gate—zero flight events plus final double support—was not met. The robot practically stops and rarely falls, but small stepping/contact oscillations remain. I keep that limitation visible in the demo.

Scientifically, this shifted my interpretation: the hard part was not primarily network capacity. It was the optimization path and the existence of separate dynamical gait basins. I also found that a teacher policy’s full exploration standard deviation is not necessarily a safe closed-loop runtime distribution; gait-specific temperature calibration was required.

This work is simulation-only in Isaac Lab. It is not a claim of perfect standing, scratch-only acquisition, Sim2Real, or general humanoid intelligence.

With these results and limitations documented, exp_012 is now closed.

[GitHub repository link]

#PhysicalAI #Robotics #ReinforcementLearning #IsaacLab #HumanoidRobot #MachineLearning
