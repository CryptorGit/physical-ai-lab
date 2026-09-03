I have closed exp_013, a study of single-policy omnidirectional locomotion for the Unitree G1 in Isaac Lab.

One memoryless actor accepts body-frame vx, vy, and yaw-rate commands and executes 360-degree walking, pure yaw, and moving turns. Translation was formally evaluated at 16 directions in 22.5-degree increments.

The video contains only the exp_013 result: 16 translation directions, pure yaw, and forward/rear moving yaw. The earlier exp_012 Stage 2Q actor achieved bidirectional WALK/RUN transitions, acceleration and deceleration, and a practical stop, but that separate result is not included in this video and is not attributed to the exp_013 actor.

The positive result is that one checkpoint can retain continuous vx/vy control together with pure and moving yaw. The negative result is equally important: although stopping itself worked with a separate policy, I did not obtain one actor that jointly retained stop maintenance, safe restart in every direction, moving yaw, and stop recovery. Command history, contact phase, a short GRU, teacher-trajectory integration, and constrained PPO were tested, but none passed the complete joint gate.

I am closing exp_013 with both the achieved capability and the unresolved integration problem documented. All results are from simulation; no real-robot performance is claimed.

#PhysicalAI #ReinforcementLearning #Robotics #IsaacLab #UnitreeG1
