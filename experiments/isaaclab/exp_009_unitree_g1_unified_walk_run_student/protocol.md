# Stage 0 protocol

- Immutable teachers: WALK, RUN_LOW, and the formal WALK_TO_RUN checkpoint.
- Student input: canonical 123D locomotion observation; no teacher or skill ID.
- Student output: one 37D normalized position action at scale 0.5.
- Dataset: at least 1.5 million physical teacher steps, grouped 70/15/15 split.
- Objective: action Huber + action-delta Huber + explicit AdamW decay only.
- Closed loop: one student controls WALK, RUN, forward ramp, intermediate
  speeds, and diagnostic reverse ramp without switching controllers.
- At most one DAgger round, and only if offline BC retention fails.

PPO, reward fine-tuning, RUN_TO_WALK transition training, capability edits, and
production promotion are prohibited.
