# EXP014 Phase 2-D29B0 zero-speed WALK gate ablation

## Classification

Classification: EXP014_D29B0_ZERO_SPEED_WALK_PRECONDITION_UNSAFE. D29A remains EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED; its artifacts were read-only.

## Identifiability and provenance

The causal actor was the single frozen Exp014 141D actor results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/dagger_checkpoints/round_2_step_10000.pt with SHA-256 7382163c649676f4e551aa438943cd5bd069e438b08469d6359e30ef4ca5f9e7 and architecture [141,256,128,128,37]. The same actor, observation schema, and action path were used for STAND and WALK. The physical command was [0,0,0] in both conditions. Only target motion mode one-hot indices 127:130 changed; previous mode, command history, ramp progress, and the 123D physical observation were held equal. The existing W_MOVE actor was used only after the fixed 100-step preconditioning interval.

## Zero-command preconditioning

P_STAND ready-like coverage was 0/8; P_WALK_ZERO coverage was 0/8. Metrics are in zero_command_preconditioning.csv/json. Periodicity is reported separately from safety and drift.

## Route comparison

Route A is STAND preconditioning followed by a hard switch to W_MOVE [0.3,0,0]; route B is WALK-zero preconditioning followed by the identical switch. Safe first-step counts were 0/8 and 0/8; W_MOVE 10-step entry confirmation counts were 7/8 and 5/8. Per-source yaw, clearance, support, action discontinuity, and first-divergence records are in start_route_comparison.csv/json.

## Gate decision

The preregistered positive-control requirements were not relaxed. The observed median route yaw p95 changed from 0.332466 to 0.229654 (relative reduction 0.309), and the 95th-percentile maximum clearance changed from 0.12096 to 0.112666. gate_effect_statistics.json records the full decision. No new training, checkpoint, WBIK/centroidal modification, validation, held-out evaluation, or RUN integration was performed.

## Repository

Expected user start HEAD was 600298f1d21acaf7389efd96ede081faa9bd90b9; actual source-of-truth HEAD at execution was c6d374c4dc77fd704c4bdac4e7fe02f5ee942141. The expected mismatch was preserved without reset. D29A/D29B protected artifacts and unrelated pre-existing worktree changes were preserved. Persistent update: 0; new checkpoint: 0; remote push: false.
