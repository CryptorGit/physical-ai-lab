# exp_014 Phase 2-D9 static evaluator correction

## Result

Classification: **EXP014_D9_STATIC_EVALUATOR_CORRECTED_S1_ELIGIBLE**. D7 remains `EXP014_D7_STATIC_CAPACITY_FAIL`; no D7 byte or semantic content was changed.

The bug was at `run_phase2_d7_bc.py:47-50` and `run_phase2_d7_s1_bc.py:39-42`: a separately trained raw-141D classifier was copied into every checkpoint row and required to exceed 99%. Static Contract V2 evaluates only checkpoint-specific action regression plus immutable dataset integrity. Six regression tests passed.

All nine S0 checkpoints remain ineligible; step 30,000 has boundary MSE 0.004867 >0.001. S1 steps 20,000, 25,000, and 30,000 are eligible. Validation-only ordering selects S1 step 30000 with overall MSE 0.00004110, cosine 0.999987, boundary MSE 0.00025028, deceleration MSE 0.00001875, acquisition MSE 0.00000708, and worst-condition MSE 0.00006947.

The selected existing checkpoint is `results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation/raw/bc_checkpoints/s1_step_30000.pt` (SHA-256 `5de37e5d0807654d370ba7a79ee9872c4029cb50e548685423e48812249959d5`). No checkpoint was written. Raw phase accuracy, action-relevant phase accuracy, and physical phase safety remain diagnostics only. Held-out remains sealed and unopened. The next single experiment is validation-only closed loop with this frozen S1 checkpoint; DAgger is allowed only after a physical validation failure.
