# exp_014 Phase 2-D7 R4 stop-oracle distillation report

## R4 oracle

R4 retained 100% STOP acquisition, conditional hold, joint success, and worst-condition success across 204 balanced local perturbation episodes (six per formal condition). Fall, slip, impact, velocity saturation, and torque saturation were zero. Switch timings 0.45/0.50/0.55 s and direction/yaw/speed perturbations passed. Raw switch action L2 p95 was 2.3024, but the physical-transient gate passed with zero root discontinuity, contact-buffer corruption, and NaN/Inf.

## Dataset

`Exp014R4StopOracleDistillationDatasetV1` contains 2,702 train, 579 validation, and 579 newly sealed held-out episodes: exactly 70/15/15. Training has 258,349 samples and validation 55,263. It uses 310,331 eligible D6 oracle labels plus 3,281 new PRE_STOP labels. The D6 diagnostic held-out labels are excluded. Material conflicts are zero at all four quantizations. Boundary pairs are distinct in 141D and Teacher/condition metadata is excluded from actor input.

## Student

S0 initialization reproduced W_MOVE with max difference 0. After 30,000 BC steps S0 reached overall MSE 0.000568276, cosine 0.999902, and worst-condition MSE 0.000826855, but boundary MSE remained 0.00486655. S1 was function-preservingly widened (max difference 7.15e-06) and after 30,000 steps reached MSE 4.10996e-05, boundary MSE 0.000250281, and worst-condition MSE 6.94723e-05. Action regression passed, but causal 141D phase classification was 97.32% versus the mandatory 99%.

## DAgger, closed loop, and held-out

The mandatory static gate failed, so closed-loop validation and DAgger were not authorized. Student-visited states were not queried. The new held-out split remains sealed and unopened; no fallback occurred. No checkpoint was selected.

## Authorization and classification

S_HOLD and W_MOVE retain their prior authorization. S_STOP_OMNI_D7 is denied, and Causal DAgger Dataset V2 remains denied and unbuilt. Classification: `EXP014_D7_STATIC_CAPACITY_FAIL`. The only recommended next experiment is an S1 phase-classification error-cluster audit; do not return to PPO.

## Protection

D6 remains `EXP014_D6_STOP_SPECIALIST_VALIDATION_FAIL` and was not modified. exp_005-exp_013, existing exp_014 assets, physics, rewards, and checkpoints were protected. Reward PPO, RUN/OMNI-RUN, final three-mode Student, Dataset V2 construction, and remote push are zero.
