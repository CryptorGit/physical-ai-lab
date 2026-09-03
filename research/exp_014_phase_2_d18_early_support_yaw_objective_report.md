# EXP014 Phase 2-D18 early support/yaw precursor objective

## Outcome

**EXP014_D18_SUPPORT_TRANSFER_OBJECTIVE_NO_EFFECT**. The registered causal preflight failed, so persistent PPO did not start: 0 updates, 0 persistent interactions, and 0 new checkpoints.

## Reference scales

The train-only reference used 64 S_HOLD source snapshots (10 contact-refresh samples each) and 10,240 safe W_MOVE forward-basin states. The deterministic dominant-support target was 0.7000. Derived scales were sigma_Lz=0.225283, sigma_dLz=7.338820, sigma_Mz=0.001000, sigma_load=0.300000, and sigma_support=1.000000.

## Gradient calibration and causal probes

One result-blind proportional calibration placed preventive yaw, support transfer, tracking, and safety/regularization gradient ratios at 33.88%, 28.23%, 39.66%, and 11.30%; all registered ranges passed.

The preventive-yaw one-update probe reduced early |Lz| p95 by 10.89%, with safety degradation within the 2 pp allowance, so that sub-gate passed. It did not reduce |dLz/dt| (change -39.96%) and yaw p95 changed from 0.5115 to 0.5113.

The support probe failed decisively: load-target error changed from 0.3441 to 0.3930, a -14.21% reduction (negative means regression), while total-support error changed from 0.4051 to 0.4267. The required 10% improvement was not present.

## Stability and safety

The all-V2 temporary update itself was numerically stable: exact KL 0.00000200, all-step KL 0.00000594, clip fraction 0.0%, mean final-action shift 1.7066, with 0% fall, dangerous slip, and torque saturation. Failure is therefore causal ineffectiveness of the support precursor, not optimizer instability.

## Decision

Per the registered stop rule, no persistent update, C2 expansion, or checkpoint selection was allowed. The next single experiment is a focused diagnosis of why the symmetric support-transfer reward produces the wrong load trajectory; D18 does not meet the separate 40-update criterion needed to declare the entire additive-residual PPO route No-Go.

## Protection

exp_005-exp_013, D6-D17 artifacts, W_MOVE, S_HOLD, S_STOP_OMNI, datasets, optimizer, physics, PD gains, friction, robot assets, command/observation contracts, and formal gates were unchanged. RUN, integrated Student, C2+, and remote push are zero.
