# EXP 013 Phase W2-P1-A9 Observation/History Contract Preflight

## Outcome

Primary classification: `EXP013_W2_P1_A9_NO_CONTRACT_SOLVES_INTEGRATION`. No student checkpoint, PPO update, overlay, runtime teacher, router, blending, or promotion was created.

## Observation and dataset

The policy input is 124D: base linear/angular velocity (3+3), projected gravity (3), calibrated current actor command (3), joint position/velocity (37+37), prior action (37), and gait (1). Contact and command history are absent. ReplayV2 live collection produced 74,666 exact-control-step samples from 1009 recipes across eight contexts. The split is recipe-disjoint. Of 189 A8 local points, 162 were labelable and 27 remained unlabelable after the opposite-checkpoint counterfactual.

## Representation result

O0 worst validation MSE was 0.009160. O1 command history reduced it to 0.001313; O3 reached 0.001311. Both nevertheless missed the fixed 0.001 gate in stop recovery and moving-yaw retention. Contact alone (O2) and the 8-step GRU residual (O4) also failed. Every expanded actor matched A4 bitwise at initialization.

## Authorization

There was no all-static-pass contract, so candidate-only physical validation, causal ablation, and frozen held-out physical confirmation were not run. `exp013_observation_contract_v2.json` was not created because it is PASS-only. The tested observation changes improve fit but do not authorize a final architecture.
