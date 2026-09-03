# Exp014 Phase 2-D24B native START contract and recovery

## Result

Classification: `EXP014_D24B_NATIVE_TIME_CONTRACT_MISMATCH`.

The original exp012 100% result used a 10 s episode gait classification (`flight_fraction < 0.10`) with no velocity/yaw confirmation and no explicit acquisition deadline. D24A instead held 0.6 m/s, imposed 25 consecutive forward/lateral tracking steps before 3.0 s episode time, and added torque saturation. The contracts therefore do not match.

D24A raw reanalysis found 77 first-step events and 38 conservative N3 trajectories, but zero N4 trajectories: no episode maintained the full forward/lateral/yaw criterion for 25 steps (maximum was 10). Per-joint native torque dwell cannot be reconstructed because D24A did not persist applied-torque tensors; the 31 episode-level flags are preserved without extrapolation.

The 64 fixed S_HOLD sources were evaluated on R0-R4. Continuous 0.5 s source validity was 0/64 per route. Stage2Q routes produced 30/64 first-step events but zero confirmed acquisitions, and all had long-dwell torque saturation under the exp014 definition. Consequently, safe S_HOLD demonstration coverage and W_MOVE handoffs were both zero. No static distillation was run.

W_MOVE 0.3 m/s is in the formal command family; 0.6 m/s is not authorized as a complete omnidirectional scope. The preregistered 0.6->0.3 bridge was represented, but no eligible demonstration reached handoff.

No persistent policy update, checkpoint, validation or held-out access, D25 training, RUN integration, or remote push occurred. The recommended next experiment is a model-based S_HOLD-to-native-prestart bridge using centroidal/ZMP or whole-body IK supervision.
