# Exp014 Phase 2-D22 direct START transition actor preflight

Classification: `EXP014_D22_DIRECT_START_TRANSITION_UNRESOLVED`.

## Architecture and endpoints

The diagnostic actor is a direct `141 -> 512 -> 512 -> 256 -> 37` action policy. It does not add, blend, or route a W_MOVE base action. Phase names P0--P4 are metadata over existing causal 141D timing/command fields. I_HOLD/I_MOVE exact expansion errors were 0/0. I_DUAL passed endpoint capacity after 100 temporary supervised steps: source MSE 9.07722e-06, steady MSE 9.81017e-05, cosine 0.9999758.

## Authority and reachability

The mean S_HOLD-to-nearest-W_MOVE action L2 gap was 3.888885. Bound-0.50 residual coverage was 62.88%; direct normalized coverage was 99.87%. Per-snapshot 12-PCA-basis CEM produced 25/50-step success rates [0.0, 0.0], safe rates [0.0, 0.0], and no acquisition-confirmed trajectory. Therefore no oracle generality or lead-foot classifier result was eligible.

## Temporary causal probes

All three one-update clones passed numerical stability. Their final basin-distance improvements relative to I_DUAL were [0.028159, 0.027571, 0.016757], and yaw improvements were [0.107027, 0.155055, 0.145995]; none met the 20% causal gate. Acquisition-confirmation remained zero. All updates were temporary; persistent update/checkpoint, validation and held-out access were zero.
