# EXP014 Phase 2-D1 reset-boundary causal DAgger V2 preflight

## Outcome

Classification: **EXP014_RESET_BOUNDARY_SPECIALIST_SCOPE_FAIL**.

The frozen exp_012 Stage 2Q Specialist S was evaluated from all 680 unchanged exp_014 reset recipes for 100 control steps. It achieved practical STAND **58.24%**, below the preregistered 95% positive-control gate. Fall was 0.00%, dangerous slip 0.15%, impact 0.00%, and long-dwell saturation 0.00%.

## Reset boundary

All 2720 candidate actions at control steps 0-3 were finite, within the configured action contract, and had no reset-buffer corruption. They were **not published as labels** because physical teacher scope was not established.

## Causal experiment

The first-divergence audit, Dataset V2, C0/C1/C2 training, validation, and held-out evaluation were not executed. This is the required fail-closed behavior and leaves H1-H3 unidentified; it is not evidence that reset-boundary labels are causal or noncausal.

## Protection and next experiment

V1 remained byte-identical at `75f5fd09de6f6bc3edd517910c8303f503232c8cdfdee46cd853d90caa8cdfca` and the fixed parent matched `7382163c649676f4e551aa438943cd5bd069e438b08469d6359e30ef4ca5f9e7`. Protected paths were unchanged during D1 finalization. The next single experiment is a read-only reset observation/action/history initialization and Specialist-S action-contract parity audit under the same reset distribution.
