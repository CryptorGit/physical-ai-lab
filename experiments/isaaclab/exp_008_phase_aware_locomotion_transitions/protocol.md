# Stage 0 frozen protocol

## Observability

- Source: preserved `exp_007` formal graph route to RUN_LOW 2.6/2.8 m/s.
- Policy: frozen Stage 8C `model_10.pt`.
- Dataset split unit: complete diagnostic episode; reset seed, source speed,
  checkpoint, and neighboring steps never cross splits.
- Window: 16 steps before first WALK-compatible contact through 8 steps after
  the first contract break.
- Primary endpoint: contact break within three control steps.
- Timing leakage is audited with full 152D, timing-field ablation, legacy 123D,
  legacy plus applied action, and analysis-only phase upper-bound conditions.
- Age-matched metrics are reported independently.

## Controllability

Counterfactual branches are replayed from reset with identical source route and
identical action history before the branch. No simulator state is copied or
injected. Branch ages 4/5/6 are tested with baseline, frozen WALK, frozen RUN,
bounded joint-group perturbations, and target-WALK alignment. Success requires
the unchanged 20-step WALK contract and all safety gates.

## Prohibitions

No PPO, optimizer update, actor update, reward change, completion-detector
change, production feature addition, production GRU, artifact creation, or
capability update is permitted.
