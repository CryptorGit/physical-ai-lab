# exp_012 G1 true STAND / STOP integration — Stage 2R

## Outcome

Stage 2R stopped at the mandatory source positive-control gate. No dataset was
collected, no supervised update or DAgger round was run, and no Stage 2R
checkpoint was created. The formal classification is
`G1_FINAL_STAND_POSITIVE_CONTROL_FAIL`.

## Sources

The unique contract-compatible true-STAND candidate was exp_007's Stage 1
reference `734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621` (`model_4246.pt`). It had exp_007 formal
retention evidence (98% hold, 2% fall, zero recorded flight, 98% final double
support), but it is also the zero-speed teacher already used by Stage 2Q.

The unique formal WALK_TO_STAND candidate was exp_007 Stage 4
`bb1bf713119b7980cfac4c1f43eb0d415bc32abe97a54509ee45d13061e858bd`, supported by WALK source
`9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa`. Its original formal evaluation reported 100%
completion, 0% fall, zero final flight, and 100% final double support.

Both sources match the G1 123D observation, 37D joint-position action, action
scale 0.5, joint order, asset, 0.005 s physics step and 0.02 s control step.

## Same-environment positive control

The sources were reevaluated deterministically in
`Isaac-Exp012-G1-Reverse-PhaseR1-v0`, using the exp_007 formal seed and the
same contact definition (two ankle-roll links; maximum sensor-history force
above 5 N). Each condition used 100 episodes.

| Metric | TRUE_STAND | WALK_TO_STAND |
|---|---:|---:|
| Formal success/completion | 3.0% | 3.0% |
| Fall | 1.0% | 2.0% |
| Mean final/hold speed | 0.005513 m/s | 0.001217 m/s |
| Flight-zero | 89.0% | 97.0% |
| Final double support | 3.0% | 0.0% |

TRUE_STAND failed the required 95% flight-zero and final-double-support gates.
WALK_TO_STAND failed the required 95% completion/final-double-support gate.
Speed and fall alone were not treated as sufficient, and no gate was relaxed.

## Student, endpoints, transitions, and final sequence

The Stage 2Q parent hash was verified as
`66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698`. Because the source gate failed, it was not
cloned or updated. Training steps, DAgger rounds, new checkpoints, endpoint
evaluations, transition evaluations and integrated-sequence evaluations were
all zero.

## Protection

exp_005–exp_011 and exp_012 Stage 0–2Q were not changed by Stage 2R. Existing
checkpoints, optimizers, reward, physics, Isaac Lab and RSL-RL were unchanged.
There was no production update, runtime routing, checkpoint switching, action
blending or remote push. Pre-existing unrelated dirty paths were preserved.
