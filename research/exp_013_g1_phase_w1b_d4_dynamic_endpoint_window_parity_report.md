# exp_013 Phase W1B-D4 dynamic endpoint-window parity preflight

## Outcome

Classification: `FINAL_HOLD_STATIC_METRIC_PARITY_FOUND`.

The selected preregistered candidate is `W1_FINAL_HOLD_ALL × M1_STATIC_ENDPOINT_EQUIVALENT`.
Its mean static/dynamic pass-rate difference is 0.778%,
paired disagreement is 0.889%, and negative-control
false-PASS is 0.331%.

## Contract finding

Static evaluation uses mean yaw sign and yaw MAE over a constant-command endpoint.
The old dynamic metric used instantaneous sign fraction over an episode that included
pre-hold, ramp, zero crossing, acquisition, and endpoint retention. The paired dataset
shows that acquisition and endpoint capability must be reported independently.

The selected endpoint contract uses only the preregistered window and applies the same
mean-yaw/MAE, translation, and safety criteria as the static endpoint evaluator.
No threshold, formal gate, production evaluator, policy, or command calibration was changed.

## Rear and random controls

Rear true endpoint partial after parity correction: `False`.
Random segment mean success is 32.7%,
whereas full-episode all-segment success is 0.0%. The large difference is
reported separately from endpoint parity.

## Protection

No PPO update or checkpoint was created. Existing checkpoints, optimizers, sampler,
reward, curriculum, network, physics, static/dynamic production evaluators, and
formal gates remain unchanged. Remote push was not performed.
