# exp_010_unitree_g1_post_run_walk_attractor

This experiment tests whether RUN-derived states can enter and retain a
distinct low-speed attractor named `POST_RUN_WALK`.

It does not modify the original WALK expert or require the new state to match
the original WALK action or contact trace. Source occupancy is generated with
the frozen formal graph and the frozen Stage 8C `model_10.pt`; PPO storage
starts only after first WALK-compatible contact.

Stage 0 freezes and validates the Pilot 1 protocol. Stage 1 executes at most
one 100-iteration Pilot. A second Pilot is permitted only by an explicit later
stage, and the experiment-wide maximum is two.

The experiment is diagnostic until a later formal capability stage. Existing
capability manifests and production artifacts are not changed.

## Pilot 1 result

`POST_RUN_WALK_STATE_FAIL`

The live in-place route and first-contact handoff passed, but no deterministic
durable checkpoint acquired the POST_RUN_WALK contract at either 2.6 or
2.8 m/s source speed. The initial parent remained periodic RUN; learned
checkpoints terminated that pattern by moving into excessive-flight,
saturation, and heading-failure regimes. Training stopped fail-closed after
iteration 77 when trainable exploration standard deviation crossed the frozen
0.75 abort limit. No Pilot 2, formal evaluation, production enablement, or
capability-manifest update was performed.

## Stage 2 optimization preflight

`OPTIMIZATION_FAILURE_MULTIPLE`

The Stage 2 fixed-checkpoint replay separated exploration growth from two
earlier failures. The first update already produced a non-local deterministic
mean-policy shift and extreme PPO KL/clip metrics; critic value and return
scales later diverged. The diagnostic reward return was partly correlated with
progress, but normalized advantage was not, so the complete reward
directionality gate failed. Entropy gradients contributed less than 0.60 of
the log-std gradient at every audited checkpoint.

Because reward directionality and critic/mean-policy stability both fail,
fixing std at 0.25 is not an isolated valid intervention. The shadow update
and Pilot 2 were not executed. `POST_RUN_WALK_V1` is therefore closed as
`NO_GO`; existing capabilities remain unchanged.

```powershell
.\experiments\isaaclab\exp_010_unitree_g1_post_run_walk_attractor\scripts\train_post_run_walk_pilot1.ps1 -ValidateOnly
```
