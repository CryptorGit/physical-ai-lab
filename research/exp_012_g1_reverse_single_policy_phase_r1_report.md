# exp_012 Stage 2I — Reverse single-policy continuation Phase R1

## Result

`REVERSE_SINGLE_POLICY_WALK_RECOVERY_FAIL`

The exp_005 Stage-4 parent was uniquely resolved as `90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9`. Its
17-state Adam optimizer was strictly restored at step 105,000 and LR `1.5e-5`.
The run completed 100 iterations / 2,457,600 interactions. First-update exact
KL was 0.01269, clip fraction 0.18852, and all stability/LR gates passed.

The precedence-selected checkpoint is iteration 1,
`707bd50a8a168f2b247965ff6977e41da1d560094a1d5328737eaa76963f3ecd`.

## Capabilities

- WALK 1.2: success 0%, fall 0%,
  speed MAE 0.067 m/s, but gait remained periodic running
  in 100% of episodes.
- RUN 2.4: periodic 100%, fall
  0%, completion fires
  2220.
- RUN 2.6: periodic 100%, fall
  0%, completion fires
  2218.
- WALK_TO_RUN 1.2→2.4 / 2.6: 100% / 100%.
- RUN_TO_WALK 2.4 / 2.6→1.2: 0% / 0%; RUN gait remained at the endpoint.

RUN periodicity changed by 0 points at both speeds versus the parent and fall
changed by 0 points. There was no catastrophic RUN loss. The failure is a
low-speed gait-manifold recovery failure, not speed tracking: 1.2 m/s MAE was
only 0.067 m/s.

## Contract and next

One selected actor SHA and one actor parameter hash were used throughout formal
evaluation. Teacher, expert, router, blend, and checkpoint switch counts were
zero. Phase R2 was not run.

Next: `low-speed action-manifold reachability diagnosis`.

All earlier stages/checkpoints and unrelated dirty paths were preserved. No
remote push was performed.
