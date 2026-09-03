# exp_009 final report

## Final classification

`CLOSED_NO_GO_UNIFIED_ACTION_MANIFOLD`

exp_009 tested whether the formally successful WALK, RUN_LOW, and WALK_TO_RUN
capabilities could be represented by one continuous-speed action manifold
without reinforcement learning. The tested unified and morphing families are
closed as a No-Go.

## Evidence chain

The single-head student achieved low one-step imitation loss but destroyed
closed-loop WALK retention. Increasing raw capacity and separating output heads
did not explain or repair the failure. Short-horizon finite-difference weighting
did not restore the WALK attractor, and the nonlinear surrogate was not
accurate enough to supervise rollouts safely.

Preserving the WALK expert exactly and learning only a bounded residual also
failed at the representation gate. Stage 5 showed that this was not a
raw-output, clipping, scaling, joint-order, previous-action, or observation
semantics mistake: WALK and RUN differ substantially at the actual applied
joint target.

Stage 6 tested the favorable oracle interpolation between the WALK and RUN
internal bases. Steady endpoints were consistent, but the scalar oracle covered
only 84.94% of WALK_TO_RUN full action vectors inside the existing bound; fixed
joint-group oracles covered 71.76%. The WALK_TO_RUN oracle coefficient was
already RUN-side at entry. The transition teacher therefore is not a trajectory
along the tested two-base manifold.

## Preserved positive results

The frozen WALK, RUN_LOW, and WALK_TO_RUN teachers remain valid in their
previously established scopes. exp_009 does not retract or modify those
capabilities. It rejects only the tested unified action-manifold,
dynamics-loss, surrogate-loss, residual, and two-base morph formulations.

## Research pivot

The next experiment is `exp_010_unitree_g1_post_run_walk_attractor`. Instead of
forcing RUN-derived states into the original WALK expert's acceptance basin, it
asks whether a separate low-speed steady-state expert can maintain a safe
1.2 m/s attractor after RUN-cycle termination and first WALK-compatible contact.
