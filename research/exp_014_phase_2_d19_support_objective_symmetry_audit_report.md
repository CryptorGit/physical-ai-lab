# EXP014 Phase 2-D19 support objective symmetry audit

## Outcome

**EXP014_D19_SUPPORT_TIMING_OR_IMPLEMENTATION_BUG**. The synthetic fail-closed gate failed before any temporary policy update or physics probe. Persistent updates and checkpoints are zero.

## Implementation and timing

D18 reads world-frame ankle contact forces after `wrapped.step(action)`, so the reward observes the just-applied action and refreshed physics. No pre-action sensor timing bug was found. However, the registered schedule retains the 0.7 target while its weight decays from 0.50 to 0.75 s; the implementation hard-resets the target to zero immediately after 0.50 s while retaining a positive envelope. At 0.60 s the implemented target is 0 rather than 0.7.

The load term also lacks a valid-total-support mask. With F_L=F_R=0 and target=0 its value is exactly 1.0, allowing flight/no-contact to maximize the load term. Additive total-support reward does not remove that false optimum from the load term itself.

## Synthetic and mirror tests

11 synthetic tests ran: 9 passed and 2 failed (`zero_total_support_not_high_load_reward`, `schedule_t_0.60`). Algebraic mirror tests passed for 32 pairs: unsigned reward is invariant, signed-left maps to signed-right, and the repository joint permutation/sign map is involutive.

## Conditional diagnostics

Per protocol, Q_LOAD_ABS through Q_SUPPORT_FULL, gradient conflicts, advantage lags, action-to-load timing, signed left/right probes, and support-side reversal physics were not executed. Consequently symmetry cancellation and family conflict were not inferred from incomplete evidence.

## Positive references

D15 preserves the single acquisition only as aggregate outcome; A5 preserves aggregate profile results. Neither contains the raw contact-force/Lz trajectory needed to recover the actual support schedule, so `SUPPORT_POSITIVE_REFERENCE_INSUFFICIENT` is diagnostic only and no schedule was estimated.

## Decision

Decision precedence selects the implementation/timing bug. The next single experiment is to fix only the support reward implementation—keep the peak target through the decay window and gate load reward on valid total support—then rerun the D18 one-update causal preflight. Persistent PPO remains unauthorized.

## Protection

exp_005-exp_013, D6-D18 artifacts, policies, datasets, optimizers, physics, formal contracts, and actor inputs were unchanged. Persistent updates, checkpoints, RUN, Causal DAgger V2, and remote push are zero.
