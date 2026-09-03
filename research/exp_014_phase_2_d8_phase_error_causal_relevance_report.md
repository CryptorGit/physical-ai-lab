# exp_014 Phase 2-D8 phase-error causal relevance audit

## Result

Classification: **EXP014_D8_PHASE_CLASSIFIER_IMPLEMENTATION_BUG**. D7 remains `EXP014_D7_STATIC_CAPACITY_FAIL` and is not retroactively passed.

## Phase semantics and classifier

The seven labels mix clock-defined, physical-event-defined, and Teacher-route-defined boundaries. `W_MOVE_TO_STAGE2Q_BOUNDARY` spans both T_MOVE and T_STOP, while acquisition and post-acquisition split nearly action-equivalent steadying behavior. The D7 97.32% value was produced by an independent raw-141D MLP trained for 3,000 balanced steps. It was neither an S1 hidden probe nor an S1 auxiliary head, yet the same scalar was attached to all S1 checkpoints and used in static eligibility.

Validation accuracy was 97.32%, macro F1 98.02%, and balanced accuracy 98.31%. The best of three raw-input diagnostics was also 97.32%; the labels are not 99%-separable under this audit. Errors were not concentrated at the 0.50 s switch: 0.00% occurred within ±4 steps.

## Action and physical relevance

Of 1,482 phase errors, 1,470 (99.19%) were action-safe and zero were action-critical. `ACTION_RELEVANT_PHASE_ACCURACY` was 100.00%.

Read-only counterfactuals covered 545 local-neighborhood states, up to 100 per observed error pair. Six met the preregistered physical-critical definition, giving `PHYSICAL_PHASE_SAFETY=98.90%`. Most were loss of acquisition-progress parity after small S1/oracle deviations; no diagnostic classifier output was supplied to S1, so these deviations cannot be caused by classifier routing. Because physical safety was below 99%, shadow closed loop was not authorized and the sealed held-out remained unopened.

## Decision

The next single experiment is evaluator-only: remove the erroneous attribution of an independent phase classifier to each actor checkpoint and recompute D7 static eligibility from unchanged action-regression artifacts. Do not retrain the policy, add S2, return to PPO, or open held-out.
