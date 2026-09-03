# exp_013 Phase W2-P1-D3 initialization-gap diagnosis

## Outcome

Primary classification: `CANONICAL_BALANCED_TRAINING_TOO_SHORT`.

The canonical W1B-R2 actor did reach the preregistered joint static gate under the unchanged balanced objective, first at the 10,000-step checkpoint and again at 40,000 steps. Old-objective pretraining made the 2,000-step consolidation reliable from old step 10,000 onward, but was not necessary for reachability. No persistent policy checkpoint, closed-loop rollout, DAgger round, PPO update, or promotion was performed.

## Initializations

- Canonical parent: W1B-R2 iteration 200, SHA-256 `61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`.
- Old supervised checkpoints 0/500/1k/2k/5k/10k/15k/20k/25k were all AVAILABLE.
- Parameter L2 from parent increased from 2.366 at step 500 to 6.525 at step 20,000 and 7.098 at step 25,000.
- Old step 0 is actor-tensor identical to the canonical parent.

## P3 replay matrix

- canonical: FAIL; start 0.00099963; stop-recovery 0.00103521; exact-zero 0.06777176
- old_step_0: FAIL; start 0.00099963; stop-recovery 0.00103521; exact-zero 0.06777176
- old_step_500: FAIL; start 0.00100086; stop-recovery 0.00102685; exact-zero 0.06805842
- old_step_1000: FAIL; start 0.00099885; stop-recovery 0.00102314; exact-zero 0.06796927
- old_step_2000: FAIL; start 0.00099785; stop-recovery 0.00101680; exact-zero 0.06804920
- old_step_5000: FAIL; start 0.00099339; stop-recovery 0.00100207; exact-zero 0.06800533
- old_step_10000: PASS; start 0.00098653; stop-recovery 0.00099422; exact-zero 0.06762248
- old_step_15000: PASS; start 0.00098257; stop-recovery 0.00099356; exact-zero 0.06723306
- old_step_20000: PASS; start 0.00098288; stop-recovery 0.00098732; exact-zero 0.06708939
- old_step_25000: PASS; start 0.00098633; stop-recovery 0.00097308; exact-zero 0.06738832

The success boundary across existing initializations is old step 10,000. Old step-20,000 reproduced tensor hash `975f2cb165e48853f87d79cb93de83ed50954627b5b3a37f38c3b2bd6d4a159b` and trace hash `50d15a131577d64015c5793af01da4db20c1d811cbcb6b105af362517f0b724c` exactly.

## Balanced-only horizon

- 2,000: FAIL; start 0.00099963; stop-recovery 0.00103521
- 5,000: FAIL; start 0.00101056; stop-recovery 0.00101894
- 10,000: PASS; start 0.00099945; stop-recovery 0.00099174
- 15,000: FAIL; start 0.00099706; stop-recovery 0.00100352
- 20,000: FAIL; start 0.00096998; stop-recovery 0.00101309
- 25,000: FAIL; start 0.00101073; stop-recovery 0.00097696
- 40,000: PASS; start 0.00098479; stop-recovery 0.00097355

The gate is narrow and non-monotonic at preregistered checkpoints: 10k and 40k pass, while 15k/20k/25k straddle the 0.001 boundary. This establishes reachability without old pretraining but not a stable selection plateau.

## Two-stage paths

- PATH_A_BALANCED_ONLY: FAIL; start 0.00101073; stop-recovery 0.00097696
- PATH_B_ORIGINAL_THEN_BALANCED: PASS; start 0.00098288; stop-recovery 0.00098732
- PATH_C_ORIGINAL_25K_THEN_BALANCED: PASS; start 0.00098633; stop-recovery 0.00097308
- PATH_D_BALANCED_THEN_ORIGINAL: FAIL; start 0.00102042; stop-recovery 0.00095699
- PATH_E_LINEAR_WEIGHT_SCHEDULE: FAIL; start 0.00100685; stop-recovery 0.00096637

Original-then-balanced is the most reproducible path among tested paths. Nevertheless, because canonical balanced-only reaches the gate, pretraining is an optimizer-path stabilizer rather than a required representation stage under the decision rules.

## Parameter and layer analysis

- Interpolation first passes after P3 at lambda 0.7; improvement is continuous, with stop-recovery MSE falling from 0.00103521 at lambda 0 to 0.00099824 at lambda 0.7.
- There is no intervening loss barrier on canonical-to-old20 linear interpolation; worst-group loss decreases from the canonical endpoint.
- L0_CANONICAL_ALL: FAIL; start 0.00099963; stop-recovery 0.00103521
- L1_OLD_TRUNK_CANONICAL_HEAD: PASS; start 0.00098542; stop-recovery 0.00098445
- L2_CANONICAL_TRUNK_OLD_HEAD: FAIL; start 0.00099317; stop-recovery 0.00103620
- L3_OLD_FIRST_LAYER_ONLY: PASS; start 0.00099683; stop-recovery 0.00099724
- L4_OLD_LAST_HIDDEN_AND_HEAD: FAIL; start 0.00099036; stop-recovery 0.00102299
- L5_OLD_ALL: PASS; start 0.00098288; stop-recovery 0.00098732

The old trunk with canonical head passes, while canonical trunk with old head fails. Old first layer alone also passes. The useful warm-start effect is therefore primarily feature/trunk initialization, not a special output head.

## Latent and gradient

- Layer-3 start/stop AUROC: canonical 0.991100, formal R1-1750 0.995803, old20 0.995947, old20+P3 0.995872.
- Layer-3 centroid distance grows from 1.9131 at canonical to 3.0038 at old20.
- Exact-zero start and steady-stop gradients are antagonistic; cosine is -0.706 at canonical, -0.977 at R1-1750, -0.863 at old20, and -0.989 after P3. Training succeeds by better feature separation and balancing, not by eliminating this local gradient conflict.

## Optimizer state

- old20_O1_FRESH_ADAM: AVAILABLE; joint=True
- old20_O2_OLD_OPTIMIZER_STATE: AVAILABLE; joint=True
- old20_O3_ZERO_MOMENT_SAME_STEP: AVAILABLE; joint=True
- canonical_fresh_adam: AVAILABLE; joint=False
- canonical_parent_optimizer_state: NOT_AVAILABLE; joint=not_evaluable

Fresh Adam, old Adam state, and zero-moment/old-step-counter variants all pass from old20. The canonical PPO optimizer cannot be strictly loaded because its parameter-group structure differs. Optimizer state is not primary.

## Warm-start validity

`VALID_REPRODUCIBLE_INTERMEDIATE`. The step-20,000 actor is traceable to the canonical parent, uses the resolved immutable dataset and current label contract, has saved optimizer/objective/seed information, and reproduces P3 exactly. The preferred next action remains a formal long-horizon balanced run from canonical because it has stronger end-to-end provenance.

## Protection and current artifact

- Dataset, labels, split, manifests, existing checkpoints, and optimizers: unchanged.
- New persistent policy checkpoint: 0.
- Closed-loop evaluation / DAgger / promotion: 0 / 0 / 0.
- Canonical parent remains W1B-R2 iteration 200. Candidate students remain diagnostic only.
