# exp_014 Phase 2-D2: Specialist-S action-contract parity and scope audit

## Result

**EXP014_NO_EXISTING_STAND_SPECIALIST_PASSES**

Stage 2Q's original 124D adapter and the exp014 wrapper are bitwise identical on all 680 reset states: observation and deterministic mean-action maximum differences are both 0. The original exp012 moving-to-stop path reproduces at 99% practical success (speed 0.005452 m/s, yaw 0.002317 rad/s, fall 0%). The exp014 reset result remains 58.24%; this is scope mismatch, not adapter mismatch.

## Contract audit

The source contract is 123D manager observation plus one gait scalar. Positions are default-pose-relative, velocity and command scales are 1, base velocities and projected gravity are body-frame values, previous action is the normalized policy action, and the actuator maps it using scale 0.5 plus default joint positions. gait=0 selects walk/stand; gait=1 is the run negative control. The exp014-only 17D suffix is never passed to Stage 2Q.

Reset order is reset, command zeroing, sensor refresh/observation, deterministic mean action, then physics step. Original and current previous-action initialization are both zero. No observation/history buffer discrepancy was found.

## Counterfactuals

P0/P1/P2 are bitwise identical at 58.24%. Initializing previous action with the first policy mean (P3) reduces success to 51.32%. G0/G1/G3 are bitwise identical; gait=1 (G2) yields 0.00%. Neither counterfactual repairs the failure.

## Teacher scope and candidates

Stage 2Q remains authorized for moving-to-stop, WALK_TO_STAND_DECELERATION, and WALK_TO_STAND_RECOVERY. It is not authorized for RESET/STAND_HOLD. The sole predeclared eligible candidate, exp007 Stage 1 `model_4246.pt` (734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621), achieves 55.44% on the same 680 recipes and also fails the 95% gate. No S_HOLD teacher is selected; therefore the dual-teacher boundary test is not entered.

## Authorization and next experiment

Reset steps 0-3 labels and Dataset V2 are denied. The next experiment is exactly one action: train and formally evaluate a dedicated exp014 STAND specialist on the unchanged 680-recipe reset distribution before collecting more unified-Student DAgger data.

## Protection

No policy training, PPO, DAgger dataset construction, RUN integration, checkpoint creation, reset-distribution change, reward change, or physics change occurred. Existing protected artifacts were read only. No remote push was performed.
