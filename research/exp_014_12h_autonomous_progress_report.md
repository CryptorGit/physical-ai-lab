# EXP 014 — 12h autonomous progress report

## Outcome

Classification: **EXP014_STATIC_PASS_PHYSICAL_FAIL**. The causal 141D contract and formal S/W dataset passed integrity. S0 (90,570 parameters) passed every static group at step 30,000; S1/S2 were therefore not authorized. Teacher-free closed loop failed practical STAND and downstream retention. Two registered DAgger retries improved aggregate STAND hold from 36.9% to 55.7%, but did not approach 95% and aggregate fall remained 50.2%.

## Central result

At identical physical B0 states, 680/680 STAND/WALK pairs had different 141D inputs and different S/W actions, with zero material collision at 1e-6 through 1e-3 quantization. Static dual-mode classification was 99.51%. Thus explicit mode resolves the representation-level zero-command ambiguity, but this run does not establish a unified physical actor.

## Diagnosis

The DAgger fail-closed proxy rejected reset steps 0–3 because the robot had not made foot contact. Rounds 1/2 began labeling STAND at step 4. The resulting policy improved STAND monotonically while WALK acquisition stayed nonzero, which localizes the next experiment to reset-boundary labelability rather than capacity or RUN.

## Scope

RUN integration, local-neighborhood promotion, and OMNI-RUN audit were not authorized because Phase 2 failed. No protected asset was changed and no runtime router, teacher, checkpoint switching, or action blend was introduced.
