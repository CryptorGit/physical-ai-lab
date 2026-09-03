# LAND_SHALLOW scripted v0 audit

Result: `NOT_SUPPORTED`.

The frozen Stage 2 standing actor survived the collision-calibrated 0.02 m drop in 10/10 episodes. The selected scripted pre-flex/absorption controller passed only 8/10 at 0.02 m, 7/10 at 0.04 m, and 0/10 at 0.06 m. It reduced peak load modestly but introduced recovery/rebound failures, so it was not connected to the production route. A 0.08 m test was correctly skipped.

The original reset calculation treated the legacy toe/sole/heel keypoint plane as world ground. The final evaluator instead calibrates its offset from the settled USD collision contact plane and rejects stale post-teleport contact samples. This is required for a physical drop-height interpretation.

Raw outputs are under `results/exp_006_unitree_g1_command_skills/land_shallow/` (`baseline_final`, `grid_prepare_040`, `grid_prepare_060`, and `pilot_final`).
