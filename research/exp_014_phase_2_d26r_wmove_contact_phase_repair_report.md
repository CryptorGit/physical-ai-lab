# Phase 2-D26R — W_MOVE native contact-phase repair

## W_MOVE parity

The protected exp013 `evaluate_w1b.py` adapter was run in an isolated D26R output directory for 100 fresh episodes at forward 0.3 m/s, WALK, zero yaw. Formal tracking success was **1.000**, fall **0.000**, and the original gate was **PASS**.

The mandatory capture-harness parity gate was **FAIL**. The original evaluator used seed `20274021` and the exp013 default reset entrypoint; the D26 capture used seed `20282601` and D3 reset recipes. The first registered contract divergence is control step 0, `command_trace`: original direct exposure `[0.3, 0, 0]` versus the D26 capture contract `[0.0, 0, 0]` (25-step minimum-jerk ramp). Because source lifecycle, seed, and command identity are not established, actor/action hashes and contact-event interpretation are not authorized.

## Sensor mapping

The protected D26 collision audit names `left_ankle_roll_link` and `right_ankle_roll_link` and their collision prims, with mirrored numeric sole polygons. Runtime force-tensor mapping was intentionally not promoted to PASS after the parity stop.

## Contact phase

Strict touchdown, hysteretic onset, support-dominance, and kinematic-force detectors were not run. No event source or gait classification is inferred from the D26 six strict events.

## Reference population

No new steady trace was collected. The protected D26 forensics remain 59 identity-complete states from 256 attempted episodes and are unchanged; they are not a D26R reference population.

## Geometry/DCM

D26 CoM, polygon, DCM, and WBIK implementations remain protected and unchanged. D26R does not recompute offline plans after the parity failure.

## Classification

**EXP014_D26R_WMOVE_NATIVE_PARITY_FAIL**

## Authorization

Bilateral and single-phase D27 physics are not authorized. The next experiment is capture-harness repair and a fresh original/capture parity run; no policy, physics, reward, WBIK, or checkpoint changes are authorized.

## Repository

Starting HEAD: `05715d92bad02280c4b5da9d117a2885207f1acd`

Ending HEAD at artifact generation: `05715d92bad02280c4b5da9d117a2885207f1acd` (the D26R commit is created after this report). Protected paths were not modified; persistent update 0; remote push false.
