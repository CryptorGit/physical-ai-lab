# OpenDuckMini gait-quality PDCA

Status: **ACTIVE — current H3 release is safety-qualified but gait-quality rejected.**

Hardware deployment remains **PROHIBITED** until the simulation gates below pass and a
separate hardware-in-the-loop review is completed.

## Problem statement

The current evaluator proves physical safety, route closure, and coarse signed motion.
It does not prove straight tracking, useful reverse propulsion, alternating steps, or
low stance-foot slip.  The V2 showcase therefore must not be used as gait-quality
acceptance evidence.

Observed deterministic exact-home failures:

- Forward: `dx=+0.175 m`, `dy=+0.168 m`, final yaw `+48.3 deg` over 6 s.
- Reverse: `dx=-0.118 m`, `dy=+0.093 m` versus the nominal `-0.300/0.000 m` endpoint.
- Forward's pre-guard left-knee target exceeds the 0.050 rad target-margin envelope in
  `101/120` retained motion ticks, while the right knee exceeds it in `0/120`.  The
  asymmetric flattening is a primary causal hypothesis for lateral/yaw drift.
- All 40 straight-reverse segments in the current formal 20x30 evidence fail the new
  quality screen: speed ratio `0.440–0.644` and cross velocity
  `0.0167–0.0240 m/s`.

## Frozen quality gates

These gates are additional to every existing zero-tolerance safety, finite-state,
route, target-margin, slew, provenance, and hardware-prohibition gate.

- Commanded linear/yaw DOF tracking: `0.75 <= actual / requested <= 1.25`.
- Pure translation cross velocity: at most `0.012 m/s` and at most `20%` of commanded
  speed.
- Compound-motion cross velocity: at most `0.015 m/s` and at most `25%` of commanded
  linear speed.
- Uncommanded yaw: at most `0.05 rad/s`; steady heading error at most `0.15 rad`.
- Yaw-only translation: at most `0.012 m/s` and at most `0.050 m` over 6 s.
- Single support: `25–60%`; flight: at most `1%`.
- Requested/effective command error: at most `0.005` per axis.
- Rise time: `T30 <= 0.4 s`, `T75 <= 1.0 s`, first single support `<= 0.8 s`.
- Both feet must alternate support; left/right step-count difference at most one and
  duty-factor imbalance at most `10 percentage points`.
- Stance-foot tangential slip: RMS `<= 0.015 m/s`, p95 `<= 0.030 m/s`, and cumulative
  slip per stance `<= 0.020 m`.
- Leg motion must be periodic and non-degenerate: both legs must produce repeated
  joint/foot cycles rather than one-sided shuffling.

No warmup interval may hide rise-time or initial-direction failures.  Steady-state
metrics may use a documented window only after the startup gates pass.

## PDCA sequence

1. **Measure:** add all-substep foot kinematics, per-foot contacts, step segmentation,
   rise-time, command fidelity, and strict acceptance to the evaluator.
2. **Diagnose:** reproduce forward/reverse failures under exact-home and formal reset
   perturbations; compare raw action, guard input/output, per-joint saturation, contact,
   and stance slip.
3. **Improve:** search the smallest route-specific observation/target calibration.
   If calibration cannot pass robustly, train a dedicated expert with the exact final
   margin/slew composition in the training loop.
4. **Screen:** exact-home and fixed failure seeds, then at least 5 seeds x 15 s.  Any
   fall, nonfinite value, qpos/target/margin/slew violation, or quality-gate failure
   rejects the candidate.
5. **Integrate:** evaluate all 12 motions and the complete continuous transition
   schedule.  A local improvement that regresses another route is rejected.
6. **Qualify:** run the frozen 20 seeds x 30 s perturbation suite with exact source,
   model, policy, profile, and runtime provenance.
7. **Show:** generate a fixed-world overview plus foot close-up/top-view MP4 from the
   qualified build, with startup, endpoint, step, and slip telemetry visible.
8. **Release:** only the independent post-adoption 20x30 artifact may authorize a new
   simulation package.  Hardware remains prohibited.

## Promotion states

`DIAGNOSTIC` -> `COMPONENT_5X15` -> `ALL_MOTIONS_5X15` ->
`CONTINUOUS_5X15` -> `FORMAL_20X30` -> `POST_ADOPTION_20X30` ->
`SIMULATION_PACKAGE`.

Every transition is fail-closed.  Existing H3 artifacts remain immutable historical
evidence and are not reinterpreted under these stricter gates.
