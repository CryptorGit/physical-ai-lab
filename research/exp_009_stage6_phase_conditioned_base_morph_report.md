# exp_009 Stage 6 — Phase-conditioned base morph feasibility

## Classification

**BASE_PAIR_INADEQUATE**

Stage 4 remains `RESIDUAL_PARAMETERIZATION_INADEQUATE`; Stage 5 remains
`PIECEWISE_OR_PHASE_CONDITIONED_BASE_REQUIRED`.

## Oracle scalar morph

The audit reconstructed 100,000 common states without Isaac stepping:
40,000 WALK, 30,000 RUN, and 30,000 states from 300 complete WALK_TO_RUN
episodes.  All actions are actual normalized actions before the common 0.5
environment scale.

Steady endpoints are consistent: WALK alpha p95 is 0.000
and RUN alpha p05 is 0.960.  WALK and RUN full-vector
coverage are 100%.  WALK_TO_RUN coverage is only
84.94%, below both
the 99% feasibility requirement and 95% early-stop threshold.

The oracle alpha is already near RUN throughout the transition (mean
0.990);
zero-to-one progression occurs in 0.0% of
episodes.  Its numerical trajectory passes the variation/jump gate only because
it begins near the RUN base rather than morphing from WALK.

## Speed and phase

Fixed speed smoothstep provides only
54.71% WALK_TO_RUN
coverage.  The specified scalar oracle itself is infeasible, so the protocol
correctly stopped before fitting speed, 123D, transition-scalar, or explicit
phase alpha probes.  No identifiability claim is made.

## Groupwise diagnostic

Five fixed joint-group least-squares alphas reduce neither the bound violation
nor the full-vector requirement: WALK_TO_RUN coverage is
71.76%.  Thus the
requested groupwise fallback also fails its 99% gate.

## Next

No morph controller is authorized.  Retain the modular frozen experts; do not
implement a scalar or groupwise phase-conditioned morph from this result.
