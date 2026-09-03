# Phase 2-D26 — W_MOVE reference and deterministic WBIK

## Result

Classification: `EXP014_D26_WMOVE_REFERENCE_CAPTURE_FAIL`.

## W_MOVE reference

The fresh reset-recipe lifecycle attempted 256 episodes and durably captured 59 identity-complete post-touchdown transitions. The preregistered minimum was 20,000, so the reference gate failed; no padding or reuse of restored D17 states was performed. Bundle SHA-256: `cca8edd382ed78c6d69940db0f5aa90781da044c5b3d14b7197cdae6ba531242`.

## CoM/DCM

CoM is mass-weighted over body-local CoM offsets; centroidal momentum is explicitly unavailable. Body Jacobian shape is 44×6×43 and the point-corrected CoM Jacobian implementation has a synthetic finite-difference PASS.

## Foot geometry

USD collision cubes were found for both ankle-roll links. Numeric sole polygons were extracted from collision mesh vertices; left/right areas match and the mirror test passes.

## WBIK

`Exp014DeterministicHierarchicalWBIKV1` implements deterministic SVD damped least-squares hierarchy, SO(3) log error, active-set joint limits, velocity limits, and exact action conversion. Unit/property tests pass.

## Offline plans

All 432 pre-registered plan IDs were emitted, but none were marked eligible because medoid/reference capture did not meet the identity-complete population gate. No D25 plan physics was run.

## Authorization

D27 model-based START physics is **not authorized**. Persistent updates/checkpoints, raw snapshot restore, PPO/CEM, validation, held-out, and RUN were all zero.

## Repository

Starting HEAD: `34af62c4def27fbdf34d6bad67b91eb1618e3aff`. Ending HEAD before commit: `34af62c4def27fbdf34d6bad67b91eb1618e3aff`. Remote push: false.
