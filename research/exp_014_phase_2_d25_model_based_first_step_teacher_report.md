# Exp 014 Phase 2-D25 Model-Based First-Step Teacher Preflight

## Outcome

Primary classification: `EXP014_D25_KINEMATICS_INTERFACE_UNAVAILABLE`. The mandatory interface/reference gate failed before unit-test instantiation, offline IK, or physics. This is an infrastructure/input-contract result, not evidence that a model-based first step is physically impossible.

## Robot model

The runtime G1 has mass 32.238930 kg, 44 rigid bodies, 37 actions, control dt 0.02 s, physics dt 0.005 s, and decimation 4. PhysX exposes a 44-body Jacobian tensor of shape `[1, 44, 6, 43]`, per-body masses and inertias, and contact forces. It does not expose `get_mass_matrices()` on this ArticulationView. A registered constrained whole-body IK/QP interface and a versioned numeric sole polygon were not found.

## W_MOVE target

D17 registered 10,240 forward-0.3 states and a 122D distance representation, but its persisted artifact contains only the manifest plus the original D6 physical snapshot source. It does not contain the per-state foot poses, contact phase, CoM, centroidal momentum, DCM, support polygon, or next action required to select the requested real post-touchdown medoids. Recreating those quantities would require the raw state restoration prohibited by D25, or guessing target geometry, so neither was done.

## Reduced-order plan and WBIK

The LIPM/DCM equations, four phases, hard-task hierarchy, and fixed 27-plan grid were preregistered. Numeric DCM offsets, support polygons, step geometry, IK unit tests, and offline feasibility were not instantiated because the required target and solver contracts were incomplete. No constraint violation was hidden by clipping.

## Execution and protection

Physics candidates: 0. Development, handoff, generality, and distillation were not executed. Persistent updates, checkpoints, PPO, CEM, raw snapshot restores, validation access, and held-out access are all zero. D6-D24D artifacts and all Teacher checkpoints were left unchanged.
