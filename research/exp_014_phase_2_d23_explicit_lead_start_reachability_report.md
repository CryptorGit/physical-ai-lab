# Exp014 Phase 2-D23 explicit lead-foot START reachability audit

Classification: `EXP014_D23_EXPLICIT_LEAD_PHASE_NO_REACHABILITY`.

The versioned planner command appends LEFT/RIGHT first-swing semantics to the unchanged 141D causal contract, yielding a diagnostic 143D direct actor. Its `[0,0]` function-preserving expansion matched D22 I_DUAL exactly (max difference 0). No Teacher, condition, phase, or checkpoint identifier was introduced.

The search replaced D22's PCA-only space with twelve mirrored START modes over 0--0.8 s and eight W_MOVE PCA modes over 0.8--1.5 s. Eight train snapshots (four original/four mirrored) were each searched under both lead commands with 12 x 128 CEM candidates and hard safety rejection: 24,576 candidate evaluations.

No candidate reached FIRST_STEP_ACCEPTANCE or WALK acquisition. Final global-best replay was safe in 3/16 lead conditions and at least one command was safe for 2/8 snapshots, but staged progress remained 0/8. Generality and temporary distillation were therefore not executed. Success-class mirror classification was 100%, vacuously due to universal failure; contact-step match was 63.5%, so no positive mirrored solution exists.

Persistent updates/checkpoints, validation and held-out access were zero. D6--D22 artifacts and W_MOVE/S_HOLD/S_STOP_OMNI remained read-only. The next experiment is acquisition of an intermediate low-speed STEP specialist; oracle distillation and the second W_MOVE segment are not authorized.
