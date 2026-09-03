# exp_009 Stage 1 — single-head interference diagnosis

## Scope

This stage used the immutable Stage 0 teacher dataset and grouped
episode/seed/trajectory split. It trained diagnostic students and probes only.
PPO updates, reward optimization, teacher updates, production promotion, and
capability changes remained zero.

## Findings

### H1 — raw capacity

The medium and large single-head models reduce some offline errors, but both
score 0% WALK retention at 0.6/0.8/1.0/1.2 m/s. Model size alone does not
recover the task-isolated closed-loop failures. Raw capacity is not the primary
explanation.

### H2 — hidden-mode aliasing

No exact or quantized cross-regime collision was found in the fixed diagnostic
sample. A small MLP predicts WALK, RUN, or WALK_TO_RUN identity from 123D with
99.998% test accuracy. The nearest cross-regime observations are not genuinely
near in the 123D metric. Hidden-mode aliasing is therefore not the primary
cause. Expanded-context models are retained as diagnostic upper bounds; because
their input layers cannot inherit the exact WALK initialization, their absolute
loss comparison is not treated as a causal context test.

### H3 — gradient interference

Regime gradients conflict. Across the audited checkpoints/layers, roughly 30%
of cosine measurements are negative; the worst whole-network cosine is about
-0.54. Sequential training is strongly asymmetric: training a new regime
raises the held-out error of prior regimes. This is a real failure mechanism.

The oracle-regime diagnostic multi-head also fails its relevant deterministic
closed-loop regimes. Output-head separation alone therefore does not restore
behavior; the failure already appears inside isolated BC.

### H4 — closed-loop dynamic sensitivity

Task isolation does not solve the key problem. RUN-only retains periodic
running, but WALK-only and WALK_TO_RUN-only fail their closed-loop capability
despite small held-out one-step errors. The strongest causal diagnostic is the
WALK joint substitution:

- substituting teacher ankle-roll actions into the Stage 0 student raises WALK
  success to 70% at 1.0 m/s and 80% at 1.2 m/s;
- substituting only hip, knee, or ankle-pitch actions does not provide the same
  recovery;
- the reverse ankle-roll substitution destroys the teacher-side behavior.

This demonstrates that small joint-local errors can produce rapid
contact/attractor divergence that average 37D BC loss underweights.

## Classification

`MULTIPLE_FAILURE_MODES`

Gradient interference and closed-loop sensitivity are both supported. Stage 1
prioritizes the latter because it remains after regime isolation.

## One next design

Use a **dynamics-sensitive distillation loss with short-horizon contact/state
matching**. This should be evaluated before gradient-conflict mitigation,
larger production students, conditional adapters, or multi-head designs.

The diagnostic multi-head and all larger models remain non-production upper
bounds.
