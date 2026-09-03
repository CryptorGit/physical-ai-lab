# Exp014 Phase 2-D24D fresh START revalidation

## Result

Main classification: `EXP014_D24D_FRESH_START_REACHABILITY_ZERO`.

The lineage audit found raw START-source restoration without warm-up in D15-D19 and D21-D24B. D16 persistent training, D18/D19/D21 causal probes, and D22/D23 reachability therefore have contaminated early-trajectory evidence. Source-code fixes in D19/D20, endpoint-capacity findings, and the D24A native contract result remain independent.

All eight D23 development sources were rebuilt by `Exp014FreshS_HOLDSourceLifecycleV2` in the same process and environment. Direct W_MOVE produced four first-step events but zero acquisitions, with no safety failures. The full D23 search consumed 24,576 candidates. It produced 151 safe candidate occurrences and six in-search first-step occurrences, versus 24 and zero in D23, but none of those first-step sequences reproduced under independent fresh verification. Verified FIRST_STEP acceptance, WALK acquisition, and confirmation were all 0/8 recipes. The decisive no-reachability conclusion therefore did not change.

No verified trajectory was available for distillation. The pure policy-gradient/action-search START route is closed. The only next experiment is a model-based S_HOLD-to-W_MOVE-compatible first-step Teacher preflight; no controller is implemented in D24D.
