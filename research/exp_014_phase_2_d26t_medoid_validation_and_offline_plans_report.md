# Phase 2-D26T — W_MOVE medoid replay and offline-plan eligibility

## Medoid replay

D26S's LEFT episode 52/step 111 and RIGHT episode 187/step 115 were replayed through the original exp013 evaluator from fresh resets. All 100 selected references reproduced the stored raw state/action/contact hashes bitwise; the detached CPU CoM/DCM reduction also matched after using the D26S reduction order. The 50-step native-window aggregate tracking gate passed for 100/100 references, phase retention was 1.000, alternation 1.000, and safety failures were zero. The stricter per-control-step fraction is retained as a diagnostic (0.929) and is not substituted for the original evaluator's aggregate gate.

## Neighborhood and mirror

Fifty physical-feature nearest references per side were selected from the protected E0 event+2..6 population. Both side medoids and all neighborhood identities replayed successfully. Native DCM/action geometry is bilaterally valid but not a simple exact mirror; no artificial averaging or target symmetrization was performed.

## Offline plans

All 432 fixed D26 plans were registered with the original 0.30/0.40/0.50 second shift, 0.8/1.0/1.2 T_ref swing multipliers, and p50/p75/p90 clearances. They were fail-closed before WBIK execution because the protected D24D FreshS_HOLD artifact contains lifecycle/observation hashes only, not identity-complete q/root/contact states required for FK/Jacobian evaluation. No numeric trajectory was invented and no raw snapshot was restored. Therefore eligible plans are 0/432 and no D27 physics authorization is issued.

## Classification

**EXP014_D26T_OFFLINE_START_KINEMATICS_FAIL**

## Authorization

No bilateral or single-side D27 physics is authorized. The one next experiment is a fresh-lifecycle identity-complete S_HOLD source capture, followed by the unchanged D26T ledger/WBIK evaluation.

## Repository

Starting HEAD: `c9a247f42fe29a23f3a69fa728f94d9ab734c706`

Ending HEAD before commit: `c9a247f42fe29a23f3a69fa728f94d9ab734c706`

Protected D6–D26S, S_HOLD, W_MOVE, WBIK, CoM, polygon, and action-conversion artifacts were not modified. Persistent update 0; model-based START physics 0; raw snapshot restore 0; PPO/CEM 0; validation/held-out 0; remote push false.
