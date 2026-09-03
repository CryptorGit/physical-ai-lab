# Exp 013 Phase W2-P1-A2 — start-boundary physical capability diagnosis

## Scope and protection

This stage is a diagnostic-only evaluation of the frozen W2-P1-R2 step-37,000 student. It performed no student training, checkpoint selection, dataset or label mutation, formal closed-loop authorization, DAgger, or canonical promotion. The canonical parent remains W1B-R2 iteration 200.

## A1 reconstruction

The A1 exact-zero one-step branch protocol was reconstructed with seed 20279001, observation corruption disabled, 24 direction/yaw conditions, 200 matched states per condition, and four branches. The maximum difference from the recorded A1 aggregate metrics was exactly 0.0.

| Branch | Endpoint | Acquisition | Fall | Dangerous slip | Impact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Student | 89.604% | 88.417% | 9.292% | 8.417% | 0% |
| W1B start label | 93.104% | 89.562% | 6.146% | 5.646% | 0% |
| Stop teacher | 89.833% | 88.458% | 9.083% | 7.917% | 0% |
| Canonical parent | 93.312% | 89.500% | 5.917% | 5.542% | 0% |

The student is not uniquely unsafe: the stop-teacher branch is similarly unsafe. Conversely, the W1B/canonical start action is materially safer and more successful.

## Condition localization

The largest student-parent endpoint deficit was direction 270° with positive yaw (-26.0 pp); the corresponding acquisition deficit was -23.5 pp. Other endpoint parity failures were 270°/zero yaw (-18.5 pp), 90°/positive yaw (-15.0 pp), 270°/negative yaw (-10.5 pp), and 135°/positive yaw (-10.5 pp). The failure is therefore direction/yaw dependent, with the strongest concentration in lateral motion and positive yaw. It is not explained by yaw sign alone.

## Action discontinuity

The immediate student action jump from the saved previous action was small (mean L2 0.0507; estimated torque-jump L2 0.892). The stop teacher was similarly small (0.0710; 7.36). The much safer canonical/W1B actions had much larger jumps (L2 1.6486/1.6383; torque-jump L2 30.04/29.81). Holding the previous action produced zero action jump but retained a 9.06% fall rate. Skipping the zero boundary retained a 9.35% fall rate. These observations reject action-rate or PD-target discontinuity as the primary cause.

## State divergence and precursors

Branch state differences appear within the first 1–2 control steps and become materially separated by 2–4 steps. The decisive observation is basin entry: two or four complete canonical/W1B steps reduce falls to 0.458% and 0.250%, while student and stop-teacher actions held for the same durations retain approximately 9% falls. Fall precursors are dominated by coupled tilt/contact/slip development rather than an isolated initial torque impulse. The divergence is amplified by subsequent student closed-loop dynamics after the boundary.

## Contact phase and state coverage

Almost all reconstructed boundary states were double support (4,784/4,800 student branch samples). Single-support and flight strata contained too few samples for a causal phase claim. Contact phase is therefore secondary, not the primary explanation.

Training exact-zero states and A1 branch states were not cleanly separable: linear AUROC was 0.612, energy distance 0.00128, and mean nearest-neighbor distance 0.0776 in the stored observation scale. Validation-versus-held-out AUROC was approximately chance. The A1 failures are not explained by a clear state-distribution mismatch.

## Joint-group ablation

Replacing only lower body plus waist with the parent/W1B action did not recover safety (fall 9.42%/9.13%). Replacing only the upper body improved fall modestly (8.73%/8.46%) but did not reproduce the safe whole-body trajectory. Stop-teacher lower/upper substitutions also remained near the student baseline. The registered lower-body joint-localization trigger was therefore false, so no post-hoc hip/knee/ankle combinations were run. The evidence supports a whole-body start-action interaction rather than a single joint group.

## Timing and command onset

The current profile, previous-action hold, skip-zero boundary, and stop-until-first-nonzero profiles all remained near 9% fall. Delaying the student switch until command magnitude thresholds from 0.001 through 0.05 m/s also failed to remove the unsafe basin. Thus zero-command switch timing alone is not primary.

The best diagnostic intervention was the complete canonical/W1B action for 2–4 steps: endpoint success rose to 99.52–99.75%, acquisition to 91.04–91.96%, fall fell to 0.46–0.25%, and dangerous slip to 0.27–0.04%. A single W1B step improved outcomes but remained above the safety requirement.

## Student zero-command bounded diagnostic

The student survived all 400 standard-reset two-second zero-command trials. From exp_012 steady-stop states it survived 99.25%, and from start-boundary-equivalent exact-zero states 99.0%. Mean speed was approximately 0.045 m/s and mean absolute yaw approximately 0.049 rad/s in the latter two groups. The candidate is not generally unsafe at zero command; the failure emerges when the stop-like boundary state must enter a moving-start basin.

## Classification

`START_BOUNDARY_W1B_ACTION_REQUIRED`

The result is not an action-jump failure, a clear training-state coverage mismatch, a contact-phase-only failure, or a generic protocol failure. A short complete W1B/canonical start-action trajectory is required to enter the safe start basin; student- or stop-like boundary behavior does not do so reliably. Because lower- or upper-body substitution alone is insufficient, the required effect is whole-body and persists for more than one control step.

## Current artifact and next action

The W2-P1-R2 step-37,000 student remains diagnostic-only. Nonzero start imitation, stop recovery, steady stop, and moving imitation remain static PASS, while exact-zero physical capability remains FAIL. Closed-loop authorization is not granted and canonical promotion remains none.

The single next action is an **exact-zero W1B start-action retention preflight**, restricted to the boundary stratum and retaining one runtime actor. No runtime teacher, router, action blending, or checkpoint switching is authorized by this diagnosis.
