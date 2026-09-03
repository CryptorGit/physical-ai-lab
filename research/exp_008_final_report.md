# exp_008 final report

## 1. Executive summary

`exp_008_phase_aware_locomotion_transitions` is closed as
`CLOSED_DIAGNOSTIC_COMPLETE`, with stage gate `PASS_DIAGNOSTIC_COMPLETE`.
Its local phase-aware correction approach is a No-Go. It created no production
locomotion capability.

The experiment diagnosed the failed G1 RUN_TO_WALK edge from `exp_007`. A
classifier could rank states close to contact break with high AUROC, but could
not predict break timing accurately enough for the preregistered gate. None of
the five bounded corrective-action families produced a safe 20-step WALK
contract.

## 2. Background from exp_007

`exp_007` ended as `PARTIAL_SUCCESS_ASYMMETRIC_STATE_GRAPH`. It formally
established STAND, WALK at 0.6/0.8/1.0/1.2 m/s, RUN_LOW at 2.4/2.6/2.8 m/s,
STAND_TO_WALK, WALK_TO_STAND, and WALK_TO_RUN to 2.6/2.8 m/s.

WALK_TO_RUN at 2.4 m/s was not supported. RUN_TO_WALK was `NO_GO_V1`;
graph-based STOP remained blocked and a full bidirectional graph was not
achieved.

## 3. Research questions

The experiment separated two questions:

1. Can the existing 152-dimensional observation predict when the
   WALK-compatible contact state will break?
2. Can a safe bounded action correction extend that state to the unchanged
   20-step/0.40-second WALK contract?

This was a diagnostic study, not a new controller-training study.

## 4. Dataset and protocol

Frozen diagnostic replay produced 2,048 episodes and 201,882 steps:

- 1,024 episodes from RUN_LOW 2.6 m/s and 1,024 from 2.8 m/s.
- 2,048 failed segments and zero successful 20-step WALK-valid segments.
- Break reasons: contact 1,579; speed 426; heading 41; safety 2.
- Launch phase: left support 726; right support 651; flight 671; double support 0.
- Episode/reset-seed split 60/20/20; step leakage 0.

PPO training, production actor updates, reward optimization, and capability
updates were all zero.

## 5. Observability results

For break within three steps:

| Feature condition | AUROC | AUPRC | time-to-break MAE |
|---|---:|---:|---:|
| Full 152D | 0.980 | 0.845 | 5.18 steps |
| Timing fields removed | 0.982 | 0.871 | 5.18 steps |
| Legacy 123D | 0.978 | 0.878 | 5.22 steps |
| Legacy 123D + action | 0.980 | 0.879 | 5.21 steps |
| Explicit phase upper bound | 0.981 | 0.872 | 5.16 steps |
| 16-step GRU | 0.989 | 0.918 | — |

Near-term break risk was rankable. Removing elapsed/remaining/progress fields
did not reduce performance, so the result was not merely elapsed-time leakage.
However, all static time-to-break estimates missed the required MAE of at most
1.5 steps. The GRU improved AUROC by only about 0.008, and explicit phase
features did not materially close the timing gap.

The fixed classification is `BREAK_NOT_PREDICTABLE`: high AUROC alone is not
enough to determine the corrective timing or action.

## 6. Controllability results

Counterfactual replays compared:

- baseline action;
- frozen WALK action;
- frozen RUN action;
- bounded joint-group correction;
- bounded target-WALK alignment correction.

Every family achieved 0 safe 20-step contracts out of 512 audited branch
states. Phase-limited successes were also zero. Prebranch state matching passed
without state copying, setters, or teleportation.

The fixed classification is `NO_LOCAL_CORRECTION_FOUND`.

## 7. What was ruled out

Within the audited scope, the failure was not resolved by:

- removing timing leakage;
- using 4/8/16-step observation history;
- adding explicit contact/air-time/landing phase features;
- substituting frozen WALK or RUN actions;
- bounded joint-group perturbations;
- bounded action movement toward the WALK expert.

This rules out the tested local phase-aware correction route, not every
possible G1 controller.

## 8. What remains possible

The results do not rule out a non-local trajectory controller, a jointly
trained locomotion policy, a different state representation, or a different
embodiment. They also do not show that Go2 will succeed.

## 9. Final capability graph

```text
STAND
  ↕
WALK
  └── WALK_TO_RUN ──→ RUN_LOW
```

Supported WALK_TO_RUN targets are 2.6 and 2.8 m/s. RUN_TO_WALK, graph-based
STOP, and the full bidirectional graph remain unavailable.

## 10. Showcase scope

The closeout showcase replays only formal `exp_007` capabilities:
STAND_TO_WALK, WALK at 0.6/0.8/1.0/1.2 m/s, WALK_TO_STAND, and WALK_TO_RUN to
2.6/2.8 m/s with RUN_LOW hold. Scene resets are explicitly labeled as new
scenes, not locomotion transitions.

## 11. Limitations

There were no successful 20-step segments in the frozen dataset, limiting the
streak-success classification problem. Counterfactual action coverage was
bounded and local. Results are from Isaac Sim, not hardware.

## 12. Why the next embodiment is Go2

The G1 reverse connection combined high-dimensional whole-body action,
alternating contacts, and narrow expert acceptance basins. The project will
next retest bidirectional gait transitions on Unitree Go2, whose lower degree
of freedom and simpler contact structure should make failure attribution more
tractable. This is a project decision, not an `exp_008` experimental claim.

## 13. Final classification

```text
exp_008: CLOSED_DIAGNOSTIC_COMPLETE
stage gate: PASS_DIAGNOSTIC_COMPLETE
observability: BREAK_NOT_PREDICTABLE
controllability: NO_LOCAL_CORRECTION_FOUND
local phase-aware correction: NO_GO
new production capability: NONE
next embodiment: UNITREE_GO2
```

## 14. Reproduction

Stage 0:

```powershell
.\experiments\isaaclab\exp_008_phase_aware_locomotion_transitions\scripts\reproduce_stage0.ps1
```

Closeout showcase:

```powershell
.\experiments\isaaclab\exp_008_phase_aware_locomotion_transitions\scripts\play_exp008_closeout_showcase.ps1
```

## 15. Repository provenance

Stage 0 commit:
`2795ec858f9390aeec3994f7fdd5dd97e3cd9cc5`.

Closeout work started from the actual repository HEAD
`10a5510661142d3f99929c7d65219ab9928273ec`, preserving the existing unrelated
dirty state. Commit hashes and video metadata are recorded in the final
closeout manifests. No remote push is performed.
