# exp_008_phase_aware_locomotion_transitions

```text
STATUS:
CLOSED_DIAGNOSTIC_COMPLETE

OBSERVABILITY:
BREAK_NOT_PREDICTABLE

CONTROLLABILITY:
NO_LOCAL_CORRECTION_FOUND

NEXT EMBODIMENT:
UNITREE_GO2
```

`exp_008` is a new, diagnostic-only successor to the closed `exp_007`.
Stage 0 asks two separate questions:

1. Can the next RUN-to-WALK contact break be predicted from the frozen 152D
   policy observation?
2. At a replay-matched pre-break state, can a bounded action correction extend
   the WALK-valid streak to the unchanged 20-step/0.40-second contract?

Stage 0 performs no PPO, optimizer update, reward modification, transition
actor update, production observation change, or capability registration.

Stage 0 result:

```text
observability: BREAK_NOT_PREDICTABLE
controllability: NO_LOCAL_CORRECTION_FOUND
next single direction: UNIFIED_WALK_RUN_DISTILLATION
```

The timing-ablated 152D probe ranks near-term break risk well, but misses the
fixed time-to-break accuracy gate. No prebranch-matched bounded correction,
including frozen WALK/RUN actions, extended the unchanged contract to 20 safe
steps.

`exp_008` did not create a new production locomotion capability.
The closeout showcase replays capabilities formally achieved in `exp_007`.
It does not replay or imply support for RUN_TO_WALK.

The local phase-aware correction route is closed as `NO_GO`. Moving to Unitree
Go2 is a project decision to retest bidirectional gait transitions with a
simpler embodiment; `exp_008` does not establish that Go2 will succeed.

The immutable reference policy is the `exp_007` Stage 8C diagnostic
`model_10.pt`. Raw trajectory columns missing from Stage 8C/8D are regenerated
only by diagnostic replay with the frozen policy and source graph.

Reproduce:

```powershell
.\experiments\isaaclab\exp_008_phase_aware_locomotion_transitions\scripts\reproduce_stage0.ps1
```

Results:

```text
results/exp_008_phase_aware_locomotion_transitions/
stage0_observability_and_controllability/
```

Final report and closeout:

```text
research/exp_008_final_report.md
results/exp_008_phase_aware_locomotion_transitions/final_closeout/
```
