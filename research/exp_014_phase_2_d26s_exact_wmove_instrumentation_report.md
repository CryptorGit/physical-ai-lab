# Phase 2-D26S exact exp013 instrumentation

- Classification: **EXP014_D26S_PASSIVE_CAPTURE_PARITY_PASS_REFERENCE_READY**
- Starting HEAD: `0ed51ce49c42fb83bb126a85e9f4d4346a6a15dd`
- Original evaluator: `evaluate_w1b.py`, checkpoint SHA `61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`

## Passive parity

Capture OFF/ON used independent fresh processes and seed `20274021`. Pre-physics hashes and 202 before/after-step trace points were bitwise identical; hook mutation counters are all zero.

## Formal W_MOVE

Capture-ON formal reproduction: 100/100 success, forward vector error 0.08217118 m/s (original 0.08217118), fall/slip/impact/long-dwell saturation all zero.

## Native collection

Collection index 0 used the original lifecycle, direct `[0.3, 0, 0]` command and seed `20274021`. The durable bundle contains 20000 identity-complete steady states and 27734 phase events; SHA-256 is `e4f2250a35a5feee2d1adb415d11121e52164018648bc7678dcf91a47e0894f6`.

## Foot mapping

Runtime mapping resolves sensor indices [6, 13] to robot body indices [24, 25]; D26 numeric sole geometry is reused read-only.

## Contact phases

Strict E0 events: L=6961, R=6906, alternation=1.000000, mean interval=7.3288 steps. Hysteretic event counts match. The selected source is `E0_STRICT_TOUCHDOWN` and gait classification is `ALTERNATING_TOUCHDOWN_WALK`.

## Entry reference

Event+2..6 candidate populations and physical medoids are recorded. Fresh replay validation of 50 references per side was not executed in this passive-capture stage, so no bilateral D27 authorization is issued. D26's non-native 59-state bundle remains quarantined.

## Offline plans

The fixed 432-plan ledger is present but marked NOT_EXECUTED pending medoid replay validation; no physics, WBIK plan execution, PPO, CEM, or checkpoint was run.

## Repository

D26/D26R and protected paths are unchanged; persistent updates 0; remote push false.
