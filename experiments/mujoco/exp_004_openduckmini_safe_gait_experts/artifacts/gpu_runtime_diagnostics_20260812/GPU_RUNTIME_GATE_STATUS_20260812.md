# GPU Runtime Gate Status — 2026-08-12

## Decision

H5 PPO training is **not authorized**.  The required OpenDuck GPU raw-parity
gate has not passed on either investigated JAX/CUDA closure.  No threshold,
batch topology, source binding, action, or hardware constraint was weakened.

## Controls that passed

| Stack | Evidence | Result |
| --- | --- | --- |
| C0: JAX 0.5.3 / MuJoCo MJX 3.11.0 | `c0v2_cold_baseline/result.json` | GPU JAX B=1/B=2 and minimal MJX contact B=1/B=2 repeated raw equality passed. |
| C1: JAX 0.6.2 / MuJoCo MJX 3.11.0 | `c1v1_cold_baseline/result.json` | The same GPU JAX and minimal MJX exact raw gates passed. |

Both controls use `cuda:0`, the canonical B=1/B=2 raw checks, and no OpenDuck
model, PPO, checkpoint, or hardware operation.

## OpenDuck gate results

| Stack / diagnostic | Result | Exact boundary |
| --- | --- | --- |
| C0 V4 fused T=1 | timeout | `warm_b1:start` after `batched_rollout` compiled. |
| C0 direct host-synchronized ladder | timeout | substep 0, `b2_first.block.start`; dispatch returned but `block_until_ready` never finished. Python SIGUSR1 stack is preserved. |
| C1 host-synchronized ladder | raw fail | All 40 direct executions (10 substeps × B1-first/B1-second/B2-first/B2-second) synchronized and returned. |
| C1 leaf-evidence repeat | raw fail | First failure at substep 0 with raw-identical inputs. |

The C1 leaf evidence is in
`c1v2_openduck_host_synchronized_ladder_leaf_evidence/result.json`, SHA-256
`3d3915f50a434ba6ef3f9843ef3e62c17ce4ce2a88314d2d17873e7d1aa5b746`.

At C1 substep 0:

- B1 same-arm run differed in `._impl.cfrc_int` and `._impl.cfrc_ext`.
- B2 same-arm run differed in `.qfrc_bias`, `.qfrc_smooth`, and `._impl.cfrc_int`.
- B1 versus the canonical B2 lane differed in 15 leaves, including `.qpos`,
  `.qvel`, `.qacc_warmstart`, `.qacc`, and force terms.

These are finite ULP-scale differences, but the contract requires literal raw
bytes to match.  They are therefore failures, not tolerance candidates.

## Causal conclusion

C1 removes the C0 B=2 synchronization noncompletion, so the JAX/PJRT closure
is causally relevant to the timeout.  It does **not** establish a viable
training stack because the strict raw reproducibility gate still fails from the
first OpenDuck physics substep.  The evidence cannot distinguish a remaining
JAX/MJX GPU issue from WSL GPU-bridge behavior without a native-Linux replay.

## Required next external check

Run the unchanged C1 host-synchronized ladder on native Linux with the same
GPU/driver class and the exact C1 package closure.  Preserve:

- JAX/JAXlib/PJRT/plugin 0.6.2, MuJoCo/MJX 3.11.0, Brax 0.14.2;
- OpenDuck joystick SHA
  `95890569d971725308b5a9c0996bfa5fd9520479f014f325e810aa1db272eb9d`;
- canonical B=2 reset, B=1 lane 1 slice, seed 20260823, zero action;
- exact raw comparisons and host-stage logging.

Native PASS would permit re-opening the V4 fused scan parity gate.  Native raw
FAIL would prove the present strict GPU gate cannot be met by this model/stack
without a justified contract-level redesign.  Native results are the only
currently meaningful next evidence; CPU substitution, XLA tuning, PPO,
checkpoints, and hardware commands are not valid substitutes.

## Safety scope

No PPO run, policy training, checkpoint creation, servo command, or real-robot
deployment was invoked in any diagnostic listed above.
