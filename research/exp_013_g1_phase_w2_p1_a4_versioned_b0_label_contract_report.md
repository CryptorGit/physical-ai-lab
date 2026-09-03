# exp_013 Phase W2-P1-A4 — versioned B0 label-contract preflight

## Outcome

**VERSIONED_B0_TWO_NONZERO_STEPS_INSUFFICIENT**

`StartBoundaryLabelContractV2` removes the static stop/start contradiction at B0. The selected in-memory step-500 probe passes B0, B1, B2, stop recovery, steady stop, every moving subgroup, and start-nonboundary. It nevertheless fails the preregistered matched-state physical start gate. No held-out fallback, persistent policy checkpoint, formal closed-loop evaluation, or promotion was performed.

## Versioned label overlay

- Base: resolved immutable W2-P1 dataset, unchanged.
- Overlay: exactly 2,373 B0 labels; train/validation/held-out = 1,893/240/240.
- B0: exp_012 Stage 2Q stop-maintenance mean action.
- B1/B2/B3+: unchanged W1B-R2 mean action.
- Overlay SHA-256: `c6bc9c68b70ea119f919e3e3b7d9b3734b8d33127915559160b0b1fd59ab553b`.

## Static validation

| Group | MSE | Cosine |
|---|---:|---:|
| B0 V2 | 0.0000023186 | 0.9999991417 |
| B1 | 0.0000075637 | 0.9999992371 |
| B2 | 0.0000067534 | 0.9999990463 |
| stop recovery | 0.0007361590 | 0.9998790923 |
| steady stop | 0.0000096721 | 0.9999932183 |
| start nonboundary | 0.0000148296 | 0.9999978090 |

V2 changes B0-vs-steady gradient cosine from the V1 conflict (-0.998 at its closest candidate) to +0.990 at the selected V2 probe; B0-vs-stop-recovery is +0.444. The semantic conflict is therefore resolved statically.

## Matched physical start

| Metric | Candidate | canonical/W1B 2-step reference | Difference |
|---|---:|---:|---:|
| endpoint | 93.6875% | 99.4167% | -5.73 pp |
| acquisition | 89.8542% | 91.3958% | -1.54 pp |
| fall | 5.4375% | — | — |
| dangerous slip | 4.6042% | — | — |

The worst condition endpoint gap is 47.5 pp and the worst acquisition gap is 39.5 pp. Direction 270°, yaw 0 has 50.0% endpoint and 49.0% fall. Thus saved-state B1/B2 imitation does not guarantee the candidate follows the W1B trajectory on its own post-B0 state.

## Sequence controls

- all-stop B0/B1/B2: endpoint 89.44%, fall 9.54%.
- uninterrupted W1B B0/B1/B2: endpoint 99.60%, fall 0.40%.
- stop/stop/W1B: endpoint 92.96%, fall 6.33%.
- stop/W1B/stop: endpoint 89.67%, fall 9.21%.

The evidence supports a continuous whole-body W1B start sequence; merely assigning correct static labels at B1/B2 is insufficient for on-policy state coverage.

## Protection and authorization

Base dataset, labels, split, manifests, existing checkpoints, and optimizers remain byte-identical. The only new dataset artifact is the versioned B0 overlay. Zero-command retention and held-out authorization were not run because validation physical start failed. Formal closed-loop authorization remains denied, and the canonical parent remains W1B-R2 iteration 200.
