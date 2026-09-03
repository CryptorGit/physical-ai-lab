# exp_013 Phase W2-P1-R1 group-balanced stop integration

## Outcome

Classification: `EXP013_W2_P1_R1_DATASET_IDENTITY_FAIL`. The formal run stopped at the first immutable-input gate. No P3 replay, persistent student training, closed-loop rollout, DAgger, checkpoint creation, or promotion was executed.

## Dataset identity

The preregistered source of truth was `w2_p1_dataset_hashes.json`. Two existing chunks differ:

- `results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_practical_stop_endpoint_acquisition/raw/stop_recovery_chunk_002.pt`: expected `cc4bfe6757a01dcc05fe9721f17c597651e9de9969431090bc5ed9959872c8a1`, actual `04975de086383e1c7c436db076c2ef529efa5af5428a3a5b2eb70dfb9672156b`. Actual matches D1 baseline: `True`.
- `results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_practical_stop_endpoint_acquisition/raw/stop_recovery_chunk_003.pt`: expected `7e345d9c3ecc24e07c75174d9202f65c3f17bc476c918849ddf27c04a54b760a`, actual `ec413b90018a8faa5375d7421cb49f99c36f94bc7bad4c225af4f0820fe7b0a1`. Actual matches D1 baseline: `True`.

The current bytes match the hashes captured at both the start and end of W2-P1-D1, so this run did not modify them. Nevertheless, R1 explicitly requires agreement with the existing W2-P1 hash manifest, and that condition is false. Treating the later D1 audit as a replacement source would silently change the requested provenance contract, so the run failed closed.

## P3 and training

The committed D1 probe source contains a complete P3 contract (Adam, LR 2e-4, seed 20277717, 2,000 steps, clip 10, fixed pool/validation seeds, 25/25/25/25 objective). It was not executed because dataset identity is an earlier hard gate. No formal checkpoint exists.

## Closed loop and DAgger

Not authorized. All static-stop, moving-retention, transition, safety, and symmetry outputs explicitly record `NOT_EXECUTED_UPSTREAM_DATASET_IDENTITY_FAIL`.

## Canonical artifact

W1B-R2 iteration 200 (`61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`) remains canonical.

## Next

One method only: reconcile the immutable W2-P1 dataset hash manifest against the D1-protected chunks without changing dataset or label bytes, then request R1 authorization again.

## Protection

All current dataset bytes remained unchanged during R1; no existing checkpoint, optimizer, stage, sampler, reward, physics, calibration, evaluator, Isaac Lab core, or RSL-RL package was changed. Remote push was not performed.
