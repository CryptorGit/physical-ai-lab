# exp_013 Phase W2-P1-R1-D2 immutable dataset provenance reconciliation

## Scope and outcome

This stage performed read-only hashing, archive inspection, semantic canonicalization, split validation, and offline metric replay. It performed no training, P3 replay, rollout, DAgger, checkpoint creation, dataset serialization, or canonical promotion.

Main classification:

```text
W2_P1_STALE_HASH_MANIFEST_ACTUAL_DATASET_PROVEN
```

The original manifest is preserved as stale historical evidence. The current chunk bytes are authorized as the immutable source for a future one-time W2-P1-R1 rerun through the additive `w2_p1_dataset_hashes_resolved_v2.json` manifest.

## Hash mismatch

| Chunk | Original expected SHA-256 | Resolved actual SHA-256 |
|---|---|---|
| `stop_recovery_chunk_002.pt` | `cc4bfe6757a01dcc05fe9721f17c597651e9de9969431090bc5ed9959872c8a1` | `04975de086383e1c7c436db076c2ef529efa5af5428a3a5b2eb70dfb9672156b` |
| `stop_recovery_chunk_003.pt` | `7e345d9c3ecc24e07c75174d9202f65c3f17bc476c918849ddf27c04a54b760a` | `ec413b90018a8faa5375d7421cb49f99c36f94bc7bad4c225af4f0820fe7b0a1` |

The actual hashes were already recorded at both the beginning and end of W2-P1-D1 and remained the P3 input bytes. W2-P1-R1 detected the mismatch but did not modify the files.

## Hash generation and chronology

The original generator uses `hashlib.sha256(path.read_bytes()).hexdigest()` over the complete binary file. Paths are obtained from the lexicographically sorted `RAW.glob("*_chunk_*.pt")`; the path string is not part of the hash, symlinks receive normal file-open semantics, and no text/newline conversion is involved. No algorithm, binary-mode, glob-index, or wrong-directory capture bug was found.

The lower-precedence filesystem chronology is nevertheless decisive when combined with later cryptographic and metric evidence:

```text
manifest capture                         2026-07-31 21:59:04 +09:00
chunk 002 final on-disk write            2026-07-31 22:00:00 +09:00
chunk 003 final on-disk write            2026-07-31 22:06:28 +09:00
selected W2-P1 student final write       2026-07-31 22:14:53 +09:00
W2-P1 commit cae97ad                     2026-07-31 22:17:45 +09:00
W2-P1-D1 protected analysis be6fb47      2026-08-01 01:30:40 +09:00
W2-P1-R1 identity gate 1058584           2026-08-01 04:40:11 +09:00
```

Thus the manifest captured earlier byte versions of chunks 002/003 and was not refreshed after their final rewrite/reserialization. The raw chunks were ignored/untracked; the committed manifests, protected hashes, reports, and metrics provide the durable lineage.

## File-copy and semantic audit

The complete expected hashes were searched in repository text, git objects, results, reports, logs, source, and untracked run artifacts. Exact target filenames and all plausible dataset-size `.pt` files across the accessible workspace were byte-hashed. No expected-hash copy was found; only the two actual files matched the actual hashes. Consequently, byte-different/semantic-identical comparison against the expected bytes is not claimable and is explicitly recorded as `EXPECTED_COPY_NOT_FOUND`.

Both actual files are PyTorch ZIP archives. Their canonical whole semantic hashes are:

| Chunk | Whole semantic hash | Tensor semantic hash | Metadata semantic hash |
|---|---|---|---|
| 002 | `dbafb5d6cc702c7279dd386885ac2db39b6df12db93412eb720e7d90ec037c7c` | `d6183b1adf29c3a1f7ed947fe83feaf5f00ac142be380157add1b1015ddea08d` | `86d57597d99715c15132d93a2fa1ec3d1f0671eb0c037cd663b07f637d7a8efc` |
| 003 | `7bb05280920cb07eadf699f744b26e4a97519593fe5a14f767c6928063d005a5` | `0f5a8ef99106df0f1f7e486e1c93d6c4af0cfe6a5d2b295ae036e47081ba2be2` | `b88bf49ea447fe594877e19189d9b4003ef7919fcdc6ebaec124b69aef6a15c7` |

Canonicalization hashes tensor key path, dtype, shape, and contiguous logical values; mapping keys are sorted, sequences remain ordered, and floats use IEEE binary representation. No chunk was re-saved.

## Dataset, episodes, samples, and split

All 13 W2-P1 chunks contain 20,663 accepted episodes. Observed group counts match the committed manifest exactly after applying its documented loader alias: 600 `FORWARD_ANCHOR` episodes are included in the reported 3,800 `ZERO_YAW_TRANSLATION` episodes. Stop recovery contains 7,200 episodes.

Affected chunks 002/003 contain 3,600 episodes and 378,000 ordered samples. Their aggregate ordered sample identity is `adbf4fd157542ebe602dc75980013779813c6c51179140c87b99e95b9b0e056d`; per-episode hashes are stored in the CSV artifact. Schema, episode ranges, conditions, phases, label sources, tensor shapes, and counts pass.

The immutable split has no overlap, unknown episode, or missing episode. Its file SHA-256 is `1b19d73d68aa4cdab63c21dfc6f2602c352a88bb5758dcaadcf5f447e47b8b51`; canonical membership hash is `db238795ce761f11306e95a3ebf02fe25ee23f7fe1f9bd28e0904d70dc0e7add`.

## Metric fingerprints

Using the existing selected step-20,000 student and the actual chunks, all original W2-P1 held-out metrics, cosines, and 10,000-sample counts reproduce with zero recorded difference:

| Group | Recomputed MSE |
|---|---:|
| stop recovery | 0.0006019577849656343 |
| steady stop | 0.000014700874999107327 |
| zero-yaw translation | 0.00005135194078320637 |
| pure yaw | 0.00004602097760653123 |
| moving turns | 0.00003612534419517033 |
| independence | 0.00003375818414497189 |
| dynamic yaw endpoints | 0.00004105953485122882 |
| start retention | 0.0012912879465147853 |

The W2-P1-D1 fingerprint also reproduces exactly: start p95 `0.00010496831964701414`, top-1% loss contribution `0.5384848713874817`, and exact-zero loss share `0.9760376811027527`. Every difference is `0.0`, within the required `1e-8`.

This proves that the actual chunks are consistent with the content used for the saved original W2-P1 held-out result and D1 diagnosis.

## P3 provenance

D1 start/end protected hashes resolve every P3 source path to the current actual byte hashes, including chunks 002/003. The split hash matches, and the fixed training-pool sample index/content hashes were reconstructed from sampler seed `20276049` without replaying P3 optimization. D1 probe seed was `20277717`. An expected-hash alternative cannot have been used at D1 because its protected input hashes are the actual hashes.

## Resolution and authorization

`w2_p1_dataset_hashes.json` remains unchanged with status `PRESERVED_STALE`. The additive `w2_p1_dataset_identity_resolution.json` and `w2_p1_dataset_hashes_resolved_v2.json` record actual byte hashes, semantic hashes, split and sample identity, both metric fingerprints, and classification. Resolved status is `IMMUTABLE_RESOLVED_SOURCE_OF_TRUTH`.

The next-stage authorization gate is PASS because schema/count, split, original metrics, D1 metrics, P3 input provenance, fixed byte hashes, and fixed semantic hashes all pass. This stage authorizes—but does not execute—a single W2-P1-R1 group-balanced supervised integration rerun using the resolved immutable manifest.

## Protection audit

Dataset chunks, labels, split, original manifest, checkpoints, and optimizer artifacts are byte-identical from stage start to end. P3 replay, student training, closed-loop evaluation, DAgger, and canonical promotion are all zero. The canonical parent remains W1B-R2 iteration 200; the stop teacher positive control remains PASS; no student was created or promoted. No remote push was performed.
