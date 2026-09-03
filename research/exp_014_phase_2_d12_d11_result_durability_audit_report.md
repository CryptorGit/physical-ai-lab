# exp_014 Phase 2-D12 D11 result-durability forensic audit

## Finding

Main classification: **EXP014_D12_COMPLETION_LEDGER_TRANSACTION_BUG**. D11 is **R3_LEDGER_ONLY** and its formal outcome is permanently inconclusive under the original contract. No simulation, actor inference, physics step, sealed-payload deserialize, or held-out rerun occurred in D12.

The runner created all 579 result rows only in memory, then committed their episode IDs to the completion ledger at line 114. It persisted neither per-episode nor per-batch results. The only aggregate write was line 118, after the `launch_simulation` context. Context teardown called `app.close()` before that line and the process returned exit code 0 without reaching it. This violated `completed_episode_ids subset of durable_result_episode_ids` and is an experiment-infrastructure transaction-order bug, independent of policy, physics, and held-out content.

No R0/R1/R2 output, structured stdout record, journal, SQLite/WAL, atomic temporary result, or crash dump containing formal outcomes was found. Recovered formal records: 0/579. S_STOP_OMNI remains **NOT_AUTHORIZED**.

SQLite WAL with `synchronous=FULL` now atomically commits episode result, result hash, completed status, and completion event in one parent-owned transaction. Aggregate generation is a separate pure offline phase. All 16 synthetic forensic, transaction, six-point crash, resume, aggregate-reproducibility, and protection tests passed.

Commit 8846049 contains a NOT_AUTHORIZED interrupted result despite its historical commit subject.

The next experiment is preregistration of `D11R_REPLACEMENT_HELDOUT_PROTOCOL_V1`; the original held-out must never be reused and the candidate must remain S1 step 30000.
