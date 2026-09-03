# exp_014 Phase 2-D13 D11R preregistration report

## Result

Classification: **EXP014_D13_SEED_DERIVATION_MISMATCH**. The fixed seed gate failed before episode generation, overlap audit, seal, durability preflight, or D14 authorization.

The specified five ordered inputs, concatenated as canonical delimiter-free UTF-8 and processed with the D12 preregistered rule `int(first_8_hex(SHA-256), 16) mod 2147483647`, produce digest `0fbb49afc26045db5b9e6bd994cfa218f8efe7dcce8dd09aa4a37a6e6d212761` and seed `263932335`, not the fixed value `1940027935`. Repeating the computation gives the same result bitwise. The fixed value `1940027935` is reproduced only by the earlier three-input D12 draft that omitted the D12 commit SHA and protocol literal.

No replacement episode was generated. The required `.bin` filename is a zero-byte tombstone and explicitly not a sealed payload. Access count is zero. D14 is **NOT_AUTHORIZED**. The fixed seed, candidate, and condition contract were not changed.

Original D11 remains `PERMANENTLY_INCONCLUSIVE_UNDER_ORIGINAL_CONTRACT`; D11 and D12 classifications are unchanged. No Isaac Lab import, simulation, physics, actor inference, outcome access, training, or remote push occurred.
