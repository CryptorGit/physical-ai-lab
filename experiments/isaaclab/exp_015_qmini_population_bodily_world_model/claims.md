# Claims ledger

| ID | Claim | Evidence | Status |
|---|---|---|---|
| C015-001 | The official Qmini main source is fixed at commit f6f3fef723f8bb434f9d2679dfb6053b0aca93a8. | manifests/qmini_source.json, audit script output | PASS |
| C015-002 | The official current URDF has ten revolute locomotion joints in the frozen order. | Official URDF hash and XML parser | PASS |
| C015-003 | The official current URDF has no neck joint or transmission; the 11th physical motor is a source-level expansion mismatch. | README/DIY audit plus URDF parse | PASS |
| C015-004 | Official Qmini link mass/inertia and geometry are preserved without G1 substitution. | manifests/physics_contract.json, vendored URDF/STL hashes | PASS |
| C015-005 | 8010 nominal torque, velocity, reduction, motor model, and official PD values are not published in the audited Qmini repository. | Source audit | UNKNOWN by design |
| C015-006 | RoboTamer4Qmini is a legacy Isaac Gym reference with a different URDF and action/observation contract. | manifests/baseline_policy_contract.json | PASS |
| C015-007 | A Qmini WALK baseline passes the formal 50-episode gate. | Results table required | NOT_RUN |
| C015-008 | Hidden physics ranges are relevant and safe. | Calibration result required | NOT_RUN |
| C015-009 | Same-snapshot same-action replay is deterministic. | Unit test and backend replay result | PASS for pure contract; Isaac backend NOT_RUN |
| C015-010 | At least three measured macro actions separate at 1/5/10/25 steps. | Crossed intervention result required | NOT_RUN |
| C015-011 | Short history is necessary for every hidden factor. | Memory necessity table required | NOT_RUN |
| C015-012 | Canonical transitions exclude teacher ID and teacher-specific reward from model input. | data_schema.py and tests | PASS |

NOT_RUN is not a success claim. The Stage 1 verification result remains
NO_GO_QMINI_BASELINE until a real Qmini baseline table is supplied.
