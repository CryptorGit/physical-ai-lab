# exp_015_qmini_population_bodily_world_model

Stage 0–1 research infrastructure for Unitree Qmini. The research question
is whether experiences produced by teachers with different value criteria can
support a shared bodily state-transition model without exposing teacher IDs or
teacher-specific rewards.

This checkout stops at the Stage 1 gate. It contains no teacher training,
long-horizon PPO, world-model/RNN training, preference selector, distillation,
MoE, genetic algorithm, quality-diversity search, 100-teacher expansion, or
real-robot deployment.

## Source-of-truth rule

The official Unitree Qmini repository is the mechanical source of truth:

- repository: https://github.com/unitreerobotics/Qmini
- audited main commit: f6f3fef723f8bb434f9d2679dfb6053b0aca93a8
- URDF: assets/qmini_official/urdf/Qmini.urdf
- official URDF SHA-256: 4d1454510bf403fb0740a7a682fc1883ada0ecbdced844530dd98d484a618215

The vendored URDF and 11 STL files are copied byte-for-byte from that commit.
The repository license is CC BY-NC-SA 4.0; see
assets/qmini_official/LICENSE. STEP and DIY/BOM artifacts remain referenced
by URL and immutable hashes in manifests/qmini_source.json rather than being
modified or silently normalized.

The current official URDF contains exactly 10 revolute locomotion joints and
does not contain the 11th neck motor described by the README/DIY material as
reserved for expansion. This experiment therefore excludes the neck from
action and observation spaces and records the discrepancy as an explicit
source audit finding.

## Isaac Lab boundary

The repository-local asset/config lives here and does not patch the Isaac Lab
checkout. The audited simulator checkout is Isaac Lab
v3.0.0-beta2.patch1, commit
ffff603eafc6b74264a5261cc0183d6a65390d78. Import checks are split into:

1. XML/mesh contract checks that run without Isaac Sim.
2. An optional headless Isaac Lab import smoke test.

From the experiment directory, run:

    $env:PYTHONPATH = "$PWD\src"
    python scripts/validate_qmini_asset.py
    python scripts/validate_qmini_asset.py --isaac --headless

The second command requires the local Isaac Lab/Isaac Sim environment. It
does not train or deploy a policy.

## Stage 0–1 sequence

1. Audit official Qmini sources and freeze hashes.
2. Validate the 10-joint contract and import the unchanged URDF.
3. Reproduce or record the RoboTamer4Qmini reference baseline. A reference
   checkpoint is not treated as transferable to this exact URDF.
4. Run a real 50-episode Qmini WALK formal sweep before choosing any speed or
   macro action. No G1 speed values are copied.
5. Calibrate friction, wind, and left/right fatigue using measured metrics and
   freeze the ranges.
6. Save deterministic snapshots, branch same-snapshot interventions, and
   compare current-observation-only versus short-history audits.
7. Emit the Stage 1 classification and stop.

Until a real baseline table is supplied, the formal baseline, calibration,
macro-action, crossed-intervention, and memory gates remain NOT_RUN; the
verification script must therefore classify the experiment as
NO_GO_QMINI_BASELINE.

## Files

- protocol.md: executable Stage 0–1 protocol and stop rules.
- claims.md: auditable claims and evidence status.
- manifests/: frozen source, physics, actuator, baseline, split, and test contracts.
- configs/: declarative experiment settings; unknown values are literal UNKNOWN.
- src/qmini_population_bwm/: source contract, action logging, hidden physics,
  fatigue, snapshots, interventions, schema, and gates.
- scripts/: source audit, asset validation, calibration, and verification.
- tests/: contract and determinism tests.

Generated gate reports belong under
results/exp_015_qmini_population_bodily_world_model/; large immutable audit
artifacts belong under
artifacts/exp_015_qmini_population_bodily_world_model/.
