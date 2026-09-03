# SafeGaitRouter distribution package

> **Simulation-only release:** the current runtime status is
> `ADOPTED_SIMULATION_ONLY`.
> Phase mapping `7/4/4` and the three profile hashes are unchanged; profile cap
> remains `0.0125 rad` / `0.413034 rad`. The exit-only recovery is
> `0.0225 rad` / `0.403034 rad`, hold 13 ticks. Separate exact 20x30 H3 runs
> passed simulation adoption and release qualification. The formal-release
> allowlist contains only SHA-256 `95819b5b...`, and the unique portable
> package is `artifacts/router_packages/exp004-safe-gait-router-h3-release-20260808-v1`.
> Hardware remains `PROHIBITED`.

The current candidate-selection record is
`artifacts/h3_combined_candidate_5x15_seed20260808_v1.json`, SHA-256
`f040a9c6f9783b7d50dd5590389d3c81411e8f3a7fa9dd155e8ac78175d5ff56`.
All three suites passed: 15 episodes, 190 segments, and 190 acceptances. The
run audited 1,100,000 physics substeps and 1,100,000 contact samples, 110,000
control samples, and 11,000,000 leg samples. Falls, qpos, nonfinite,
applied-target-limit, desired-target-margin, slew, unauthorized applied-margin,
and route violations were all zero. The 40 applied-margin samples were exactly
the 40 authorized startup-margin-transition samples; the 110,403 pre-clip
margin samples are informational guard inputs, not safety violations. Minimum
height was `0.17911993 m` and minimum upright was
`0.9785479266972336`. Phase-entry produced 30 events, exactly 10 for each of
straight/left/right at `7/4/4`; recovery produced 15 exits and 195 active ticks.
This record selects a candidate only. It is not adoption, formal simulation
acceptance, or release evidence.

The independently pinned H3 adoption record is
`artifacts/h3_formal_candidate_pending_20x30_seed20260808_v1.json`, SHA-256
`1aea58904598cfba8ea4ef572f9473bba647eacc695f7fce3fcaa1b8646391aa`.
All three suites passed at the exact 20 episodes × 30 seconds scale: 60
episodes, 760/760 segments and 760/760 acceptances, 8,150,000 physics/contact
samples, 815,000 controls, and 81,500,000 leg-qpos samples. Falls, qpos,
nonfinite, target, desired-margin, unauthorized-margin, slew, and route
violations were zero. This hash is adoption evidence only; it is deliberately
absent from the formal-release allowlist.

The independently executed release-qualification record is
`artifacts/h3_formal_release_20x30_seed20260808_v1.json`, 18,611,839 bytes,
SHA-256
`95819b5bc1d0827a5ad779542a6f98c4aaebacf5f55a8303c0b5a14fba501674`.
It repeated the frozen 20x30 scale after adoption: 60 episodes, 760/760
segments and acceptances, 28,120 explicit checks, 8,150,000 physics/contact
samples, 815,000 controls, and 81,500,000 leg samples. Every fall, qpos,
target, slew, route, and nonfinite counter was zero. All 760 trajectory
metrics, physics-substep audits, and target-safety audits were bit-exact to the
adoption run. This is the only formal-release evidence; `f040...`, `090e...`,
and `1aea...` remain selection, safety-only, and adoption evidence respectively.

The H3 safety-only record is
`h2_aggressive_short_transition_recovery0225_hold13_20seed_v1.json`, SHA-256
`090e09cc2d82c1f42112a5f30a85cd93d940213956d6ec902fb4089875fb855a`.
It passed 500/500 safety segments and 370,000/370,000 physics/contact samples
with all fall, qpos, target, slew, route, and nonfinite counters at zero. Its
central short-horizon motion result is only 489/500 with 11 exact pinned
failures, so the source artifact remains `passed=false` / `DIAGNOSTIC_FAIL`.
It is allowlisted only as a safety component, never as adoption or release
evidence.

The older H2 records `bfaf...`, `6f65...`, and `bd7e...` remain immutable
superseded lineage. In particular, `bd7e...` validates as
`SUPERSEDED_H2_ADOPTION_LINEAGE`, with adoption/simulation/release false.
A prior phase-`6/4/4` bundle passed its 5x15 run (`8fe375ce...`) but then
failed formal 20x30 (`e975a078...`) with one straight fall and five transition
qpos failures. Both records are retained as rejected historical lineage and
cannot select or release the current H3 candidate.
The pre-promotion H3 no-diagnostic-flag wiring smoke (SHA-256
`b7612cac84b9b2d79e9ece887425e95b92131696cfa160c6b4cb51f09e971158`)
completed 38/38 segments and 31,500/31,500 physics/contact samples with all
safety and routing counters at zero. It confirmed CPU-only sessions, phase
events 6, recovery 3 exits / 39 ticks, and unchanged 55-file runtime closure.
One short-horizon reverse-right motion check failed, as expected. This is
historical wiring evidence, not the current 5x15 selection, adoption, or
release evidence.

The post-adoption no-flag screening smoke (SHA-256 `708e2fc2...`) binds all
three profiles, phase/recovery contracts, and six reverse CommandCases to
`1aea...`, while keeping `090e...` safety-only. It completed 38 segments with
all safety counters at zero; only the expected two-second reverse-right signed
progress check failed. Its screening scale and exit code 1 prevent promotion.
It is wiring evidence, not release evidence.

`package_manifest.py` bundles the existing
`SafeGaitRouter`, frozen safety contract, hardware-safe scene/reference, v22
base ONNX, formally adopted reverse profiles, the allowlisted formal evidence,
and an optional exp_004 reverse ONNX into one relocatable directory. The
current package contains no rejected reverse ONNX, so every executable route
resolves to base-v22. Every payload uses a package-relative path and SHA-256
digest.

The intended routing graph is closed and has no dynamic model lookup. Every
executable route resolves to `base_v22`. The superseded package design composed
straight reverse with `optimized_reverse_exact_safe_v1` at
`residual_scale: 0.0`; this is not a current adoption claim. A supplied reverse
ONNX is carried only as a disabled audit artifact: it cannot become an executed
route. The rejected v59/v60 lineages cannot be named by a model or reached by
any route.

The runtime command envelope is asymmetric: `vx` is clipped to
`[-0.050, +0.10]` m/s. The -0.050 reverse cap is the recalibrated physical
endpoint for the slower feedforward gait. The legacy -0.075 evidence is
historical and cannot qualify a package built against this endpoint.
H3 adopted simulation-only reverse-turn endpoints are atomic:

- left: `(-0.03, 0.0, +0.20)`
- right: `(-0.04, 0.0, -0.20)`

The evaluator router snaps these requests to the fixed
endpoint, transition through an exact stand when entering, leaving, or changing
a reverse-turn profile, and prohibit profile interpolation/action blending.

Every desired leg target is first clamped to
`[safe_lower + 0.050, safe_upper - 0.050]`, then passes through exactly one
stateful 2.0 rad/s slew, and finally receives a physical-SAFE clamp immediately
before actuator application. At the required 0.02 s control period, the slew
permits at most 0.040 rad per leg joint per tick. The limits come from packaged
`contract.json.safe_joint_limits_rad`; the 0.050 rad desired-target margin
applies to all ten leg joints. A runtime that omits, duplicates, or reorders
these stages does not satisfy the package contract. The package includes
`runtime/target_safety.py::FinalTargetSafetyGuard` as the canonical stateful
guard; its `step(desired_targets, dt)` method also overwrites all four head
targets to exact zero. Initialize it from the currently applied/reset targets,
and call it exactly once per control tick. Initialization and
reset preserve targets anywhere inside the physical SAFE bounds; desired
targets are clamped to the 0.050 rad inward margin, slewed from the real prior
applied state, and finally clamped to the physical SAFE bounds. Thus a SAFE
home target may remain outside the inward margin only during the bounded
transition into the margin; no instantaneous reset-to-margin jump is hidden.

Reset qpos has a separate guard. With zero joint noise,
`apply_reset_qpos_safety` preserves the exact physical-SAFE home pose. With any
positive joint-noise scale, it clips every leg qpos to a 0.005 rad inward SAFE
margin before simulation starts; head qpos remains exact zero.

Startup is single-policy control-first. Immediately after reset, observe the
reset state, route the command, infer the first command-policy target, and
compose the first desired joint targets. Call
`guard.control_first_startup(first_desired_targets, dt=0.02)` exactly once,
apply that guarded output to the actuators, and only then run the first physics
step and post-step sensor/audit pass. A home-only guard precharge is prohibited:
it would consume a second slew before the first policy target.

Every later tick uses the same ordering:

```text
observe / route / policy
-> clamp desired leg targets to the 0.050 rad inward margin
-> slew exactly once (2.0 rad/s, <= 0.040 rad/tick) and final physical-SAFE clamp
-> apply actuator control
-> physics
-> post-step sensor / audit
```

No physics or post-step sensor/audit pass may precede the guarded actuator
control, and the guard must not be called twice in one tick.

The checked-in backward-exit recovery contract is enabled by default for the
H3 simulation-only runtime. Profile composition for all three reverse routes uses
an extra `0.0125 rad` inward margin and upper target `0.413034 rad`. Independently,
the exit-only recovery uses an extra `0.0225 rad` inward margin and upper target
`0.403034 rad` for 13 control ticks / `0.26 s` after backward feedforward exits,
before the same one-call final guard. These are distinct manifest fields and
validator gates; equality between the profile and recovery caps is prohibited.
Re-entry cancels the remaining hold. Status is `ADOPTED_SIMULATION_ONLY`;
profile, phase-entry, recovery, and six reverse CommandCases are bound to
adoption evidence `1aea...` and, independently, safety evidence `090e...`.
Hardware remains `PROHIBITED`.

`yaw_right` applies this correction before policy inference:

```text
policy observation yaw_rate += -0.30 rad/s
requested command remains unchanged
```

The manifest and packaged `contract.json` both keep hardware deployment at
`PROHIBITED`. Packaging or verification never constitutes hardware approval.

The selected scene/model/reference and base policy match the formal evaluator's immutable
provenance: the pinned generated root, manifest/scene/model/reference hashes,
complete transitive XML/mesh/hfield closure root, and base-v22 SHA-256 for all
eight policy roles. H3 profiles are adopted evaluator defaults, and the
distinct exact-20x30 release record `95819b5b...` binds them as package inputs.

The evaluator additionally hard-gates the exp003/playground source closure,
formal WSL package/native-binary versions, ORT CPU-only sessions, and the eager
`optimized_backward_gait.json` read. Exp004 source/contract and all runtime
model/data inputs are re-hashed after evaluation; any pre/post change is a hard
error. The package manifest carries immutable hashes from the accepted
run, not hashes from an earlier diagnostic snapshot. The formal evidence loader
also requires the exact 20×30 scale, master seed `20260808`, all three suite
acceptances, all CommandCase validation statuses, pre/post provenance, eight
base-v22 policy hashes, all three profile/evidence bindings, scene/reference
hashes, formal/default phase-entry and recovery contracts, and hardware
`PROHIBITED`. The adoption allowlist contains only H3 record `1aea...`; the
formal-release evidence SHA-256 allowlist contains only `95819b5b...`. H3
selection `f040...`, safety-only component `090e...`, adoption `1aea...`, and
release `95819...` remain four distinct bindings. `6f65...` and `bd7e...` are
superseded H2 selection/adoption lineage.

## Build

The public builder requires `--formal-evidence PATH`, rejects every hash except
the frozen `95819b5b...` release record, and refuses to overwrite an existing
directory. Selection, safety-only, adoption, superseded, malformed, and
tampered records fail before an output directory is created. Build with a new
package ID and output directory:

```powershell
python experiments/mujoco/exp_004_openduckmini_safe_gait_experts/scripts/build_router_package.py build `
  --output experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/router_packages/exp004-safe-gait-router-h3-release-20260808-v1 `
  --package-id exp004-safe-gait-router-h3-release-20260808-v1 `
  --formal-evidence experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h3_formal_release_20x30_seed20260808_v1.json
```

The resulting manifest SHA-256 is
`44b64ad794d83c518ec7deeac14b18d6afcbcccafa235359c3d8075cd25334fd`.
It declares 39 files; with the manifest itself the directory contains 40 files,
with zero missing, undeclared, or hash-mismatched files. The canonical sorted
declared-file closure SHA-256 is
`40b5e15461c0685cf18119d64e3d113b300392db3d0ff7e37caccdce4a5c7837`.

An adjacent `reverse.onnx.json` produced by `export_expert_onnx.py` is detected
and verified automatically. Without it, the manifest honestly records
`sha256_only_export_report_not_supplied`. In both cases the ONNX remains
`DISABLED`, `REJECTED_NOT_ADOPTED`, and causally inert at residual scale zero.

Existing output directories are never overwritten. Build occurs in a temporary
sibling directory and is published only after all hashes, route closure,
corrections, and safety fields validate.

## Verify after copying

```powershell
python experiments/mujoco/exp_004_openduckmini_safe_gait_experts/scripts/build_router_package.py verify `
  path/to/copied/package
```
