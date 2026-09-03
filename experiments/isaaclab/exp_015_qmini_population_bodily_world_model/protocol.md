# Stage 0–1 protocol

## Scope and exclusions

The sole embodiment is Unitree Qmini. No G1 checkpoint, G1 observation,
action, reward, torque limit, velocity threshold, or PD value may enter this
experiment. Isaac Lab upstream is read-only.

No teacher population training, long PPO, world-model training, recurrent
model scaling, preference selection, student distillation, MoE, GA, QD,
100-teacher extension, or real-hardware policy deployment is allowed.

## Stage 0A — source and physics contract

Run scripts/audit_qmini_sources.py against the official Qmini checkout.
Freeze the commit, URDF hash, mesh hashes, STEP hashes, DIY PDF hash, joint
order, limits, link inertias, collision/visual meshes, and source findings.

Values absent from official Qmini sources remain UNKNOWN. RoboTamer values
are permitted only as explicitly labelled ROBOTAMER_REFERENCE and are never
renamed as official Qmini specifications.

The contract must distinguish:

- URDF joint effort/velocity limits from 8010 motor nominal limits.
- official mechanical data from simulator/importer choices.
- official source values from RoboTamer reference values.
- a provisional root height from an official physical specification.

## Stage 0B — baseline locomotion

Audit the public RoboTamer4Qmini simulator, observation, action, reward,
decimation, domain randomization, ONNX export, and SDK deployment path.
The reference uses legacy Isaac Gym and a different URDF/schema, so a public
checkpoint is a reference baseline only unless exact asset/environment
compatibility is proven.

For Isaac Lab, establish only a short Qmini WALK baseline sufficient to
generate calibration data. The minimal evaluation records velocity tracking,
upright/fall, torque, action rate, slip, contact, joint-limit violations,
power, and mechanical work. A missing baseline is a hard NO_GO.

## Formal WALK gate

Use at least 50 episodes with fixed evaluation seeds and a measured command
velocity sweep. Pass requires:

- fall rate at or below 2%;
- finite state on 100% of episodes;
- recorded tracking error, slip, torque, saturation dwell, contact pattern,
  orientation, power, and mechanical work.

The safe speed range is measured from these data. No speed values from G1 are
valid defaults for Qmini. Only after this gate may three measured macro
actions be constructed.

## Hidden physics

The hidden factors are floor friction, lateral/head wind, left actuator
fatigue, and right actuator fatigue. The normal policy observation excludes
their ground truth. The analysis record may retain ground truth with an
explicit analysis-only field.

Fatigue is per locomotion actuator and uses:

f[t+1] = clip(f[t] + alpha * P[t] - beta * f[t], 0, 1)

with effectiveness:

eta[t] = 1 - c * f[t].

P[t] is a measured dimensionless power input after a calibration reference
is frozen. No G1 or unverified 8010 threshold is used.

Calibrate ranges until the worst condition has baseline fall at or below 10%,
each factor affects at least one physical metric, and no factor is
unobservable or permanently destabilizing. Freeze the result before
counterfactual collection.

## Snapshot and intervention

Each snapshot stores root pose/velocity, ten q/dq values, actuator/controller
state, previous actually applied action, current command, contact-related
state, friction, wind, both fatigue ledgers, RNG state, episode time, and any
baseline recurrent state.

For one snapshot, clone independently and apply at least three measured
macro actions at 1, 5, 10, and 25 policy steps. Store state delta, progress,
velocity/error, energy/work, torque RMS/peak, saturation dwell, impact, slip,
contact, fatigue delta, and stability/fall.

## Memory necessity

Compare:

- A: current observation only;
- B: short observation/action history.

Evaluate friction, wind, left fatigue, and right fatigue. If history does not
show a preregistered, reproducible improvement, emit
NO_GO_MEMORY_NECESSITY; do not enlarge an RNN to manufacture a difference.

## Stage 1 stop gate

The required gate keys are:

qmini_source_hash, qmini_joint_contract, isaaclab_asset_import,
baseline_walk_formal, source_protected_write, snapshot_deterministic_replay,
hidden_factor_relevance, memory_necessity, crossed_action_separation,
worst_hidden_fall, fatigue_ledger, action_proposed_applied_logging,
canonical_schema, train_dev_test_contract, and failure_taxonomy.

After emitting the classification, stop. A PASS authorizes only the next
research review; it does not authorize teacher training, world-model
training, MoE, GA, QD, or deployment.
