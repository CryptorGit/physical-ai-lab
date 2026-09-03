# Exp 011 Go2 bidirectional baseline report

## Go2 baseline discovery

The live Gymnasium registry contained four standard Go2 tasks:
`Isaac-Velocity-{Flat,Rough}-Unitree-Go2-{v0,Play-v0}`. The selected task was
`Isaac-Velocity-Flat-Unitree-Go2-v0`; no task ID was guessed or introduced.

The baseline was selected before formal results by selection-rule rank 1: the
official Isaac Lab RSL-RL Go2 flat-velocity checkpoint. Its SHA-256 is
`32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0`,
saved iteration is 999, and documented training command range is
`vx, vy, yaw-rate ∈ [-1, 1]`. Thus 1.2–2.5 m/s evaluation is explicitly OOD.

Strict loading passed. The deterministic actor contract is:

- observation: 48D, ordered as base linear velocity (3), base angular velocity
  (3), projected gravity (3), velocity command (3), relative joint position
  (12), relative joint velocity (12), previous action (12);
- network: 48 → 128 → 128 → 128 → 12, ELU, diagonal Gaussian with no
  observation normalizer; evaluation uses its deterministic mean;
- action: 12D default-offset joint-position action, scale 0.25;
- joint order: all FL/FR/RL/RR hip joints, then thigh joints, then calf joints.

The flat PhysX environment uses physics dt 0.005 s, decimation 4, control dt
0.020 s, contact-sensor dt 0.005 s, 20 s default timeout, plane terrain with
static/dynamic friction 1.0/1.0, and base-contact fall termination. Go2 uses
23.5 Nm DC-motor limits with 25.0 stiffness and 0.5 damping in the asset
configuration.

## Steady-state map

Fifty deterministic 8-second episodes were evaluated at every command with the
frozen seed set 20260901–20260950.

Zero-command STAND failed: hold success was 86%, fall was 14%, mean forward
speed was -0.001 m/s, aggregate speed p95 was 0.238 m/s, heading-drift p95 was
1.918 rad, and long-dwell saturation was 0%. Diagnostic gait counts were 27
STAND, 9 WALK_LIKE, 7 IRREGULAR, and 7 FALL.

| Command (m/s) | Actual mean | Mean abs. error | Fall | Dominant gait count | Status |
|---:|---:|---:|---:|---|---|
| 0.0 | -0.001 | 0.015 | 14% | STAND 27/50 | STAND FAIL |
| 0.4 | 0.325 | 0.092 | 0% | IRREGULAR 48/50 | PARTIAL |
| 0.6 | 0.609 | 0.052 | 0% | IRREGULAR 50/50 | PARTIAL |
| 0.8 | 0.816 | 0.062 | 0% | IRREGULAR 48/50 | PARTIAL |
| 1.0 | 1.017 | 0.077 | 0% | IRREGULAR 50/50 | PARTIAL |
| 1.2 | 1.215 | 0.087 | 0% | IRREGULAR 50/50 | PARTIAL |
| 1.5 | 1.480 | 0.082 | 0% | IRREGULAR 46/50 | PARTIAL |
| 2.0 | 1.850 | 0.154 | 0% | WALK_LIKE 31/50 | PARTIAL |
| 2.5 | 2.130 | 0.370 | 0% | WALK_LIKE 37/50 | UNSUPPORTED |

The speed tracking error gate alone is met through 2.0 m/s, but the complete
steady gate is not: contact-foot slip is the dominant failure, with additional
heading failures at some endpoints. No nonzero test point is `SUPPORTED`, so
the formal continuous support envelope is empty. The 2.5 m/s result is retained
and reported; it is not hidden or relabeled.

Gait names above are diagnostic classifications from measured four-foot contact
traces. They are not used as the primary success gate.

## Bidirectional transitions

Because neither endpoint of any planned pair was steady-state `SUPPORTED`, zero
transitions were formally gate-eligible. The predeclared pairs were nevertheless
run as diagnostics with the same policy and a 1.5-second minimum-jerk command
ramp. There was no policy/hidden-state/previous-action reset inside an episode.

| Direction | Completion | Fall | Acquisition | Target hold | Formal |
|---|---:|---:|---:|---:|---|
| STAND→1.2 | 90% | 8% | 92% | 90% | No |
| 1.2→2.0 | 100% | 0% | 100% | 100% | No |
| 2.0→1.2 | 100% | 0% | 100% | 100% | No |
| 1.2→STAND | 98% | 2% | 98% | 98% | No |
| 1.2→2.5 | 24% | 0% | 24% | 50% | No |
| 2.5→1.2 | 100% | 0% | 100% | 100% | No |

Acceleration and deceleration are not averaged. The diagnostics do not isolate
a formal deceleration asymmetry: 1.2→2.0 and 2.0→1.2 both acquired their targets
in all episodes, while the dominant blockers were steady endpoint safety,
zero-command hold, and the 2.5 m/s acquisition limit. Contact/gait hysteresis
remains diagnostic-only because the endpoint acceptance precondition failed.

## Full sequence

The 2.5 m/s full sequence was not run because 0.0, 0.6, 1.2, 2.0, and 2.5 did
not all pass their required steady gates. The nominally reduced 2.0 m/s sequence
was also not run because its 0.0, 0.6, 1.2, and 2.0 endpoints were not all
supported. This is a fail-closed outcome, not a sequence failure and not a
successful limited sequence. Unsupported command executions and checkpoint
switches are both zero.

The GUI wiring was exercised with a short non-claim Stand recording. Tracking
camera, visual-only 1 m/5 m floor guides, lane boundaries, live telemetry
overlay, and MP4 generation completed. The MP4 is diagnostic and excluded from
the commit.

## Classification

`GO2_STEADY_STATE_ENVELOPE_INSUFFICIENT`

No nonzero tested speed passed every formal steady-state gate, including the
major 2.0 m/s endpoint. Zero-command hold independently failed as well. The
problem therefore cannot yet be classified as a formal transition or
deceleration-asymmetry failure.

## Next

Train a new continuous 0–2.0 m/s Go2 base policy.

This is the sole recommended next method. It must remain one continuous
speed-conditioned policy; this Stage does not authorize experts, switching, a
transition controller, residuals, or reward patching to the frozen baseline.

## Repository

Starting repository HEAD was
`a72f556744d51c41a003eea43180fc0b41c72897`, not the previously reported
`ff15a94ff168b9948ba4d2e3ee49b0fd57735ebd`. Existing exp_006, OpenDuck,
artifact, and media dirty state was preserved and excluded from staging.
Experiments exp_005 through exp_010, capability manifests, production
artifacts, the selected checkpoint, and Isaac Lab tracked core were not
modified. Teacher/policy gradients, PPO optimizer updates, and reward
optimization were zero. No remote push was performed.

