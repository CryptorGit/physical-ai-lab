# exp_012 Stage 2G — Event-Stratified On-Policy Minibatch Preflight

## Outcome

**EVENT_STRATIFIED_ON_POLICY_NO_EFFECT**. No multiplier was selected. Phase B remains
**PHASE_B_NOT_READY**. This stage performed no persistent policy update and wrote
no checkpoint.

## Data and contract

The frozen Phase A iteration-50 actor (`4edbb595e28e24dc09cf39e8245c7be1b1bebf792798a73af2e562075d0fe952`) generated one fresh
20-second, 1,024-environment S100 batch: **1,024,000 samples**, **69
completion events**, and **67 unique completion episodes**.
This passed the preregistered 64-event/32-episode stop gate. All yaw commands
were zero, external controllers were off, and all samples came from this one
on-policy checkpoint.

Mutually exclusive strata contained **1,245 E2**,
**64,102 E1**,
**2,433 unsafe**, and
**956,220 background** samples. Event windows span
takeoff minus 10 steps through landing plus 5 steps and are merged within an
episode. Matched backgrounds control target speed, episode time, command
segment, contact phase, and preceding flight state.

Reward values were unchanged. The effective update objective was nevertheless
reweighted by sampling. In M4/M8/M16 every minibatch retained 20% unsafe,
at least 10% precursor, and at least 30% background samples.

## Gradient amplification

| condition | completion/total | combined·completion cosine | combined·unsafe cosine |
|---|---:|---:|---:|
| M0 | 13.983% | +0.082 | +0.149 |
| M4 | 10.847% | -0.082 | +0.882 |
| M8 | 11.492% | -0.019 | +0.897 |
| M16 | 10.112% | -0.259 | +0.922 |

The magnitude ratio exceeded 1% in every condition, but stratification did not
amplify the *direction*: all stratified combined gradients point weakly against
the completion component. The mandatory unsafe quota dominates the effective
objective. Layer/joint localization remains locomotion-relevant—torso, bilateral
hip, knee, and ankle terms dominate; arm/hand parameters do not explain the
failure.

## Shadow stability

| condition | exact KL | max-step KL | clip | ratio p99 | mean shift |
|---|---:|---:|---:|---:|---:|
| M0 | 0.01286 | 0.02327 | 0.189 | 1.487 | 0.0610 |
| M4 | 0.01611 | 0.02190 | 0.235 | 1.521 | 0.0552 |
| M8 | 0.01341 | 0.02589 | 0.193 | 1.466 | 0.0518 |
| M16 | 0.01949 | 0.02757 | 0.267 | 1.597 | 0.0513 |

All four disposable updates passed the numerical hard gate with finite
parameters, critic gradients below 1e6, and value losses below 1e8. Stability
was therefore not the blocker. Adam updates remained nearly orthogonal to the
completion component (cosines +0.016,
+0.017, and
-0.003).

## Immediate cross-effect

Relative completion-window loss changed by
**+2.25% (M4)**,
**+1.94% (M8)**,
and **+3.93%
(M16)**: all worsened. Unsafe-window loss worsened by
**+6.13%**,
**+8.63%**,
and **+6.35%**,
all beyond the 5% gate. Background loss stayed within 5%.

## Temporary behavior and retention

No condition produced deterministic completion. At 2.4 m/s, no condition
produced a deterministic PERIODIC_RUNNING success, so the required +10-point
gain was absent. Across all five S100 speeds the completion counts were
M0=6, M4=4,
M8=6, and
M16=1; none reached the required
twofold increase over M0. M4/M8/M16 changed 2.4 m/s fall relative to M0 by
+20.0%,
+13.3%, and
+13.3%.

STAND, WALK 0.6, and WALK 1.2 remained 100% in all temporary clones.
WALK_TO_STAND remained 100% except M4 at 95%, exactly the allowed five-point
boundary. This limited retention preservation does not offset the absent RUN
effect and unsafe cross-effect failure.

## Fresh-process reproducibility

M8 was repeated from a second fresh on-policy collection with a different
diagnostic seed. Its full combined-gradient cosine to the primary run was
**-0.096**, below the required 0.80.
Both processes failed to improve completion-window loss and failed the
behavioral gate; the reproduction also remained within the KL/clip hard gate.
Thus numerical stability reproduced, but the proposed update direction did not.

## Decision

The event-stratified sampler route is closed for this construction. The next
single method is **completion-event short-horizon replay preflight**. That method
is not executed here. Phase B remains not ready.

## Protection

Stage 2G changed no reward, curriculum, network, observation/action contract,
physics, Isaac Lab/RSL-RL core, formal checkpoint, or optimizer state. All
shadow parameters were disposable. Production policy updates: **0**. Remote
push: **false**. Pre-existing unrelated dirty paths were preserved.
