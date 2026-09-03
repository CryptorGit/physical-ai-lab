# exp_007_unitree_g1_walk_centered_transitions

`exp_007` is an independent experiment. It does not extend the `exp_006`
dispatcher in place: `exp_006` established isolated command skills but left the
running and standing controller families disconnected. This experiment freezes a
walk-centered transition specification without changing any formal `exp_005` or
`exp_006` artifact.

## Hypothesis and state model

The central hypothesis is that `RUN -> WALK -> STAND` is more robust than a
direct `RUN -> STAND` transition: the physically meaningful intermediate gait
should reduce falls, joint saturation, and heading drift and generalize better
over contact phase, entry velocity, and command timing.

`STAND` is the home state because it is the safe idle posture and the origin for
posture commands. `WALK` is the hub because it overlaps kinematically with both
standing and low running without claiming a direct 5 m/s stop.

Persistent states:

```text
STAND <-> WALK <-> RUN_LOW
STAND <-> CROUCH_HOLD
```

Candidate transition states are `STAND_TO_WALK`, `WALK_TO_STAND`,
`WALK_TO_RUN`, `RUN_TO_WALK`, `CROUCH_DOWN`, and `CROUCH_UP`. Persistent and
transition states are separate. In Stage 0 every transition is either
`FORMAL_EVALUATION_PENDING` or `NOT_IMPLEMENTED_STAGE_0`; none is a PASS.

TURN is not a discrete state. It is a heading command over WALK or RUN:
`target_speed_mps`, `target_heading_w_rad`, and
`target_yaw_rate_radps`. This avoids inventing a separate controller state for
ordinary curved locomotion.

## Frozen expert provenance

- STAND/WALK: immutable `exp_005` Stage 2 `model_4246.pt`, 123-D observation,
  37-D normalized position action.
- RUN/RUN_TURN: immutable `exp_006` candidate A `model_0.pt`, 123 legacy state
  columns plus 29 command columns, 37-D final action.
- Candidate B `model_31.pt`: formal `command_system_v1` provenance and direct
  STOP baseline only. Stage 0 requires A/B RUN and TURN routes to be bitwise
  equal before selecting the simpler candidate A.
- CROUCH_SHALLOW: reference-only `scripted_shallow_v1`, 0.08--0.10 m. It is
  neither copied nor recalibrated.

The adapters are one-way:

```text
canonical policy-relative robot state + MotionCommand
    -> expert-specific observation
```

They do not add absolute world XY and do not force the WALK expert into a 152-D
interface. The transition bridge is a non-trainable, disconnected interface
marker that returns `NOT_IMPLEMENTED_STAGE_0`.

## Stage 0 and gate

Stage 0 covers repository provenance, checkpoint hashes, actor architecture,
observation layouts, fixed-input action equivalence, physical/action
compatibility, contract validation, and individual GUI playback. It performs no
PPO training, fine-tuning, bridge learning, cross-expert switching, or formal
transition evaluation.

Run the deterministic audit:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_experts.ps1
```

Outputs are written to:

```text
results/exp_007_unitree_g1_walk_centered_transitions/stage0_expert_audit/
```

The gate fails closed if any required checkpoint/artifact/hash, actor interface,
37-joint order, action semantics/scale, simulation timing, PD or limit
compatibility, finite adapter output, fixed action equivalence, weight
immutability, or protected repository check fails. Warnings remain distinct from
failures.

**STAND↔WALKが正式合格する前に、WALK↔RUN評価やbridge学習へ進まない。**

## Individual GUI playback

All launchers resolve the repository root and checkpoint from
`expert_manifest.json`, create exactly one environment, print actor/checkpoint/
command at startup, and use a world-orientation-fixed camera. No playback
switches between WALK and RUN in one episode.

STAND:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_walk_expert.ps1 -Mode STAND
```

WALK:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_walk_expert.ps1 -Mode WALK -Speed 1.5
```

RUN:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_run_expert.ps1 -Speed 2.6
```

RUN with heading command:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_run_expert.ps1 -Speed 2.6 -TurnDegrees 90 -Direction Left
```

The RUN player prints actual world heading and accumulated yaw. The camera
transform is set once in world coordinates and never follows robot yaw, so a
90-degree course change remains visible.

## Planned order

```text
Stage 0 expert provenance
Stage 1 STAND inheritance confirmation
Stage 2 STAND <-> WALK
Stage 3 WALK_TURN
Stage 4 WALK/RUN overlap audit
Stage 5 WALK -> RUN
Stage 6 RUN -> WALK
Stage 7 STOP reconstruction
```

Stage 0 PASS permits only Stage 1. A pivot is required if the inherited
STAND/WALK expert cannot preserve stable zero-command standing, if expert action
semantics or timing differ, or if a safe WALK overlap cannot be demonstrated.
Training and transition evaluation must not proceed while a compatibility
failure remains.

## v1 exclusions and limitations

`CROUCH_DEEP`, `STEP_OVER`, `LAND`, fall recovery, hardware transfer,
`TURN_IN_PLACE`, and direct stopping from 5 m/s are outside v1. Skills marked
`NOT_SUPPORTED` by `exp_006` are not reopened. The 2 cm standing drop observation
is not a LAND skill.

Stage 0 does not establish STAND↔WALK or WALK↔RUN performance, does not provide a
production router, and does not train a bridge. Artifact plans use:

```text
logs/rsl_rl/physical_ai_g1_walk_centered
results/exp_007_unitree_g1_walk_centered_transitions
artifacts/exp_007_unitree_g1_walk_centered_transitions
```

## Stage 1 — STAND home state

Stage 1 formally certifies the immutable Stage 2 `model_4246.pt` as the
`STAND` home-state expert. The production equation is exactly:

```text
STAND action = frozen Stage 2 expert action
```

RUN, RUN residual, transition bridge, CROUCH/STEP_OVER/LAND, and every other
scripted contribution are bitwise zero. No router or expert switch is invoked.
The 50-episode formal run uses the same task, seed, reset distribution, and
checkpoint as the exp_006 STAND reference, with a stricter 8-second hold instead
of 6 seconds.

Formal evaluation:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_stand.ps1
```

GUI playback:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_stand.ps1
```

The PASS artifact contains references and hashes only; it never contains a
checkpoint:

```text
artifacts/exp_007_unitree_g1_walk_centered_transitions/stand_home_state_v1/
```

Stage 1 PASS certifies only STAND. `STAND_TO_WALK` and `WALK_TO_STAND` remain
`FORMAL_EVALUATION_PENDING`; WALK/RUN transitions remain unimplemented.

## Stage 2 — STAND ↔ WALK audit

Stage 2 tested a single frozen Stage 2 expert with only its legacy forward
velocity command changed. RUN, bridge, and scripted contributions remained
bitwise zero. The controller uses a fixed 2.0-second minimum-jerk ramp and the
eight guarded phases in `stand_walk_controller.py`.

The 50-episode formal result is **FAIL**: no speed category reached 90% full
sequence completion, so no supported WALK speed range is published. In
particular, 0.3 m/s did not initiate walking, while higher speeds showed
heading drift and saturation. `STAND_TO_WALK` and `WALK_TO_STAND` therefore
remain pending and no transition artifact or production-router connection is
created. WALK↔RUN work must not begin from this result.

Reproduce the diagnostic, pilot, and formal runs:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_stand_walk.ps1 -Mode all
```

Play one audited command in the GUI (diagnostic only):

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_stand_walk.ps1 -Speed 1.2
```

The next decision is to consider a STAND/WALK-specific expert initialized from
the Stage 2 checkpoint. This README records that pivot; it does not authorize
or perform training.

## Stage 2B — WALK operating-envelope audit

Stage 2B preserves Stage 2A as a committed negative baseline and audits a
world-heading hold outside the frozen actor:

```text
yaw_rate_cmd = clamp(
    K_heading * wrap(target_heading - current_heading)
    - K_yaw_rate * current_yaw_rate,
    +/- yaw_rate_limit
)
```

The limited candidate audit selected `K_heading=1.25`, `K_yaw_rate=0.10`, and
`yaw_rate_limit=0.5 rad/s` as the least-bad diagnostic controller. It is not a
TURN command and absolute world XY is not added to the policy observation.

The sweep found a command dead zone through 0.3 m/s, isolated steps at
0.4–0.5 m/s, and one safe diagnostic point at 0.6 m/s. Commands from 0.8 m/s
upward were saturation-dominated, principally at the ankle-pitch joints. A
single safe point is not a continuous operating range, so the transition pilot
was skipped and no Stage 2C capability candidate was published.

The fixed-heading comparison reduced cross-track drift at some speeds but
increased yaw activity, action rate, slip, and fall risk at 1.0 m/s. Stage 2B is
therefore classified `HEADING_CONTROLLER_UNSTABLE`; a dedicated STAND/WALK
expert should be considered before WALK/RUN work.

Reproduction:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_operating_envelope.ps1 -Mode all -Seed 20260725
```

Stage 2A playback remains the default `ZeroYaw` mode. Fixed-heading diagnostic
playback is explicit:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_stand_walk.ps1 -Speed 1.2 -HeadingMode FixedTarget
```

## Stage 2R — unified STAND/WALK retraining

Stage 2R tested a unified STAND/WALK actor warm-started from Stage 2
`model_4246`. The actor and critic were loaded strictly and the optimizer was
reset. The parent exploration standard deviation was reset once to a trainable
0.25 before R1 because its audited maximum was 1.421.

A bounded heading preflight selected a smoothed fixed-world-heading controller:

```text
K_heading = 0.8
K_yaw_rate = 0.10
yaw_rate_limit = 0.3 rad/s
low_pass_alpha = 0.15
slew_limit = 0.01 rad/s per control step
```

R0 passed its wiring-only check. Two 1024-environment, 100-update R1 pilots
were then run. WALK-capable checkpoints reached 96.25–97.5% diagnostic
steady-WALK success, but every such checkpoint produced non-zero flight during
the protected STAND evaluation. The second pilot also raised STAND fall rate
to 4% at its best WALK checkpoint.

```text
Stage 2R = NO_GO_RETRAIN
R2 / R3 / R4 / formal = NOT RUN
supported WALK range = none
artifact = not created
capability manifest = unchanged
```

The retained checkpoint is diagnostic only:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_unified_stand_walk.ps1 `
  -Speed 1.0
```

Playback prints `NO_GO_RETRAIN`, uses a world-orientation-fixed camera, and
does not load the RUN expert or transition bridge. The gate and exact
reproduction commands are under
`results/exp_007_unitree_g1_walk_centered_transitions/stage2r_unified_stand_walk/`.

## Formal strategy revision: modular locomotion state graph

Stage 2A, Stage 2B, and Stage 2R reject only the shared-weight design in which
one expert must maintain STAND and WALK and perform both directed transitions.
They do not reject exp_007. The experiment is now defined as an integrated
runtime composed of specialized models:

```text
External Command
  -> Command Planner
  -> State Graph Path
  -> Current Controller State
  -> Expert Router
  -> Steady-State Expert or Transition Expert
  -> Safety Gate
  -> Final Action
```

The word "integrated" describes this complete execution unit, not one neural
network. STAND, WALK, RUN, and CROUCH_HOLD are independent steady-state
experts. STAND_TO_WALK, WALK_TO_STAND, WALK_TO_RUN, RUN_TO_WALK,
STAND_TO_CROUCH, and CROUCH_TO_STAND are independent directed edges.

The router uses hard controller switching. It may use blending or residuals
inside a transition controller only when boundary continuity requires it; it
does not continuously mix model parameters. A target steady-state expert may
be activated only after the active transition's completion contract passes.
Unsupported edges fail closed and keep the current controller active.

States and edges are registered through `transition_graph.json`,
`state_contracts.json`, and `transition_contracts.json`. The Python graph,
planner, and router consume these registrations, so adding a future movement
requires a state model, incoming/outgoing directed edges, and contracts rather
than another central routing `if` chain.

The revised roadmap is:

```text
Stage 0      Expert provenance                                  COMPLETE
Stage 1      STAND steady-state expert                          PASS
Stage 2A/B/R Shared STAND/WALK feasibility                      NO_GO
Stage 2W     Independent WALK steady-state expert
Stage 3      STAND_TO_WALK transition expert
Stage 4      WALK_TO_STAND transition expert
Stage 5      STAND -> WALK -> STAND integration
Stage 6      RUN steady-state expert formalization
Stage 7      WALK_TO_RUN transition expert
Stage 8      RUN_TO_WALK transition expert
Stage 9      Graph-based STOP: RUN -> WALK -> STAND
Stage 10     TURN integration
Stage 11     CROUCH integration
Stage 12     Integrated motion sequence
Stage 13     OOD transition timing and initial-phase evaluation
Stage 14     Limited robustness
```

Stage 2W evaluates WALK maintenance only. It must not gate on zero-command
STAND, stopping, starting, RUN, or any transition behavior.

## Stage 2W result: independent WALK expert

Stage 2W compared the Stage 2R pilot-1 `model_50` parent against the frozen
Stage 2 `model_4246` using identical seeds and steady-WALK-only criteria.
`model_4246` was selected because it had higher initial WALK success (15%
versus 10%) and lower heading error; STAND performance was deliberately
excluded from parent selection.

Two bounded 1024-environment pilots were run. Pilot 1 used 150 updates and its
`model_150` passed the 40-episode pilot screen. A 50-episode formal evaluation
then failed:

```text
overall WALK success       86%       (required >=95%)
0.6 m/s success            92.31%    (required >=90%)
0.8 m/s success            92.31%    (required >=90%)
1.0 m/s success            75%       (required >=90%)
1.2 m/s success            83.33%    (required >=90%)
fall                       0%
heading error p95          0.182207 rad (required <=0.12)
speed error mean           0.168992 m/s
path-drift failure         4%
long-dwell saturation      0%
dangerous slip             0%
excessive flight           0%
```

Pilot 2 increased heading, lateral, and cross-track reward weights for 100
updates, but degraded pilot success to 80% with heading p95 0.279 rad. The
branch was rejected and no further search was run.

```text
Stage 2W = FAIL
supported WALK range = none
Stage 3 eligibility = false
artifact = not created
WALK capability = FORMAL_EVALUATION_PENDING
```

The best checkpoint is retained only as a diagnostic candidate. It is not
registered as `walk_steady_state_expert_v1`:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_walk_steady_state.ps1 `
  -Speed 1.0
```

The player shows `STATE: WALK`, `TRANSITION: NONE`, and
`STAGE2W_STATUS=FORMAL_FAIL`; it does not load RUN or a transition expert.
Stage 3 must not begin until WALK initialization, reward shaping, or low-level
heading control is redesigned and Stage 2W passes formal evaluation.

## Stage 2W-B result: WALK direction stabilization

Stage 2W-B preserved the modular state graph and investigated only the
independent WALK steady-state expert. The seven Stage 2W failure episodes were
replayed from the unchanged formal seed. Six heading failures accumulated
during the command ramp, entered steady hold above 0.12 rad, and subsequently
recovered. They were not caused by sustained controller saturation or
high-frequency command reversal.

The fixed-policy controller comparison used the same seed and eight episodes
per speed:

```text
ZeroYaw:             0% success, heading p95 2.890 rad
Current FixedTarget: 87.5% success, heading p95 0.158 rad
Lower bandwidth:     50% success, heading p95 0.631 rad
```

This established `POLICY_RESPONSE_DOMINATED`: zero yaw command permits severe
open-loop drift, while lower-bandwidth feedback cannot arrest it. Two bounded
1024-environment pilots were therefore run. Training retained the 123-D
observation and 37-D position action, replayed speeds with weights
20/20/30/30%, and exposed the policy to small smooth 0.08–0.15 Hz heading
corrections. Pilot 2 changed only the yaw-rate oscillation penalty from -0.02
to -0.05.

The selected Pilot 2 `model_100` passed a new 50-episode full-range formal
evaluation:

```text
overall success              96%
0.6 / 0.8 / 1.0 / 1.2 m/s   100% / 92.31% / 91.67% / 100%
fall                         0%
heading error p95            0.087342 rad
speed error mean             0.038195 m/s
path-drift failure           2%
long-dwell saturation        0%
dangerous slip               0%
excessive flight             0%
```

```text
Stage 2W-B = FULL_PASS
supported WALK commands = 0.6 / 0.8 / 1.0 / 1.2 m/s
artifact = walk_steady_state_expert_v1
Stage 3 eligibility = true
```

WALK_LOW was not evaluated because the full range passed. Unsupported speeds
are rejected, never clamped. STAND_TO_WALK and WALK_TO_STAND remain distinct
`NOT_IMPLEMENTED` transition edges.

## Stage 3 result: directional STAND_TO_WALK edge

Stage 3 preloads three separately owned controllers and uses hard switching:

```text
frozen model_4246
→ independent stand_to_walk_transition_v1
→ frozen walk_steady_state_expert_v1
```

The source and target experts were hash-protected throughout. Two bounded
1024-environment, 100-update pilots were run from a strict WALK warm-start.
Later checkpoints degraded transition acquisition, so they were rejected.
RSL-RL's first post-update auto-save, `model_0` (one effective PPO update),
was the best independent checkpoint in the frozen sweep.

The direct STAND-to-WALK hard-switch baseline completed the kinematic
acquisition but had 50% ankle long-dwell saturation failures and only 50% full
edge success. The selected edge uses a 1.3–1.7 s minimum-jerk command ramp and
the independent transition checkpoint, with no runtime action blend.

The 50-episode formal evaluation passed:

```text
source STAND settle          100%
transition completion        100%
WALK takeover                100%
full edge success             98%
fall                           0%
heading error p95          0.09073 rad
long-dwell saturation          2%
dangerous slip                 0%
entry / exit discontinuity     0% / 0%
```

Only the four evaluated targets `0.6`, `0.8`, `1.0`, and `1.2 m/s` are
supported. No continuous speed interval is claimed. `WALK_TO_STAND` remains
`NOT_IMPLEMENTED` in the immutable Stage 3 result; Stage 3 alone did not
imply stop capability.

## Stage 4 result: directional WALK_TO_STAND edge

Stage 4 first evaluated a parameter-free direct hard switch. It passed at
0.6/0.8 m/s but failed at 1.2 m/s with falls, ankle saturation, reverse
motion, and timeout. Its 32-episode full-edge success was 81.25%, so it was
not registered.

One bounded 1024-environment, 100-update pilot was then run from a strict
STAND `model_4246` warm-start. The independent edge owns every transition
action; WALK and STAND weights remain frozen and runtime action blending is
not used. The selected first post-update `model_0` uses a 1.6 s minimum-jerk
deceleration command. Later checkpoints degraded and were rejected. Pilot 2
was unnecessary because the selected checkpoint passed every pilot gate.

The 50-episode formal evaluation passed:

```text
source WALK hold             100%
transition completion        100%
STAND takeover / hold        100% / 100%
full edge success            100%
fall / slip / saturation       0% / 0% / 0%
reverse motion / flight        0% / 0%
heading error p95          0.08335 rad
final speed mean / p95     0.00647 / 0.01306 m/s
entry / exit jump p95      2.0048 / 0.1208
```

Only discrete source commands `0.6`, `0.8`, `1.0`, and `1.2 m/s` are
supported. Stage 4 does not claim intermediate speeds or any RUN transition.
With both directed edges now formalized, Stage 5 may evaluate the integrated
`STAND -> WALK -> STAND` route without changing any expert.

GUI playback:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_walk_to_stand.ps1 `
  -Speed 1.0
```

Unsupported source speeds are rejected by PowerShell validation and are never
clamped.

## Stage 5 result: integrated round trip

Stage 5 connects the four frozen PASS modules through
`modular_state_graph_v1`. External `WALK(speed)` and `STOP` commands are
decomposed into manifest edges by the command planner. The router advances its
cursor only after the owning completion detector passes. Exactly one
controller supplies each applied action; runtime action blending is absent.
The previous-action columns always contain the preceding global applied
action, including all four switch boundaries.

The frozen 50-episode formal evaluation did not pass every gate:

```text
full sequence                    47/50 (94%)
all segment retention rates             98%
fall                                     2%
long-dwell saturation                    6%
WALK heading p95                   0.05860 rad
final speed p95                    0.01297 m/s
final double support                     98%
```

The 0.6 m/s category completed 11/13 (84.6%), below its 90% category
requirement, and accounted for the only fall plus two of the three
long-dwell saturation failures. The other categories achieved at least 92.3%
full-sequence success. Router errors, controller overlap, unsupported-command
misexecution, and stale previous-action observations were all zero.

The separate 20-episode, three-cycle diagnostic completed every cycle. It
found no state leakage, route-cursor reset error, completion-history reset
error, or previous-action mismatch. Mean cumulative cross-track drift was
0.0837 m. This diagnostic does not override the frozen single-cycle formal
FAIL. No integration artifact or capability entry is created, and Stage 6 is
not authorized.

Formal reproduction:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_stand_walk_stand.ps1 `
  -Mode formal `
  -Seed 20260901 `
  -Output ".\results\exp_007_unitree_g1_walk_centered_transitions\stage5_stand_walk_stand_integration"
```

GUI playback:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_stand_walk_stand.ps1 `
  -Speed 1.0 `
  -Cycles 3
```

## Stage 5D result: 0.6 m/s failure diagnosis

Stage 5D is diagnostic-only. It changes no checkpoint, router contract, gate,
capability, or production artifact. The frozen Stage 5 seed reproduced the
same three failures exactly:

```text
episode 1   0.6 m/s   initial STAND failure
episode 10  0.6 m/s   long-dwell saturation
episode 22  0.8 m/s   long-dwell saturation
```

The first physical abnormality in all three episodes occurs during
`INITIAL_STAND` under `stage2_model_4246`, before any WALK command or switch
boundary. The critical signal is the frozen aggregate left/right ankle-pitch
effort dwell. The dominant side can alternate while the per-step maximum
remains above 95%. Previous-action mismatches remain zero.

Matched controlled diagnostics found:

```text
STAND-only, 100 episodes        fall 3%, saturation 5%
0.6 forward edge, 50 episodes  completion 100%, saturation 12%
0.8 forward control            edge saturation 0%
0.6 reverse edge               completion 100%, reverse-edge saturation 0%
0.6 full integration           all segments 100%, safety success 92%
```

The direct Stage 5 failure cause is classified as
`STAND_BASELINE_VARIANCE`. A secondary, independently reproducible
`STAND_TO_WALK_0P6_FRAGILITY` remains because the same ankle-pitch saturation
appears during the 0.6 forward edge, consistent with Stage 3's 1/13 event.
`WALK_TO_STAND_0P6_FRAGILITY`, causal entry-distribution shift, boundary
saturation, and router/state-contract bugs are excluded.

The next recommended work is preregistered multi-seed confirmation of the
STAND variance plus a local 0.6 forward-edge decision. Stage 5D itself does
not remove 0.6 support or authorize further training.

## Stage 5E result: state-contract-conditioned integration

Stage 5E separates system startup from the locomotion graph:

```text
startup diagnostic:
RESET -> UNINITIALIZED -> valid STAND source state

production state graph:
STAND -> STAND_TO_WALK -> WALK -> WALK_TO_STAND -> STAND
```

`UNINITIALIZED` is not a steady-state capability. A graph route starts only
after the frozen STAND controller has produced horizontal speed at most
0.08 m/s, vertical speed at most 0.05 m/s, roll/pitch at most 0.10 rad,
double support, finite observations/actions, and no flight, torso contact,
fall, or long-dwell saturation continuously for 0.4 seconds. Timeout never
forces promotion to STAND. `RESET_TO_STAND` remains `NOT_IMPLEMENTED`;
startup recovery is diagnostic-only.

The preregistered evaluation completed 4 seeds x 50 startup episodes,
3 seeds x 60 conditioned round trips, and 3 seeds x 50 separate 0.6 m/s
forward-edge episodes. Startup reached a valid STAND contract in 199/200
episodes; one episode was rejected after a fall with ankle saturation, and
reset itself had double support in 0/200. The conditioned
0.8/1.0/1.2 m/s main evaluation completed 180/180 round trips with no fall,
long-dwell saturation, routing error, controller overlap, or previous-action
mismatch. The 0.6 m/s confirmation completed 144/150 safety-qualified edges:
completion and WALK takeover were 150/150, fall was 0%, and long-dwell
saturation was 4%. Therefore Stage 5E is `GRAPH_INTEGRATION_PASS` and retains
the discrete supported commands 0.6/0.8/1.0/1.2 m/s.

This result does not replace the Stage 5 reset-inclusive FAIL. Both claims
are retained:

```text
Stage 5   FAIL under reset-inclusive protocol
Stage 5E  PASS under state-contract-conditioned protocol
```

GUI playback with the source-state contract:

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_stand_walk_stand.ps1 `
  -Speed 1.0 `
  -RequireValidStandContract
```

Exact headless commands are saved in
`results/exp_007_unitree_g1_walk_centered_transitions/stage5e_state_conditioned_confirmation/reproduction_commands.ps1`.
## Stage 6 — independent RUN_LOW steady state

Stage 6 freezes exp_006 candidate A as a RUN-only steady-state expert. It does
not claim reset-to-RUN acquisition, TURN, STOP, or either WALK/RUN edge. The
policy retains the 123-dimensional legacy robot observation plus the
29-dimensional command-system extension, 37 normalized position actions, and
action scale 0.5. Straight-line heading uses the formal exp_006 legacy mapping
`clamp(1.5 * heading_error, -1.5, 1.5)`; TURN fields remain zero.

The four pre-registered points were audited independently. Formal evaluation
classified the expert `PARTIAL_PASS`: 2.4, 2.6, and 2.8 m/s passed; 3.0 m/s
failed the frozen heading gate and is rejected rather than clamped.
`WALK_TO_RUN` and `RUN_TO_WALK` remain `NOT_IMPLEMENTED`.

```powershell
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_run_low_steady_state.ps1 -Speed 2.6
```
## Stage 7 — WALK_TO_RUN directional edge

Stage 7 kept WALK at the normalized 1.2 m/s source contract and audited hard
switches into the frozen RUN_LOW expert at 2.4, 2.6, and 2.8 m/s. The direct
switch failed its pre-registered pilot gate (58.6% full-edge success); a
1.4-second minimum-jerk command profile also failed (25.9%). No production
edge was registered. A valid learned retry requires a dedicated trainable
152-dimensional `G1CommandResidualActor` action term initialized from real
frozen-WALK occupancy. The existing 123-dimensional Stage 3/4 task was not
misused for that purpose. `RUN_TO_WALK` remains unimplemented.
## Stage 7R — dedicated 152D transition infrastructure

Stage 7R selected Route A: the existing 29D command contract already exposes
RUN identity, previous skill, heading, local target, elapsed/remaining time,
phase, and transition progress without redefining any field. The dedicated
`WalkToRunTransitionActor152` strict-copies the RUN actor and permits gradients
only through the RUN command encoder, RUN state adapter, and RUN residual
head. `WalkToRunTransitionAction` enforces the global previous-action contract
and produces one 37D action with scale 0.5 and no runtime blending.

Static gradient, frozen-gradient, finite-output, and checkpoint/optimizer
save-reload checks passed. R0 nevertheless failed because valid WALK occupancy
is not yet connected to a transition-only PPO rollout/advantage buffer.
Consequently no PPO pilot, formal evaluation, artifact, or capability update
was performed.
## Stage 7R2 — transition-only PPO storage

Stage 7R2 implements cohort-scoped storage that cannot contain source
preparation steps. Segment-local GAE, prefix-reward contamination, and source
duration invariance tests pass, with zero invalid stored steps. R0 remains
failed because live Isaac Sim physical-state handoff from asynchronously
prepared WALK environments into a fixed 512-environment ready cohort is not
implemented or end-to-end verified. No PPO pilot or formal claim was made.

## Final closeout — asymmetric state graph

Later Stage 7R work completed live in-place transition-only PPO and established
a limited formal `WALK_TO_RUN` edge:

```text
WALK@1.2 -> RUN_LOW@2.6 / 2.8
```

`WALK_TO_RUN@2.4` remains unsupported and is never clamped. The immutable
RUN_LOW steady-state expert can still maintain 2.4, 2.6, and 2.8 m/s.

The reverse direction did not succeed. The Stage 8A parameter-free switch
reached only 46.7% and 40.0% full-edge success from 2.6 and 2.8 m/s,
respectively, with saturation above 50%. The learned Stage 8C controller
terminated the RUN cycle and reached WALK-compatible contact in every
deterministic evaluation, but never maintained the 0.40-second WALK contract:

```text
RUN-cycle termination       100%
WALK-compatible contact     100%
WALK contract                 0%
completion / takeover         0%
timeout                     100%
```

Stage 8D reproduced this result over 80 valid-source episodes. Every episode
reached exactly seven WALK-valid steps (0.14 seconds) against the required 20
steps. Contact broke the streak 1,539 times; speed broke it four times. The
WALK-acquisition-progress reward fired in all near-contract episodes, but its
correlation with the invariant seven-step maximum was zero. Its required
reachability gate therefore stopped the weight-only Pilot 2 before optimization.

The formal closure is:

```text
exp_007                         PARTIAL_SUCCESS_ASYMMETRIC_STATE_GRAPH
RUN_TO_WALK v1                  RUN_TO_WALK_V1_NO_GO
primary failure                 TARGET_WALK_BASIN_RETENTION_FAILURE
supporting failure              WALK_ACQUISITION_REWARD_NOT_REACHABLE
graph-based STOP                BLOCKED_BY_RUN_TO_WALK
full bidirectional graph        NOT_ACHIEVED
```

The achieved graph is:

```text
STAND
  ↕
WALK
  └── WALK_TO_RUN ──→ RUN_LOW
```

No RUN_TO_WALK production artifact exists, and Stage 8C/8D checkpoints are not
production checkpoints. Stage 9 STOP and the full integrated sequence must not
proceed through this missing edge.

The recommended next project is
`exp_008_phase_aware_locomotion_transitions`: first use the preserved Stage 8D
trajectories to test whether the next contact break and corrective action are
predictable from the existing 152D observation. This separates an
actor/reward-optimization problem from partial observability before adding
contact/history features, recurrence, or a phase estimator.
