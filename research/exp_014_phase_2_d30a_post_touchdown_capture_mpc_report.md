# EXP014 Phase 2-D30A post-touchdown capture MPC

## Classification

`EXP014_D30A_LOCAL_CAPTURE_MODEL_INVALID`

## Local dynamics

- Runtime: `C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe`; fresh physics branches: `3`.
- Model: `D30ALocalDynamicsV1` with LEFT/RIGHT × early/late bins.
- Hold-out one-step max error: `1.4900354628698338`.
- Hold-out three-step max error: `1.0240549363756202`.
- Normalized three-step error: `0.979200331851726`.
- Required gates: `{'control_response_sign_agreement': 0.9, 'three_step_normalized_max_error': 0.3, 'velocity_yaw_sign_agreement': 0.95}`; observed gates: `{'control_response_sign': False, 'one_step': False, 'three_step': False, 'three_step_normalized': False, 'velocity_yaw_sign': False}`.
- Feature-group errors: `{'com_relative_support': {'max_abs_error': 0.39865565483770304, 'mean_abs_error': 0.13757028479110242}, 'dcm_relative_support': {'max_abs_error': 0.43076621436337703, 'mean_abs_error': 0.1111471513850844}, 'projected_gravity': {'max_abs_error': 0.06414662342722632, 'mean_abs_error': 0.01596224509452729}, 'velocity': {'max_abs_error': 1.0240549363756202, 'mean_abs_error': 0.32585125787321084}, 'yaw': {'max_abs_error': 0.6766118701608193, 'mean_abs_error': 0.16897603764963137}}`.

## Baseline

Route A reused the D29B `S_HOLD -> W_MOVE` lifecycle and exact frozen actors.
Baseline available: `True`; baseline gate: `False`.
The baseline ledger covers R0-R7 and preserves first-failure decomposition.
D29C Route A progression reference: `{'available': True, 'progression': {'L0_liftoff': {'count': 8, 'total': 8}, 'L1_touchdown': {'count': 8, 'total': 8}, 'L2_wmove_neighborhood_crossed': {'count': 0, 'total': 8}, 'L3_multiple_alternating_contacts': {'count': 8, 'total': 8}, 'L4_stable_limit_cycle_captured': {'count': 0, 'total': 8}, 'L5_100_step_retention': {'count': 0, 'total': 8}}, 'return_map': {'classifications': {'CONTRACTING': 2, 'DIVERGING': 6, 'UNAVAILABLE': 0}, 'median_phase_conditioned_distance': 15690420.95176733, 'median_same_side_ratio': 0.7369794862356073, 'reading': 'DIVERGING_OR_MIXED', 'rows': 48}, 'route': 'A_CONTINUE_WMOVE', 'source': 'D29C route_level_progression.json', 'source_count': 8}`.

## Capture MPC

The canonical controller is `D30AFiniteHorizonBoundedLQRMPCV1`: 16-step bounded LQR, at most
40 steps after TD0, then a hard W_MOVE switch and 100-step retention.
It was not executed because the local-model hold-out gates failed; no positive
MPC physics result is claimed. Capture result available: `False`.

## Stable capture

Stable capture requires finite state/action, all required model gates, safe
handoff, and 100-step W_MOVE retention. This run is not eligible
for stable-capture promotion.

## Handoff

Handoff available: `False`; pass: `False`.
Retention available: `False`; pass: `False`.

## Failure decomposition

- `LOCAL_CAPTURE_MODEL_VALIDATION_FAILED`

## Recommended next action

nonlinear short-horizon trajectory optimization

## Repository

- Starting HEAD: `6665455444dbfdc9e5ca9433f8974c5820be6fad`; ending HEAD: `6665455444dbfdc9e5ca9433f8974c5820be6fad`.
- Pre-existing worktree status is filtered to exclude D30A paths.
- D26T phase tube: `WMove03PhaseTubeV1`, 50 LEFT / 50 RIGHT references with
  explicit phase/reference fields.
- Protected hashes unchanged: see `protected_hashes.json`.
- Reproduction command: `reproduction_commands.ps1`.

Artifacts are under `results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d30a_post_touchdown_capture_mpc`.
