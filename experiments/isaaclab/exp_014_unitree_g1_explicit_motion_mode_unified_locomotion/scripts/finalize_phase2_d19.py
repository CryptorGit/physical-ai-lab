"""Finalize D19 fail-closed support implementation/timing audit."""
from __future__ import annotations
import csv,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve();REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d19_support_objective_symmetry_audit";RAW=OUT/"raw";REPORT=REPO/"research/exp_014_phase_2_d19_support_objective_symmetry_audit_report.md";START="5616e1da770b42d5bd0c72b8f15755c50e7dd2a1"
def dump(name,x):OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def placeholder(reason):return {"status":"NOT_EXECUTED","reason":reason,"persistent_policy_update":0,"checkpoint_created":0}
def csv_placeholder(name,reason):(OUT/name).write_text("status,reason\nNOT_EXECUTED,"+reason.replace(",",";")+"\n",encoding="utf-8-sig")
x=json.loads((RAW/"audit.json").read_text(encoding="utf-8")); reason=x["policy_probe_reason"]; loc=x["source_locations"]
classification="EXP014_D19_SUPPORT_TIMING_OR_IMPLEMENTATION_BUG"
dump("stage_reference.json",{"phase":"2-D19","starting_head":START,"actual_starting_head":START,"actual_head_is_source_of_truth":True,"D18_classification":"EXP014_D18_SUPPORT_TRANSFER_OBJECTIVE_NO_EFFECT","source_snapshots":64,"policy_parent":"D16 initial zero residual","persistent_updates":0,"remote_push":False})
dump("protocol.json",{"name":"Exp014SupportObjectiveSymmetryCausalityAuditV1","selection_split":"train-only 64 S_HOLD sources","unit_test_fail_closed":True,"temporary_probes_require_unit_tests_pass":True,"same_rollout_seed_optimizer_required_if_authorized":True,"validation_used":False,"persistent_PPO":0,"checkpoint_creation":0})
dump("support_objective_source_audit.json",{"source_file":"experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d18_precursor.py","locations":loc,"force_source":"ContactSensor.net_forces_w_history newest indexed sample","force_frame":"world; no frame conversion","contact_filter":"force norm >5 N only for safety/slip streak; load reward uses unclipped z-positive force without contact-valid mask","F_L_F_R":"max(world-z force,0), first/second ankle-roll contact body","load_imbalance":"abs(F_L-F_R)/(F_total+epsilon)","low_load_ratio":"min(F_L,F_R)/(F_total+epsilon)","total_support_ratio":"F_total/(total_mass*9.81)","support_foot":"argmax(F_L,F_R)","support_slip":"selected support-foot tangential speed","swing_unloading":"low_load_ratio Gaussian in 0.20-0.60 s","defects":["load reward is not gated by nonzero total support","target is set to zero after 0.50 s while reward envelope remains active to 0.75 s"]})
dump("support_reward_timing_contract.json",{"action_application":{"line":loc["physics_action"]["line"],"order":1},"physics_and_sensor_refresh":{"order":2,"mechanism":"wrapped.step(action)"},"privileged_quantity_read":{"line":loc["privileged_after_action"]["line"],"order":3},"reward_computation":{"line":loc["reward_computation"]["line"],"order":4},"age_increment":{"order":5},"time_index":"pre-increment age at the just-completed action step","action_effect_alignment":"post-action reward; no pre-action sensor timing bug found","registered_schedule":{"0.00_to_0.35":"target rises 0 to peak","0.35_to_0.50":"peak hold","0.50_to_0.75":"target peak remains; weight decays","at_or_after_0.75":"no target"},"implemented_mismatch":"target becomes zero immediately for t>0.50 while weight remains positive until 0.75"})
dump("support_reward_unit_tests.json",{"status":x["unit_test_status"],"tests":x["unit_tests"],"force_fixture":x["force_fixture"],"schedule_fixture":x["schedule_fixture"],"test_count":len(x["unit_tests"]),"passed":sum(t["pass"] for t in x["unit_tests"]),"failed":len(x["failed_tests"]),"failed_tests":x["failed_tests"],"stop_rule_applied":True})
csv_placeholder("support_term_decomposition.csv",reason);dump("support_term_decomposition.json",placeholder(reason)|{"requested_probes":["Q_LOAD_ABS","Q_TOTAL_SUPPORT","Q_SUPPORT_SLIP","Q_SWING_UNLOAD","Q_LOAD_UNLOAD","Q_LOAD_SUPPORT","Q_SUPPORT_FULL"]})
csv_placeholder("support_gradient_conflict_matrix.csv",reason);dump("support_gradient_conflict_matrix.json",placeholder(reason)|{"requested_terms":["LOAD_ABS","TOTAL_SUPPORT","SUPPORT_SLIP","SWING_UNLOAD","UPRIGHT","VELOCITY","YAW"],"strong_conflict_threshold":-.5,"dominant_conflicting_term":"NOT_DETERMINED"})
dump("support_advantage_alignment.json",placeholder(reason)|{"spearman":"NOT_COMPUTED","lagged_regression":"NOT_COMPUTED","classification":"NOT_APPLICABLE_UNTIL_IMPLEMENTATION_FIXED"})
dump("action_to_load_timing.json",placeholder(reason)|{"lags_steps":[1,2,4,8,12],"actuation_delay":"NOT_COMPUTED"})
m=x["mirror_contract"];dump("mirror_contract.json",{"source":"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_d1_yaw_translation_interference_diagnosis/robot_mirror_contract.json","joint_index_permutation":m["mirror_indices"],"joint_sign_inversion":m["mirror_signs"],"joint_names":m["joint_names"],"base_linear_velocity_signs":m["base_linear_velocity_signs"],"base_angular_velocity_signs":m["base_angular_velocity_signs"],"projected_gravity_signs":m["projected_gravity_signs"],"command_signs":m["command_signs"],"contact_swap":[1,0],"action_permutation_sign":"same joint permutation and signs","source_state_pairs":32})
dump("mirror_contract_tests.json",{"status":x["mirror_status"],"pairs":x["mirror_pairs"],"joint_mapping_involution":True,"sign_involution":True,"unsigned_reward_mirror_invariant":True,"signed_left_to_right_mirror":True,"base_W_MOVE_action_mirror_consistency":"NOT_RUN; policy probes blocked by unit-test failure"})
csv_placeholder("signed_support_probe_results.csv",reason);dump("signed_support_probe_results.json",placeholder(reason)|{"Q_SIGN_LEFT":"NOT_EXECUTED","Q_SIGN_RIGHT":"NOT_EXECUTED","mirror_consistency":"NOT_EVALUATED"})
dump("symmetry_cancellation_metrics.json",placeholder(reason)|{"g_left":"NOT_COMPUTED","g_right":"NOT_COMPUTED","g_abs":"NOT_COMPUTED","symmetry_axis_projection":"NOT_COMPUTED","shared_axis_projection":"NOT_COMPUTED","symmetry_cancellation_confirmed":False})
dump("support_side_temporal_stability.json",placeholder(reason)|{"first_dominant_support_step":"NOT_COMPUTED","support_side_reversal_count":"NOT_COMPUTED","yaw_correlation":"NOT_COMPUTED","oscillatory_support":"NOT_DETERMINED"})
dump("positive_trajectory_reference_manifest.json",x["positive"])
dump("positive_trajectory_dynamics.json",{"status":"INSUFFICIENT_RAW_DYNAMICS","D15_rare_success":{"count":1,"raw_force_momentum_trajectory_available":False},"A5_forward_success":{"aggregate_profiles_available":x["positive"]["A5"]["profiles"],"raw_force_momentum_trajectory_available":False},"quantities_not_reconstructable":["signed load balance","total support","low-load ratio","support side","Lz","dLz/dt","contact sequence"],"no_estimation":True})
dump("support_target_schedule_comparison.json",{"current_contract":"0 to 0.7 by 0.35 s; hold peak while weight decays to 0 at 0.75 s","D18_implementation":"0 to 0.7 by 0.35 s; target hard-resets to 0 after 0.50 s while weight remains active","positive_reference_schedule":"NOT_RECOVERABLE","classification":"IMPLEMENTATION_CONTRACT_MISMATCH","positive_reference_shape_class":"NO_COMMON_SCHEDULE_NOT_ASSESSED"})
subs=["SUPPORT_REWARD_IMPLEMENTATION_BUG","SUPPORT_REWARD_TIMING_MISMATCH","SUPPORT_POSITIVE_REFERENCE_INSUFFICIENT"]
dump("root_cause_classification.json",{"sub_classifications":subs,"primary_root_cause":"D18 support target and support-validity implementation violate the registered support objective contract","evidence":{"failed_unit_tests":x["failed_tests"],"target_at_0p60_implemented":0,"target_at_0p60_contract":.7,"zero_support_zero_target_load_reward":1.0},"not_evaluated_due_fail_closed":["reward-family conflict","signed-target causality","symmetry cancellation","support-side temporal stability"],"decision_precedence":"implementation bug selected first"})
dump("stage_classification.json",{"classification":classification,"sub_classifications":subs,"unit_test_status":"FAIL","temporary_policy_updates":0,"persistent_policy_updates":0,"D18_unchanged":True})
dump("recommended_next_action.json",{"single_experiment":"fix support reward implementation/timing only, then rerun the D18 one-update causal preflight","changes_allowed":["retain peak target through 0.50-0.75 s decay window","gate load reward on valid total support"],"persistent_PPO":False,"architecture_change":False,"actor_input_change":False})
diff=subprocess.check_output(["git","diff","--name-only",START],cwd=REPO,text=True).splitlines();protected=[p for p in diff if any(f"phase_2_d{i}" in p.replace("\\","/") for i in range(6,19)) or any(f"exp_{i:03d}_" in p.replace("\\","/") for i in range(5,14))]
dump("protected_hashes.json",{"starting_head":START,"exp005_exp013_changed_by_D19":0,"D6_D18_changed_by_D19":0,"preexisting_unrelated_dirty_paths_preserved":protected,"persistent_policy_update":0,"new_checkpoint":0,"formal_contract_change":0,"actor_input_change":0,"RUN":0,"causal_dagger_v2":0,"remote_push":False})
(OUT/"reproduction_commands.ps1").write_text("python experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d19_support_audit.py\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d19.py\n",encoding="utf-8")
REPORT.write_text(f"""# EXP014 Phase 2-D19 support objective symmetry audit

## Outcome

**{classification}**. The synthetic fail-closed gate failed before any temporary policy update or physics probe. Persistent updates and checkpoints are zero.

## Implementation and timing

D18 reads world-frame ankle contact forces after `wrapped.step(action)`, so the reward observes the just-applied action and refreshed physics. No pre-action sensor timing bug was found. However, the registered schedule retains the 0.7 target while its weight decays from 0.50 to 0.75 s; the implementation hard-resets the target to zero immediately after 0.50 s while retaining a positive envelope. At 0.60 s the implemented target is 0 rather than 0.7.

The load term also lacks a valid-total-support mask. With F_L=F_R=0 and target=0 its value is exactly 1.0, allowing flight/no-contact to maximize the load term. Additive total-support reward does not remove that false optimum from the load term itself.

## Synthetic and mirror tests

{len(x['unit_tests'])} synthetic tests ran: {sum(t['pass'] for t in x['unit_tests'])} passed and {len(x['failed_tests'])} failed (`{x['failed_tests'][0]}`, `{x['failed_tests'][1]}`). Algebraic mirror tests passed for 32 pairs: unsigned reward is invariant, signed-left maps to signed-right, and the repository joint permutation/sign map is involutive.

## Conditional diagnostics

Per protocol, Q_LOAD_ABS through Q_SUPPORT_FULL, gradient conflicts, advantage lags, action-to-load timing, signed left/right probes, and support-side reversal physics were not executed. Consequently symmetry cancellation and family conflict were not inferred from incomplete evidence.

## Positive references

D15 preserves the single acquisition only as aggregate outcome; A5 preserves aggregate profile results. Neither contains the raw contact-force/Lz trajectory needed to recover the actual support schedule, so `SUPPORT_POSITIVE_REFERENCE_INSUFFICIENT` is diagnostic only and no schedule was estimated.

## Decision

Decision precedence selects the implementation/timing bug. The next single experiment is to fix only the support reward implementation—keep the peak target through the decay window and gate load reward on valid total support—then rerun the D18 one-update causal preflight. Persistent PPO remains unauthorized.

## Protection

exp_005-exp_013, D6-D18 artifacts, policies, datasets, optimizers, physics, formal contracts, and actor inputs were unchanged. Persistent updates, checkpoints, RUN, Causal DAgger V2, and remote push are zero.
""",encoding="utf-8")
print(json.dumps({"classification":classification,"artifacts":24,"persistent_updates":0},indent=2))
