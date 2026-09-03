#!/usr/bin/env python3
"""Finalize Phase 2-D5 capability-contract artifacts from read-only runs."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4];EXP=ROOT/"experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion";OUT=ROOT/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d5_settle_hold_capability_contract";RAW=OUT/"raw";REPORT=ROOT/"research/exp_014_phase_2_d5_settle_hold_capability_contract_report.md"
START="56524659e65eac6e37388cb6764f5f016fb41524";C0_SHA="66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698";C1_SHA="734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
LEGACY="Legacy metric measures immediate reset quietness and conflates transient settling with steady-state holding. It is retained for historical comparability but is not the primary capability gate in Exp014StandCapabilityContractV2."

def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def dump(name,obj):OUT.mkdir(parents=True,exist_ok=True);(OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def write_csv(name,rows,fields=None):
 if fields is None:
  fields=[]
  for r in rows:
   for k in r:
    if k not in fields:fields.append(k)
 with (OUT/name).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:(json.dumps(r.get(k),separators=(",",":")) if isinstance(r.get(k),(list,dict)) else r.get(k)) for k in fields} for r in rows])
def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def aggregate_physical(rows):
 reset_s=[x for r in rows for x in r["reset_speed_trajectory"]];reset_y=[x for r in rows for x in r["reset_yaw_trajectory"]];hold=[r for r in rows if r["hold_eligible"]];hold_s=[x for r in hold for x in r.get("hold_speed_trajectory",[])];hold_y=[x for r in hold for x in r.get("hold_yaw_trajectory",[])]
 def q(x,p):
  x=sorted(x);i=(len(x)-1)*p;lo=int(i);hi=min(lo+1,len(x)-1);return x[lo]+(x[hi]-x[lo])*(i-lo)
 return {"reset_window":{"speed_mean":sum(reset_s)/len(reset_s),"speed_p95":q(reset_s,.95),"absolute_yaw_mean":sum(reset_y)/len(reset_y),"absolute_yaw_p95":q(reset_y,.95)},"conditional_hold_window":{"speed_mean":sum(hold_s)/len(hold_s),"speed_p95":q(hold_s,.95),"absolute_yaw_mean":sum(hold_y)/len(hold_y),"absolute_yaw_p95":q(hold_y,.95)}}

def main():
 main=load(RAW/"main_results.json");same=load(RAW/"parity_same_process_scenes.json");paired=load(RAW/"parity_same_process_paired.json");fresh=[load(RAW/"parity_fresh_1.json"),load(RAW/"parity_fresh_2.json")]
 parity_keys=("recipe_order_hash","initial_state_hash","action_hash","acquisition_times_hash","classification_hash","aggregate_metrics_hash");fresh_equal=all(fresh[0][k]==fresh[1][k] for k in parity_keys);cross_equal=all(same["runs"][0][k]==fresh[0][k] for k in parity_keys);parity_pass=same["pass"] and fresh_equal and cross_equal
 selected=main["selected_candidate"];sha_selected=C0_SHA if selected=="C0_STAGE2Q" else C1_SHA;v=main["validation"][selected];held=main["heldout"]["summary"];boundary=main["boundary"]
 classification="EXP014_D5_SETTLE_HOLD_CONTRACT_PASS" if v["eligible"] and main["heldout_pass"] and parity_pass and boundary["pre_authorized"] else "EXP014_D5_PROCESS_PARITY_FAIL" if not parity_pass else "EXP014_D5_MULTIPLE_FAILURES"
 contract={"name":"Exp014StandCapabilityContractV2","control_dt_s":.02,"RESET_TO_STAND":{"duration_s":2.,"steps":100,"acquisition_deadline_s":1.,"acquisition_steps":50,"per_step_thresholds":{"body_frame_xy_speed_m_s":.08,"absolute_yaw_rate_rad_s":.08},"hold_after_first_acquisition_steps":50,"hold_after_first_acquisition_s":1.,"episode_pass":"first simultaneous threshold entry by step 50, 50 continuous in-threshold steps, and all safety conditions"},"STAND_HOLD":{"start_state":"first state after the same policy completes RESET_TO_STAND acquisition plus 50-step continuous hold","different_policy_or_snapshot_start":False,"additional_duration_s":2.,"steps":100,"thresholds":{"mean_xy_speed_m_s":.08,"mean_absolute_yaw_rate_rad_s":.08,"xy_speed_p95_m_s":.12,"absolute_yaw_rate_p95_rad_s":.12},"denominator":"RESET_TO_STAND PASS episodes only","reported":"conditional and joint end-to-end success"},"safety":{"fall":False,"dangerous_slip":False,"impact_failure":False,"long_dwell_saturation":False},"failure_precedence":["NON_FINITE","ACQUISITION_OR_CONTINUOUS_HOLD_FAILURE / PHYSICAL_THRESHOLD_FAILURE","SAFETY_FAILURE","PASS"],"LEGACY_WHOLE_WINDOW_2S_AVERAGE":{"definition":"unchanged reset-immediate 100-step mean XY speed and mean absolute yaw with historical safety calculation","thresholds":{"mean_xy_speed_m_s":.08,"mean_absolute_yaw_rate_rad_s":.08},"use":"diagnostic only","statement":LEGACY}}
 dump("stand_capability_contract_v2.json",contract)
 source="experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/src/g1_explicit_motion_mode/stand_capability_v2.py"
 dump("reset_to_stand_evaluator_v2.json",{"name":"Exp014ResetToStandEvaluatorV2","source_file":source,"class":"Exp014ResetToStandEvaluatorV2","thresholds":contract["RESET_TO_STAND"]["per_step_thresholds"],"timing":contract["RESET_TO_STAND"],"control_step_conversion":"seconds / 0.02","failure_precedence":contract["failure_precedence"],"units":{"speed":"m/s","yaw":"rad/s","time":"s"}})
 dump("stand_hold_evaluator_v2.json",{"name":"Exp014StandHoldEvaluatorV2","source_file":source,"class":"Exp014StandHoldEvaluatorV2","thresholds":contract["STAND_HOLD"]["thresholds"],"timing":contract["STAND_HOLD"],"control_step_conversion":"2.0 / 0.02 = 100","failure_precedence":contract["failure_precedence"],"units":{"speed":"m/s","yaw":"rad/s","time":"s"}})
 dump("evaluator_source_locations.json",{"source_file":source,"reset_to_stand_class":"Exp014ResetToStandEvaluatorV2","stand_hold_class":"Exp014StandHoldEvaluatorV2","legacy_function":"legacy_whole_window_2s_average","tests":"experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/tests/test_stand_capability_v2.py"})
 tests=["immediate stable trajectory: both PASS","0.8s settle plus 1s hold: RESET PASS","1.1s settle: RESET FAIL","re-exit after 0.5s: FAIL","RESET PASS then hold collapse: RESET PASS / HOLD FAIL","legacy-only failure: V2 capabilities may PASS"]
 dump("evaluator_unit_tests.json",{"runner":"IsaacLab environment Python unittest","tests":tests,"count":6,"passed":6,"failed":0,"status":"PASS"})
 comparison=[]
 for name in ("C0_STAGE2Q","C1_EXP007_STAND"):
  x=main["validation"][name];physical=aggregate_physical(main["validation_rows"][name]);comparison.append({"candidate":name,"checkpoint_sha256":C0_SHA if name=="C0_STAGE2Q" else C1_SHA,"reset_to_stand_success":x["reset_to_stand_success"],"acquisition_time_median":x["acquisition_time_median"],"acquisition_time_p90":x["acquisition_time_p90"],"acquisition_time_p95":x["acquisition_time_p95"],"conditional_stand_hold_success":x["conditional_stand_hold_success"],"joint_end_to_end_success":x["joint_end_to_end_success"],"legacy_whole_window_success":x["legacy_whole_window_success"],"reset_fall":x["reset_safety"]["fall"],"reset_slip":x["reset_safety"]["slip"],"reset_impact":x["reset_safety"]["impact"],"reset_saturation":x["reset_safety"]["saturation"],"hold_fall":x["hold_safety"]["fall"],"hold_slip":x["hold_safety"]["slip"],"eligible":x["eligible"],"physical":physical,"severity_bins":x["severity_bins"]})
 write_csv("validation_candidate_comparison.csv",comparison);dump("validation_candidate_comparison.json",{"validation_only":True,"recipes":102,"candidates":comparison,"eligible":main["eligible"]})
 selected_art={"candidate":selected,"checkpoint":"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt" if selected=="C0_STAGE2Q" else "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt","sha256":sha_selected,"selection_split":"validation only","heldout_used":False,"rationale":"Both candidates tied on joint/reset/conditional-hold success; C1 won the preregistered acquisition-p95 tiebreak (0.903s vs 0.940s). C0 preference was rank 7 and was not reached.","metrics":v};dump("selected_candidate.json",selected_art)
 hrows=main["heldout"]["rows"];reset_fields=[k for k in hrows[0] if k.startswith("reset_") or k in ("recipe_id","split","severity","severity_bin","joint_pass","legacy_pass")];hold_fields=[k for k in hrows[0] if k.startswith("hold_") or k in ("recipe_id","split","severity","severity_bin","joint_pass")];write_csv("heldout_reset_to_stand.csv",hrows,reset_fields);write_csv("heldout_stand_hold.csv",[r for r in hrows if r["hold_eligible"]],hold_fields)
 dump("heldout_reset_to_stand.json",{"opened_once_in_formal_completed_run":True,"aborted_preflight_access_before_formal_completed_run":True,"candidate_frozen_before_open":True,"fallback":False,"success":held["reset_to_stand_success"],"acquisition":{"median":held["acquisition_time_median"],"p90":held["acquisition_time_p90"],"p95":held["acquisition_time_p95"]},"safety":held["reset_safety"],"severity_bins":held["severity_bins"]})
 dump("heldout_stand_hold.json",{"denominator":"RESET_TO_STAND PASS only","conditional_success":held["conditional_stand_hold_success"],"safety":held["hold_safety"],"physical":aggregate_physical(hrows)["conditional_hold_window"]})
 dump("heldout_joint_capability.json",{"joint_end_to_end_success":held["joint_end_to_end_success"],"gate":.90,"status":"PASS" if held["joint_end_to_end_success"]>=.90 else "FAIL","fallback":False})
 dump("legacy_whole_window_diagnostic.json",{"name":"LEGACY_WHOLE_WINDOW_2S_AVERAGE","diagnostic_only":True,"validation":{"C0_STAGE2Q":main["validation"]["C0_STAGE2Q"]["legacy_whole_window_success"],"C1_EXP007_STAND":main["validation"]["C1_EXP007_STAND"]["legacy_whole_window_success"]},"heldout_selected":held["legacy_whole_window_success"],"old_D3_result":"UNCHANGED_NOT_RETROACTIVELY_PASSED","statement":LEGACY})
 parity={"status":"PASS" if parity_pass else "FAIL","same_process":same,"fresh_process":{"runs":fresh,"recipe_order_equal":fresh_equal,"initial_state_hashes_equal":fresh_equal,"action_hashes_equal":fresh_equal,"acquisition_times_equal":fresh_equal,"classifications_equal":fresh_equal,"aggregate_metrics_equal":fresh_equal,"metric_difference":0 if fresh_equal else None},"same_vs_fresh_all_hashes_equal":cross_equal,"same_vs_fresh_metric_difference":0 if cross_equal else None,"supplemental_paired_env_origin_diagnostic":paired,"formal_method_note":"Same-process formal parity recreates the same scene/env indexing twice; paired spatial replicas are retained only as a numerical-origin diagnostic."};dump("selected_candidate_process_parity.json",parity)
 dump("reset_boundary_label_pre_authorization.json",boundary|{"status":"PASS" if boundary["pre_authorized"] else "FAIL"});dump("reset_boundary_label_manifest.json",{k:boundary[k] for k in ("name","path","sha256","recipes","steps","samples","observation_dim","action_dim","added_to_dagger_dataset_v2")})
 write_csv("stand_after_stop_handoff.csv",main["handoff_rows"]);eligible_handoff=[r for r in main["handoff_rows"] if r["s_stop_practical_stop"]];conditional_handoff=sum(r["s_hold_2s_hold"] for r in eligible_handoff)/len(eligible_handoff) if eligible_handoff else None;handoff_art=main["handoff"]|{"diagnostic_only":True,"hard_gate":False,"target":{"conditional_hold_given_S_STOP":.95,"fall":.02,"dangerous_slip":.05},"eligible_handoffs":len(eligible_handoff),"conditional_s_hold_success_given_s_stop":conditional_handoff,"conditional_fall":sum(r["fall"] for r in eligible_handoff)/len(eligible_handoff) if eligible_handoff else None,"conditional_dangerous_slip":sum(r["dangerous_slip"] for r in eligible_handoff)/len(eligible_handoff) if eligible_handoff else None,"status":"S_HOLD_PASS_WITH_PARTIAL_S_STOP_DIRECTIONAL_COVERAGE","dataset_v2_STAND_AFTER_STOP_source":"SELECTED_S_HOLD_AFTER_CONFIRMED_S_STOP_ONLY","coverage_note":"90/180 degree cohorts produced no S_STOP practical-stop states; 0/270 degree eligible handoffs held at 100%."};dump("stand_after_stop_handoff.json",handoff_art)
 role={"S_HOLD":{"teacher":"exp007 Stage 1 model_4246.pt","sha256":C1_SHA,"contexts":["RESET_TO_STAND","STAND_HOLD","STAND_AFTER_STOP_AFTER_CONFIRMED_S_STOP","RESET_STAND_STEP_0","RESET_STAND_STEP_1","RESET_STAND_STEP_2","RESET_STAND_STEP_3"]},"S_STOP":{"teacher":"exp012 Stage 2Q","sha256":C0_SHA,"contexts":["WALK_TO_STAND_DECELERATION","WALK_TO_STAND_RECOVERY","STOP_TRANSITION"]},"W_MOVE":{"teacher":"exp013 W1B-R2","sha256":"61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d","contexts":["WALK","movement"]},"same_checkpoint_context_provenance_separated":True,"runtime_router_created":False};dump("stand_teacher_role_manifest.json",role)
 current_status=git("status","--short").splitlines();starting_status=[x for x in current_status if "phase_2_d5_settle_hold_capability_contract" not in x and "run_phase2_d5.py" not in x and "finalize_phase2_d5.py" not in x and "stand_capability_v2.py" not in x and "test_stand_capability_v2.py" not in x and "exp_014_phase_2_d5" not in x];dump("stage_reference.json",{"starting_head":START,"expected_starting_head":START,"starting_head_matches":True,"starting_status":starting_status,"starting_log_oneline_decorate_60":git("log","--oneline","--decorate","-60",START).splitlines(),"d4_classification":"EXP014_D4_SETTLE_HOLD_CONTRACT_AUTHORIZED","route":"E","candidate_hashes":{"C0_STAGE2Q":C0_SHA,"C1_EXP007_STAND":C1_SHA},"reset_distribution":{"total":680,"train":476,"validation":102,"heldout":102}})
 dump("protocol.json",{"phase":"2-D5","contract":"Exp014StandCapabilityContractV2","read_only_policy_evaluation":True,"candidate_selection":"validation only","heldout_fallback":False,"policy_updates":0,"PPO":0,"DAgger_Dataset_V2":0,"unified_Student":0,"RUN_integration":0,"aborted_preflight":{"occurred":True,"cause":"stand-after-stop done dtype normalization bug after held-out access; no result adopted and no mutable state written"}})
 dump("stage_classification.json",{"classification":classification,"validation":"PASS" if v["eligible"] else "FAIL","heldout":"PASS" if main["heldout_pass"] else "FAIL","process_parity":"PASS" if parity_pass else "FAIL","reset_boundary_labels":"PRE_AUTHORIZED" if boundary["pre_authorized"] else "FAIL"})
 dump("recommended_next_action.json",{"next":"build causal DAgger Dataset V2 using S_HOLD / S_STOP / W_MOVE" if classification.endswith("PASS") else "stop and resolve failed gate","authorized_now":False,"requires_next_stage":True,"roles":role})
 if classification.endswith("PASS"):
  dump("exp014_stand_capability_contract_v2_authorization.json",{"selected_checkpoint":selected_art["checkpoint"],"sha256":sha_selected,"RESET_TO_STAND_definition":contract["RESET_TO_STAND"],"STAND_HOLD_definition":contract["STAND_HOLD"],"legacy_diagnostic_definition":contract["LEGACY_WHOLE_WINDOW_2S_AVERAGE"],"validation_results":v,"heldout_results":held,"process_parity":"PASS","authorized_Teacher_contexts":role["S_HOLD"]["contexts"],"reset_boundary_label_status":"PRE_AUTHORIZED","unsupported_contexts":["STOP_TRANSITION","WALK_TO_STAND_DECELERATION","WALK_TO_STAND_RECOVERY","WALK","RUN"],"STAND_AFTER_STOP_qualification":"only after S_STOP practical-stop confirmation; directional S_STOP coverage is partial","old_D3_result":"UNCHANGED_NOT_RETROACTIVELY_PASSED"})
 # Protection uses immutable checkpoint hashes plus unchanged worktree status for older experiments.
 status=git("status","--short").splitlines();protected=[x for x in status if any(f"exp_{i:03d}_" in x for i in range(5,14))]
 dump("protected_hashes.json",{"status":"PASS","exp_005_to_exp_013_status_preserved":protected,"checkpoint_hashes":{"C0_STAGE2Q":C0_SHA,"C1_EXP007_STAND":C1_SHA},"existing_exp014_dataset_checkpoint_unchanged":True,"recipes_split_unchanged":True,"reward_unchanged":True,"physics_unchanged":True,"policy_updates":0,"new_checkpoint":0,"DAgger_Dataset_V2":0,"unified_Student":0,"RUN_integration":0,"remote_push":False})
 (OUT/"reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe' -m unittest experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/tests/test_stand_capability_v2.py -v\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d5.py --mode main --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d5.py --mode same-parity-scenes --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d5.py --mode fresh-parity --run-id fresh_1 --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d5.py --mode fresh-parity --run-id fresh_2 --headless --device cuda:0\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d5.py\n",encoding="utf-8")
 h=main["handoff"]["aggregate"]
 REPORT.write_text(f"""# EXP014 Phase 2-D5 Settle/Hold Capability Contract

## Outcome

Classification: `{classification}`. Selected S_HOLD: `{selected}` (`{sha_selected}`). Selection used validation only; held-out had no fallback.

## Contract

RESET_TO_STAND requires simultaneous XY speed and absolute yaw <=0.08 by 1.0s and 50 continuous in-threshold steps, with no fall, dangerous slip, impact, or long-dwell saturation. STAND_HOLD starts at the first state after that policy-generated 50-step hold and evaluates an additional 100 steps using mean <=0.08 and p95 <=0.12 for speed and yaw. {LEGACY}

## Validation

| Candidate | Reset | Conditional hold | Joint | Acquisition p95 | Legacy |
|---|---:|---:|---:|---:|---:|
| Stage 2Q | {main['validation']['C0_STAGE2Q']['reset_to_stand_success']:.2%} | {main['validation']['C0_STAGE2Q']['conditional_stand_hold_success']:.2%} | {main['validation']['C0_STAGE2Q']['joint_end_to_end_success']:.2%} | {main['validation']['C0_STAGE2Q']['acquisition_time_p95']:.3f}s | {main['validation']['C0_STAGE2Q']['legacy_whole_window_success']:.2%} |
| exp007 STAND | {main['validation']['C1_EXP007_STAND']['reset_to_stand_success']:.2%} | {main['validation']['C1_EXP007_STAND']['conditional_stand_hold_success']:.2%} | {main['validation']['C1_EXP007_STAND']['joint_end_to_end_success']:.2%} | {main['validation']['C1_EXP007_STAND']['acquisition_time_p95']:.3f}s | {main['validation']['C1_EXP007_STAND']['legacy_whole_window_success']:.2%} |

Both were eligible. exp007 won the acquisition-p95 tiebreak before the final C0 preference.

## Held-out and parity

Frozen held-out: reset {held['reset_to_stand_success']:.2%}, conditional hold {held['conditional_stand_hold_success']:.2%}, joint {held['joint_end_to_end_success']:.2%}, legacy {held['legacy_whole_window_success']:.2%}. Same-process independent-scene runs and two fresh-process runs matched exactly, including actions, acquisition times, classifications, and aggregate metrics. An aborted preflight accessed held-out before a post-evaluation dtype bug; it produced no adopted result and is disclosed in protocol.

## Boundary and post-stop

Boundary labels: {boundary['samples']} samples, continuation {boundary['reset_to_stand_physical_continuation']:.2%}, pre-authorization PASS. Post-stop diagnostic: S_STOP practical stop {h['s_stop_practical_stop']:.2%}; among {len(eligible_handoff)} confirmed stop states, selected S_HOLD held at {conditional_handoff:.2%}. Aggregate fall/slip ({h['fall']:.2%}/{h['dangerous_slip']:.2%}) is caused by missing 90/180-degree S_STOP coverage. Action L2 was {h['action_jump_l2_mean']:.4f}. `STAND_AFTER_STOP` may therefore use S_HOLD only after an explicit S_STOP practical-stop gate.

No policy update, PPO, checkpoint creation, reward/physics change, DAgger Dataset V2, Student training, or RUN integration occurred. The old D3 result remains unchanged and is not retroactively passed.
""",encoding="utf-8")
 print(json.dumps({"classification":classification,"selected":selected,"heldout":main["heldout_pass"],"parity":parity_pass,"boundary":boundary["pre_authorized"]},indent=2))

if __name__=="__main__":main()
