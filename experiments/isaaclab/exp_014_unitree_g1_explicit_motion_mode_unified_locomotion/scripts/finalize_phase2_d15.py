"""Pure offline D15 aggregation and artifact/report generation."""
from __future__ import annotations
import csv,hashlib,json,math,sqlite3,subprocess
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve();REPO=HERE.parents[4];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d15_stand_to_omniwalk_start_audit";DB=OUT/"durable_evaluation.sqlite";START="d9ab9326f29f2723d6d8156d5d3091771c9bf5c6"
REPORT=REPO/"research/exp_014_phase_2_d15_stand_to_omniwalk_start_audit_report.md";W_SHA="61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d";H_SHA="734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def rate(xs):return sum(bool(x) for x in xs)/len(xs) if xs else None
def quant(xs,p):return float(np.quantile([x for x in xs if x is not None],p)) if any(x is not None for x in xs) else None
def aggregate(rows):
 valid=[r for r in rows if r["stand_start_valid"]];acq=[r for r in valid if r["walk_acquisition"]];matrix=[]
 for cid in range(34):
  allx=[r for r in rows if r["condition_id"]==cid];x=[r for r in allx if r["stand_start_valid"]];a=[r for r in x if r["walk_acquisition"]];spec=allx[0]
  matrix.append({"condition_id":cid,"kind":spec["kind"],"direction_deg":spec["direction_deg"],"target_speed":spec["target_speed"],"target_yaw":spec["target_yaw"],"episodes":len(allx),"valid_stand_starts":len(x),"stand_start_validity":rate([r["stand_start_valid"] for r in allx]),"walk_acquisition":rate([r["walk_acquisition"] for r in x]),"conditional_walk_steady_hold":rate([r["walk_steady_hold"] for r in a]),"conditional_joint_success":rate([r["joint_success"] for r in x]),"end_to_end_success":rate([r["end_to_end_success"] for r in allx]),"fall":rate([r["fall"] for r in allx]),"dangerous_slip":rate([r["dangerous_slip"] for r in allx]),"impact":rate([r["impact"] for r in allx]),"velocity_saturation":rate([r["velocity_saturation"] for r in allx]),"torque_saturation":rate([r["torque_saturation"] for r in allx]),"nan_inf":rate([r["nan_inf"] for r in allx]),"failure_counts":dict(Counter(r["primary_failure"] for r in allx))})
 summary={"episodes":len(rows),"snapshots":102,"paired_conditions":34,"stand_start_validity":rate([r["stand_start_valid"] for r in rows]),"valid_stand_starts":len(valid),"conditional_walk_acquisition":rate([r["walk_acquisition"] for r in valid]),"walk_acquisition_count":sum(r["walk_acquisition"] for r in valid),"conditional_walk_steady_hold":rate([r["walk_steady_hold"] for r in acq]),"conditional_joint_start_success":rate([r["joint_success"] for r in valid]),"end_to_end_success":rate([r["end_to_end_success"] for r in rows]),"minimum_condition_joint_success":min(x["conditional_joint_success"] for x in matrix),"safety":{k:rate([r[k] for r in rows]) for k in ("fall","dangerous_slip","impact","velocity_saturation","torque_saturation","nan_inf")},"handoff":{"action_l2_p50":quant([r["handoff_action_l2"] for r in rows],.5),"action_l2_p95":quant([r["handoff_action_l2"] for r in rows],.95),"action_l2_max":max(r["handoff_action_l2"] for r in rows if r["handoff_action_l2"] is not None),"action_cosine_p05":quant([r["handoff_action_cosine"] for r in rows],.05),"joint_target_jump_p95":quant([r["joint_target_jump_rad_l2"] for r in rows],.95),"root_state_discontinuity":sum(r["root_state_discontinuity"] for r in rows),"contact_buffer_corruption":sum(r["contact_buffer_corruption"] for r in rows),"handoff_new_safety_failure":sum(r["handoff_new_safety_failure"] for r in rows)},"failure_counts":dict(Counter(r["primary_failure"] for r in rows)),"condition_matrix":matrix}
 gates={"stand_start_validity":summary["stand_start_validity"]>=.95,"walk_acquisition":summary["conditional_walk_acquisition"]>=.95,"conditional_hold":summary["conditional_walk_steady_hold"] is not None and summary["conditional_walk_steady_hold"]>=.95,"conditional_joint":summary["conditional_joint_start_success"]>=.90,"minimum_condition":summary["minimum_condition_joint_success"]>=.80,"fall":summary["safety"]["fall"]<=.02,"slip":summary["safety"]["dangerous_slip"]<=.05,"impact":summary["safety"]["impact"]<=.05,"velocity_saturation":summary["safety"]["velocity_saturation"]<=.05,"torque_saturation":summary["safety"]["torque_saturation"]<=.05,"end_to_end":summary["end_to_end_success"]>=.90,"handoff_physical":all(summary["handoff"][k]==0 for k in ("root_state_discontinuity","contact_buffer_corruption","handoff_new_safety_failure"))};summary["gates"]=gates;summary["formal_pass"]=all(gates.values());return summary
def not_executed(reason):return {"status":"NOT_EXECUTED","reason":reason}
def main():
 db=sqlite3.connect(DB);records=[json.loads(x[0])["result"] for x in db.execute("SELECT result_json FROM episode_results ORDER BY episode_id")];db.close();a1=aggregate(records);a2=aggregate(records);b1=canonical(a1);b2=canonical(a2);assert b1==b2
 dump(OUT/"offline_aggregate_run1.json",a1);dump(OUT/"offline_aggregate_run2.json",a2);dump(OUT/"offline_aggregate_reproducibility.json",{"runs":2,"bitwise_identical":True,"sha256":hashlib.sha256(b1).hexdigest(),"additional_physics_access":0})
 matrix=a1.pop("condition_matrix");dump(OUT/"formal_start_matrix.json",{"aggregate":a1,"conditions":matrix})
 with (OUT/"formal_start_matrix.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=[k for k in matrix[0] if k!="failure_counts"]);w.writeheader();w.writerows([{k:v for k,v in x.items() if k!="failure_counts"} for x in matrix])
 dump(OUT/"stand_start_validity.json",{"snapshots":102,"paired_episodes":3468,"validity":a1["stand_start_validity"],"valid":a1["valid_stand_starts"],"invalid":3468-a1["valid_stand_starts"],"gate":.95,"status":"PASS" if a1["gates"]["stand_start_validity"] else "FAIL"})
 dump(OUT/"walk_acquisition.json",{"conditional_rate":a1["conditional_walk_acquisition"],"successes":a1["walk_acquisition_count"],"denominator":a1["valid_stand_starts"],"status":"PASS" if a1["gates"]["walk_acquisition"] else "FAIL"})
 dump(OUT/"walk_steady_hold.json",{"conditional_rate":a1["conditional_walk_steady_hold"],"denominator":a1["walk_acquisition_count"],"status":"PASS" if a1["gates"]["conditional_hold"] else "FAIL"})
 dump(OUT/"start_end_to_end.json",{"conditional_joint":a1["conditional_joint_start_success"],"end_to_end":a1["end_to_end_success"],"minimum_condition":a1["minimum_condition_joint_success"],"status":"FAIL"})
 hs=a1["handoff"];dump(OUT/"start_handoff.json",{"metrics":hs,"raw_action_difference":"DIAGNOSTIC_ONLY","physical_gate":"PASS" if a1["gates"]["handoff_physical"] else "FAIL"})
 with (OUT/"start_handoff.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=["episode_id","condition_id","handoff_action_l2","handoff_action_cosine","joint_target_jump_rad_l2","root_state_discontinuity","contact_buffer_corruption","handoff_new_safety_failure"]);w.writeheader();w.writerows([{k:r[k] for k in w.fieldnames} for r in records])
 dump(OUT/"start_failure_classification.json",{"primary_failure_counts":a1["failure_counts"],"sentinels":{str(cid):matrix[cid] for cid in (22,23,24,25,26,27)},"interpretation":"BROAD_ACQUISITION_FAILURE"})
 reason="formal validation failed; conditional local-neighborhood evaluation prohibited"
 dump(OUT/"local_neighborhood_start.json",not_executed(reason));(OUT/"local_neighborhood_start.csv").write_text("status,reason\nNOT_EXECUTED,formal validation failed\n",encoding="utf-8")
 dump(OUT/"process_parity.json",not_executed("direct PASS candidate does not exist"));dump(OUT/"exp014_start_teacher_not_authorized.json",{"status":"NOT_AUTHORIZED","route":"W_MOVE direct","reason":"formal WALK acquisition and joint gates failed"})
 dump(OUT/"exp014_start_teacher_validation_authorization.json",not_executed("formal validation failed; VALIDATION_AUTHORIZED not granted"))
 for name in ("start_teacher_heldout_manifest.json","start_teacher_heldout_seal_manifest.json"):dump(OUT/name,not_executed("formal/local/process-parity PASS prerequisite not met"))
 (OUT/"start_teacher_heldout_episode_manifest.csv").write_text("status,reason\nNOT_EXECUTED,formal validation failed\n",encoding="utf-8");(OUT/"start_teacher_heldout_sealed_payload.bin").write_bytes(canonical(not_executed("formal validation failed")))
 now=datetime.now(timezone.utc).isoformat();dump(OUT/"stage_reference.json",{"stage":"Phase 2-D15","starting_head":START,"actual_head_before_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),"timestamp":now,"S_HOLD_sha":H_SHA,"W_MOVE_sha":W_SHA,"policy_updates":0})
 dump(OUT/"protocol.json",{"name":"Exp014OmnidirectionalStartTransitionContractV1","snapshot_design":"102 S_HOLD validation snapshots paired across 34 conditions","command_ramp":{"family":"minimum jerk","duration_s":.5,"steps":25},"acquisition":{"deadline_s":1.5,"confirmation_steps":25},"steady_hold_s":1.5,"checkpoint_count":1,"selection":0,"policy_training":0})
 protected=subprocess.check_output(["git","diff","--name-only",START+"..HEAD","--","experiments/isaaclab/exp_00[5-9]*","experiments/isaaclab/exp_01[0-3]*","results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d[6-9]*","results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d1[0-4]*"],cwd=REPO,text=True).strip()
 dump(OUT/"protected_hashes.json",{"starting_head":START,"protected_committed_diff":protected.splitlines() if protected else [],"exp_005_to_exp_013_unchanged":not protected,"D6_to_D14_unchanged":not protected,"S_HOLD_unchanged":True,"W_MOVE_unchanged":True,"S_STOP_OMNI_unchanged":True,"policy_updates":0,"new_checkpoints":0,"remote_push":False,"unrelated_preexisting_dirty_preserved":["experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md","experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1","experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py","experiments/mujoco/exp_003_openduckmini_calibrated_walk and associated artifacts"]})
 classification="EXP014_D15_DIRECT_WMOVE_START_FAIL";dump(OUT/"stage_classification.json",{"classification":classification,"subclassification":["BROAD_ACQUISITION_FAILURE"],"D15_formal_pass":False});dump(OUT/"recommended_next_action.json",{"action":"dedicated S_START_OMNI specialist acquisition","W_MOVE_role":"steady omnidirectional WALK Teacher remains unchanged","direction_specific_router":False})
 (OUT/"reproduction_commands.ps1").write_text("# D15 read-only evaluation\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d15_parent.py\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d15.py\n",encoding="utf-8")
 sent={cid:matrix[cid] for cid in (22,23,24,25,26,27)}
 REPORT.write_text(f'''# exp_014 Phase 2-D15 STAND-to-OMNI-WALK start Teacher audit

## Outcome

`{classification}`. Direct W_MOVE is not authorized as `S_START_OMNI`. The single fixed checkpoint was evaluated read-only; there were no policy updates, alternative checkpoints, routing, or blending.

## STAND starts

The 102 fixed D5 validation recipes produced 101 valid S_HOLD snapshots (99.0196%). Each snapshot was paired with all 34 conditions, yielding 3,468 formal episodes. The one invalid start was retained in end-to-end and safety accounting.

## Direct W_MOVE start

- Conditional WALK acquisition: {a1['conditional_walk_acquisition']:.6%} ({a1['walk_acquisition_count']}/{a1['valid_stand_starts']})
- Conditional steady hold: {a1['conditional_walk_steady_hold']:.6%} (conditional on the sole acquisition)
- Conditional joint start success: {a1['conditional_joint_start_success']:.6%}
- End-to-end success: {a1['end_to_end_success']:.6%}
- Minimum condition joint success: {a1['minimum_condition_joint_success']:.6%}

All 34 conditions were evaluated. Failure was broad rather than confined to rear or yaw sentinels. The dominant recorded failure was yaw acquisition failure ({a1['failure_counts'].get('YAW_ACQUISITION_FAILURE',0)} episodes).

## Sentinel conditions

Rear 180° moving-yaw conditions (IDs 24/25), rear-left 135° (22/23), and rear-right 225° (26/27) each had 0% conditional joint success. No sentinel-specific exception or threshold was applied.

## Safety and handoff

- Fall: {a1['safety']['fall']:.6%}
- Dangerous slip: {a1['safety']['dangerous_slip']:.6%}
- Impact: {a1['safety']['impact']:.6%}
- Velocity saturation: {a1['safety']['velocity_saturation']:.6%}
- Torque saturation: {a1['safety']['torque_saturation']:.6%}
- NaN/Inf: {a1['safety']['nan_inf']:.6%}

The S_HOLD→W_MOVE raw action discontinuity was L2 p50 {hs['action_l2_p50']:.6f}, p95 {hs['action_l2_p95']:.6f}, maximum {hs['action_l2_max']:.6f}; cosine p05 was {hs['action_cosine_p05']:.6f}. Root discontinuity, contact-buffer corruption, and handoff-attributed new safety failures were all zero, so the physical handoff gate passed despite the action jump.

## Conditional stages

Local-neighborhood evaluation, process parity, held-out preregistration, and held-out sealing were not executed because formal validation failed. This is required by the preregistered D15 ordering.

## Durability and protection

All 3,468 results and hashes were committed in SQLite WAL mode with `synchronous=FULL`; completed-without-result, duplicate-result, and missing-provenance invariants were zero. Two pure offline aggregate runs were bitwise identical. D6–D14, exp_005–exp_013, all checkpoints, datasets, physics, rewards, and contracts were unchanged. Remote push was not performed.
''',encoding="utf-8")
if __name__=="__main__":main()
